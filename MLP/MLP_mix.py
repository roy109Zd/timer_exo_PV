import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================== 配置参数 =====================
PRED_DIR = "/root/timer+exo/pred"
STATION = "station00"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
EPOCHS = 150
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-5

# ===================== 评估指标 =====================
def compute_metrics(true, pred, power_range):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    return mae, rmse, nmae, nrmse, r2

# ===================== 新模型定义 =====================
class TwoBranchMLP(nn.Module):
    """
    分支结构：
    - timer 分支: 1 -> 8 -> 16
    - nwp 分支: 7 -> 16 -> 16
    - 合并: 32 维
    - 合并后: 32 -> 64 -> 1
    """
    def __init__(self):
        super().__init__()
        # Timer 分支: 输入1维，输出16维
        self.timer_branch = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU()
        )
        # NWP 分支: 输入7维，输出16维
        self.nwp_branch = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        # 合并后 MLP: 32 -> 64 -> 1
        self.combined = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_timer, x_nwp):
        # x_timer: (batch, 1)
        # x_nwp:   (batch, 7)
        feat_timer = self.timer_branch(x_timer)   # (batch, 16)
        feat_nwp   = self.nwp_branch(x_nwp)       # (batch, 16)
        combined   = torch.cat([feat_timer, feat_nwp], dim=1)  # (batch, 32)
        out = self.combined(combined)             # (batch, 1)
        return out.squeeze(-1)                    # (batch,)

# ===================== 数据加载 =====================
def load_station_data(station_name):
    csv_path = os.path.join(PRED_DIR, f"{station_name}_timer_pred_with_info.csv")
    df = pd.read_csv(csv_path)
    nwp_cols = ['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 'nwp_humidity',
                'nwp_windspeed', 'nwp_winddirection', 'nwp_pressure']
    # 注意：原始数据中 power_pred 是 Timer 预测值， power_true 是真实值
    timer_pred = df['power_pred'].values.astype(np.float32).reshape(-1, 1)   # (N,1)
    nwp_data = df[nwp_cols].values.astype(np.float32)                        # (N,7)
    targets = df['power_true'].values.astype(np.float32)                     # (N,)
    return timer_pred, nwp_data, targets

# ===================== 训练与评估 =====================
def main():
    print(f"单站 {STATION} 双分支 MLP 逐点预测 (Timer分支 + NWP分支)")

    # 加载数据
    timer_pred, nwp_data, targets = load_station_data(STATION)
    n_samples = len(targets)
    print(f"总样本数: {n_samples}")

    # 时间顺序划分 8:1:1
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)

    timer_train, timer_val, timer_test = timer_pred[:train_end], timer_pred[train_end:val_end], timer_pred[val_end:]
    nwp_train, nwp_val, nwp_test = nwp_data[:train_end], nwp_data[train_end:val_end], nwp_data[val_end:]
    y_train, y_val, y_test = targets[:train_end], targets[train_end:val_end], targets[val_end:]

    print(f"训练样本: {len(timer_train)}, 验证: {len(timer_val)}, 测试: {len(timer_test)}")

    # 可选：对 NWP 特征进行归一化（基于训练集）
    # 这里对 NWP 的每个维度独立归一化
    nwp_mean = nwp_train.mean(axis=0, keepdims=True)
    nwp_std = nwp_train.std(axis=0, keepdims=True) + 1e-8
    nwp_train_norm = (nwp_train - nwp_mean) / nwp_std
    nwp_val_norm   = (nwp_val   - nwp_mean) / nwp_std
    nwp_test_norm  = (nwp_test  - nwp_mean) / nwp_std

    # Timer 预测值不归一化（本身已经是功率量纲，也可以选择归一化，这里不做）
    # 如果想归一化，可以类似处理，但为了简洁，保持原始

    # 转换为 Tensor
    timer_train_t = torch.tensor(timer_train).to(DEVICE)
    nwp_train_t   = torch.tensor(nwp_train_norm).to(DEVICE)
    y_train_t     = torch.tensor(y_train).to(DEVICE)

    timer_val_t   = torch.tensor(timer_val).to(DEVICE)
    nwp_val_t     = torch.tensor(nwp_val_norm).to(DEVICE)
    y_val_t       = torch.tensor(y_val).to(DEVICE)

    timer_test_t  = torch.tensor(timer_test).to(DEVICE)
    nwp_test_t    = torch.tensor(nwp_test_norm).to(DEVICE)
    y_test_t      = torch.tensor(y_test).to(DEVICE)

    train_loader = DataLoader(TensorDataset(timer_train_t, nwp_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(timer_val_t,   nwp_val_t,   y_val_t),   batch_size=BATCH_SIZE, shuffle=False)

    model = TwoBranchMLP().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for tb, nb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(tb, nb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(tb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tb, nb, yb in val_loader:
                pred = model(tb, nb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(tb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ===== 评估 =====
    model.eval()
    with torch.no_grad():
        pred_test = model(timer_test_t, nwp_test_t).cpu().numpy()
    y_test_np = y_test_t.cpu().numpy()

    # 1. 原始 Timer 预测值的评估
    timer_pred_test = timer_test.flatten()
    print("\n" + "="*60)
    print("基准评估：直接使用 Timer 预测值 (power_pred)")
    power_range_timer = y_test_np.max() - y_test_np.min()
    mae_timer, rmse_timer, nmae_timer, nrmse_timer, r2_timer = compute_metrics(
        y_test_np, timer_pred_test, power_range_timer
    )
    print(f"Timer Raw    -> MAE:{mae_timer:.4f} RMSE:{rmse_timer:.4f} NMAE:{nmae_timer:.4f} NRMSE:{nrmse_timer:.4f} R2:{r2_timer:.4f}")

    # 2. 最终模型评估
    print("\n最终模型 (双分支 MLP) 评估")
    power_range_model = y_test_np.max() - y_test_np.min()
    mae_model, rmse_model, nmae_model, nrmse_model, r2_model = compute_metrics(
        y_test_np, pred_test, power_range_model
    )
    print(f"Our Model    -> MAE:{mae_model:.4f} RMSE:{rmse_model:.4f} NMAE:{nmae_model:.4f} NRMSE:{nrmse_model:.4f} R2:{r2_model:.4f}")

    print("\n不保存模型权重和结果文件。")

if __name__ == "__main__":
    main()