import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
# ==================== 配置 ====================
PRED_DIR = "/root/timer+exo/pred"
STATION = "station00"                     # 单站测试，可改为多站
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 1500
PATIENCE = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
LAMBDA_DIFF = 0.3                    # 差分正则项系数

WINDOW = 96
STRIDE = 96
FEAT_DIM = 8
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)
# ==================== 模型 ====================
class ResidualMLP(nn.Module):
    def __init__(self, in_features=FEAT_DIM, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.mlp(x)   # (batch, window, 1)

# ==================== 损失函数（加入差分正则） ====================
def loss_with_diff(pred_delta, target_delta, timer_pred):
    # pred_delta, target_delta: (batch, window)
    mse = nn.functional.mse_loss(pred_delta, target_delta)
    # 最终预测值 = timer_pred + pred_delta
    final_pred = timer_pred + pred_delta
    # 计算相邻时间点的差分 (沿window维度)
    diff = final_pred[:, 1:] - final_pred[:, :-1]   # (batch, window-1)
    diff_reg = torch.abs(diff).mean()               # L1 差分正则
    loss = mse + LAMBDA_DIFF * diff_reg
    return loss, mse, diff_reg

# ==================== 评估指标 ====================
def compute_metrics(true, pred, name=""):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    power_range = true.max() - true.min()
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    print(f"{name:20s} | MAE:{mae:7.4f} | RMSE:{rmse:7.4f} | NMAE:{nmae:6.4f} | NRMSE:{nrmse:6.4f} | R2:{r2:6.4f}")
    return mae, rmse, nmae, nrmse, r2

# ==================== 数据加载 ====================
def load_windows(station_name, window=96, stride=96):
    csv_path = os.path.join(PRED_DIR, f"{station_name}_timer_pred_with_info.csv")
    df = pd.read_csv(csv_path)
    nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
                'nwp_windspeed','nwp_winddirection','nwp_pressure']
    feat_cols = ['power_pred'] + nwp_cols
    X = df[feat_cols].values.astype(np.float32)
    y = df['power_true'].values.astype(np.float32)
    n = len(X)
    X_windows, y_windows = [], []
    for start in range(0, n - window + 1, stride):
        X_windows.append(X[start:start+window])
        y_windows.append(y[start:start+window])
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    return X_windows, y_windows

# ==================== 训练主函数 ====================
def main():
    print(f"单站 {STATION} 残差MLP (窗口={WINDOW}, 差分正则λ={LAMBDA_DIFF})")
    X_all, y_all = load_windows(STATION, WINDOW, STRIDE)
    n_samples = len(X_all)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    X_train, X_val, X_test = X_all[:train_end], X_all[train_end:val_end], X_all[val_end:]
    y_train, y_val, y_test = y_all[:train_end], y_all[train_end:val_end], y_all[val_end:]

    # 残差标签
    timer_train = X_train[:, :, 0]
    timer_val   = X_val[:, :, 0]
    timer_test  = X_test[:, :, 0]
    residual_train = y_train - timer_train
    residual_val   = y_val   - timer_val
    residual_test  = y_test  - timer_test

    # 输入归一化
    X_train_flat = X_train.reshape(-1, FEAT_DIM)
    feat_mean = X_train_flat.mean(axis=0, keepdims=True)
    feat_std = X_train_flat.std(axis=0, keepdims=True) + 1e-8
    def norm(X):
        shape = X.shape
        flat = X.reshape(-1, FEAT_DIM)
        flat_norm = (flat - feat_mean) / feat_std
        return flat_norm.reshape(shape)
    X_train_norm = norm(X_train)
    X_val_norm   = norm(X_val)
    X_test_norm  = norm(X_test)

    # 转换为Tensor
    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32, device=DEVICE)
    residual_train_t = torch.tensor(residual_train, dtype=torch.float32, device=DEVICE)
    timer_train_t = torch.tensor(timer_train, dtype=torch.float32, device=DEVICE)

    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32, device=DEVICE)
    residual_val_t = torch.tensor(residual_val, dtype=torch.float32, device=DEVICE)
    timer_val_t = torch.tensor(timer_val, dtype=torch.float32, device=DEVICE)

    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32, device=DEVICE)
    timer_test_t = torch.tensor(timer_test, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)  # 用于最终指标

    train_loader = DataLoader(TensorDataset(X_train_t, residual_train_t, timer_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, residual_val_t, timer_val_t), batch_size=BATCH_SIZE, shuffle=False)

    model = ResidualMLP().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_diff = 0.0
        for Xb, rb, tb in train_loader:
            optimizer.zero_grad()
            delta = model(Xb).squeeze(-1)   # (batch, window)
            loss, mse, diff_reg = loss_with_diff(delta, rb, tb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
            train_mse += mse.item() * len(Xb)
            train_diff += diff_reg.item() * len(Xb)
        train_loss /= len(train_loader.dataset)
        train_mse /= len(train_loader.dataset)
        train_diff /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, rb, tb in val_loader:
                delta = model(Xb).squeeze(-1)
                loss, _, _ = loss_with_diff(delta, rb, tb)
                val_loss += loss.item() * len(Xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} (MSE:{train_mse:.6f} Diff:{train_diff:.6f}) | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 测试评估
    model.eval()
    with torch.no_grad():
        delta_test = model(X_test_t).squeeze(-1).cpu().numpy()   # (n_test, window)
    timer_test_np = timer_test
    y_test_np = y_test
    final_pred = timer_test_np + delta_test

    flat_timer = timer_test_np.flatten()
    flat_final = final_pred.flatten()
    flat_true = y_test_np.flatten()

    print("\n" + "="*60)
    compute_metrics(flat_true, flat_timer, "Timer原始预测")
    compute_metrics(flat_true, flat_final, "残差MLP+差分正则")

    # 可选：打印差分正则项的实际贡献（最后一批）
    print(f"\n差分正则系数 λ = {LAMBDA_DIFF}")
    print("不保存任何结果文件。")

if __name__ == "__main__":
    main()