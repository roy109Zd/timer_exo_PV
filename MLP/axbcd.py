import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random

# ==================== 固定随机种子 ====================
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

# ==================== 配置 ====================
PRED_DIR = "/root/timer+exo/pred"
STATION = "station01"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5

WINDOW = 96
STRIDE = 96
FEAT_DIM = 8

# ==================== 条件 Affine 模型（四个参数） ====================
class ConditionalAffine(nn.Module):
    """
    输入特征 (timer_pred + 7个NWP)，输出四个标量 a, b, c, d
    最终预测 = a * timer_pred / c + b - d
    """
    def __init__(self, in_features=FEAT_DIM, hidden_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)   # 输出 [a, b, c, d]
        )
    def forward(self, x):
        # x: (batch, window, in_features) -> 逐点预测 a,b,c,d
        abcd = self.fc(x)                # (batch, window, 4)
        a = abcd[..., 0:1]               # (batch, window, 1)
        b = abcd[..., 1:2]
        c = abcd[..., 2:3]
        d = abcd[..., 3:4]
        # 对 a 施加 softplus 确保为正，并加上偏置使初始 a 接近1
        a = nn.functional.softplus(a) + 0.5
        # 对 c 施加 softplus 确保为正，并加一个小常数防止除零
        c = nn.functional.softplus(c) + 1e-2
        # d 可以是任意实数，不做额外约束
        return a, b, c, d

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
    print(f"单站 {STATION} | 条件 Affine 变换 (y = a*x/c + b - d) | 窗口={WINDOW}")
    X_all, y_all = load_windows(STATION, WINDOW, STRIDE)
    n_samples = len(X_all)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    X_train, X_val, X_test = X_all[:train_end], X_all[train_end:val_end], X_all[val_end:]
    y_train, y_val, y_test = y_all[:train_end], y_all[train_end:val_end], y_all[val_end:]

    # 提取 Timer 预测值（特征第一列）
    timer_train = X_train[:, :, 0:1]   # (n, 96, 1)
    timer_val   = X_val[:, :, 0:1]
    timer_test  = X_test[:, :, 0:1]

    # 输入归一化（基于训练集所有时间步）
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

    # 转换为 Tensor
    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    timer_train_t = torch.tensor(timer_train, dtype=torch.float32, device=DEVICE)

    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)
    timer_val_t = torch.tensor(timer_val, dtype=torch.float32, device=DEVICE)

    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)
    timer_test_t = torch.tensor(timer_test, dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t, timer_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t, timer_val_t), batch_size=BATCH_SIZE, shuffle=False)

    model = ConditionalAffine(in_features=FEAT_DIM, hidden_dim=16).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for Xb, yb, tb in train_loader:
            optimizer.zero_grad()
            a, b, c, d = model(Xb)          # 每个 shape: (batch, window, 1)
            pred = a * tb / (c + 1e-6) + b - d   # 新公式
            loss = criterion(pred.view(-1), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb, tb in val_loader:
                a, b, c, d = model(Xb)
                pred = a * tb / (c + 1e-6) + b - d
                loss = criterion(pred.view(-1), yb.view(-1))
                val_loss += loss.item() * len(Xb)
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

    # 测试评估
    model.eval()
    with torch.no_grad():
        a_test, b_test, c_test, d_test = model(X_test_t)   # 每个 (n_test, 96, 1)
        pred_test = (a_test * timer_test_t / (c_test + 1e-6) + b_test - d_test).cpu().numpy().squeeze(-1)
        a_test_np = a_test.cpu().numpy().squeeze(-1)
        b_test_np = b_test.cpu().numpy().squeeze(-1)
        c_test_np = c_test.cpu().numpy().squeeze(-1)
        d_test_np = d_test.cpu().numpy().squeeze(-1)

    y_test_np = y_test_t.cpu().numpy()
    timer_test_np = timer_test_t.cpu().numpy().squeeze(-1)

    flat_timer = timer_test_np.flatten()
    flat_affine = pred_test.flatten()
    flat_true = y_test_np.flatten()

    print("\n" + "="*60)
    compute_metrics(flat_true, flat_timer, "Timer原始预测")
    compute_metrics(flat_true, flat_affine, "条件Affine修正")

    # 打印四个参数的统计
    a_flat = a_test_np.flatten()
    b_flat = b_test_np.flatten()
    c_flat = c_test_np.flatten()
    d_flat = d_test_np.flatten()

    print("\n" + "="*60)
    print(f"乘性因子 a 的统计 (测试集所有预测点):")
    print(f"  均值: {a_flat.mean():.4f} | 标准差: {a_flat.std():.4f} | 中位数: {np.median(a_flat):.4f}")
    print(f"  最小值: {a_flat.min():.4f} | 最大值: {a_flat.max():.4f}")

    print(f"\n加性因子 b 的统计:")
    print(f"  均值: {b_flat.mean():.4f} | 标准差: {b_flat.std():.4f} | 中位数: {np.median(b_flat):.4f}")
    print(f"  最小值: {b_flat.min():.4f} | 最大值: {b_flat.max():.4f}")

    print(f"\n除数因子 c 的统计 (c > 0):")
    print(f"  均值: {c_flat.mean():.4f} | 标准差: {c_flat.std():.4f} | 中位数: {np.median(c_flat):.4f}")
    print(f"  最小值: {c_flat.min():.4f} | 最大值: {c_flat.max():.4f}")

    print(f"\n减性因子 d 的统计:")
    print(f"  均值: {d_flat.mean():.4f} | 标准差: {d_flat.std():.4f} | 中位数: {np.median(d_flat):.4f}")
    print(f"  最小值: {d_flat.min():.4f} | 最大值: {d_flat.max():.4f}")

    print(f"\n随机种子 = {SEED} | 不保存任何结果文件。")

if __name__ == "__main__":
    main()