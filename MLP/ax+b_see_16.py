import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
PRED_DIR = "/root/timer+exo/pred"
STATION = "station00"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 150
PATIENCE = 15
LR = 1e-4
WEIGHT_DECAY = 1e-5

# 两种窗口设置
WINDOW_CONFIGS = [
    {"name": "窗口96 (步长96)", "window": 96, "stride": 96},
    {"name": "窗口16 (步长1)",  "window": 16, "stride": 1},
]

FEAT_DIM = 8

# ==================== 条件 Affine 模型 ====================
class ConditionalAffine(nn.Module):
    def __init__(self, in_features=FEAT_DIM, hidden_dim=32, window=96):
        super().__init__()
        # 输入形状: (batch, window, in_features)
        self.window = window
        self.fc = nn.Sequential(
            nn.Linear(window * in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, window * 2)   # 输出 (a, b) 对每个时间步
        )
    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        ab = self.fc(x_flat)                     # (batch, window*2)
        ab = ab.view(batch_size, self.window, 2) # (batch, window, 2)
        a = torch.sigmoid(ab[:, :, 0:1]) * 2.0   # (0,2) 范围，初始约1
        b = ab[:, :, 1:2]
        return a, b

# ==================== 评估指标 ====================
def compute_metrics(true, pred, name=""):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    power_range = true.max() - true.min()
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    print(f"{name:25s} | MAE:{mae:7.4f} | RMSE:{rmse:7.4f} | NMAE:{nmae:6.4f} | NRMSE:{nrmse:6.4f} | R2:{r2:6.4f}")
    return mae, rmse, nmae, nrmse, r2

# ==================== 数据加载与窗口构建 ====================
def load_windows(station, window, stride):
    csv_path = os.path.join(PRED_DIR, f"{station}_timer_pred_with_info.csv")
    df = pd.read_csv(csv_path)
    df = df.sort_values('datetime').reset_index(drop=True)
    feat_cols = ['power_pred', 'nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature',
                 'nwp_humidity', 'nwp_windspeed', 'nwp_winddirection', 'nwp_pressure']
    X = df[feat_cols].values.astype(np.float32)
    y = df['power_true'].values.astype(np.float32)
    n = len(X)
    X_windows, y_windows = [], []
    for start in range(0, n - window + 1, stride):
        X_windows.append(X[start:start+window])
        y_windows.append(y[start:start+window])
    return np.array(X_windows), np.array(y_windows)

# ==================== 训练与评估 ====================
def train_and_evaluate(cfg):
    print(f"\n{'='*70}")
    print(f"训练配置: {cfg['name']} (window={cfg['window']}, stride={cfg['stride']})")
    X_all, y_all = load_windows(STATION, cfg['window'], cfg['stride'])
    n_samples = len(X_all)
    print(f"总窗口数: {n_samples}")

    # 时间顺序 8:1:1 划分
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    X_train, X_val, X_test = X_all[:train_end], X_all[train_end:val_end], X_all[val_end:]
    y_train, y_val, y_test = y_all[:train_end], y_all[train_end:val_end], y_all[val_end:]

    # 输入归一化 (基于训练集)
    X_train_flat = X_train.reshape(-1, FEAT_DIM)
    feat_mean = X_train_flat.mean(axis=0, keepdims=True)
    feat_std = X_train_flat.std(axis=0, keepdims=True) + 1e-8
    def norm(X):
        shape = X.shape
        flat = X.reshape(-1, FEAT_DIM)
        flat_norm = (flat - feat_mean) / feat_std
        return flat_norm.reshape(shape)
    X_train_norm = norm(X_train)
    X_val_norm = norm(X_val)
    X_test_norm = norm(X_test)

    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

    model = ConditionalAffine(in_features=FEAT_DIM, hidden_dim=32, window=cfg['window']).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            a, b = model(Xb)
            timer_pred = Xb[:, :, 0:1]            # (batch, window, 1)
            pred = a * timer_pred + b
            loss = criterion(pred.squeeze(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                a, b = model(Xb)
                timer_pred = Xb[:, :, 0:1]
                pred = a * timer_pred + b
                loss = criterion(pred.squeeze(), yb)
                val_loss += loss.item() * len(Xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if (epoch+1) % 20 == 0:
            print(f"{cfg['name']:15s} Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"{cfg['name']} Early stopping at epoch {epoch+1}")
                break

    # 测试集评估
    model.eval()
    with torch.no_grad():
        a_test, b_test = model(X_test_t)
        timer_pred_test = X_test_t[:, :, 0:1]
        pred_test = (a_test * timer_pred_test + b_test).cpu().numpy().squeeze(-1)
    y_test_np = y_test_t.cpu().numpy()
    pred_flat = pred_test.flatten()
    true_flat = y_test_np.flatten()

    # Timer 原始预测（作为基线）
    timer_raw = X_test[:, :, 0].flatten()
    print(f"\n--- {cfg['name']} 评估 ---")
    compute_metrics(true_flat, timer_raw, "Timer原始(测试集)")
    compute_metrics(true_flat, pred_flat, f"条件Affine ({cfg['name']})")
    return

def main():
    print(f"站点: {STATION} | 对比不同窗口长度")
    for cfg in WINDOW_CONFIGS:
        train_and_evaluate(cfg)
    print("\n所有实验完成，不保存任何文件。")

if __name__ == "__main__":
    main()