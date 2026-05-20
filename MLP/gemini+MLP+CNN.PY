import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================== 配置 ====================
PRED_DIR = "/root/timer+exo/pred"          # 存放 stationXX_timer_pred_with_info.csv
STATIONS = ["station00"]                   # 可扩展为多个，此处只演示 station00
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 150
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-5

WINDOW_LEN = 96            # 每个样本包含 96 个连续时间点
FEAT_DIM = 8               # [power_pred] + 7 个 NWP
STRIDE = 96                # 不重叠窗口（避免过度相关），也可改为1，但数据量大

# ==================== 模型定义 ====================
class FusionMLP(nn.Module):
    """逐点共享权重的 MLP，预测残差（修正量）"""
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
        # x: (batch, WINDOW_LEN, FEAT_DIM)
        timer_pred = x[:, :, 0:1]                 # 第0个特征为 power_pred
        delta = self.mlp(x)                       # (batch, WINDOW_LEN, 1)
        out = timer_pred + delta
        return torch.relu(out)                    # 功率非负

class FusionCNN(nn.Module):
    """1D-CNN，利用时间邻域信息预测残差"""
    def __init__(self, in_channels=FEAT_DIM, hidden_channels=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (batch, WINDOW_LEN, FEAT_DIM)
        timer_pred = x[:, :, 0:1]                 # (batch, 96, 1)
        x_conv = x.transpose(1, 2)                # (batch, FEAT_DIM, 96)
        hidden = self.relu(self.conv1(x_conv))    # (batch, hidden_channels, 96)
        delta = self.conv2(hidden)                # (batch, 1, 96)
        delta = delta.transpose(1, 2)             # (batch, 96, 1)
        out = timer_pred + delta
        return torch.relu(out)

# ==================== 评估指标 ====================
def compute_metrics(true, pred, power_range):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    return mae, rmse, nmae, nrmse, r2

# ==================== 数据加载与窗口构建 ====================
def load_sequence_data(station_name, window=96, stride=96):
    csv_path = os.path.join(PRED_DIR, f"{station_name}_timer_pred_with_info.csv")
    df = pd.read_csv(csv_path)
    nwp_cols = ['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 'nwp_humidity',
                'nwp_windspeed', 'nwp_winddirection', 'nwp_pressure']
    # 提取特征和标签
    feat_cols = ['power_pred'] + nwp_cols
    X_raw = df[feat_cols].values.astype(np.float32)   # (N, 8)
    y_raw = df['power_true'].values.astype(np.float32) # (N,)

    n = len(X_raw)
    X_seq, y_seq = [], []
    for start in range(0, n - window + 1, stride):
        X_seq.append(X_raw[start:start+window])   # (window, 8)
        y_seq.append(y_raw[start:start+window])   # (window,)
    X_seq = np.array(X_seq)   # (num_samples, window, 8)
    y_seq = np.array(y_seq)   # (num_samples, window)
    return X_seq, y_seq

# ==================== 训练与评估 ====================
def train_and_evaluate(model, train_loader, val_loader, test_loader, model_name):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb).squeeze(-1)   # (batch, window)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = model(Xb).squeeze(-1)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(Xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if (epoch+1) % 20 == 0:
            print(f"{model_name} Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"{model_name} Early stopping at epoch {epoch+1}")
                break

    # 测试
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE)
            pred = model(Xb).squeeze(-1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(yb.numpy())
    all_pred = np.concatenate(all_pred, axis=0).flatten()
    all_true = np.concatenate(all_true, axis=0).flatten()
    power_range = all_true.max() - all_true.min()
    mae, rmse, nmae, nrmse, r2 = compute_metrics(all_true, all_pred, power_range)
    print(f"{model_name} 测试集: MAE={mae:.4f}, RMSE={rmse:.4f}, NMAE={nmae:.4f}, NRMSE={nrmse:.4f}, R2={r2:.4f}")
    return mae, rmse, nmae, nrmse, r2

# ==================== 主程序 ====================
def main():
    station = STATIONS[0]
    print(f"处理站点: {station} (窗口长度={WINDOW_LEN}, 步长={STRIDE})")
    X_seq, y_seq = load_sequence_data(station, window=WINDOW_LEN, stride=STRIDE)
    n_samples = len(X_seq)
    print(f"总样本数: {n_samples}")

    # 时间顺序 8:1:1 划分
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    X_train, X_val, X_test = X_seq[:train_end], X_seq[train_end:val_end], X_seq[val_end:]
    y_train, y_val, y_test = y_seq[:train_end], y_seq[train_end:val_end], y_seq[val_end:]

    # 输入归一化 (基于训练集，对每个特征维度独立)
    # 将 X_train 重塑为 (所有时间步, 8)
    X_train_flat = X_train.reshape(-1, FEAT_DIM)
    feat_mean = X_train_flat.mean(axis=0, keepdims=True)
    feat_std = X_train_flat.std(axis=0, keepdims=True) + 1e-8
    def norm(X):
        orig_shape = X.shape
        flat = X.reshape(-1, FEAT_DIM)
        flat_norm = (flat - feat_mean) / feat_std
        return flat_norm.reshape(orig_shape)
    X_train_norm = norm(X_train)
    X_val_norm = norm(X_val)
    X_test_norm = norm(X_test)

    # 转换为 Tensor
    X_train_t = torch.tensor(X_train_norm).float()
    y_train_t = torch.tensor(y_train).float()
    X_val_t = torch.tensor(X_val_norm).float()
    y_val_t = torch.tensor(y_val).float()
    X_test_t = torch.tensor(X_test_norm).float()
    y_test_t = torch.tensor(y_test).float()

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    # 评估 Timer 原始预测值 (直接从测试集的 power_pred 列)
    # 注意：测试集中的 power_pred 就是 Timer 对每个时间点的预测值
    # 从 X_test_norm 反推原始 power_pred？更简单：直接从原始 X_test 中取第0列
    timer_pred_raw = X_test[:, :, 0].flatten()   # 原始尺度（未归一化）
    y_true_flat = y_test.flatten()
    power_range_timer = y_true_flat.max() - y_true_flat.min()
    mae_t, rmse_t, nmae_t, nrmse_t, r2_t = compute_metrics(y_true_flat, timer_pred_raw, power_range_timer)
    print("\n" + "="*60)
    print("【Timer 原始预测值】评估 (测试集):")
    print(f"MAE={mae_t:.4f}, RMSE={rmse_t:.4f}, NMAE={nmae_t:.4f}, NRMSE={nrmse_t:.4f}, R2={r2_t:.4f}")

    # 训练 FusionMLP
    print("\n" + "="*60)
    print("训练 FusionMLP ...")
    mlp_model = FusionMLP(in_features=FEAT_DIM, hidden_dim=32)
    train_and_evaluate(mlp_model, train_loader, val_loader, test_loader, "FusionMLP")

    # 训练 FusionCNN
    print("\n" + "="*60)
    print("训练 FusionCNN ...")
    cnn_model = FusionCNN(in_channels=FEAT_DIM, hidden_channels=32)
    train_and_evaluate(cnn_model, train_loader, val_loader, test_loader, "FusionCNN")

    print("\n不保存任何文件，脚本结束。")

if __name__ == "__main__":
    main()