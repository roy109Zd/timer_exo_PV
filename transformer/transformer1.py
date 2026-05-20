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
BATCH_SIZE = 128
EPOCHS = 150
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-5

# Transformer 超参数
SEQ_LEN = 16          # 历史窗口长度 (输入多少个时间步)
D_MODEL = 64          # Transformer 的嵌入维度
NHEAD = 4             # 注意力头数
NUM_LAYERS = 3        # 编码器层数
DIM_FEEDFORWARD = 128 # 前馈网络维度
DROPOUT = 0.1

# 特征维度 (power_pred + 7个NWP)
FEAT_DIM = 8

# ===================== 评估指标 =====================
def compute_metrics(true, pred, power_range):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    return mae, rmse, nmae, nrmse, r2

# ===================== Transformer 模型 =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class PointwiseTransformer(nn.Module):
    """输入历史序列，输出最后一个时间步的预测值"""
    def __init__(self, feat_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, feat_dim)
        x = self.input_proj(x)                # (batch, seq_len, d_model)
        x = self.pos_encoder(x)               # 添加位置编码
        x = self.transformer(x)               # (batch, seq_len, d_model)
        last_out = x[:, -1, :]                # 取最后一个时间步 (batch, d_model)
        out = self.output_proj(last_out)      # (batch, 1)
        return out.squeeze(-1)                # (batch,)

# ===================== 数据加载 =====================
def load_station_data(station_name):
    csv_path = os.path.join(PRED_DIR, f"{station_name}_timer_pred_with_info.csv")
    df = pd.read_csv(csv_path)
    nwp_cols = ['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 'nwp_humidity',
                'nwp_windspeed', 'nwp_winddirection', 'nwp_pressure']
    features = df[['power_pred'] + nwp_cols].values.astype(np.float32)
    targets = df['power_true'].values.astype(np.float32)
    return features, targets

def create_sequences(features, targets, seq_len):
    """输入 seq_len 个历史特征，输出下一个点的目标值（即最后一个时间步的功率）"""
    X, y = [], []
    n = len(features)
    # 需要至少 seq_len 个历史点才能预测下一个点
    for i in range(n - seq_len):
        X.append(features[i:i+seq_len])   # (seq_len, 8)
        y.append(targets[i+seq_len])      # 下一个时间点的真实功率
    return np.array(X), np.array(y)

# ===================== 训练与评估 =====================
def main():
    print(f"单站 {STATION} 逐点 Transformer 预测 (历史窗口长度={SEQ_LEN})")
    features, targets = load_station_data(STATION)
    X_all, y_all = create_sequences(features, targets, SEQ_LEN)
    n_samples = len(X_all)
    print(f"总样本数: {n_samples}")

    # 时间顺序划分 8:1:1
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all[val_end:], y_all[val_end:]
    print(f"训练样本: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")

    # 输入归一化（基于训练集）
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_mean = X_train_flat.mean(axis=0, keepdims=True)
    X_std = X_train_flat.std(axis=0, keepdims=True) + 1e-8
    def normalize_X(X):
        shape = X.shape
        X_flat = X.reshape(len(X), -1)
        X_norm = (X_flat - X_mean) / X_std
        return X_norm.reshape(shape)
    X_train_norm = normalize_X(X_train)
    X_val_norm = normalize_X(X_val)
    X_test_norm = normalize_X(X_test)

    # 转换为 Tensor
    X_train_t = torch.tensor(X_train_norm).to(DEVICE)
    y_train_t = torch.tensor(y_train).to(DEVICE)
    X_val_t = torch.tensor(X_val_norm).to(DEVICE)
    y_val_t = torch.tensor(y_val).to(DEVICE)
    X_test_t = torch.tensor(X_test_norm).to(DEVICE)
    y_test_t = torch.tensor(y_test).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

    # 构建模型
    model = PointwiseTransformer(
        feat_dim=FEAT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 训练
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)
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
                pred = model(Xb)
                loss = criterion(pred, yb)
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

    # 测试
    model.eval()
    with torch.no_grad():
        pred_test = model(X_test_t).cpu().numpy()
    y_test_np = y_test_t.cpu().numpy()
    power_range = y_test_np.max() - y_test_np.min()
    mae, rmse, nmae, nrmse, r2 = compute_metrics(y_test_np, pred_test, power_range)
    print(f"\n测试集结果: MAE={mae:.4f}, RMSE={rmse:.4f}, NMAE={nmae:.4f}, NRMSE={nrmse:.4f}, R2={r2:.4f}")

if __name__ == "__main__":
    main()