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
LR = 5e-4                # 降低学习率适应更大模型
WEIGHT_DECAY = 1e-4      # 增加权重衰减

# 增强后的隐藏层配置（更深更宽）
HIDDEN_DIMS = {
    1:  [256, 128, 64],   # 输入8 -> 256 -> 128 -> 64 -> 输出1
    16: [256, 128, 64],   # 输入128 -> 256 -> 128 -> 64 -> 输出16
    96: [256, 128, 64]    # 输入768 -> 256 -> 128 -> 64 -> 输出96
}
DROPOUT = {
    1:  0.1,              # 逐点加入轻微dropout
    16: 0.1,
    96: 0.2
}

# ===================== 评估指标 =====================
def compute_metrics(true, pred, power_range):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    return mae, rmse, nmae, nrmse, r2

# ===================== 模型定义 =====================
class FlattenMLP(nn.Module):
    def __init__(self, input_len, feat_dim=8, hidden_dims=[512, 256], dropout=0.0):
        super().__init__()
        self.input_dim = input_len * feat_dim
        self.output_dim = input_len
        layers = []
        prev_dim = self.input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, input_len, feat_dim)
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return self.net(x_flat)   # (batch, output_len)

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
    X, y = [], []
    n = len(features)
    for i in range(n - seq_len + 1):
        X.append(features[i:i+seq_len])
        y.append(targets[i:i+seq_len])
    return np.array(X), np.array(y)

# ===================== 训练与评估 =====================
def train_and_evaluate(seq_len):
    print(f"\n{'='*60}")
    print(f"序列长度: {seq_len} (输入 {seq_len}点 → 输出 {seq_len}点)")

    features, targets = load_station_data(STATION)
    X_all, y_all = create_sequences(features, targets, seq_len)
    n_samples = len(X_all)

    # 时间顺序划分 8:1:1
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all[val_end:], y_all[val_end:]
    print(f"样本数: 训练 {len(X_train)}, 验证 {len(X_val)}, 测试 {len(X_test)}")

    # --- 输入归一化（基于训练集）---
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

    # 转换为Tensor
    X_train_t = torch.tensor(X_train_norm).to(DEVICE)
    y_train_t = torch.tensor(y_train).to(DEVICE)
    X_val_t = torch.tensor(X_val_norm).to(DEVICE)
    y_val_t = torch.tensor(y_val).to(DEVICE)
    X_test_t = torch.tensor(X_test_norm).to(DEVICE)
    y_test_t = torch.tensor(y_test).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

    # 构建模型
    model = FlattenMLP(input_len=seq_len, feat_dim=8,
                       hidden_dims=HIDDEN_DIMS[seq_len],
                       dropout=DROPOUT[seq_len]).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 训练循环
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

        if (epoch + 1) % 20 == 0:
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
    pred_flat = pred_test.flatten()
    true_flat = y_test_np.flatten()
    power_range = true_flat.max() - true_flat.min()
    mae, rmse, nmae, nrmse, r2 = compute_metrics(true_flat, pred_flat, power_range)
    print(f"测试集结果: MAE={mae:.4f}, RMSE={rmse:.4f}, NMAE={nmae:.4f}, NRMSE={nrmse:.4f}, R2={r2:.4f}")
    return mae, rmse, nmae, nrmse, r2

# ===================== 主程序 =====================
def main():
    print(f"单站 {STATION} 不同输入长度 MLP 对比 (增强版: 隐藏层更深更宽)")
    seq_lengths = [1, 16, 96]
    results = []
    for L in seq_lengths:
        metrics = train_and_evaluate(L)
        results.append({'seq_len': L,
                        'MAE': metrics[0],
                        'RMSE': metrics[1],
                        'NMAE': metrics[2],
                        'NRMSE': metrics[3],
                        'R2': metrics[4]})
    print("\n" + "="*80)
    print("汇总对比 (测试集):")
    print(f"{'SeqLen':<8} {'MAE':>10} {'RMSE':>10} {'NMAE':>10} {'NRMSE':>10} {'R2':>10}")
    for r in results:
        print(f"{r['seq_len']:<8} {r['MAE']:10.4f} {r['RMSE']:10.4f} {r['NMAE']:10.4f} {r['NRMSE']:10.4f} {r['R2']:10.4f}")

if __name__ == "__main__":
    main()