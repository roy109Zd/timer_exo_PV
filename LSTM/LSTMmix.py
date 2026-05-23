import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random

# ========== 固定随机种子 ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ========== 配置 ==========
DATA_DIR = "/root/timer/甘肃光伏"
ALL_STATIONS = [f"station{i:02d}" for i in range(10)]   # 00-09
TARGET_STATION = "station00"                            # 评估目标站
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128        # 合并后样本多，可以增大batch
EPOCHS = 100
LR = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0
LOOKBACK_POINTS = 96    # 历史1天
PRED_LEN = 96
NWP_DIM = 7

# ========== 1. 加载单个站点的数据并标准化 ==========
def load_and_scale(station):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{station}.csv"))
    power = df['power'].values.astype(np.float32)
    nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
                'nwp_windspeed','nwp_winddirection','nwp_pressure']
    nwp = df[nwp_cols].values.astype(np.float32)
    # 功率标准化
    power_scaler = StandardScaler()
    power_scaled = power_scaler.fit_transform(power.reshape(-1, 1)).flatten()
    # NWP 标准化（每个变量独立）
    nwp_scaler = StandardScaler()
    nwp_scaled = nwp_scaler.fit_transform(nwp)
    return power_scaled, nwp_scaled, power_scaler, nwp_scaler

# ========== 2. 构造单个站点的样本（返回 pandas 格式的时间索引，便于对齐） ==========
def create_samples_with_time(station, lookback_len, pred_len):
    """
    返回:
        samples_df: DataFrame, 包含 start_idx, hist, nwp_future, label
    """
    power_scaled, nwp_scaled, _, _ = load_and_scale(station)
    total_len = len(power_scaled)
    stride = pred_len
    samples = []
    for start in range(0, total_len - lookback_len - pred_len + 1, stride):
        hist = power_scaled[start:start+lookback_len]
        nwp_future = nwp_scaled[start+lookback_len:start+lookback_len+pred_len]
        label = power_scaled[start+lookback_len:start+lookback_len+pred_len]
        samples.append({
            'start_idx': start,
            'hist': hist,
            'nwp_future': nwp_future,
            'label': label
        })
    return samples

# ========== 3. 合并所有站点的样本（直接拼接，不按时间对齐，因为每个站点独立样本） ==========
def merge_all_stations_samples(stations, lookback_len, pred_len):
    """
    将所有站点的所有样本合并为一个大的数据集
    返回:
        hist_list: list of numpy arrays
        nwp_list: list of numpy arrays
        label_list: list of numpy arrays
    """
    all_hist = []
    all_nwp = []
    all_label = []
    for st in stations:
        samples = create_samples_with_time(st, lookback_len, pred_len)
        for s in samples:
            all_hist.append(s['hist'])
            all_nwp.append(s['nwp_future'])
            all_label.append(s['label'])
        print(f"{st} 贡献样本数: {len(samples)}")
    # 转换为 tensor
    hist_tensor = torch.tensor(np.array(all_hist), dtype=torch.float32)
    nwp_tensor = torch.tensor(np.array(all_nwp), dtype=torch.float32)
    label_tensor = torch.tensor(np.array(all_label), dtype=torch.float32)
    return hist_tensor, nwp_tensor, label_tensor

# ========== 4. 构建目标站的测试集（仅后10%时间段） ==========
def create_target_testset(station, lookback_len, pred_len, test_ratio=0.1):
    """
    返回目标站后 test_ratio 时间段的测试样本
    """
    power_scaled, nwp_scaled, power_scaler, _ = load_and_scale(station)
    total_len = len(power_scaled)
    # 计算用于测试的起始索引范围
    total_samples = (total_len - lookback_len - pred_len) // pred_len + 1
    test_start_sample = int(total_samples * (1 - test_ratio))
    test_start_idx = test_start_sample * pred_len   # 原始序列中的起始索引
    samples = []
    for start in range(test_start_idx, total_len - lookback_len - pred_len + 1, pred_len):
        hist = power_scaled[start:start+lookback_len]
        nwp_future = nwp_scaled[start+lookback_len:start+lookback_len+pred_len]
        label = power_scaled[start+lookback_len:start+lookback_len+pred_len]
        samples.append((hist, nwp_future, label))
    if len(samples) == 0:
        raise ValueError("测试集为空，请减小 test_ratio 或检查数据长度")
    hist_test = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)
    nwp_test = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)
    label_test = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.float32)
    return hist_test, nwp_test, label_test, power_scaler

# ========== 5. 基础 LSTM 模型（与之前相同） ==========
class BasicLSTM(nn.Module):
    def __init__(self, lookback_len, pred_len, nwp_dim, hidden_size, num_layers):
        super().__init__()
        self.pred_len = pred_len
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + nwp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, hist, nwp_future):
        B = hist.shape[0]
        _, (h_n, _) = self.lstm(hist.unsqueeze(-1))
        context = h_n[-1]
        context_exp = context.unsqueeze(1).expand(-1, self.pred_len, -1)
        x = torch.cat([context_exp, nwp_future], dim=-1)
        out = self.mlp(x.view(-1, x.shape[-1]))
        out = out.view(B, self.pred_len)
        return out

# ========== 6. 评估指标 ==========
def compute_metrics(y_true, y_pred, power_scaler):
    y_true_orig = power_scaler.inverse_transform(y_true.reshape(-1,1)).flatten()
    y_pred_orig = power_scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2 = r2_score(y_true_orig, y_pred_orig)
    power_range = y_true_orig.max() - y_true_orig.min()
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    return mae, rmse, nmae, nrmse, r2

# ========== 7. 训练函数（带早停和轮数显示） ==========
def train_model(model, train_loader, val_loader, test_loader, epochs, lr, device, power_scaler, patience=10):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_val_loss = float('inf')
    best_state = None
    counter = 0
    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        for hist, nwp_future, label in train_loader:
            hist = hist.to(device)
            nwp_future = nwp_future.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out = model(hist, nwp_future)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hist, nwp_future, label in val_loader:
                hist = hist.to(device)
                nwp_future = nwp_future.to(device)
                label = label.to(device)
                out = model(hist, nwp_future)
                loss = criterion(out, label)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
            print(f"  -> 验证损失下降，保存最佳模型")
        else:
            counter += 1
            if counter >= patience:
                print(f"早停触发: 验证损失连续 {patience} 轮未改善，停止训练")
                break
    if best_state:
        model.load_state_dict(best_state)
        print(f"加载最佳模型，验证损失: {best_val_loss:.6f}")
    # 测试
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for hist, nwp_future, label in test_loader:
            hist = hist.to(device)
            nwp_future = nwp_future.to(device)
            pred = model(hist, nwp_future).cpu().numpy()
            true = label.numpy()
            all_pred.append(pred)
            all_true.append(true)
    all_pred = np.concatenate(all_pred, axis=0).flatten()
    all_true = np.concatenate(all_true, axis=0).flatten()
    return compute_metrics(all_true, all_pred, power_scaler)

# ========== 8. 主程序 ==========
def main():
    print(f"设备: {DEVICE}")
    print(f"训练站点: {ALL_STATIONS} (共{len(ALL_STATIONS)}个)")
    print(f"目标评估站: {TARGET_STATION}")
    print(f"历史点数: {LOOKBACK_POINTS} (1天), 预测点数: {PRED_LEN}")
    
    # 1. 合并所有站点的训练数据（全部样本，不划分训练/验证，因为我们后面要随机划分）
    print("\n合并所有站点样本...")
    hist_all, nwp_all, label_all = merge_all_stations_samples(ALL_STATIONS, LOOKBACK_POINTS, PRED_LEN)
    total_samples = hist_all.shape[0]
    print(f"总样本数: {total_samples}")
    
    # 2. 从合并数据中随机划分训练集和验证集（80% 训练，20% 验证）
    indices = np.random.permutation(total_samples)
    train_size = int(0.8 * total_samples)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    train_hist, train_nwp, train_label = hist_all[train_idx], nwp_all[train_idx], label_all[train_idx]
    val_hist, val_nwp, val_label = hist_all[val_idx], nwp_all[val_idx], label_all[val_idx]
    print(f"训练样本: {len(train_idx)}, 验证样本: {len(val_idx)}")
    
    # 3. 创建目标站的测试集（最后10%时间段）
    print(f"\n构建目标站 {TARGET_STATION} 的测试集（后10%时间）...")
    test_hist, test_nwp, test_label, target_scaler = create_target_testset(
        TARGET_STATION, LOOKBACK_POINTS, PRED_LEN, test_ratio=0.1)
    print(f"测试样本数: {test_hist.shape[0]}")
    
    # 4. DataLoader
    train_ds = TensorDataset(train_hist, train_nwp, train_label)
    val_ds   = TensorDataset(val_hist,   val_nwp,   val_label)
    test_ds  = TensorDataset(test_hist,  test_nwp,  test_label)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. 构建模型
    model = BasicLSTM(LOOKBACK_POINTS, PRED_LEN, NWP_DIM, HIDDEN_SIZE, NUM_LAYERS)
    
    # 6. 训练与评估
    mae, rmse, nmae, nrmse, r2 = train_model(model, train_loader, val_loader, test_loader,
                                             EPOCHS, LR, DEVICE, target_scaler, patience=10)
    print("\n===== 多站联合训练 + 单站评估结果 =====")
    print(f"目标站: {TARGET_STATION}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"NMAE: {nmae:.4f}")
    print(f"NRMSE:{nrmse:.4f}")
    print(f"R²:   {r2:.4f}")

if __name__ == "__main__":
    main()