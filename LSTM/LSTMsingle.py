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
import warnings
warnings.filterwarnings('ignore')

# ========== 固定随机种子 ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ========== 全局配置 ==========
DATA_DIR = "/root/timer/甘肃光伏"
STATIONS = [f"station{i:02d}" for i in range(10)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
POINTWISE_BATCH_MULTIPLIER = 4
EPOCHS = 100
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

HISTORY_DAYS = [1, 2, 3, 5, 7]
PREDICT_LENS = [1, 16, 96]
POINTS_PER_DAY = 96

# ========== 1. 数据加载与标准化 ==========
def load_and_scale(station):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{station}.csv"))
    power = df['power'].values.astype(np.float32)
    nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
                'nwp_windspeed','nwp_winddirection','nwp_pressure']
    nwp = df[nwp_cols].values.astype(np.float32)
    power_scaler = StandardScaler()
    power_scaled = power_scaler.fit_transform(power.reshape(-1, 1)).flatten()
    nwp_scaler = StandardScaler()
    nwp_scaled = nwp_scaler.fit_transform(nwp)
    return power_scaled, nwp_scaled, power_scaler, nwp_scaler

# ========== 2. 构造多步样本（滑动窗口，步长=预测长度） ==========
def create_samples(power_scaled, nwp_scaled, lookback_len, pred_len):
    nwp_dim = nwp_scaled.shape[1]
    total_len = len(power_scaled)
    samples = []
    stride = pred_len
    for start in range(0, total_len - lookback_len - pred_len + 1, stride):
        hist = power_scaled[start:start+lookback_len]
        nwp_future = nwp_scaled[start+lookback_len:start+lookback_len+pred_len]
        label = power_scaled[start+lookback_len:start+lookback_len+pred_len]
        samples.append((hist, nwp_future, label))
    if len(samples) == 0:
        return None
    hist_seq = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)
    nwp_future = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)
    label = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.float32)
    return hist_seq, nwp_future, label

# ========== 3. 模型定义 ==========
class MultiStepLSTM(nn.Module):
    def __init__(self, lookback_len, pred_len, nwp_dim, hidden_size, num_layers, dropout, mlp_hidden=256):
        super().__init__()
        self.pred_len = pred_len
        self.nwp_dim = nwp_dim
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + nwp_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.ReLU(),
            nn.Linear(mlp_hidden//2, 1)
        )
    def forward(self, hist, nwp_future):
        B = hist.shape[0]
        lstm_out, (h_n, _) = self.lstm(hist.unsqueeze(-1))
        context = h_n[-1]
        context_exp = context.unsqueeze(1).expand(-1, self.pred_len, -1)
        x = torch.cat([context_exp, nwp_future], dim=-1)
        out = self.mlp(x.view(-1, x.shape[-1]))
        out = out.view(-1, self.pred_len)
        return out

class SingleStepLSTM(nn.Module):
    def __init__(self, lookback_len, nwp_dim, hidden_size, num_layers, dropout, mlp_hidden=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + nwp_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.ReLU(),
            nn.Linear(mlp_hidden//2, 1)
        )
    def forward(self, hist, nwp_t):
        lstm_out, (h_n, _) = self.lstm(hist.unsqueeze(-1))
        context = h_n[-1]
        x = torch.cat([context, nwp_t], dim=-1)
        out = self.mlp(x).squeeze(-1)
        return out

# ========== 4. 评估指标 ==========
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

# ========== 5. 训练：一次性预测模型 ==========
def train_multistep(model, train_loader, val_loader, test_loader, epochs, lr, device, power_scaler):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_val_loss = float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        for hist, nwp_future, label in train_loader:
            hist = hist.to(device)
            nwp_future = nwp_future.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out = model(hist, nwp_future)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
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
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
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

# ========== 6. 训练：滚动单步预测模型（训练用单步数据，测试用多步数据） ==========
def train_singlestep(model, train_loader, val_loader, test_multistep_loader, epochs, lr, device, power_scaler, pred_len):
    """
    train_loader: 单步数据加载器，每个batch为 (hist, nwp_t, label_t)
    test_multistep_loader: 多步数据加载器，每个batch为 (hist, nwp_future, label)
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_val_loss = float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        for hist, nwp_t, label_t in train_loader:
            hist = hist.to(device)
            nwp_t = nwp_t.to(device)
            label_t = label_t.to(device)
            optimizer.zero_grad()
            out = model(hist, nwp_t)
            loss = criterion(out, label_t)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hist, nwp_t, label_t in val_loader:
                hist = hist.to(device)
                nwp_t = nwp_t.to(device)
                label_t = label_t.to(device)
                out = model(hist, nwp_t)
                loss = criterion(out, label_t)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    # 测试：滚动预测（使用多步测试数据）
    model.eval()
    all_pred_sequences = []
    all_true_sequences = []
    with torch.no_grad():
        for hist, nwp_future, label in test_multistep_loader:
            hist = hist.to(device)
            nwp_future = nwp_future.to(device)
            B = hist.shape[0]
            current_hist = hist.clone()
            preds = []
            for t in range(pred_len):
                nwp_t = nwp_future[:, t, :]      # [B, nwp_dim]
                next_pred = model(current_hist, nwp_t)
                preds.append(next_pred.cpu().numpy())
                next_pred_tensor = next_pred.unsqueeze(1)
                current_hist = torch.cat([current_hist[:, 1:], next_pred_tensor], dim=1)
            pred_seq = np.stack(preds, axis=1)
            all_pred_sequences.append(pred_seq)
            all_true_sequences.append(label.cpu().numpy())
    all_pred = np.concatenate(all_pred_sequences, axis=0).flatten()
    all_true = np.concatenate(all_true_sequences, axis=0).flatten()
    return compute_metrics(all_true, all_pred, power_scaler)

# ========== 7. 主程序 ==========
def main():
    results = []
    for station in STATIONS:
        print(f"\n{'#'*60}\n处理站点: {station}\n{'#'*60}")
        try:
            power_scaled, nwp_scaled, power_scaler, nwp_scaler = load_and_scale(station)
        except Exception as e:
            print(f"站点 {station} 加载失败: {e}")
            continue
        nwp_dim = nwp_scaled.shape[1]

        for hist_days in HISTORY_DAYS:
            lookback_len = hist_days * POINTS_PER_DAY
            for pred_len in PREDICT_LENS:
                print(f"\n--- 历史天数: {hist_days}天 ({lookback_len}点) | 预测长度: {pred_len}点 ---")
                samples = create_samples(power_scaled, nwp_scaled, lookback_len, pred_len)
                if samples is None:
                    print("样本不足，跳过")
                    continue
                hist_seq, nwp_future, label = samples
                n_samples = hist_seq.shape[0]
                train_end = int(n_samples * 0.7)
                val_end = int(n_samples * 0.85)
                train_idx = list(range(train_end))
                val_idx = list(range(train_end, val_end))
                test_idx = list(range(val_end, n_samples))
                print(f"样本总数: {n_samples}, 训练: {len(train_idx)}, 验证: {len(val_idx)}, 测试: {len(test_idx)}")

                # ----- (1) 一次性预测模型 -----
                model_multi = MultiStepLSTM(lookback_len, pred_len, nwp_dim,
                                            HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
                train_ds = TensorDataset(hist_seq[train_idx], nwp_future[train_idx], label[train_idx])
                val_ds   = TensorDataset(hist_seq[val_idx],   nwp_future[val_idx],   label[val_idx])
                test_ds  = TensorDataset(hist_seq[test_idx],  nwp_future[test_idx],  label[test_idx])
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
                val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
                test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
                mae, rmse, nmae, nrmse, r2 = train_multistep(model_multi, train_loader, val_loader, test_loader,
                                                             EPOCHS, LR, DEVICE, power_scaler)
                print(f"一次性预测 -> MAE:{mae:.4f} RMSE:{rmse:.4f} NMAE:{nmae:.4f} NRMSE:{nrmse:.4f} R2:{r2:.4f}")
                results.append((station, hist_days, pred_len, "MultiStep", mae, rmse, nmae, nrmse, r2))

                # ----- (2) 滚动单步预测模型 -----
                # 准备单步训练/验证数据（展平）
                def flatten_to_singlestep(hist_sub, nwp_sub, label_sub):
                    N, T = hist_sub.shape[0], pred_len
                    hist_list, nwp_t_list, label_t_list = [], [], []
                    for i in range(N):
                        for t in range(T):
                            hist_list.append(hist_sub[i])
                            nwp_t_list.append(nwp_sub[i, t])
                            label_t_list.append(label_sub[i, t])
                    return (torch.stack(hist_list),
                            torch.stack(nwp_t_list),
                            torch.tensor(label_t_list, dtype=torch.float32))
                train_hist_s, train_nwp_s, train_label_s = flatten_to_singlestep(
                    hist_seq[train_idx], nwp_future[train_idx], label[train_idx])
                val_hist_s, val_nwp_s, val_label_s = flatten_to_singlestep(
                    hist_seq[val_idx], nwp_future[val_idx], label[val_idx])
                # 测试仍然使用原始多步格式（hist_seq[test_idx], nwp_future[test_idx], label[test_idx]）
                test_ds_multistep = TensorDataset(hist_seq[test_idx], nwp_future[test_idx], label[test_idx])
                test_loader_multistep = DataLoader(test_ds_multistep, batch_size=BATCH_SIZE, shuffle=False)

                batch_pt = BATCH_SIZE * POINTWISE_BATCH_MULTIPLIER
                train_ds_s = TensorDataset(train_hist_s, train_nwp_s, train_label_s)
                val_ds_s   = TensorDataset(val_hist_s,   val_nwp_s,   val_label_s)
                train_loader_s = DataLoader(train_ds_s, batch_size=batch_pt, shuffle=True)
                val_loader_s   = DataLoader(val_ds_s,   batch_size=batch_pt, shuffle=False)

                model_single = SingleStepLSTM(lookback_len, nwp_dim, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
                mae, rmse, nmae, nrmse, r2 = train_singlestep(model_single, train_loader_s, val_loader_s,
                                                              test_loader_multistep, EPOCHS, LR, DEVICE,
                                                              power_scaler, pred_len)
                print(f"滚动单步预测 -> MAE:{mae:.4f} RMSE:{rmse:.4f} NMAE:{nmae:.4f} NRMSE:{nrmse:.4f} R2:{r2:.4f}")
                results.append((station, hist_days, pred_len, "Iterative", mae, rmse, nmae, nrmse, r2))

    # 输出汇总
    print("\n\n" + "="*120)
    print("完整实验结果汇总")
    print("="*120)
    header = f"{'站点':<12} {'历史天数':<8} {'预测长度':<8} {'模型':<12} {'MAE':>8} {'RMSE':>8} {'NMAE':>8} {'NRMSE':>8} {'R2':>8}"
    print(header)
    for row in results:
        station, hist_days, pred_len, model_type, mae, rmse, nmae, nrmse, r2 = row
        print(f"{station:<12} {hist_days:<8} {pred_len:<8} {model_type:<12} {mae:8.4f} {rmse:8.4f} {nmae:8.4f} {nrmse:8.4f} {r2:8.4f}")

    df = pd.DataFrame(results, columns=['station', 'history_days', 'pred_len', 'model',
                                        'MAE', 'RMSE', 'NMAE', 'NRMSE', 'R2'])
    df.to_csv("/root/timer+exo/LSTM/LSTM_compare_experiment.csv", index=False)
    print("\n结果已保存至 /root/timer+exo/LSTM/LSTM_compare_experiment.csv")

if __name__ == "__main__":
    main()