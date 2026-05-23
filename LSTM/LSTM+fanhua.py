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
ALL_STATIONS = [f"station{i:02d}" for i in range(10)]   # 00~09
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LOOKBACK_POINTS = 96
PRED_LEN = 96
NWP_DIM = 7

# ========== 1. 加载站点数据并标准化 ==========
def load_and_scale_station(station):
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

# ========== 2. 构造样本 ==========
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
        raise ValueError(f"样本数为0：总长度{total_len}, lookback{lookback_len}, pred_len{pred_len}")
    hist_seq = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)
    nwp_future = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)
    label = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.float32)
    return hist_seq, nwp_future, label

# ========== 3. 模型 ==========
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

# ========== 4. 评估指标 ==========
def compute_metrics(y_true_scaled, y_pred_scaled, power_scaler):
    y_true = power_scaler.inverse_transform(y_true_scaled.reshape(-1,1)).flatten()
    y_pred = power_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    power_range = y_true.max() - y_true.min()
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    return mae, rmse, nmae, nrmse, r2

# ========== 5. 训练函数 ==========
def train_model(model, train_loader, val_loader, epochs, lr, device, patience=10):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_val_loss = float('inf')
    best_state = None
    counter = 0
    for epoch in range(1, epochs+1):
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
        else:
            counter += 1
            if counter >= patience:
                print(f"早停触发，停止训练")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model

# ========== 6. 评估函数（在给定数据加载器上） ==========
def evaluate_model(model, test_loader, power_scaler, device):
    model.eval()
    all_pred_scaled = []
    all_true_scaled = []
    with torch.no_grad():
        for hist, nwp_future, label in test_loader:
            hist = hist.to(device)
            nwp_future = nwp_future.to(device)
            pred = model(hist, nwp_future).cpu().numpy()
            true = label.numpy()
            all_pred_scaled.append(pred)
            all_true_scaled.append(true)
    all_pred = np.concatenate(all_pred_scaled, axis=0).flatten()
    all_true = np.concatenate(all_true_scaled, axis=0).flatten()
    return compute_metrics(all_true, all_pred, power_scaler)

# ========== 7. 准备单个站点的数据（返回训练/验证/测试集和scaler） ==========
def prepare_station_data(station, lookback, pred_len, train_ratio=0.7, val_ratio=0.15):
    power_scaled, nwp_scaled, power_scaler, _ = load_and_scale_station(station)
    hist_seq, nwp_future, label = create_samples(power_scaled, nwp_scaled, lookback, pred_len)
    n = hist_seq.shape[0]
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_hist = hist_seq[:train_end]
    train_nwp  = nwp_future[:train_end]
    train_label = label[:train_end]
    val_hist   = hist_seq[train_end:val_end]
    val_nwp    = nwp_future[train_end:val_end]
    val_label  = label[train_end:val_end]
    test_hist  = hist_seq[val_end:]
    test_nwp   = nwp_future[val_end:]
    test_label = label[val_end:]
    return (train_hist, train_nwp, train_label,
            val_hist, val_nwp, val_label,
            test_hist, test_nwp, test_label,
            power_scaler)

# ========== 8. 合并多个站点的训练数据（用于联合训练） ==========
def merge_stations_training(stations, lookback, pred_len):
    """
    将所有指定站点的训练+验证数据合并（测试数据分开返回）
    返回:
        train_hist, train_nwp, train_label
        val_hist, val_nwp, val_label
        test_data_list: 每个站点一个元组 (test_hist, test_nwp, test_label, scaler)
    """
    all_train_hist, all_train_nwp, all_train_label = [], [], []
    all_val_hist, all_val_nwp, all_val_label = [], [], []
    test_data_list = []
    for st in stations:
        (tr_h, tr_n, tr_l, v_h, v_n, v_l, te_h, te_n, te_l, scaler) = prepare_station_data(st, lookback, pred_len)
        all_train_hist.append(tr_h)
        all_train_nwp.append(tr_n)
        all_train_label.append(tr_l)
        all_val_hist.append(v_h)
        all_val_nwp.append(v_n)
        all_val_label.append(v_l)
        test_data_list.append((te_h, te_n, te_l, scaler))
    train_hist = torch.cat(all_train_hist, dim=0)
    train_nwp  = torch.cat(all_train_nwp, dim=0)
    train_label = torch.cat(all_train_label, dim=0)
    val_hist   = torch.cat(all_val_hist, dim=0)
    val_nwp    = torch.cat(all_val_nwp, dim=0)
    val_label  = torch.cat(all_val_label, dim=0)
    return (train_hist, train_nwp, train_label,
            val_hist, val_nwp, val_label,
            test_data_list)

# ========== 9. 主程序：泛化能力测试（所有站点） ==========
def main():
    print("="*70)
    print("泛化能力测试：LSTM 模型（评估所有站点）")
    print("="*70)
    lookback = LOOKBACK_POINTS
    pred_len = PRED_LEN

    # 预先准备所有站点的测试数据（用于后续统一评估）
    all_test_loaders = {}
    all_scalers = {}
    for st in ALL_STATIONS:
        (_, _, _, _, _, _, test_h, test_n, test_l, scaler) = prepare_station_data(st, lookback, pred_len)
        test_ds = TensorDataset(test_h, test_n, test_l)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        all_test_loaders[st] = test_loader
        all_scalers[st] = scaler

    # ---------- 实验1：仅用 station00 训练，评估所有站点 ----------
    print("\n[实验1] 训练集: station00，评估所有站点")
    data00 = prepare_station_data("station00", lookback, pred_len)
    (tr_h, tr_n, tr_l, v_h, v_n, v_l, _, _, _, _) = data00
    train_ds = TensorDataset(tr_h, tr_n, tr_l)
    val_ds   = TensorDataset(v_h, v_n, v_l)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model1 = BasicLSTM(lookback, pred_len, NWP_DIM, HIDDEN_SIZE, NUM_LAYERS)
    model1 = train_model(model1, train_loader, val_loader, EPOCHS, LR, DEVICE)

    results1 = {}
    for st in ALL_STATIONS:
        mae, rmse, nmae, nrmse, r2 = evaluate_model(model1, all_test_loaders[st], all_scalers[st], DEVICE)
        results1[st] = (mae, rmse, r2)
        print(f"  {st}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # ---------- 实验2：用 station00+station01 联合训练，评估所有站点 ----------
    print("\n[实验2] 训练集: station00+station01，评估所有站点")
    train_stations = ["station00", "station01"]
    (tr_h, tr_n, tr_l, v_h, v_n, v_l, test_list) = merge_stations_training(train_stations, lookback, pred_len)
    train_ds2 = TensorDataset(tr_h, tr_n, tr_l)
    val_ds2   = TensorDataset(v_h, v_n, v_l)
    train_loader2 = DataLoader(train_ds2, batch_size=BATCH_SIZE, shuffle=True)
    val_loader2   = DataLoader(val_ds2, batch_size=BATCH_SIZE, shuffle=False)

    model2 = BasicLSTM(lookback, pred_len, NWP_DIM, HIDDEN_SIZE, NUM_LAYERS)
    model2 = train_model(model2, train_loader2, val_loader2, EPOCHS, LR, DEVICE)

    results2 = {}
    for st in ALL_STATIONS:
        mae, rmse, nmae, nrmse, r2 = evaluate_model(model2, all_test_loaders[st], all_scalers[st], DEVICE)
        results2[st] = (mae, rmse, r2)
        print(f"  {st}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # 汇总对比表格
    print("\n" + "="*90)
    print("泛化能力对比 (MAE / RMSE / R2)")
    print("="*90)
    print(f"{'测试站点':<10} {'仅 station00 训练':<30} {'station00+01 联合训练':<30}")
    print(f"{'':<10} {'MAE':>10} {'RMSE':>10} {'R2':>8}  {'MAE':>10} {'RMSE':>10} {'R2':>8}")
    print("-"*90)
    for st in ALL_STATIONS:
        mae1, rmse1, r21 = results1[st]
        mae2, rmse2, r22 = results2[st]
        print(f"{st:<10} {mae1:10.4f} {rmse1:10.4f} {r21:8.4f}  {mae2:10.4f} {rmse2:10.4f} {r22:8.4f}")

if __name__ == "__main__":
    main()