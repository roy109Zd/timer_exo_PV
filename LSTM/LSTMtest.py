import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
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
STATIONS = ["station00", "station01", "station02"]
# 各站起始时间（格式：YYYY-MM-DD HH:MM:SS）
STATION_START_TIMES = {
    "station00": "2018-08-15 16:00:00",
    "station01": "2018-06-30 16:00:00",
    "station02": "2018-07-22 16:00:00",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 500
LR = 1e-4
HIDDEN_SIZE = 64          # LSTM 隐层维度
NUM_LAYERS = 2            # LSTM 层数
LOOKBACK_POINTS = 96      # 历史1天 = 96点
PRED_LEN = 96             # 预测96点
NWP_DIM = 7               # 每个站点的 NWP 变量数
PATIENCE = 10             # 早停耐心

# ========== 1. 加载单个站点数据并生成时间索引 ==========
def load_station_with_time(station):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{station}.csv"))
    power = df['power'].values.astype(np.float32)
    nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
                'nwp_windspeed','nwp_winddirection','nwp_pressure']
    nwp = df[nwp_cols].values.astype(np.float32)
    # 根据起始时间生成时间索引（15分钟间隔）
    start_time = pd.to_datetime(STATION_START_TIMES[station])
    time_index = pd.date_range(start=start_time, periods=len(power), freq='15T')
    return time_index, power, nwp

# ========== 2. 多站时间对齐，返回标准化后的数据及每个站的功率范围 ==========
def align_and_scale_stations(stations, lookback_len, pred_len):
    times_dict = {}
    power_dict = {}
    nwp_dict = {}
    for st in stations:
        times, power, nwp = load_station_with_time(st)
        times_dict[st] = times
        power_dict[st] = power
        nwp_dict[st] = nwp

    # 公共时间范围
    common_start = max([ts.min() for ts in times_dict.values()])
    common_end = min([ts.max() for ts in times_dict.values()])
    print(f"公共时间范围: {common_start} 到 {common_end}")

    data_dict = {}       # station -> (power_scaled, nwp_scaled, scaler)
    power_ranges = {}    # station -> 全局功率范围（用于归一化指标）
    for st in stations:
        mask = (times_dict[st] >= common_start) & (times_dict[st] <= common_end)
        power_cut = power_dict[st][mask]
        nwp_cut = nwp_dict[st][mask, :]

        power_ranges[st] = power_cut.max() - power_cut.min()

        scaler = StandardScaler()
        power_scaled = scaler.fit_transform(power_cut.reshape(-1,1)).flatten()
        nwp_scaler = StandardScaler()
        nwp_scaled = nwp_scaler.fit_transform(nwp_cut)

        data_dict[st] = (power_scaled, nwp_scaled, scaler)

    # 截取到相同长度
    min_len = min([len(data_dict[st][0]) for st in stations])
    for st in stations:
        p, n, s = data_dict[st]
        data_dict[st] = (p[:min_len], n[:min_len], s)

    return data_dict, power_ranges

# ========== 3. 构造多站样本（滑动窗口，步长=预测长度） ==========
def create_multistation_samples(data_dict, lookback_len, pred_len):
    stations = list(data_dict.keys())
    nwp_dim = data_dict[stations[0]][1].shape[1]
    total_len = len(data_dict[stations[0]][0])
    stride = pred_len
    samples = []
    for start in range(0, total_len - lookback_len - pred_len + 1, stride):
        hist_list = []
        nwp_list = []
        label_list = []
        for st in stations:
            power_scaled, nwp_scaled, _ = data_dict[st]
            hist = power_scaled[start:start+lookback_len]
            nwp_future = nwp_scaled[start+lookback_len:start+lookback_len+pred_len]
            label = power_scaled[start+lookback_len:start+lookback_len+pred_len]
            hist_list.append(hist)
            nwp_list.append(nwp_future)
            label_list.append(label)
        samples.append((np.stack(hist_list), np.stack(nwp_list), np.stack(label_list)))
    if len(samples) == 0:
        raise ValueError("样本数为0，请检查数据长度")
    hist_tensor = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)   # [N, num_stations, lookback]
    nwp_tensor = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)    # [N, num_stations, pred_len, nwp_dim]
    label_tensor = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.float32)  # [N, num_stations, pred_len]
    return hist_tensor, nwp_tensor, label_tensor

# ========== 4. 多站融合模型 ==========
class MultiStationFusion(nn.Module):
    def __init__(self, num_stations, lookback_len, pred_len, nwp_dim, hidden_size, num_layers):
        super().__init__()
        self.num_stations = num_stations
        self.pred_len = pred_len
        self.nwp_dim = nwp_dim
        # 每个站点独立的 LSTM 编码器
        self.station_lstms = nn.ModuleList()
        for _ in range(num_stations):
            lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
            self.station_lstms.append(lstm)
        # 共享的融合网络：输入 = 所有站点的隐状态拼接 + 当前步所有站点的NWP拼接
        fusion_input_dim = num_stations * hidden_size + num_stations * nwp_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_stations)   # 输出该步各站预测值
        )
    def forward(self, hist, nwp_future):
        """
        hist: [B, num_stations, lookback]
        nwp_future: [B, num_stations, pred_len, nwp_dim]
        """
        B = hist.shape[0]
        # 1. 编码各站历史，得到隐状态 [B, hidden] 每个站
        contexts = []
        for i in range(self.num_stations):
            hist_i = hist[:, i, :]                 # [B, lookback]
            _, (h_n, _) = self.station_lstms[i](hist_i.unsqueeze(-1))
            context_i = h_n[-1]                    # [B, hidden]
            contexts.append(context_i)
        all_context = torch.cat(contexts, dim=1)   # [B, num_stations * hidden]

        # 2. 逐时间步预测
        outputs = []
        for t in range(self.pred_len):
            # 提取该时间步所有站点的 NWP
            nwp_t = nwp_future[:, :, t, :]         # [B, num_stations, nwp_dim]
            nwp_t_flat = nwp_t.view(B, -1)         # [B, num_stations * nwp_dim]
            # 拼接上下文和当前步NWP
            fusion_input = torch.cat([all_context, nwp_t_flat], dim=1)  # [B, num_stations*(hidden+nwp_dim)]
            # 融合网络输出该步所有站点的预测值
            step_out = self.fusion(fusion_input)   # [B, num_stations]
            outputs.append(step_out)
        # 堆叠时间步并转置为 [B, num_stations, pred_len]
        out = torch.stack(outputs, dim=2)          # [B, num_stations, pred_len]
        return out

# ========== 5. 数据集 ==========
class MultiStationDataset(Dataset):
    def __init__(self, hist, nwp, label):
        self.hist = hist
        self.nwp = nwp
        self.label = label
        self.length = hist.shape[0]
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.hist[idx], self.nwp[idx], self.label[idx]

# ========== 6. 评估指标（使用各站全局功率范围） ==========
def compute_metrics(y_true, y_pred, power_range):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    if power_range > 0:
        nmae = mae / power_range
        nrmse = rmse / power_range
    else:
        nmae = nrmse = np.nan
    return mae, rmse, nmae, nrmse, r2

# ========== 7. 训练函数（损失为各站MSE之和，带早停） ==========
def train_model(model, train_loader, val_loader, test_loader, epochs, lr, device,
                station_scalers, station_power_ranges, patience=10):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    counter = 0

    for epoch in range(1, epochs+1):
        # 训练
        model.train()
        train_loss = 0.0
        for hist, nwp, label in train_loader:
            hist = hist.to(device)
            nwp = nwp.to(device)
            label = label.to(device)       # [B, num_stations, pred_len]
            optimizer.zero_grad()
            pred = model(hist, nwp)        # [B, num_stations, pred_len]
            loss = 0.0
            for i in range(pred.shape[1]):
                loss += criterion(pred[:, i, :], label[:, i, :])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hist, nwp, label in val_loader:
                hist = hist.to(device)
                nwp = nwp.to(device)
                label = label.to(device)
                pred = model(hist, nwp)
                loss = 0.0
                for i in range(pred.shape[1]):
                    loss += criterion(pred[:, i, :], label[:, i, :])
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # 早停
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
    all_pred = []
    all_true = []
    with torch.no_grad():
        for hist, nwp, label in test_loader:
            hist = hist.to(device)
            nwp = nwp.to(device)
            pred = model(hist, nwp).cpu().numpy()   # [B, num_stations, pred_len]
            all_pred.append(pred)
            all_true.append(label.cpu().numpy())
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    # 分别评估每个站点
    station_metrics = []
    for i, st in enumerate(STATIONS):
        pred_station = all_pred[:, i, :].flatten()
        true_station = all_true[:, i, :].flatten()
        scaler = station_scalers[st]
        true_orig = scaler.inverse_transform(true_station.reshape(-1,1)).flatten()
        pred_orig = scaler.inverse_transform(pred_station.reshape(-1,1)).flatten()
        metrics = compute_metrics(true_orig, pred_orig, station_power_ranges[st])
        station_metrics.append(metrics)
    return station_metrics

# ========== 8. 主程序 ==========
def main():
    print(f"设备: {DEVICE}")
    lookback_len = LOOKBACK_POINTS
    pred_len = PRED_LEN

    # 1. 对齐并标准化数据
    print("\n正在对齐三个站点的时间序列...")
    data_dict, power_ranges = align_and_scale_stations(STATIONS, lookback_len, pred_len)
    station_scalers = {st: data_dict[st][2] for st in STATIONS}
    clean_data = {st: (data_dict[st][0], data_dict[st][1]) for st in STATIONS}

    # 2. 构造样本
    print("构造样本...")
    hist_tensor, nwp_tensor, label_tensor = create_multistation_samples(clean_data, lookback_len, pred_len)
    N = hist_tensor.shape[0]
    print(f"总样本数: {N}")

    # 3. 划分训练/验证/测试 (时间顺序)
    train_end = int(N * 0.7)
    val_end = int(N * 0.85)
    train_hist = hist_tensor[:train_end]
    train_nwp  = nwp_tensor[:train_end]
    train_label = label_tensor[:train_end]
    val_hist   = hist_tensor[train_end:val_end]
    val_nwp    = nwp_tensor[train_end:val_end]
    val_label  = label_tensor[train_end:val_end]
    test_hist  = hist_tensor[val_end:]
    test_nwp   = nwp_tensor[val_end:]
    test_label = label_tensor[val_end:]
    print(f"训练: {len(train_hist)} 样本, 验证: {len(val_hist)} 样本, 测试: {len(test_hist)} 样本")

    # 4. DataLoader
    train_ds = MultiStationDataset(train_hist, train_nwp, train_label)
    val_ds   = MultiStationDataset(val_hist,   val_nwp,   val_label)
    test_ds  = MultiStationDataset(test_hist,  test_nwp,  test_label)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # 5. 构建模型
    model = MultiStationFusion(num_stations=len(STATIONS),
                               lookback_len=lookback_len,
                               pred_len=pred_len,
                               nwp_dim=NWP_DIM,
                               hidden_size=HIDDEN_SIZE,
                               num_layers=NUM_LAYERS)

    # 6. 训练并评估
    station_metrics = train_model(model, train_loader, val_loader, test_loader,
                                  EPOCHS, LR, DEVICE, station_scalers, power_ranges,
                                  patience=PATIENCE)

    # 7. 输出结果
    print("\n" + "="*70)
    print("各站独立预测结果 (损失: 各站MSE之和)")
    print("="*70)
    print(f"{'站点':<12} {'C值':>10} {'MAE':>10} {'RMSE':>10} {'NMAE':>10} {'NRMSE':>10} {'R2':>10}")
    for i, st in enumerate(STATIONS):
        mae, rmse, nmae, nrmse, r2 = station_metrics[i]
        c_val = power_ranges[st]
        print(f"{st:<12} {c_val:10.4f} {mae:10.4f} {rmse:10.4f} {nmae:10.4f} {nrmse:10.4f} {r2:10.4f}")

if __name__ == "__main__":
    main()