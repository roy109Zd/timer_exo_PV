import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========== 配置 ==========
PRED_DIR = "/root/timer+exo/pred"
STATIONS = [f"station{i:02d}" for i in range(10)]   # station00 ~ station09
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
EPOCHS = 1000
LR = 1e-4
PATIENCE = 10                     # 早停耐心值
HISTORY_LENS = [1, 4, 16, 96]     # 历史窗口长度（包含当前点）

# ========== 指标计算 ==========
def compute_metrics(true, pred, power_range):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    return mae, rmse, nmae, nrmse, r2

# ========== MLP 模型 ==========
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ========== 构建滑动窗口数据集（按时间顺序） ==========
def create_sequences(df, n):
    """
    df : DataFrame 包含 'power_pred' 和 7 个 nwp 列，已按时间排序
    n   : 历史窗口长度（包含当前时刻）
    返回 X (n_samples, n*8), y (n_samples,)
    """
    feature_cols = ['power_pred'] + [f'nwp_{c}' for c in ['globalirrad','directirrad','temperature',
                                                           'humidity','windspeed','winddirection','pressure']]
    # 实际列名：nwp_globalirrad 等，需匹配
    actual_nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
                       'nwp_windspeed','nwp_winddirection','nwp_pressure']
    feature_cols = ['power_pred'] + actual_nwp_cols
    data = df[feature_cols].values.astype(np.float32)  # (T, 8)
    T = data.shape[0]
    if T < n:
        raise ValueError(f"序列长度 {T} 小于窗口长度 {n}")
    X_list, y_list = [], []
    for i in range(n-1, T):
        # 窗口 [i-n+1, i] 共 n 个点
        window = data[i-n+1:i+1]          # (n, 8)
        X_list.append(window.flatten())   # (n*8,)
        y_list.append(df['power_true'].iloc[i])
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y

# ========== 加载原始数据（不构造窗口，直接返回完整 DataFrame） ==========
def load_station_df(station_name):
    csv_path = os.path.join(PRED_DIR, f"{station_name}_timer_pred_with_info.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    # 确保按时间升序排序（假设有 'time' 列，若无则按索引顺序）
    if 'time' in df.columns:
        df = df.sort_values('time')
    return df

# ========== 训练单个模型（给定窗口后的 X, y）并返回测试集指标 ==========
def train_evaluate_for_window(X, y, n, station_name):
    """
    X, y 已经是由 create_sequences 得到的窗口化数据
    按时间顺序划分为 80% 训练，10% 验证，10% 测试
    """
    n_samples = len(X)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # 转换为 PyTorch tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                            batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X.shape[1]
    model = SimpleMLP(input_dim=input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
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
            print(f"[{station_name}, n={n}] Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # 测试集评估
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred_test = model(X_test_t).cpu().numpy()
    power_range = y_test.max() - y_test.min()
    mae, rmse, nmae, nrmse, r2 = compute_metrics(y_test, pred_test, power_range)
    return mae, rmse, nmae, nrmse, r2

# ========== 主函数 ==========
def main():
    results = []   # 存储每行结果
    for station in STATIONS:
        print(f"\n===== 处理站点: {station} =====")
        try:
            df = load_station_df(station)
        except FileNotFoundError as e:
            print(f"跳过 {station}: {e}")
            continue

        for n in HISTORY_LENS:
            print(f"  历史长度 n={n} ...")
            try:
                X, y = create_sequences(df, n)
            except ValueError as e:
                print(f"    跳过: {e}")
                continue
            if len(X) == 0:
                print(f"    无有效样本，跳过")
                continue
            # 训练并评估
            mae, rmse, nmae, nrmse, r2 = train_evaluate_for_window(X, y, n, station)
            results.append({
                'station': station,
                'n': n,
                'MAE': mae,
                'RMSE': rmse,
                'NMAE': nmae,
                'NRMSE': nrmse,
                'R2': r2
            })
            print(f"    -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, NMAE: {nmae:.4f}, NRMSE: {nrmse:.4f}, R2: {r2:.4f}")

    # 保存结果到 CSV
    df_results = pd.DataFrame(results)
    output_path = "/root/timer+exo/MLP/evaluation_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n所有结果已保存至: {output_path}")
    print(df_results)

if __name__ == "__main__":
    main()