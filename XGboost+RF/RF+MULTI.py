import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# ==================== 全局配置 ====================
PRED_DIR = "/root/timer+exo/pred"
STATIONS_ALL = [f"station{i:02d}" for i in range(10)]   # station00 ~ station09
STATIONS_3 = [f"station{i:02d}" for i in range(3)]      # station00,01,02

WINDOW = 96
STRIDE = 96
TEST_RATIO = 0.1
VAL_RATIO = 0.1          # 用于保持划分一致，RF 不使用验证集

FEAT_COLS = ['power_pred', 'nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature',
             'nwp_humidity', 'nwp_windspeed', 'nwp_winddirection', 'nwp_pressure']

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

# ==================== 数据加载与窗口构建 ====================
def load_windows(station, window=WINDOW, stride=STRIDE):
    """返回 (X_windows, y_windows)，形状 (n_windows, window, feat_dim) 和 (n_windows, window)"""
    csv_path = os.path.join(PRED_DIR, f"{station}_timer_pred_with_info.csv")
    df = pd.read_csv(csv_path)
    df = df.sort_values('datetime').reset_index(drop=True)
    X_raw = df[FEAT_COLS].values.astype(np.float32)
    y_raw = df['power_true'].values.astype(np.float32)
    n = len(X_raw)
    X_windows, y_windows = [], []
    for start in range(0, n - window + 1, stride):
        X_windows.append(X_raw[start:start+window])
        y_windows.append(y_raw[start:start+window])
    return np.array(X_windows), np.array(y_windows)

def split_temporal(X, y, test_ratio, val_ratio):
    """按时间顺序划分，返回展平后的训练/验证/测试集（验证集仅用于保持划分一致）"""
    n = len(X)
    test_start = int(n * (1 - test_ratio))
    val_start = int(n * (1 - test_ratio - val_ratio))
    X_train = X[:val_start]
    y_train = y[:val_start]
    X_val = X[val_start:test_start]
    y_val = y[val_start:test_start]
    X_test = X[test_start:]
    y_test = y[test_start:]
    # 展平用于 RF 训练（将窗口和特征展平为一维）
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    y_train_flat = y_train.reshape(len(y_train), -1)
    y_val_flat = y_val.reshape(len(y_val), -1)
    y_test_flat = y_test.reshape(len(y_test), -1)
    return X_train_flat, X_val_flat, X_test_flat, y_train_flat, y_val_flat, y_test_flat, y_test

# ==================== 单站独立训练 ====================
def train_single_station(station):
    print(f"\n>>> 单站训练: {station}")
    X, y = load_windows(station)
    X_train, _, X_test, y_train, _, y_test_flat, y_test = split_temporal(X, y, TEST_RATIO, VAL_RATIO)
    print(f"训练窗口: {len(X_train)}, 测试窗口: {len(X_test)}")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)   # (n_test, window)
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_test_flat.flatten()
    compute_metrics(y_true_flat, y_pred_flat, f"{station} (RF独立)")
    return y_true_flat, y_pred_flat

# ==================== 多站协同训练 ====================
def train_multi_station(stations, model_name):
    """stations: 站点名列表，将特征和标签沿列拼接"""
    print(f"\n>>> 多站协同训练: {model_name} (站点: {stations})")
    # 加载所有站点的数据，确保窗口数一致（取最小窗口数）
    data = [load_windows(s) for s in stations]
    min_windows = min(len(d[0]) for d in data)
    X_list, y_list = [], []
    for X, y in data:
        X_list.append(X[:min_windows])
        y_list.append(y[:min_windows])
    # 拼接特征（沿特征维度）和标签（沿窗口维度）
    X_concat = np.concatenate(X_list, axis=-1)   # (n_windows, window, 8*num_stations)
    y_concat = np.concatenate(y_list, axis=-1)   # (n_windows, window*num_stations)
    # 划分
    n = len(X_concat)
    test_start = int(n * (1 - TEST_RATIO))
    val_start = int(n * (1 - TEST_RATIO - VAL_RATIO))
    X_train = X_concat[:val_start]
    y_train = y_concat[:val_start]
    X_test = X_concat[test_start:]
    y_test = y_concat[test_start:]
    # 展平
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    y_train_flat = y_train.reshape(len(y_train), -1)
    y_test_flat = y_test.reshape(len(y_test), -1)
    print(f"训练窗口: {len(X_train)}, 测试窗口: {len(X_test)}")
    print(f"输入维度: {X_train_flat.shape[1]}, 输出维度: {y_train_flat.shape[1]}")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_flat, y_train_flat)
    y_pred_flat = model.predict(X_test_flat)   # (n_test, window*num_stations)
    # 拆分回每个站点的预测
    window = WINDOW
    num_stations = len(stations)
    y_test_reshaped = y_test_flat.reshape(-1, window, num_stations)   # (n_test, window, num_stations)
    y_pred_reshaped = y_pred_flat.reshape(-1, window, num_stations)
    # 逐站评估
    for idx, st in enumerate(stations):
        true_st = y_test_reshaped[:, :, idx].flatten()
        pred_st = y_pred_reshaped[:, :, idx].flatten()
        compute_metrics(true_st, pred_st, f"{st} ({model_name})")
    return

# ==================== 主程序 ====================
def main():
    print("="*70)
    print("实验1: 单站独立 RF (10站)")
    for st in STATIONS_ALL:
        train_single_station(st)

    print("\n" + "="*70)
    print("实验2: 多站协同 RF (3站: station00,01,02)")
    train_multi_station(STATIONS_3, "RF协同3站")

    print("\n" + "="*70)
    print("实验3: 多站协同 RF (10站: station00~09)")
    train_multi_station(STATIONS_ALL, "RF协同10站")

    print("\n所有实验完成，不保存任何文件。")

if __name__ == "__main__":
    main()