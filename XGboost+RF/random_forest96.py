import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

PRED_DIR = "/root/timer+exo/pred"
STATION = "station02"
WINDOW = 96
STRIDE = 96
TEST_RATIO = 0.1
VAL_RATIO = 0.1

FEAT_COLS = ['power_pred', 'nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature',
             'nwp_humidity', 'nwp_windspeed', 'nwp_winddirection', 'nwp_pressure']

def compute_metrics(true, pred, name=""):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    power_range = true.max() - true.min()
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    print(f"{name:20s} | MAE:{mae:7.4f} | RMSE:{rmse:7.4f} | NMAE:{nmae:6.4f} | NRMSE:{nrmse:6.4f} | R2:{r2:6.4f}")
    return mae, rmse, nmae, nrmse, r2

def load_windows(station):
    csv_path = os.path.join(PRED_DIR, f"{station}_timer_pred_with_info.csv")
    df = pd.read_csv(csv_path)
    df = df.sort_values('datetime').reset_index(drop=True)
    X_raw = df[FEAT_COLS].values.astype(np.float32)
    y_raw = df['power_true'].values.astype(np.float32)
    n = len(X_raw)
    X_windows, y_windows = [], []
    for start in range(0, n - WINDOW + 1, STRIDE):
        X_windows.append(X_raw[start:start+WINDOW])
        y_windows.append(y_raw[start:start+WINDOW])
    X = np.array(X_windows)
    y = np.array(y_windows)
    return X, y

def split_temporal(X, y, test_ratio, val_ratio):
    n = len(X)
    test_start = int(n * (1 - test_ratio))
    val_start = int(n * (1 - test_ratio - val_ratio))
    X_train = X[:val_start]
    y_train = y[:val_start]
    X_val = X[val_start:test_start]
    y_val = y[val_start:test_start]
    X_test = X[test_start:]
    y_test = y[test_start:]
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    return X_train_flat, X_val_flat, X_test_flat, y_train, y_val, y_test, X_test, y_test

def main():
    print(f"站点: {STATION} | Random Forest 多输出回归 (窗口={WINDOW}, 步长={STRIDE})")
    X, y = load_windows(STATION)
    print(f"总窗口数: {len(X)}")
    X_train_flat, X_val_flat, X_test_flat, y_train, y_val, y_test, X_test_orig, y_test_orig = split_temporal(X, y, TEST_RATIO, VAL_RATIO)
    print(f"训练窗口: {len(X_train_flat)}, 验证窗口: {len(X_val_flat)}, 测试窗口: {len(X_test_flat)}")

    # Timer 原始预测值
    timer_pred_test = X_test_orig[:, :, 0].flatten()
    y_test_true = y_test_orig.flatten()
    print("\n" + "="*60)
    print("【Timer 原始预测值】评估 (测试集):")
    compute_metrics(y_test_true, timer_pred_test, "Timer原始")

    print("\n训练 Random Forest 多输出模型...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_flat, y_train)

    y_pred = model.predict(X_test_flat)
    y_pred_flat = y_pred.flatten()
    print("\n" + "="*60)
    print("【Random Forest 多输出】评估 (测试集):")
    compute_metrics(y_test_true, y_pred_flat, "RandomForest")

    # 每个时间步 MAE
    per_step_mae = []
    for t in range(WINDOW):
        true_step = y_test_orig[:, t]
        pred_step = y_pred[:, t]
        mae = mean_absolute_error(true_step, pred_step)
        per_step_mae.append(mae)
    print(f"\n每个时间步的平均 MAE (共{WINDOW}步):")
    print(f"  整体平均: {np.mean(per_step_mae):.4f} | 前10步: {per_step_mae[100:110]}")
    print(f"  步数最小MAE: {np.argmin(per_step_mae)} (MAE={np.min(per_step_mae):.4f})")
    print(f"  步数最大MAE: {np.argmax(per_step_mae)} (MAE={np.max(per_step_mae):.4f})")

    print("\n不保存任何模型文件。")

if __name__ == "__main__":
    main()