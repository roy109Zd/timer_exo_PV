import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

PRED_DIR = "/root/timer+exo/pred"
ALL_STATIONS = [f"station{i:02d}" for i in range(10)]
FEAT_COLS = ['power_pred', 'nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature',
             'nwp_humidity', 'nwp_windspeed', 'nwp_winddirection', 'nwp_pressure']
TEST_RATIO = 0.1
VAL_RATIO = 0.1

def load_station_data(station):
    csv_path = os.path.join(PRED_DIR, f"{station}_timer_pred_with_info.csv")
    df = pd.read_csv(csv_path)
    df = df.sort_values('datetime').reset_index(drop=True)
    X = df[FEAT_COLS].values.astype(np.float32)
    y = df['power_true'].values.astype(np.float32)
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
    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_metrics(y_true, y_pred, name=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    power_range = y_true.max() - y_true.min()
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    print(f"{name:25s} | MAE:{mae:7.4f} | RMSE:{rmse:7.4f} | NMAE:{nmae:6.4f} | NRMSE:{nrmse:6.4f} | R2:{r2:6.4f}")
    return mae, rmse, nmae, nrmse, r2

def train_and_evaluate(train_stations, model_name_prefix):
    print(f"\n{'='*70}")
    print(f"训练配置: {model_name_prefix} (训练站点: {train_stations})")
    X_list, y_list = [], []
    for st in train_stations:
        X, y = load_station_data(st)
        X_list.append(X)
        y_list.append(y)
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    X_train, X_val, X_test, y_train, y_val, y_test = split_temporal(X_all, y_all, TEST_RATIO, VAL_RATIO)
    print(f"训练样本: {len(X_train)}, 验证样本: {len(X_val)}, 测试样本: {len(X_test)}")

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        objective='reg:squarederror',
        eval_metric='rmse',
        early_stopping_rounds=30,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_round = model.best_iteration if model.best_iteration else model.n_estimators
    print(f"最佳迭代次数: {best_round}")

    # 评估训练站点的测试集
    y_pred_test = model.predict(X_test)
    print(f"\n--- {model_name_prefix} 在训练站点的测试集上 (最后10%数据) ---")
    compute_metrics(y_test, y_pred_test, f"{model_name_prefix} (测试集)")

    # 评估其他站点全部数据
    other_stations = [st for st in ALL_STATIONS if st not in train_stations]
    print(f"\n--- {model_name_prefix} 在其他站点上的评估 (全部数据) ---")
    for st in other_stations:
        X_other, y_other = load_station_data(st)
        y_pred_other = model.predict(X_other)
        compute_metrics(y_other, y_pred_other, f"{st} (全部数据)")

def main():
    train_and_evaluate(['station00'], "XGB_仅00站")
    train_and_evaluate(['station00', 'station01'], "XGB_00+01联合")
    print("\n所有评估完成，不保存任何文件。")

if __name__ == "__main__":
    main()