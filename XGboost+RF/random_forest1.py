import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
PRED_DIR = "/root/timer+exo/pred"
STATION = "station02"
TEST_RATIO = 0.1          # 测试集比例（最后10%）
VAL_RATIO = 0.1           # 验证集比例（仅用于与XGBoost保持一致划分，RF不使用早停）

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

# ==================== 数据加载与划分 ====================
def load_data(station):
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

# ==================== 主程序 ====================
def main():
    print(f"站点: {STATION} | Random Forest 回归")
    X, y = load_data(STATION)
    X_train, X_val, X_test, y_train, y_val, y_test = split_temporal(X, y, TEST_RATIO, VAL_RATIO)
    print(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")
    print("注意: Random Forest 不使用验证集，仅用于划分一致性。")

    # Timer 原始预测基线
    timer_pred_test = X_test[:, 0]
    print("\n" + "="*60)
    print("【Timer 原始预测值】评估 (测试集):")
    compute_metrics(y_test, timer_pred_test, "Timer原始")

    # Random Forest 模型
    print("\n训练 Random Forest...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n" + "="*60)
    print("【Random Forest】评估 (测试集):")
    compute_metrics(y_test, y_pred, "RandomForest")

    # 特征重要性
    print("\n" + "="*60)
    print("特征重要性 (Random Forest):")
    for name, imp in zip(FEAT_COLS, model.feature_importances_):
        print(f"  {name:25s}: {imp:.4f}")

    print("\n不保存任何模型文件。")

if __name__ == "__main__":
    main()