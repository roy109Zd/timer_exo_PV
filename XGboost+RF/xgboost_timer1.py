import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
PRED_DIR = "/root/timer+exo/pred"
STATION = "station00"
TEST_RATIO = 0.1          # 测试集比例（最后10%）
VAL_RATIO = 0.1           # 验证集比例（用于早停）

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
    print(f"站点: {STATION} | XGBoost 回归 (损失函数: MSE)")
    X, y = load_data(STATION)
    X_train, X_val, X_test, y_train, y_val, y_test = split_temporal(X, y, TEST_RATIO, VAL_RATIO)
    print(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")

    # Timer 原始预测基线
    timer_pred_test = X_test[:, 0]
    print("\n" + "="*60)
    print("【Timer 原始预测值】评估 (测试集):")
    compute_metrics(y_test, timer_pred_test, "Timer原始")

    # XGBoost 模型
    print("\n训练 XGBoost (每10轮打印一次评估)...")
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        objective='reg:squarederror',   # MSE 损失
        eval_metric='rmse',             # 验证集评估指标使用 RMSE
        early_stopping_rounds=30,
        verbose=False                   # 关闭默认打印，手动控制
    )
    # 训练并设置 verbose_eval=10 每10轮打印一次
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False                   # 不使用默认 verbose
    )
    # 手动实现每10轮打印：使用回调函数或直接使用 xgboost 的 verbose_eval
    # 更简单：重新用原生 xgboost 的 train 方法，但这里使用 sklearn 接口并设置 callbacks
    # 由于 sklearn 接口的 fit 不直接支持 verbose_eval，我们改用 xgboost 原生方式
    # 但为了简单，我们重新用原生 API 并设置 verbose_eval=10
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'alpha': 0.1,
        'lambda': 1.0,
        'seed': 42
    }
    evals = [(dtrain, 'train'), (dval, 'eval')]
    model_native = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=10           # 每10轮打印一次
    )
    best_round = model_native.best_iteration
    print(f"最佳迭代次数: {best_round}")

    # 预测
    dtest = xgb.DMatrix(X_test)
    y_pred = model_native.predict(dtest)

    print("\n" + "="*60)
    print("【XGBoost】评估 (测试集):")
    compute_metrics(y_test, y_pred, "XGBoost")

    # 特征重要性
    #importance = model_native.get_score(importance_type='weight')
    importance = model_native.get_score(importance_type='gain')
    # 转换为与特征顺序对应的列表
    feature_importance = [importance.get(f'f{i}', 0) for i in range(len(FEAT_COLS))]
    total = sum(feature_importance)
    if total > 0:
        feature_importance = [f/total for f in feature_importance]
    print("\n" + "="*60)
    print("特征重要性 (XGBoost):")
    for name, imp in zip(FEAT_COLS, feature_importance):
        print(f"  {name:25s}: {imp:.4f}")

    print("\n不保存任何模型文件。")

if __name__ == "__main__":
    main()