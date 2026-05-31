import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 配置
PRED_DIR = "/root/timer+exo/pred_stride=1"
STATIONS = [f"station{i:02d}" for i in range(10)]   # station00 ~ station09
OUTPUT_CSV = os.path.join(PRED_DIR, "evaluation_metrics.csv")

def compute_metrics(true, pred, power_range):
    """
    计算 MAE, RMSE, NMAE, NRMSE, R2
    power_range: 功率极差 (max - min)
    """
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    if power_range > 0:
        nmae = mae / power_range
        nrmse = rmse / power_range
    else:
        nmae = nrmse = np.nan
    return mae, rmse, nmae, nrmse, r2

def main():
    results = []
    all_true = []
    all_pred = []
    
    for st in STATIONS:
        csv_path = os.path.join(PRED_DIR, f"{st}_timer_pred_with_info.csv")
        if not os.path.exists(csv_path):
            print(f"警告: 文件 {csv_path} 不存在，跳过 {st}")
            continue
        df = pd.read_csv(csv_path)
        # 确保有时间顺序（已排序），提取真实值和预测值
        true = df['power_true'].values.astype(float)
        pred = df['power_pred'].values.astype(float)
        # 计算该站点的功率极差（基于真实值）
        power_range = true.max() - true.min()
        mae, rmse, nmae, nrmse, r2 = compute_metrics(true, pred, power_range)
        results.append({
            'station': st,
            'MAE': mae,
            'RMSE': rmse,
            'NMAE': nmae,
            'NRMSE': nrmse,
            'R2': r2,
            'power_min': true.min(),
            'power_max': true.max(),
            'power_range': power_range
        })
        # 收集整体数据
        all_true.extend(true)
        all_pred.extend(pred)
        print(f"{st:10s} | MAE:{mae:7.4f} | RMSE:{rmse:7.4f} | NMAE:{nmae:6.4f} | NRMSE:{nrmse:6.4f} | R2:{r2:6.4f}")
    
    # 计算整体指标（所有站点合并）
    if len(all_true) > 0:
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        overall_range = all_true.max() - all_true.min()
        overall_mae, overall_rmse, overall_nmae, overall_nrmse, overall_r2 = compute_metrics(all_true, all_pred, overall_range)
        results.append({
            'station': 'overall',
            'MAE': overall_mae,
            'RMSE': overall_rmse,
            'NMAE': overall_nmae,
            'NRMSE': overall_nrmse,
            'R2': overall_r2,
            'power_min': all_true.min(),
            'power_max': all_true.max(),
            'power_range': overall_range
        })
        print("\n" + "="*60)
        print(f"Overall     | MAE:{overall_mae:7.4f} | RMSE:{overall_rmse:7.4f} | NMAE:{overall_nmae:6.4f} | NRMSE:{overall_nrmse:6.4f} | R2:{overall_r2:6.4f}")
    
    # 保存结果
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n评估结果已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()