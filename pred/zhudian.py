import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/root/timer/timer84m')
from modeling_timer import TimerForPrediction

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)

BASE_CONFIG = {
    "model_path": "/root/timer/timer84m",
    "data_dir": "/root/timer/甘肃光伏",
    "stations": [f"station{i:02d}" for i in range(10)],
    "lookback": 672,
    "predict_len": 1,
    "nwp_dim": 7,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,
    "stride": 1,
    "time_col": "date_time",
}

def compute_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    power_range = true.max() - true.min()
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    return mae, rmse, nmae, nrmse, r2

def predict_pointwise_and_save(station_name):
    print(f"\n处理站点: {station_name}")
    csv_path = os.path.join(BASE_CONFIG["data_dir"], f"{station_name}.csv")
    df_raw = pd.read_csv(csv_path)
    time_col = BASE_CONFIG["time_col"]
    if time_col not in df_raw.columns:
        raise ValueError(f"CSV中未找到时间列 '{time_col}'，可用列: {df_raw.columns.tolist()}")
    df_raw[time_col] = pd.to_datetime(df_raw[time_col])
    power = df_raw['power'].values.astype(np.float32)
    nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
                'nwp_windspeed','nwp_winddirection','nwp_pressure']
    
    model = TimerForPrediction.from_pretrained(BASE_CONFIG["model_path"]).to(BASE_CONFIG["device"])
    model.eval()
    
    lookback = BASE_CONFIG["lookback"]
    stride = BASE_CONFIG["stride"]
    total_points = len(power)
    start_indices = range(0, total_points - lookback, stride)
    
    preds = []
    trues = []
    datetimes = []
    nwp_records = []  # 存储每个预测点的NWP值
    
    for start in start_indices:
        input_seq = power[start:start+lookback]
        input_tensor = torch.tensor(input_seq).float().unsqueeze(0).to(BASE_CONFIG["device"])
        with torch.no_grad():
            mean, std = input_tensor.mean(dim=-1, keepdim=True), input_tensor.std(dim=-1, keepdim=True)
            norm_input = (input_tensor - mean) / std
            outputs = model.model(input_ids=norm_input, return_dict=True)
            feat = outputs.last_hidden_state[:, -1, :]
            pred_norm = model.lm_heads[0](feat)
            pred_raw = (pred_norm * std + mean).cpu().numpy().flatten()[0]
        target = power[start + lookback]
        preds.append(pred_raw)
        trues.append(target)
        orig_idx = start + lookback
        datetimes.append(df_raw[time_col].iloc[orig_idx])
        nwp_records.append([df_raw[col].iloc[orig_idx] for col in nwp_cols])
        
        if len(preds) % 10000 == 0:
            print(f"  已处理 {len(preds)} 个预测点")
    
    preds = np.array(preds)
    trues = np.array(trues)
    mae, rmse, nmae, nrmse, r2 = compute_metrics(trues, preds)
    print(f"{station_name:10s} | MAE:{mae:7.4f} | RMSE:{rmse:7.4f} | NMAE:{nmae:6.4f} | NRMSE:{nrmse:6.4f} | R2:{r2:6.4f}")
    
    # 保存到 /root/timer+exo/pred/zhudian/
    output_dir = "/root/timer+exo/pred/zhudian"
    os.makedirs(output_dir, exist_ok=True)
    out_df = pd.DataFrame({
        'datetime': datetimes,
        'power_true': trues,
        'power_pred': preds,
        'nwp_globalirrad': [x[0] for x in nwp_records],
        'nwp_directirrad': [x[1] for x in nwp_records],
        'nwp_temperature': [x[2] for x in nwp_records],
        'nwp_humidity': [x[3] for x in nwp_records],
        'nwp_windspeed': [x[4] for x in nwp_records],
        'nwp_winddirection': [x[5] for x in nwp_records],
        'nwp_pressure': [x[6] for x in nwp_records],
    })
    out_csv = os.path.join(output_dir, f"{station_name}_pointwise_pred_with_nwp.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"  已保存至 {out_csv}")
    return trues, preds, (mae, rmse, nmae, nrmse, r2)

def main():
    all_trues = []
    all_preds = []
    for st in BASE_CONFIG["stations"]:
        trues, preds, metrics = predict_pointwise_and_save(st)
        all_trues.extend(trues)
        all_preds.extend(preds)
    
    all_trues = np.array(all_trues)
    all_preds = np.array(all_preds)
    overall_mae, overall_rmse, overall_nmae, overall_nrmse, overall_r2 = compute_metrics(all_trues, all_preds)
    print("\n" + "="*70)
    print("总体评估 (所有站点合并):")
    print(f"Overall     | MAE:{overall_mae:7.4f} | RMSE:{overall_rmse:7.4f} | NMAE:{overall_nmae:6.4f} | NRMSE:{overall_nrmse:6.4f} | R2:{overall_r2:6.4f}")
    
    summary_path = os.path.join("/root/timer+exo/pred/zhudian", "summary_metrics.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Overall MAE: {overall_mae:.4f}\n")
        f.write(f"Overall RMSE: {overall_rmse:.4f}\n")
        f.write(f"Overall NMAE: {overall_nmae:.4f}\n")
        f.write(f"Overall NRMSE: {overall_nrmse:.4f}\n")
        f.write(f"Overall R2: {overall_r2:.4f}\n")
    print(f"\n总体指标已保存至 {summary_path}")

if __name__ == "__main__":
    main()