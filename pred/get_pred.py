import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
     "stations": [f"station{i:02d}" for i in range(10)],   # station00 到 station09
    "output_dir": "/root/timer+exo/pred",
    "lookback": 672,
    "predict_len": 96,
    "nwp_dim": 7,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,
    "force_extract": False,      # 该参数已无用，可删除
    "stride": 96,
    "time_col": "date_time",     # 根据实际CSV中的时间列名修改
}

def extract_and_save_timer_with_info(station_name):
    """提取 Timer 预测值，并与原始数据中的日期、真实值、NWP协变量对齐保存（无缓存）"""
    stride = BASE_CONFIG["stride"]
    time_col = BASE_CONFIG["time_col"]
    output_dir = BASE_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取原始 CSV
    csv_path = os.path.join(BASE_CONFIG["data_dir"], f"{station_name}.csv")
    df_raw = pd.read_csv(csv_path)
    if time_col not in df_raw.columns:
        raise ValueError(f"CSV 中未找到时间列 '{time_col}'，可用列: {df_raw.columns.tolist()}")
    df_raw[time_col] = pd.to_datetime(df_raw[time_col])
    power = df_raw['power'].values.astype(np.float32)
    nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
                'nwp_windspeed','nwp_winddirection','nwp_pressure']
    nwp_data = df_raw[nwp_cols].values.astype(np.float32)
    # 对 NWP 进行标准化（与提取特征时保持一致，但预测时不需要保存标准化后的值）
    scaler = StandardScaler()
    nwp_data_scaled = scaler.fit_transform(nwp_data)

    # 2. 直接提取 Timer 预测值（不缓存）
    print(f"开始提取 {station_name} Timer 预测值 (stride={stride})...")
    model = TimerForPrediction.from_pretrained(BASE_CONFIG["model_path"]).to(BASE_CONFIG["device"])
    model.eval()
    preds = []
    total_steps = (len(power) - BASE_CONFIG["lookback"] - BASE_CONFIG["predict_len"]) // stride + 1
    print(f"预计样本数: {total_steps}")
    for start in range(0, len(power) - BASE_CONFIG["lookback"] - BASE_CONFIG["predict_len"] + 1, stride):
        input_seq = power[start:start+BASE_CONFIG["lookback"]]
        input_tensor = torch.tensor(input_seq).float().unsqueeze(0).to(BASE_CONFIG["device"])
        with torch.no_grad():
            mean, std = input_tensor.mean(dim=-1, keepdim=True), input_tensor.std(dim=-1, keepdim=True)
            norm_input = (input_tensor - mean) / std
            outputs = model.model(input_ids=norm_input, return_dict=True)
            feat = outputs.last_hidden_state[:, -1, :]
            pred_norm = model.lm_heads[0](feat)
            pred_raw = (pred_norm * std + mean).cpu().numpy().flatten()   # (96,)
        preds.append(pred_raw)
        if len(preds) % 200 == 0:
            print(f"{station_name} 已提取 {len(preds)} 个样本")
    preds = np.array(preds)   # (n_samples, 96)
    print(f"{station_name} 预测提取完成，样本数 {len(preds)}")

    # 3. 构建输出 DataFrame：每个预测点一行
    rows = []
    for i, start in enumerate(range(0, len(power) - BASE_CONFIG["lookback"] - BASE_CONFIG["predict_len"] + 1, stride)):
        idx_start = start + BASE_CONFIG["lookback"]
        idx_end = idx_start + BASE_CONFIG["predict_len"]
        for t in range(BASE_CONFIG["predict_len"]):
            orig_idx = idx_start + t
            row = {
                'station': station_name,
                'datetime': df_raw[time_col].iloc[orig_idx],
                'power_true': power[orig_idx],
                **{col: df_raw[col].iloc[orig_idx] for col in nwp_cols},
                'power_pred': preds[i, t]
            }
            rows.append(row)
    df_out = pd.DataFrame(rows)
    df_out.sort_values('datetime', inplace=True)

    # 4. 保存 CSV
    out_csv = os.path.join(output_dir, f"{station_name}_timer_pred_with_info.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"已保存带真实值、协变量和预测值的 CSV 至: {out_csv}")
    print(f"共 {len(df_out)} 行数据（每个预测点一行）")

def main():
    for st in BASE_CONFIG["stations"]:
        extract_and_save_timer_with_info(st)
    print("所有站点处理完成。")

if __name__ == "__main__":
    main()