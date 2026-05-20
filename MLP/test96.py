import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 配置
PRED_DIR = "/root/timer+exo/pred"
STATION = "station00"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 150
LR = 5e-4
PATIENCE = 15
SEQ_LEN = 96
HIDDEN_SIZE = 128
NUM_LAYERS = 2

def compute_metrics(true, pred, power_range):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    return mae, rmse, nmae, nrmse, r2

class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_dim=8, hidden_size=128, num_layers=2, output_len=96):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, output_len)
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]        # (batch, hidden)
        out = self.linear(last_out)          # (batch, output_len)
        return out

def load_station_data(station_name):
    csv_path = os.path.join(PRED_DIR, f"{station_name}_timer_pred_with_info.csv")
    df = pd.read_csv(csv_path)
    nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
                'nwp_windspeed','nwp_winddirection','nwp_pressure']
    features = df[['power_pred'] + nwp_cols].values.astype(np.float32)
    targets = df['power_true'].values.astype(np.float32)
    return features, targets

def create_sequences(features, targets, seq_len):
    X, y = [], []
    n = len(features)
    for i in range(n - seq_len + 1):
        X.append(features[i:i+seq_len])
        y.append(targets[i:i+seq_len])
    return np.array(X), np.array(y)

def main():
    print(f"单站 {STATION} LSTM 序列预测 (seq_len={SEQ_LEN})")
    features, targets = load_station_data(STATION)
    X_all, y_all = create_sequences(features, targets, SEQ_LEN)
    
    # 归一化输出（真实功率）
    y_mean = y_all.mean()
    y_std = y_all.std() + 1e-8
    y_all_norm = (y_all - y_mean) / y_std
    
    n_samples = len(X_all)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    X_train, y_train = X_all[:train_end], y_all_norm[:train_end]
    X_val, y_val = X_all[train_end:val_end], y_all_norm[train_end:val_end]
    X_test, y_test = X_all[val_end:], y_all_norm[val_end:]
    print(f"样本数: 训练 {len(X_train)}, 验证 {len(X_val)}, 测试 {len(X_test)}")
    
    X_train_t = torch.tensor(X_train).to(DEVICE)
    y_train_t = torch.tensor(y_train).to(DEVICE)
    X_val_t = torch.tensor(X_val).to(DEVICE)
    y_val_t = torch.tensor(y_val).to(DEVICE)
    X_test_t = torch.tensor(X_test).to(DEVICE)
    y_test_t = torch.tensor(y_test).to(DEVICE)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    
    model = LSTMSeq2Seq(input_dim=8, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_len=SEQ_LEN).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)          # (batch, seq_len)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 测试
    model.eval()
    with torch.no_grad():
        pred_test_norm = model(X_test_t).cpu().numpy()
    # 反归一化
    pred_test = pred_test_norm * y_std + y_mean
    y_test_true = y_test_t.cpu().numpy() * y_std + y_mean
    pred_flat = pred_test.flatten()
    true_flat = y_test_true.flatten()
    power_range = true_flat.max() - true_flat.min()
    mae, rmse, nmae, nrmse, r2 = compute_metrics(true_flat, pred_flat, power_range)
    print(f"\n测试集结果: MAE={mae:.4f}, RMSE={rmse:.4f}, NMAE={nmae:.4f}, NRMSE={nrmse:.4f}, R2={r2:.4f}")

if __name__ == "__main__":
    main()