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
STATIONS = [f"station{i:02d}" for i in range(10)]   # station00 ~ station09
TRAIN_STATION = "station00"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
PATIENCE = 10   # 早停耐心值

def compute_metrics(true, pred, power_range):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    return mae, rmse, nmae, nrmse, r2

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=8, hidden_dims=[64, 32]):
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

def load_station_data(station_name):
    """加载单个站点的特征和标签：X = [power_pred, nwp1...nwp7], y = power_true"""
    csv_path = os.path.join(PRED_DIR, f"{station_name}_timer_pred_with_info.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
                'nwp_windspeed','nwp_winddirection','nwp_pressure']
    X = df[['power_pred'] + nwp_cols].values.astype(np.float32)
    y = df['power_true'].values.astype(np.float32)
    return X, y

def main():
    # 加载训练站点的完整数据
    X_train_full, y_train_full = load_station_data(TRAIN_STATION)
    n_samples = len(X_train_full)
    # 8:1:1 划分
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    indices = np.random.permutation(n_samples)   # 随机打乱
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]
    X_test, y_test = X_train_full[test_idx], y_train_full[test_idx]
    
    # 转换为 Tensor
    X_train_t = torch.tensor(X_train).to(DEVICE)
    y_train_t = torch.tensor(y_train).to(DEVICE)
    X_val_t = torch.tensor(X_val).to(DEVICE)
    y_val_t = torch.tensor(y_val).to(DEVICE)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    
    model = SimpleMLP(input_dim=8).to(DEVICE)
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
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 评估函数
    def evaluate_on_station(station_name):
        X, y = load_station_data(station_name)
        X_t = torch.tensor(X).to(DEVICE)
        model.eval()
        with torch.no_grad():
            preds = model(X_t).cpu().numpy()
        power_range = y.max() - y.min()
        mae, rmse, nmae, nrmse, r2 = compute_metrics(y, preds, power_range)
        return mae, rmse, nmae, nrmse, r2
    
    # 在训练站的测试集上评估
    print("\n" + "="*60)
    print(f"评估 {TRAIN_STATION} 测试集 (8:1:1 中的测试部分):")
    X_test_t = torch.tensor(X_test).to(DEVICE)
    with torch.no_grad():
        pred_test = model(X_test_t).cpu().numpy()
    test_range = y_test.max() - y_test.min()
    mae, rmse, nmae, nrmse, r2 = compute_metrics(y_test, pred_test, test_range)
    print(f"MAE:{mae:7.4f} RMSE:{rmse:7.4f} NMAE:{nmae:6.4f} NRMSE:{nrmse:6.4f} R2:{r2:6.4f}")
    
    # 在其他站上评估
    print("\n" + "="*60)
    print("在其他站点上评估 (全部数据):")
    for st in STATIONS:
        if st == TRAIN_STATION:
            continue
        try:
            mae, rmse, nmae, nrmse, r2 = evaluate_on_station(st)
            print(f"{st:10s} | MAE:{mae:7.4f} RMSE:{rmse:7.4f} NMAE:{nmae:6.4f} NRMSE:{nrmse:6.4f} R2:{r2:6.4f}")
        except FileNotFoundError as e:
            print(f"{st:10s} 数据缺失: {e}")
    
    print("\n不保存模型权重和结果文件。")

if __name__ == "__main__":
    main()