# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # 配置
# PRED_DIR = "/root/timer+exo/pred"
# STATIONS = [f"station{i:02d}" for i in range(10)]
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 256
# EPOCHS = 100
# LR = 1e-3
# PATIENCE = 10

# # 方案选择: 1 = 只使用station00 + 归一化; 2 = 使用station00+station01训练, 不归一化
# SCHEME = 2   # 根据需要修改为1或2

# def load_station_data(station_name):
#     csv_path = os.path.join(PRED_DIR, f"{station_name}_timer_pred_with_info.csv")
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"文件不存在: {csv_path}")
#     df = pd.read_csv(csv_path)
#     nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
#                 'nwp_windspeed','nwp_winddirection','nwp_pressure']
#     X = df[['power_pred'] + nwp_cols].values.astype(np.float32)
#     y = df['power_true'].values.astype(np.float32)
#     return X, y

# class SimpleMLP(nn.Module):
#     def __init__(self, input_dim=8, hidden_dims=[64, 32]):
#         super().__init__()
#         layers = []
#         prev_dim = input_dim
#         for hdim in hidden_dims:
#             layers.append(nn.Linear(prev_dim, hdim))
#             layers.append(nn.ReLU())
#             prev_dim = hdim
#         layers.append(nn.Linear(prev_dim, 1))
#         self.net = nn.Sequential(*layers)
#     def forward(self, x):
#         return self.net(x).squeeze(-1)

# def compute_metrics(true, pred, power_range):
#     mae = mean_absolute_error(true, pred)
#     rmse = np.sqrt(mean_squared_error(true, pred))
#     r2 = r2_score(true, pred)
#     nmae = mae / power_range if power_range > 0 else np.nan
#     nrmse = rmse / power_range if power_range > 0 else np.nan
#     return mae, rmse, nmae, nrmse, r2

# def train_model(X_train, y_train, X_val, y_val, input_dim=8):
#     train_loader = DataLoader(TensorDataset(torch.tensor(X_train).to(DEVICE), 
#                                             torch.tensor(y_train).to(DEVICE)), 
#                               batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(TensorDataset(torch.tensor(X_val).to(DEVICE), 
#                                           torch.tensor(y_val).to(DEVICE)), 
#                             batch_size=BATCH_SIZE, shuffle=False)
    
#     model = SimpleMLP(input_dim=input_dim).to(DEVICE)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=LR)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
#     best_val_loss = float('inf')
#     patience_counter = 0
#     for epoch in range(EPOCHS):
#         model.train()
#         train_loss = 0.0
#         for Xb, yb in train_loader:
#             optimizer.zero_grad()
#             pred = model(Xb)
#             loss = criterion(pred, yb)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * len(Xb)
#         train_loss /= len(train_loader.dataset)
        
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for Xb, yb in val_loader:
#                 pred = model(Xb)
#                 loss = criterion(pred, yb)
#                 val_loss += loss.item() * len(Xb)
#         val_loss /= len(val_loader.dataset)
#         scheduler.step(val_loss)
        
#         if (epoch+1) % 20 == 0:
#             print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= PATIENCE:
#                 print(f"Early stopping at epoch {epoch+1}")
#                 break
#     return model

# def scheme1():
#     print("方案1: 只使用 station00 训练 + 输入归一化")
#     X, y = load_station_data("station00")
#     n_samples = len(X)
#     train_end = int(n_samples * 0.8)
#     val_end = int(n_samples * 0.9)
#     # 按时间顺序划分
#     X_train, y_train = X[:train_end], y[:train_end]
#     X_val, y_val = X[train_end:val_end], y[train_end:val_end]
#     X_test, y_test = X[val_end:], y[val_end:]
    
#     # 计算训练集的均值和标准差
#     X_mean = X_train.mean(axis=0, keepdims=True)
#     X_std = X_train.std(axis=0, keepdims=True) + 1e-8
#     X_train_norm = (X_train - X_mean) / X_std
#     X_val_norm = (X_val - X_mean) / X_std
#     X_test_norm = (X_test - X_mean) / X_std
    
#     print(f"训练样本: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")
#     model = train_model(X_train_norm, y_train, X_val_norm, y_val, input_dim=8)
    
#     # 评估 station00 测试集
#     model.eval()
#     with torch.no_grad():
#         pred_test = model(torch.tensor(X_test_norm).to(DEVICE)).cpu().numpy()
#     test_range = y_test.max() - y_test.min()
#     mae, rmse, nmae, nrmse, r2 = compute_metrics(y_test, pred_test, test_range)
#     print(f"\nstation00 测试集结果: MAE={mae:.4f}, RMSE={rmse:.4f}, NMAE={nmae:.4f}, NRMSE={nrmse:.4f}, R2={r2:.4f}")
    
#     # 评估其他站
#     print("\n在其他站点上评估 (全部数据，使用训练集均值和标准差归一化输入):")
#     for st in STATIONS:
#         if st == "station00":
#             continue
#         X_other, y_other = load_station_data(st)
#         X_other_norm = (X_other - X_mean) / X_std
#         with torch.no_grad():
#             pred_other = model(torch.tensor(X_other_norm).to(DEVICE)).cpu().numpy()
#         power_range = y_other.max() - y_other.min()
#         mae, rmse, nmae, nrmse, r2 = compute_metrics(y_other, pred_other, power_range)
#         print(f"{st:10s} | MAE:{mae:7.4f} RMSE:{rmse:7.4f} NMAE:{nmae:6.4f} NRMSE:{nrmse:6.4f} R2:{r2:6.4f}")

# def scheme2():
#     print("方案2: 使用 station00 + station01 训练, 不归一化")
#     X0, y0 = load_station_data("station00")
#     X1, y1 = load_station_data("station01")
#     X = np.concatenate([X0, X1], axis=0)
#     y = np.concatenate([y0, y1], axis=0)
#     indices = np.random.permutation(len(X))
#     X, y = X[indices], y[indices]
#     n_samples = len(X)
#     train_end = int(n_samples * 0.8)
#     val_end = int(n_samples * 0.9)
#     X_train, y_train = X[:train_end], y[:train_end]
#     X_val, y_val = X[train_end:val_end], y[train_end:val_end]
#     X_test, y_test = X[val_end:], y[val_end:]
#     print(f"训练样本: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)} (来自00+01合并)")
    
#     model = train_model(X_train, y_train, X_val, y_val, input_dim=8)
    
#     # 在合并测试集上评估
#     model.eval()
#     with torch.no_grad():
#         pred_test = model(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
#     test_range = y_test.max() - y_test.min()
#     mae, rmse, nmae, nrmse, r2 = compute_metrics(y_test, pred_test, test_range)
#     print(f"\n合并测试集结果: MAE={mae:.4f}, RMSE={rmse:.4f}, NMAE={nmae:.4f}, NRMSE={nrmse:.4f}, R2={r2:.4f}")
    
#     # 在其他站点上评估
#     print("\n在其他站点上评估 (全部数据，不归一化):")
#     for st in STATIONS:
#         if st in ["station00", "station01"]:
#             continue
#         X_other, y_other = load_station_data(st)
#         with torch.no_grad():
#             pred_other = model(torch.tensor(X_other).to(DEVICE)).cpu().numpy()
#         power_range = y_other.max() - y_other.min()
#         mae, rmse, nmae, nrmse, r2 = compute_metrics(y_other, pred_other, power_range)
#         print(f"{st:10s} | MAE:{mae:7.4f} RMSE:{rmse:7.4f} NMAE:{nmae:6.4f} NRMSE:{nrmse:6.4f} R2:{r2:6.4f}")

# def main():
#     if SCHEME == 1:
#         scheme1()
#     elif SCHEME == 2:
#         scheme2()
#     else:
#         print("SCHEME 必须是 1 或 2")

# if __name__ == "__main__":
#     main()
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
STATIONS = [f"station{i:02d}" for i in range(10)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
PATIENCE = 10

def load_station_data(station_name):
    csv_path = os.path.join(PRED_DIR, f"{station_name}_timer_pred_with_info.csv")
    df = pd.read_csv(csv_path)
    nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
                'nwp_windspeed','nwp_winddirection','nwp_pressure']
    X = df[['power_pred'] + nwp_cols].values.astype(np.float32)
    y = df['power_true'].values.astype(np.float32)
    return X, y

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

def compute_metrics(true, pred, power_range):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    return mae, rmse, nmae, nrmse, r2

def split_data(X, y, train_ratio=0.8, val_ratio=0.1):
    """随机划分训练、验证、测试集"""
    n = len(X)
    indices = np.random.permutation(n)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]

def scheme3():
    print("方案3: 使用 station00 + station01 训练 + 输入归一化，分别测试每个站点")
    # 加载00和01数据
    X0, y0 = load_station_data("station00")
    X1, y1 = load_station_data("station01")
    
    # 分别划分
    X0_train, y0_train, X0_val, y0_val, X0_test, y0_test = split_data(X0, y0)
    X1_train, y1_train, X1_val, y1_val, X1_test, y1_test = split_data(X1, y1)
    
    # 合并训练集和验证集
    X_train = np.concatenate([X0_train, X1_train], axis=0)
    y_train = np.concatenate([y0_train, y1_train], axis=0)
    X_val = np.concatenate([X0_val, X1_val], axis=0)
    y_val = np.concatenate([y0_val, y1_val], axis=0)
    
    # 计算归一化参数（基于训练集）
    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std = X_train.std(axis=0, keepdims=True) + 1e-8
    
    # 归一化
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std
    
    print(f"训练样本数: {len(X_train)} (station00: {len(X0_train)}, station01: {len(X1_train)})")
    print(f"验证样本数: {len(X_val)}")
    
    # 训练模型
    model = SimpleMLP(input_dim=8).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_norm).to(DEVICE),
                                            torch.tensor(y_train).to(DEVICE)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_norm).to(DEVICE),
                                          torch.tensor(y_val).to(DEVICE)),
                            batch_size=BATCH_SIZE, shuffle=False)
    
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
    
    # 定义一个评估函数
    def evaluate(station_name, X_data, y_data, desc):
        X_norm = (X_data - X_mean) / X_std
        with torch.no_grad():
            pred = model(torch.tensor(X_norm).to(DEVICE)).cpu().numpy()
        power_range = y_data.max() - y_data.min()
        mae, rmse, nmae, nrmse, r2 = compute_metrics(y_data, pred, power_range)
        print(f"{desc:15s} | MAE:{mae:7.4f} RMSE:{rmse:7.4f} NMAE:{nmae:6.4f} NRMSE:{nrmse:6.4f} R2:{r2:6.4f}")
        return mae, rmse, nmae, nrmse, r2
    
    print("\n" + "="*70)
    print("测试结果:")
    # 评估 station00 测试集
    evaluate("station00", X0_test, y0_test, "station00(test)")
    # 评估 station01 测试集
    evaluate("station01", X1_test, y1_test, "station01(test)")
    # 评估其他站点的全部数据
    for st in STATIONS:
        if st in ["station00", "station01"]:
            continue
        X_other, y_other = load_station_data(st)
        evaluate(st, X_other, y_other, f"{st}(all)")

if __name__ == "__main__":
    scheme3()