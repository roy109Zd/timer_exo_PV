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
BATCH_SIZE = 64
EPOCHS = 150
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-5

WINDOW = 96
STRIDE = 96
FEAT_DIM = 8

class ResidualMLP(nn.Module):
    def __init__(self, in_features=FEAT_DIM, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.mlp(x)   # (batch, window, 1)

def compute_metrics(true, pred, name=""):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    power_range = true.max() - true.min()
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    print(f"{name:20s} | MAE:{mae:7.4f} | RMSE:{rmse:7.4f} | NMAE:{nmae:6.4f} | NRMSE:{nrmse:6.4f} | R2:{r2:6.4f}")
    return mae, rmse, nmae, nrmse, r2

def load_windows(station_name):
    csv_path = os.path.join(PRED_DIR, f"{station_name}_timer_pred_with_info.csv")
    df = pd.read_csv(csv_path)
    nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
                'nwp_windspeed','nwp_winddirection','nwp_pressure']
    feat_cols = ['power_pred'] + nwp_cols
    X = df[feat_cols].values.astype(np.float32)
    y = df['power_true'].values.astype(np.float32)
    n = len(X)
    X_w, y_w = [], []
    for start in range(0, n - WINDOW + 1, STRIDE):
        X_w.append(X[start:start+WINDOW])
        y_w.append(y[start:start+WINDOW])
    return np.array(X_w), np.array(y_w)

def main():
    print(f"站点: {STATION}, 窗口={WINDOW}, 步长={STRIDE}")
    X_all, y_all = load_windows(STATION)
    n_samples = len(X_all)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    X_train, X_val, X_test = X_all[:train_end], X_all[train_end:val_end], X_all[val_end:]
    y_train, y_val, y_test = y_all[:train_end], y_all[train_end:val_end], y_all[val_end:]

    # 残差标签
    timer_train = X_train[:, :, 0]
    timer_val   = X_val[:, :, 0]
    timer_test  = X_test[:, :, 0]
    residual_train = y_train - timer_train
    residual_val   = y_val   - timer_val
    residual_test  = y_test  - timer_test

    # 输入归一化
    X_train_flat = X_train.reshape(-1, FEAT_DIM)
    feat_mean = X_train_flat.mean(axis=0, keepdims=True)
    feat_std = X_train_flat.std(axis=0, keepdims=True) + 1e-8
    def norm(X):
        shape = X.shape
        flat = X.reshape(-1, FEAT_DIM)
        flat_norm = (flat - feat_mean) / feat_std
        return flat_norm.reshape(shape)
    X_train_norm = norm(X_train)
    X_val_norm   = norm(X_val)
    X_test_norm  = norm(X_test)

    # 转换为 Tensor 并移到设备
    X_train_t = torch.tensor(X_train_norm).float().to(DEVICE)
    residual_train_t = torch.tensor(residual_train).float().to(DEVICE)
    X_val_t   = torch.tensor(X_val_norm).float().to(DEVICE)
    residual_val_t   = torch.tensor(residual_val).float().to(DEVICE)
    X_test_t  = torch.tensor(X_test_norm).float().to(DEVICE)
    residual_test_t  = torch.tensor(residual_test).float().to(DEVICE)
    timer_test_tensor = torch.tensor(timer_test).float().to(DEVICE)   # 用于最终预测

    train_loader = DataLoader(TensorDataset(X_train_t, residual_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, residual_val_t), batch_size=BATCH_SIZE, shuffle=False)

    model = ResidualMLP().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for Xb, rb in train_loader:
            optimizer.zero_grad()
            delta = model(Xb).squeeze(-1)
            loss = criterion(delta, rb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, rb in val_loader:
                delta = model(Xb).squeeze(-1)
                loss = criterion(delta, rb)
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
        delta_test = model(X_test_t).cpu().numpy().squeeze(-1)
    timer_test_np = timer_test
    y_test_np = y_test
    final_pred = timer_test_np + delta_test

    flat_timer = timer_test_np.flatten()
    flat_final = final_pred.flatten()
    flat_true = y_test_np.flatten()

    print("\n" + "="*60)
    compute_metrics(flat_true, flat_timer, "Timer原始预测")
    compute_metrics(flat_true, flat_final, "Timer + 残差MLP")

if __name__ == "__main__":
    main()