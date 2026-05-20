import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
PRED_DIR = "/root/timer+exo/pred"
STATION = "station00"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 200
PATIENCE = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

WINDOW = 96
STRIDE = 96
FEAT_DIM = 8

# ==================== 模型（每个时间步独立，输入整个窗口）====================
class StepMLPWithContext(nn.Module):
    """输入整个窗口 (batch, window, feat_dim) -> 展平 -> MLP -> 输出 (a, b)"""
    def __init__(self, input_dim=WINDOW*FEAT_DIM, hidden_dims=[128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 2))   # 输出 [a, b]
        self.net = nn.Sequential(*layers)
        # 初始化最后一层偏置使初始 a≈1, b≈0
        with torch.no_grad():
            self.net[-1].bias.data[0] = 1.0   # a 偏置初始1
            self.net[-1].bias.data[1] = 0.0   # b 偏置初始0

    def forward(self, x):
        # x: (batch, window, feat_dim)
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)      # (batch, window*feat_dim)
        ab = self.net(x_flat)                # (batch, 2)
        a = torch.sigmoid(ab[:, 0:1]) * 2.0  # 限制 a 在 (0,2)，初始约1
        b = ab[:, 1:2]
        return a, b

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

# ==================== 数据加载与窗口构建 ====================
def load_windows(station):
    csv_path = os.path.join(PRED_DIR, f"{station}_timer_pred_with_info.csv")
    df = pd.read_csv(csv_path)
    df = df.sort_values('datetime').reset_index(drop=True)
    feat_cols = ['power_pred', 'nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature',
                 'nwp_humidity', 'nwp_windspeed', 'nwp_winddirection', 'nwp_pressure']
    X = df[feat_cols].values.astype(np.float32)
    y = df['power_true'].values.astype(np.float32)
    n = len(X)
    X_windows, y_windows = [], []
    for start in range(0, n - WINDOW + 1, STRIDE):
        X_windows.append(X[start:start+WINDOW])
        y_windows.append(y[start:start+WINDOW])
    X_windows = np.array(X_windows)  # (num_samples, WINDOW, FEAT_DIM)
    y_windows = np.array(y_windows)  # (num_samples, WINDOW)
    return X_windows, y_windows

# ==================== 训练单个时间步的模型 ====================
def train_step_model(X_train, y_train, X_val, y_val, timer_train, timer_val, step_idx):
    """
    X_train: (n_train, window, feat_dim)
    y_train: (n_train,)  该时间步的真实功率
    timer_train: (n_train,) 该时间步的 timer_pred
    """
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(timer_train, dtype=torch.float32)
    ), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(timer_val, dtype=torch.float32)
    ), batch_size=BATCH_SIZE, shuffle=False)

    model = StepMLPWithContext(input_dim=WINDOW*FEAT_DIM, hidden_dims=[128,64], dropout=DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for Xb, yb, tb in train_loader:
            Xb, yb, tb = Xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            optimizer.zero_grad()
            a, b = model(Xb)
            pred = a.squeeze() * tb + b.squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb, tb in val_loader:
                Xb, yb, tb = Xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
                a, b = model(Xb)
                pred = a.squeeze() * tb + b.squeeze()
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(Xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if (epoch+1) % 50 == 0:
            print(f"  时间步 {step_idx:2d} Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  时间步 {step_idx:2d} 早停于 epoch {epoch+1}")
                break
    return model

# ==================== 主程序 ====================
def main():
    print(f"站点: {STATION} | 每个时间步独立MLP，输入整个窗口 (96x8)")
    X_windows, y_windows = load_windows(STATION)
    n_samples = len(X_windows)
    print(f"总窗口数: {n_samples}")

    # 时间顺序 8:1:1 划分窗口
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    X_train_win = X_windows[:train_end]   # (train_win, 96, 8)
    y_train_win = y_windows[:train_end]   # (train_win, 96)
    X_val_win   = X_windows[train_end:val_end]
    y_val_win   = y_windows[train_end:val_end]
    X_test_win  = X_windows[val_end:]
    y_test_win  = y_windows[val_end:]

    print(f"训练窗口数: {len(X_train_win)}, 验证窗口数: {len(X_val_win)}, 测试窗口数: {len(X_test_win)}")

    # 对每个时间步 t，构造训练集和验证集（输入为整个窗口，输出为该时间步的功率和timer_pred）
    models = []
    for t in range(WINDOW):
        # 提取该时间步的标签和 timer_pred
        y_t_train = y_train_win[:, t]          # (n_train,)
        y_t_val   = y_val_win[:, t]
        timer_t_train = X_train_win[:, t, 0]   # 每个窗口第t个点的第一列（power_pred）
        timer_t_val   = X_val_win[:, t, 0]
        print(f"训练时间步 {t+1}/{WINDOW} (样本数: {len(y_t_train)})...")
        model = train_step_model(X_train_win, y_t_train, X_val_win, y_t_val,
                                 timer_t_train, timer_t_val, t)
        models.append(model)

    # 测试：对每个测试窗口的每个时间步，用对应模型预测
    all_preds = []
    all_trues = []
    for i in range(len(X_test_win)):
        X_win = X_test_win[i]           # (96, 8)
        y_win = y_test_win[i]           # (96,)
        pred_win = np.zeros(WINDOW)
        # 对每个时间步独立预测
        for t in range(WINDOW):
            model = models[t]
            model.eval()
            # 输入整个窗口（但需要增加batch维度）
            x_input = torch.tensor(X_win, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1,96,8)
            timer_pred = X_win[t, 0]    # 当前时间步的原始Timer预测值
            with torch.no_grad():
                a, b = model(x_input)
                pred = a.squeeze().cpu().numpy() * timer_pred + b.squeeze().cpu().numpy()
                pred_win[t] = pred
        all_preds.extend(pred_win)
        all_trues.extend(y_win)

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    print("\n" + "="*60)
    print("整体测试集评估 (展平所有96点):")
    compute_metrics(all_trues, all_preds, "独立MLP(全窗口)")

    # Timer 原始预测基线
    timer_pred_test = X_test_win[:, :, 0].flatten()
    print("\nTimer 原始预测基线:")
    compute_metrics(all_trues, timer_pred_test, "Timer原始")

    print("\n不保存任何模型文件。")

if __name__ == "__main__":
    main()