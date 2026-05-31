import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random

# ==================== 固定随机种子 ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

# ==================== 配置 ====================
# PRED_DIR = "/root/timer+exo/pred"
PRED_DIR = "/root/timer+exo/pred_stride=1"
STATION = "station06"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5

WINDOW = 96
STRIDE = 96
FEAT_DIM = 8

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

# ==================== 数据加载 ====================
def load_windows(station_name, window=96, stride=96):
    csv_path = os.path.join(PRED_DIR, f"{station_name}_timer_pred_with_info.csv")
    df = pd.read_csv(csv_path)
    nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
                'nwp_windspeed','nwp_winddirection','nwp_pressure']
    feat_cols = ['power_pred'] + nwp_cols
    X = df[feat_cols].values.astype(np.float32)
    y = df['power_true'].values.astype(np.float32)
    n = len(X)
    X_windows, y_windows = [], []
    for start in range(0, n - window + 1, stride):
        X_windows.append(X[start:start+window])
        y_windows.append(y[start:start+window])
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    return X_windows, y_windows

# ==================== 模型定义 ====================

# 1. 原模型：逐点MLP仿射
class ConditionalAffine(nn.Module):
    def __init__(self, in_features=FEAT_DIM, hidden_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        ab = self.fc(x)                # (batch, window, 2)
        a = ab[..., 0:1]
        b = ab[..., 1:2]
        a = nn.functional.softplus(a) + 0.5
        return a, b

# 2. 整天->整点MLP：直接输出96个修正后的功率
class WindowToWindowMLP(nn.Module):
    def __init__(self, input_dim=WINDOW*FEAT_DIM, hidden_dims=[128, 64], output_dim=WINDOW):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        # x: (batch, window, feat_dim) -> 展平
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        out = self.net(x_flat)          # (batch, window)
        return out.unsqueeze(-1)         # (batch, window, 1) 方便与y比较

# 3. 整天->标量仿射：输出全局a,b，对整个窗口应用同一变换
class WindowToScalarAffine(nn.Module):
    def __init__(self, input_dim=WINDOW*FEAT_DIM, hidden_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # [a, b]
        )
    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        ab = self.fc(x_flat)            # (batch, 2)
        a = ab[:, 0:1]                  # (batch, 1)
        b = ab[:, 1:2]
        a = nn.functional.softplus(a) + 0.5
        # 扩展为 (batch, window, 1)
        a = a.unsqueeze(1).expand(-1, WINDOW, -1)
        b = b.unsqueeze(1).expand(-1, WINDOW, -1)
        return a, b

# 4. LSTM仿射：利用时序信息逐点输出a,b
class LSTMAffine(nn.Module):
    def __init__(self, input_size=FEAT_DIM, hidden_size=32, num_layers=2, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        lstm_out = hidden_size * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(lstm_out, 2)   # 每个时间步输出 a,b
    def forward(self, x):
        # x: (batch, window, input_size)
        out, _ = self.lstm(x)             # (batch, window, lstm_out)
        ab = self.fc(out)                 # (batch, window, 2)
        a = ab[..., 0:1]
        b = ab[..., 1:2]
        a = nn.functional.softplus(a) + 0.5
        return a, b

# ==================== 通用训练评估函数 ====================
def train_and_evaluate(model_class, model_kwargs, model_name,
                       X_train, y_train, timer_train,
                       X_val, y_val, timer_val,
                       X_test, y_test, timer_test,
                       device, batch_size, epochs, patience, lr, weight_decay):
    """
    训练给定模型并返回测试集上的指标字典
    """
    # 归一化（基于训练集）
    X_train_flat = X_train.reshape(-1, FEAT_DIM)
    feat_mean = X_train_flat.mean(axis=0, keepdims=True)
    feat_std = X_train_flat.std(axis=0, keepdims=True) + 1e-8
    def norm(X):
        shape = X.shape
        flat = X.reshape(-1, FEAT_DIM)
        flat_norm = (flat - feat_mean) / feat_std
        return flat_norm.reshape(shape)
    X_train_norm = norm(X_train)
    X_val_norm = norm(X_val)
    X_test_norm = norm(X_test)

    # 转换为Tensor
    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    timer_train_t = torch.tensor(timer_train, dtype=torch.float32, device=device)

    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    timer_val_t = torch.tensor(timer_val, dtype=torch.float32, device=device)

    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)
    timer_test_t = torch.tensor(timer_test, dtype=torch.float32, device=device)

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t, timer_train_t),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t, timer_val_t),
                            batch_size=batch_size, shuffle=False)

    # 模型初始化
    model = model_class(**model_kwargs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb, tb in train_loader:
            optimizer.zero_grad()
            # 模型输出取决于具体类，需要统一接口
            if model_name == "WindowToWindowMLP":
                pred = model(Xb)          # (batch, window, 1)
                loss = criterion(pred.view(-1), yb.view(-1))
            else:  # 其他模型返回 (a,b)
                a, b = model(Xb)
                pred = a * tb + b
                loss = criterion(pred.view(-1), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb, tb in val_loader:
                if model_name == "WindowToWindowMLP":
                    pred = model(Xb)
                    loss = criterion(pred.view(-1), yb.view(-1))
                else:
                    a, b = model(Xb)
                    pred = a * tb + b
                    loss = criterion(pred.view(-1), yb.view(-1))
                val_loss += loss.item() * len(Xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if (epoch+1) % 50 == 0:
            print(f"[{model_name}] Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{model_name}] Early stopping at epoch {epoch+1}")
                break

    # 测试评估
    model.eval()
    with torch.no_grad():
        if model_name == "WindowToWindowMLP":
            pred_test = model(X_test_t).cpu().numpy().squeeze(-1)
        else:
            a_test, b_test = model(X_test_t)
            pred_test = (a_test * timer_test_t + b_test).cpu().numpy().squeeze(-1)
    y_test_np = y_test_t.cpu().numpy()
    flat_pred = pred_test.flatten()
    flat_true = y_test_np.flatten()
    # 原始Timer预测
    timer_test_np = timer_test_t.cpu().numpy().squeeze(-1)
    flat_timer = timer_test_np.flatten()
    # 计算指标
    timer_mae, timer_rmse, _, _, timer_r2 = compute_metrics(flat_true, flat_timer, f"Timer({model_name})")
    model_mae, model_rmse, _, _, model_r2 = compute_metrics(flat_true, flat_pred, model_name)
    return {
        'name': model_name,
        'timer_mae': timer_mae,
        'timer_rmse': timer_rmse,
        'timer_r2': timer_r2,
        'model_mae': model_mae,
        'model_rmse': model_rmse,
        'model_r2': model_r2,
    }

# ==================== 主程序 ====================
def main():
    print(f"站点 {STATION} | 窗口={WINDOW} | 对比四种模型")
    # 加载数据
    X_all, y_all = load_windows(STATION, WINDOW, STRIDE)
    n_samples = len(X_all)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    X_train, X_val, X_test = X_all[:train_end], X_all[train_end:val_end], X_all[val_end:]
    y_train, y_val, y_test = y_all[:train_end], y_all[train_end:val_end], y_all[val_end:]

    timer_train = X_train[:, :, 0:1]
    timer_val   = X_val[:, :, 0:1]
    timer_test  = X_test[:, :, 0:1]

    # 定义四种模型配置
    models_info = [
        ("ConditionalAffine", ConditionalAffine, {'in_features': FEAT_DIM, 'hidden_dim': 16}),
        ("WindowToWindowMLP", WindowToWindowMLP, {'input_dim': WINDOW*FEAT_DIM, 'hidden_dims': [128, 64], 'output_dim': WINDOW}),
        ("WindowToScalarAffine", WindowToScalarAffine, {'input_dim': WINDOW*FEAT_DIM, 'hidden_dim': 32}),
        ("LSTMAffine", LSTMAffine, {'input_size': FEAT_DIM, 'hidden_size': 32, 'num_layers': 2, 'bidirectional': True})
    ]

    results = []
    for name, model_class, kwargs in models_info:
        print("\n" + "="*80)
        print(f"训练模型: {name}")
        print("="*80)
        # 每次训练前固定种子保证模型初始化一致
        set_seed(SEED)
        metrics = train_and_evaluate(
            model_class, kwargs, name,
            X_train, y_train, timer_train,
            X_val, y_val, timer_val,
            X_test, y_test, timer_test,
            DEVICE, BATCH_SIZE, EPOCHS, PATIENCE, LR, WEIGHT_DECAY
        )
        results.append(metrics)

    # 汇总表格
    print("\n\n" + "="*100)
    print("四种模型在测试集上的性能对比")
    print("="*100)
    print(f"{'模型':<22} | Timer MAE | Model MAE | Timer RMSE | Model RMSE | Timer R2 | Model R2")
    print("-"*100)
    for r in results:
        print(f"{r['name']:<22} | {r['timer_mae']:9.4f} | {r['model_mae']:9.4f} | {r['timer_rmse']:10.4f} | {r['model_rmse']:10.4f} | {r['timer_r2']:8.4f} | {r['model_r2']:8.4f}")

    # 额外打印每个模型相对于Timer的提升百分比
    print("\n" + "="*100)
    print("相对于Timer原始预测的MAE改善率 (负值表示变差)")
    print("-"*100)
    for r in results:
        improvement = (r['timer_mae'] - r['model_mae']) / r['timer_mae'] * 100
        print(f"{r['name']:<22} | MAE改善: {improvement:6.2f}%")

if __name__ == "__main__":
    main()