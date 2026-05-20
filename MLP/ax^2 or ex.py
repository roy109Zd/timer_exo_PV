import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random

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

PRED_DIR = "/root/timer+exo/pred"
STATION = "station00"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 30
LR = 1e-4
WEIGHT_DECAY = 1e-5
WINDOW = 96
STRIDE = 96
FEAT_DIM = 8

# ==================== 模型定义 ====================
class LinearAffine(nn.Module):
    """a*x + b，残差形式 a=1+tanh(delta_a), b=delta_b"""
    def __init__(self, in_features=FEAT_DIM, hidden_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        ab = self.fc(x)
        delta_a = ab[..., 0:1]
        delta_b = ab[..., 1:2]
        a = 1.0 + torch.tanh(delta_a)
        b = delta_b
        return a, b

class QuadraticModel(nn.Module):
    """a*x + b*x^2 + c"""
    def __init__(self, in_features=FEAT_DIM, hidden_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [a, b, c]
        )
    def forward(self, x):
        abc = self.fc(x)              # (batch, window, 3)
        a = abc[..., 0:1]
        b = abc[..., 1:2]
        c = abc[..., 2:3]
        # 为保证二次项方向合理，不对系数做额外约束
        return a, b, c

class ExpModel(nn.Module):
    """a * exp(c * x) + b，为防止指数爆炸，对 c*x 限制范围"""
    def __init__(self, in_features=FEAT_DIM, hidden_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [a, b, c]
        )
    def forward(self, x, timer_pred):
        abc = self.fc(x)              # (batch, window, 3)
        a = abc[..., 0:1]
        b = abc[..., 1:2]
        c = abc[..., 2:3]
        # 限制 c * timer_pred 的范围，避免 exp 溢出
        exponent = c * timer_pred
        exponent = torch.clamp(exponent, -10.0, 10.0)  # exp(10) ~ 22026，可接受
        exp_term = torch.exp(exponent)
        out = a * exp_term + b
        return out

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

def train_model(model, train_loader, val_loader, model_name):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for Xb, yb, tb in train_loader:
            optimizer.zero_grad()
            # 根据模型类型调用不同的 forward
            if model_name == "LinearAffine":
                a, b = model(Xb)
                pred = a * tb + b
            elif model_name == "Quadratic":
                a, b, c = model(Xb)
                pred = a * tb + b * (tb ** 2) + c
            elif model_name == "Exp":
                pred = model(Xb, tb)
            else:
                raise ValueError
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
                if model_name == "LinearAffine":
                    a, b = model(Xb)
                    pred = a * tb + b
                elif model_name == "Quadratic":
                    a, b, c = model(Xb)
                    pred = a * tb + b * (tb ** 2) + c
                elif model_name == "Exp":
                    pred = model(Xb, tb)
                loss = criterion(pred.view(-1), yb.view(-1))
                val_loss += loss.item() * len(Xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if (epoch+1) % 50 == 0:
            print(f"{model_name:15s} Epoch {epoch+1:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"{model_name} Early stopping at epoch {epoch+1}")
                break
    return model

def evaluate_model(model, X_test_t, timer_test_t, y_test_t, model_name):
    model.eval()
    with torch.no_grad():
        if model_name == "LinearAffine":
            a, b = model(X_test_t)
            pred = (a * timer_test_t + b).cpu().numpy().squeeze(-1)
        elif model_name == "Quadratic":
            a, b, c = model(X_test_t)
            pred = (a * timer_test_t + b * (timer_test_t ** 2) + c).cpu().numpy().squeeze(-1)
        elif model_name == "Exp":
            pred = model(X_test_t, timer_test_t).cpu().numpy().squeeze(-1)
        else:
            pred = None
    y_test_np = y_test_t.cpu().numpy()
    timer_test_np = timer_test_t.cpu().numpy().squeeze(-1)
    flat_true = y_test_np.flatten()
    flat_pred = pred.flatten()
    flat_timer = timer_test_np.flatten()
    print(f"\n--- {model_name} 评估 ---")
    compute_metrics(flat_true, flat_timer, "Timer原始预测")
    compute_metrics(flat_true, flat_pred, model_name)
    return flat_true, flat_pred

def main():
    print(f"单站 {STATION} | 对比线性/平方/指数模型 | 窗口={WINDOW}")
    X_all, y_all = load_windows(STATION, WINDOW, STRIDE)
    n_samples = len(X_all)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    X_train, X_val, X_test = X_all[:train_end], X_all[train_end:val_end], X_all[val_end:]
    y_train, y_val, y_test = y_all[:train_end], y_all[train_end:val_end], y_all[val_end:]

    timer_train = X_train[:, :, 0:1]
    timer_val   = X_val[:, :, 0:1]
    timer_test  = X_test[:, :, 0:1]

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

    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    timer_train_t = torch.tensor(timer_train, dtype=torch.float32, device=DEVICE)

    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)
    timer_val_t = torch.tensor(timer_val, dtype=torch.float32, device=DEVICE)

    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)
    timer_test_t = torch.tensor(timer_test, dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t, timer_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t, timer_val_t), batch_size=BATCH_SIZE, shuffle=False)

    # 训练线性模型
    print("\n" + "="*60)
    print("训练线性 Affine 模型 (a*x+b)")
    model_linear = LinearAffine()
    model_linear = train_model(model_linear, train_loader, val_loader, "LinearAffine")
    evaluate_model(model_linear, X_test_t, timer_test_t, y_test_t, "LinearAffine")

    # 训练平方模型
    print("\n" + "="*60)
    print("训练平方模型 (a*x + b*x^2 + c)")
    model_quad = QuadraticModel()
    model_quad = train_model(model_quad, train_loader, val_loader, "Quadratic")
    evaluate_model(model_quad, X_test_t, timer_test_t, y_test_t, "Quadratic")

    # 训练指数模型
    print("\n" + "="*60)
    print("训练指数模型 (a*exp(c*x)+b)")
    model_exp = ExpModel()
    model_exp = train_model(model_exp, train_loader, val_loader, "Exp")
    evaluate_model(model_exp, X_test_t, timer_test_t, y_test_t, "Exp")

    print("\n所有实验完成，不保存任何文件。")

if __name__ == "__main__":
    main()