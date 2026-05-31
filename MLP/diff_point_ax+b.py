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

# ==================== 全局配置 ====================
PRED_DIR = "/root/timer+exo/pred"   # 请根据实际路径修改
STATIONS = [f"station{i:02d}" for i in range(10)]   # station00 ~ station09
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
HISTORY_LENS = [1,4,16,  96]   # 窗口长度
FEAT_DIM = 8                     # 每个时间步的特征数: power_pred + 7个NWP

# ==================== 条件仿射模型 (a * timer_pred + b) ====================
class ConditionalAffine(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        """
        input_dim: 窗口长度 * FEAT_DIM (展平后的特征维度)
        实际上为了处理窗口内逐点预测，输入形状 (batch, window, FEAT_DIM)
        我们在 forward 中对每个时间步独立计算 a,b
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # 输出 [a, b]
        )
    def forward(self, x):
        # x: (batch, window, FEAT_DIM)
        batch, window, _ = x.shape
        # 将窗口维度展平到 batch 维度，以便每个时间步独立计算
        x_flat = x.view(-1, FEAT_DIM)          # (batch*window, FEAT_DIM)
        ab = self.fc(x_flat)                   # (batch*window, 2)
        a = ab[:, 0:1]                         # (batch*window, 1)
        b = ab[:, 1:2]
        # 恢复窗口形状
        a = a.view(batch, window, 1)           # (batch, window, 1)
        b = b.view(batch, window, 1)
        # 对 a 施加 softplus 确保为正，并加偏置使初始 a 接近 1
        a = nn.functional.softplus(a) + 0.5
        return a, b

# ==================== 评估指标 ====================
def compute_metrics(true, pred, name=""):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    power_range = true.max() - true.min()
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    if name:
        print(f"{name:20s} | MAE:{mae:7.4f} | RMSE:{rmse:7.4f} | NMAE:{nmae:6.4f} | NRMSE:{nrmse:6.4f} | R2:{r2:6.4f}")
    return mae, rmse, nmae, nrmse, r2

# ==================== 构造滑动窗口数据（步长=1） ====================
def create_windows(df, window_len):
    """
    参数:
        df: DataFrame，包含特征列和 'power_true' 列，已按时间排序
        window_len: 窗口长度（历史点数，包含当前点）
    返回:
        X: (n_samples, window_len, FEAT_DIM)  特征窗口
        y: (n_samples, window_len)            真实功率窗口（与X时间对齐）
        timer_pred: (n_samples, window_len, 1) 原始timer预测值窗口
    """
    nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
                'nwp_windspeed','nwp_winddirection','nwp_pressure']
    feat_cols = ['power_pred'] + nwp_cols
    data = df[feat_cols].values.astype(np.float32)   # (T, 8)
    y = df['power_true'].values.astype(np.float32)   # (T,)
    T = len(data)
    if T < window_len:
        raise ValueError(f"序列长度 {T} 小于窗口长度 {window_len}")
    X_list, y_list, timer_list = [], [], []
    for i in range(window_len - 1, T):
        start = i - window_len + 1
        X_list.append(data[start:i+1])               # (window_len, 8)
        y_list.append(y[start:i+1])                  # (window_len,)
        timer_list.append(data[start:i+1, 0:1])      # (window_len, 1)
    X = np.array(X_list)                             # (N, window_len, 8)
    y = np.array(y_list)                             # (N, window_len)
    timer_pred = np.array(timer_list)                # (N, window_len, 1)
    return X, y, timer_pred

# ==================== 训练单个窗口配置 ====================
def train_for_window(station_name, window_len):
    print(f"\n>>> 处理站点: {station_name}, 窗口长度: {window_len}")
    # 加载原始数据
    csv_path = os.path.join(PRED_DIR, f"{station_name}_timer_pred_with_info.csv")
    if not os.path.exists(csv_path):
        print(f"  文件不存在，跳过")
        return None
    df = pd.read_csv(csv_path)
    # 确保按时间排序（如果有时间列）
    if 'time' in df.columns:
        df = df.sort_values('time')
    
    try:
        X, y, timer_pred = create_windows(df, window_len)
    except ValueError as e:
        print(f"  窗口构造失败: {e}")
        return None
    
    n_samples = len(X)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    timer_train, timer_val, timer_test = timer_pred[:train_end], timer_pred[train_end:val_end], timer_pred[val_end:]
    
    # 归一化（基于训练集所有时间步的特征）
    X_train_flat = X_train.reshape(-1, FEAT_DIM)
    feat_mean = X_train_flat.mean(axis=0, keepdims=True)
    feat_std = X_train_flat.std(axis=0, keepdims=True) + 1e-8
    def norm(x):
        shape = x.shape
        flat = x.reshape(-1, FEAT_DIM)
        flat_norm = (flat - feat_mean) / feat_std
        return flat_norm.reshape(shape)
    
    X_train_norm = norm(X_train)
    X_val_norm   = norm(X_val)
    X_test_norm  = norm(X_test)
    
    # 转换为Tensor
    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    timer_train_t = torch.tensor(timer_train, dtype=torch.float32, device=DEVICE)
    
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)
    timer_val_t = torch.tensor(timer_val, dtype=torch.float32, device=DEVICE)
    
    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)
    timer_test_t = torch.tensor(timer_test, dtype=torch.float32, device=DEVICE)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t, timer_train_t),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t, timer_val_t),
                            batch_size=BATCH_SIZE, shuffle=False)
    
    model = ConditionalAffine(input_dim=window_len*FEAT_DIM, hidden_dim=16).to(DEVICE)
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
            a, b = model(Xb)                     # (batch, window, 1)
            pred = a * tb + b                    # (batch, window, 1)
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
                a, b = model(Xb)
                pred = a * tb + b
                loss = criterion(pred.view(-1), yb.view(-1))
                val_loss += loss.item() * len(Xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        if (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # 测试集评估
    model.eval()
    with torch.no_grad():
        a_test, b_test = model(X_test_t)                    # (n_test, window, 1)
        pred_test = (a_test * timer_test_t + b_test).cpu().numpy().squeeze(-1)  # (n_test, window)
        a_test_np = a_test.cpu().numpy().squeeze(-1)        # (n_test, window)
        b_test_np = b_test.cpu().numpy().squeeze(-1)
    
    y_test_np = y_test_t.cpu().numpy()
    timer_test_np = timer_test_t.cpu().numpy().squeeze(-1)
    
    flat_true = y_test_np.flatten()
    flat_timer = timer_test_np.flatten()
    flat_pred = pred_test.flatten()
    
    # 计算指标
    mae, rmse, nmae, nrmse, r2 = compute_metrics(flat_true, flat_pred, f"{station_name}_n={window_len}")
    
    # 统计 a, b
    a_flat = a_test_np.flatten()
    b_flat = b_test_np.flatten()
    a_stats = {
        'a_mean': a_flat.mean(), 'a_std': a_flat.std(), 'a_min': a_flat.min(),
        'a_max': a_flat.max(), 'a_median': np.median(a_flat)
    }
    b_stats = {
        'b_mean': b_flat.mean(), 'b_std': b_flat.std(), 'b_min': b_flat.min(),
        'b_max': b_flat.max(), 'b_median': np.median(b_flat)
    }
    
    result = {
        'station': station_name,
        'window_len': window_len,
        'MAE': mae, 'RMSE': rmse, 'NMAE': nmae, 'NRMSE': nrmse, 'R2': r2,
        **a_stats, **b_stats
    }
    return result

# ==================== 主函数 ====================
def main():
    all_results = []
    for station in STATIONS:
        for n in HISTORY_LENS:
            res = train_for_window(station, n)
            if res is not None:
                all_results.append(res)
    
    if not all_results:
        print("没有产生任何结果，请检查数据路径。")
        return
    
    df_results = pd.DataFrame(all_results)
    output_path = "/root/timer+exo/evaluation_results_ax+b.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n所有结果已保存至: {output_path}")
    print(df_results.to_string())

if __name__ == "__main__":
    main()