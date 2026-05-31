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
PRED_DIR = "/root/timer+exo/pred"
STATION = "station00"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5

WINDOW = 96
STRIDE = 96
FEAT_DIM = 8

# ==================== 条件 Affine 模型 ====================
class ConditionalAffine(nn.Module):
    def __init__(self, in_features=FEAT_DIM, hidden_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        ab = self.fc(x)
        a = ab[..., 0:1]
        b = ab[..., 1:2]
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

# ==================== 训练函数（给定训练数据子集） ====================
def train_and_evaluate(X_train_sub, y_train_sub, timer_train_sub,
                       X_val, y_val, timer_val,
                       X_test, y_test, timer_test,
                       ratio_name):
    """
    使用指定的训练子集训练模型，返回测试集上的各项指标
    """
    # 归一化：基于训练子集
    X_train_flat = X_train_sub.reshape(-1, FEAT_DIM)
    feat_mean = X_train_flat.mean(axis=0, keepdims=True)
    feat_std = X_train_flat.std(axis=0, keepdims=True) + 1e-8

    def norm(X):
        shape = X.shape
        flat = X.reshape(-1, FEAT_DIM)
        flat_norm = (flat - feat_mean) / feat_std
        return flat_norm.reshape(shape)

    X_train_norm = norm(X_train_sub)
    X_val_norm   = norm(X_val)
    X_test_norm  = norm(X_test)

    # 转换为 Tensor
    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train_sub, dtype=torch.float32, device=DEVICE)
    timer_train_t = torch.tensor(timer_train_sub, dtype=torch.float32, device=DEVICE)

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

    # 重新初始化模型（保证每个比例独立训练）
    model = ConditionalAffine(in_features=FEAT_DIM, hidden_dim=16).to(DEVICE)
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
                a, b = model(Xb)
                pred = a * tb + b
                loss = criterion(pred.view(-1), yb.view(-1))
                val_loss += loss.item() * len(Xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # 测试评估
    model.eval()
    with torch.no_grad():
        a_test, b_test = model(X_test_t)
        pred_test = (a_test * timer_test_t + b_test).cpu().numpy().squeeze(-1)

    y_test_np = y_test_t.cpu().numpy()
    flat_affine = pred_test.flatten()
    flat_true = y_test_np.flatten()

    # 计算原始 Timer 预测在测试集上的指标（用于对比）
    timer_test_np = timer_test_t.cpu().numpy().squeeze(-1)
    flat_timer = timer_test_np.flatten()
    timer_mae, timer_rmse, timer_nmae, timer_nrmse, timer_r2 = compute_metrics(flat_true, flat_timer, f"Timer原始({ratio_name})")
    affine_mae, affine_rmse, affine_nmae, affine_nrmse, affine_r2 = compute_metrics(flat_true, flat_affine, f"条件Affine({ratio_name})")

    return {
        'ratio': ratio_name,
        'timer_mae': timer_mae, 'timer_rmse': timer_rmse, 'timer_r2': timer_r2,
        'affine_mae': affine_mae, 'affine_rmse': affine_rmse, 'affine_r2': affine_r2
    }
# ==================== 主程序（修改后） ====================
def main():
    print(f"单站 {STATION} | 条件 Affine 变换 | 窗口={WINDOW}")
    X_all, y_all = load_windows(STATION, WINDOW, STRIDE)
    n_samples = len(X_all)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    X_train_full, X_val, X_test = X_all[:train_end], X_all[train_end:val_end], X_all[val_end:]
    y_train_full, y_val, y_test = y_all[:train_end], y_all[train_end:val_end], y_all[val_end:]

    # 提取 Timer 预测值（特征第一列）
    timer_train_full = X_train_full[:, :, 0:1]
    timer_val = X_val[:, :, 0:1]
    timer_test = X_test[:, :, 0:1]

    full_train_size = len(X_train_full)
    ratios = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    # 固定索引打乱的随机种子（保证每个比例的子集可重复）
    shuffle_seed = 42
    rng = np.random.RandomState(shuffle_seed)
    indices = np.arange(full_train_size)
    rng.shuffle(indices)   # 打乱索引

    results = []
    for ratio in ratios:
        sub_size = int(full_train_size * ratio)
        if sub_size == 0:
            continue
        # 从打乱后的索引中取前 sub_size 个
        sub_indices = indices[:sub_size]
        X_train_sub = X_train_full[sub_indices]
        y_train_sub = y_train_full[sub_indices]
        timer_train_sub = timer_train_full[sub_indices]

        ratio_name = f"{int(ratio*100)}%"
        print(f"\n{'='*60}\n训练集比例: {ratio_name} (样本数 {sub_size}/{full_train_size})\n{'='*60}")
        # 每次训练前固定随机种子，保证模型初始化一致
        set_seed(SEED)
        metrics = train_and_evaluate(X_train_sub, y_train_sub, timer_train_sub,
                                     X_val, y_val, timer_val,
                                     X_test, y_test, timer_test,
                                     ratio_name)
        results.append(metrics)

    # 汇总结果表格
    print("\n\n" + "="*80)
    print("不同训练集比例下的测试集性能汇总 (训练集样本随机打乱抽取)")
    print("="*80)
    print(f"{'比例':<8} | {'Timer MAE':>10} | {'Affine MAE':>10} | {'Timer RMSE':>10} | {'Affine RMSE':>10} | {'Timer R2':>8} | {'Affine R2':>8}")
    print("-"*80)
    for r in results:
        print(f"{r['ratio']:<8} | {r['timer_mae']:10.4f} | {r['affine_mae']:10.4f} | {r['timer_rmse']:10.4f} | {r['affine_rmse']:10.4f} | {r['timer_r2']:8.4f} | {r['affine_r2']:8.4f}")

if __name__ == "__main__":
    main()
# # ==================== 主程序 ====================
# def main():
#     print(f"单站 {STATION} | 条件 Affine 变换 | 窗口={WINDOW}")
#     X_all, y_all = load_windows(STATION, WINDOW, STRIDE)
#     n_samples = len(X_all)
#     train_end = int(n_samples * 0.8)
#     val_end = int(n_samples * 0.9)
#     X_train_full, X_val, X_test = X_all[:train_end], X_all[train_end:val_end], X_all[val_end:]
#     y_train_full, y_val, y_test = y_all[:train_end], y_all[train_end:val_end], y_all[val_end:]

#     # 提取 Timer 预测值（特征第一列）
#     timer_train_full = X_train_full[:, :, 0:1]
#     timer_val = X_val[:, :, 0:1]
#     timer_test = X_test[:, :, 0:1]

#     # 训练集大小
#     full_train_size = len(X_train_full)
#     ratios = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

#     results = []
#     for ratio in ratios:
#         sub_size = int(full_train_size * ratio)
#         if sub_size == 0:
#             continue
#         # 顺序取前 sub_size 个样本（保持时序，可重复）
#         X_train_sub = X_train_full[:sub_size]
#         y_train_sub = y_train_full[:sub_size]
#         timer_train_sub = timer_train_full[:sub_size]

#         ratio_name = f"{int(ratio*100)}%"
#         print(f"\n{'='*60}\n训练集比例: {ratio_name} (样本数 {sub_size}/{full_train_size})\n{'='*60}")
#         # 每次训练前固定随机种子，保证模型初始化一致
#         set_seed(SEED)
#         metrics = train_and_evaluate(X_train_sub, y_train_sub, timer_train_sub,
#                                      X_val, y_val, timer_val,
#                                      X_test, y_test, timer_test,
#                                      ratio_name)
#         results.append(metrics)

#     # 汇总结果表格
#     print("\n\n" + "="*80)
#     print("不同训练集比例下的测试集性能汇总")
#     print("="*80)
#     print(f"{'比例':<8} | {'Timer MAE':>10} | {'Affine MAE':>10} | {'Timer RMSE':>10} | {'Affine RMSE':>10} | {'Timer R2':>8} | {'Affine R2':>8}")
#     print("-"*80)
#     for r in results:
#         print(f"{r['ratio']:<8} | {r['timer_mae']:10.4f} | {r['affine_mae']:10.4f} | {r['timer_rmse']:10.4f} | {r['affine_rmse']:10.4f} | {r['timer_r2']:8.4f} | {r['affine_r2']:8.4f}")

# if __name__ == "__main__":
#     main()