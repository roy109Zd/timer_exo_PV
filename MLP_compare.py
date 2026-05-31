import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import warnings
warnings.filterwarnings('ignore')

# ========== 配置 ==========
STATION = "station00"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 1
LR = 1e-3
PATIENCE = 10
PRED_LEN = 96
NWP_DIM = 7

CACHE_DIR = "/root/timer+exo/stride1"
CACHE_STRIDE1 = os.path.join(CACHE_DIR, f"{STATION}_stride1.pt")
CACHE_STRIDE96 = os.path.join(CACHE_DIR, f"{STATION}_stride96.pt")

# 训练集步长选择：1 或 96
TRAIN_STRIDE = 1   # 可选 1 或 96

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ========== 1. 加载缓存 ==========
def load_cache(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data['feats'], data['preds'], data['nwps'], data['labels'], data['power_range']

# ========== 2. 数据划分 ==========
def prepare_data(train_stride):
    # 加载训练集数据（根据选择的步长）
    if train_stride == 1:
        feats_train, preds_train, nwps_train, labels_train, _ = load_cache(CACHE_STRIDE1)
    else:
        feats_train, preds_train, nwps_train, labels_train, _ = load_cache(CACHE_STRIDE96)
    
    # 加载验证/测试集数据（固定 stride=96）
    feats_valtest, preds_valtest, nwps_valtest, labels_valtest, power_range = load_cache(CACHE_STRIDE96)
    
    # 训练集取前80%（时间顺序）
    n_train = feats_train.shape[0]
    train_end = int(n_train * 0.8)
    feats_train = feats_train[:train_end]
    preds_train = preds_train[:train_end]
    nwps_train = nwps_train[:train_end]
    labels_train = labels_train[:train_end]
    
    # 验证/测试集（stride=96）按时间顺序前50%验证，后50%测试
    n_valtest = feats_valtest.shape[0]
    val_end = int(n_valtest * 0.5)
    feats_val = feats_valtest[:val_end]
    preds_val = preds_valtest[:val_end]
    nwps_val = nwps_valtest[:val_end]
    labels_val = labels_valtest[:val_end]
    
    feats_test = feats_valtest[val_end:]
    preds_test = preds_valtest[val_end:]
    nwps_test = nwps_valtest[val_end:]
    labels_test = labels_valtest[val_end:]
    
    print(f"训练步长={train_stride}: 训练集 {len(feats_train)}, 验证集 {len(feats_val)}, 测试集 {len(feats_test)}")
    return (feats_train, preds_train, nwps_train, labels_train,
            feats_val, preds_val, nwps_val, labels_val,
            feats_test, preds_test, nwps_test, labels_test,
            power_range)

# ========== 3. 逐点展平辅助 ==========
def flatten_sequence(feat, pred, nwp, label):
    T = PRED_LEN
    label_flat = label.reshape(-1)                     # [N*T]
    nwp_flat = nwp.reshape(-1, nwp.shape[-1])          # [N*T, 7]
    feat_flat = None
    if feat is not None:
        feat_flat = feat.unsqueeze(1).expand(-1, T, -1).reshape(-1, feat.shape[-1])
    pred_flat = None
    if pred is not None:
        pred_flat = pred.reshape(-1, 1)
    return feat_flat, pred_flat, nwp_flat, label_flat

# ========== 4. 定义所有模型 ==========
class MLP_OnlyPredNWP(nn.Module):
    def __init__(self, pred_len=96, nwp_dim=7, hidden=512):
        super().__init__()
        input_dim = pred_len + pred_len * nwp_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Linear(256, pred_len)
        )
    def forward(self, pred, nwp):
        nwp_flat = nwp.view(pred.shape[0], -1)
        x = torch.cat([pred, nwp_flat], dim=-1)
        return self.net(x)

class MLP_OnlyFeatNWP(nn.Module):
    def __init__(self, feat_dim=1024, pred_len=96, nwp_dim=7, hidden=512):
        super().__init__()
        input_dim = feat_dim + pred_len * nwp_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Linear(256, pred_len)
        )
    def forward(self, feat, nwp):
        nwp_flat = nwp.view(feat.shape[0], -1)
        x = torch.cat([feat, nwp_flat], dim=-1)
        return self.net(x)

class MLP_LatentPredNWP(nn.Module):
    def __init__(self, feat_dim=1024, pred_len=96, nwp_dim=7, hidden_dim=256):
        super().__init__()
        self.pred_len = pred_len
        self.nwp_dim = nwp_dim
        self.feat_transform = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, pred_len * nwp_dim)
        )
        self.global_fc = nn.Sequential(
            nn.Linear(pred_len * (2 * nwp_dim + 1), 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, pred_len)
        )
    def forward(self, feat, timer_pred, nwp):
        B = feat.shape[0]
        latent = self.feat_transform(feat).view(B, self.pred_len, self.nwp_dim)
        timer_pred_exp = timer_pred.unsqueeze(-1)
        combined = torch.cat([latent, timer_pred_exp, nwp], dim=-1).view(B, -1)
        return self.global_fc(combined)

class Pointwise_Feat(nn.Module):
    def __init__(self, feat_dim=1024, nwp_dim=7, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + nwp_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, feat, nwp_t):
        x = torch.cat([feat, nwp_t], dim=-1)
        return self.net(x).squeeze(-1)

class Pointwise_PredNWP(nn.Module):
    def __init__(self, nwp_dim=7, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + nwp_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, pred_t, nwp_t):
        x = torch.cat([pred_t, nwp_t], dim=-1)
        return self.net(x).squeeze(-1)

class Pointwise_1024_1Step(nn.Module):
    def __init__(self, feat_dim=1024, nwp_dim=7, hidden=256):
        super().__init__()
        self.feat_map = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(1 + nwp_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1)
        )
    def forward(self, feat, nwp_t):
        feat_out = self.feat_map(feat)          # [B, 1]
        x = torch.cat([feat_out, nwp_t], dim=-1)
        return self.fusion(x).squeeze(-1)

# ========== 5. 训练与评估函数 ==========
def train_model(model, train_loader, val_loader, test_loader, epochs, lr, device, power_range, patience=10):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_state = None
    counter = 0
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if len(batch) == 3:  # (x1, x2, label)
                x1, x2, label = batch
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                out = model(x1, x2)
            elif len(batch) == 4: # (x1, x2, x3, label)
                x1, x2, x3, label = batch
                x1, x2, x3, label = x1.to(device), x2.to(device), x3.to(device), label.to(device)
                out = model(x1, x2, x3)
            else:
                raise ValueError("batch length not supported")
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x1, x2, label = batch
                    x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                    out = model(x1, x2)
                else:
                    x1, x2, x3, label = batch
                    x1, x2, x3, label = x1.to(device), x2.to(device), x3.to(device), label.to(device)
                    out = model(x1, x2, x3)
                loss = criterion(out, label)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"早停触发 (epoch {epoch})")
                break
    if best_state:
        model.load_state_dict(best_state)
    
    # 测试
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x1, x2, label = batch
                x1, x2 = x1.to(device), x2.to(device)
                out = model(x1, x2)
            else:
                x1, x2, x3, label = batch
                x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
                out = model(x1, x2, x3)
            all_pred.append(out.cpu().numpy())
            all_true.append(label.numpy())
    all_pred = np.concatenate(all_pred, axis=0).flatten()
    all_true = np.concatenate(all_true, axis=0).flatten()
    
    # 计算指标，C = 训练集真实值极差（power_range 是原始数据的极差，这里用传入的 power_range）
    mae = mean_absolute_error(all_true, all_pred)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    r2 = r2_score(all_true, all_pred)
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    return mae, rmse, nmae, nrmse, r2

# ========== 6. 主程序 ==========
def main():
    set_seed(42)
    # 准备数据（训练步长由 TRAIN_STRIDE 决定）
    (feats_train, preds_train, nwps_train, labels_train,
     feats_val, preds_val, nwps_val, labels_val,
     feats_test, preds_test, nwps_test, labels_test,
     power_range) = prepare_data(TRAIN_STRIDE)
    
    # 计算训练集真实值的极差（作为归一化分母）
    train_true = labels_train.numpy().flatten()
    C = train_true.max() - train_true.min()
    print(f"训练集真实值极差 C = {C:.4f}")
    
    # 定义实验配置
    experiments = [
        {
            "name": "MLP_OnlyPredNWP",
            "model": MLP_OnlyPredNWP(pred_len=PRED_LEN, nwp_dim=NWP_DIM),
            "train_data": (preds_train, nwps_train, labels_train),
            "val_data": (preds_val, nwps_val, labels_val),
            "test_data": (preds_test, nwps_test, labels_test),
            "batch_len": 3
        },
        {
            "name": "MLP_OnlyFeatNWP",
            "model": MLP_OnlyFeatNWP(feat_dim=1024, pred_len=PRED_LEN, nwp_dim=NWP_DIM),
            "train_data": (feats_train, nwps_train, labels_train),
            "val_data": (feats_val, nwps_val, labels_val),
            "test_data": (feats_test, nwps_test, labels_test),
            "batch_len": 3
        },
        {
            "name": "MLP_LatentPredNWP",
            "model": MLP_LatentPredNWP(feat_dim=1024, pred_len=PRED_LEN, nwp_dim=NWP_DIM),
            "train_data": (feats_train, preds_train, nwps_train, labels_train),
            "val_data": (feats_val, preds_val, nwps_val, labels_val),
            "test_data": (feats_test, preds_test, nwps_test, labels_test),
            "batch_len": 4
        },
        {
            "name": "Pointwise_Feat",
            "model": Pointwise_Feat(feat_dim=1024, nwp_dim=NWP_DIM),
            "train_data": flatten_sequence(feats_train, None, nwps_train, labels_train),
            "val_data": flatten_sequence(feats_val, None, nwps_val, labels_val),
            "test_data": flatten_sequence(feats_test, None, nwps_test, labels_test),
            "batch_len": 3
        },
        {
            "name": "Pointwise_PredNWP",
            "model": Pointwise_PredNWP(nwp_dim=NWP_DIM),
            "train_data": flatten_sequence(None, preds_train, nwps_train, labels_train),
            "val_data": flatten_sequence(None, preds_val, nwps_val, labels_val),
            "test_data": flatten_sequence(None, preds_test, nwps_test, labels_test),
            "batch_len": 3
        },
        {
            "name": "Pointwise_1024_1Step",
            "model": Pointwise_1024_1Step(feat_dim=1024, nwp_dim=NWP_DIM),
            "train_data": flatten_sequence(feats_train, None, nwps_train, labels_train),
            "val_data": flatten_sequence(feats_val, None, nwps_val, labels_val),
            "test_data": flatten_sequence(feats_test, None, nwps_test, labels_test),
            "batch_len": 3
        }
    ]
    
    results = []
    for exp in experiments:
        print(f"\n--- 训练模型: {exp['name']} ---")
        # 构建 DataLoader
        if exp['batch_len'] == 3:
            train_ds = TensorDataset(exp['train_data'][0], exp['train_data'][1], exp['train_data'][2])
            val_ds = TensorDataset(exp['val_data'][0], exp['val_data'][1], exp['val_data'][2])
            test_ds = TensorDataset(exp['test_data'][0], exp['test_data'][1], exp['test_data'][2])
        else:
            train_ds = TensorDataset(exp['train_data'][0], exp['train_data'][1], exp['train_data'][2], exp['train_data'][3])
            val_ds = TensorDataset(exp['val_data'][0], exp['val_data'][1], exp['val_data'][2], exp['val_data'][3])
            test_ds = TensorDataset(exp['test_data'][0], exp['test_data'][1], exp['test_data'][2], exp['test_data'][3])
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        mae, rmse, nmae, nrmse, r2 = train_model(
            exp['model'], train_loader, val_loader, test_loader,
            EPOCHS, LR, DEVICE, C, PATIENCE
        )
        print(f"结果: MAE={mae:.4f}, RMSE={rmse:.4f}, NMAE={nmae:.4f}, NRMSE={nrmse:.4f}, R2={r2:.4f}")
        results.append({
            'model': exp['name'],
            'train_stride': TRAIN_STRIDE,
            'MAE': mae,
            'RMSE': rmse,
            'NMAE': nmae,
            'NRMSE': nrmse,
            'R2': r2,
            'train_samples': len(train_ds),
            'val_samples': len(val_ds),
            'test_samples': len(test_ds)
        })
    
    # 保存结果
    df = pd.DataFrame(results)
    csv_path = os.path.join(CACHE_DIR, f"comparison_train_stride_{TRAIN_STRIDE}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n结果保存至: {csv_path}")

if __name__ == "__main__":
    main()