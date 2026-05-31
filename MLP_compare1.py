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
EPOCHS = 200
LR = 1e-3
PATIENCE = 10
PRED_LEN = 96
NWP_DIM = 7

CACHE_DIR = "/root/timer+exo/stride1"
CACHE_STRIDE1 = os.path.join(CACHE_DIR, f"{STATION}_stride1.pt")
CACHE_STRIDE96 = os.path.join(CACHE_DIR, f"{STATION}_stride96.pt")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ========== 1. 加载缓存 ==========
def load_cache(path):
    print(f"加载缓存: {path}")
    data = torch.load(path, map_location='cpu', weights_only=False)
    return (data['start_indices'], data['feats'], data['preds'],
            data['nwps'], data['labels'], data['power_range'])

# ========== 2. 数据准备 ==========
def prepare_data(train_stride):
    # 训练集数据
    if train_stride == 1:
        _, train_feat, train_pred, train_nwp, train_label, _ = load_cache(CACHE_STRIDE1)
    else:
        _, train_feat, train_pred, train_nwp, train_label, _ = load_cache(CACHE_STRIDE96)
    # 验证/测试集固定使用 stride96
    _, valtest_feat, valtest_pred, valtest_nwp, valtest_label, power_range = load_cache(CACHE_STRIDE96)
    
    # 划分验证/测试集（前50%验证，后50%测试）
    n_valtest = len(valtest_label)
    split = n_valtest // 2
    val_feat = valtest_feat[:split]
    val_pred = valtest_pred[:split]
    val_nwp = valtest_nwp[:split]
    val_label = valtest_label[:split]
    test_feat = valtest_feat[split:]
    test_pred = valtest_pred[split:]
    test_nwp = valtest_nwp[split:]
    test_label = valtest_label[split:]
    
    # 训练集取全部（已经按时间顺序）
    print(f"训练步长={train_stride}: 训练集 {len(train_label)} 样本, 验证集 {len(val_label)} 样本, 测试集 {len(test_label)} 样本")
    return (train_feat, train_pred, train_nwp, train_label,
            val_feat, val_pred, val_nwp, val_label,
            test_feat, test_pred, test_nwp, test_label,
            power_range)

# ========== 3. 逐点展平 ==========
def flatten_sequence(feat, pred, nwp, label):
    T = PRED_LEN
    label_flat = label.reshape(-1)                     # [N*T]
    nwp_flat = nwp.reshape(-1, nwp.shape[-1])          # [N*T, 7]
    feat_flat = None
    if feat is not None:
        # feat: [N, 1024] -> 复制 T 次 -> [N*T, 1024]
        feat_flat = feat.unsqueeze(1).expand(-1, T, -1).reshape(-1, feat.shape[-1])
    pred_flat = None
    if pred is not None:
        # pred: [N, T] -> [N*T, 1]
        pred_flat = pred.reshape(-1, 1)
    return feat_flat, pred_flat, nwp_flat, label_flat

# ========== 4. 模型定义 ==========
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
        feat_out = self.feat_map(feat)
        x = torch.cat([feat_out, nwp_t], dim=-1)
        return self.fusion(x).squeeze(-1)

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

# ========== 5. 评估指标 ==========
def compute_metrics(y_true, y_pred, C):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    nmae = mae / C if C > 0 else np.nan
    nrmse = rmse / C if C > 0 else np.nan
    return mae, rmse, nmae, nrmse, r2

# ========== 6. 训练函数 ==========
def train_model(model, train_loader, val_loader, test_loader, epochs, lr, device, C, patience=10):
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
            inputs = [x.to(device) for x in batch[:-1]]
            labels = batch[-1].to(device)
            optimizer.zero_grad()
            out = model(*inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = [x.to(device) for x in batch[:-1]]
                labels = batch[-1].to(device)
                out = model(*inputs)
                loss = criterion(out, labels)
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
            inputs = [x.to(device) for x in batch[:-1]]
            labels = batch[-1].to(device)
            pred = model(*inputs).cpu().numpy()
            all_pred.append(pred)
            all_true.append(labels.cpu().numpy())
    all_pred = np.concatenate(all_pred, axis=0).flatten()
    all_true = np.concatenate(all_true, axis=0).flatten()
    return compute_metrics(all_true, all_pred, C)

# ========== 7. 主程序：对 train_stride in [1, 96] 分别实验 ==========
def main():
    set_seed(42)
    results = []
    for train_stride in [1, 96]:
        print("\n" + "="*70)
        print(f"训练集步长 = {train_stride} (验证/测试步长=96)")
        print("="*70)
        train_feat, train_pred, train_nwp, train_label, \
        val_feat, val_pred, val_nwp, val_label, \
        test_feat, test_pred, test_nwp, test_label, \
        power_range = prepare_data(train_stride)
        C = power_range

        # ===== 展平逐点模型所需数据 =====
        # 特征 + NWP 展平 (用于 Pointwise_Feat 和 Pointwise_1024_1Step)
        train_feat_flat, _, train_nwp_flat, train_label_flat = flatten_sequence(train_feat, None, train_nwp, train_label)
        val_feat_flat, _, val_nwp_flat, val_label_flat = flatten_sequence(val_feat, None, val_nwp, val_label)
        test_feat_flat, _, test_nwp_flat, test_label_flat = flatten_sequence(test_feat, None, test_nwp, test_label)

        # Timer预测值 + NWP 展平 (用于 Pointwise_PredNWP)
        _, train_pred_flat, train_nwp_flat_pred, train_label_flat_pred = flatten_sequence(None, train_pred, train_nwp, train_label)
        _, val_pred_flat, val_nwp_flat_pred, val_label_flat_pred = flatten_sequence(None, val_pred, val_nwp, val_label)
        _, test_pred_flat, test_nwp_flat_pred, test_label_flat_pred = flatten_sequence(None, test_pred, test_nwp, test_label)

        # ===== 构建实验列表，所有数据均不为 None =====
        experiments = [
            ("MLP_OnlyPredNWP", MLP_OnlyPredNWP,
             (train_pred, train_nwp, train_label),
             (val_pred, val_nwp, val_label),
             (test_pred, test_nwp, test_label)),
            ("MLP_OnlyFeatNWP", MLP_OnlyFeatNWP,
             (train_feat, train_nwp, train_label),
             (val_feat, val_nwp, val_label),
             (test_feat, test_nwp, test_label)),
            ("MLP_LatentPredNWP", MLP_LatentPredNWP,
             (train_feat, train_pred, train_nwp, train_label),
             (val_feat, val_pred, val_nwp, val_label),
             (test_feat, test_pred, test_nwp, test_label)),
            ("Pointwise_Feat", Pointwise_Feat,
             (train_feat_flat, train_nwp_flat, train_label_flat),
             (val_feat_flat, val_nwp_flat, val_label_flat),
             (test_feat_flat, test_nwp_flat, test_label_flat)),
            ("Pointwise_1024_1Step", Pointwise_1024_1Step,
             (train_feat_flat, train_nwp_flat, train_label_flat),
             (val_feat_flat, val_nwp_flat, val_label_flat),
             (test_feat_flat, test_nwp_flat, test_label_flat)),
            ("Pointwise_PredNWP", Pointwise_PredNWP,
             (train_pred_flat, train_nwp_flat_pred, train_label_flat_pred),
             (val_pred_flat, val_nwp_flat_pred, val_label_flat_pred),
             (test_pred_flat, test_nwp_flat_pred, test_label_flat_pred)),
        ]

        for name, model_class, train_data, val_data, test_data in experiments:
            print(f"\n--- 训练模型: {name} ---")
            # 构建 DataLoader
            train_inputs = [x for x in train_data[:-1] if x is not None]
            train_labels = train_data[-1]
            train_ds = TensorDataset(*train_inputs, train_labels)
            val_inputs = [x for x in val_data[:-1] if x is not None]
            val_labels = val_data[-1]
            val_ds = TensorDataset(*val_inputs, val_labels)
            test_inputs = [x for x in test_data[:-1] if x is not None]
            test_labels = test_data[-1]
            test_ds = TensorDataset(*test_inputs, test_labels)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

            # 实例化模型
            if name == "MLP_OnlyPredNWP":
                model = model_class(pred_len=PRED_LEN, nwp_dim=NWP_DIM)
            elif name == "MLP_OnlyFeatNWP":
                model = model_class(feat_dim=1024, pred_len=PRED_LEN, nwp_dim=NWP_DIM)
            elif name == "MLP_LatentPredNWP":
                model = model_class(feat_dim=1024, pred_len=PRED_LEN, nwp_dim=NWP_DIM)
            elif name in ["Pointwise_Feat", "Pointwise_1024_1Step"]:
                model = model_class(feat_dim=1024, nwp_dim=NWP_DIM)
            elif name == "Pointwise_PredNWP":
                model = model_class(nwp_dim=NWP_DIM)
            else:
                raise ValueError(name)

            mae, rmse, nmae, nrmse, r2 = train_model(
                model, train_loader, val_loader, test_loader,
                EPOCHS, LR, DEVICE, C, PATIENCE
            )
            print(f"结果: MAE={mae:.4f}, RMSE={rmse:.4f}, NMAE={nmae:.4f}, NRMSE={nrmse:.4f}, R2={r2:.4f}")
            results.append({
                'train_stride': train_stride,
                'model': name,
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
    csv_path = os.path.join(CACHE_DIR, "all_comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n所有结果已保存至: {csv_path}")
    print(df.to_string())

if __name__ == "__main__":
    main()