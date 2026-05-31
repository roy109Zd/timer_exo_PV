import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/root/timer/timer84m')
from modeling_timer import TimerForPrediction

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ========== 配置 ==========
DATA_DIR = "/root/timer/甘肃光伏"
MODEL_PATH = "/root/timer/timer84m"
STATIONS = [f"station{i:02d}" for i in range(10)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-3
PATIENCE = 10
LOOKBACK = 672
PRED_LEN = 96
NWP_DIM = 7
TRAIN_STRIDE = 1
VAL_TEST_STRIDE = 96
SAVE_ROOT = "/root/timer+exo/stride1"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ========== 1. 提取特征（给定步长，返回所有样本的起始索引和特征） ==========
def extract_features_with_start_idx(station, stride):
    """返回 (start_indices, feats, preds, nwps, labels, power_range)"""
    cache_file = os.path.join(SAVE_ROOT, f"{station}_stride{stride}.pt")
    if os.path.exists(cache_file):
        print(f"加载缓存: {cache_file}")
        data = torch.load(cache_file, weights_only=False)
        return (data['start_indices'], data['feats'], data['preds'], 
                data['nwps'], data['labels'], data['power_range'])
    
    print(f"提取 {station} 特征，步长={stride}...")
    model = TimerForPrediction.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    df = pd.read_csv(os.path.join(DATA_DIR, f"{station}.csv"))
    power = df['power'].values.astype(np.float32)
    power_range = power.max() - power.min()
    nwp_cols = ['nwp_globalirrad','nwp_directirrad','nwp_temperature','nwp_humidity',
                'nwp_windspeed','nwp_winddirection','nwp_pressure']
    nwp_data = df[nwp_cols].values.astype(np.float32)
    scaler = StandardScaler()
    nwp_data = scaler.fit_transform(nwp_data)
    
    start_indices = []
    feats, preds, nwps, labels = [], [], [], []
    total_len = len(power)
    for start in range(0, total_len - LOOKBACK - PRED_LEN + 1, stride):
        if start % 5000 == 0:
            print(f"进度: {start}/{total_len - LOOKBACK - PRED_LEN + 1}")
        input_seq = power[start:start+LOOKBACK]
        input_tensor = torch.tensor(input_seq).float().unsqueeze(0).to(DEVICE)
        nwp_future = torch.tensor(nwp_data[start+LOOKBACK:start+LOOKBACK+PRED_LEN]).float().unsqueeze(0)
        label = torch.tensor(power[start+LOOKBACK:start+LOOKBACK+PRED_LEN]).float().unsqueeze(0)
        with torch.no_grad():
            mean, std = input_tensor.mean(dim=-1, keepdim=True), input_tensor.std(dim=-1, keepdim=True)
            norm_input = (input_tensor - mean) / std
            outputs = model.model(input_ids=norm_input, return_dict=True)
            feat = outputs.last_hidden_state[:, -1, :]
            pred_norm = model.lm_heads[0](feat)
            pred_raw = pred_norm * std + mean
        start_indices.append(start)
        feats.append(feat.cpu())
        preds.append(pred_raw.cpu())
        nwps.append(nwp_future.cpu())
        labels.append(label.cpu())
    feats = torch.cat(feats, dim=0)
    preds = torch.cat(preds, dim=0)
    nwps = torch.cat(nwps, dim=0)
    labels = torch.cat(labels, dim=0)
    data = {
        'start_indices': start_indices,
        'feats': feats,
        'preds': preds,
        'nwps': nwps,
        'labels': labels,
        'power_range': power_range
    }
    torch.save(data, cache_file)
    print(f"保存缓存: {cache_file}, 样本数: {len(start_indices)}")
    return start_indices, feats, preds, nwps, labels, power_range

# ========== 2. 按时间顺序划分8:1:1，训练用步长1，验证/测试用步长96 ==========
def prepare_station_data(station):
    # 获取步长1的所有样本（用于确定时间顺序）
    start_idx_1, feats_1, preds_1, nwps_1, labels_1, power_range = extract_features_with_start_idx(station, TRAIN_STRIDE)
    # 获取步长96的所有样本及其起始索引
    start_idx_96, feats_96, preds_96, nwps_96, labels_96, _ = extract_features_with_start_idx(station, VAL_TEST_STRIDE)
    
    # 步长1的样本按起始索引排序（本身就是递增的），取前80%作为训练集
    n_train_1 = int(len(start_idx_1) * 0.8)
    train_start_set = set(start_idx_1[:n_train_1])   # 训练集覆盖的时间起始点
    
    # 验证集和测试集从步长96中选取，其起始索引分别属于接下来的10%和最后10%
    # 需要知道时间边界：训练集最大起始索引，然后按比例划分剩余的时间范围
    # 由于步长1的样本索引基本连续，我们可以用训练集的最大起始索引作为分隔点
    max_train_start = start_idx_1[n_train_1 - 1] if n_train_1 > 0 else -1
    # 剩余时间范围的总长度（按步长1样本数）为 n_total_1 - n_train_1
    total_rem = len(start_idx_1) - n_train_1
    val_size_1 = int(total_rem * 0.5)   # 剩下的一半作为验证，一半作为测试（因为8:1:1，剩余20%中一半验证一半测试）
    # 实际上 8:1:1 意味着训练80%，验证10%，测试10%。所以剩余20%中验证和测试各占一半。
    val_end_1 = n_train_1 + val_size_1
    val_start_set = set(start_idx_1[n_train_1:val_end_1])
    test_start_set = set(start_idx_1[val_end_1:])
    
    # 从步长96中筛选
    val_indices = [i for i, start in enumerate(start_idx_96) if start in val_start_set]
    test_indices = [i for i, start in enumerate(start_idx_96) if start in test_start_set]
    
    # 训练集直接使用步长1的前80%样本
    train_feat = feats_1[:n_train_1]
    train_pred = preds_1[:n_train_1]
    train_nwp = nwps_1[:n_train_1]
    train_label = labels_1[:n_train_1]
    
    # 验证集
    val_feat = feats_96[val_indices]
    val_pred = preds_96[val_indices]
    val_nwp = nwps_96[val_indices]
    val_label = labels_96[val_indices]
    
    # 测试集
    test_feat = feats_96[test_indices]
    test_pred = preds_96[test_indices]
    test_nwp = nwps_96[test_indices]
    test_label = labels_96[test_indices]
    
    print(f"{station}: 训练集 {len(train_feat)}, 验证集 {len(val_feat)}, 测试集 {len(test_feat)}")
    return (train_feat, train_pred, train_nwp, train_label,
            val_feat, val_pred, val_nwp, val_label,
            test_feat, test_pred, test_nwp, test_label,
            power_range)

# ========== 3. 保存每个站点的三个数据集 ==========
def save_station_features(station):
    save_dir = os.path.join(SAVE_ROOT, station)
    os.makedirs(save_dir, exist_ok=True)
    train_file = os.path.join(save_dir, "train.pt")
    val_file = os.path.join(save_dir, "val.pt")
    test_file = os.path.join(save_dir, "test.pt")
    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        print(f"{station} 特征已存在，跳过")
        return
    data = prepare_station_data(station)
    (tr_f, tr_p, tr_n, tr_l,
     va_f, va_p, va_n, va_l,
     te_f, te_p, te_n, te_l, pr) = data
    torch.save({'feat': tr_f, 'pred': tr_p, 'nwp': tr_n, 'label': tr_l, 'power_range': pr}, train_file)
    torch.save({'feat': va_f, 'pred': va_p, 'nwp': va_n, 'label': va_l, 'power_range': pr}, val_file)
    torch.save({'feat': te_f, 'pred': te_p, 'nwp': te_n, 'label': te_l, 'power_range': pr}, test_file)
    print(f"保存 {station} 完成")

# ========== 4. 模型定义 ==========
class TimerHybridModel(nn.Module):
    def __init__(self, feat_dim=1024, pred_len=96, nwp_dim=7, hidden_dim=256):
        super().__init__()
        self.pred_len = pred_len
        self.nwp_dim = nwp_dim
        self.feat_transform = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, pred_len * nwp_dim)
        )
        self.global_fc = nn.Sequential(
            nn.Linear(pred_len * (2 * nwp_dim + 1), 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, pred_len)
        )
    def forward(self, feat, timer_pred, nwp):
        B = feat.shape[0]
        latent = self.feat_transform(feat)
        latent = latent.view(B, self.pred_len, self.nwp_dim)
        timer_pred_exp = timer_pred.unsqueeze(-1)
        combined = torch.cat([latent, timer_pred_exp, nwp], dim=-1)
        combined_flat = combined.view(B, -1)
        out = self.global_fc(combined_flat)
        return out

# ========== 5. 评估指标 ==========
def compute_metrics(y_true, y_pred, power_range):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    nmae = mae / power_range if power_range > 0 else np.nan
    nrmse = rmse / power_range if power_range > 0 else np.nan
    return mae, rmse, nmae, nrmse, r2

# ========== 6. 训练函数（早停） ==========
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
        for feat, timer_pred, nwp, label in train_loader:
            feat = feat.to(device)
            timer_pred = timer_pred.to(device)
            nwp = nwp.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out = model(feat, timer_pred, nwp)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feat, timer_pred, nwp, label in val_loader:
                feat = feat.to(device)
                timer_pred = timer_pred.to(device)
                nwp = nwp.to(device)
                label = label.to(device)
                out = model(feat, timer_pred, nwp)
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
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for feat, timer_pred, nwp, label in test_loader:
            feat = feat.to(device)
            timer_pred = timer_pred.to(device)
            nwp = nwp.to(device)
            pred = model(feat, timer_pred, nwp).cpu().numpy()
            all_pred.append(pred)
            all_true.append(label.numpy())
    all_pred = np.concatenate(all_pred, axis=0).flatten()
    all_true = np.concatenate(all_true, axis=0).flatten()
    return compute_metrics(all_true, all_pred, power_range)

# ========== 7. 主程序 ==========
def main():
    # 1. 提取所有站点的特征（如果未缓存则提取）
    for st in STATIONS:
        try:
            save_station_features(st)
        except Exception as e:
            print(f"站点 {st} 失败: {e}")
    
    # 2. 训练模型（示例：训练 station00，可循环所有站）
    station = "station00"
    print(f"\n开始训练 {station}")
    data_dir = os.path.join(SAVE_ROOT, station)
    train_data = torch.load(os.path.join(data_dir, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(data_dir, "val.pt"), weights_only=False)
    test_data = torch.load(os.path.join(data_dir, "test.pt"), weights_only=False)
    
    power_range = train_data['power_range']
    train_loader = DataLoader(TensorDataset(train_data['feat'], train_data['pred'], train_data['nwp'], train_data['label']),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data['feat'], val_data['pred'], val_data['nwp'], val_data['label']),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data['feat'], test_data['pred'], test_data['nwp'], test_data['label']),
                             batch_size=BATCH_SIZE, shuffle=False)
    
    model = TimerHybridModel(feat_dim=1024, pred_len=PRED_LEN, nwp_dim=NWP_DIM, hidden_dim=256)
    mae, rmse, nmae, nrmse, r2 = train_model(model, train_loader, val_loader, test_loader,
                                             EPOCHS, LR, DEVICE, power_range, PATIENCE)
    print(f"\n{station} 测试结果: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

if __name__ == "__main__":
    main()