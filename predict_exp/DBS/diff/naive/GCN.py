import os
import re
import json
import time
import logging

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, average_precision_score
)
from sklearn.metrics import roc_curve

# ======================
# 日志配置
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("treatment_response_prediction.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================
# 全局配置
# ======================
BASE_DIR = "/ailab/group/medai-share/syDu/ruijin/DBS/AAL_VTB"
UPDRS_CSV_PATH = "/ailab/group/medai-share/syDu/ruijin/DBS/DBS_UPDRS.csv"
RANDOM_SEED = 777
N_SPLITS = 5

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using device: {device}")

# 临床子项列名（完全匹配CSV表头，注意中文冒号）
CLINICAL_COLS = [
    '言语', '面部表情', '强直_脖子', '强直_上肢：右', '强直_上肢：左', '强直_下肢：右', '强直_下肢：左',
    '手指拍打_右', '手指拍打_左', '手掌运动_右', '手掌运动_左', '前臂回旋_右', '前臂回旋_左',
    '脚趾拍地_右', '脚趾拍地_左', '脚灵敏度_右', '脚灵敏度_左', '起立', '步态', '步态冻结_步态冻结的评估',
    '姿势平稳度', '姿势', '全身自发性的动作评估_身体动作迟缓', '姿态性震颤_右', '姿态性震颤_左',
    '动作性震颤_右', '动作性震颤_左', '静止型震颤_上肢：右', '静止型震颤_上肢：左', '静止型震颤_下肢：右',
    '静止型震颤_下肢：左', '静止型震颤_嘴唇/下巴', '静止型震颤_持续性'
]

# ======================
# 超参数配置 (集中管理) - 只保留GCN相关
# ======================
HYPERPARAMS = {
    'batch_size': 8,
    'learning_rate': 0.0005,
    'num_epochs': 100,
    'hidden_dim': 256,
    'dropout': 0.3,
    'feature_type': 'ALL',           # 'FC' or 'ALL'（GCN不支持UPDRS-only）
    'improvement_threshold': 0.25,   # 改善率阈值
    'model_type': 'GCN',             # 固定GCN
    'gcn_params': {
        'gcn_hidden': 128,
        'gcn_layers': 2
    }
}

logger.info(f"Hyperparameters: {json.dumps(HYPERPARAMS, indent=2, ensure_ascii=False)}")

# ======================
# 辅助函数：构建FC文件索引
# ======================
def build_fc_index(base_dir):
    """
    构建FC文件索引字典
    Returns:
        dict: {lowercase_id: fc_file_path}
    """
    fc_index = {}
    folder_pattern = re.compile(r'^([a-zA-Z0-9]+)_fmri')

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        match = folder_pattern.match(folder_name)
        if not match:
            logger.warning(f"Skipping folder with invalid name format: {folder_name}")
            continue

        pid = match.group(1).lower()
        patient_subfolder = os.path.join(folder_path, 'fine_tune', folder_name)
        fc_file = os.path.join(patient_subfolder, f"patient_{folder_name}_empirical_FC.npy")

        if not os.path.exists(fc_file):
            logger.warning(f"FC file not found: {fc_file}")
            continue

        if pid not in fc_index:
            fc_index[pid] = fc_file
        else:
            logger.warning(f"Duplicate patient ID found: {pid}, skipping additional entry")

    logger.info(f"Built FC index with {len(fc_index)} patients")
    return fc_index

# ======================
# 解析UPDRS数据 - 使用FC索引
# ======================
def parse_updrs_data(csv_path, fc_index):
    """
    解析UPDRS CSV数据，匹配FC文件
    """
    df = pd.read_csv(csv_path)

    required_cols = ['ID', '评估时间', '手术情况', 'UPDRS-III改善率'] + CLINICAL_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in CSV: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.sort_values(by=['ID', '评估时间'])

    samples = []
    improvement_threshold = HYPERPARAMS['improvement_threshold']

    for pid, group in df.groupby('ID'):
        pid_lower = str(pid).strip().lower()

        if pid_lower not in fc_index:
            logger.warning(f"No FC records found for patient {pid}")
            continue

        if len(group) < 2:
            logger.warning(f"Patient {pid} has only {len(group)} records, skipping")
            continue

        off_row = group.iloc[0]
        if off_row['手术情况'] != 'DBS off':
            logger.warning(f"First record for {pid} is not 'DBS off', skipping")
            continue

        if pd.isna(off_row['UPDRS-III改善率']):
            logger.warning(f"Missing improvement rate for {pid}, skipping")
            continue

        label = 1 if float(off_row['UPDRS-III改善率']) >= improvement_threshold else 0

        clinical_features = []
        for col in CLINICAL_COLS:
            val = off_row[col]
            if pd.isna(val):
                logger.warning(f"Missing clinical feature {col} for {pid}, skipping")
                clinical_features = None
                break
            try:
                clinical_features.append(float(val))
            except (ValueError, TypeError):
                logger.warning(f"Non-numeric value in {col} for {pid}: {val}, skipping")
                clinical_features = None
                break

        if clinical_features is None:
            continue

        fc_file = fc_index[pid_lower]

        samples.append({
            'id': pid,
            'fc_file': fc_file,
            'clinical_features': np.array(clinical_features, dtype=np.float32),
            'label': int(label),
            'improvement_rate': float(off_row['UPDRS-III改善率']),
            'updrs_total_off': off_row.get('UPDRS总分', None)
        })

    logger.info(f"Found {len(samples)} valid samples")
    return samples

# ======================
# 数据准备 - 用于K折划分（也可用于日志/检查）
# ======================
def prepare_data(samples):
    """
    准备特征和标签 (用于K折划分)
    - FC: 使用 166x166 的行均值 -> 166维
    - UPDRS: clinical 维
    - ALL: 拼接
    """
    X_fc_list = []
    X_updrs_list = []
    y_list = []
    meta_list = []

    feature_type = HYPERPARAMS['feature_type'].upper()

    for sample in samples:
        try:
            fc_matrix = np.load(sample['fc_file'])
            if fc_matrix.shape != (166, 166):
                logger.warning(f"Unexpected FC matrix shape {fc_matrix.shape} for {sample['id']}, skipping")
                continue
            fc_avg = np.mean(fc_matrix, axis=1).astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading FC file for {sample['id']}: {str(e)}")
            continue

        clinical_vec = sample['clinical_features']

        if feature_type == 'FC':
            _ = fc_avg
        elif feature_type == 'UPDRS':
            _ = clinical_vec
        elif feature_type == 'ALL':
            _ = np.concatenate([fc_avg, clinical_vec])
        else:
            raise ValueError(f"Invalid feature_type: {feature_type}. Choose from 'FC', 'UPDRS', 'ALL'")

        X_fc_list.append(fc_avg)
        X_updrs_list.append(clinical_vec)
        y_list.append(sample['label'])
        meta_list.append(sample)

    X_fc = np.array(X_fc_list, dtype=np.float32)
    X_updrs = np.array(X_updrs_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    if feature_type == 'FC':
        X = X_fc
    elif feature_type == 'UPDRS':
        X = X_updrs
    else:
        X = np.concatenate([X_fc, X_updrs], axis=1)

    logger.info(f"Feature matrix shape: {X.shape}, Class distribution: {np.bincount(y)}")
    return X, y, meta_list

# ======================
# 深度学习数据集（按需读取FC矩阵）
# ======================
class PatientDataset(Dataset):
    """
    供 GCN 使用：
    - 返回 fc_matrix (166,166) float32
    - 返回 clinical (C,) float32
    - 返回 label int64
    """
    def __init__(self, meta_list, feature_type):
        self.meta = meta_list
        self.feature_type = feature_type.upper()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        m = self.meta[idx]
        y = int(m['label'])
        clinical = m['clinical_features'].astype(np.float32)

        fc = None
        if self.feature_type in ['FC', 'ALL']:
            fc_matrix = np.load(m['fc_file']).astype(np.float32)
            if fc_matrix.shape != (166, 166):
                raise ValueError(f"Unexpected FC shape {fc_matrix.shape} for {m['id']}")
            fc = fc_matrix

        return {
            'id': m['id'],
            'fc': fc,
            'clinical': clinical,
            'label': y
        }

def collate_fn(batch):
    ids = [b['id'] for b in batch]
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)

    clinical = torch.tensor(np.stack([b['clinical'] for b in batch], axis=0), dtype=torch.float32)

    if batch[0]['fc'] is None:
        fc = None
    else:
        fc = torch.tensor(np.stack([b['fc'] for b in batch], axis=0), dtype=torch.float32)

    return ids, fc, clinical, labels

# ======================
# GCN 模型（不依赖 torch_geometric，手写归一化）
# ======================
def normalize_adjacency(A):
    """
    A: (B,N,N) adjacency (float)
    returns: (B,N,N) normalized with self-loop: D^{-1/2}(A+I)D^{-1/2}
    """
    B, N, _ = A.shape
    I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
    A_hat = A + I

    # ReLU to remove negative edges if desired (optional). 这里保持权重但把负值截断，避免度为负导致奇怪归一化
    A_hat = torch.relu(A_hat)

    deg = A_hat.sum(dim=-1)  # (B,N)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)  # (B,N,N)

    return D_inv_sqrt @ A_hat @ D_inv_sqrt


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A_norm):
        # X: (B,N,F), A_norm: (B,N,N)
        out = A_norm @ X
        out = self.lin(out)
        out = torch.relu(out)
        out = self.dropout(out)
        return out


class GCNClassifier(nn.Module):
    """
    将 FC(166x166) 视为图：
    - adjacency = FC（加自环后归一化）
    - node features = FC 的每行（每个节点的 connectivity profile，166维）
    - graph embedding = mean pool
    - 若 feature_type == ALL：将 clinical 经过MLP后拼接再分类
    """
    def __init__(self, feature_type, clinical_dim, params, dropout=0.3):
        super().__init__()
        self.feature_type = feature_type.upper()
        if self.feature_type == 'UPDRS':
            raise ValueError("GCN requires FC features (feature_type must be 'FC' or 'ALL').")

        gcn_hidden = params['gcn_hidden']
        gcn_layers = params['gcn_layers']

        in_dim = 166
        layers = []
        for i in range(gcn_layers):
            layers.append(GCNLayer(in_dim if i == 0 else gcn_hidden, gcn_hidden, dropout=dropout))
        self.gcn = nn.ModuleList(layers)

        if self.feature_type == 'ALL':
            self.clinical_proj = nn.Sequential(
                nn.Linear(clinical_dim, gcn_hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            head_in = gcn_hidden + gcn_hidden
        else:
            head_in = gcn_hidden

        self.head = nn.Sequential(
            nn.Linear(head_in, HYPERPARAMS['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HYPERPARAMS['hidden_dim'], 1)
        )

    def forward(self, fc, clinical):
        # fc: (B,166,166)
        A_norm = normalize_adjacency(fc)  # (B,N,N)
        X = fc  # node features: (B,N,166)

        for layer in self.gcn:
            X = layer(X, A_norm)  # (B,N,H)

        graph_emb = X.mean(dim=1)  # mean pool (B,H)

        if self.feature_type == 'ALL':
            c = self.clinical_proj(clinical)  # (B,H)
            graph_emb = torch.cat([graph_emb, c], dim=1)

        logits = self.head(graph_emb).squeeze(-1)
        return logits

# ======================
# 训练/评估：GCN
# ======================
def train_dl_model(model, train_loader, num_epochs, lr):
    model.train()

    all_labels = []
    for _, _, _, y in train_loader:
        all_labels.append(y)
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    n_pos = np.sum(all_labels == 1)
    n_neg = np.sum(all_labels == 0)
    if n_pos == 0:
        pos_weight = 1.0
    else:
        pos_weight = float(n_neg) / float(n_pos)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for _, fc, clinical, y in train_loader:
            y = y.to(device)
            if fc is not None:
                fc = fc.to(device)
            clinical = clinical.to(device)

            optimizer.zero_grad()
            logits = model(fc, clinical)
            loss = criterion(logits, y.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item() * y.size(0)

        epoch_loss /= max(1, len(train_loader.dataset))
        if epoch % 20 == 0 or epoch == 1 or epoch == num_epochs:
            logger.info(f"Epoch {epoch:03d}/{num_epochs} - loss: {epoch_loss:.6f}")

@torch.no_grad()
def predict_dl_model(model, data_loader):
    model.eval()
    all_probs = []
    all_labels = []
    all_ids = []

    for ids, fc, clinical, y in data_loader:
        if fc is not None:
            fc = fc.to(device)
        clinical = clinical.to(device)

        logits = model(fc, clinical)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        all_probs.append(probs)
        all_labels.append(y.numpy())
        all_ids.extend(ids)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_ids, all_probs, all_labels

# ======================
# 交叉验证实验 (GCN ONLY)
# ======================
def cross_validate_experiment(X, y, meta, experiment_name):
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Cross-Validation Experiment: {experiment_name}")
    logger.info(f"Total samples: {len(X)} | Class distribution: {np.bincount(y)}")
    logger.info("Using model type: GCN (only)")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = {'acc': [], 'f1': [], 'precision': [], 'recall': [], 'auc': [], 'ap': []}

    use_Youden = True
    feature_type = HYPERPARAMS['feature_type'].upper()
    clinical_dim = len(CLINICAL_COLS)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")

        y_train, y_test = y[train_idx], y[test_idx]

        logger.info(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
        logger.info(f"Train class balance - 0: {np.sum(y_train==0)}, 1: {np.sum(y_train==1)}")
        logger.info(f"Test class balance  - 0: {np.sum(y_test==0)}, 1: {np.sum(y_test==1)}")

        train_meta = [meta[i] for i in train_idx]
        test_meta = [meta[i] for i in test_idx]

        train_ds = PatientDataset(train_meta, feature_type=feature_type)
        test_ds = PatientDataset(test_meta, feature_type=feature_type)

        train_loader = DataLoader(
            train_ds,
            batch_size=HYPERPARAMS['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=False
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=HYPERPARAMS['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=False
        )

        model = GCNClassifier(
            feature_type=feature_type,
            clinical_dim=clinical_dim,
            params=HYPERPARAMS['gcn_params'],
            dropout=HYPERPARAMS['dropout']
        ).to(device)

        logger.info(f"Training GCN model for fold {fold+1}...")
        train_dl_model(
            model=model,
            train_loader=train_loader,
            num_epochs=HYPERPARAMS['num_epochs'],
            lr=HYPERPARAMS['learning_rate']
        )

        _, y_probs, _ = predict_dl_model(model, test_loader)

        # 阈值（与原逻辑一致：用 test 的roc找 Youden）
        if use_Youden and len(np.unique(y_test)) > 1:
            fpr, tpr, thresholds = roc_curve(y_test, y_probs)
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
            best_threshold = thresholds[best_idx]
            logger.info(f"Optimal Youden threshold: {best_threshold:.4f}")
        else:
            best_threshold = 0.5
            logger.info("Using default threshold 0.5")

        y_preds = (y_probs >= best_threshold).astype(int)

        acc = accuracy_score(y_test, y_preds)
        f1 = f1_score(y_test, y_preds, zero_division=0)
        precision = precision_score(y_test, y_preds, zero_division=0)
        recall = recall_score(y_test, y_preds, zero_division=0)

        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_probs)
            ap = average_precision_score(y_test, y_probs)
        else:
            auc = float('nan')
            ap = float('nan')

        fold_metrics['acc'].append(acc)
        fold_metrics['f1'].append(f1)
        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)
        fold_metrics['auc'].append(auc)
        fold_metrics['ap'].append(ap)

        # 保存预测结果
        fold_meta = [meta[i] for i in test_idx]
        result_df = pd.DataFrame(fold_meta)
        result_df['true_label'] = y_test
        result_df['pred_prob'] = y_probs
        result_df['pred_label'] = y_preds
        result_df['threshold'] = best_threshold
        result_df.to_csv(f"{experiment_name}_fold{fold+1}_predictions.csv", index=False)

        logger.info(f"Fold {fold+1} Results - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")

    summary = {}
    for metric, values in fold_metrics.items():
        values = np.array(values)
        valid_vals = values[~np.isnan(values)]
        if len(valid_vals) > 0:
            mean_val = np.mean(valid_vals)
            std_val = np.std(valid_vals) if len(valid_vals) > 1 else 0.0
        else:
            mean_val, std_val = float('nan'), float('nan')
        summary[metric] = (mean_val, std_val)

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL CROSS-VALIDATION RESULTS FOR: {experiment_name}")
    for metric, (mean, std) in summary.items():
        if np.isnan(mean):
            logger.info(f"{metric.upper():<10}: N/A")
        else:
            logger.info(f"{metric.upper():<10}: {mean:.4f} ± {std:.4f}")
    logger.info(f"{'='*60}\n")

    return summary

# ======================
# 主函数
# ======================
def main():
    start_time = time.time()
    logger.info("Starting treatment response prediction pipeline (GCN only)")

    model_type = HYPERPARAMS['model_type'].upper()
    feature_type = HYPERPARAMS['feature_type'].upper()

    if model_type != 'GCN':
        raise ValueError("This script is GCN-only. Please set model_type to 'GCN'.")

    if feature_type == 'UPDRS':
        raise ValueError("GCN model requires FC features. Please set feature_type to 'FC' or 'ALL'.")

    # 1. 构建FC文件索引
    logger.info("Building FC file index...")
    fc_index = build_fc_index(BASE_DIR)

    # 2. 解析UPDRS数据
    logger.info("Parsing UPDRS data with FC features...")
    samples = parse_updrs_data(UPDRS_CSV_PATH, fc_index)

    if len(samples) == 0:
        logger.error("No valid samples found. Exiting.")
        return

    # 3. 准备数据（用于分层K折）
    logger.info("Preparing feature matrices with FC features...")
    X, y, meta = prepare_data(samples)

    # 4. 交叉验证
    experiment_name = f"Treatment_Response_{feature_type}_GCN_Thresh{HYPERPARAMS['improvement_threshold']}"
    summary = cross_validate_experiment(X, y, meta, experiment_name)

    # 5. 保存最终结果
    result_file = f"final_results_{experiment_name}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {k: [float(v) for v in vs] for k, vs in summary.items()},
            'hyperparameters': HYPERPARAMS,
            'total_samples': int(len(X)),
            'class_distribution': np.bincount(y).tolist()
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {result_file}")
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
