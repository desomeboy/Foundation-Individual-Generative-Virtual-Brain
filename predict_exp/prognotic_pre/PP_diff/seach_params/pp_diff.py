import os
import re
import json
import random
import logging
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from sklearn.metrics import roc_curve


# ======================
# 命令行参数解析
# ======================
def parse_args():
    parser = argparse.ArgumentParser(description='Treatment Response Prediction with Seed/Epoch Search')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size for MLP')
    parser.add_argument('--num_blocks', type=int, default=4, help='Number of residual blocks')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # 训练参数（其余保持不变；num_epochs会在搜索时被覆盖）
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30, help='(Unused in search) Default number of epochs')
    parser.add_argument('--use_young', action='store_true', default=True, help='Use Youden index for threshold selection')
    parser.add_argument('--no_use_young', dest='use_young', action='store_false', help='Disable Youden index (use 0.5 threshold)')

    # 数据与特征参数
    parser.add_argument('--sample_type', type=str, choices=['all', 'diff', 'clinical'], default='all',
                        help="Feature type: 'all' (diff+clinical), 'diff' (only diff features), 'clinical' (only clinical features)")
    parser.add_argument('--zscore_normalize', action='store_true', default=False, help='Apply z-score normalization to diff features')

    # 实验设置
    parser.add_argument('--random_seed', type=int, default=777, help='(Unused in search) Default random seed')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--experiment_name', type=str, default='Treatment_Response_Diff_Clinical', help='Experiment name prefix for output files')

    # 路径参数
    parser.add_argument('--base_dir_diff', type=str, default='/ailab/group/medai-share/syDu/ruijin/Final_data/AAL3_VTB',
                        help='Base directory for diff features')
    parser.add_argument('--excel_path', type=str, default='/ailab/user/dusiyuan/code/Brain/EC/ruijin/TI_gt_1_10.xlsx',
                        help='Path to Excel metadata file')

    # 搜索范围（按你的要求给默认值）
    parser.add_argument('--epoch_candidates', type=str, default='20,25,30,40,50',
                        help='Comma-separated epoch candidates, e.g., "20,25,30,40,50"')
    parser.add_argument('--seed_start', type=int, default=1, help='Random seed start (inclusive)')
    parser.add_argument('--seed_end', type=int, default=5, help='Random seed end (inclusive)')

    args = parser.parse_args()

    # 验证参数合理性
    if args.dropout < 0 or args.dropout > 1:
        raise ValueError("Dropout rate must be between 0 and 1")
    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if args.n_splits < 2:
        raise ValueError("Number of splits must be at least 2")
    if args.seed_start < 1 or args.seed_end < args.seed_start:
        raise ValueError("Invalid seed range")
    # 解析epoch列表
    try:
        args.epoch_candidates = [int(x.strip()) for x in args.epoch_candidates.split(",") if x.strip() != ""]
    except Exception as e:
        raise ValueError(f"Invalid epoch_candidates: {args.epoch_candidates}") from e
    if not args.epoch_candidates or any(e <= 0 for e in args.epoch_candidates):
        raise ValueError("epoch_candidates must be positive integers")

    return args


# ======================
# 日志配置
# ======================
def setup_logging(experiment_name):
    log_filename = f"{experiment_name}_training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ======================
# 随机种子设置
# ======================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 可选：追求更强可复现（可能会变慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================
# 全局配置
# ======================
args = parse_args()
logger = setup_logging(args.experiment_name)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 路径配置
BASE_DIR_DIFF = args.base_dir_diff
EXCEL_PATH = args.excel_path
TARGET_MAP = {
    'L_STN': 'L_stn',
    'L_Gpi': 'L_gpi',
    'R_STN': 'R_stn',
    'R_Gpi': 'R_gpi',
    'L_GPi': 'L_gpi',
    'R_GPi': 'R_gpi',
}

# 临床子项列名（含总分）
CLINICAL_COLS = [
    '3.1_言语', '3.2_面部表情', '3.3a_强直-颈部', '3.3b_强直-右上肢', '3.3c_强直-左上肢',
    '3.3d_强直-右下肢', '3.3e_强直-左下肢', '3.4a_手指拍打-右手', '3.4b_手指拍打-左手',
    '3.5a_手掌运动-右手', '3.5b_手掌运动-左手', '3.6a_前臂回旋运动-右手', '3.6b_前臂回旋运动-左手',
    '3.7a_脚趾拍地运动-右脚', '3.7b_脚趾拍地运动-左脚', '3.8a_两脚灵敏度测试-右下肢', '3.8b_两脚灵敏度测试-左下肢',
    '3.9_起立', '3.10_步态', '3.11_步态冻结的评估', '3.12_姿势平稳度', '3.13_姿势',
    '3.14_全身自发性的动作评估(身体动作迟缓)', '3.15a_双手姿态性震颤-右上肢', '3.15b_双手姿态性震颤-左上肢',
    '3.16a_双手动作性震颤-右上肢', '3.16b_双手动作性震颤-左上肢', '3.17a_静止性震颤幅度-右上肢',
    '3.17b_静止性震颤幅度-左上肢', '3.17c_静止性震颤幅度-右下肢', '3.17d_静止性震颤幅度-左下肢',
    '3.17e_静止性震颤幅度-嘴唇或下颌', '3.18_静止性震颤持续性', '总分'
]


# ======================
# 辅助函数
# ======================
def clean_check_name(check_val):
    clean = re.sub(r'\s+', '_', str(check_val).strip())
    clean = re.sub(r'[^\w\-_\.]', '_', clean)
    return clean


def build_diff_paths(clean_check):
    patient_dir = os.path.join(BASE_DIR_DIFF, clean_check, 'fine_tune', clean_check)
    anomaly = os.path.join(patient_dir, f"patient_{clean_check}_mean_anomaly.npy")
    distortion = os.path.join(patient_dir, f"patient_{clean_check}_mean_distortion.npy")
    return anomaly, distortion


def zscore_normalize_array(arr: np.ndarray) -> np.ndarray:
    mean = np.mean(arr)
    std = np.std(arr)
    if std > 1e-8:
        return (arr - mean) / std
    else:
        return arr - mean


# ======================
# 数据解析函数
# ======================
def parse_excel_data(excel_path):
    """解析Excel数据，返回病人级别的记录"""
    df = pd.read_excel(excel_path)

    if '姓名' not in df.columns:
        logger.error("Column '姓名' is missing in Excel.")
        raise ValueError("Required column '姓名' not found.")

    df['姓名'] = df['姓名'].astype(str).str.strip()
    df['日期'] = df['日期'].astype(str).str.strip()

    required_cols = ['Check', '总分', '靶点位置', '治疗状态', '姓名', '日期'] + CLINICAL_COLS
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"Missing columns in Excel: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    groups = df.groupby(['姓名', '日期'])
    patients = []

    logger.info("Parsing patient data from Excel (grouped by 姓名 + 日期)...")
    for (name, date), group in tqdm(groups, desc="Processing patients"):
        if len(group) != 2:
            continue

        pre_row = group[group['治疗状态'] == '前']
        post_row = group[group['治疗状态'] == '后']
        if len(pre_row) != 1 or len(post_row) != 1:
            continue

        pre_row = pre_row.iloc[0]
        post_row = post_row.iloc[0]

        if pre_row[CLINICAL_COLS].isnull().any() or post_row[CLINICAL_COLS].isnull().any():
            continue

        clean_check_pre = clean_check_name(pre_row['Check'])
        clean_check_post = clean_check_name(post_row['Check'])

        anomaly_path_pre, distortion_path_pre = build_diff_paths(clean_check_pre)
        anomaly_path_post, distortion_path_post = build_diff_paths(clean_check_post)

        if not (os.path.exists(anomaly_path_pre) and os.path.exists(distortion_path_pre) and
                os.path.exists(anomaly_path_post) and os.path.exists(distortion_path_post)):
            logger.warning(f"Missing diff files for name={name}, date={date}")
            continue

        target_folder = TARGET_MAP.get(pre_row['靶点位置'])
        if target_folder is None:
            logger.warning(f"Unknown target location: {pre_row['靶点位置']} for {name}")
            continue

        target_original = pre_row['靶点位置']

        id_date = f"{name}_{date}"

        patients.append({
            'id_date': id_date,
            'timepoint': 'pre',
            'pre_score': int(pre_row['总分']),
            'post_score': int(post_row['总分']),
            'anomaly_path': anomaly_path_pre,
            'distortion_path': distortion_path_pre,
            'clinical_features': pre_row[CLINICAL_COLS].values.astype(np.float32),
            'target': target_folder,
            'target_original': target_original,
            'name': name,
            'check_name': clean_check_pre
        })

    logger.info(f"Found {len(patients)} valid pre-treatment samples with complete data")
    return patients


# ======================
# 数据准备函数
# ======================
def prepare_response_data(patients, sample_type='all', apply_zscore=False):
    """准备治疗响应预测数据"""
    X_diff, y, meta = [], [], []

    logger.info(f"Preparing features and labels (sample_type={sample_type}, zscore_normalize={apply_zscore})...")
    for patient in tqdm(patients, desc="Loading diff features"):
        anomaly = np.load(patient['anomaly_path']).flatten()
        distortion = np.load(patient['distortion_path']).flatten()

        if anomaly.shape[0] != 166 or distortion.shape[0] != 166:
            logger.warning(f"Invalid shape for {patient['check_name']}: "
                           f"anomaly={anomaly.shape}, distortion={distortion.shape}")
            continue

        # 应用z-score归一化
        if apply_zscore:
            anomaly = zscore_normalize_array(anomaly)
            distortion = zscore_normalize_array(distortion)

        clinical_vec = patient['clinical_features'].astype(np.float32) / 4.0
        response = 1 if (patient['pre_score'] - patient['post_score']) >= 5 else 0

        if sample_type == 'all':
            features = np.concatenate([anomaly, distortion, clinical_vec])
        elif sample_type == 'diff':
            features = np.concatenate([anomaly, distortion])
        elif sample_type == 'clinical':
            features = clinical_vec
        else:
            raise ValueError("sample_type must be 'all', 'diff', or 'clinical'")

        X_diff.append(features)
        y.append(response)
        meta.append({
            **patient,
            'response_label': response,
            'feature_type': sample_type
        })

    X = np.array(X_diff, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    logger.info(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    if len(y) > 0:
        logger.info(f"Class distribution: {np.bincount(y)}")

    return X, y, meta


# ======================
# Dataset
# ======================
class DiffDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


# ======================
# 模型定义
# ======================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)


class DeeperMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_blocks=4, dropout=0.1):
        super(DeeperMLP, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_head(x)
        return x.squeeze(-1)


# ======================
# 交叉验证：只计算AUC和AP（不保存fold预测）
# ======================
def cross_validate_auc_ap_only(
    X, y,
    batch_size=8, lr=0.001, num_epochs=30,
    hidden_dim=512, num_blocks=4, dropout=0.1,
    use_young=True, n_splits=5, random_seed=777
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    input_dim = X.shape[1]

    fold_auc = []
    fold_ap = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_dataset = DiffDataset(X_train, y_train)
        test_dataset = DiffDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = DeeperMLP(input_dim, hidden_dim, num_blocks, dropout).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练
        for _epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

        # 推理
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                all_probs.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # 如果use_young且两类都存在，可用Youden阈值（这里只是保持一致；AUC/AP不受阈值影响）
        if use_young and len(np.unique(all_labels)) > 1:
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
            _best_threshold = thresholds[best_idx]
        else:
            _best_threshold = 0.5

        # 计算AUC/AP
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
            ap = average_precision_score(all_labels, all_probs)
        else:
            auc = float('nan')
            ap = float('nan')

        fold_auc.append(auc)
        fold_ap.append(ap)

    return fold_auc, fold_ap


# ======================
# 搜索 random_seed & num_epochs，并保存结果为CSV
# ======================
def search_seed_and_epochs(X, y, out_csv_path: str):
    results = []

    epoch_list = args.epoch_candidates
    seeds = list(range(args.seed_start, args.seed_end + 1))

    total_runs = len(epoch_list) * len(seeds)
    logger.info(f"Search space: num_epochs={epoch_list}, random_seed=[{args.seed_start}..{args.seed_end}]")
    logger.info(f"Total combinations: {total_runs}")

    pbar = tqdm(total=total_runs, desc="Searching (seed, epochs)")

    for num_epochs in epoch_list:
        for seed in seeds:
            # 每个组合都重新设随机种子，保证可复现
            set_global_seed(seed)

            fold_auc, fold_ap = cross_validate_auc_ap_only(
                X, y,
                batch_size=args.batch_size,
                lr=args.lr,
                num_epochs=num_epochs,
                hidden_dim=args.hidden_dim,
                num_blocks=args.num_blocks,
                dropout=args.dropout,
                use_young=args.use_young,
                n_splits=args.n_splits,
                random_seed=seed
            )

            # 组装一行：5折AUC/AP + mean（仍然只包含AUC/AP信息）
            row = {
                "random_seed": seed,
                "num_epochs": num_epochs,
            }
            for i in range(args.n_splits):
                row[f"auc_fold{i+1}"] = fold_auc[i]
                row[f"ap_fold{i+1}"] = fold_ap[i]

            # mean（忽略nan）
            auc_vals = np.array(fold_auc, dtype=float)
            ap_vals = np.array(fold_ap, dtype=float)
            row["auc_mean"] = np.nanmean(auc_vals) if not np.all(np.isnan(auc_vals)) else float('nan')
            row["ap_mean"] = np.nanmean(ap_vals) if not np.all(np.isnan(ap_vals)) else float('nan')

            results.append(row)
            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(results)
    df.to_csv(out_csv_path, index=False)
    logger.info(f"Saved search results to: {out_csv_path}")

    # 也顺便把当前最优（按auc_mean优先、其次ap_mean）打印出来
    df_sorted = df.sort_values(by=["auc_mean", "ap_mean"], ascending=[False, False])
    best = df_sorted.iloc[0].to_dict()
    logger.info(f"Best combo (by auc_mean then ap_mean): seed={best['random_seed']}, epochs={best['num_epochs']}, "
                f"auc_mean={best['auc_mean']:.6f}, ap_mean={best['ap_mean']:.6f}")


# ======================
# 主函数
# ======================
def main():
    logger.info("Starting experiment with parameters:")
    logger.info(f"Base directory: {BASE_DIR_DIFF}")
    logger.info(f"Excel path: {EXCEL_PATH}")
    logger.info(f"Sample type: {args.sample_type}")
    logger.info(f"Z-score normalize: {args.zscore_normalize}")
    logger.info(f"Model: hidden_dim={args.hidden_dim}, num_blocks={args.num_blocks}, dropout={args.dropout}")
    logger.info(f"Train: lr={args.lr}, batch_size={args.batch_size}, n_splits={args.n_splits}, use_young={args.use_young}")
    logger.info(f"Search: epochs={args.epoch_candidates}, seeds=[{args.seed_start}..{args.seed_end}]")

    patients = parse_excel_data(EXCEL_PATH)
    if not patients:
        logger.error("No valid patients found. Exiting.")
        return

    X, y, _meta = prepare_response_data(
        patients,
        sample_type=args.sample_type,
        apply_zscore=args.zscore_normalize
    )
    if len(X) == 0:
        logger.error("No features extracted. Exiting.")
        return

    out_csv = f"{args.experiment_name}_seed_epoch_search.csv"
    search_seed_and_epochs(X, y, out_csv_path=out_csv)


if __name__ == "__main__":
    main()
