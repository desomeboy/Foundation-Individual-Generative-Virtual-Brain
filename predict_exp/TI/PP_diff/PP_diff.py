import os
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
import re
from tqdm import tqdm
import logging
import json
import argparse  


def parse_args():
    parser = argparse.ArgumentParser(description='Treatment Response Prediction with Hyperparameter Tuning')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size for MLP')
    parser.add_argument('--num_blocks', type=int, default=4, help='Number of residual blocks')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--use_young', action='store_true', default=True, help='Use Youden index for threshold selection')
    parser.add_argument('--no_use_young', dest='use_young', action='store_false', help='Disable Youden index (use 0.5 threshold)')
    
    # Data and feature parameters
    parser.add_argument('--sample_type', type=str, choices=['all', 'diff', 'clinical'], default='all',
                        help="Feature type: 'all' (diff+clinical), 'diff' (only CBM features), 'clinical' (only clinical features)")
    parser.add_argument('--zscore_normalize', action='store_true', default=False, help='Apply z-score normalization to diff features')
    
    # Experimental settings
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--experiment_name', type=str, default='Treatment_Response_Diff_Clinical', help='Experiment name prefix for output files')
    
    # Path parameters
    parser.add_argument('--base_dir_diff', type=str, default='./ruijin/TI_data/AAL3_VTB',
                        help='The root directory of the VTB folder obtained after running train_iVB.py')
    parser.add_argument('--excel_path', type=str, default='./ruijin/TI_gt.xlsx',
                        help='Clinical information corresponding to the data')
    
    args = parser.parse_args()
    
    # Validate parameter reasonableness
    if args.dropout < 0 or args.dropout > 1:
        raise ValueError("Dropout rate must be between 0 and 1")
    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if args.num_epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    if args.n_splits < 2:
        raise ValueError("Number of splits must be at least 2")
    
    return args


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


args = parse_args()
logger = setup_logging(args.experiment_name)


torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
logger.info(f"Random seed set to: {args.random_seed}")


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")


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

# Clinical sub-item column names
CLINICAL_COLS = [
    '3.1_言语', '3.2_面部表情', '3.3a_强直-颈部', '3.3b_强直-右上肢', '3.3c_强直-左上肢',
    '3.3d_强直-右下肢', '3.3e_强直-左下肢', '3.4a_手指拍打-右手', '3.4b_手指拍打-左手',
    '3.5a_手掌运动-右手', '3.5b_手掌运动-左手', '3.6a_前臂回旋运动-右手', '3.6b_前臂回旋运动-左手',
    '3.7a_脚趾拍地运动-右脚', '3.7b_脚趾拍地运动-左脚', '3.8a_两脚灵敏度测试-右下肢', '3.8b_两脚灵敏度测试-左下肢',
    '3.9_起立', '3.10_步态', '3.11_步态冻结的评估', '3.12_姿势平稳度', '3.13_姿势',
    '3.14_全身自发性的动作评估(身体动作迟缓)', '3.15a_双手姿态性震颤-右上肢', '3.15b_双手姿态性震颤-左上肢',
    '3.16a_双手动作性震颤-右上肢', '3.16b_双手动作性震颤-左上肢', '3.17a_静止性震颤幅度-右上肢',
    '3.17b_静止性震颤幅度-左上肢', '3.17c_静止性震颤幅度-右下肢', '3.17d_静止性震颤幅度-左下肢',
    '3.17e_静止性震颤幅度-嘴唇或下颌', '3.18_静止性震颤持续性','总分'
]


def clean_check_name(check_val):
    clean = re.sub(r'\s+', '_', str(check_val).strip())
    clean = re.sub(r'[^\w\-_\.]', '_', clean)
    return clean

def build_diff_paths(clean_check):
    patient_dir = os.path.join(BASE_DIR_DIFF, clean_check, 'fine_tune', clean_check)
    anomaly = os.path.join(patient_dir, f"patient_{clean_check}_mean_anomaly.npy")
    distortion = os.path.join(patient_dir, f"patient_{clean_check}_mean_distortion.npy")
    return anomaly, distortion

def zscore_normalize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    if std > 1e-8:
        return (arr - mean) / std
    else:
        return arr - mean       


def parse_excel_data(excel_path):
    """Parse Excel data, return patient-level records"""
    df = pd.read_excel(excel_path)
    original_check_set = set(df['Check'].dropna().astype(str).str.strip())

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
    processed_check_set = set()

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

        clinical_scores = pre_row[CLINICAL_COLS].values.astype(np.float32)
        target_original = pre_row['靶点位置']

        processed_check_set.add(str(pre_row['Check']).strip())
        processed_check_set.add(str(post_row['Check']).strip())

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


def prepare_response_data(patients, sample_type='all', zscore_normalize=False):
    """Prepare treatment response prediction data"""
    X_diff, X_clinical, y, meta = [], [], [], []
    
    logger.info(f"Preparing features and labels (sample_type={sample_type}, zscore_normalize={zscore_normalize})...")
    for patient in tqdm(patients, desc="Loading diff features"):
        anomaly = np.load(patient['anomaly_path']).flatten()
        distortion = np.load(patient['distortion_path']).flatten()
        
        if anomaly.shape[0] != 166 or distortion.shape[0] != 166:
            logger.warning(f"Invalid shape for {patient['check_name']}: "
                          f"anomaly={anomaly.shape}, distortion={distortion.shape}")
            continue
        

        if zscore_normalize:
            anomaly = zscore_normalize(anomaly)
            distortion = zscore_normalize(distortion)
        
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
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    return X, y, meta


class DiffDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


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


def cross_validate_experiment(X, y, meta, experiment_name, 
                             batch_size=8, lr=0.001, num_epochs=30,
                             hidden_dim=512, num_blocks=4, dropout=0.1,
                             use_young=True, n_splits=5, random_seed=777):
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Cross-Validation Experiment: {experiment_name}")
    logger.info(f"Total samples: {len(X)} | Class distribution: {np.bincount(y)}")
    logger.info(f"Parameters: batch_size={batch_size}, lr={lr}, epochs={num_epochs}, "
                f"hidden_dim={hidden_dim}, num_blocks={num_blocks}, dropout={dropout}, "
                f"use_young={use_young}, n_splits={n_splits}, random_seed={random_seed}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_metrics = {
        'acc': [], 'f1': [], 'precision': [], 'recall': [], 'auc': [], 'ap': []
    }
    input_dim = X.shape[1]

    all_fold_dfs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        logger.info(f"Train set: {len(y_train)} samples | Class 0: {np.sum(y_train==0)}, Class 1: {np.sum(y_train==1)}")
        logger.info(f"Test set: {len(y_test)} samples | Class 0: {np.sum(y_test==0)}, Class 1: {np.sum(y_test==1)}")
        
        test_meta = [meta[i] for i in test_idx]

        train_dataset = DiffDataset(X_train, y_train)
        test_dataset = DiffDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = DeeperMLP(input_dim, hidden_dim, num_blocks, dropout).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        logger.info(f"Training fold {fold+1}...")
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                logger.info(f"Fold {fold+1} | Epoch {epoch+1}/{num_epochs} | "
                            f"Train Loss: {epoch_loss/len(train_loader):.4f}")


        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)


        if use_young and len(np.unique(all_labels)) > 1:
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
            best_threshold = thresholds[best_idx]
            all_preds = (all_probs >= best_threshold).astype(int)
            logger.info(f"Fold {fold+1} - Optimal Youden threshold: {best_threshold:.4f}")
        else:
            best_threshold = 0.5
            all_preds = (all_probs > 0.5).astype(int)


        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)

        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
            ap = average_precision_score(all_labels, all_probs)
        else:
            auc = float('nan')
            ap = float('nan')

        fold_metrics['acc'].append(acc)
        fold_metrics['f1'].append(f1)
        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)
        fold_metrics['auc'].append(auc)
        fold_metrics['ap'].append(ap)


        result_df = pd.DataFrame({
            'id_date': [m['id_date'] for m in test_meta],
            'name': [m['name'] for m in test_meta],
            'target_original': [m['target_original'] for m in test_meta],
            'target_mapped': [m['target'] for m in test_meta],
            'pre_score': [m['pre_score'] for m in test_meta],
            'post_score': [m['post_score'] for m in test_meta],
            'timepoint': [m['timepoint'] for m in test_meta],
            'true_label': all_labels,
            'pred_prob': all_probs,
            'pred_label': all_preds,
            'threshold': best_threshold,
        })
        result_df.to_csv(f"{experiment_name}_fold{fold+1}_predictions.csv", index=False)
        all_fold_dfs.append(result_df)

        logger.info(f"Fold {fold+1} - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")


    combined_df = pd.concat(all_fold_dfs, ignore_index=True)
    combined_df.to_csv(f"{experiment_name}_all_folds_combined.csv", index=False)
    logger.info(f"Combined results saved to {experiment_name}_all_folds_combined.csv")

    # Calculate summary metrics
    summary = {}
    for metric, values in fold_metrics.items():
        values = np.array(values)
        if np.all(np.isnan(values)):
            mean_val, std_val = float('nan'), float('nan')
        else:
            valid_vals = values[~np.isnan(values)]
            mean_val = np.mean(valid_vals) if len(valid_vals) > 0 else float('nan')
            std_val = np.std(valid_vals) if len(valid_vals) > 1 else 0.0
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


def main():

    global BASE_DIR_DIFF, EXCEL_PATH
    
    logger.info("Starting experiment with parameters:")
    logger.info(f"Base directory: {BASE_DIR_DIFF}")
    logger.info(f"Excel path: {EXCEL_PATH}")
    logger.info(f"Sample type: {args.sample_type}")
    logger.info(f"Z-score normalize: {args.zscore_normalize}")
    
    patients = parse_excel_data(EXCEL_PATH)
    
    if not patients:
        logger.error("No valid patients found. Exiting.")
        return
    
    X, y, meta = prepare_response_data(
        patients, 
        sample_type=args.sample_type,
        zscore_normalize=args.zscore_normalize
    )
    
    if len(X) == 0:
        logger.error("No features extracted. Exiting.")
        return
        
    cross_validate_experiment(
        X, y, meta, args.experiment_name,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        use_young=args.use_young,
        n_splits=args.n_splits,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main()