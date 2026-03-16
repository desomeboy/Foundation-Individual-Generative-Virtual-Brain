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

# ======================
# 日志配置
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_response_with_diff.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================
# 全局配置
# ======================
BASE_DIR_DIFF = "/ailab/group/medai-share/syDu/ruijin/10.10/AAL3_VTB"  # 新的diff文件根目录
EXCEL_PATH = "/ailab/user/dusiyuan/code/Brain/EC/ruijin/fmri_symptoms.xlsx"
RANDOM_SEED = 42
N_SPLITS = 5
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

TARGET_MAP = {
    '左侧stn20min': 'L_stn',
    '左侧gpi20min': 'L_gpi',
    '右侧stn20min': 'R_stn',
    '右侧gpi20min': 'R_gpi'
}

# 临床子项列名（排除“总分”）
CLINICAL_COLS = [
    '3.1_言语', '3.2_面部表情', '3.3a_强直-颈部', '3.3b_强直-右上肢', '3.3c_强直-左上肢',
    '3.3d_强直-右下肢', '3.3e_强直-左下肢', '3.4a_手指拍打-右手', '3.4b_手指拍打-左手',
    '3.5a_手掌运动-右手', '3.5b_手掌运动-左手', '3.6a_前臂回旋运动-右手', '3.6b_前臂回旋运动-左手',
    '3.7a_脚趾拍地运动-右脚', '3.7b_脚趾拍地运动-左脚', '3.8a_两脚灵敏度测试-右下肢', '3.8b_两脚灵敏度测试-左下肢',
    '3.9_起立', '3.10_步态', '3.11_步态冻结的评估', '3.12_姿势平稳度', '3.13_姿势',
    '3.14_全身自发性的动作评估(身体动作迟缓)', '3.15a_双手姿态性震颤-右上肢', '3.15b_双手姿态性震颤-左上肢',
    '3.16a_双手动作性震颤-右上肢', '3.16b_双手动作性震颤-左上肢', '3.17a_静止性震颤幅度-右上肢',
    '3.17b_静止性震颤幅度-左上肢', '3.17c_静止性震颤幅度-右下肢', '3.17d_静止性震颤幅度-左下肢',
    '3.17e_静止性震颤幅度-嘴唇或下颌', '3.18_静止性震颤持续性'
]

# ======================
# 辅助函数：解析Excel表格（病人级别）
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

def zscore_nomalize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    if std > 1e-8:
        return (arr - mean) / std
    else:
        return arr - mean       


def parse_excel_data(excel_path):
    """
    解析Excel数据，返回病人级别的记录（每个病人一条记录）
    只保留同时有治疗前/后记录、且diff文件存在的病人
    """
    df = pd.read_excel(excel_path)
    df['序号'] = df['序号'].ffill()
    df['日期'] = df['日期'].ffill()
    
    # 检查必要列
    required_cols = ['Check', '总分', '靶点位置', '治疗状态', '序号', '日期'] + CLINICAL_COLS
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"Missing columns in Excel: {missing}")
        raise ValueError(f"Missing required columns: {missing}")
    
    # 按病人+日期分组
    groups = df.groupby(['序号', '日期'])
    patients = []
    
    logger.info("Parsing patient data from Excel...")
    for (id_num, date), group in tqdm(groups, desc="Processing patients"):
        # 验证组内有且仅有两条记录（治疗前/后）
        if len(group) != 2:
            continue
            
        # 分离治疗前/后记录
        pre_row = group[group['治疗状态'] == '前']
        post_row = group[group['治疗状态'] == '后']
        if len(pre_row) != 1 or len(post_row) != 1:
            continue
        
        pre_row = pre_row.iloc[0]
        post_row = post_row.iloc[0]
        
        # 检查临床子项完整性
        if pre_row[CLINICAL_COLS].isnull().any() or post_row[CLINICAL_COLS].isnull().any():
            continue
            
        # 清理Check名称（用于路径）
        clean_check_pre = clean_check_name(pre_row['Check'])
        clean_check_post = clean_check_name(post_row['Check'])     
        
        
        # 构建diff文件路径
        anomaly_path_pre, distortion_path_pre = build_diff_paths(clean_check_pre)
        anomaly_path_post, distortion_path_post = build_diff_paths(clean_check_post)
        

        
        if not (os.path.exists(anomaly_path_pre) and os.path.exists(distortion_path_pre) and
                os.path.exists(anomaly_path_post) and os.path.exists(distortion_path_post)):
            logger.debug(f"Missing diff files for pre: {anomaly_path_pre}, post: {anomaly_path_post}")
            continue
            
        # 映射靶点位置
        target_folder = TARGET_MAP.get(pre_row['靶点位置'])
        if not target_folder:
            continue
            
        # 提取临床特征（治疗前）
        clinical_scores = pre_row[CLINICAL_COLS].values.astype(np.float32)
        
        patients.append({
            'id_date': f"{id_num}_{date}",
            'timepoint': 'pre',
            'pre_score': int(pre_row['总分']),
            'post_score': int(post_row['总分']),
            'anomaly_path': anomaly_path_pre,
            'distortion_path': distortion_path_pre,
            'clinical_features': pre_row[CLINICAL_COLS].values.astype(np.float32),
            'target': target_folder,
            'check_name': clean_check_pre
        })

        patients.append({
            'id_date': f"{id_num}_{date}",
            'timepoint': 'post',
            'pre_score': int(pre_row['总分']),
            'post_score': int(post_row['总分']),
            'anomaly_path': anomaly_path_post,
            'distortion_path': distortion_path_post,
            'clinical_features': post_row[CLINICAL_COLS].values.astype(np.float32),
            'target': target_folder,
            'check_name': clean_check_post
        })
    
    logger.info(f"Found {len(patients)} valid patients with complete data")
    return patients

# ======================
# 数据准备函数：只使用治疗前数据预测响应
# ======================
def prepare_response_data(patients, sample_type='all'):
    """
    准备治疗响应预测数据
    每个病人一个样本（治疗前diff + 临床特征）
    标签 = 1 if (pre_score - post_score) >= 5 else 0
    """
    X_diff, X_clinical, y, meta = [], [], [], []
    
    logger.info("Preparing features and labels...")
    for patient in tqdm(patients, desc="Loading diff features"):
        # 加载diff文件 (166x1x1)
        anomaly = np.load(patient['anomaly_path']).flatten()  # (166,)
        distortion = np.load(patient['distortion_path']).flatten()  # (166,)
        
        # 检查维度
        if anomaly.shape[0] != 166 or distortion.shape[0] != 166:
            logger.warning(f"Invalid shape for {patient['check_name']}: "
                          f"anomaly={anomaly.shape}, distortion={distortion.shape}")
            continue
        
        
        #z_score
        # anomaly = zscore_nomalize(anomaly)
        # distortion = zscore_nomalize(distortion)
        
        # 临床特征归一化 (0-4 -> 0-1)
        clinical_vec = patient['clinical_features'].astype(np.float32) / 4.0
        
        # 计算响应标签
        response = 1 if (patient['pre_score'] - patient['post_score']) >= 5 else 0
        
        # 根据sample_type选择特征
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
# 残差块 (保持不变)
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
        return x + self.block(x)  # residual connection

# ======================
# DEEP MLP 模型（自动适配输入维度）
# ======================
class DeeperMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_blocks=4, dropout=0.3):
        super(DeeperMLP, self).__init__()
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Stack residual blocks
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        
        # Final output head
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
# 交叉验证函数
# ======================
def cross_validate_experiment(X, y, meta, experiment_name):
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Cross-Validation Experiment: {experiment_name}")
    logger.info(f"Total samples: {len(X)} | Class distribution: {np.bincount(y)}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = {
        'acc': [], 'f1': [], 'precision': [], 'recall': [], 'auc': [], 'ap': []
    }
    use_Youden = True
    input_dim = X.shape[1]

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 打印数据分布
        logger.info(f"Train set: {len(y_train)} samples | Class 0: {np.sum(y_train==0)}, Class 1: {np.sum(y_train==1)}")
        logger.info(f"Test set: {len(y_test)} samples | Class 0: {np.sum(y_test==0)}, Class 1: {np.sum(y_test==1)}")
        
        test_meta = [meta[i] for i in test_idx]

        train_dataset = DiffDataset(X_train, y_train)
        test_dataset = DiffDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        model = DeeperMLP(input_dim).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 100
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
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Fold {fold+1} | Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss/len(train_loader):.4f}")

        # Evaluation
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

        # Youden threshold
        if use_Youden and len(np.unique(all_labels)) > 1:
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
            best_threshold = thresholds[best_idx]
            all_preds = (all_probs >= best_threshold).astype(int)
            logger.info(f"Fold {fold+1} - Optimal Youden threshold: {best_threshold:.4f}")
        else:
            best_threshold = 0.5
            all_preds = (all_probs > 0.5).astype(int)

        # Metrics
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

        # Save predictions
        result_df = pd.DataFrame(test_meta)
        result_df['true_label'] = all_labels
        result_df['pred_prob'] = all_probs
        result_df['pred_label'] = all_preds
        result_df['threshold'] = best_threshold
        result_df.to_csv(f"{experiment_name}_fold{fold+1}_predictions.csv", index=False)

        logger.info(f"Fold {fold+1} - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    # Summary
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

# ======================
# 主函数
# ======================
def main():
    logger.info("Parsing Excel data with diff features...")
    patients = parse_excel_data(EXCEL_PATH)
    
    if not patients:
        logger.error("No valid patients found. Exiting.")
        return
    
    # 准备数据 (使用diff+临床特征)
    X, y, meta = prepare_response_data(patients, sample_type='all')
    
    if len(X) == 0:
        logger.error("No features extracted. Exiting.")
        return
        
    # 运行实验
    experiment_name = "Treatment_Response_Diff_Clinical"
    cross_validate_experiment(X, y, meta, experiment_name)

if __name__ == "__main__":
    main()