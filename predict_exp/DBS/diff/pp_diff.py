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
import time
from datetime import datetime

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
# 全局配置 - 更新基础目录
# ======================
BASE_DIR = "/ailab/group/medai-share/syDu/ruijin/DBS/AAL_VTB"  # Updated to new base directory
UPDRS_CSV_PATH = "/ailab/group/medai-share/syDu/ruijin/DBS/DBS_UPDRS.csv"
RANDOM_SEED = 17
N_SPLITS = 5
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 临床子项列名（完全匹配CSV表头，注意中文冒号）
CLINICAL_COLS = [
    '言语', '面部表情', '强直_脖子', '强直_上肢：右', '强直_上肢：左', '强直_下肢：右', '强直_下肢：左',
    '手指拍打_右', '手指拍打_左', '手掌运动_右', '手掌运动_左', '前臂回旋_右', '前臂回旋_左',
    '脚趾拍地_右', '脚趾拍地_左', '脚灵敏度_右', '脚灵敏度_左', '起立', '步态', '步态冻结_步态冻结的评估',
    '姿势平稳度', '姿势', '全身自发性的动作评估_身体动作迟缓', '姿态性震颤_右', '姿态性震颤_左',
    '动作性震颤_右', '动作性震颤_左', '静止型震颤_上肢：右', '静止型震颤_上肢：左', '静止型震颤_下肢：右',
    '静止型震颤_下肢：左', '静止型震颤_嘴唇/下巴', '静止型震颤_持续性'
    # ,'UPDRS总分'
]

# ======================
# 超参数配置 (集中管理)
# ======================
HYPERPARAMS = {
    'batch_size': 8,        # 可调整
    'learning_rate': 0.0005, # 可调整
    'num_epochs': 200,       # 可调整
    'hidden_dim': 512,       # 模型隐藏层维度
    'num_blocks': 6,         # 残差块数量
    'dropout': 0.3,          # Dropout率
    'feature_type': 'ALL',  # 'DIFF', 'UPDRS', or 'ALL' - Updated to DIFF
    'improvement_threshold': 0.25 # 改善率阈值
}

logger.info(f"Hyperparameters: {json.dumps(HYPERPARAMS, indent=2)}")

# ======================
# 辅助函数：构建diff文件索引 (mean_anomaly & mean_distortion)
# ======================
def build_diff_index(base_dir):
    """
    构建diff文件索引字典 (mean_anomaly & mean_distortion)
    
    Returns:
        dict: {lowercase_id: {'anomaly': file_path, 'distortion': file_path}}
    """
    diff_index = {}
    
    # 规则: 从文件夹名提取患者ID (如 gpi02_fmri_20190901_143922_651000 -> gpi02)
    folder_pattern = re.compile(r'^([a-zA-Z0-9]+)_fmri')
    
    # 遍历一级子文件夹
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        # 尝试匹配患者ID
        match = folder_pattern.match(folder_name)
        if not match:
            logger.warning(f"Skipping folder with invalid name format: {folder_name}")
            continue
        
        pid = match.group(1).lower()
        
        # 构建文件路径
        patient_subfolder = os.path.join(folder_path, 'fine_tune', folder_name)
        file_prefix = f"patient_{folder_name}"
        
        anomaly_file = os.path.join(patient_subfolder, f"{file_prefix}_mean_anomaly.npy")
        distortion_file = os.path.join(patient_subfolder, f"{file_prefix}_mean_distortion.npy")
        
        # 验证文件存在
        if not os.path.exists(anomaly_file):
            logger.warning(f"Anomaly file not found: {anomaly_file}")
            continue
        if not os.path.exists(distortion_file):
            logger.warning(f"Distortion file not found: {distortion_file}")
            continue
        
        # 存储到索引
        if pid not in diff_index:
            diff_index[pid] = {
                'anomaly': anomaly_file,
                'distortion': distortion_file
            }
        else:
            logger.warning(f"Duplicate patient ID found: {pid}, skipping additional entry")
    
    logger.info(f"Built diff index with {len(diff_index)} patients")
    return diff_index

# ======================
# 解析UPDRS数据 - 更新为使用diff索引
# ======================
def parse_updrs_data(csv_path, diff_index):
    """
    解析UPDRS CSV数据，匹配diff文件
    
    Args:
        csv_path: UPDRS CSV文件路径
        diff_index: diff文件索引字典
    
    Returns:
        list: 有效样本列表
    """
    df = pd.read_csv(csv_path)
    
    # 验证必要列存在
    required_cols = ['ID', '评估时间', '手术情况', 'UPDRS-III改善率'] + CLINICAL_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in CSV: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # 按ID和评估时间排序 (确保DBS off在前)
    df = df.sort_values(by=['ID', '评估时间'])
    
    samples = []
    improvement_threshold = HYPERPARAMS['improvement_threshold']
    
    for pid, group in df.groupby('ID'):
        pid_lower = pid.strip().lower()
        
        # 检查diff文件是否存在
        if pid_lower not in diff_index:
            logger.warning(f"No diff records found for patient {pid}")
            continue
        
        # 每个病人应该有两行 (DBS off, DBS on)
        if len(group) < 2:
            logger.warning(f"Patient {pid} has only {len(group)} records, skipping")
            continue
        
        # 取DBS off行 (第一行)
        off_row = group.iloc[0]
        
        # 验证手术情况
        if off_row['手术情况'] != 'DBS off':
            logger.warning(f"First record for {pid} is not 'DBS off', skipping")
            continue
        
        # 获取改善率标签
        if pd.isna(off_row['UPDRS-III改善率']):
            logger.warning(f"Missing improvement rate for {pid}, skipping")
            continue
        
        label = 1 if off_row['UPDRS-III改善率'] >= improvement_threshold else 0
        
        # 提取临床特征 (DBS off状态)
        clinical_features = []
        for col in CLINICAL_COLS:
            val = off_row[col]
            if pd.isna(val):
                logger.warning(f"Missing clinical feature {col} for {pid}, skipping")
                clinical_features = None
                break
            try:
                # 尝试转换为数值
                clinical_features.append(float(val))
            except (ValueError, TypeError):
                logger.warning(f"Non-numeric value in {col} for {pid}: {val}, skipping")
                clinical_features = None
                break
        
        if clinical_features is None:
            continue
        
        # 获取diff文件路径
        anomaly_file = diff_index[pid_lower]['anomaly']
        distortion_file = diff_index[pid_lower]['distortion']
        
        samples.append({
            'id': pid,
            'anomaly_file': anomaly_file,
            'distortion_file': distortion_file,
            'clinical_features': np.array(clinical_features, dtype=np.float32),
            'label': label,
            'improvement_rate': off_row['UPDRS-III改善率'],
            'updrs_total_off': off_row.get('UPDRS总分', None)
        })
    
    logger.info(f"Found {len(samples)} valid samples")
    return samples

# ======================
# 数据准备 - 更新为加载diff特征
# ======================
def prepare_data(samples):
    """
    准备特征和标签 - 使用diff特征替代EC
    
    Args:
        samples: 样本列表
    
    Returns:
        X: 特征矩阵 (diff, UPDRS, or combined)
        y: 标签向量
        meta: 元数据列表
    """
    X_diff_list = []
    X_updrs_list = []
    y_list = []
    meta_list = []
    
    feature_type = HYPERPARAMS['feature_type'].upper()
    
    for sample in samples:
        # 加载并处理diff特征
        try:
            # 加载anomaly特征 (166x1x1 -> 166)
            anomaly = np.load(sample['anomaly_file']).flatten().astype(np.float32)
            # 加载distortion特征 (166x1x1 -> 166)
            distortion = np.load(sample['distortion_file']).flatten().astype(np.float32)
            
            # 合并为332维特征
            diff_combined = np.concatenate([anomaly, distortion])
            
            # 对整个diff特征进行z-score标准化
            # mean_val = diff_combined.mean()
            # std_val = diff_combined.std()
            # if std_val > 1e-8:
            #     diff_combined = (diff_combined - mean_val) / std_val
            # else:
            #     diff_combined = np.zeros_like(diff_combined)
                
        except Exception as e:
            logger.error(f"Error loading diff files for {sample['id']}: {str(e)}")
            continue
        

        clinical_vec = sample['clinical_features']
        
        # 根据特征类型选择
        if feature_type == 'DIFF':
            features = diff_combined
        elif feature_type == 'UPDRS':
            features = clinical_vec
        elif feature_type == 'ALL':
            features = np.concatenate([diff_combined, clinical_vec])
        else:
            raise ValueError(f"Invalid feature_type: {feature_type}. Choose from 'DIFF', 'UPDRS', 'ALL'")
        
        X_diff_list.append(diff_combined)
        X_updrs_list.append(clinical_vec)
        y_list.append(sample['label'])
        meta_list.append(sample)
    
    # 转换为numpy数组
    X_diff = np.array(X_diff_list, dtype=np.float32)
    X_updrs = np.array(X_updrs_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    # 根据特征类型组合
    if feature_type == 'DIFF':
        X = X_diff
    elif feature_type == 'UPDRS':
        X = X_updrs
    elif feature_type == 'ALL':
        X = np.concatenate([X_diff, X_updrs], axis=1)
    
    logger.info(f"Feature matrix shape: {X.shape}, Class distribution: {np.bincount(y)}")
    return X, y, meta_list

# ======================
# Dataset
# ======================
class ECMapDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ======================
# 模型定义 (保持不变)
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
# 交叉验证实验 (保持不变)
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
    
    # 早停参数
    PATIENCE = 50  # 容忍多少个epoch没有改善
    MIN_EPOCHS = 50  # 最小训练轮数
    BEST_MODEL_PATH = "best_model_temp.pth"  # 临时保存最佳模型
    
    # 超参数
    batch_size = HYPERPARAMS['batch_size']
    lr = HYPERPARAMS['learning_rate']
    num_epochs = HYPERPARAMS['num_epochs']
    hidden_dim = HYPERPARAMS['hidden_dim']
    num_blocks = HYPERPARAMS['num_blocks']
    dropout = HYPERPARAMS['dropout']
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 打印数据分布
        logger.info(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
        logger.info(f"Train class balance - 0: {np.sum(y_train==0)}, 1: {np.sum(y_train==1)}")
        logger.info(f"Test class balance  - 0: {np.sum(y_test==0)}, 1: {np.sum(y_test==1)}")
        
        # 创建数据集
        train_dataset = ECMapDataset(X_train, y_train)
        test_dataset = ECMapDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型
        model = DeeperMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout
        ).to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 早停相关变量
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = 0
        early_stop = False
        
        # 训练循环
        logger.info(f"Training fold {fold+1} with early stopping (patience={PATIENCE})...")
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 每个epoch后在测试集(验证集)上评估
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device).float()
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels)
                    val_loss += batch_loss.item() * inputs.size(0)
            
            val_loss = val_loss / len(test_loader.dataset)
            
            # 早停逻辑
            if epoch >= MIN_EPOCHS:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_no_improve = 0
                    # 保存最佳模型
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= PATIENCE:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}! "
                                    f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
                        early_stop = True
                        break
            
            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == num_epochs or early_stop:
                logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                            f"Train Loss: {epoch_loss/len(train_loader):.6f} | "
                            f"Val Loss: {val_loss:.6f} | "
                            f"Best Val Loss: {best_val_loss:.6f} (epoch {best_epoch+1})")
        
        # 加载最佳模型（如果有）
        if os.path.exists(BEST_MODEL_PATH):
            model.load_state_dict(torch.load(BEST_MODEL_PATH))
            os.remove(BEST_MODEL_PATH)  # 清理临时文件
            logger.info(f"Loaded best model from epoch {best_epoch+1}")
        else:
            logger.warning("No best model saved, using final model weights")
        
        # 评估
        model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 确定阈值
        if use_Youden and len(np.unique(all_labels)) > 1:
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
            best_threshold = thresholds[best_idx]
            logger.info(f"Optimal Youden threshold: {best_threshold:.4f}")
        else:
            best_threshold = 0.5
            logger.info("Using default threshold 0.5")
        
        all_preds = (all_probs >= best_threshold).astype(int)
        
        # 计算指标
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
            ap = average_precision_score(all_labels, all_probs)
        else:
            auc = float('nan')
            ap = float('nan')
        
        # 记录指标
        fold_metrics['acc'].append(acc)
        fold_metrics['f1'].append(f1)
        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)
        fold_metrics['auc'].append(auc)
        fold_metrics['ap'].append(ap)
        
        # 保存预测结果
        fold_meta = [meta[i] for i in test_idx]
        result_df = pd.DataFrame(fold_meta)
        result_df['true_label'] = all_labels
        result_df['pred_prob'] = all_probs
        result_df['pred_label'] = all_preds
        result_df['threshold'] = best_threshold
        result_df['best_epoch'] = best_epoch + 1  # 记录最佳epoch
        result_df.to_csv(f"{experiment_name}_fold{fold+1}_predictions.csv", index=False)
        
        logger.info(f"Fold {fold+1} Results - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")

    
    # 汇总结果
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
    logger.info("Starting treatment response prediction pipeline")
    
    # 1. 构建diff文件索引 (替代原EC索引)
    logger.info("Building diff file index (mean_anomaly & mean_distortion)...")
    diff_index = build_diff_index(BASE_DIR)  # Updated function call
    
    # 2. 解析UPDRS数据 - 使用diff索引
    logger.info("Parsing UPDRS data with diff features...")
    samples = parse_updrs_data(UPDRS_CSV_PATH, diff_index)  # Updated parameter
    
    if len(samples) == 0:
        logger.error("No valid samples found. Exiting.")
        return
    
    # 3. 准备数据 - 使用diff特征
    logger.info("Preparing feature matrices with diff features...")
    X, y, meta = prepare_data(samples)
    
    # 4. 交叉验证
    experiment_name = f"Treatment_Response_{HYPERPARAMS['feature_type']}_Thresh{HYPERPARAMS['improvement_threshold']}"
    summary = cross_validate_experiment(X, y, meta, experiment_name)
    
    # 5. 保存最终结果
    result_file = f"final_results_{experiment_name}.json"
    with open(result_file, 'w') as f:
        json.dump({
            'summary': {k: [float(v) for v in vs] for k, vs in summary.items()},
            'hyperparameters': HYPERPARAMS,
            'total_samples': len(X),
            'class_distribution': np.bincount(y).tolist()
        }, f, indent=2)
    
    logger.info(f"Results saved to {result_file}")
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()