import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import re
from tqdm import tqdm
import logging
import json
import time
from datetime import datetime

class SoftSpearmanLoss(nn.Module):
    """
    使用可微分的soft ranking近似Spearman相关系数
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def soft_rank(self, x):
        """可微分的软排名"""
        n = x.size(0)
        # 计算每个元素比其他元素小的概率之和
        x_diff = x.unsqueeze(1) - x.unsqueeze(0)  # (n, n)
        # 使用sigmoid近似阶跃函数
        ranks = torch.sigmoid(x_diff / self.temperature).sum(dim=1) + 1
        return ranks
    
    def forward(self, pred, target):
        pred_rank = self.soft_rank(pred)
        target_rank = self.soft_rank(target)
        
        # 计算Spearman相关系数（负值作为损失）
        pred_centered = pred_rank - pred_rank.mean()
        target_centered = target_rank - target_rank.mean()
        
        cov = (pred_centered * target_centered).mean()
        pred_std = pred_centered.std() + 1e-8
        target_std = target_centered.std() + 1e-8
        
        spearman = cov / (pred_std * target_std)
        
        # 返回负相关作为损失（最大化相关 = 最小化负相关）
        return 1 - spearman


class HybridRegressionLoss(nn.Module):
    """
    MSE + 排名损失的组合
    alpha: MSE权重
    beta: 排名损失权重
    """
    def __init__(self, alpha=0.3, beta=0.7, margin=0.05, temperature=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        # self.ranking = PairwiseRankingLoss(margin=margin)
        # 可选：加入soft spearman
        self.soft_spearman = SoftSpearmanLoss(temperature=temperature)
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        # rank_loss = self.ranking(pred, target)
        spearman_loss = self.soft_spearman(pred, target)
        
        return self.alpha * mse_loss + self.beta * spearman_loss


# ======================
# 日志配置
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("treatment_response_regression.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================
# 全局配置
# ======================
BASE_DIR = "/ailab/group/medai-share/syDu/ruijin/DBS/AAL_VTB"
UPDRS_CSV_PATH = "/ailab/group/medai-share/syDu/ruijin/DBS/DBS_UPDRS.csv"
RANDOM_SEED = 17
N_SPLITS = 5
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 临床子项列名
CLINICAL_COLS = [
    '言语', '面部表情', '强直_脖子', '强直_上肢：右', '强直_上肢：左', '强直_下肢：右', '强直_下肢：左',
    '手指拍打_右', '手指拍打_左', '手掌运动_右', '手掌运动_左', '前臂回旋_右', '前臂回旋_左',
    '脚趾拍地_右', '脚趾拍地_左', '脚灵敏度_右', '脚灵敏度_左', '起立', '步态', '步态冻结_步态冻结的评估',
    '姿势平稳度', '姿势', '全身自发性的动作评估_身体动作迟缓', '姿态性震颤_右', '姿态性震颤_左',
    '动作性震颤_右', '动作性震颤_左', '静止型震颤_上肢：右', '静止型震颤_上肢：左', '静止型震颤_下肢：右',
    '静止型震颤_下肢：左', '静止型震颤_嘴唇/下巴', '静止型震颤_持续性'
]

# ======================
# 超参数配置
# ======================
HYPERPARAMS = {
    'batch_size': 8,
    'learning_rate': 0.0005,
    'num_epochs': 200,
    'hidden_dim': 512,
    'num_blocks': 6,
    'dropout': 0.3,
    'feature_type': 'ALL',  # 'DIFF', 'UPDRS', or 'ALL'
    # 回归任务相关参数
    'normalize_target': True,  # 是否归一化目标值到0-1
    'target_min': 0.0,         # 改善率最小值（用于归一化）
    'target_max': 1.0,         # 改善率最大值（用于归一化，如果改善率是百分比如0.5表示50%，则max=1.0）
}

logger.info(f"Hyperparameters: {json.dumps(HYPERPARAMS, indent=2)}")

# ======================
# 辅助函数：构建diff文件索引
# ======================
def build_diff_index(base_dir):
    """构建diff文件索引字典"""
    diff_index = {}
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
        file_prefix = f"patient_{folder_name}"
        
        anomaly_file = os.path.join(patient_subfolder, f"{file_prefix}_mean_anomaly.npy")
        distortion_file = os.path.join(patient_subfolder, f"{file_prefix}_mean_distortion.npy")
        
        if not os.path.exists(anomaly_file):
            logger.warning(f"Anomaly file not found: {anomaly_file}")
            continue
        if not os.path.exists(distortion_file):
            logger.warning(f"Distortion file not found: {distortion_file}")
            continue
        
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
# 解析UPDRS数据 - 回归版本
# ======================
def parse_updrs_data(csv_path, diff_index):
    """
    解析UPDRS CSV数据，标签为改善率（连续值）
    """
    df = pd.read_csv(csv_path)
    
    required_cols = ['ID', '评估时间', '手术情况', 'UPDRS-III改善率'] + CLINICAL_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in CSV: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df.sort_values(by=['ID', '评估时间'])
    
    samples = []
    improvement_rates = []  # 收集所有改善率用于统计
    
    for pid, group in df.groupby('ID'):
        pid_lower = pid.strip().lower()
        
        if pid_lower not in diff_index:
            logger.warning(f"No diff records found for patient {pid}")
            continue
        
        if len(group) < 2:
            logger.warning(f"Patient {pid} has only {len(group)} records, skipping")
            continue
        
        off_row = group.iloc[0]
        
        if off_row['手术情况'] != 'DBS off':
            logger.warning(f"First record for {pid} is not 'DBS off', skipping")
            continue
        
        # 获取改善率（连续值标签）
        if pd.isna(off_row['UPDRS-III改善率']):
            logger.warning(f"Missing improvement rate for {pid}, skipping")
            continue
        
        improvement_rate = float(off_row['UPDRS-III改善率'])
        improvement_rates.append(improvement_rate)
        
        # 提取临床特征
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
        
        anomaly_file = diff_index[pid_lower]['anomaly']
        distortion_file = diff_index[pid_lower]['distortion']
        
        samples.append({
            'id': pid,
            'anomaly_file': anomaly_file,
            'distortion_file': distortion_file,
            'clinical_features': np.array(clinical_features, dtype=np.float32),
            'improvement_rate': improvement_rate,  # 连续值标签
            'updrs_total_off': off_row.get('UPDRS总分', None)
        })
    
    # 打印改善率统计信息
    if improvement_rates:
        logger.info(f"Improvement rate statistics:")
        logger.info(f"  Min: {min(improvement_rates):.4f}")
        logger.info(f"  Max: {max(improvement_rates):.4f}")
        logger.info(f"  Mean: {np.mean(improvement_rates):.4f}")
        logger.info(f"  Std: {np.std(improvement_rates):.4f}")
    
    logger.info(f"Found {len(samples)} valid samples")
    return samples

# ======================
# 数据准备 - 回归版本
# ======================
def prepare_data(samples):
    """
    准备特征和标签（回归任务）
    """
    X_diff_list = []
    X_updrs_list = []
    y_list = []
    meta_list = []
    
    feature_type = HYPERPARAMS['feature_type'].upper()
    
    for sample in samples:
        try:
            anomaly = np.load(sample['anomaly_file']).flatten().astype(np.float32)
            distortion = np.load(sample['distortion_file']).flatten().astype(np.float32)
            diff_combined = np.concatenate([anomaly, distortion])
        except Exception as e:
            logger.error(f"Error loading diff files for {sample['id']}: {str(e)}")
            continue
        
        clinical_vec = sample['clinical_features']
        
        X_diff_list.append(diff_combined)
        X_updrs_list.append(clinical_vec)
        y_list.append(sample['improvement_rate'])  # 连续值标签
        meta_list.append(sample)
    
    X_diff = np.array(X_diff_list, dtype=np.float32)
    X_updrs = np.array(X_updrs_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)  # 改为float32
    
    # 归一化目标值到0-1（如果需要）
    if HYPERPARAMS['normalize_target']:
        y_min = HYPERPARAMS['target_min']
        y_max = HYPERPARAMS['target_max']
        # 裁剪到指定范围
        y = np.clip(y, y_min, y_max)
        # 归一化到0-1
        y = (y - y_min) / (y_max - y_min + 1e-8)
        logger.info(f"Normalized target values to [0, 1], original range: [{y_min}, {y_max}]")
    
    if feature_type == 'DIFF':
        X = X_diff
    elif feature_type == 'UPDRS':
        X = X_updrs
    elif feature_type == 'ALL':
        X = np.concatenate([X_diff, X_updrs], axis=1)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target range: [{y.min():.4f}, {y.max():.4f}], Mean: {y.mean():.4f}")
    return X, y, meta_list

# ======================
# Dataset - 回归版本
# ======================
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  # 改为float
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ======================
# 模型定义 - 回归版本
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

class RegressionMLP(nn.Module):
    """回归模型，输出归一化到0-1的预测值"""
    def __init__(self, input_dim, hidden_dim=512, num_blocks=4, dropout=0.3):
        super(RegressionMLP, self).__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        
        # 回归输出头
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 保持Sigmoid，输出0-1
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_head(x)
        return x.squeeze(-1)

# ======================
# 评估指标计算
# ======================
def compute_regression_metrics(y_true, y_pred):
    """计算回归评估指标"""
    metrics = {}
    
    # MSE & RMSE
    mse = mean_squared_error(y_true, y_pred)
    metrics['mse'] = mse
    metrics['rmse'] = np.sqrt(mse)
    
    # MAE
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # R² (决定系数)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Pearson相关系数
    if len(y_true) > 2:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        metrics['pearson_r'] = pearson_r
        metrics['pearson_p'] = pearson_p
    else:
        metrics['pearson_r'] = float('nan')
        metrics['pearson_p'] = float('nan')
    
    # Spearman相关系数
    if len(y_true) > 2:
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        metrics['spearman_r'] = spearman_r
        metrics['spearman_p'] = spearman_p
    else:
        metrics['spearman_r'] = float('nan')
        metrics['spearman_p'] = float('nan')
    
    return metrics


def evaluate_model(model, data_loader, device):
    """评估模型并返回预测值和真实值"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


# ======================
# 交叉验证实验 - 回归版本（修改版：记录测试集最佳时的in-sample指标）
# ======================
def cross_validate_experiment(X, y, meta, experiment_name):
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Regression Cross-Validation: {experiment_name}")
    logger.info(f"Total samples: {len(X)} | Target range: [{y.min():.4f}, {y.max():.4f}]")
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    # Out-of-sample (测试集) 指标
    fold_metrics = {
        'mse': [], 'rmse': [], 'mae': [], 'r2': [],
        'pearson_r': [], 'pearson_p': [], 'spearman_r': [], 'spearman_p': []
    }
    
    # In-sample (训练集) 指标 - 在测试集最佳epoch时的值
    fold_insample_at_best = {
        'mse': [], 'rmse': [], 'mae': [], 'r2': [],
        'pearson_r': [], 'pearson_p': [], 'spearman_r': [], 'spearman_p': [],
        'best_epoch': []  # 记录每个fold的best epoch
    }
    
    input_dim = X.shape[1]
    
    PATIENCE = 50
    MIN_EPOCHS = 50
    BEST_MODEL_PATH = "best_model_temp.pth"
    
    batch_size = HYPERPARAMS['batch_size']
    lr = HYPERPARAMS['learning_rate']
    num_epochs = HYPERPARAMS['num_epochs']
    hidden_dim = HYPERPARAMS['hidden_dim']
    num_blocks = HYPERPARAMS['num_blocks']
    dropout = HYPERPARAMS['dropout']
    
    all_predictions = []
    all_insample_predictions = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        logger.info(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        logger.info(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
        logger.info(f"Train target - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
        logger.info(f"Test target  - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
        
        train_dataset = RegressionDataset(X_train, y_train)
        test_dataset = RegressionDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # 用于评估的train loader（不shuffle）
        train_loader_eval = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        model = RegressionMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout
        ).to(device)
        
        criterion = HybridRegressionLoss(alpha=0.1, beta=0.9)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = 0
        best_insample_metrics = None  # 记录测试集最佳时的in-sample指标
        
        logger.info(f"Training fold {fold+1} with early stopping (patience={PATIENCE})...")
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 计算验证集loss
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels)
                    val_loss += batch_loss.item() * inputs.size(0)
            
            val_loss = val_loss / len(test_loader.dataset)
            
            if epoch >= MIN_EPOCHS:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                    
                    # ========== 关键修改：在测试集最佳时，计算并保存in-sample指标 ==========
                    train_preds, train_labels = evaluate_model(model, train_loader_eval, device)
                    best_insample_metrics = compute_regression_metrics(train_labels, train_preds)
                    best_insample_metrics['predictions'] = train_preds.copy()
                    best_insample_metrics['labels'] = train_labels.copy()
                    # =====================================================================
                    
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= PATIENCE:
                        logger.info(f"Early stopping at epoch {epoch+1}! "
                                    f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
                        break
            
            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
                # 显示当前in-sample spearman（用于监控过拟合）
                current_train_preds, current_train_labels = evaluate_model(model, train_loader_eval, device)
                current_insample_metrics = compute_regression_metrics(current_train_labels, current_train_preds)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                            f"Train Loss: {epoch_loss/len(train_loader):.6f} | "
                            f"Val Loss: {val_loss:.6f} | "
                            f"Train Spearman: {current_insample_metrics['spearman_r']:.4f} | "
                            f"Best Val: {best_val_loss:.6f} (ep {best_epoch+1})")
        
        # 加载最佳模型
        if os.path.exists(BEST_MODEL_PATH):
            model.load_state_dict(torch.load(BEST_MODEL_PATH))
            os.remove(BEST_MODEL_PATH)
            logger.info(f"Loaded best model from epoch {best_epoch+1}")
        
        # ============ 评估 Out-of-sample (Test) - 使用最佳模型 ============
        test_preds, test_labels = evaluate_model(model, test_loader, device)
        test_metrics = compute_regression_metrics(test_labels, test_preds)
        
        for key in fold_metrics.keys():
            if key in test_metrics:
                fold_metrics[key].append(test_metrics[key])
        
        # 保存测试集预测结果
        fold_meta = [meta[i] for i in test_idx]
        result_df = pd.DataFrame(fold_meta)
        result_df['true_value'] = test_labels
        result_df['pred_value'] = test_preds
        result_df['residual'] = test_labels - test_preds
        result_df['best_epoch'] = best_epoch + 1
        result_df.to_csv(f"{experiment_name}_fold{fold+1}_test_predictions.csv", index=False)
        
        for i, idx in enumerate(test_idx):
            all_predictions.append({
                'id': meta[idx]['id'],
                'fold': fold + 1,
                'true_value': test_labels[i],
                'pred_value': test_preds[i]
            })
        
        logger.info(f"Fold {fold+1} TEST Results (best epoch {best_epoch+1}) - "
                    f"MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}, "
                    f"R²: {test_metrics['r2']:.4f}, "
                    f"Pearson r: {test_metrics['pearson_r']:.4f} (p={test_metrics['pearson_p']:.4e}), "
                    f"Spearman r: {test_metrics['spearman_r']:.4f} (p={test_metrics['spearman_p']:.4e})")
        
        # ============ 记录测试集最佳时的 In-sample 指标 ============
        if best_insample_metrics is not None:
            for key in fold_insample_at_best.keys():
                if key == 'best_epoch':
                    fold_insample_at_best[key].append(best_epoch + 1)
                elif key in best_insample_metrics:
                    fold_insample_at_best[key].append(best_insample_metrics[key])
            
            # 保存in-sample预测结果（测试集最佳时的）
            train_meta = [meta[i] for i in train_idx]
            insample_df = pd.DataFrame(train_meta)
            insample_df['true_value'] = best_insample_metrics['labels']
            insample_df['pred_value'] = best_insample_metrics['predictions']
            insample_df['residual'] = best_insample_metrics['labels'] - best_insample_metrics['predictions']
            insample_df['best_epoch'] = best_epoch + 1
            insample_df.to_csv(f"{experiment_name}_fold{fold+1}_insample_at_best_predictions.csv", index=False)
            
            # 收集所有in-sample预测
            for i, idx in enumerate(train_idx):
                all_insample_predictions.append({
                    'id': meta[idx]['id'],
                    'fold': fold + 1,
                    'best_epoch': best_epoch + 1,
                    'true_value': best_insample_metrics['labels'][i],
                    'pred_value': best_insample_metrics['predictions'][i]
                })
            
            logger.info(f"Fold {fold+1} IN-SAMPLE @ best epoch {best_epoch+1} - "
                        f"MSE: {best_insample_metrics['mse']:.4f}, MAE: {best_insample_metrics['mae']:.4f}, "
                        f"R²: {best_insample_metrics['r2']:.4f}, "
                        f"Pearson r: {best_insample_metrics['pearson_r']:.4f} (p={best_insample_metrics['pearson_p']:.4e}), "
                        f"Spearman r: {best_insample_metrics['spearman_r']:.4f} (p={best_insample_metrics['spearman_p']:.4e})")
        
        # 计算过拟合差距
        overfit_gap = best_insample_metrics['spearman_r'] - test_metrics['spearman_r'] if best_insample_metrics else float('nan')
        logger.info(f"Fold {fold+1} Overfit Gap (In-sample - Test Spearman): {overfit_gap:.4f}")
    
    # ============ 汇总结果 ============
    
    # 汇总out-of-sample结果
    test_summary = {}
    for metric, values in fold_metrics.items():
        values = np.array(values)
        valid_vals = values[~np.isnan(values)]
        if len(valid_vals) > 0:
            mean_val = np.mean(valid_vals)
            std_val = np.std(valid_vals) if len(valid_vals) > 1 else 0.0
        else:
            mean_val, std_val = float('nan'), float('nan')
        test_summary[metric] = (mean_val, std_val)
    
    # 汇总in-sample at best结果
    insample_at_best_summary = {}
    for metric, values in fold_insample_at_best.items():
        if metric == 'best_epoch':
            insample_at_best_summary[metric] = values  # 保留原始列表
            continue
        values = np.array(values)
        valid_vals = values[~np.isnan(values)]
        if len(valid_vals) > 0:
            mean_val = np.mean(valid_vals)
            std_val = np.std(valid_vals) if len(valid_vals) > 1 else 0.0
        else:
            mean_val, std_val = float('nan'), float('nan')
        insample_at_best_summary[metric] = (mean_val, std_val)
    
    # 计算平均过拟合差距
    insample_spearman = np.array(fold_insample_at_best['spearman_r'])
    test_spearman = np.array(fold_metrics['spearman_r'])
    overfit_gaps = insample_spearman - test_spearman
    mean_overfit_gap = np.nanmean(overfit_gaps)
    std_overfit_gap = np.nanstd(overfit_gaps)
    
    # 保存所有预测结果
    all_pred_df = pd.DataFrame(all_predictions)
    all_pred_df.to_csv(f"{experiment_name}_all_test_predictions.csv", index=False)
    
    all_insample_pred_df = pd.DataFrame(all_insample_predictions)
    all_insample_pred_df.to_csv(f"{experiment_name}_all_insample_at_best_predictions.csv", index=False)
    
    # 打印最终结果
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL CROSS-VALIDATION RESULTS FOR: {experiment_name}")
    logger.info("-" * 40)
    logger.info("OUT-OF-SAMPLE (Test) Metrics:")
    for metric, (mean, std) in test_summary.items():
        if np.isnan(mean):
            logger.info(f"  {metric.upper():<12}: N/A")
        else:
            logger.info(f"  {metric.upper():<12}: {mean:.4f} ± {std:.4f}")
    
    logger.info("-" * 40)
    logger.info("IN-SAMPLE @ Test-Best-Epoch Metrics:")
    for metric, value in insample_at_best_summary.items():
        if metric == 'best_epoch':
            logger.info(f"  Best Epochs: {value}")
        elif np.isnan(value[0]):
            logger.info(f"  {metric.upper():<12}: N/A")
        else:
            logger.info(f"  {metric.upper():<12}: {value[0]:.4f} ± {value[1]:.4f}")
    
    logger.info("-" * 40)
    logger.info(f"OVERFIT GAP (In-sample - Test Spearman): {mean_overfit_gap:.4f} ± {std_overfit_gap:.4f}")
    logger.info(f"  Per-fold gaps: {[f'{g:.4f}' for g in overfit_gaps]}")
    logger.info(f"{'='*60}\n")
    
    return test_summary, insample_at_best_summary, {
        'mean_overfit_gap': mean_overfit_gap,
        'std_overfit_gap': std_overfit_gap,
        'per_fold_gaps': overfit_gaps.tolist()
    }

# ======================
# 主函数
# ======================
def main():
    start_time = time.time()
    logger.info("Starting treatment response REGRESSION pipeline")
    
    logger.info("Building diff file index...")
    diff_index = build_diff_index(BASE_DIR)
    
    logger.info("Parsing UPDRS data...")
    samples = parse_updrs_data(UPDRS_CSV_PATH, diff_index)
    
    if len(samples) == 0:
        logger.error("No valid samples found. Exiting.")
        return
    
    logger.info("Preparing feature matrices...")
    X, y, meta = prepare_data(samples)
    
    experiment_name = f"Treatment_Regression_{HYPERPARAMS['feature_type']}"
    test_summary, insample_summary, overfit_info = cross_validate_experiment(X, y, meta, experiment_name)
    
    result_file = f"final_results_{experiment_name}.json"
    with open(result_file, 'w') as f:
        # 处理insample_summary中的best_epoch列表
        insample_summary_for_json = {}
        for k, v in insample_summary.items():
            if k == 'best_epoch':
                insample_summary_for_json[k] = [int(e) for e in v]
            else:
                insample_summary_for_json[k] = [float(val) for val in v]
        
        json.dump({
            'out_of_sample_summary': {k: [float(v) for v in vs] for k, vs in test_summary.items()},
            'in_sample_at_best_summary': insample_summary_for_json,
            'overfit_analysis': {
                'mean_gap': float(overfit_info['mean_overfit_gap']),
                'std_gap': float(overfit_info['std_overfit_gap']),
                'per_fold_gaps': [float(g) for g in overfit_info['per_fold_gaps']]
            },
            'hyperparameters': HYPERPARAMS,
            'total_samples': int(len(X)),
            'target_stats': {
                'min': float(y.min()),
                'max': float(y.max()),
                'mean': float(y.mean()),
                'std': float(y.std())
            }
        }, f, indent=2)
    
    logger.info(f"Results saved to {result_file}")
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()