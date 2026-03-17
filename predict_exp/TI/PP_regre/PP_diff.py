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

# ======================
# 自定义损失函数
# ======================
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
    def __init__(self, alpha=0.3, beta=0.7, temperature=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.soft_spearman = SoftSpearmanLoss(temperature=temperature)
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        spearman_loss = self.soft_spearman(pred, target)
        
        return self.alpha * mse_loss + self.beta * spearman_loss


# ======================
# 日志配置
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ti_treatment_response_regression.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================
# 全局配置
# ======================
BASE_DIR_DIFF = "/ailab/group/medai-share/syDu/ruijin/Final_data/AAL3_VTB"  # diff文件根目录
EXCEL_PATH = "/ailab/user/dusiyuan/code/Brain/EC/ruijin/TI_gt_1_10_filled.xlsx"
RANDOM_SEED = 42
N_SPLITS = 5
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")



# 临床子项列名（排除"总分"）
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
# 超参数配置
# ======================
HYPERPARAMS = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 200,
    'hidden_dim': 512,
    'num_blocks': 4,
    'dropout': 0.3,
    'feature_type': 'all',  # 'all', 'diff', or 'clinical'
    # 回归任务相关参数
    'normalize_target': True,  # 是否归一化目标值到0-1
    'target_min': 0.0,         # 改善率最小值
    'target_max': 1.0,         # 改善率最大值
    # 损失函数参数
    'loss_alpha': 0.6,         # MSE权重
    'loss_beta': 0.4,          # Spearman排序损失权重
    'loss_temperature': 0.1,   # Soft ranking温度参数
}

logger.info(f"Hyperparameters: {json.dumps(HYPERPARAMS, indent=2)}")

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

def zscore_normalize(arr):
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
    required_cols = ['Check', '总分', '靶点位置', '治疗状态', '序号', '日期', '姓名'] + CLINICAL_COLS
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"Missing columns in Excel: {missing}")
        raise ValueError(f"Missing required columns: {missing}")
    
    # 按病人+日期分组
    groups = df.groupby(['序号', '日期'])
    patients = []
    improvement_rates = []  # 收集所有改善率用于统计
    
    logger.info("Parsing patient data from Excel...")
    
    for (id_num, date), group in tqdm(groups, desc="Processing patients"):
        if len(group) != 2:
            print(f"[SKIP] Patient {id_num}_{date}: Group size != 2 ({group}size={len(group)})")
            continue
            
        pre_row = group[group['治疗状态'] == '前']
        post_row = group[group['治疗状态'] == '后']
        if len(pre_row) != 1 or len(post_row) != 1:
            print(f"[SKIP] Patient {id_num}_{date}: Invalid pre/post rows (pre={len(pre_row)}, post={len(post_row)})")
            continue
        
        pre_row = pre_row.iloc[0]
        post_row = post_row.iloc[0]
        
        if pre_row[CLINICAL_COLS].isnull().any() or post_row[CLINICAL_COLS].isnull().any():
            pre_null = pre_row[CLINICAL_COLS].isnull().sum()
            post_null = post_row[CLINICAL_COLS].isnull().sum()
            print(f"[SKIP] Patient {id_num}_{date}: Missing clinical data (pre_nulls={pre_null}, post_nulls={post_null})")
            continue
            
        clean_check_pre = clean_check_name(pre_row['Check'])
        clean_check_post = clean_check_name(post_row['Check'])     
        print(clean_check_pre)
        anomaly_path_pre, distortion_path_pre = build_diff_paths(clean_check_pre)
        anomaly_path_post, distortion_path_post = build_diff_paths(clean_check_post)
        
        if not (os.path.exists(anomaly_path_pre) and os.path.exists(distortion_path_pre) and
                os.path.exists(anomaly_path_post) and os.path.exists(distortion_path_post)):
            missing_files = []
            if not os.path.exists(anomaly_path_pre):
                missing_files.append(f"pre_anomaly")
            if not os.path.exists(distortion_path_pre):
                missing_files.append(f"pre_distortion")
            if not os.path.exists(anomaly_path_post):
                missing_files.append(f"post_anomaly")
            if not os.path.exists(distortion_path_post):
                missing_files.append(f"post_distortion")
            print(f"[SKIP] Patient {id_num}_{date}: Missing diff files - {', '.join(missing_files)}")
            logger.debug(f"Missing diff files for pre: {anomaly_path_pre}, post: {anomaly_path_post}")
            continue
            
        target_folder = pre_row['靶点位置']
        if not target_folder:
            print(f"[SKIP] Patient {id_num}_{date}: Unknown target location '{pre_row['靶点位置']}'")
            continue
        
        # 计算改善率（连续值标签）
        pre_score = int(pre_row['总分'])
        post_score = int(post_row['总分'])
        
        # 改善率 = (pre_score - post_score) / pre_score
        # 避免除零
        if pre_score > 0:
            improvement_rate = (pre_score - post_score) / pre_score
        else:
            improvement_rate = 0.0
        
        improvement_rates.append(improvement_rate)
        
        # --- 姓名和原始靶点 ---
        name = pre_row.get('姓名', 'Unknown')
        target_original = pre_row['靶点位置']
        
        # 治疗前样本
        patients.append({
            'id_date': f"{id_num}_{date}",
            'timepoint': 'pre',
            'pre_score': pre_score,
            'post_score': post_score,
            'improvement_rate': improvement_rate,  # 连续值标签
            'anomaly_path': anomaly_path_pre,
            'distortion_path': distortion_path_pre,
            'clinical_features': pre_row[CLINICAL_COLS].values.astype(np.float32),
            'target': target_folder,
            'target_original': target_original,
            'name': name,
            'check_name': clean_check_pre
        })

        # 治疗后样本
        # patients.append({
        #     'id_date': f"{id_num}_{date}",
        #     'timepoint': 'post',
        #     'pre_score': pre_score,
        #     'post_score': post_score,
        #     'improvement_rate': improvement_rate,  # 同一病人改善率相同
        #     'anomaly_path': anomaly_path_post,
        #     'distortion_path': distortion_path_post,
        #     'clinical_features': post_row[CLINICAL_COLS].values.astype(np.float32),
        #     'target': target_folder,
        #     'target_original': target_original,
        #     'name': name,
        #     'check_name': clean_check_post
        # })
    # 打印改善率统计信息
    if improvement_rates:
        logger.info(f"Improvement rate statistics:")
        logger.info(f"  Min: {min(improvement_rates):.4f}")
        logger.info(f"  Max: {max(improvement_rates):.4f}")
        logger.info(f"  Mean: {np.mean(improvement_rates):.4f}")
        logger.info(f"  Std: {np.std(improvement_rates):.4f}")
    
    logger.info(f"Found {len(patients)} valid patient samples with complete data")
    return patients

# ======================
# 数据准备函数 - 回归版本
# ======================
def prepare_response_data(patients, sample_type='all'):
    """
    准备治疗响应预测数据（回归任务）
    每个样本包含 diff + 临床特征
    标签 = 改善率（连续值）
    """
    X_list, y_list, meta = [], [], []
    
    logger.info("Preparing features and labels for regression...")
    for patient in tqdm(patients, desc="Loading diff features"):
        anomaly = np.load(patient['anomaly_path']).flatten()
        distortion = np.load(patient['distortion_path']).flatten()
        
        if anomaly.shape[0] != 166 or distortion.shape[0] != 166:
            logger.warning(f"Invalid shape for {patient['check_name']}: "
                          f"anomaly={anomaly.shape}, distortion={distortion.shape}")
            continue
        
        clinical_vec = patient['clinical_features'].astype(np.float32) / 4.0
        improvement_rate = patient['improvement_rate']  # 连续值标签
        
        if sample_type == 'all':
            features = np.concatenate([anomaly, distortion, clinical_vec])
        elif sample_type == 'diff':
            features = np.concatenate([anomaly, distortion])
        elif sample_type == 'clinical':
            features = clinical_vec
        else:
            raise ValueError("sample_type must be 'all', 'diff', or 'clinical'")
        
        X_list.append(features)
        y_list.append(improvement_rate)
        meta.append({
            **patient,
            'feature_type': sample_type
        })
    
    X = np.array(X_list, dtype=np.float32)
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
    
    logger.info(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Target range: [{y.min():.4f}, {y.max():.4f}], Mean: {y.mean():.4f}, Std: {y.std():.4f}")
    
    return X, y, meta

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
# 残差块
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

# ======================
# 回归模型
# ======================
# class RegressionMLP(nn.Module):
#     """回归模型，输出归一化到0-1的预测值"""
#     def __init__(self, input_dim, hidden_dim=512, num_blocks=4, dropout=0.3):
#         super(RegressionMLP, self).__init__()
        
#         self.input_proj = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )
        
#         self.blocks = nn.Sequential(
#             *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
#         )
        
#         # 回归输出头
#         self.output_head = nn.Sequential(
#             nn.Linear(hidden_dim, 256),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, 1),
#             nn.Sigmoid()  # 保持Sigmoid，输出0-1
#         )
    
#     def forward(self, x):
#         x = self.input_proj(x)
#         x = self.blocks(x)
#         x = self.output_head(x)
#         return x.squeeze(-1)

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
            nn.Sigmoid()
        )
        
        # 使用更稳定的初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # 减小初始化范围
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 调试：检查输入
        # print(f"Input - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
        # print(f"Input has NaN: {torch.isnan(x).any().item()}")
        
        # 逐层检查
        for i, layer in enumerate(self.input_proj):
            x = layer(x)
            # print(f"After layer {i} ({layer.__class__.__name__}) - min: {x.min().item():.4f}, max: {x.max().item():.4f}, has NaN: {torch.isnan(x).any().item()}")
            if torch.isnan(x).any():
                # print(f"NaN detected at layer {i}!")
                break
        
        if not torch.isnan(x).any():
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

# ======================
# 交叉验证函数 - 回归版本
# ======================
def cross_validate_experiment(X, y, meta, experiment_name):
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Regression Cross-Validation: {experiment_name}")
    logger.info(f"Total samples: {len(X)} | Target range: [{y.min():.4f}, {y.max():.4f}]")

    # 使用KFold而不是StratifiedKFold（回归任务）
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_metrics = {
        'mse': [], 'rmse': [], 'mae': [], 'r2': [],
        'pearson_r': [], 'pearson_p': [], 'spearman_r': [], 'spearman_p': []
    }
    
    input_dim = X.shape[1]
    
    # 早停参数
    PATIENCE = 50
    MIN_EPOCHS = 50
    BEST_MODEL_PATH = "best_model_temp.pth"
    
    # 超参数
    batch_size = HYPERPARAMS['batch_size']
    lr = HYPERPARAMS['learning_rate']
    num_epochs = HYPERPARAMS['num_epochs']
    hidden_dim = HYPERPARAMS['hidden_dim']
    num_blocks = HYPERPARAMS['num_blocks']
    dropout = HYPERPARAMS['dropout']
    
    all_fold_dfs = []  # 用于合并所有 fold
    all_predictions = []  # 收集所有预测结果

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        logger.info(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        logger.info(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
        logger.info(f"Train target - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
        logger.info(f"Test target  - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
        
        test_meta = [meta[i] for i in test_idx]

        train_dataset = RegressionDataset(X_train, y_train)
        test_dataset = RegressionDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 初始化回归模型
        model = RegressionMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout
        ).to(device)
        
        # 使用混合损失
        criterion = HybridRegressionLoss(
            alpha=HYPERPARAMS['loss_alpha'],
            beta=HYPERPARAMS['loss_beta'],
            temperature=HYPERPARAMS['loss_temperature']
        )
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 早停相关变量
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = 0
        
        # 训练循环
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
            
            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
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
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= PATIENCE:
                        logger.info(f"Early stopping at epoch {epoch+1}! "
                                    f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
                        break
            
            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
                logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                            f"Train Loss: {epoch_loss/len(train_loader):.6f} | "
                            f"Val Loss: {val_loss:.6f} | "
                            f"Best: {best_val_loss:.6f} (ep {best_epoch+1})")

        # 加载最佳模型
        if os.path.exists(BEST_MODEL_PATH):
            model.load_state_dict(torch.load(BEST_MODEL_PATH))
            os.remove(BEST_MODEL_PATH)
            logger.info(f"Loaded best model from epoch {best_epoch+1}")

        # 评估
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 计算回归指标
        metrics = compute_regression_metrics(all_labels, all_preds)
        
        # 记录指标
        for key in fold_metrics.keys():
            if key in metrics:
                fold_metrics[key].append(metrics[key])

        # 构建预测结果 DataFrame
        result_df = pd.DataFrame({
            'id_date': [m['id_date'] for m in test_meta],
            'name': [m['name'] for m in test_meta],
            'target_original': [m['target_original'] for m in test_meta],
            'target_mapped': [m['target'] for m in test_meta],
            'pre_score': [m['pre_score'] for m in test_meta],
            'post_score': [m['post_score'] for m in test_meta],
            'timepoint': [m['timepoint'] for m in test_meta],
            'true_value': all_labels,
            'pred_value': all_preds,
            'residual': all_labels - all_preds,
            'best_epoch': best_epoch + 1
        })
        result_df.to_csv(f"{experiment_name}_fold{fold+1}_predictions.csv", index=False)
        all_fold_dfs.append(result_df)

        # 收集所有预测
        for i, idx in enumerate(test_idx):
            all_predictions.append({
                'id': meta[idx]['id_date'],
                'fold': fold + 1,
                'true_value': all_labels[i],
                'pred_value': all_preds[i]
            })

        logger.info(f"Fold {fold+1} Results - MSE: {metrics['mse']:.4f}, "
                    f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}, "
                    f"Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e}), "
                    f"Spearman r: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4e})")

    # 合并所有 fold
    combined_df = pd.concat(all_fold_dfs, ignore_index=True)
    combined_df.to_csv(f"{experiment_name}_all_folds_combined.csv", index=False)
    logger.info(f"Combined results saved to {experiment_name}_all_folds_combined.csv")
    
    # 保存所有预测结果
    all_pred_df = pd.DataFrame(all_predictions)
    all_pred_df.to_csv(f"{experiment_name}_all_predictions.csv", index=False)

    # Summary
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
    logger.info("-" * 40)
    for metric, (mean, std) in summary.items():
        if np.isnan(mean):
            logger.info(f"{metric.upper():<12}: N/A")
        else:
            logger.info(f"{metric.upper():<12}: {mean:.4f} ± {std:.4f}")
    logger.info(f"{'='*60}\n")

    return summary

# ======================
# 主函数
# ======================
def main():
    start_time = time.time()
    logger.info("Starting TI treatment response REGRESSION pipeline")
    
    # 1. 解析Excel数据
    logger.info("Parsing Excel data with diff features...")
    patients = parse_excel_data(EXCEL_PATH)
    
    if not patients:
        logger.error("No valid patients found. Exiting.")
        return
    
    # 2. 准备数据
    feature_type = HYPERPARAMS['feature_type']
    X, y, meta = prepare_response_data(patients, sample_type=feature_type)
    
    if len(X) == 0:
        logger.error("No features extracted. Exiting.")
        return
    
    # 3. 交叉验证
    experiment_name = f"TI_Treatment_Regression_{feature_type}"
    summary = cross_validate_experiment(X, y, meta, experiment_name)
    
    # 4. 保存最终结果
    result_file = f"final_results_{experiment_name}.json"
    with open(result_file, 'w') as f:
        json.dump({
            'summary': {k: [float(v) for v in vs] for k, vs in summary.items()},
            'hyperparameters': HYPERPARAMS,
            'total_samples': len(X),
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