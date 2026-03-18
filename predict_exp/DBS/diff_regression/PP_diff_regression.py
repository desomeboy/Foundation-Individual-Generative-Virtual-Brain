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
    Use differentiable soft ranking to approximate Spearman correlation coefficient
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def soft_rank(self, x):
        """Differentiable soft ranking"""
        n = x.size(0)
        x_diff = x.unsqueeze(1) - x.unsqueeze(0)  # (n, n)
        ranks = torch.sigmoid(x_diff / self.temperature).sum(dim=1) + 1
        return ranks
    
    def forward(self, pred, target):
        pred_rank = self.soft_rank(pred)
        target_rank = self.soft_rank(target)
        
        pred_centered = pred_rank - pred_rank.mean()
        target_centered = target_rank - target_rank.mean()
        
        cov = (pred_centered * target_centered).mean()
        pred_std = pred_centered.std() + 1e-8
        target_std = target_centered.std() + 1e-8
        
        spearman = cov / (pred_std * target_std)
        
        # Return negative correlation as loss (maximizing correlation = minimizing negative correlation)
        return 1 - spearman


class HybridRegressionLoss(nn.Module):
    """
    Combination of MSE + ranking loss
    alpha: weight for MSE
    beta: weight for ranking loss
    """
    def __init__(self, alpha=0.3, beta=0.7, margin=0.05, temperature=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        # self.ranking = PairwiseRankingLoss(margin=margin)
        self.soft_spearman = SoftSpearmanLoss(temperature=temperature)
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        # rank_loss = self.ranking(pred, target)
        spearman_loss = self.soft_spearman(pred, target)
        
        return self.alpha * mse_loss + self.beta * spearman_loss



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("treatment_response_regression.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


BASE_DIR = "./ruijin/DBS/AAL_VTB" # The root directory of the VTB folder obtained after running train_iVB.py
UPDRS_CSV_PATH = "./ruijin/DBS/DBS_UPDRS.csv" # Clinical information corresponding to the data
RANDOM_SEED = 17
N_SPLITS = 5
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Clinical sub-item column names 
CLINICAL_COLS = [
    '言语', '面部表情', '强直_脖子', '强直_上肢：右', '强直_上肢：左', '强直_下肢：右', '强直_下肢：左',
    '手指拍打_右', '手指拍打_左', '手掌运动_右', '手掌运动_左', '前臂回旋_右', '前臂回旋_左',
    '脚趾拍地_右', '脚趾拍地_左', '脚灵敏度_右', '脚灵敏度_左', '起立', '步态', '步态冻结_步态冻结的评估',
    '姿势平稳度', '姿势', '全身自发性的动作评估_身体动作迟缓', '姿态性震颤_右', '姿态性震颤_左',
    '动作性震颤_右', '动作性震颤_左', '静止型震颤_上肢：右', '静止型震颤_上肢：左', '静止型震颤_下肢：右',
    '静止型震颤_下肢：左', '静止型震颤_嘴唇/下巴', '静止型震颤_持续性'
]


HYPERPARAMS = {
    'batch_size': 8,
    'learning_rate': 0.0005,
    'num_epochs': 200,
    'hidden_dim': 512,
    'num_blocks': 6,
    'dropout': 0.3,
    'feature_type': 'ALL',  # 'DIFF', 'UPDRS', or 'ALL'

    'normalize_target': True,  
    'target_min': 0.0,         
    'target_max': 1.0,         
}

logger.info(f"Hyperparameters: {json.dumps(HYPERPARAMS, indent=2)}")


def build_diff_index(base_dir):
    """Build diff file index dictionary"""
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
# Parse UPDRS data - Regression version
# ======================
def parse_updrs_data(csv_path, diff_index):
    """
    Parse UPDRS CSV data, with labels as improvement rate (continuous values)
    """
    df = pd.read_csv(csv_path)
    
    required_cols = ['ID', '评估时间', '手术情况', 'UPDRS-III改善率'] + CLINICAL_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in CSV: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df.sort_values(by=['ID', '评估时间'])
    
    samples = []
    improvement_rates = []  
    
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
        
        # Get improvement rate (continuous value label)
        if pd.isna(off_row['UPDRS-III改善率']):
            logger.warning(f"Missing improvement rate for {pid}, skipping")
            continue
        
        improvement_rate = float(off_row['UPDRS-III改善率'])
        improvement_rates.append(improvement_rate)
        
        # Extract clinical features
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
            'improvement_rate': improvement_rate,  
            'updrs_total_off': off_row.get('UPDRS总分', None)
        })
    

    if improvement_rates:
        logger.info(f"Improvement rate statistics:")
        logger.info(f"  Min: {min(improvement_rates):.4f}")
        logger.info(f"  Max: {max(improvement_rates):.4f}")
        logger.info(f"  Mean: {np.mean(improvement_rates):.4f}")
        logger.info(f"  Std: {np.std(improvement_rates):.4f}")
    
    logger.info(f"Found {len(samples)} valid samples")
    return samples

# ======================
# Data preparation - Regression version
# ======================
def prepare_data(samples):
    """
    Prepare features and labels (regression task)
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
        y_list.append(sample['improvement_rate'])  
        meta_list.append(sample)
    
    X_diff = np.array(X_diff_list, dtype=np.float32)
    X_updrs = np.array(X_updrs_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) 
    
    
    if HYPERPARAMS['normalize_target']:
        y_min = HYPERPARAMS['target_min']
        y_max = HYPERPARAMS['target_max']

        y = np.clip(y, y_min, y_max)

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
# Dataset - Regression version
# ======================
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ======================
# Model definition - Regression version
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
    """Regression model, output normalized to [0, 1]"""
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
        
        # Regression output head
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
# Evaluation metrics computation
# ======================
def compute_regression_metrics(y_true, y_pred):
    """Compute regression evaluation metrics"""
    metrics = {}
    
    # MSE & RMSE
    mse = mean_squared_error(y_true, y_pred)
    metrics['mse'] = mse
    metrics['rmse'] = np.sqrt(mse)
    
    # MAE
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # R² 
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Pearson
    if len(y_true) > 2:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        metrics['pearson_r'] = pearson_r
        metrics['pearson_p'] = pearson_p
    else:
        metrics['pearson_r'] = float('nan')
        metrics['pearson_p'] = float('nan')
    
    # Spearman
    if len(y_true) > 2:
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        metrics['spearman_r'] = spearman_r
        metrics['spearman_p'] = spearman_p
    else:
        metrics['spearman_r'] = float('nan')
        metrics['spearman_p'] = float('nan')
    
    return metrics

# ======================
# Cross-validation experiment - Regression version
# ======================
def cross_validate_experiment(X, y, meta, experiment_name):
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Regression Cross-Validation: {experiment_name}")
    logger.info(f"Total samples: {len(X)} | Target range: [{y.min():.4f}, {y.max():.4f}]")
    

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_metrics = {
        'mse': [], 'rmse': [], 'mae': [], 'r2': [],
        'pearson_r': [], 'pearson_p': [], 'spearman_r': [], 'spearman_p': []
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
        

        model = RegressionMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout
        ).to(device)
        
        # criterion = nn.MSELoss()
        criterion = HybridRegressionLoss(alpha=0.1, beta=0.9)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = 0
        

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
        

        if os.path.exists(BEST_MODEL_PATH):
            model.load_state_dict(torch.load(BEST_MODEL_PATH))
            os.remove(BEST_MODEL_PATH)
            logger.info(f"Loaded best model from epoch {best_epoch+1}")
        
        # eval
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute regression metrics
        metrics = compute_regression_metrics(all_labels, all_preds)
        

        for key in fold_metrics.keys():
            if key in metrics:
                fold_metrics[key].append(metrics[key])
        
        # Save prediction results
        fold_meta = [meta[i] for i in test_idx]
        result_df = pd.DataFrame(fold_meta)
        result_df['true_value'] = all_labels
        result_df['pred_value'] = all_preds
        result_df['residual'] = all_labels - all_preds
        result_df['best_epoch'] = best_epoch + 1
        result_df.to_csv(f"{experiment_name}_fold{fold+1}_predictions.csv", index=False)
        

        for i, idx in enumerate(test_idx):
            all_predictions.append({
                'id': meta[idx]['id'],
                'fold': fold + 1,
                'true_value': all_labels[i],
                'pred_value': all_preds[i]
            })
        
        logger.info(f"Fold {fold+1} Results - MSE: {metrics['mse']:.4f}, "
                    f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}, "
                    f"Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e}), "
                    f"Spearman r: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4e})")
    

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
    
    # Save all prediction results
    all_pred_df = pd.DataFrame(all_predictions)
    all_pred_df.to_csv(f"{experiment_name}_all_predictions.csv", index=False)
    
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

def main():
    start_time = time.time()
    logger.info("Starting treatment response REGRESSION pipeline")
    

    logger.info("Building CBM file index...")
    diff_index = build_diff_index(BASE_DIR)
    

    logger.info("Parsing UPDRS data...")
    samples = parse_updrs_data(UPDRS_CSV_PATH, diff_index)
    
    if len(samples) == 0:
        logger.error("No valid samples found. Exiting.")
        return
    

    logger.info("Preparing feature matrices...")
    X, y, meta = prepare_data(samples)
    

    experiment_name = f"Treatment_Regression_{HYPERPARAMS['feature_type']}"
    summary = cross_validate_experiment(X, y, meta, experiment_name)
    

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