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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("treatment_response_prediction.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


BASE_DIR = "./ruijin/DBS/AAL_VTB"  # The root directory of the VTB folder obtained after running train_iVB.py
UPDRS_CSV_PATH = "./ruijin/DBS/DBS_UPDRS.csv" # Clinical information corresponding to the data
RANDOM_SEED = 42
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
    'improvement_threshold': 0.25 
}

logger.info(f"Hyperparameters: {json.dumps(HYPERPARAMS, indent=2)}")


def build_diff_index(base_dir):
    """
    Build diff file index dictionary (mean_anomaly & mean_distortion)
    
    Returns:
        dict: {lowercase_id: {'anomaly': file_path, 'distortion': file_path}}
    """
    diff_index = {}
    
    # Rule: Extract patient ID from folder name (e.g., gpi02_fmri_20190901_143922_651000 -> gpi02)
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
        
        # Build file paths
        patient_subfolder = os.path.join(folder_path, 'fine_tune', folder_name)
        file_prefix = f"patient_{folder_name}"
        
        anomaly_file = os.path.join(patient_subfolder, f"{file_prefix}_mean_anomaly.npy")
        distortion_file = os.path.join(patient_subfolder, f"{file_prefix}_mean_distortion.npy")
        
        # Verify file existence
        if not os.path.exists(anomaly_file):
            logger.warning(f"Anomaly file not found: {anomaly_file}")
            continue
        if not os.path.exists(distortion_file):
            logger.warning(f"Distortion file not found: {distortion_file}")
            continue
        
        # Store in index
        if pid not in diff_index:
            diff_index[pid] = {
                'anomaly': anomaly_file,
                'distortion': distortion_file
            }
        else:
            logger.warning(f"Duplicate patient ID found: {pid}, skipping additional entry")
    
    logger.info(f"Built diff index with {len(diff_index)} patients")
    return diff_index


def parse_updrs_data(csv_path, diff_index):
    """
    Parse UPDRS CSV data and match with diff files
    
    Args:
        csv_path: Path to UPDRS CSV file
        diff_index: Dictionary of diff file index
    
    Returns:
        list: List of valid samples
    """
    df = pd.read_csv(csv_path)
    
    # Verify required columns exist
    required_cols = ['ID', '评估时间', '手术情况', 'UPDRS-III改善率'] + CLINICAL_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in CSV: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sort by ID and assessment time (ensure DBS off comes first)
    df = df.sort_values(by=['ID', '评估时间'])
    
    samples = []
    improvement_threshold = HYPERPARAMS['improvement_threshold']
    
    for pid, group in df.groupby('ID'):
        pid_lower = pid.strip().lower()
        
        # Check if diff files exist
        if pid_lower not in diff_index:
            logger.warning(f"No diff records found for patient {pid}")
            continue
        
        # Each patient should have two rows (DBS off, DBS on)
        if len(group) < 2:
            logger.warning(f"Patient {pid} has only {len(group)} records, skipping")
            continue
        
        off_row = group.iloc[0]
        
        # Verify surgery status
        if off_row['手术情况'] != 'DBS off':
            logger.warning(f"First record for {pid} is not 'DBS off', skipping")
            continue
        
        # Get improvement rate label
        if pd.isna(off_row['UPDRS-III改善率']):
            logger.warning(f"Missing improvement rate for {pid}, skipping")
            continue
        
        label = 1 if off_row['UPDRS-III改善率'] >= improvement_threshold else 0
        
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
            'label': label,
            'improvement_rate': off_row['UPDRS-III改善率'],
            'updrs_total_off': off_row.get('UPDRS总分', None)
        })
    
    logger.info(f"Found {len(samples)} valid samples")
    return samples


def prepare_data(samples):
    """
    Prepare features and labels - use diff features instead of EC
    
    Args:
        samples: List of samples
    
    Returns:
        X: Feature matrix (diff, UPDRS, or combined)
        y: Label vector
        meta: List of metadata
    """
    X_diff_list = []
    X_updrs_list = []
    y_list = []
    meta_list = []
    
    feature_type = HYPERPARAMS['feature_type'].upper()
    
    for sample in samples:
        try:
            # Load PtC feature (166x1x1 -> 166)
            anomaly = np.load(sample['anomaly_file']).flatten().astype(np.float32)
            # Load CtP feature (166x1x1 -> 166)
            distortion = np.load(sample['distortion_file']).flatten().astype(np.float32)
            
            diff_combined = np.concatenate([anomaly, distortion])
            
                
        except Exception as e:
            logger.error(f"Error loading diff files for {sample['id']}: {str(e)}")
            continue
        

        clinical_vec = sample['clinical_features']
        

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
    

    X_diff = np.array(X_diff_list, dtype=np.float32)
    X_updrs = np.array(X_updrs_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    

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
# Cross-Validation Experiment
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
    

    PATIENCE = 50  
    MIN_EPOCHS = 50  
    BEST_MODEL_PATH = "best_model_temp.pth"  

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
        

        logger.info(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
        logger.info(f"Train class balance - 0: {np.sum(y_train==0)}, 1: {np.sum(y_train==1)}")
        logger.info(f"Test class balance  - 0: {np.sum(y_test==0)}, 1: {np.sum(y_test==1)}")
        

        train_dataset = ECMapDataset(X_train, y_train)
        test_dataset = ECMapDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        

        model = DeeperMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout
        ).to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = 0
        early_stop = False
        

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
            

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device).float()
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
                        logger.info(f"Early stopping triggered at epoch {epoch+1}! "
                                    f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
                        early_stop = True
                        break
            
            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == num_epochs or early_stop:
                logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                            f"Train Loss: {epoch_loss/len(train_loader):.6f} | "
                            f"Val Loss: {val_loss:.6f} | "
                            f"Best Val Loss: {best_val_loss:.6f} (epoch {best_epoch+1})")
        

        if os.path.exists(BEST_MODEL_PATH):
            model.load_state_dict(torch.load(BEST_MODEL_PATH))
            os.remove(BEST_MODEL_PATH)  
            logger.info(f"Loaded best model from epoch {best_epoch+1}")
        else:
            logger.warning("No best model saved, using final model weights")
        
        # eval
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
        
        # Determine threshold
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
        
        # Calculate metrics
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
        
        # Record metrics
        fold_metrics['acc'].append(acc)
        fold_metrics['f1'].append(f1)
        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)
        fold_metrics['auc'].append(auc)
        fold_metrics['ap'].append(ap)
        
        # Save prediction results
        fold_meta = [meta[i] for i in test_idx]
        result_df = pd.DataFrame(fold_meta)
        result_df['true_label'] = all_labels
        result_df['pred_prob'] = all_probs
        result_df['pred_label'] = all_preds
        result_df['threshold'] = best_threshold
        result_df['best_epoch'] = best_epoch + 1 
        result_df.to_csv(f"{experiment_name}_fold{fold+1}_predictions.csv", index=False)
        
        logger.info(f"Fold {fold+1} Results - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")

    
    # Summarize results
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


def main():
    start_time = time.time()
    logger.info("Starting treatment response prediction pipeline")
    
    # 1. Build CBM file index (replaces original EC index)
    logger.info("Building CBM file index (mean_anomaly & mean_distortion)...")
    diff_index = build_diff_index(BASE_DIR) 
    
    # 2. Parse UPDRS data - use CBM index
    logger.info("Parsing UPDRS data with CBM features...")
    samples = parse_updrs_data(UPDRS_CSV_PATH, diff_index)  
    
    if len(samples) == 0:
        logger.error("No valid samples found. Exiting.")
        return
    
    # 3. Prepare data - use CBM features
    logger.info("Preparing feature matrices with CBM features...")
    X, y, meta = prepare_data(samples)
    
    # 4. Cross-validation
    experiment_name = f"Treatment_Response_{HYPERPARAMS['feature_type']}_Thresh{HYPERPARAMS['improvement_threshold']}"
    summary = cross_validate_experiment(X, y, meta, experiment_name)
    
    # 5. Save final results
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