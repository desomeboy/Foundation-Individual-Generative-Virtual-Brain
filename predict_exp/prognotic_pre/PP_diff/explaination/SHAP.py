#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, average_precision_score,
    roc_curve
)
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm  # 新增tqdm用于进度显示

# =========================================================
# 日志
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_and_shap.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ShapModelWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        out = self.base_model(x)
        # base_model 可能输出 (B,)；GradientExplainer 需要 (B,1)
        if out.dim() == 1:
            out = out.unsqueeze(1)
        return out


# =========================================================
# 全局配置
# =========================================================
# 新数据集路径配置
BASE_DIR_DIFF = "/ailab/group/medai-share/syDu/ruijin/Final_data/AAL3_VTB"
EXCEL_PATH = "/ailab/user/dusiyuan/code/Brain/EC/ruijin/TI_gt_1_10.xlsx"
AAL_INDEX2REGION_JSON = "/ailab/group/medai-share/syDu/Brain_EC/AAL_atlas/index2region.json"

RANDOM_SEED = 20
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using device: {device}")

# 临床子项列名（不包括总分）
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

# 仅用于画图显示名（与原始代码保持一致）
CLINICAL_COLS_EN = [
    "Speech",
    "Facial expression",
    "Rigidity (neck)",
    "Rigidity (right upper limb)",
    "Rigidity (left upper limb)",
    "Rigidity (right lower limb)",
    "Rigidity (left lower limb)",
    "Finger tapping (right)",
    "Finger tapping (left)",
    "Hand movements (right)",
    "Hand movements (left)",
    "Pronation-supination (right)",
    "Pronation-supination (left)",
    "Toe tapping (right)",
    "Toe tapping (left)",
    "Leg agility (right)",
    "Leg agility (left)",
    "Arising from chair",
    "Gait",
    "Freezing of gait",
    "Postural stability",
    "Posture",
    "Global spontaneity of movement (body bradykinesia)",
    "Postural tremor (right)",
    "Postural tremor (left)",
    "Kinetic tremor (right)",
    "Kinetic tremor (left)",
    "Rest tremor (right upper limb)",
    "Rest tremor (left upper limb)",
    "Rest tremor (right lower limb)",
    "Rest tremor (left lower limb)",
    "Rest tremor (lip/jaw)",
    "Rest tremor (constancy)"
]
UPDRS_NAME_MAP = {zh: f"UPDRS-{en}" for zh, en in zip(CLINICAL_COLS, CLINICAL_COLS_EN)}

# 靶点位置映射
TARGET_MAP = {
    'L_STN': 'L_stn',
    'L_Gpi': 'L_gpi',
    'R_STN': 'R_stn',
    'R_Gpi': 'R_gpi',
    'L_GPi': 'L_gpi',
    'R_GPi': 'R_gpi',    
}

# 超参数配置
HYPERPARAMS = {
    "batch_size": 8,
    "learning_rate": 0.005,
    "num_epochs": 200,
    "hidden_dim": 512,
    "num_blocks": 4,
    "dropout": 0.1,
    "feature_type": "DIFF",  # DIFF / UPDRS / ALL
    "zscore_normalize": False,  # 新增z-score归一化选项
    "response_threshold": 5,  # 新增响应阈值
    # 训练设置
    "val_ratio": 0.2,
    "early_stop_patience": 150,
    "early_stop_min_epochs": 1,
}

logger.info(f"Hyperparameters: {json.dumps(HYPERPARAMS, indent=2, ensure_ascii=False)}")

# =========================================================
# 新数据集辅助函数
# =========================================================
def clean_check_name(check_val):
    """清理检查名称，使其成为有效的文件名"""
    clean = re.sub(r'\s+', '_', str(check_val).strip())
    clean = re.sub(r'[^\w\-_\.]', '_', clean)
    return clean

def build_diff_paths(clean_check):
    """构建diff文件路径"""
    patient_dir = os.path.join(BASE_DIR_DIFF, clean_check, 'fine_tune', clean_check)
    anomaly = os.path.join(patient_dir, f"patient_{clean_check}_mean_anomaly.npy")
    distortion = os.path.join(patient_dir, f"patient_{clean_check}_mean_distortion.npy")
    return anomaly, distortion

def zscore_normalize(arr):
    """对数组进行z-score归一化"""
    mean = np.mean(arr)
    std = np.std(arr)
    if std > 1e-8:
        return (arr - mean) / std
    else:
        return arr - mean

# =========================================================
# 新数据集解析函数
# =========================================================
def parse_excel_data(excel_path):
    """解析Excel数据，返回病人级别的记录"""
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

def prepare_response_data(patients, feature_type='ALL', zscore_normalize=False, response_threshold=5):
    """准备治疗响应预测数据"""
    X_diff, X_clinical, y, meta = [], [], [], []
    
    # 映射feature_type到新数据集类型
    sample_type_map = {
        "ALL": "all",
        "DIFF": "diff",
        "UPDRS": "clinical"
    }
    sample_type = sample_type_map.get(feature_type.upper(), "all")
    
    logger.info(f"Preparing features and labels (sample_type={sample_type}, zscore_normalize={zscore_normalize})...")
    for patient in tqdm(patients, desc="Loading diff features"):
        try:
            anomaly = np.load(patient['anomaly_path']).flatten()
            distortion = np.load(patient['distortion_path']).flatten()
        except Exception as e:
            logger.warning(f"Error loading diff files for {patient['check_name']}: {e}")
            continue
            
        if anomaly.shape[0] != 166 or distortion.shape[0] != 166:
            logger.warning(f"Invalid shape for {patient['check_name']}: "
                          f"anomaly={anomaly.shape}, distortion={distortion.shape}")
            continue
        
        # 应用z-score归一化
        if zscore_normalize:
            anomaly = zscore_normalize(anomaly)
            distortion = zscore_normalize(distortion)
        
        clinical_vec = patient['clinical_features'].astype(np.float32) / 4.0
        
        # 计算响应标签: 总分下降 >= 阈值
        response = 1 if (patient['pre_score'] - patient['post_score']) >= response_threshold else 0
        
        # 根据sample_type准备特征
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
            'id': patient['id_date'],
            'clinical_features': clinical_vec,
            'label': response,
            'pre_score': patient['pre_score'],
            'post_score': patient['post_score'],
            'check_name': patient['check_name']
        })
    
    if len(X_diff) == 0:
        logger.error("No valid samples after processing.")
        return np.array([]), np.array([]), []
    
    X = np.array(X_diff, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    logger.info(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    return X, y, meta

# =========================================================
# Dataset
# =========================================================
class ECMapDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================
# 模型（与原始代码一致）
# =========================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class DeeperMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_blocks=6, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_head(x)
        return x.squeeze(-1)


# =========================================================
# 训练（与原始代码一致）
# =========================================================
def train_final_model(X, y, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y,
        test_size=float(HYPERPARAMS["val_ratio"]),
        random_state=RANDOM_SEED,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    train_loader = DataLoader(ECMapDataset(X_tr, y_tr), batch_size=int(HYPERPARAMS["batch_size"]), shuffle=True)
    val_loader = DataLoader(ECMapDataset(X_va, y_va), batch_size=int(HYPERPARAMS["batch_size"]), shuffle=False)

    model = DeeperMLP(
        input_dim=X.shape[1],
        hidden_dim=int(HYPERPARAMS["hidden_dim"]),
        num_blocks=int(HYPERPARAMS["num_blocks"]),
        dropout=float(HYPERPARAMS["dropout"])
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(HYPERPARAMS["learning_rate"]))

    patience = int(HYPERPARAMS["early_stop_patience"])
    min_epochs = int(HYPERPARAMS["early_stop_min_epochs"])
    num_epochs = int(HYPERPARAMS["num_epochs"])
    best_path = os.path.join(out_dir, "best_model.pth")

    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0

    logger.info(f"Start training final model. Train={len(X_tr)}, Val={len(X_va)}")

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            prob = model(xb)
            loss = criterion(prob, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * xb.size(0)

        train_loss = train_loss_sum / max(1, len(train_loader.dataset))

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                prob = model(xb)
                loss = criterion(prob, yb)
                val_loss_sum += float(loss.item()) * xb.size(0)

        val_loss = val_loss_sum / max(1, len(val_loader.dataset))

        # early stop
        if epoch >= min_epochs:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve = 0
                torch.save(model.state_dict(), best_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}, best epoch={best_epoch+1}, best val loss={best_val_loss:.6f}")
                    break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} | train loss={train_loss:.6f} | val loss={val_loss:.6f} | best={best_val_loss:.6f}")

    # load best
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        logger.info(f"Loaded best model from {best_path} (epoch {best_epoch+1})")

    # 保存最终模型（带时间戳）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(out_dir, f"final_model_{ts}.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final model to: {final_path}")

    # 在val上给个指标（便于 sanity check）
    model.eval()
    with torch.no_grad():
        probs = []
        labels = []
        for xb, yb in val_loader:
            xb = xb.to(device)
            p = model(xb).detach().cpu().numpy()
            probs.append(p)
            labels.append(yb.numpy())
    probs = np.concatenate(probs) if len(probs) else np.array([])
    labels = np.concatenate(labels) if len(labels) else np.array([])

    metrics = {}
    if len(labels) > 0:
        # youden阈值
        if len(np.unique(labels)) > 1:
            fpr, tpr, thr = roc_curve(labels, probs)
            j = tpr - fpr
            best_thr = float(thr[int(np.argmax(j))])
        else:
            best_thr = 0.5

        pred = (probs >= best_thr).astype(int)

        metrics["val_threshold"] = best_thr
        metrics["val_acc"] = float(accuracy_score(labels, pred))
        metrics["val_f1"] = float(f1_score(labels, pred, zero_division=0))
        metrics["val_precision"] = float(precision_score(labels, pred, zero_division=0))
        metrics["val_recall"] = float(recall_score(labels, pred, zero_division=0))
        if len(np.unique(labels)) > 1:
            metrics["val_auc"] = float(roc_auc_score(labels, probs))
            metrics["val_ap"] = float(average_precision_score(labels, probs))
        else:
            metrics["val_auc"] = float("nan")
            metrics["val_ap"] = float("nan")

        logger.info(f"Validation metrics: {json.dumps(metrics, ensure_ascii=False, indent=2)}")

    with open(os.path.join(out_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "hyperparameters": HYPERPARAMS,
            "val_metrics": metrics,
            "n_total": int(len(X)),
            "class_distribution": np.bincount(y).tolist() if len(y) else [],
        }, f, ensure_ascii=False, indent=2)

    return model


# =========================================================
# AAL region names (166)
# =========================================================
def load_aal_region_names(index2region_json_path: str, expected_n: int = 166):
    with open(index2region_json_path, "r", encoding="utf-8") as f:
        idx2reg = json.load(f)
    return [idx2reg[str(i)] for i in range(expected_n)]


# =========================================================
# DIFF SHAP聚合（与原始代码一致）
# =========================================================
def aggregate_diff_shap_and_x(shap_values: np.ndarray, X: np.ndarray, n_regions=166):
    n_diff_total = 2 * n_regions
    if shap_values.shape[1] < n_diff_total:
        raise ValueError(f"Expect at least {n_diff_total} dims for DIFF, got {shap_values.shape[1]}")

    shap_diff = shap_values[:, :n_diff_total]
    X_diff = X[:, :n_diff_total]

    shap_region = shap_diff[:, :n_regions] + shap_diff[:, n_regions:n_diff_total]
    X_region = X_diff[:, :n_regions] + X_diff[:, n_regions:n_diff_total]

    if shap_values.shape[1] > n_diff_total:
        shap_updrs = shap_values[:, n_diff_total:]
        X_updrs = X[:, n_diff_total:]
        shap_agg = np.concatenate([shap_region, shap_updrs], axis=1)
        X_agg = np.concatenate([X_region, X_updrs], axis=1)
    else:
        shap_agg = shap_region
        X_agg = X_region

    return shap_agg, X_agg


# =========================================================
# SHAP分析（与原始代码一致，只做微小适配）
# =========================================================
def run_shap(model: nn.Module, X: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ft = HYPERPARAMS["feature_type"].upper()

    # 背景数据 & 解释数据（控制规模，避免太慢）
    bg_size = min(128, len(X))
    ex_size = min(512, len(X))
    
    if bg_size < 1 or ex_size < 1:
        logger.error("Not enough samples for SHAP analysis")
        return
        
    bg_idx = np.random.choice(len(X), size=bg_size, replace=False)
    ex_idx = np.random.choice(len(X), size=ex_size, replace=False)

    background = torch.tensor(X[bg_idx], dtype=torch.float32).to(device)
    X_explain = X[ex_idx]
    X_explain_t = torch.tensor(X_explain, dtype=torch.float32).to(device)

    model.eval()
    shap_model = ShapModelWrapper(model).to(device)
    shap_model.eval()

    explainer = shap.GradientExplainer(shap_model, background)
    shap_values = explainer.shap_values(X_explain_t)

    # 统一成 numpy (N, D)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if torch.is_tensor(shap_values):
        shap_values = shap_values.detach().cpu().numpy()
    else:
        shap_values = np.array(shap_values)

    # 保险：处理可能的 (N, D, 1) 形状
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]

    # ---------- 特征名 + 聚合 ----------
    if ft == "UPDRS":
        feature_names = [UPDRS_NAME_MAP[c] for c in CLINICAL_COLS]
        shap_plot_values = shap_values
        X_plot = X_explain

    elif ft == "DIFF":
        region_names = load_aal_region_names(AAL_INDEX2REGION_JSON, expected_n=166)
        shap_plot_values, X_plot = aggregate_diff_shap_and_x(shap_values, X_explain, n_regions=166)
        feature_names = region_names

    elif ft == "ALL":
        region_names = load_aal_region_names(AAL_INDEX2REGION_JSON, expected_n=166)
        updrs_names = [UPDRS_NAME_MAP[c] for c in CLINICAL_COLS]
        shap_plot_values, X_plot = aggregate_diff_shap_and_x(shap_values, X_explain, n_regions=166)
        feature_names = region_names + updrs_names

    else:
        raise ValueError("feature_type must be one of DIFF/UPDRS/ALL")

    shap_plot_values = np.asarray(shap_plot_values)
    X_plot = np.asarray(X_plot)

    # 再保险：聚合后也可能出现 (N, D, 1)
    if shap_plot_values.ndim == 3 and shap_plot_values.shape[-1] == 1:
        shap_plot_values = shap_plot_values[..., 0]

    assert shap_plot_values.shape[1] == len(feature_names), \
        f"Feature name length mismatch: shap D={shap_plot_values.shape[1]}, names={len(feature_names)}"

    # ---------- 画图：Top30 + 全部 ----------
    max_all = len(feature_names)

    # Top30
    # plt.figure()
    # shap.summary_plot(shap_plot_values, X_plot, feature_names=feature_names, show=False, max_display=30)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, f"shap_{ft}_beeswarm_top30.png"), dpi=300)
    # plt.close()

    # plt.figure()
    # shap.summary_plot(shap_plot_values, X_plot, feature_names=feature_names, show=False, plot_type="bar", max_display=30)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, f"shap_{ft}_bar_top30.png"), dpi=300)
    # plt.close()

    # # 全部特征
    # plt.figure()
    # shap.summary_plot(shap_plot_values, X_plot, feature_names=feature_names, show=False, max_display=max_all)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, f"shap_{ft}_beeswarm_ALL.png"), dpi=300)
    # plt.close()

    # plt.figure()
    # shap.summary_plot(shap_plot_values, X_plot, feature_names=feature_names, show=False, plot_type="bar", max_display=max_all)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, f"shap_{ft}_bar_ALL.png"), dpi=300)
    # plt.close()

    # ---------- ALL 模式：单独把 UPDRS 画出来 ----------
    if ft in  ["DIFF","ALL"]:
        n_regions = 166
        updrs_names = [UPDRS_NAME_MAP[c] for c in CLINICAL_COLS]

        shap_updrs = shap_plot_values[:, n_regions:]
        X_updrs = X_plot[:, n_regions:]

        # plt.figure()
        # shap.summary_plot(shap_updrs, X_updrs, feature_names=updrs_names, show=False, max_display=len(updrs_names))
        # plt.tight_layout()
        # plt.savefig(os.path.join(out_dir, "shap_UPDRS_only_beeswarm_ALL.png"), dpi=300)
        # plt.close()

        # plt.figure()
        # shap.summary_plot(shap_updrs, X_updrs, feature_names=updrs_names, show=False, plot_type="bar", max_display=len(updrs_names))
        # plt.tight_layout()
        # plt.savefig(os.path.join(out_dir, "shap_UPDRS_only_bar_ALL.png"), dpi=300)
        # plt.close()
        
        # ---------- 单独把 AAL 脑区（region）部分画出来 ----------
        region_names = load_aal_region_names(AAL_INDEX2REGION_JSON, expected_n=166)
        shap_region = shap_plot_values[:, :n_regions]
        X_region = X_plot[:, :n_regions]

        # plt.figure()
        # shap.summary_plot(shap_region, X_region, feature_names=region_names, show=False, max_display=n_regions)
        # plt.tight_layout()
        # plt.savefig(os.path.join(out_dir, "shap_REGION_only_beeswarm_ALL.png"), dpi=300)
        # plt.close()

        # plt.figure()
        # shap.summary_plot(shap_region, X_region, feature_names=region_names, show=False, plot_type="bar", max_display=n_regions)
        # plt.tight_layout()
        # plt.savefig(os.path.join(out_dir, "shap_REGION_only_bar_ALL.png"), dpi=300)
        # plt.close()
        
        # ========== 新增：仅绘制帕金森关键脑区中的前15个（按SHAP重要性排序） ==========
        try:
            # 读取帕金森关键脑区表 (170个label_index, 含无效区)
            parkinson_df = pd.read_csv("/ailab/user/dusiyuan/code/Brain/PD_brain_map/1mm_map/parkinson_disease_by_AAL3.csv")

            # 生成有效脑区索引 (跳过0,35,36,81,82)
            valid_labels = sorted(set(range(1, 171)) - {0,35,36,81,82})  # 166个有效label_index
            parkinson_df = parkinson_df[parkinson_df['label_index'].isin(valid_labels)]

            # 按病理重要性排序，取Top30
            top30_df = parkinson_df.sort_values('abs_mean_z', ascending=False).head(30)
            top30_labels = top30_df['label_index'].values  # 帕金森关键脑区label_index

            # 构建CBM特征索引映射
            cbm_index_map = {label: idx for idx, label in enumerate(valid_labels)}
            top30_cbm_indices = [cbm_index_map[label] for label in top30_labels]  
            
            # 从当前 shap_region 和 X_region 中提取这30个脑区
            shap_region_top30 = shap_region[:, top30_cbm_indices]  # (N, 30)
            X_region_top30 = X_region[:, top30_cbm_indices]        # (N, 30)
            region_names_top30 = [region_names[i] for i in top30_cbm_indices]
            
            # 在这30个中，按 SHAP 绝对值均值排序，取前15
            mean_abs_top30 = np.mean(np.abs(shap_region_top30), axis=0)  # (30,)
            top15_idx_in30 = np.argsort(-mean_abs_top30)[:15]           # 前15的索引（在30内的位置）
            shap_region_top15 = shap_region_top30[:, top15_idx_in30]
            X_region_top15 = X_region_top30[:, top15_idx_in30]
            region_names_top15 = [region_names_top30[i] for i in top15_idx_in30]
 
            # 计算每个脑区的平均绝对SHAP值（特征重要性）和平均SHAP值（影响方向）
            mean_abs_shap = np.mean(np.abs(shap_region_top15), axis=0)
            mean_shap = np.mean(shap_region_top15, axis=0)

            # 创建DataFrame
            shap_df = pd.DataFrame({
                'Region_Name': region_names_top15,
                'Mean_Abs_SHAP': mean_abs_shap,      # 特征重要性（用于排序）
                'Mean_SHAP': mean_shap,              # 平均影响方向（正/负）
                'SHAP_STD': np.std(shap_region_top15, axis=0)  # 可选：SHAP值标准差
            })

            # 按平均绝对SHAP值降序排序
            shap_df = shap_df.sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

            # 保存为CSV
            csv_path = os.path.join(out_dir, "shap_REGION_ParkinsonTop30_SHAPTop15.csv")
            shap_df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig支持中文脑区名

            print(f"SHAP统计结果已保存至: {csv_path}")
            print("\nTop 5 重要脑区:")
            print(shap_df[['Region_Name', 'Mean_Abs_SHAP', 'Mean_SHAP']].head())
 
            
            # 帕金森 Top30 中 SHAP 前15 的图
            plt.figure(figsize=(3, 12))  # 宽 x 高（单位：英寸）
            shap.summary_plot(shap_region_top15, X_region_top15, feature_names=region_names_top15, show=False, max_display=15)
            plt.xlabel("SHAP value", fontsize=17)  # 自定义 x 轴标签并增大字体
            plt.yticks(fontsize=16)  # ←←← 增大 y 轴标签字体
            cbar = plt.gcf().axes[-1]  # SHAP summary_plot 将 colorbar 放在最后一个 axes
            cbar.tick_params(labelsize=14)  # 调整 colorbar 刻度字体
            cbar.set_ylabel("Feature value", fontsize=16, labelpad=12)  # 竖向 colorbar 常用
            cbar.yaxis.label.set_size(16)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "shap_REGION_ParkinsonTop30_SHAPTop15_beeswarm.pdf"), dpi=300)
            plt.close()

            plt.figure()
            shap.summary_plot(shap_region_top15, X_region_top15, feature_names=region_names_top15, show=False, plot_type="bar", max_display=15)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "shap_REGION_ParkinsonTop30_SHAPTop15_bar.png"), dpi=300)
            plt.close()


        except Exception as e:
            logger.warning(f"Error generating Parkinson-specific SHAP plots: {str(e)}")

    # ---------- 导出：全特征重要性 ----------
    mean_abs_all = np.mean(np.abs(shap_plot_values), axis=0)
    df_all = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_all})
    df_all = df_all.sort_values("mean_abs_shap", ascending=False)
    df_all.to_csv(os.path.join(out_dir, f"feature_importance_{ft}_ALLFEATURES.csv"), index=False)

    # 导出脑区重要性
    if ft in ("DIFF", "ALL"):
        region_names = load_aal_region_names(AAL_INDEX2REGION_JSON, expected_n=166)
        mean_abs_region = np.mean(np.abs(shap_plot_values[:, :166]), axis=0)
        df_region = pd.DataFrame({"region": region_names, "mean_abs_shap": mean_abs_region})
        df_region = df_region.sort_values("mean_abs_shap", ascending=False)
        df_region.to_csv(os.path.join(out_dir, f"diff_region_importance_{ft}.csv"), index=False)

    # 导出UPDRS重要性
    if ft == "ALL":
        updrs_names = [UPDRS_NAME_MAP[c] for c in CLINICAL_COLS]
        mean_abs_updrs = np.mean(np.abs(shap_plot_values[:, 166:]), axis=0)
        df_updrs = pd.DataFrame({"updrs_item": updrs_names, "mean_abs_shap": mean_abs_updrs})
        df_updrs = df_updrs.sort_values("mean_abs_shap", ascending=False)
        df_updrs.to_csv(os.path.join(out_dir, "updrs_importance_ALL.csv"), index=False)

    logger.info(f"SHAP outputs saved to: {out_dir}")


# =========================================================
# main
# =========================================================
def main():
    start = time.time()
    out_root = "results"
    os.makedirs(out_root, exist_ok=True)

    logger.info("1) Parsing Excel data ...")
    patients = parse_excel_data(EXCEL_PATH)
    if len(patients) == 0:
        logger.error("No valid patients found. Exit.")
        return

    logger.info("2) Preparing data matrix ...")
    X, y, meta = prepare_response_data(
        patients,
        feature_type=HYPERPARAMS["feature_type"],
        zscore_normalize=HYPERPARAMS["zscore_normalize"],
        response_threshold=HYPERPARAMS["response_threshold"]
    )
    if len(X) == 0:
        logger.error("Empty X after preparation. Exit.")
        return

    logger.info("3) Training final model on all data (with train/val split) ...")
    model = train_final_model(X, y, out_dir=os.path.join(out_root, "model"))

    logger.info("4) Running SHAP ...")
    run_shap(model, X, out_dir=os.path.join(out_root, "shap"))

    logger.info(f"Done. Total time: {time.time() - start:.2f}s")
    logger.info(f"Output dir: {out_root}")


if __name__ == "__main__":
    main()