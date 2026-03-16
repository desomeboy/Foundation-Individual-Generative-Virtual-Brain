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
# 全局配置（按你原始脚本默认值）
# =========================================================
BASE_DIR = "/ailab/group/medai-share/syDu/ruijin/DBS/AAL_VTB"
UPDRS_CSV_PATH = "/ailab/group/medai-share/syDu/ruijin/DBS/DBS_UPDRS.csv"
AAL_INDEX2REGION_JSON = "/ailab/group/medai-share/syDu/Brain_EC/AAL_atlas/index2region.json"

RANDOM_SEED = 17
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

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

# 仅用于画图显示名（读取仍使用中文列名）
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

# 超参数（你可按需改）
HYPERPARAMS = {
    "batch_size": 8,
    "learning_rate": 0.0005,
    "num_epochs": 200,
    "hidden_dim": 512,
    "num_blocks": 6,
    "dropout": 0.3,
    "feature_type": "ALL",  # DIFF / UPDRS / ALL
    "improvement_threshold": 0.25,
    # 训练设置
    "val_ratio": 0.2,
    "early_stop_patience": 100,
    "early_stop_min_epochs": 100,
}

logger.info(f"Hyperparameters: {json.dumps(HYPERPARAMS, indent=2, ensure_ascii=False)}")


# =========================================================
# 数据读取：diff索引 & UPDRS解析（与你原始逻辑一致）
# =========================================================
def build_diff_index(base_dir: str):
    """
    构建diff文件索引字典 (mean_anomaly & mean_distortion)
    Returns: {pid_lower: {'anomaly': path, 'distortion': path}}
    """
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

        patient_subfolder = os.path.join(folder_path, "fine_tune", folder_name)
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
            diff_index[pid] = {"anomaly": anomaly_file, "distortion": distortion_file}
        else:
            logger.warning(f"Duplicate patient ID found: {pid}, skipping additional entry")

    logger.info(f"Built diff index with {len(diff_index)} patients")
    return diff_index


def parse_updrs_data(csv_path: str, diff_index: dict):
    """
    解析UPDRS CSV数据，匹配diff文件。
    每个病人应至少两行(off/on)，取第一行(DBS off)作为特征与标签来源。
    """
    df = pd.read_csv(csv_path)

    required_cols = ["ID", "评估时间", "手术情况", "UPDRS-III改善率"] + CLINICAL_COLS
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.sort_values(by=["ID", "评估时间"])

    samples = []
    thr = float(HYPERPARAMS["improvement_threshold"])

    for pid, group in df.groupby("ID"):
        pid_lower = str(pid).strip().lower()

        if pid_lower not in diff_index:
            logger.warning(f"No diff records found for patient {pid}")
            continue

        if len(group) < 2:
            logger.warning(f"Patient {pid} has only {len(group)} records, skipping")
            continue

        off_row = group.iloc[0]
        if off_row["手术情况"] != "DBS off":
            logger.warning(f"First record for {pid} is not 'DBS off', skipping")
            continue

        if pd.isna(off_row["UPDRS-III改善率"]):
            logger.warning(f"Missing improvement rate for {pid}, skipping")
            continue

        label = 1 if float(off_row["UPDRS-III改善率"]) >= thr else 0

        clinical_features = []
        bad = False
        for col in CLINICAL_COLS:
            val = off_row[col]
            if pd.isna(val):
                bad = True
                logger.warning(f"Missing clinical feature {col} for {pid}, skipping")
                break
            try:
                clinical_features.append(float(val))
            except (ValueError, TypeError):
                bad = True
                logger.warning(f"Non-numeric value in {col} for {pid}: {val}, skipping")
                break
        if bad:
            continue

        samples.append({
            "id": pid,
            "anomaly_file": diff_index[pid_lower]["anomaly"],
            "distortion_file": diff_index[pid_lower]["distortion"],
            "clinical_features": np.array(clinical_features, dtype=np.float32),
            "label": int(label),
            "improvement_rate": float(off_row["UPDRS-III改善率"]),
            "updrs_total_off": off_row.get("UPDRS总分", None),
        })

    logger.info(f"Found {len(samples)} valid samples")
    return samples


def prepare_data(samples):
    """
    根据feature_type返回X, y。
    DIFF: 332 (anomaly166 + distortion166)
    UPDRS: 33
    ALL: 332+33
    """
    X_diff_list, X_updrs_list, y_list, meta_list = [], [], [], []
    ft = HYPERPARAMS["feature_type"].upper()

    for s in samples:
        try:
            anomaly = np.load(s["anomaly_file"]).flatten().astype(np.float32)
            distortion = np.load(s["distortion_file"]).flatten().astype(np.float32)
            diff_combined = np.concatenate([anomaly, distortion], axis=0)  # 332
        except Exception as e:
            logger.error(f"Error loading diff files for {s['id']}: {e}")
            continue

        clinical_vec = s["clinical_features"].astype(np.float32)/4

        X_diff_list.append(diff_combined)
        X_updrs_list.append(clinical_vec)
        y_list.append(s["label"])
        meta_list.append(s)

    X_diff = np.array(X_diff_list, dtype=np.float32)
    X_updrs = np.array(X_updrs_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    if ft == "DIFF":
        X = X_diff
    elif ft == "UPDRS":
        X = X_updrs
    elif ft == "ALL":
        X = np.concatenate([X_diff, X_updrs], axis=1)
    else:
        raise ValueError("feature_type must be one of DIFF/UPDRS/ALL")

    logger.info(f"Feature matrix shape: {X.shape}, Class distribution: {np.bincount(y)}")
    return X, y, meta_list


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
# 模型（与你原始一致）
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
# 训练（全量数据 -> train/val early stop -> 最佳模型）
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
# DIFF SHAP聚合：166 + 166 -> 166（同脑区相加）
# 同时把X也做同样聚合（用于summary_plot配合显示）
# =========================================================
def aggregate_diff_shap_and_x(shap_values: np.ndarray, X: np.ndarray, n_regions=166):
    """
    shap_values: (N, D), X: (N, D)
    DIFF部分假设前332维 = anomaly(166) + distortion(166)
    返回:
      shap_agg: (N, 166 + updrs_dim)
      X_agg:    (N, 166 + updrs_dim)
    """
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




def run_shap(model: nn.Module, X: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ft = HYPERPARAMS["feature_type"].upper()

    # 背景数据 & 解释数据（控制规模，避免太慢）
    bg_size = min(128, len(X))
    ex_size = min(512, len(X))
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

    # 关键：有时是 (N, D, 1)，画图前先 squeeze
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

    # Top30（保留你原来的）
    plt.figure()
    shap.summary_plot(shap_plot_values, X_plot, feature_names=feature_names, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"shap_{ft}_beeswarm_top30.png"), dpi=300)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_plot_values, X_plot, feature_names=feature_names, show=False, plot_type="bar", max_display=30)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"shap_{ft}_bar_top30.png"), dpi=300)
    plt.close()

    # 全部特征（你要“最终结果全部显示”就看这两个）
    plt.figure()
    shap.summary_plot(shap_plot_values, X_plot, feature_names=feature_names, show=False, max_display=max_all)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"shap_{ft}_beeswarm_ALL.png"), dpi=300)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_plot_values, X_plot, feature_names=feature_names, show=False, plot_type="bar", max_display=max_all)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"shap_{ft}_bar_ALL.png"), dpi=300)
    plt.close()

    # ---------- ALL 模式：单独把 UPDRS 画出来（保证可见） ----------
    if ft == "ALL":
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
        
        # # ---------- 单独把 AAL 脑区（region）部分画出来 ----------
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
        
        # 3. 从当前 shap_region 和 X_region 中提取这30个脑区
        shap_region_top30 = shap_region[:, top30_cbm_indices]  # (N, 30)
        X_region_top30 = X_region[:, top30_cbm_indices]        # (N, 30)
        region_names_top30 = [region_names[i] for i in top30_cbm_indices]
        
        # 4. 在这30个中，按 SHAP 绝对值均值排序，取前15
        mean_abs_top30 = np.mean(np.abs(shap_region_top30), axis=0)  # (30,)
        top15_idx_in30 = np.argsort(-mean_abs_top30)[:25]           # 前15的索引（在30内的位置）
        shap_region_top15 = shap_region_top30[:, top15_idx_in30]
        X_region_top15 = X_region_top30[:, top15_idx_in30]
        region_names_top15 = [region_names_top30[i] for i in top15_idx_in30]
        
        # 计算每个脑区的平均绝对SHAP值（特征重要性）和平均SHAP值（影响方向）
        mean_abs_shap = np.mean(np.abs(shap_region_top15), axis=0)
        mean_shap = np.mean(shap_region_top15, axis=0)     
           
        # ==============================================================        
        # === 新增：帕金森 Top30 中 SHAP 前15 的图 ===
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
        
        import inspect

        # ---------- 你可以在这里调样式 ----------
        bar_color = "#8a508f"
        bar_alpha = 0.95
        xlabel_fs = 18
        bar_tick_fs = 14
        ytick_fs = 16
        bee_xlabel_fs = 18
        cbar_tick_fs = 14
        cbar_label_fs = 16

        # 画布：bee更宽 + 右侧放bar
        fig = plt.figure(figsize=(17, 7))

        # 4个区域：cbar(最左) / bee(左) / ytick-label轴(中) / bar(右)
        pos_cbar = [0.05, 0.18, 0.018, 0.64]
        pos_bee  = [0.20, 0.08, 0.77, 0.84]
        pos_lbl  = [0.87, 0.08, 0.13, 0.84]
        pos_bar  = [1.10, 0.08, 0.35, 0.84]

        ax_bee = fig.add_axes(pos_bee)
        ax_lbl = fig.add_axes(pos_lbl, sharey=ax_bee)
        ax_bar = fig.add_axes(pos_bar, sharey=ax_bee)

        # ---------- 1) beeswarm 画到 ax_bee ----------
        plt.sca(ax_bee)
        axes_before = fig.axes.copy()

        kw_bee = dict(
            feature_names=region_names_top15,
            show=False,
            max_display=15
        )
        sig = inspect.signature(shap.summary_plot).parameters
        if "ax" in sig:
            kw_bee["ax"] = ax_bee

        shap.summary_plot(shap_region_top15, X_region_top15, **kw_bee)

        # shap 自动生成的 colorbar axes：通常是新增的 axes
        new_axes = [a for a in fig.axes if a not in axes_before and a not in (ax_bee, ax_lbl, ax_bar)]
        cax = new_axes[-1] if len(new_axes) else fig.axes[-1]

        # 把 cbar 挪到最左
        cax.set_position(pos_cbar)
        cax.tick_params(labelsize=cbar_tick_fs)
        cax.set_ylabel("Feature value", fontsize=cbar_label_fs, labelpad=10)

        # beeswarm x 轴 label
        ax_bee.set_xlabel("SHAP value", fontsize=bee_xlabel_fs)
        ax_bee.tick_params(axis='x', labelsize=14)

        # ---------- 2) 从 beeswarm 轴里抓到 shap 排序后的 yticks / yticklabels ----------
        yticks = ax_bee.get_yticks()
        yticklabels = [t.get_text() for t in ax_bee.get_yticklabels()]

        # bee 左边不显示 ytick（因为要放中间）
        ax_bee.tick_params(axis='y', left=False, labelleft=False, right=False, labelright=False)

        # ---------- 3) 中间标签轴：只显示 yticks（文字在中间） ----------
        ax_lbl.set_xlim(0, 1)
        ax_lbl.set_xticks([])
        ax_lbl.set_yticks(yticks)
        ax_lbl.set_yticklabels(yticklabels, fontsize=ytick_fs)

        ax_lbl.tick_params(axis='y', length=0, pad=0)
        ax_lbl.yaxis.set_ticks_position('left')

        for t in ax_lbl.get_yticklabels():
            t.set_horizontalalignment('center')
            t.set_x(0.5)

        # 去掉边框、刻度线
        for spine in ax_lbl.spines.values():
            spine.set_visible(False)
        ax_lbl.tick_params(axis='y', length=0)
        ax_lbl.yaxis.set_ticks_position('none')

        # ---------- 4) 右侧柱状图：mean(|SHAP|)，并和 beeswarm y 对齐 ----------
        mean_abs_shap_15 = np.mean(np.abs(shap_region_top15), axis=0)
        mean_abs_map = {name: val for name, val in zip(region_names_top15, mean_abs_shap_15)}

        # 按 beeswarm 实际显示顺序对齐 bar
        bar_vals = np.array([mean_abs_map.get(name, 0.0) for name in yticklabels], dtype=float)

        ax_bar.barh(yticks, bar_vals, color=bar_color, alpha=bar_alpha)
        ax_bar.set_xlabel("mean(|SHAP value|)", fontsize=xlabel_fs)
        ax_bar.tick_params(axis='x', labelsize=bar_tick_fs)

        ax_bar.tick_params(axis='y', left=False, labelleft=False, right=False, labelright=False)
        for spine in ["top", "right", "left"]:
            ax_bar.spines[spine].set_visible(False)

        # 强制 y 范围一致
        ax_lbl.set_ylim(ax_bee.get_ylim())
        ax_bar.set_ylim(ax_bee.get_ylim())

        # ---------- 5) 保存 ----------
        plt.savefig(
            os.path.join(out_dir, "shap_REGION_ParkinsonTop30_SHAPTop15_cbar_bee_ticks_bar.pdf"),
            dpi=300, bbox_inches="tight", pad_inches=0.4
        )
        plt.close(fig)
        # ===================== 以上：完全复制你的绘图方式 =====================   
        
        

    # ---------- 导出：全特征重要性（脑区 + UPDRS 全都有） ----------
    mean_abs_all = np.mean(np.abs(shap_plot_values), axis=0)
    df_all = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_all})
    df_all = df_all.sort_values("mean_abs_shap", ascending=False)
    df_all.to_csv(os.path.join(out_dir, f"feature_importance_{ft}_ALLFEATURES.csv"), index=False)

    # 你原来只导出脑区（保留），但也建议 ALL 时同时导出 UPDRS 的 CSV
    if ft in ("DIFF", "ALL"):
        region_names = load_aal_region_names(AAL_INDEX2REGION_JSON, expected_n=166)
        mean_abs_region = np.mean(np.abs(shap_plot_values[:, :166]), axis=0)
        df_region = pd.DataFrame({"region": region_names, "mean_abs_shap": mean_abs_region})
        df_region = df_region.sort_values("mean_abs_shap", ascending=False)
        df_region.to_csv(os.path.join(out_dir, f"diff_region_importance_{ft}.csv"), index=False)

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

    logger.info("1) Building diff index ...")
    diff_index = build_diff_index(BASE_DIR)

    logger.info("2) Parsing UPDRS CSV ...")
    samples = parse_updrs_data(UPDRS_CSV_PATH, diff_index)
    if len(samples) == 0:
        logger.error("No valid samples found. Exit.")
        return

    logger.info("3) Preparing data matrix ...")
    X, y, meta = prepare_data(samples)
    if len(X) == 0:
        logger.error("Empty X after preparation. Exit.")
        return

    logger.info("4) Training final model on all data (with train/val split) ...")
    model = train_final_model(X, y, out_dir=os.path.join(out_root, "model"))

    logger.info("5) Running SHAP ...")
    run_shap(model, X, out_dir=os.path.join(out_root, "shap"))

    logger.info(f"Done. Total time: {time.time() - start:.2f}s")
    logger.info(f"Output dir: {out_root}")


if __name__ == "__main__":
    main()