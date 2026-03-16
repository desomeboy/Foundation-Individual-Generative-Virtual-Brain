import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# 配置路径
base_path = "/ailab/user/dusiyuan/code/Brain/EC/predict_exp/DBS/diff"
experiment_prefix = "Treatment_Response_ALL_Thresh0.25"

# 存储每折的指标
metrics_per_fold = {
    'auc': [],
    'ap': [],
    'acc': [],
    'f1': [],
    'precision': [],
    'recall': []
}

# 遍历 5 折
for fold in range(1, 6):
    file_path = os.path.join(base_path, f"{experiment_prefix}_fold{fold}_predictions.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prediction file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    y_true = df['true_label'].values
    y_prob = df['pred_prob'].values
    y_pred = (y_prob >= df['threshold'].iloc[0]).astype(int)  # 使用该 fold 记录的阈值

    # 确保至少有两个类别（避免指标报错）
    if len(np.unique(y_true)) < 2:
        print(f"Warning: Fold {fold} has only one class. Skipping metric computation.")
        continue

    # 计算各项指标
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # 存入列表
    metrics_per_fold['auc'].append(auc)
    metrics_per_fold['ap'].append(ap)
    metrics_per_fold['acc'].append(acc)
    metrics_per_fold['f1'].append(f1)
    metrics_per_fold['precision'].append(prec)
    metrics_per_fold['recall'].append(rec)

    print(f"Fold {fold}: AUC={auc:.4f}, AP={ap:.4f}, ACC={acc:.4f}, F1={f1:.4f}")

# 计算均值 ± 标准差
print("\n=== Cross-Validation Summary (Mean ± Std) ===")
summary = {}
for metric, values in metrics_per_fold.items():
    if len(values) == 0:
        mean, std = float('nan'), float('nan')
    else:
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0  # 无偏标准差
    summary[metric] = (mean, std)
    print(f"{metric.upper():<10}: {mean:.4f} ± {std:.4f}")