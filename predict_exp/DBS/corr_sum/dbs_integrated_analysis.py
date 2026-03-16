import os
import numpy as np
import pandas as pd
import re
from scipy.stats import pearsonr, spearmanr, linregress
from scipy.stats import t as stt
from tqdm import tqdm
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 日志配置
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dbs_integrated_analysis.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================
# 全局配置
# ======================
FMRI_CSV_DIR = "/ailab/group/medai-share/syDu/ruijin/DBS_FMRI2MNI/dbs_csv"
DBS_UPDRS_CSV = "/ailab/group/medai-share/syDu/ruijin/DBS/DBS_UPDRS.csv"

# AI模型预测文件配置
AI_MODEL_FOLD_FILES = [
    '/ailab/user/dusiyuan/code/Brain/EC/predict_exp/DBS/diff_regression/Treatment_Regression_ALL_fold1_predictions.csv', 
    '/ailab/user/dusiyuan/code/Brain/EC/predict_exp/DBS/diff_regression/Treatment_Regression_ALL_fold2_predictions.csv',
    '/ailab/user/dusiyuan/code/Brain/EC/predict_exp/DBS/diff_regression/Treatment_Regression_ALL_fold3_predictions.csv',
    '/ailab/user/dusiyuan/code/Brain/EC/predict_exp/DBS/diff_regression/Treatment_Regression_ALL_fold4_predictions.csv',
    '/ailab/user/dusiyuan/code/Brain/EC/predict_exp/DBS/diff_regression/Treatment_Regression_ALL_fold5_predictions.csv'
]
AI_MODEL_POOLED_FILE = "/ailab/user/dusiyuan/code/Brain/EC/predict_exp/DBS/diff_regression/Treatment_Regression_ALL_all_predictions.csv"

# FC ROI对配置（ROI编号从1开始，与fMRI一致）
FC_ROI_PAIRS = {
    'pair1': {
        'name': 'STN-L, GPi-L',
        'roi1': [215], 
        'roi2': [223]
    },
    'pair2': {
        'name': 'STN-L, Thalamus-L',
        'roi1': [215], 
        'roi2': [225]
    },
    'pair3': {
        'name': 'STN, M1',
        'roi1': [215, 216], 
        'roi2': [55, 56, 63, 64]
    },
    'pair4': {
        'name': 'STN, SMA',
        'roi1': [215, 216], 
        'roi2': [53, 54, 57, 58, 59, 60, 61, 62, 67, 68]
    },
}

# ======================
# 辅助函数：FC计算
# ======================
def compute_fc_between_rois(timeseries, roi1_indices, roi2_indices):
    """
    计算两组ROI之间的功能连接(FC)
    
    参数:
        timeseries: (n_timepoints, n_rois) 时间序列矩阵（ROI编号从1开始）
        roi1_indices: 第一组ROI的编号列表（从1开始）
        roi2_indices: 第二组ROI的编号列表（从1开始）
    
    返回:
        fc_value: FC相关系数（Pearson相关）
    """
    # 将ROI编号转换为CSV列索引（ROI编号从1开始，CSV列索引从0开始）
    roi1_csv_indices = [idx - 1 for idx in roi1_indices]
    roi2_csv_indices = [idx - 1 for idx in roi2_indices]
    
    # 检查索引是否在范围内
    max_idx = timeseries.shape[1] - 1
    if any(idx > max_idx for idx in roi1_csv_indices + roi2_csv_indices):
        logger.warning(f"ROI index out of range. Max available: {max_idx+1}")
        return np.nan
    
    # 提取ROI时间序列
    roi1_ts = timeseries[:, roi1_csv_indices]  # (n_timepoints, len(roi1_indices))
    roi2_ts = timeseries[:, roi2_csv_indices]  # (n_timepoints, len(roi2_indices))
    
    # 对每组ROI取平均
    roi1_mean = np.mean(roi1_ts, axis=1)  # (n_timepoints,)
    roi2_mean = np.mean(roi2_ts, axis=1)  # (n_timepoints,)
    
    # 计算Pearson相关系数
    if len(roi1_mean) > 2:
        fc_value, _ = pearsonr(roi1_mean, roi2_mean)
    else:
        fc_value = np.nan
    
    return fc_value


def id_to_filename(patient_id):
    """
    将病人ID转换为fMRI文件名
    例如: GPi02 -> gpi02, STN01 -> stn01
    
    参数:
        patient_id: 病人ID (如 'GPi02', 'STN01')
    
    返回:
        filename: fMRI文件名前缀 (如 'gpi02', 'stn01')
    """
    return patient_id


def load_fmri_timeseries(patient_id, fmri_dir):
    """
    加载fMRI时间序列数据
    
    参数:
        patient_id: 病人ID (如 'GPi02', 'STN01')
        fmri_dir: fMRI CSV文件所在目录
    
    返回:
        timeseries: (n_timepoints, n_rois) numpy数组，如果文件不存在返回None
    """
    # 将ID转换为文件名
    filename_prefix = id_to_filename(patient_id)
    csv_pattern = f"{filename_prefix}_ts_combined.csv"
    csv_path = os.path.join(fmri_dir, csv_pattern)
    
    if not os.path.exists(csv_path):
        logger.debug(f"fMRI file not found: {csv_path}")
        return None
    
    try:
        # 读取CSV文件（无表头）
        df = pd.read_csv(csv_path, header=None)
        timeseries = df.values  # (n_timepoints, n_rois)
        logger.debug(f"Loaded fMRI data: {csv_path}, shape: {timeseries.shape}")
        return timeseries
    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        return None


def compute_all_fc_features(timeseries):
    """
    计算所有定义的FC特征
    
    参数:
        timeseries: (n_timepoints, n_rois) 时间序列矩阵
    
    返回:
        fc_features: dict，包含所有FC特征
    """
    fc_features = {}
    
    for pair_name, roi_config in FC_ROI_PAIRS.items():
        roi1_indices = roi_config['roi1']
        roi2_indices = roi_config['roi2']
        
        fc_value = compute_fc_between_rois(timeseries, roi1_indices, roi2_indices)
        fc_features[pair_name] = fc_value
    
    return fc_features


# ======================
# 辅助函数：解析DBS_UPDRS CSV
# ======================
def parse_dbs_updrs_csv(csv_path):
    """
    解析DBS_UPDRS CSV数据，返回病人级别的记录
    只保留同时有DBS off/on记录、且fMRI文件存在的病人
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path, encoding='utf-8-sig')  # 处理BOM
    
    # 检查必要列
    required_cols = ['ID', '姓名', '手术情况', '靶点', 'UPDRS总分', 'UPDRS-III改善率']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"Missing columns in CSV: {missing}")
        raise ValueError(f"Missing required columns: {missing}")
    
    # 按病人ID分组
    groups = df.groupby('ID')
    patients = []
    improvement_rates = []
    
    logger.info("Parsing patient data from CSV...")
    
    for patient_id, group in tqdm(groups, desc="Processing patients"):
        # 筛选 DBS off 和 DBS on 记录
        dbs_off = group[group['手术情况'] == 'DBS off']
        dbs_on = group[group['手术情况'] == 'DBS on']
        
        if len(dbs_off) != 1 or len(dbs_on) != 1:
            logger.debug(f"[SKIP] Patient {patient_id}: Invalid DBS off/on records (off={len(dbs_off)}, on={len(dbs_on)})")
            continue
        
        off_row = dbs_off.iloc[0]
        on_row = dbs_on.iloc[0]
        
        # 检查fMRI文件是否存在
        fmri_csv_path = os.path.join(FMRI_CSV_DIR, f"{id_to_filename(patient_id)}_ts_combined.csv")
        
        if not os.path.exists(fmri_csv_path):
            logger.debug(f"[SKIP] Patient {patient_id}: Missing fMRI file")
            continue
        
        # 获取靶点
        target = off_row['靶点']
        if pd.isna(target) or not target:
            logger.debug(f"[SKIP] Patient {patient_id}: Unknown target")
            continue
        
        # 计算改善率
        pre_score = float(off_row['UPDRS总分'])
        post_score = float(on_row['UPDRS总分'])
        
        # 检查是否有预先计算的改善率
        if 'UPDRS-III改善率' in off_row.index and pd.notna(off_row['UPDRS-III改善率']):
            improvement_rate = float(off_row['UPDRS-III改善率'])
        else:
            # 手动计算改善率
            if pre_score > 0:
                improvement_rate = (pre_score - post_score) / pre_score
            else:
                improvement_rate = 0.0
        
        improvement_rates.append(improvement_rate)
        
        # 获取姓名
        name = off_row.get('姓名', 'Unknown')
        
        # 添加到患者列表
        patients.append({
            'id': patient_id,
            'name': name,
            'target': target,
            'pre_score': pre_score,
            'post_score': post_score,
            'improvement_rate': improvement_rate,
            'fmri_csv_path': fmri_csv_path,
        })
    
    # 打印改善率统计信息
    if improvement_rates:
        logger.info(f"\nImprovement rate statistics:")
        logger.info(f"  N samples: {len(improvement_rates)}")
        logger.info(f"  Min: {min(improvement_rates):.4f}")
        logger.info(f"  Max: {max(improvement_rates):.4f}")
        logger.info(f"  Mean: {np.mean(improvement_rates):.4f}")
        logger.info(f"  Std: {np.std(improvement_rates):.4f}")
        logger.info(f"  Median: {np.median(improvement_rates):.4f}")
    
    logger.info(f"\nFound {len(patients)} valid patient samples with complete data")
    return patients


# ======================
# FC相关性分析
# ======================
def analyze_fc_correlation(patients):
    """
    分析FC与UPDRS III改善率的相关性
    """
    logger.info("\n" + "="*80)
    logger.info("FC-UPDRS III Improvement Rate Correlation Analysis")
    logger.info("="*80)
    
    # 收集数据
    all_data = []
    
    logger.info("\nComputing FC features for all patients...")
    for patient in tqdm(patients, desc="Computing FC"):
        # 加载fMRI时间序列
        timeseries = load_fmri_timeseries(patient['id'], FMRI_CSV_DIR)
        
        if timeseries is None:
            logger.warning(f"Could not load fMRI for {patient['id']}, skipping")
            continue
        
        # 计算FC特征
        fc_features = compute_all_fc_features(timeseries)
        
        # 检查是否有NaN值
        if any(np.isnan(v) for v in fc_features.values()):
            logger.warning(f"NaN in FC features for {patient['id']}, skipping")
            continue
        
        all_data.append({
            'id': patient['id'],
            'name': patient['name'],
            'target': patient['target'],
            'pre_score': patient['pre_score'],
            'post_score': patient['post_score'],
            'improvement_rate': patient['improvement_rate'],
            'fc_pair1': fc_features['pair1'],
            'fc_pair2': fc_features['pair2'],
            'fc_pair3': fc_features['pair3'],
            'fc_pair4': fc_features['pair4'],
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 保存原始数据
    df.to_csv('dbs_fc_improvement_data.csv', index=False)
    logger.info(f"\nSaved raw data to dbs_fc_improvement_data.csv (N={len(df)})")
    
    if len(df) < 3:
        logger.error("Not enough valid samples for correlation analysis!")
        return None, None
    
    # 计算相关性
    logger.info("\n" + "-"*80)
    logger.info("Correlation Analysis Results:")
    logger.info("-"*80)
    
    correlation_results = {}
    
    for pair_id in ['pair1', 'pair2', 'pair3', 'pair4']:
        pair_name = FC_ROI_PAIRS[pair_id]['name']
        fc_values = df[f'fc_{pair_id}'].values
        improvement_rates = df['improvement_rate'].values
        
        # Pearson相关
        pearson_r, pearson_p = pearsonr(fc_values, improvement_rates)
        
        # Spearman相关
        spearman_r, spearman_p = spearmanr(fc_values, improvement_rates)
        
        correlation_results[pair_id] = {
            'pair_name': pair_name,
            'roi1': FC_ROI_PAIRS[pair_id]['roi1'],
            'roi2': FC_ROI_PAIRS[pair_id]['roi2'],
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'fc_mean': float(np.mean(fc_values)),
            'fc_std': float(np.std(fc_values)),
            'fc_min': float(np.min(fc_values)),
            'fc_max': float(np.max(fc_values)),
        }
        
        logger.info(f"\n{pair_id.upper()}: {pair_name}")
        logger.info(f"  ROI1: {FC_ROI_PAIRS[pair_id]['roi1']}")
        logger.info(f"  ROI2: {FC_ROI_PAIRS[pair_id]['roi2']}")
        logger.info(f"  FC range: [{np.min(fc_values):.4f}, {np.max(fc_values):.4f}]")
        logger.info(f"  FC mean ± std: {np.mean(fc_values):.4f} ± {np.std(fc_values):.4f}")
        logger.info(f"  Pearson  r = {pearson_r:7.4f}, p = {pearson_p:.4e} {'***' if pearson_p < 0.001 else '**' if pearson_p < 0.01 else '*' if pearson_p < 0.05 else 'ns'}")
        logger.info(f"  Spearman r = {spearman_r:7.4f}, p = {spearman_p:.4e} {'***' if spearman_p < 0.001 else '**' if spearman_p < 0.01 else '*' if spearman_p < 0.05 else 'ns'}")
    
    logger.info("\n" + "="*80)
    
    # 保存结果
    with open('dbs_fc_correlation_results.json', 'w') as f:
        json.dump({
            'n_samples': len(df),
            'improvement_rate_stats': {
                'mean': float(df['improvement_rate'].mean()),
                'std': float(df['improvement_rate'].std()),
                'min': float(df['improvement_rate'].min()),
                'max': float(df['improvement_rate'].max()),
                'median': float(df['improvement_rate'].median()),
            },
            'correlations': correlation_results
        }, f, indent=2)
    
    logger.info("\nSaved correlation results to dbs_fc_correlation_results.json")
    
    return df, correlation_results


# ======================
# AI模型预测分析
# ======================
def analyze_ai_model_predictions():
    """
    分析AI模型5折交叉验证的预测结果
    """
    logger.info("\n" + "="*80)
    logger.info("AI Model Prediction Analysis (5-Fold CV)")
    logger.info("="*80)
    
    # 存储每个fold的相关系数和p值（仅用于记录）
    r_values = []
    p_values = []
    fold_data = []  # 新增：存储每个fold的数据用于可视化
    
    logger.info("\nProcessing individual folds...")
    for i, file in enumerate(AI_MODEL_FOLD_FILES, 1):
        if not os.path.exists(file):
            logger.warning(f"Fold {i} file not found: {file}")
            continue
            
        df = pd.read_csv(file)
        r, p = spearmanr(df['true_value'], df['pred_value'])
        r_values.append(r)
        p_values.append(p)
        fold_data.append(df)  # 保存fold数据
        logger.info(f"Fold {i}: Spearman R = {r:.3f}, P = {p:.4f}")
    
    # 保存fold结果（可选）
    if r_values:
        fold_results = pd.DataFrame({
            'Fold': [f'Fold {i}' for i in range(1, len(r_values)+1)],
            'Spearman R': r_values,
            'P-value': p_values
        })
        fold_results.to_csv('ai_model_fold_results.csv', index=False)
        logger.info("\nSaved fold results to ai_model_fold_results.csv")
    
    # 读取pooled数据并计算统计量
    pooled_data = None
    pooled_r = None
    pooled_p = None
    
    if os.path.exists(AI_MODEL_POOLED_FILE):
        pooled_data = pd.read_csv(AI_MODEL_POOLED_FILE)
        logger.info(f"\nLoaded pooled data: {len(pooled_data)} samples")
        
        # 计算pooled数据的相关性
        pooled_r, pooled_p = spearmanr(pooled_data['true_value'], pooled_data['pred_value'])
        logger.info("\n" + "-"*80)
        logger.info(f"Pooled Data Summary:")
        logger.info(f"  Spearman R = {pooled_r:.3f}")
        logger.info(f"  P-value = {pooled_p:.4f}")
        logger.info(f"  N samples = {len(pooled_data)}")
        logger.info("-"*80)
    else:
        logger.warning(f"Pooled data file not found: {AI_MODEL_POOLED_FILE}")
    
    return {
        'r_values': r_values,  # 保留用于记录
        'p_values': p_values,  # 保留用于记录
        'fold_data': fold_data,  # 新增：返回fold数据
        'pooled_data': pooled_data,
        'pooled_r': pooled_r,
        'pooled_p': pooled_p
    }


# ======================
# 整合可视化
# ======================
def create_integrated_visualizations(fc_df, fc_correlation_results, ai_model_results):
    """
    创建整合的可视化图表，包含FC相关性和AI模型预测
    第一行：四个FC相关性图 + AI模型pooled数据图
    第二行：五折交叉验证的每一折相关性分析结果
    """
    logger.info("\nGenerating integrated visualizations...")
    
    # 设置绘图风格
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    
    # 创建2行5列的子图，每行共享纵轴
    # fig, axes = plt.subplots(2, 5, figsize=(25, 8), sharey='row')
    fig, axes = plt.subplots(1, 5, figsize=(25, 4), sharey='row')
    plt.subplots_adjust(wspace=0.1, hspace=0.3)  # 调整子图间距
    
    # 用于存储图例的handles和labels
    legend_handles = None
    legend_labels = None
    
    # ======================
    # 第一行：1-4: FC相关性图
    # ======================
    for idx, pair_id in enumerate(['pair1', 'pair2', 'pair3', 'pair4']):
        # ax = axes[0, idx]
        ax = axes[idx]
        
        pair_name = FC_ROI_PAIRS[pair_id]['name']
        fc_values = fc_df[f'fc_{pair_id}'].values
        improvement_rates = fc_df['improvement_rate'].values
        
        # 散点图
        scatter = ax.scatter(fc_values, improvement_rates, c='#0C4842', s=40, alpha=1,
                  edgecolors='none', linewidth=1.5, zorder=3, label='Data points')
        
        # 拟合线和置信区间
        z = np.polyfit(fc_values, improvement_rates, 1)
        p = np.poly1d(z)
        x_line = np.linspace(fc_values.min(), fc_values.max(), 100)
        y_pred = p(x_line)
        
        # 计算残差标准误差和置信区间
        y_fit = p(fc_values)
        residuals = improvement_rates - y_fit
        n = len(fc_values)
        
        residual_std = np.sqrt(np.sum(residuals**2) / (n - 2))
        x_mean = np.mean(fc_values)
        sxx = np.sum((fc_values - x_mean)**2)
        se_line = residual_std * np.sqrt(1/n + (x_line - x_mean)**2 / sxx)
        
        t_val = stt.ppf(0.975, n - 2)
        ci_upper = y_pred + t_val * se_line
        ci_lower = y_pred - t_val * se_line
        
        # 绘制拟合线
        line = ax.plot(x_line, y_pred, "k-", alpha=0.8, linewidth=2.5, 
                label='Linear fit', zorder=2)[0]
        
        # 绘制置信区间
        fill = ax.fill_between(x_line, ci_lower, ci_upper, 
                        color='gray', alpha=0.15, 
                        label='95% CI', zorder=1)
        
        # 获取相关性数据
        spearman_r = fc_correlation_results[pair_id]['spearman_r']
        spearman_p = fc_correlation_results[pair_id]['spearman_p']
        
        # 只在第一个子图收集图例信息
        if idx == 0:
            legend_handles = [scatter, line, fill]
            legend_labels = ['Data points', 'Linear fit', '95% CI']
        
        # 设置标题（显示Spearman R和p值）
        ax.set_title(r'$\rho$'f' = {spearman_r:.3f}, p = {spearman_p:.3f}', 
                    fontsize=18)
        
        # 标签
        # ax.set_xlabel(f'{pair_name} Connectivity\nR = {spearman_r:.3f}, p = {spearman_p:.3f}', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{pair_name} Connectivity', fontsize=14, fontweight='bold')
        if idx == 0:  # 只在第一个子图显示y轴标签
            ax.set_ylabel('Improvement (Empirical)', fontsize=14, fontweight='bold')
        
        ax.grid(False)
        ax.tick_params(labelsize=12)
        
        # 美化坐标轴
        if idx in [0,1]:
            pcolor = '#2B5F75'
        elif idx in [2,3]:
            pcolor = '#1B813E'
            
        ax.spines['bottom'].set_color(pcolor)  # 或其他颜色
        ax.spines['left'].set_color(pcolor)
        ax.spines['top'].set_color(pcolor)
        ax.spines['right'].set_color(pcolor)
        for spine in ax.spines.values():
            spine.set_linewidth(3)  
    
    # ======================
    # 第一行：5: AI模型pooled数据图
    # ======================
    # ax_ai = axes[0, 4]
    ax_ai = axes[4]
    
    if ai_model_results['pooled_data'] is not None:
        pooled_df = ai_model_results['pooled_data']
        x = pooled_df['pred_value'].values  # x轴：预测值
        y = pooled_df['true_value'].values  # y轴：实际值
        
        # 计算相关性
        r_spearman, p_spearman = spearmanr(y, x)
        r_pearson, p_pearson = pearsonr(y, x)
        
        # 线性回归
        slope, intercept, _, _, se = linregress(x, y)
        
        # 散点图
        ax_ai.scatter(x, y, c='#0C4842', s=40, alpha=1,
                     edgecolors='none', linewidth=1.5, zorder=3)
        
        # 生成拟合线
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        
        # 计算95%置信区间
        n = len(x)
        x_mean = np.mean(x)
        ssx = np.sum((x - x_mean)**2)
        
        y_fit = slope * x + intercept
        residuals = y - y_fit
        residual_std = np.sqrt(np.sum(residuals**2) / (n - 2))
        
        se_line = residual_std * np.sqrt(1/n + (x_line - x_mean)**2 / ssx)
        
        t_val = stt.ppf(0.975, n - 2)
        ci_upper = y_line + t_val * se_line
        ci_lower = y_line - t_val * se_line
        
        # 绘制拟合线
        ax_ai.plot(x_line, y_line, 'k-', linewidth=2.5, label='Linear fit', zorder=2)
        
        # 绘制置信区间
        ax_ai.fill_between(x_line, ci_lower, ci_upper, 
                          alpha=0.15, color='gray', label='95% CI', zorder=1)
        
        # 设置标题（显示Spearman R和p值）
        ax_ai.set_title(r'$\boldsymbol{\rho}$'f' = {r_spearman:.3f}, p = {p_spearman:.2e}', 
                       fontsize=18, fontweight='bold')
        
        # 标签
        # ax_ai.set_xlabel(f'iVB Prediction(Pooled)\nR = {r_spearman:.3f}, p = {p_spearman:.2e}', fontsize=14, fontweight='bold')
        ax_ai.set_xlabel(f'iVB Model(Pooled)', fontsize=14, fontweight='bold')
        ax_ai.grid(False)
        ax_ai.tick_params(labelsize=12)
        
        # 美化坐标轴
        pcolor = '#D9AB42'
        ax_ai.spines['bottom'].set_color(pcolor)  # 或其他颜色
        ax_ai.spines['left'].set_color(pcolor)
        ax_ai.spines['top'].set_color(pcolor)
        ax_ai.spines['right'].set_color(pcolor)
        for spine in ax_ai.spines.values():
            spine.set_linewidth(3) 
    else:
        ax_ai.text(0.5, 0.5, 'AI Model Data\nNot Available', 
                  ha='center', va='center', fontsize=14, color='red')
        ax_ai.set_xticks([])
        ax_ai.set_title('iVB Model', fontsize=12, fontweight='bold', pad=10)
    
    # ======================
    # 第二行：五折交叉验证结果
    # ======================
    # fold_data = ai_model_results.get('fold_data', [])
    # r_values = ai_model_results.get('r_values', [])
    # p_values = ai_model_results.get('p_values', [])
    
    # for fold_idx in range(5):
    #     ax_fold = axes[1, fold_idx]
        
    #     if fold_idx < len(fold_data):
    #         fold_df = fold_data[fold_idx]
    #         x = fold_df['pred_value'].values
    #         y = fold_df['true_value'].values
            
    #         # 计算相关性
    #         r_spearman = r_values[fold_idx]
    #         p_spearman = p_values[fold_idx]
            
    #         # 线性回归
    #         slope, intercept, _, _, se = linregress(x, y)
            
    #         # 散点图
    #         ax_fold.scatter(x, y, c='#2E86AB', s=40, alpha=0.6,
    #                       edgecolors='none', linewidth=1.5, zorder=3)
            
    #         # 生成拟合线
    #         x_line = np.linspace(x.min(), x.max(), 100)
    #         y_line = slope * x_line + intercept
            
    #         # 计算95%置信区间
    #         n = len(x)
    #         x_mean = np.mean(x)
    #         ssx = np.sum((x - x_mean)**2)
            
    #         y_fit = slope * x + intercept
    #         residuals = y - y_fit
    #         residual_std = np.sqrt(np.sum(residuals**2) / (n - 2))
            
    #         se_line = residual_std * np.sqrt(1/n + (x_line - x_mean)**2 / ssx)
            
    #         t_val = stt.ppf(0.975, n - 2)
    #         ci_upper = y_line + t_val * se_line
    #         ci_lower = y_line - t_val * se_line
            
    #         # 绘制拟合线
    #         ax_fold.plot(x_line, y_line, 'k-', linewidth=2, zorder=2)
            

    #         # 绘制置信区间
    #         ax_fold.fill_between(x_line, ci_lower, ci_upper, 
    #                            alpha=0.15, color='gray', zorder=1)
            

    #         # 设置标题
    #         # ax_fold.set_title(f'R = {r_spearman:.3f}, p = {p_spearman:.3f}', 
    #         #                 fontsize=14, fontweight='bold', pad=10)
            
    #         # 标签
    #         ax_fold.set_xlabel(f'iVB Prediction (Fold {fold_idx+1})\nR = {r_spearman:.3f}, p = {p_spearman:.3f}', 
    #                          fontsize=14, fontweight='bold')
    #         if fold_idx == 0:
    #             ax_fold.set_ylabel('Improvement (Empirical)', 
    #                              fontsize=14, fontweight='bold')
            
    #     else:
    #         # 如果没有数据，显示占位文本
    #         ax_fold.text(0.5, 0.5, f'Fold {fold_idx+1}\nData Not Available', 
    #                     ha='center', va='center', fontsize=14, color='red')
    #         ax_fold.set_xticks([])
    #         ax_fold.set_yticks([])
    #         ax_fold.set_title(f'Fold {fold_idx+1}', fontsize=12, fontweight='bold', pad=10)
        
    #     ax_fold.grid(False)
    #     ax_fold.tick_params(labelsize=12)
        
    #     # 美化坐标轴
    #     ax_fold.spines['top'].set_visible(False)
    #     ax_fold.spines['right'].set_visible(False)
    
    # ======================
    # 在图的底部中央添加统一的图例（如果需要）
    # ======================
    # if legend_handles is not None:
    #     fig.legend(legend_handles, legend_labels, 
    #               loc='lower center', 
    #               bbox_to_anchor=(0.5, -0.05),
    #               ncol=3, 
    #               fontsize=11,
    #               frameon=True,
    #               fancybox=True,
    #               shadow=True)
    
    # 保存图片
    plt.savefig('dbs_integrated_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('dbs_integrated_analysis.pdf', dpi=300, bbox_inches='tight')
    logger.info("Saved integrated visualization to dbs_integrated_analysis.png and .pdf")
    plt.close()
# ======================
# 分组分析（按靶点）
# ======================
def analyze_by_target(df):
    """
    按靶点分组进行相关性分析
    """
    logger.info("\n" + "="*80)
    logger.info("Target-Specific Correlation Analysis")
    logger.info("="*80)
    
    target_results = {}
    
    for target in df['target'].unique():
        target_df = df[df['target'] == target]
        
        if len(target_df) < 3:
            logger.info(f"\n{target}: Insufficient samples (N={len(target_df)}), skipping")
            continue
        
        logger.info(f"\n{target} (N={len(target_df)}):")
        logger.info("-" * 40)
        
        target_correlations = {}
        
        for pair_id in ['pair1', 'pair2', 'pair3', 'pair4']:
            fc_values = target_df[f'fc_{pair_id}'].values
            improvement_rates = target_df['improvement_rate'].values
            
            pearson_r, pearson_p = pearsonr(fc_values, improvement_rates)
            spearman_r, spearman_p = spearmanr(fc_values, improvement_rates)
            
            target_correlations[pair_id] = {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p),
            }
            
            logger.info(f"  {pair_id}: Pearson r={pearson_r:.3f} (p={pearson_p:.3e}), "
                       f"Spearman r={spearman_r:.3f} (p={spearman_p:.3e})")
        
        target_results[target] = {
            'n_samples': len(target_df),
            'correlations': target_correlations
        }
    
    # 保存结果
    with open('dbs_fc_correlation_by_target.json', 'w') as f:
        json.dump(target_results, f, indent=2)
    
    logger.info("\nSaved target-specific results to dbs_fc_correlation_by_target.json")
    
    return target_results


# ======================
# 主函数
# ======================
def main():
    logger.info("="*80)
    logger.info("Starting Integrated DBS Analysis (FC + AI Model)")
    logger.info("="*80)
    
    # 1. 解析CSV数据
    logger.info("\n[Step 1] Parsing DBS_UPDRS CSV data...")
    patients = parse_dbs_updrs_csv(DBS_UPDRS_CSV)
    
    if not patients:
        logger.error("No valid patients found. Exiting.")
        return
    
    # 2. 计算FC并分析相关性
    logger.info("\n[Step 2] Computing FC and analyzing correlations...")
    fc_df, fc_correlation_results = analyze_fc_correlation(patients)
    
    if fc_df is None:
        logger.error("FC analysis failed. Exiting.")
        return
    
    # 3. 分析AI模型预测
    logger.info("\n[Step 3] Analyzing AI model predictions...")
    ai_model_results = analyze_ai_model_predictions()
    
    # 4. 创建整合可视化
    logger.info("\n[Step 4] Creating integrated visualizations...")
    create_integrated_visualizations(fc_df, fc_correlation_results, ai_model_results)
    
    # 5. 按靶点分组分析
    logger.info("\n[Step 5] Analyzing by target...")
    target_results = analyze_by_target(fc_df)
    
    # 6. 总结
    logger.info("\n" + "="*80)
    logger.info("Analysis Complete!")
    logger.info("="*80)
    logger.info("\nGenerated files:")
    logger.info("  - dbs_fc_improvement_data.csv          : Raw FC and improvement rate data")
    logger.info("  - dbs_fc_correlation_results.json      : FC correlation statistics")
    logger.info("  - dbs_fc_correlation_by_target.json    : Target-specific FC correlation")
    logger.info("  - ai_model_fold_results.csv            : AI model 5-fold CV results")
    logger.info("  - dbs_integrated_analysis.png/pdf      : Integrated visualization (FC + AI)")
    logger.info("  - dbs_integrated_analysis.log          : Detailed log file")
    logger.info("\n" + "="*80)


if __name__ == "__main__":
    main()
