import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress, pearsonr
import seaborn as sns

# ============================================
# 第一部分：读取5个fold的数据并计算平均R和P
# ============================================

# 假设你的5个CSV文件名为：fold_0.csv, fold_1.csv, ..., fold_4.csv
fold_files = ['/ailab/user/dusiyuan/code/Brain/EC/predict_exp/prognotic_pre/PP_regre/TI_Treatment_Regression_all_fold1_predictions.csv', 
              '/ailab/user/dusiyuan/code/Brain/EC/predict_exp/prognotic_pre/PP_regre/TI_Treatment_Regression_all_fold2_predictions.csv',
              '/ailab/user/dusiyuan/code/Brain/EC/predict_exp/prognotic_pre/PP_regre/TI_Treatment_Regression_all_fold3_predictions.csv',
              '/ailab/user/dusiyuan/code/Brain/EC/predict_exp/prognotic_pre/PP_regre/TI_Treatment_Regression_all_fold4_predictions.csv',
              '/ailab/user/dusiyuan/code/Brain/EC/predict_exp/prognotic_pre/PP_regre/TI_Treatment_Regression_all_fold5_predictions.csv']

# 存储每个fold的相关系数和p值
r_values = []
p_values = []

for file in fold_files:
    df = pd.read_csv(file)
    r, p = spearmanr(df['true_value'], df['pred_value'])
    r_values.append(r)
    p_values.append(p)
    print(f"{file}: R = {r:.3f}, P = {p:.4f}")

# 计算平均值
mean_r = np.mean(r_values)
mean_p = np.mean(p_values)
std_r = np.std(r_values)

print("\n" + "="*50)
print(f"5-fold CV Results:")
print(f"Mean Spearman R = {mean_r:.3f} ± {std_r:.3f}")
print(f"Mean P-value = {mean_p:.4f}")
print("="*50)

# ============================================
# 第二部分：用全部数据画示意图
# ============================================

# 读取数据用于画图
df_plot = pd.read_csv("/ailab/user/dusiyuan/code/Brain/EC/predict_exp/prognotic_pre/PP_regre/TI_Treatment_Regression_all_all_predictions.csv")

# 提取数据 - 注意：x轴为预测值，y轴为真实值
x = df_plot['pred_value'].values  # x轴：预测值
y = df_plot['true_value'].values  # y轴：实际改善率

# 计算Spearman相关（用于标注）
r_spearman, p_spearman = spearmanr(x, y)

# 线性回归（用于画拟合线）
slope, intercept, r_pearson, p_pearson, se = linregress(x, y)

# 计算R²
r_squared = r_pearson ** 2

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 画散点图
scatter = ax.scatter(x, y, c='#2E86AB', s=80, alpha=0.6, 
                    edgecolors='white', linewidth=1.5, zorder=3)

# 生成拟合线的x值
x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept

# 画拟合线
ax.plot(x_line, y_line, 'r-', linewidth=2, label='Linear fit', zorder=2)

# 计算95%置信区间（使用t分布更准确）
from scipy import stats
n = len(x)
x_mean = np.mean(x)
sxx = np.sum((x - x_mean)**2)

# 残差标准误
y_fit = slope * x + intercept
residuals = y - y_fit
residual_std = np.sqrt(np.sum(residuals**2) / (n - 2))

# 预测标准误
se_line = residual_std * np.sqrt(1/n + (x_line - x_mean)**2 / sxx)

# t值（95%置信区间）
t_val = stats.t.ppf(0.975, n - 2)
ci_upper = y_line + t_val * se_line
ci_lower = y_line - t_val * se_line

# 画置信区间
ax.fill_between(x_line, ci_lower, ci_upper, 
                alpha=0.2, color='red', label='95% CI', zorder=1)

# 设置标题和标签
ax.set_xlabel('Model Prediction', fontsize=18)
ax.set_ylabel('Improvement Rate (empirical)', fontsize=18)
ax.set_title(f'Pooled CV (R = {r_spearman:.2f}; P = {p_spearman:.4f})', 
             fontsize=18, pad=20)

# 添加图例
ax.legend(loc='best', frameon=True, fontsize=11)

# 美化坐标轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=16)

# 调整布局
plt.tight_layout()

# 保存图片
# plt.savefig('correlation_plot_with_ci.png', dpi=300, bbox_inches='tight')
plt.savefig('correlation_plot_with_ci.pdf', bbox_inches='tight')
# print("\n图片已保存：correlation_plot_with_ci.png 和 correlation_plot_with_ci.pdf")

plt.show()

# ============================================
# 第三部分：生成结果表格
# ============================================

# 创建结果DataFrame
results_df = pd.DataFrame({
    'Fold': ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', '5-fold CV (Mean)'],
    'Spearman R': r_values + [mean_r],
    'P-value': p_values + [mean_p]
})

print("\n" + "="*50)
print("Detailed Results Table:")
print("="*50)
print(results_df.to_string(index=False))

# 保存结果表格
results_df.to_csv('correlation_results.csv', index=False)
print("\n结果表格已保存：correlation_results.csv")