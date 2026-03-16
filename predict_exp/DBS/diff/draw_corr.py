import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress
import seaborn as sns

# ============================================
# 第一部分：读取5个fold的数据并计算平均R和P
# ============================================

# 假设你的5个CSV文件名为：fold_0.csv, fold_1.csv, ..., fold_4.csv
fold_files = ['/ailab/user/dusiyuan/code/Brain/EC/predict_exp/DBS/diff/Treatment_Response_UPDRS_Thresh0.25_fold1_predictions.csv', 
              '/ailab/user/dusiyuan/code/Brain/EC/predict_exp/DBS/diff/Treatment_Response_UPDRS_Thresh0.25_fold2_predictions.csv',
              '/ailab/user/dusiyuan/code/Brain/EC/predict_exp/DBS/diff/Treatment_Response_UPDRS_Thresh0.25_fold3_predictions.csv',
              '/ailab/user/dusiyuan/code/Brain/EC/predict_exp/DBS/diff/Treatment_Response_UPDRS_Thresh0.25_fold4_predictions.csv',
              '/ailab/user/dusiyuan/code/Brain/EC/predict_exp/DBS/diff/Treatment_Response_UPDRS_Thresh0.25_fold5_predictions.csv']

# 存储每个fold的相关系数和p值
r_values = []
p_values = []

for file in fold_files:
    df = pd.read_csv(file)
    r, p = spearmanr(df['improvement_rate'], df['pred_prob'])
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
# 第二部分：用第一个fold画示意图
# ============================================

# 读取第一个fold的数据用于画图
df_plot = pd.read_csv(fold_files[0])

# 提取数据
x = df_plot['improvement_rate'].values
y = df_plot['pred_prob'].values

# 计算Spearman相关（用于标注）
r_spearman, p_spearman = spearmanr(x, y)

# 线性回归（用于画拟合线）
slope, intercept, r_pearson, p_pearson, se = linregress(x, y)

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 画散点图
scatter = ax.scatter(x, y, c='#2E86AB', s=80, alpha=0.6, edgecolors='white', linewidth=1.5)

# 生成拟合线的x值
x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept

# 画拟合线
ax.plot(x_line, y_line, 'r-', linewidth=2, label='Linear fit')

# 计算95%置信区间
n = len(x)
x_mean = np.mean(x)
ssx = np.sum((x - x_mean)**2)
se_line = se * np.sqrt(1/n + (x_line - x_mean)**2 / ssx)
margin = 1.96 * se_line  # 95% CI

# 画置信区间
ax.fill_between(x_line, y_line - margin, y_line + margin, 
                alpha=0.2, color='gray', label='95% CI')

# 设置标题和标签
ax.set_xlabel('Improvement Rate (empirical)', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted Probability (biomarker)', fontsize=14, fontweight='bold')
ax.set_title(f'In-sample (R = {r_spearman:.2f}; P = {p_spearman:.4e})', 
             fontsize=15, fontweight='bold', pad=20)

# 设置网格
ax.grid(True, linestyle='--', alpha=0.3)

# 添加图例
ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)

# 美化坐标轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=12)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('correlation_plot.png', dpi=300, bbox_inches='tight')
# plt.savefig('correlation_plot.pdf', bbox_inches='tight')
print("\n图片已保存：correlation_plot.png 和 correlation_plot.pdf")

plt.show()

# ============================================
# 第三部分：生成结果表格（类似你截图中的表格）
# ============================================

# 创建结果DataFrame
results_df = pd.DataFrame({
    'Fold': ['Fold 0', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', '5-fold CV (Mean)'],
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