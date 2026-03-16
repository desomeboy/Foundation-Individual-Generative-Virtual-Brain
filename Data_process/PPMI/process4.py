import pandas as pd

# 读取CSV文件
df = pd.read_csv('/ailab/group/medai-share/syDu/Brain_EC/PPMI/Process/patients_with_fmri_and_t1.csv')  # 替换为你的文件路径

# 定义严格匹配的Resting-State fMRI模态列表
strict_rsfmri_list = [
    "rsfMRI_RL",
    "R_L_RESTING_STATE_FMRI_ep2d_fid_basic_bold",
    "rsFMRI_ep2d",
    "ep2d_bold_rest",
    "ep2d_RESTING_STATE",
    "RESTING_STATE_fMRI_FAT_SHIFT_LEFT",
    "rsfMRI",
    "rsfMRI_PA",
    "rsfMRI_R-L"
]

# 定义模糊匹配关键词（用于识别并丢弃的fMRI）
fuzzy_fMRI_keywords = ["RESTING_STATE", "bold_rest", "rsfMRI", "fMRI"]

# 定义T1相关关键词
t1_keywords = ["MPRAGE", "T1"]

# 步骤1：标记需要保留并重命名的行
df['keep'] = False
df['new_modality'] = df['modality']  # 初始化新列

# 1. 严格匹配Resting-State fMRI → 重命名为 Resting_State_fMRI
mask_strict_rsfmri = df['modality'].isin(strict_rsfmri_list)
df.loc[mask_strict_rsfmri, 'new_modality'] = 'Resting_State_fMRI'
df.loc[mask_strict_rsfmri, 'keep'] = True

# 2. 匹配T1/MPRAGE → 重命名为 MPRAGE
mask_t1 = df['modality'].str.contains('|'.join(t1_keywords), case=False, na=False)
df.loc[mask_t1, 'new_modality'] = 'MPRAGE'
df.loc[mask_t1, 'keep'] = True

# 3. 模糊匹配fMRI关键词但不在严格列表中的 → 删除（不标记keep）
mask_fuzzy_fMRI = df['modality'].str.contains('|'.join(fuzzy_fMRI_keywords), case=False, na=False)
mask_discard_fMRI = mask_fuzzy_fMRI & ~mask_strict_rsfmri
# 这些行不标记keep=True，将在下一步被过滤掉

# 步骤2：只保留标记为keep的行
df_cleaned = df[df['keep']].copy()

# 步骤3：更新modality列为标准化后的值
df_cleaned['modality'] = df_cleaned['new_modality']

# 步骤4：删除辅助列
df_cleaned = df_cleaned.drop(columns=['keep', 'new_modality'])

# 可选：重置索引
df_cleaned = df_cleaned.reset_index(drop=True)

# 保存结果
df_cleaned.to_csv('final_csv.csv', index=False)

print("预处理完成，已保存为 'preprocessed_file.csv'")