import pandas as pd

# 读取数据
df = pd.read_csv('/ailab/group/medai-share/syDu/Brain_EC/PPMI/Process/dataset_index1.csv')  # 替换为你的文件路径

# 定义 fMRI 关键词（不区分大小写匹配）
fMRI_keywords = ["RESTING_STATE", "bold_rest", "rsfMRI", "fMRI"]
fmri_pattern = '|'.join(fMRI_keywords)

# 步骤1：标记哪些行是 fMRI
df['is_fmri'] = df['modality'].str.contains(fmri_pattern, case=False, na=False)

# 步骤2：找出所有有 fMRI 的患者
fmri_patients = df[df['is_fmri']]['patient_id'].unique()

# 步骤3：对这些患者，检查是否拥有至少一个 非-fMRI 的记录
valid_patients = []
for pid in fmri_patients:
    patient_rows = df[df['patient_id'] == pid]
    # 如果存在至少一行不是 fMRI → 合格
    if patient_rows[~patient_rows['is_fmri']].shape[0] > 0:
        valid_patients.append(pid)

# 步骤4：保留这些合格患者的所有行
result_df = df[df['patient_id'].isin(valid_patients)].copy()

# 可选：删除临时列
result_df = result_df.drop(columns=['is_fmri'])

# 保存结果
result_df.to_csv('patients_with_fmri_and_t1.csv', index=False)

# 输出统计
print(f"总患者数: {df['patient_id'].nunique()}")
print(f"有fMRI的患者数: {len(fmri_patients)}")
print(f"同时有fMRI+其他模态的患者数: {len(valid_patients)}")
print(f"最终保留行数: {len(result_df)}")