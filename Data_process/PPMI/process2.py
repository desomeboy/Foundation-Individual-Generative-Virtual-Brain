import pandas as pd
import re

# 读取 CSV
df = pd.read_csv('/ailab/group/medai-share/syDu/Brain_EC/PPMI/Process/dataset_index.csv')

# 定义函数：提取所有数字字符
def extract_digits(s):
    return ''.join(re.findall(r'\d', str(s)))

# 应用到 patient_id 列
df['patient_id'] = df['patient_id'].apply(extract_digits)

# 可选：去掉前导零（如果希望是整数形式）
# df['patient_id'] = df['patient_id'].str.lstrip('0').replace('', '0')  # 防止全零变空

# 保存回 CSV
df.to_csv('/ailab/group/medai-share/syDu/Brain_EC/PPMI/Process/dataset_index1.csv', index=False)