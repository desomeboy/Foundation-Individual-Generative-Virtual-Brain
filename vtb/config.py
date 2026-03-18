# npi/config.py
# 只放默认常量和路径

ROI_NUM = 166

# 数据与缓存
DEFAULT_STEPS = 7
DEFAULT_SKIP_FIRST = 30
DEFAULT_CACHE_DIR = "/ailab/user/dusiyuan/code/Brain/EC/data_cache"
DEFAULT_OUTPUT_DIR = "/ailab/user/dusiyuan/code/Brain/EC/EC_results_v3"

# 训练超参
DEFAULT_BATCH_SIZE = 768
DEFAULT_EPOCHS = 200
DEFAULT_LR = 5e-4
DEFAULT_L2 = 5e-5
DEFAULT_PATIENCE = 77

#模型参数
DEFAULT_MODEL_TYPE = 'transformer'
DEFAULT_D_MODEL = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_CROSS_LAYERS = 2
DEFAULT_NHEAD = 8

#数据集参数
DEFAULT_DATASET_PATHS = {
        'HCP': '/ailab/group/medai-share/syDu/Brain_EC/HCP/HCP_csv_out',
        'PPMI': '/ailab/group/medai-share/syDu/Brain_EC/PPMI/PPMI_CSV',
        'ABIDE': '/ailab/group/medai-share/syDu/Brain_EC/ABIDE/ABIDE_csv_out',
        'ADNI': '/ailab/group/medai-share/syDu/Brain_EC/ADNI/ADNI_CSV'
    }

DEFAULT_DATASET_PATHS_AAL = {
        'HCP': '/ailab/group/medai-share/syDu/Brain_EC/HCP/HCP_AAL3_csv_out',
        'PPMI': '/ailab/group/medai-share/syDu/Brain_EC/PPMI/PPMI_AAL3_CSV',
        'ABIDE': '/ailab/group/medai-share/syDu/Brain_EC/ABIDE/ABIDE_AAL3_csv_out',
        'ADNI': '/ailab/group/medai-share/syDu/Brain_EC/ADNI/ADNI_AAL3_CSV'    
}



DEFAULT_LABEL_PATH = '/ailab/group/medai-share/syDu/Brain_EC/source_label_AAL3.csv'