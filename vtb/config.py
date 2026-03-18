# vtb/config.py


ROI_NUM = 166

# Data and Cache
DEFAULT_STEPS = 7
DEFAULT_SKIP_FIRST = 30
DEFAULT_CACHE_DIR = "./data_cache"
DEFAULT_OUTPUT_DIR = "./EC_results_v3"

# Training Hyperparameters
DEFAULT_BATCH_SIZE = 768
DEFAULT_EPOCHS = 200
DEFAULT_LR = 5e-4
DEFAULT_L2 = 5e-5
DEFAULT_PATIENCE = 77

# Model Hyperparameters
DEFAULT_MODEL_TYPE = 'transformer'
DEFAULT_D_MODEL = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_CROSS_LAYERS = 2
DEFAULT_NHEAD = 8

#Dataset Parameters

DEFAULT_DATASET_PATHS = {
        'HCP': '',
        'PPMI': '',
        'ABIDE': '',
        'ADNI': ''    
}

DEFAULT_DATASET_PATHS_AAL = {
        'HCP': './Data_process/HCP/HCP_AAL3_CSV',
        'PPMI': './Data_process/PPMI/PPMI_AAL3_CSV',
        'ABIDE': './Data_process/ABIDE/ABIDE_AAL3_CSV',
        'ADNI': './Data_process/ADNI/ADNI_AAL3_CSV'    
}

DEFAULT_LABEL_PATH = './Data_process/Data_label.csv'