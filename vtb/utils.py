# vtb/utils.py
import os
import torch
import numpy as np
import random


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[vtb.utils] Using device: {device}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
