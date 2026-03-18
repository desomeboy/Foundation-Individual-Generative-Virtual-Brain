# vtb/data.py
import os
import time
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd # 在文件顶部导入
# 数据切片：multi2one / 单病人加载 / 缓存 / 划分
import json

# >>> paste from original: def multi2one <<<
def multi2one(time_series, steps, skip_first=30):
    n_area = time_series.shape[1]
    n_step = time_series.shape[0]
    start_idx = skip_first
    end_idx = n_step - steps
    input_X = np.zeros((end_idx - start_idx, n_area * steps))
    target_Y = np.zeros((end_idx - start_idx, n_area))
    for i in range(start_idx, end_idx):
        input_X[i - start_idx] = time_series[i-steps:i].flatten()
        target_Y[i - start_idx] = time_series[i].flatten()
    return np.array(input_X), np.array(target_Y)

# >>> paste from original: def load_patient_data <<<
def load_patient_data(patient_file, steps=7, skip_first=30, dataset_type='HCP', label_map=None):
    try:
                  
        scan_data = np.loadtxt(patient_file, delimiter=',', skiprows=1)
        n_timepoints, n_rois = scan_data.shape
        
        # 验证ROI数量（所有数据集应该都是360个ROI）
        if n_rois not in  [166,360]:
            print(f"Warning: {patient_file} has incorrect ROI dimensions. Expected 166 or 360 , got {n_rois}")
            return None,None, None, 0            
          
        # 标准化
        mean = np.mean(scan_data, axis=0, keepdims=True)
        std = np.std(scan_data, axis=0, keepdims=True)
        std[std == 0] = 1.0  # 防止标准差为0
        normalized_data = (scan_data - mean) / std
        
        all_inputs, all_targets = [], []
        num_scans = 0
        
        # HCP数据集：4800个时间点，分成4段，每段1200
        if dataset_type == 'HCP':
            if n_timepoints != 4800:
                print(f"Warning: HCP dataset expected 4800 timepoints, got {n_timepoints}")
                return None, None,None, 0
            
            for i in range(4):
                start_idx = i * 1200
                end_idx = (i + 1) * 1200
                single_scan = normalized_data[start_idx:end_idx, :]
                if single_scan.shape[0] < steps + skip_first:
                    continue
                inputs, targets = multi2one(single_scan, steps, skip_first)
                all_inputs.append(inputs)
                all_targets.append(targets)
            num_scans = 4
            
        # 其他数据集：整个序列作为一次扫描
        else:
            # 检查是否有足够的时间点
            if n_timepoints < steps + skip_first:
                print(f"Warning: {patient_file} has too few timepoints ({n_timepoints}) for steps={steps} and skip_first={skip_first}")
                return None, None, None, 0
            
            inputs, targets = multi2one(normalized_data, steps, skip_first)
            all_inputs.append(inputs)
            all_targets.append(targets)
            num_scans = 1
        
        # 检查是否有有效数据
        if not all_inputs or len(all_inputs[0]) == 0:
            return None, None, None, 0
            
        all_inputs = np.vstack(all_inputs)
        all_targets = np.vstack(all_targets)
        
        # 获取病人ID (去掉路径和.csv后缀)
        patient_id = os.path.splitext(os.path.basename(patient_file))[0]
        
        # 初始化标签
        patient_label = 2 # 默认为2 (HCP)
        if label_map is not None:
            if patient_id in label_map:
                patient_label = label_map[patient_id]
            else:
                print(f"Warning: Label for patient {patient_id} not found, using default label 0.")   
                 
        # 返回时，将标签作为一个与每个样本关联的数组
        if all_inputs is not None and len(all_targets) > 0:
            # 创建一个与 inputs 行数相同的标签数组
            labels = np.full((len(all_targets),), patient_label, dtype=np.int64)
            return all_inputs, all_targets, labels, num_scans
        else:
            assert(0)
            return None, None, None, 0
                            
    
    
    
    except Exception as e:
        print(f"Error loading {patient_file}: {str(e)}")
        return None, None, None, 0

def cache_patient_data(patient_path, cache_dir, steps, skip_first, dataset_type, label_map):
    """Cache data for a single patient"""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a unique cache filename for this patient
    patient_id = os.path.splitext(os.path.basename(patient_path))[0]
    cache_filename = f"{patient_id}_steps{steps}_skip{skip_first}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Check if cache is up-to-date
    if os.path.exists(cache_path) and os.path.getmtime(patient_path) <= os.path.getmtime(cache_path):
        return cache_path, True
    
    # Process patient data
    inputs, targets, labels, num_scans = load_patient_data(
        patient_path, steps, skip_first, dataset_type, label_map
    )
    
    if inputs is None or len(inputs) == 0:
        return None, False
    
    # Save to cache
    cache_data = {
        'inputs': inputs,
        'targets': targets,
        'labels': labels,
        'patient_id': patient_id,
        'steps': steps,
        'skip_first': skip_first,
        'timestamp': time.time()
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    return cache_path, False

def cache_dataset(data_dirs, cache_dir, label_file_path, steps=3, skip_first=30, force_reprocess=False):
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load label map
    label_map = {}
    if os.path.exists(label_file_path):
        df_labels = pd.read_csv(label_file_path)
        label_map = dict(zip(df_labels['ID'], df_labels['Label']))
        print(f"Loaded labels for {len(label_map)} patients from {label_file_path}")
    else:
        print(f"Warning: Label file not found at {label_file_path}. All patients will be assigned label 0.")
    
    # Create a manifest file to track all patient cache files
    manifest_filename = f"dataset_manifest_steps{steps}_skip{skip_first}.json"
    manifest_path = os.path.join(cache_dir, manifest_filename)
    
    # Check if manifest exists and is up-to-date
    manifest_needs_update = force_reprocess
    patient_cache_paths = []
    
    if not force_reprocess and os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            manifest_timestamp = manifest.get('timestamp', 0)
            
            # Check if any source file is newer than the manifest
            for data_dir in data_dirs:
                for filename in os.listdir(data_dir):
                    if filename.lower().endswith('.csv'):
                        file_path = os.path.join(data_dir, filename)
                        if os.path.getmtime(file_path) > manifest_timestamp:
                            manifest_needs_update = True
                            break
                if manifest_needs_update:
                    break
    
    if not manifest_needs_update and os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            patient_cache_paths = manifest['patient_cache_paths']
            print(f"Using dataset manifest from {manifest_path}")
            return manifest_path, False
    
    print("Processing dataset and creating cache...")
    
    # Process each patient and collect cache paths
    for data_dir in data_dirs:
        # Determine dataset type
        parent_dir = os.path.dirname(data_dir)
        dataset_name = os.path.basename(parent_dir)
        is_hcp = 'HCP' in dataset_name.upper()
        
        print(f"Loading data from {data_dir} (dataset type: {'HCP' if is_hcp else 'OTHER'})...")
        patient_files = [f for f in os.listdir(data_dir) 
                         if f.lower().endswith('.csv') and os.path.isfile(os.path.join(data_dir, f))]
        
        if not patient_files:
            print(f"Warning: No patient CSV files found in {data_dir}")
            continue
        
        for patient_file in tqdm(patient_files, desc=f"Processing {os.path.basename(data_dir)}"):
            patient_path = os.path.join(data_dir, patient_file)
            dataset_type = 'HCP' if is_hcp else 'OTHER'
            
            # Cache this patient's data
            cache_path, _ = cache_patient_data(
                patient_path, cache_dir, steps, skip_first, dataset_type, label_map
            )
            
            if cache_path:
                patient_cache_paths.append(cache_path)
    
    if not patient_cache_paths:
        raise ValueError("No patient data found in any of the provided directories")
    
    # Save manifest
    manifest = {
        'patient_cache_paths': patient_cache_paths,
        'steps': steps,
        'skip_first': skip_first,
        'timestamp': time.time(),
        'data_dirs': data_dirs
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Dataset manifest created at {manifest_path}")
    print(f"Total patients cached: {len(patient_cache_paths)}")
    return manifest_path, True


# >>> paste from original: def prepare_dataset <<<
def prepare_dataset(data_dirs, label_file_path, steps=3, skip_first=30, test_size=0.2, 
                    random_state=42, cache_dir="./data_cache", force_reprocess=False):
    manifest_path, is_cached = cache_dataset(
        data_dirs, cache_dir, label_file_path, steps, skip_first, force_reprocess
    )
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Instead of loading all data, just store the cache paths
    patient_cache_paths = manifest['patient_cache_paths']
    patient_ids = [os.path.basename(p).split('_steps')[0] for p in patient_cache_paths]
    
    # Split the patient IDs for train/test
    
    if test_size==0:
        return patient_cache_paths, [], patient_ids, []
    
    else:
        train_indices, test_indices = train_test_split(
            range(len(patient_cache_paths)), test_size=test_size, random_state=random_state
        )
        
        # Return the cache paths instead of the actual data
        train_data = [patient_cache_paths[i] for i in train_indices]
        test_data = [patient_cache_paths[i] for i in test_indices]
        train_patient_ids = [patient_ids[i] for i in train_indices]
        test_patient_ids = [patient_ids[i] for i in test_indices]
        
        return train_data, test_data, train_patient_ids, test_patient_ids


def load_patient_from_cache(cache_path):
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    return (cache_data['inputs'], cache_data['targets'], cache_data['labels'])






