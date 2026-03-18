import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pickle
import numpy as np
import os
import argparse
import shutil
import time
import torch


from vtb import (
    device, ensure_dir,
    ANN_MLP,ANN_Transformer,  # 可换成 ANN_CNN/ANN_RNN/ANN_VAR
    prepare_dataset,
    train_NN, load_model, plot_training_curves,
    analyze_single_patient
)
from vtb.config import *

import hashlib

class TeeStream:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()



def main():
    parser = argparse.ArgumentParser(description='Train and evaluate NPI model on fMRI data')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--model_path', type=str, default=os.path.join(DEFAULT_OUTPUT_DIR, 'best_model.pth'))
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--atlas', type=str, default='AAL3', 
                        choices=['HCP-MMP', 'AAL3'],
                        help='Brain atlas to use: HCP-MMP (360 regions) or aal3 (170 regions)')
    
    parser.add_argument('--test_patient_ids', type=str, default='187547_AAL3_ts')
    
    parser.add_argument('--use_specified_test_ids', action='store_true',
                        help='If set, use the specified test_patient_ids as test set instead of random split')    
    
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    parser.add_argument('--skip_first', type=int, default=DEFAULT_SKIP_FIRST)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--l2', type=float, default=DEFAULT_L2)
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    parser.add_argument('--use_last_token', type=bool, default=False)

    parser.add_argument('--cache_dir', type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument('--force_reprocess', action='store_true')
    parser.add_argument('--model_type',type=str, default=DEFAULT_MODEL_TYPE)
    
    parser.add_argument('--dataset', type=str, default='all', 
                        choices=['HCP', 'PPMI', 'ABIDE', 'ADNI', 'all'],
                        help='Dataset to use: HCP, PPMI, ABIDE, ADNI, or all') 
    parser.add_argument('--label_path', type=str, default=DEFAULT_LABEL_PATH)
       
    #model params    
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument('--num_cross_layers', type=int, default=DEFAULT_NUM_CROSS_LAYERS)
    parser.add_argument('--nhead', type=int, default=DEFAULT_NHEAD)     

         
    args = parser.parse_args()

    # 生成一个基于超参数的唯一输出目录
    folder_name = f"{args.atlas}_lr_{args.lr}_batch_{args.batch_size}_epochs_{args.num_epochs}_l2_{args.l2}_patience_{args.patience}_steps_{args.steps}_dmodel_{args.d_model}"
    args.output_dir = os.path.join(args.output_dir, folder_name)
    
    
    args.model_path = os.path.join(args.output_dir, 'best_model.pth')
    
    args.cache_dir = f"{args.cache_dir}/{args.dataset}_{args.atlas}_{args.steps}steps_{args.skip_first}skip"
    
    # 创建输出目录
    ensure_dir(args.output_dir)
    print(f"Saving results to: {args.output_dir}")
    
    # 重定向输出到日志文件
    log_path = os.path.join(args.output_dir, 'training.log')
    sys.stdout = TeeStream(log_path)
    sys.stderr = TeeStream(os.path.join(args.output_dir, 'error.log'))    
    
    
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")    

    
    # 根据atlas选择数据集路径
    ROI_NUM_MAPPING = {'HCP-MMP': 360, 'AAL3': 166}
    ROI_NUM = ROI_NUM_MAPPING[args.atlas]
    
    if args.atlas == 'HCP-MMP':
        DATASET_PATHS = DEFAULT_DATASET_PATHS
    elif args.atlas == 'AAL3':
        DATASET_PATHS = DEFAULT_DATASET_PATHS_AAL
    else:
        raise ValueError(f"Unsupported atlas: {args.atlas}")
    
    
    if args.dataset == 'all':
        data_dirs = list(DATASET_PATHS.values())
    else:
        data_dirs = [DATASET_PATHS[args.dataset]]    
        
    print(f"Using dataset(s): {args.dataset}")
    print(f"Data directories: {data_dirs}")
        

    # 保存脚本快照
    this_file = __file__
    shutil.copy(this_file, os.path.join(args.output_dir, 'NPI_demo.py'))

    print("="*50)
    print("="*50)
    print(f"Data directory: {data_dirs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Using device: {device}")
    print(f"Parameters:")
    print(f"  steps = {args.steps}")
    print(f"  skip_first = {args.skip_first}")
    print(f"  test_size = {args.test_size}")
    print(f"  batch_size = {args.batch_size}")
    print(f"  num_epochs = {args.num_epochs}")
    print(f"  lr = {args.lr}")
    print(f"  l2 = {args.l2}")
    print(f"  patience = {args.patience}")
    print(f"  cache_dir = {args.cache_dir}")
    print(f"  force_reprocess = {args.force_reprocess}")
    print("="*50)

    print("\nStep 1: Preparing dataset...")
    start_time = time.time()
    
    train_data, test_data, train_patient_ids, test_patient_ids = prepare_dataset(
        data_dirs , args.label_path, steps=args.steps, skip_first=args.skip_first,
        test_size=0.0, cache_dir=args.cache_dir, force_reprocess=args.force_reprocess
    )
    
    # 如果指定了使用特定的测试ID，则重新划分
    if args.use_specified_test_ids:
        # 解析测试患者ID
        specified_test_ids = [pid.strip() for pid in args.test_patient_ids.split(',') if pid.strip()]
        print(f"Using specified test patient IDs: {specified_test_ids}")
        
        # 读取manifest文件获取所有患者信息
        import json
        manifest_path = os.path.join(args.cache_dir, f"dataset_manifest_steps{args.steps}_skip{args.skip_first}.json")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        patient_cache_paths = manifest['patient_cache_paths']
        all_patient_ids = [os.path.basename(p).split('_steps')[0] for p in patient_cache_paths]
        
        # 根据指定ID划分训练集和测试集
        train_indices = []
        test_indices = []
        
        for i, pid in enumerate(all_patient_ids):
            if pid in specified_test_ids:
                test_indices.append(i)
            else:
                train_indices.append(i)
        
        # 重建训练集和测试集
        train_data = [patient_cache_paths[i] for i in train_indices]
        test_data = [patient_cache_paths[i] for i in test_indices]
        train_patient_ids = [all_patient_ids[i] for i in train_indices]
        test_patient_ids = [all_patient_ids[i] for i in test_indices]
        
        print(f"Manually split dataset: {len(train_patient_ids)} training patients, {len(test_patient_ids)} test patients")
        print(f"Test patients: {test_patient_ids}")
    else:
        # 原有的随机划分方式
        train_data, test_data, train_patient_ids, test_patient_ids = prepare_dataset(
            data_dirs, args.label_path, steps=args.steps, skip_first=args.skip_first,
            test_size=args.test_size,
            cache_dir=args.cache_dir, force_reprocess=args.force_reprocess
        )    
    
    
    
    
    
    
    print(f"Dataset preparation completed in {time.time() - start_time:.2f} seconds")
    print(f"Training patients: {len(train_patient_ids)}")
    print(f"Testing patients: {len(test_patient_ids)}")

    with open(os.path.join(args.output_dir, 'train_patient_ids.txt'), 'w') as f:
        f.writelines([pid + "\n" for pid in train_patient_ids])
    with open(os.path.join(args.output_dir, 'test_patient_ids.txt'), 'w') as f:
        f.writelines([pid + "\n" for pid in test_patient_ids])

    # Step 2: Create model

    print("\nStep 2: Creating model...")
    
    input_dim = args.steps * ROI_NUM
    print(f"Using {args.atlas} atlas with {ROI_NUM} regions")
        
    if args.model_type == 'MLP':    
        
        model = ANN_MLP(
            input_dim=input_dim,
            hidden_dim=2 * ROI_NUM,
            latent_dim=int(0.8 * ROI_NUM),
            output_dim=ROI_NUM
        )
    elif args.model_type == 'transformer':
        
        model = ANN_Transformer(
        input_dim=input_dim,
        steps=args.steps,
        roi_num=ROI_NUM,
        d_model=args.d_model,          
        nhead=args.nhead,              # 确保 d_model % nhead == 0
        num_layers=args.num_layers,         # 2~6 之间按数据量/显存调
        dim_feedforward=2*args.d_model,  # 2x~4x d_model
        dropout=0.1,
        use_layernorm = True,
        use_last_token = args.use_last_token,
        num_labels=4, 
        num_cross_layers=args.num_cross_layers, #
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")


    # Step 3: Train or load model
    if args.train:
        print("\nStep 3: Training model...")
        start_time = time.time()
        model, train_losses, test_losses = train_NN(
            model, train_data, test_data,
            batch_size=args.batch_size, num_epochs=args.num_epochs,
            lr=args.lr, l2=args.l2, model_path=args.model_path, patience=args.patience
        )
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        plot_training_curves(train_losses, test_losses, output_path=os.path.join(args.output_dir, 'training_curves.png'))
    else:
        if os.path.exists(args.model_path):
            print(f"\nStep 3: Loading pre-trained model from {args.model_path}...")
            model, train_losses, test_losses, best_loss = load_model(model, args.model_path)
            if best_loss is not None:
                print(f"Loaded model with best test loss: {best_loss:.6f}")
            if train_losses:
                plot_training_curves(train_losses, test_losses, output_path=os.path.join(args.output_dir, 'training_curves.png'))
        else:
            raise FileNotFoundError(f"Model file not found at {args.model_path}")

    # Step 4: Analyze a patient from test set
    test_patient_ids_to_analyze = []
    if args.test_patient_ids:
        test_patient_ids_to_analyze = [pid.strip() for pid in args.test_patient_ids.split(',')]    
    
    
    if test_patient_ids_to_analyze:
        print(f"\nStep 4: Analyzing {len(test_patient_ids_to_analyze)} patients from test set...")
        
        for test_patient_id in test_patient_ids_to_analyze:
            if test_patient_id in test_patient_ids:
                print(f"\nAnalyzing patient {test_patient_id} from test set...")
                patient_idx = test_patient_ids.index(test_patient_id)
                test_data_path = test_data[patient_idx]
                with open(test_data_path, 'rb') as f:
                    test_data_actual = pickle.load(f)
                test_data_input = [test_data_actual['inputs'], test_data_actual['targets'], test_data_actual['labels']]
                
                # First, show zero-shot results
                print(f"  Evaluating pre-trained model on patient {test_patient_id} (zero-shot)...")
                zero_shot_dir = os.path.join(args.output_dir, 'zero_shot_analysis', test_patient_id)
                os.makedirs(zero_shot_dir, exist_ok=True)
                
                empirical_FC, model_FC_matrix, NPI_EC = analyze_single_patient(
                    model, test_data_input, test_patient_id,
                    steps=args.steps, output_dir=zero_shot_dir,
                    fine_tune=False  # No fine-tuning for zero-shot
                )
                
                
                # Then, fine-tune and analyze
                print(f"  Fine-tuning model on patient {test_patient_id} data...")
                fine_tune_params = {'batch_size': 64, 'num_epochs': 12, 'lr': 1e-3, 'l2': 5e-5, 'patience': 100}
                fine_tune_dir = os.path.join(args.output_dir, 'patient_analysis', test_patient_id)
                os.makedirs(fine_tune_dir, exist_ok=True)
                _, _, _, fine_tuned_model = analyze_single_patient(
                    model, test_data_input, test_patient_id,
                    steps=args.steps, output_dir=fine_tune_dir,
                    fine_tune=True, fine_tune_params=fine_tune_params
                )
                
                # Save fine-tuned model
                patient_model_dir = os.path.join(args.output_dir, 'patient_models')
                os.makedirs(patient_model_dir, exist_ok=True)
                patient_model_path = os.path.join(patient_model_dir, f"model_{test_patient_id}.pth")
                torch.save({
                    'model_state_dict': {k: v.cpu() for k, v in fine_tuned_model.state_dict().items()},
                    'init_args': fine_tuned_model.init_args
                }, patient_model_path)
                print(f"  Fine-tuned model for patient {test_patient_id} saved to {patient_model_path}")
            else:
                print(f"\nWarning: Patient ID {test_patient_id} not found in test set")
                print(f"\nAnalyzing patient {test_patient_id} from train set...")
                patient_idx = train_patient_ids.index(test_patient_id)
                test_data_path = train_data[patient_idx]
                with open(test_data_path, 'rb') as f:
                    test_data_actual = pickle.load(f)
                test_data_input = [test_data_actual['inputs'], test_data_actual['targets'], test_data_actual['labels']]
                
                # First, show zero-shot results
                print(f"  Evaluating pre-trained model on patient {test_patient_id} (zero-shot)...")
                zero_shot_dir = os.path.join(args.output_dir, 'zero_shot_analysis', test_patient_id)
                os.makedirs(zero_shot_dir, exist_ok=True)
                
                empirical_FC, model_FC_matrix, NPI_EC = analyze_single_patient(
                    model, test_data_input, test_patient_id,
                    steps=args.steps, output_dir=zero_shot_dir,
                    fine_tune=False  # No fine-tuning for zero-shot
                )
                
                
                # Then, fine-tune and analyze
                print(f"  Fine-tuning model on patient {test_patient_id} data...")
                fine_tune_params = {'batch_size': 64, 'num_epochs': 50, 'lr': 1e-3, 'l2': 5e-5, 'patience': 64}
                fine_tune_dir = os.path.join(args.output_dir, 'patient_analysis', test_patient_id)
                os.makedirs(fine_tune_dir, exist_ok=True)
                _, _, _, fine_tuned_model = analyze_single_patient(
                    model, test_data_input, test_patient_id,
                    steps=args.steps, output_dir=fine_tune_dir,
                    fine_tune=True, fine_tune_params=fine_tune_params
                )
                
                # Save fine-tuned model
                patient_model_dir = os.path.join(args.output_dir, 'patient_models')
                os.makedirs(patient_model_dir, exist_ok=True)
                patient_model_path = os.path.join(patient_model_dir, f"model_{test_patient_id}.pth")
                torch.save({
                    'model_state_dict': {k: v.cpu() for k, v in fine_tuned_model.state_dict().items()},
                    'init_args': fine_tuned_model.init_args
                }, patient_model_path)
                print(f"  Fine-tuned model for patient {test_patient_id} saved to {patient_model_path}")                
                
                
                
                
                
    else:
        print("\nStep 4: No test patients to analyze")

    print("\nvtb analysis completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    
    # 在main函数结束时关闭日志
    try:
        sys.stdout.log.close()
        sys.stderr.log.close()
    except:
        pass
    # 恢复原始输出
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=0 python scripts/train_FVB \
#   --train \
#   --batch_size 256\
#   --num_epochs 300\
#   --lr 5e-5 \
#   --l2 1e-4 \
#   --patience 100 \
#   --steps 7 \
#   --skip_first 30 --output_dir /ailab/user/dusiyuan/code/Brain/EC/AAL3/EC_results_num_layer_2  --d_model 256 --num_layers 2 --num_cross_layers 1 --use_specified_test_ids




# CUDA_VISIBLE_DEVICES=0 python scripts/train_FVB \
#   --train \
#   --batch_size 256\
#   --num_epochs 300\
#   --lr 5e-5 \
#   --l2 1e-4 \
#   --patience 100 \
#   --steps 7 \
#   --atlas HCP-MMP \
#   --cache_dir /ailab/user/dusiyuan/code/Brain/EC/data_cache \
#   --model_path /ailab/user/dusiyuan/code/Brain/EC/HCP/EC_results_num_layer_2/best_model.pth \
#   --label_path /ailab/group/medai-share/syDu/Brain_EC/source_label.csv \
#   --skip_first 30 --output_dir /ailab/user/dusiyuan/code/Brain/EC/HCP/EC_results_num_layer_2  --d_model 256 --num_layers 2 --num_cross_layers 1 --use_specified_test_ids



#fintune iVB
# CUDA_VISIBLE_DEVICES=0 python scripts/train_FVB \
#   --batch_size 256\
#   --num_epochs 300\
#   --lr 5e-5 \
#   --l2 1e-4 \
#   --patience 100 \
#   --steps 7 \
#   --cache_dir /ailab/user/dusiyuan/code/Brain/EC/data_cache \
#   --model_path /ailab/user/dusiyuan/code/Brain/EC/HCP/EC_results_num_layer_2/best_model.pth \
#   --label_path /ailab/group/medai-share/syDu/Brain_EC/source_label.csv \
#   --skip_first 30 --output_dir /ailab/user/dusiyuan/code/Brain/EC/HCP/EC_results_num_layer_2  --d_model 256 --num_layers 2 --num_cross_layers 1 --use_specified_test_ids





# CUDA_VISIBLE_DEVICES=0 python scripts/train_FVB \
#   --batch_size 512\
#   --num_epochs 100\
#   --lr 1e-3 \
#   --l2 5e-5 \
#   --patience 100 \
#   --steps 7 --model_type MLP \
#   --skip_first 30 --output_dir /ailab/user/dusiyuan/code/Brain/EC/AAL3/ANN_baseline --use_specified_test_ids

