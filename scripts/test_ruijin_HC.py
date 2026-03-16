# scripts/train_healthy_virtual_brain.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import time
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from npi import (
    device, ensure_dir,
    ANN_MLP, ANN_Transformer,
    load_model
)
from npi.config import *
from npi.data import load_patient_data
from npi.utils import set_seed

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

def build_model(args, input_dim):
    """构建模型结构"""
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
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=2 * args.d_model,
            dropout=0.1,
            use_layernorm=True,
            use_last_token=args.use_last_token,
            num_labels=4,            # 保持与预训练一致
            num_cross_layers=args.num_cross_layers
        )
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")
    return model

def train_healthy_model(model, train_data, val_data, args, output_dir):
    """在健康人数据上训练模型"""
    # 准备数据
    train_inputs, train_targets, train_labels = train_data
    val_inputs, val_targets, val_labels = val_data
    
    # 转换为张量
    train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    
    val_inputs = torch.tensor(val_inputs, dtype=torch.float32)
    val_targets = torch.tensor(val_targets, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    
    # 创建数据集
    train_dataset = data.TensorDataset(train_inputs, train_targets, train_labels)
    val_dataset = data.TensorDataset(val_inputs, val_targets, val_labels)
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 初始化训练组件
    model = model.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    
    # 训练记录
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    print(f"\nStarting training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}, L2: {args.l2}")
    
    for epoch in range(args.num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        for X, y, labels in train_loader:
            X, y, labels = X.to(device), y.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(X, labels)  # 传递标签（健康人均为0）
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * X.size(0)
        
        avg_train_loss = total_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X, y, labels in val_loader:
                X, y, labels = X.to(device), y.to(device), labels.to(device)
                outputs = model(X, labels)
                loss = loss_fn(outputs, y)
                total_val_loss += loss.item() * X.size(0)
        
        avg_val_loss = total_val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)
        
        # 早停检查
        if avg_val_loss < best_val_loss - args.min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            
        # 日志输出
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.num_epochs - 1:
            print(f"Epoch [{epoch+1}/{args.num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"Best Val: {best_val_loss:.6f}")
        
        # 早停触发
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation loss: {best_val_loss:.6f}")
    
    # 保存训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curves for Healthy Virtual Brain')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, train_losses, val_losses, best_val_loss

def main():
    parser = argparse.ArgumentParser(description="Train a virtual brain model on healthy subjects")
    # 路径与基础参数
    parser.add_argument('--data_dir', type=str, required=True, help='目录包含所有健康人CSV文件 (hc_*.csv)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(DEFAULT_OUTPUT_DIR, 'healthy_virtual_brain'))
    parser.add_argument('--pretrained_model', type=str, default=None, help='预训练模型路径（可选）')
    
    # 预处理参数
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    parser.add_argument('--skip_first', type=int, default=30)
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例（按病人划分）')
    
    # 模型结构参数
    parser.add_argument('--model_type', type=str, default=DEFAULT_MODEL_TYPE, choices=['MLP', 'transformer'])
    parser.add_argument('--use_last_token', type=bool, default=False)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument('--num_cross_layers', type=int, default=DEFAULT_NUM_CROSS_LAYERS)
    parser.add_argument('--nhead', type=int, default=DEFAULT_NHEAD)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--min_delta', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    ensure_dir(args.output_dir)
    log_path = os.path.join(args.output_dir, 'training.log')
    sys.stdout = TeeStream(log_path)
    sys.stderr = TeeStream(os.path.join(args.output_dir, 'error.log'))
    
    print("="*80)
    print("Healthy Virtual Brain Training")
    print("="*80)
    print(f"Data directory     : {args.data_dir}")
    print(f"Output directory   : {args.output_dir}")
    print(f"Pretrained model   : {args.pretrained_model if args.pretrained_model else 'None'}")
    print(f"Device             : {device}")
    print(f"Params             : steps={args.steps}, skip_first={args.skip_first}, train_ratio={args.train_ratio}")
    print(f"Model              : {args.model_type} (d_model={args.d_model}, use_last_token={args.use_last_token})")
    print(f"Training           : bs={args.batch_size}, lr={args.lr}, l2={args.l2}, epochs={args.num_epochs}")
    print("="*80)
    
    start_time = time.time()
    
    # ---------- Step 1: 加载所有健康人数据 ----------
    print("\nLoading healthy subject data...")
    hc_files = [f for f in os.listdir(args.data_dir) 
               if f.startswith('hc') and f.endswith('.csv')]
    
    if not hc_files:
        raise ValueError(f"No healthy subject files found in {args.data_dir} (expected files starting with 'hc')")
    
    print(f"Found {len(hc_files)} healthy subjects:")
    for f in hc_files:
        print(f"  - {f}")
    
    # 存储所有数据
    all_inputs, all_targets, all_labels, all_patient_ids = [], [], [], []
    
    for csv_file in hc_files:
        patient_path = os.path.join(args.data_dir, csv_file)
        patient_id = os.path.splitext(csv_file)[0]
        
        try:
            # 加载单个病人数据
            inputs, targets, labels, num_scans = load_patient_data(
                patient_file=patient_path,
                steps=args.steps,
                skip_first=args.skip_first,
                dataset_type='OTHER',  # 非HCP数据
                label_map=None
            )
            
            if inputs is None or len(inputs) == 0:
                print(f"Warning: Skipping {patient_id} - no valid data after preprocessing")
                continue
            
            # 将所有健康人标签设为0（健康状态）
            labels[:] = 2
            
            print(f"Loaded {patient_id}: {len(inputs)} samples, {num_scans} raw scans")
            
            # 存储数据
            all_inputs.append(inputs)
            all_targets.append(targets)
            all_labels.append(labels)
            all_patient_ids.append(patient_id)
            
        except Exception as e:
            print(f"Error processing {patient_id}: {str(e)}")
            continue
    
    if not all_inputs:
        raise ValueError("No valid healthy subject data loaded")
    
    print(f"\nSuccessfully loaded {len(all_inputs)} healthy subjects")
    total_samples = sum(len(x) for x in all_inputs)
    print(f"Total samples: {total_samples}")
    
    # ---------- Step 2: 按病人划分训练/验证集 ----------
    print(f"\nSplitting data by subject (train ratio={args.train_ratio})...")
    np.random.seed(args.seed)
    shuffled_indices = np.random.permutation(len(all_patient_ids))
    split_idx = int(len(shuffled_indices) * args.train_ratio)
    
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]
    
    print(f"Training subjects ({len(train_indices)}):")
    for idx in train_indices:
        print(f"  - {all_patient_ids[idx]} ({len(all_inputs[idx])} samples)")
    
    print(f"\nValidation subjects ({len(val_indices)}):")
    for idx in val_indices:
        print(f"  - {all_patient_ids[idx]} ({len(all_inputs[idx])} samples)")
    
    # 合并训练数据
    train_inputs = np.concatenate([all_inputs[i] for i in train_indices], axis=0)
    train_targets = np.concatenate([all_targets[i] for i in train_indices], axis=0)
    train_labels = np.concatenate([all_labels[i] for i in train_indices], axis=0)
    
    # 合并验证数据
    val_inputs = np.concatenate([all_inputs[i] for i in val_indices], axis=0)
    val_targets = np.concatenate([all_targets[i] for i in val_indices], axis=0)
    val_labels = np.concatenate([all_labels[i] for i in val_indices], axis=0)
    
    print(f"\nTraining set: {len(train_inputs)} samples")
    print(f"Validation set: {len(val_inputs)} samples")
    
    # ---------- Step 3: 构建模型 ----------
    input_dim = args.steps * ROI_NUM
    model = build_model(args, input_dim)
    
    # 加载预训练权重（如果提供）
    if args.pretrained_model:
        print(f"\nLoading pretrained weights from {args.pretrained_model}...")
        if not os.path.exists(args.pretrained_model):
            raise FileNotFoundError(f"Pretrained model not found: {args.pretrained_model}")
        
        try:
            model, _, _, best_loss = load_model(model, args.pretrained_model)
            if best_loss is not None:
                print(f"Loaded pretrained model with best loss: {best_loss:.6f}")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {str(e)}")
            print("Training from scratch instead...")
    
    model = model.to(device)
    print(f"\nModel initialized on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ---------- Step 4: 训练模型 ----------
    model, train_losses, val_losses, best_val_loss = train_healthy_model(
        model=model,
        train_data=(train_inputs, train_targets, train_labels),
        val_data=(val_inputs, val_targets, val_labels),
        args=args,
        output_dir=args.output_dir
    )
    
    # ---------- Step 5: 保存最终模型 ----------
    print("\nSaving final model...")
    model_path = os.path.join(args.output_dir, 'healthy_virtual_brain.pth')
    torch.save({
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'init_args': model.init_args,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'training_args': vars(args)
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # 训练摘要
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Total training time: {elapsed/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {model_path}")
    print("="*80)
    
    # 关闭日志
    try:
        sys.stdout.log.close()
        sys.stderr.log.close()
    except:
        pass
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

if __name__ == "__main__":
    main()
    


# python /ailab/user/dusiyuan/code/Brain/EC/EC_code_AAL/scripts/test_ruijin_HC.py \
#   --data_dir "/ailab/group/medai-share/syDu/ruijin/DBS/AAL3_CSV/hc" \
#   --output_dir "/ailab/group/medai-share/syDu/ruijin/DBS/AAL_VTB/healthy_virtual_brain" \
#   --pretrained_model '/ailab/user/dusiyuan/code/Brain/EC/AAL3/EC_results_num_layer_2/AAL3_lr_5e-05_batch_256_epochs_300_l2_0.0001_patience_100_steps_7_dmodel_256/best_model.pth' \
#   --model_type transformer \
#   --num_layers 2 \
#   --num_cross_layers 1 \
#   --d_model 256 \
#   --steps 7 \
#   --batch_size 256 \
#   --lr 5e-5 \
#   --l2 0.0001 \
#   --num_epochs 300 \
#   --patience 100 \
#   --use_last_token False \
#   --train_ratio 0.8 \
#   --skip_first 30