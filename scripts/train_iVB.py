# scripts/test_npi.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import time
import pickle
import torch

from npi import (
    device, ensure_dir,
    ANN_MLP, ANN_Transformer,
    load_model, analyze_single_patient
)
from npi.config import *

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
    """构建与训练阶段一致的模型结构，以确保 checkpoint 可正确加载"""
    if args.model_type == 'MLP':
        model = ANN_MLP(
            input_dim=input_dim,
            hidden_dim=2 * ROI_NUM,
            latent_dim=int(0.8 * ROI_NUM),
            output_dim=ROI_NUM
        )
    elif args.model_type == 'transformer':
        # 注意：参数需与 train_npi.py 中保持一致
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
            num_labels=4,            # 与 train_npi.py 保持一致
            num_cross_layers=args.num_cross_layers
        )
        
        
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Test a pre-trained NPI model on a single patient's CSV")
    # 路径与基础参数
    parser.add_argument('--patient_csv', type=str, required=True, help='单个新病人的 CSV 文件路径')
    parser.add_argument('--model_path', type=str, default=os.path.join(DEFAULT_OUTPUT_DIR, 'best_model.pth'))
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--label_path', type=str, default=DEFAULT_LABEL_PATH)
    parser.add_argument('--cache_dir', type=str, default=DEFAULT_CACHE_DIR)

    # 预处理与模型结构参数（需与训练阶段一致）
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    parser.add_argument('--skip_first', type=int, default=30)
    parser.add_argument('--model_type', type=str, default=DEFAULT_MODEL_TYPE, choices=['MLP', 'transformer'])
    parser.add_argument('--use_last_token', type=bool, default=False)

    # 分析控制
    parser.add_argument('--fine_tune', action='store_true', help='是否对该病人进行微调')
    parser.add_argument('--ft_batch_size', type=int, default=32)
    parser.add_argument('--ft_epochs', type=int, default=300)
    parser.add_argument('--ft_lr', type=float, default=5e-5)
    parser.add_argument('--ft_l2', type=float, default=1e-3)
    parser.add_argument('--ft_patience', type=int, default=128)
    
    #model params    
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument('--num_cross_layers', type=int, default=DEFAULT_NUM_CROSS_LAYERS)
    parser.add_argument('--nhead', type=int, default=DEFAULT_NHEAD)     
        
    args = parser.parse_args()
    if args.output_dir is None:
        patient_basename = os.path.splitext(os.path.basename(args.patient_csv))[0]
        args.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, patient_basename)
        
    # 日志与基本信息
    ensure_dir(args.output_dir)
    log_path = os.path.join(args.output_dir, 'test.log')
    sys.stdout = TeeStream(log_path)
    sys.stderr = TeeStream(os.path.join(args.output_dir, 'error.log'))

    print("="*60)
    print("NPI Single-Patient Test")
    print("="*60)
    print(f"Patient CSV : {args.patient_csv}")
    print(f"Model path  : {args.model_path}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Device      : {device}")
    print(f"Params      : steps={args.steps}, skip_first={args.skip_first}, model_type={args.model_type}, d_model={args.d_model}, use_last_token={args.use_last_token}")
    print(f"Fine-tune   : {args.fine_tune} (bs={args.ft_batch_size}, epochs={args.ft_epochs}, lr={args.ft_lr}, l2={args.ft_l2}, patience={args.ft_patience})")
    print("="*60)

    # ---------- Step 1: 读取并预处理该病人 CSV ----------
    # 复用 npi.data.load_patient_data（含 z-score、multi2one、维度检查、labels 填充）
    from npi.data import load_patient_data
    from npi.utils import set_seed
    set_seed(42)

    start = time.time()
    inputs, targets, labels, num_scans = load_patient_data(
        patient_file=args.patient_csv,
        steps=args.steps,
        skip_first=args.skip_first,
        dataset_type='OTHER',     # 单病人 CSV 一般按 OTHER 处理（非 HCP 4800 切片）
        label_map=None            # 对于新病人多数没有标签，此处默认 2
    )
    if inputs is None or len(inputs) == 0:
        raise ValueError(f"Failed to load or preprocess patient CSV: {args.patient_csv}")
    patient_id = os.path.splitext(os.path.basename(args.patient_csv))[0]
    print(f"Loaded patient '{patient_id}': inputs={inputs.shape}, targets={targets.shape}, labels={labels.shape}, scans={num_scans}")
    print(f"Preprocess done in {time.time() - start:.2f}s")

    # ---------- Step 2: 构建模型并加载 checkpoint ----------
    input_dim = args.steps * ROI_NUM
    model = build_model(args, input_dim)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")
    print(f"Loading pre-trained weights from {args.model_path} ...")
    model, train_losses, test_losses, best_loss = load_model(model, args.model_path)
    if best_loss is not None:
        print(f"Loaded model with best test loss: {best_loss:.6f}")

    # ---------- Step 3: zero-shot 分析（不微调） ----------
    zero_dir = os.path.join(args.output_dir, 'zero_shot', patient_id)
    ensure_dir(zero_dir)
    print(f"\n[Zero-shot] Analyzing patient {patient_id} ...")
    patient_data_input = (inputs, targets, labels)
    from npi.analysis import analyze_single_patient
    # fine_tune=False：只评估，不更新权重
    empirical_FC, model_FC_matrix, NPI_EC = analyze_single_patient(
        pretrained_model=model,
        patient_data=patient_data_input,
        patient_id=patient_id,
        steps=args.steps,
        output_dir=zero_dir,
        fine_tune=False
    )
    print(f"[Zero-shot] Done. Results saved to: {zero_dir}")

    # ---------- Step 4: （可选）对该病人微调并再次分析 ----------
    if args.fine_tune:
        ft_dir = os.path.join(args.output_dir, 'fine_tune', patient_id)
        ensure_dir(ft_dir)
        print(f"\n[Fine-tune] Start fine-tuning on patient {patient_id} ...")
        fine_tune_params = {
            'batch_size': args.ft_batch_size,
            'num_epochs': args.ft_epochs,
            'lr': args.ft_lr,
            'l2': args.ft_l2,
            'patience': args.ft_patience
        }
        # analyze_single_patient 内部会调用 fine_tune_for_patient 并返回 fine-tuned 模型
        empirical_FC_ft, model_FC_matrix_ft, NPI_EC_ft, fine_tuned_model = analyze_single_patient(
            pretrained_model=model,
            patient_data=patient_data_input,
            patient_id=patient_id,
            steps=args.steps,
            output_dir=ft_dir,
            fine_tune=True,
            fine_tune_params=fine_tune_params
        )
        # 保存专属微调权重（包含 init_args，便于复现）
        pt_dir = os.path.join(args.output_dir, 'patient_models')
        ensure_dir(pt_dir)
        save_path = os.path.join(pt_dir, f"model_{patient_id}.pth")
        torch.save({
            'model_state_dict': {k: v.cpu() for k, v in fine_tuned_model.state_dict().items()},
            'init_args': fine_tuned_model.init_args
        }, save_path)
        print(f"[Fine-tune] Saved patient-specific weights to: {save_path}")

    print("\nAll done!")

    # 关闭日志并恢复标准输出
    try:
        sys.stdout.log.close()
        sys.stderr.log.close()
    except:
        pass
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

if __name__ == "__main__":
    main()



