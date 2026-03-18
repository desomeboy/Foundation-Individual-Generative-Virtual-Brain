import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import time
import pickle
import torch

from vtb import (
    device, ensure_dir,
    ANN_MLP, ANN_Transformer,
    load_model, analyze_single_patient
)
from vtb.config import *

from vtb.data import load_patient_data
from vtb.utils import set_seed

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
            num_labels=4,           
            num_cross_layers=args.num_cross_layers
        )
        
        
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Test a pre-trained model on a single patient's CSV")
    # Paths and basic parameters
    parser.add_argument('--patient_csv', type=str, required=True, help='Path to the CSV file for a single new patient')
    parser.add_argument('--model_path', type=str, default=os.path.join(DEFAULT_OUTPUT_DIR, 'best_model.pth'))
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--label_path', type=str, default=DEFAULT_LABEL_PATH)
    parser.add_argument('--cache_dir', type=str, default=DEFAULT_CACHE_DIR)

    # Preprocessing and model structure parameters (must match training phase)
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    parser.add_argument('--skip_first', type=int, default=30)
    parser.add_argument('--model_type', type=str, default=DEFAULT_MODEL_TYPE, choices=['MLP', 'transformer'])
    parser.add_argument('--use_last_token', type=bool, default=False)

    # Analysis control
    parser.add_argument('--fine_tune', action='store_true', help='Whether to fine-tune on this patient')
    parser.add_argument('--ft_batch_size', type=int, default=32)
    parser.add_argument('--ft_epochs', type=int, default=300)
    parser.add_argument('--ft_lr', type=float, default=5e-5)
    parser.add_argument('--ft_l2', type=float, default=1e-3)
    parser.add_argument('--ft_patience', type=int, default=128)
    
    # Model params    
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument('--num_cross_layers', type=int, default=DEFAULT_NUM_CROSS_LAYERS)
    parser.add_argument('--nhead', type=int, default=DEFAULT_NHEAD)     
        
    args = parser.parse_args()
    if args.output_dir is None:
        patient_basename = os.path.splitext(os.path.basename(args.patient_csv))[0]
        args.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, patient_basename)
        

    ensure_dir(args.output_dir)
    log_path = os.path.join(args.output_dir, 'test.log')
    sys.stdout = TeeStream(log_path)
    sys.stderr = TeeStream(os.path.join(args.output_dir, 'error.log'))

    print("="*60)
    print("vtb Single-Patient Test")
    print("="*60)
    print(f"Patient CSV : {args.patient_csv}")
    print(f"Model path  : {args.model_path}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Device      : {device}")
    print(f"Params      : steps={args.steps}, skip_first={args.skip_first}, model_type={args.model_type}, d_model={args.d_model}, use_last_token={args.use_last_token}")
    print(f"Fine-tune   : {args.fine_tune} (bs={args.ft_batch_size}, epochs={args.ft_epochs}, lr={args.ft_lr}, l2={args.ft_l2}, patience={args.ft_patience})")
    print("="*60)

    # ---------- Step 1: Load and preprocess patient CSV ----------

    set_seed(42)

    start = time.time()
    inputs, targets, labels, num_scans = load_patient_data(
        patient_file=args.patient_csv,
        steps=args.steps,
        skip_first=args.skip_first,
        dataset_type='OTHER',     
        label_map=None           
    )
    if inputs is None or len(inputs) == 0:
        raise ValueError(f"Failed to load or preprocess patient CSV: {args.patient_csv}")
    patient_id = os.path.splitext(os.path.basename(args.patient_csv))[0]
    print(f"Loaded patient '{patient_id}': inputs={inputs.shape}, targets={targets.shape}, labels={labels.shape}, scans={num_scans}")
    print(f"Preprocess done in {time.time() - start:.2f}s")

    # ---------- Step 2: Build model and load checkpoint ----------
    input_dim = args.steps * ROI_NUM
    model = build_model(args, input_dim)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")
    print(f"Loading pre-trained weights from {args.model_path} ...")
    model, train_losses, test_losses, best_loss = load_model(model, args.model_path)
    if best_loss is not None:
        print(f"Loaded model with best test loss: {best_loss:.6f}")

    # ---------- Step 3: Zero-shot analysis ----------
    zero_dir = os.path.join(args.output_dir, 'zero_shot', patient_id)
    ensure_dir(zero_dir)
    print(f"\n[Zero-shot] Analyzing patient {patient_id} ...")
    patient_data_input = (inputs, targets, labels)
    from vtb.analysis import analyze_single_patient

    empirical_FC, model_FC_matrix, NPI_EC = analyze_single_patient(
        pretrained_model=model,
        patient_data=patient_data_input,
        patient_id=patient_id,
        steps=args.steps,
        output_dir=zero_dir,
        fine_tune=False
    )
    print(f"[Zero-shot] Done. Results saved to: {zero_dir}")

    # ---------- Step 4: Fine-tune on patient ----------
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

        empirical_FC_ft, model_FC_matrix_ft, NPI_EC_ft, fine_tuned_model = analyze_single_patient(
            pretrained_model=model,
            patient_data=patient_data_input,
            patient_id=patient_id,
            steps=args.steps,
            output_dir=ft_dir,
            fine_tune=True,
            fine_tune_params=fine_tune_params
        )

        pt_dir = os.path.join(args.output_dir, 'patient_models')
        ensure_dir(pt_dir)
        save_path = os.path.join(pt_dir, f"model_{patient_id}.pth")
        torch.save({
            'model_state_dict': {k: v.cpu() for k, v in fine_tuned_model.state_dict().items()},
            'init_args': fine_tuned_model.init_args
        }, save_path)
        print(f"[Fine-tune] Saved patient-specific weights to: {save_path}")

    print("\nAll done!")

    
    try:
        sys.stdout.log.close()
        sys.stderr.log.close()
    except:
        pass
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

if __name__ == "__main__":
    main()



