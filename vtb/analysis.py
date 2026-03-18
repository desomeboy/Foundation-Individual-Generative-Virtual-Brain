
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from .metrics import corrcoef, model_FC as _model_FC, model_EC as _model_EC, flat_without_diagonal
from .utils import ensure_dir,device
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from vtb.vis_tool import *
import csv
from .train import fine_tune_for_patient
from .data import load_patient_data


def safe_corrcoef(matrix):
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


def plot_region_time_series(empirical_targets, predicted_targets, region_dict, patient_id, output_dir, fig_size=(15,12), max_time_steps=200):
    """
    Plot a minimalist brain region time series comparison (showing up to max_time_steps time points):
    - Only the bottom subplot shows X-axis ticks, tick labels, and the "Time Point" title
    - The upper subplots completely hide the X-axis elements
    - The legend is saved as a separate image
    - Prediction curves are drawn with dashed lines
    - Axis lines are thicker, X-axis tick numbers are larger
    - No grid, no title, no Y-axis ticks, no top/right borders
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    T = min(empirical_targets.shape[0], max_time_steps)
    empirical_targets = empirical_targets[:T]
    predicted_targets = predicted_targets[:T]
    time_points = np.arange(T)

    num_regions = len(region_dict)
    fig, axes = plt.subplots(num_regions, 1, figsize=fig_size, sharex=True)
    if num_regions == 1:
        axes = [axes]


    line1, = axes[0].plot([], [], label='Ground Truth', linewidth=1.7, color='C0')
    line2, = axes[0].plot([], [], label='Model Prediction', linewidth=1.7,  color='C1')


    for i, (region_idx, region_name) in enumerate(region_dict.items()):
        ax = axes[i]

        ax.plot(time_points, empirical_targets[:, region_idx], linewidth=1.7, color='C0')
        ax.plot(time_points, predicted_targets[:, region_idx], linewidth=1.7, color='C1')


        ax.set_ylabel(region_name, rotation=0, labelpad=55, va='center', fontsize=17)
        ax.set_yticks([])  


        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)


    axes[-1].set_xlabel('Time Point', fontsize=20)
    axes[-1].tick_params(axis='x', labelsize=16)  

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"patient_{patient_id}_time_series_comparison.pdf")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Minimalist plot with x-axis only at bottom (first {T} steps) saved to {fig_path}")


    legend_fig, legend_ax = plt.subplots(figsize=(3, 1))
    legend_ax.axis('off')
    legend = legend_ax.legend(
        handles=[line1, line2],
        loc='center',
        frameon=False,
        fontsize=15,
        ncol=2
    )
    legend_fig_path = os.path.join(output_dir, f"patient_{patient_id}_legend.pdf")
    legend_fig.savefig(legend_fig_path, bbox_inches='tight')
    plt.close(legend_fig)
    print(f"Legend saved separately to {legend_fig_path}")

# >>> paste from original: def analyze_single_patient <<<
def analyze_single_patient(pretrained_model, patient_data, patient_id, steps, output_dir="./results", 
                           neworder=None, fig_size=(12, 10), fine_tune=True, fine_tune_params=None, 
                           max_analysis_steps=1200,
                           healthy_csv_path="/ailab/group/medai-share/syDu/Brain_EC/HCP/HCP_AAL3_csv_out/187547_AAL3_ts.csv",
                           healthy_model_path="/ailab/user/dusiyuan/code/Brain/EC/AAL3/EC_results_num_layer_2/AAL3_lr_5e-05_batch_256_epochs_300_l2_0.0001_patience_100_steps_7_dmodel_256/patient_models/model_187547_AAL3_ts.pth"):
 
    
    ensure_dir(output_dir)
    inputs, targets, label = patient_data

    # ====== Truncate long time series ======
    original_T = targets.shape[0]
    if original_T > max_analysis_steps:
        print(f"Truncating patient {patient_id} time series from {original_T} to {max_analysis_steps} steps.")
        inputs = inputs[:max_analysis_steps]
        targets = targets[:max_analysis_steps]
        if isinstance(label, np.ndarray) and label.ndim > 0 and len(label) == original_T:
            label = label[:max_analysis_steps]
    # ===================================


    
    if fine_tune:
        # ====== Target 1: Calculate patient anomaly using healthy VTB model ======
        print("\n===== TARGET 1: Calculating patient anomaly using healthy VTB model =====")
        # 1. Load healthy pre-trained model (kept fixed)
        healthy_model = pretrained_model.__class__(**pretrained_model.init_args)
        healthy_model.load_state_dict(torch.load(healthy_model_path, map_location=device)['model_state_dict'])
        healthy_model = healthy_model.to(device)
        healthy_model.eval()
        
        # 2. Predict patient data using healthy model
        with torch.no_grad():
            label_tensor = torch.tensor(label, dtype=torch.long).to(device)
            healthy_pred_on_patient = healthy_model(
                torch.tensor(inputs, dtype=torch.float).to(device), 
                label_tensor
            ).detach().cpu().numpy()
        
        # 3. Compute difference (ground truth - prediction) and average over time steps
        anomaly_diff = targets - healthy_pred_on_patient  # [T, ROI]
        mean_anomaly = np.mean(anomaly_diff, axis=0)      # [ROI]
        # 4. Save results

        np.save(os.path.join(output_dir, f"patient_{patient_id}_anomaly_diff.npy"), anomaly_diff)
        np.save(os.path.join(output_dir, f"patient_{patient_id}_mean_anomaly.npy"), mean_anomaly)
        print(f"Target 1 results saved to {output_dir}")
        
        print(f"\nFine-tuning model on patient {patient_id} data...")
        if fine_tune_params is None:
            fine_tune_params = {'batch_size': 64, 'num_epochs': 50, 'lr': 1e-5, 'l2': 0, 'patience': 10}
        patient_fine_tune_dir = os.path.join(output_dir, f"patient_{patient_id}_fine_tune")
        ensure_dir(patient_fine_tune_dir)
        fine_tuned_model, ft_train_losses, ft_val_losses = fine_tune_for_patient(
            pretrained_model, patient_data, output_dir=patient_fine_tune_dir, **fine_tune_params
        )
        model = fine_tuned_model
    else:
        model = pretrained_model

    print(f"Analyzing patient {patient_id}...")
    node_num = targets.shape[1]
    time_series = np.zeros((inputs.shape[0] + steps, node_num))
    for i in range(steps):
        time_series[i] = inputs[0, i*node_num:(i+1)*node_num]
    for i in range(inputs.shape[0]):
        time_series[i + steps] = targets[i]
        
    empirical_FC = safe_corrcoef(time_series)
    
    with torch.no_grad():

        label_tensor = torch.tensor(label, dtype=torch.long).to(device)
        predicted_targets = model(torch.tensor(inputs, dtype=torch.float).to(device), label_tensor).detach().cpu().numpy()
        
    model_FC_matrix = safe_corrcoef(predicted_targets)
        
    print(predicted_targets.shape, targets.shape)
    np.save(os.path.join(output_dir, f"patient_{patient_id}_predicted_targets.npy"), predicted_targets)
    np.save(os.path.join(output_dir, f"patient_{patient_id}_GT_targets.npy"), targets)
    
    print(f"Model predictions saved to {os.path.join(output_dir, f'patient_{patient_id}_predicted_targets.npy')}")
    print(f"GT saved to {os.path.join(output_dir, f'patient_{patient_id}_GT_targets.npy')}")

    mae = mean_absolute_error(targets, predicted_targets)
    r2 = r2_score(targets, predicted_targets) 
      
    NPI_EC = _model_EC(model, inputs, targets, pert_strength=1.0)
    np.fill_diagonal(NPI_EC, 0)

    r_value, p_value = pearsonr(flat_without_diagonal(model_FC_matrix), 
                                flat_without_diagonal(empirical_FC))

    plt.figure(figsize=fig_size)
    plt.subplot(2, 2, 1)
    if neworder is not None:
        sns.heatmap(empirical_FC[neworder].T[neworder].T, vmin=-1.0, vmax=1.0, cmap='RdBu_r', cbar=True)
    else:
        sns.heatmap(empirical_FC, vmin=-1.0, vmax=1.0, cmap='RdBu_r', cbar=True)
    plt.title(f'Empirical FC - Patient {patient_id}')

    plt.subplot(2, 2, 2)
    if neworder is not None:
        sns.heatmap(model_FC_matrix[neworder].T[neworder].T, vmin=-1.0, vmax=1.0, cmap='RdBu_r', cbar=True)
    else:
        sns.heatmap(model_FC_matrix, vmin=-1.0, vmax=1.0, cmap='RdBu_r', cbar=True)
    plt.title(f'Model FC - Patient {patient_id}')


    x = flat_without_diagonal(model_FC_matrix)
    y = flat_without_diagonal(empirical_FC)
    
    r_value, p_value, r2_value, slope = plot_enhanced_fc_correlation(
        x, y, patient_id, output_dir, 
        fig_size=(5, 5), 
        dpi=100,
        max_points=10000
    )
     
    scatter_fig, scatter_ax = plt.subplots(figsize=(5, 5))
    scatter_ax.scatter(x, y, s=10, color='#2C3E50', alpha=0.7)
    scatter_ax.set_xlim(-0.7, 1.1)
    scatter_ax.set_ylim(-0.7, 1.1)
    scatter_ax.set_xticks([-0.5, 0.0, 1.0])
    scatter_ax.set_yticks([-0.5, 0.0, 1.0])
    scatter_ax.tick_params(labelsize=12)
    scatter_ax.set_xlabel('Model FC', fontsize=13)
    scatter_ax.set_ylabel('Empirical FC', fontsize=13)
    scatter_ax.spines['right'].set_visible(False)
    scatter_ax.spines['top'].set_visible(False)

    scatter_fig.tight_layout()
    scatter_path = os.path.join(output_dir, f"patient_{patient_id}_FC_correlation.pdf")
    scatter_fig.savefig(scatter_path, bbox_inches='tight')
    plt.close(scatter_fig)
    print(f"FC correlation plot saved separately to {scatter_path}")


    plt.subplot(2, 2, 4)
    if neworder is not None:
        left_hemi_size = len(neworder) // 2
        sns.heatmap(NPI_EC[neworder].T[neworder].T[:left_hemi_size, :left_hemi_size], vmin=-0.1, vmax=0.1, cmap='RdBu_r', cbar=True)
    else:
        sns.heatmap(NPI_EC, vmin=-0.1, vmax=0.1, cmap='RdBu_r', cbar=True)
    plt.title(f'NPI_EC - Patient {patient_id}')
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"patient_{patient_id}_analysis.pdf")
    plt.savefig(fig_path, bbox_inches='tight'); plt.close()

    np.save(os.path.join(output_dir, f"patient_{patient_id}_empirical_FC.npy"), empirical_FC)
    np.save(os.path.join(output_dir, f"patient_{patient_id}_model_FC.npy"), model_FC_matrix)
    np.save(os.path.join(output_dir, f"patient_{patient_id}_NPI_EC.npy"), NPI_EC)
    
    # ====== TARGET 2: Calculate healthy data distortion using patient VTB model ======
    if fine_tune:  
        print("\n===== TARGET 2: Calculating healthy distortion using patient VTB model =====")
        
        # 1. Load healthy subject data (using same preprocessing as during training)
        healthy_inputs, healthy_targets, healthy_labels, _ = load_patient_data(
            patient_file=healthy_csv_path,
            steps=steps,
            skip_first=30,  # 与原始训练参数一致
            dataset_type='HCP',  # 明确指定为HCP数据
            label_map=None
        )
        
        # 2. Truncate to within max_analysis_steps
        if healthy_targets.shape[0] > max_analysis_steps:
            healthy_inputs = healthy_inputs[:max_analysis_steps]
            healthy_targets = healthy_targets[:max_analysis_steps]
            healthy_labels = healthy_labels[:max_analysis_steps]
            
        # 3. Predict healthy data using patient fine-tuned model
        with torch.no_grad():
            healthy_label_tensor = torch.tensor(healthy_labels, dtype=torch.long).to(device)
            patient_model_pred_on_healthy = model(
                torch.tensor(healthy_inputs, dtype=torch.float).to(device),
                healthy_label_tensor
            ).detach().cpu().numpy()
        
        # 4. Compute difference (prediction - ground truth) and average over time steps
        distortion_diff = patient_model_pred_on_healthy - healthy_targets  # [T, ROI]
        mean_distortion = np.mean(distortion_diff, axis=0)                # [ROI]
        
        # 5. Save results
        np.save(os.path.join(output_dir, f"patient_{patient_id}_distortion_diff.npy"), distortion_diff)
        np.save(os.path.join(output_dir, f"patient_{patient_id}_mean_distortion.npy"), mean_distortion)
        print(f"Target 2 results saved to {output_dir}")

            

    with open(os.path.join(output_dir, f"patient_{patient_id}_metrics.txt"), "w") as f:
        f.write(f"Patient ID: {patient_id}\n")
        f.write(f"FC Correlation (r): {r_value:.4f}\n")
        f.write(f"FC p-value: {p_value:.4e}\n")
        f.write(f"MAE (BOLD): {mae:.6f}\n")
        f.write(f"R² (BOLD): {r2:.6f}\n")
        f.write(f"Mean absolute EC: {np.mean(np.abs(NPI_EC)):.6f}\n")
        f.write(f"Max EC: {np.max(NPI_EC):.6f}\n")
        f.write(f"Min EC: {np.min(NPI_EC):.6f}\n")

        if fine_tune:
            f.write("\n=== TARGET 1 METRICS ===\n")
            f.write(f"Mean Anomaly (abs): {np.mean(np.abs(mean_anomaly)):.6f}\n")
            f.write(f"Max Anomaly: {np.max(mean_anomaly):.6f}\n")
            f.write(f"Min Anomaly: {np.min(mean_anomaly):.6f}\n")
            
            f.write("\n=== TARGET 2 METRICS ===\n")
            f.write(f"Mean Distortion (abs): {np.mean(np.abs(mean_distortion)):.6f}\n")
            f.write(f"Max Distortion: {np.max(mean_distortion):.6f}\n")
            f.write(f"Min Distortion: {np.min(mean_distortion):.6f}\n")
        
        
    print(f"Analysis for patient {patient_id} completed. Results saved to {output_dir}")
    return (empirical_FC, model_FC_matrix, NPI_EC, model) if fine_tune else (empirical_FC, model_FC_matrix, NPI_EC)

    summary_csv_path = "/ailab/group/medai-share/syDu/ruijin/Final_data/all_patients_summary.csv"
    file_exists = os.path.isfile(summary_csv_path)


    with open(summary_csv_path, mode='a', newline='') as csvfile:
        fieldnames = ['patient_id', 'FC_r', 'FC_p', 'MAE', 'R2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'patient_id': patient_id,
            'FC_r': f"{r_value:.6f}",
            'FC_p': f"{p_value:.4e}",
            'MAE': f"{mae:.6f}",
            'R2': f"{r2:.6f}"
        })
    
    print(f"Core metrics for patient {patient_id} appended to {summary_csv_path}")