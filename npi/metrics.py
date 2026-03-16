# npi/metrics.py
import numpy as np
import torch
from .utils import device

# >>> paste from original: def corrcoef <<<
def corrcoef(signals):
    return torch.corrcoef(torch.tensor(signals.T, dtype=torch.float).to(device)).detach().cpu().numpy()

# >>> paste from original: def model_FC <<<
def model_FC(model, node_num, steps, sim_steps=600):
    NN_sim = []
    for _ in range(steps):
        NN_sim.append(np.zeros(node_num))
    for _ in range(sim_steps):
        noise = 0.1 * np.random.randn(steps * node_num)
        model_input = np.array(NN_sim[-steps:]).flatten() + noise
        with torch.no_grad():
            pred = model(torch.tensor(model_input, dtype=torch.float).to(device)).detach().cpu().numpy()
            if pred.ndim > 1:  # 兼容 RNN 的 [1, D] 输出
                pred = pred[0]
        NN_sim.append(pred)
    NN_sim = np.array(NN_sim[steps:])
    
    # 计算功能连接
    fc = corrcoef(NN_sim)

    # 🚨 检查 FC 矩阵是否包含 NaN 或 inf
    if not np.all(np.isfinite(fc)):
        nan_count = np.sum(np.isnan(fc))
        inf_count = np.sum(np.isinf(fc))
        print(f"⚠️  Warning: model_FC matrix contains {nan_count} NaNs and {inf_count} infs. "
              f"This is often caused by constant/zero time series or model instability. "
              f"Replacing with 0.")

        # 清理：NaN/inf → 0
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    
    return fc

# >>> paste from original: def model_EC <<<
def model_EC(model, input_X, target_Y, pert_strength=1.0):
    node_num = target_Y.shape[1]
    steps = int(input_X.shape[1] / node_num)
    NPI_EC = np.zeros((node_num, node_num))
    import numpy as _np
    from tqdm import tqdm as _tqdm
    print("Calculating Effective Connectivity (EC) using perturbation...")
    for node in _tqdm(range(node_num), desc="Perturbing brain regions"):
        with torch.no_grad():
            unperturbed_output = model(torch.tensor(input_X, dtype=torch.float).to(device)).detach().cpu().numpy()
        perturbation = _np.zeros((steps, node_num))
        perturbation[-1, node] = pert_strength
        with torch.no_grad():
            perturbed_output = model(torch.tensor(input_X + perturbation.flatten(), dtype=torch.float).to(device)).detach().cpu().numpy()
        effects = perturbed_output - unperturbed_output
        NPI_EC[node] = _np.mean(effects, axis=0)
    return NPI_EC

# >>> paste from original: def model_Jacobian <<<
def model_Jacobian(model, input_X, steps):
    node_num = int(input_X.shape[1] / steps)
    jacobian = np.zeros((node_num, node_num))
    print("Calculating Jacobian matrix...")
    with torch.no_grad():
        for i in range(min(1000, input_X.shape[0])):
            x_t = torch.tensor(input_X[i], dtype=torch.float).to(device)
            if hasattr(model, "forward") and "ANN_RNN" in model.__class__.__name__:
                jac = torch.autograd.functional.jacobian(lambda x: model(x)[0, :], x_t).cpu().detach().numpy()[:, -node_num:]
            else:
                jac = torch.autograd.functional.jacobian(model, x_t).cpu().detach().numpy()[:, -node_num:]
            jacobian += jac
    jacobian_EC = jacobian.T / min(1000, input_X.shape[0])
    return jacobian_EC

# >>> paste from original: def flat_without_diagonal <<<
def flat_without_diagonal(matrix):
    n = matrix.shape[0]
    flattened = []
    for i in range(n):
        for j in list(range(i)) + list(range(i + 1, n)):
            flattened.append(matrix[i][j])
    return np.array(flattened)
