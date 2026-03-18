"""
Batch processing for HCP-PTN -> AAL3 ROI BOLD time series (CSV)
---------------------------------------------------------------
Inputs:
  1) HCP dtseries.nii files (spatial basis)
  2) HCP node_timeseries.txt files (temporal coefficients)
  3) AAL3 mapping file (.npy): an array of length 91282 with values in 0-170

Output:
  out_dir/100206_AAL3_ts.csv
     - Rows = timepoints (4800)
     - Columns = ROIs (1, 2, 3, ..., 170)

Method:
  Y_roi = T @ A
  where A is the mean of S within each AAL3 ROI.
"""

import os
import csv
import glob
import argparse
import warnings
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed

# Prevent excessive BLAS thread contention
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def find_subjects(dtseries_dir, node_dir, max_show=20):
    """Find subject IDs that exist in both input directories."""
    dt_paths = []
    for pat in ("*.dtseries.nii", "*.dtseries.nii.gz"):
        dt_paths.extend(glob.glob(os.path.join(dtseries_dir, pat)))
    dt_ids = {
        os.path.splitext(os.path.basename(p))[0].replace(".dtseries", "")
        for p in dt_paths
    }

    tx_paths = glob.glob(os.path.join(node_dir, "*.txt"))
    tx_ids = {os.path.splitext(os.path.basename(p))[0] for p in tx_paths}

    common_ids = sorted(dt_ids & tx_ids)

    if not common_ids:
        raise RuntimeError("No common subject IDs were found in the provided directories.")

    return common_ids


def load_aal3_npy(npy_path):
    """
    Load the preprocessed AAL3 .npy file (shape: 91282,).

    Returns:
      roi_names: List[str], e.g. ['1', '2', ..., '170']
      roi_indices: List[np.ndarray], vertex indices for each label
      n_grayords: int, total number of grayordinates (91282)
    """
    print(f"Loading AAL3 template: {npy_path}")
    labels = np.load(npy_path)
    n_grayords = labels.size

    unique_labels = sorted(np.unique(labels))

    roi_names = []
    roi_indices = []

    for label_val in unique_labels:
        # Label 0 is background and should be skipped
        if int(label_val) == 0:
            continue

        idx = np.where(labels == label_val)[0]
        roi_indices.append(idx)
        roi_names.append(str(int(label_val)))

    if not roi_names:
        raise RuntimeError("No valid ROI labels (> 0) were found in the npy file.")

    return roi_names, roi_indices, n_grayords


def load_S(dtseries_path, n_grayords_expected):
    """Load S (spatial basis), expected shape: (300, n_grayords)."""
    img = nib.load(dtseries_path)
    S = img.get_fdata(dtype=np.float32)

    if S.ndim != 2:
        raise ValueError(f"Unexpected dimensions in {dtseries_path}: {S.shape}")

    # Handle possible transposed layout: (n_grayords, 300)
    if S.shape[1] != n_grayords_expected and S.shape[0] == n_grayords_expected:
        S = S.T

    return S


def load_T(node_txt_path, n_components_expected=300):
    """Load T (node time series), expected shape: (4800, 300)."""
    T = np.loadtxt(node_txt_path, dtype=np.float32)

    if T.ndim != 2:
        raise ValueError(f"Unexpected dimensions in {node_txt_path}: {T.shape}")

    if T.shape[1] != n_components_expected and T.shape[0] == n_components_expected:
        T = T.T

    return T


def compute_A_S_roi_mean(S, roi_indices):
    """
    Compute the column-wise mean of S within each ROI.

    Returns:
      A in R^(300 x R)
    """
    R = len(roi_indices)
    A = np.empty((S.shape[0], R), dtype=np.float32)
    n_grayords = S.shape[1]

    for j, idx in enumerate(roi_indices):
        # Ensure indices do not exceed the dtseries grayordinate range
        valid_idx = idx[idx < n_grayords]

        if valid_idx.size == 0:
            A[:, j] = np.nan
            continue

        col = S[:, valid_idx]
        if np.isnan(col).any():
            A[:, j] = np.nanmean(col, axis=1)
        else:
            A[:, j] = col.mean(axis=1)

    return A


def save_csv(out_path, header_names, data_2d):
    """Save a 2D array to CSV."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_names)
        for i in range(data_2d.shape[0]):
            writer.writerow(data_2d[i, :].tolist())


def process_one_subject(
    sub_id,
    dtseries_dir,
    node_dir,
    roi_indices,
    roi_names,
    n_grayords,
    out_dir,
    overwrite=False,
):
    """Process a single subject."""
    dt_path = os.path.join(dtseries_dir, f"{sub_id}.dtseries.nii")
    node_path = os.path.join(node_dir, f"{sub_id}.txt")
    out_path = os.path.join(out_dir, f"{sub_id}_AAL3_ts.csv")

    if (not overwrite) and os.path.exists(out_path):
        return (sub_id, "skipped", "Output already exists, skipped.")

    try:
        # 1. Load S: (300 x 91282)
        S = load_S(dt_path, n_grayords)

        # 2. Load T: (4800 x 300)
        T = load_T(node_path, n_components_expected=S.shape[0])

        # 3. Compute A: ROI-averaged S -> (300 x 170)
        A = compute_A_S_roi_mean(S, roi_indices)

        # 4. Compute final ROI time series: (4800 x 170)
        Y_roi = T @ A

        # 5. Save result
        save_csv(out_path, roi_names, Y_roi.astype(np.float32))

        return (sub_id, "ok", f"Saved to {out_path}")
    except Exception as e:
        return (sub_id, "fail", repr(e))


def main():
    parser = argparse.ArgumentParser(
        description="Batch processing for HCP-PTN -> AAL3 (npy) ROI BOLD time series CSV"
    )

    parser.add_argument(
        "--dtseries_dir",
        default="",
        help="Directory containing *.dtseries.nii files",
    )
    parser.add_argument(
        "--node_dir",
        default="",
        help="Directory containing *.txt files",
    )
    parser.add_argument(
        "--atlas_npy",
        default="/Data_process/HCP/AAL3v1.91k_fs_LR.npy",
        help="Path to the AAL3 91k .npy file",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="Output directory for CSV files",
    )

    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    args = parser.parse_args()

    # 1. Load AAL3 npy
    roi_names, roi_indices, n_grayords = load_aal3_npy(args.atlas_npy)
    print(f"Number of ROIs: {len(roi_names)} (expected: 170), grayordinates: {n_grayords}")
    print(f"ROI name examples: {roi_names[:5]} ... {roi_names[-5:]}")

    # 2. Find subjects
    ids = find_subjects(args.dtseries_dir, args.node_dir)
    print(f"Found {len(ids)} subjects; output directory: {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    # 3. Parallel processing
    results = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = []
        for sub_id in ids:
            fut = ex.submit(
                process_one_subject,
                sub_id,
                args.dtseries_dir,
                args.node_dir,
                roi_indices,
                roi_names,
                n_grayords,
                args.out_dir,
                args.overwrite,
            )
            futs.append(fut)

        for fut in as_completed(futs):
            results.append(fut.result())
            sub_id, status, msg = results[-1]
            tag = {"ok": "✓", "skipped": "-", "fail": "✗"}.get(status, "?")
            print(f"[{tag}] {sub_id}: {msg}")

    # 4. Summary
    n_ok = sum(1 for _, s, _ in results if s == "ok")
    n_sk = sum(1 for _, s, _ in results if s == "skipped")
    n_fl = sum(1 for _, s, _ in results if s == "fail")
    print(f"Done: success {n_ok}, skipped {n_sk}, failed {n_fl}")

    if n_fl:
        print("Failure details:")
        for sub_id, s, msg in results:
            if s == "fail":
                print(f"  - {sub_id}: {msg}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()