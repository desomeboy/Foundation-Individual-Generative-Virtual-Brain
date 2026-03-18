"""
Extract regional BOLD time series from a 4D func_MNI.nii.gz file using the AAL3 atlas,
and export the result to CSV.

Dependencies:
  - nibabel>=3.2 (requires nibabel.processing.resample_from_to)

Notes:
- By default, the atlas will be resampled to the functional image grid using
  nearest-neighbor interpolation to ensure spatial alignment.
- The output CSV columns are ordered by ascending AAL3 label values.
- Column names use the atlas label values directly (for example: 1, 2, ..., 170).
- Although the AAL3 index range is 1-170, the atlas may contain only 166 regions
  with voxels, meaning a few label IDs may not appear in practice.
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

# ---- Image I/O and resampling ----
try:
    from nibabel.processing import resample_from_to
except Exception:
    resample_from_to = None


def load_img(path):
    return nib.load(path)


def resample_atlas_to_func(atlas_img, func_img):
    if resample_from_to is None:
        raise RuntimeError(
            "nibabel>=3.2 is required for nibabel.processing.resample_from_to."
        )

    # Target grid = functional image spatial grid; nearest-neighbor preserves labels.
    atlas_resampled = resample_from_to(
        atlas_img,
        (func_img.shape[:3], func_img.affine),
        order=0,
    )
    return atlas_resampled


# ---- Time series extraction ----
def extract_timeseries(func_img, atlas_img, labels_keep=None, nan_policy="omit"):
    """
    Extract mean ROI time series from a 4D functional image using a 3D atlas.

    Returns:
        ts: np.ndarray, shape (T, n_rois)
        roi_indices: list[int], extracted atlas label values
        roi_sizes: dict[int, int], number of voxels per ROI
    """
    func_data = func_img.get_fdata(dtype=np.float32)
    if func_data.ndim != 4:
        raise ValueError("Functional image must be a 4D NIfTI image.")

    x_dim, y_dim, z_dim, n_timepoints = func_data.shape

    atlas_data = atlas_img.get_fdata()
    if atlas_data.ndim != 3:
        raise ValueError("Atlas image must be a 3D NIfTI image.")

    atlas_labels = np.rint(atlas_data).astype(np.int32)
    if atlas_labels.shape != (x_dim, y_dim, z_dim):
        raise ValueError(
            f"Atlas space does not match functional image space: "
            f"atlas {atlas_labels.shape} vs func {(x_dim, y_dim, z_dim)}"
        )

    func_data = np.where(np.isfinite(func_data), func_data, np.nan)
    func_2d = func_data.reshape(-1, n_timepoints)
    atlas_1d = atlas_labels.reshape(-1)

    # Determine labels to extract: exclude 0 (background), sort in ascending order.
    if labels_keep is None:
        unique_labels = np.unique(atlas_1d)
        labels_keep = [int(label) for label in unique_labels if label != 0]
        labels_keep.sort()

    ts_list = []
    roi_sizes = {}
    kept_indices = []

    for label in labels_keep:
        roi_mask = atlas_1d == label
        n_voxels = int(roi_mask.sum())
        if n_voxels == 0:
            continue

        roi_data = func_2d[roi_mask, :]  # shape: (n_voxels, T)

        if nan_policy == "omit":
            roi_ts = np.nanmean(roi_data, axis=0)
        elif nan_policy == "zero":
            roi_ts = np.mean(np.nan_to_num(roi_data, nan=0.0), axis=0)
        else:
            raise ValueError("nan_policy must be either 'omit' or 'zero'.")

        ts_list.append(roi_ts)
        roi_sizes[label] = n_voxels
        kept_indices.append(label)

    if not ts_list:
        raise RuntimeError("No overlapping ROI voxels were found between atlas and func.")

    ts = np.vstack(ts_list).T  # shape: (T, n_kept)
    return ts, kept_indices, roi_sizes


# ---- Main program ----
def main():
    parser = argparse.ArgumentParser(
        description="Extract ROI-wise BOLD time series from a 4D image using the AAL3 atlas."
    )
    parser.add_argument(
        "--func",
        required=True,
        help="4D functional image in MNI space, e.g. func_MNI.nii.gz",
    )
    parser.add_argument(
        "--atlas",
        default="Data_process/AAL3v1_1mm.nii.gz",
        help="3D AAL3 atlas file",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path for ROI time series",
    )
    parser.add_argument(
        "--out_sizes",
        default=None,
        help="Optional output CSV path for ROI voxel counts",
    )
    parser.add_argument(
        "--tr",
        type=float,
        default=None,
        help="Optional TR in seconds; adds a time_sec column if provided",
    )
    parser.add_argument(
        "--no_resample",
        action="store_true",
        help="Skip atlas resampling if atlas and functional image are already aligned",
    )
    parser.add_argument(
        "--nan_policy",
        choices=["omit", "zero"],
        default="omit",
        help="NaN handling strategy",
    )
    args = parser.parse_args()

    # Load images
    func_img = load_img(args.func)
    atlas_img = load_img(args.atlas)

    # Resample atlas if needed
    atlas_to_func = atlas_img if args.no_resample else resample_atlas_to_func(atlas_img, func_img)

    # Collect all non-zero labels present in the atlas after resampling
    atlas_data = atlas_to_func.get_fdata()
    all_labels = np.unique(np.rint(atlas_data).astype(np.int32))
    aal3_labels = [int(label) for label in all_labels if label != 0]
    aal3_labels.sort()

    # Extract time series
    ts_array, kept_indices, roi_sizes = extract_timeseries(
        func_img=func_img,
        atlas_img=atlas_to_func,
        labels_keep=aal3_labels,
        nan_policy=args.nan_policy,
    )

    # Use atlas label values directly as column names
    col_names = [str(idx) for idx in kept_indices]

    # Build output dataframe: rows=timepoints, columns=ROIs
    ts_df = pd.DataFrame(ts_array, columns=col_names)
    ts_df.index.name = "timepoint"

    if args.tr is not None:
        ts_df.insert(0, "time_sec", ts_df.index.values * float(args.tr))

    # Save time series CSV
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ts_df.to_csv(args.out, index=False)

    # Save ROI voxel counts if requested
    if args.out_sizes:
        sizes_rows = []
        for idx in kept_indices:
            sizes_rows.append(
                {
                    "index": idx,
                    "n_voxels": roi_sizes.get(idx, 0),
                }
            )
        sizes_df = pd.DataFrame(sizes_rows).sort_values("index")

        out_sizes_dir = os.path.dirname(os.path.abspath(args.out_sizes))
        if out_sizes_dir:
            os.makedirs(out_sizes_dir, exist_ok=True)

        sizes_df.to_csv(args.out_sizes, index=False)

    # Minimal logging
    sys.stderr.write(f"[OK] Saved ROI time series: {args.out}\n")
    if args.out_sizes:
        sys.stderr.write(f"[OK] Saved ROI voxel counts: {args.out_sizes}\n")


if __name__ == "__main__":
    main()