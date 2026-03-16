#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 4D 的 func_MNI.nii.gz 使用 AAL3 脑谱图提取各脑区 BOLD 时序，导出 CSV。
依赖:
  - nibabel
  - numpy
  - pandas
  - nibabel>=3.2（需要 nibabel.processing.resample_from_to）
示例:
python extract_roi_ts_aal3.py \
--func func_MNI.nii.gz \
--atlas /ailab/group/medai-share/syDu/Brain_EC/AAL_atlas/AAL3/AAL3v1_1mm.nii.gz \
--out ts_aal3.csv \
--out_sizes roi_sizes.csv \
--tr 0.72
说明:
- 默认会将 atlas 最近邻重采样到 func 的体素网格以确保空间一致。
- 输出 CSV 的列顺序按 AAL3 标签值升序排列 (1,2,...,170)。
- 输出表格中直接使用 1,2,3,...,170 作为列名（对应 AAL3 的 170 个脑区）。
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib

# ---- 影像 I/O 与重采样 ----
try:
    from nibabel.processing import resample_from_to
except Exception:
    resample_from_to = None

def load_img(path):
    return nib.load(path)

def resample_atlas_to_func(atlas_img, func_img):
    if resample_from_to is None:
        raise RuntimeError("需要 nibabel>=3.2 提供 nibabel.processing.resample_from_to。")
    # 目标网格 = func 的 (shape, affine)；使用最近邻保持整数标签
    atlas_resamp = resample_from_to(
        atlas_img,
        (func_img.shape[:3], func_img.affine),
        order=0
    )
    return atlas_resamp

# ---- 提取时序 ----
def extract_timeseries(func_img, atlas_img, labels_keep=None, nan_policy="omit"):
    """
    从 4D func 中按 atlas 分区提取均值时序
    返回:
      ts: np.ndarray (T, n_rois)
      roi_indices: List[int] (实际提取的标签值)
      roi_sizes: Dict[index, n_vox]
    """
    func_data = func_img.get_fdata(dtype=np.float32)
    if func_data.ndim != 4:
        raise ValueError("功能像应为 4D NIfTI。")
    X, Y, Z, T = func_data.shape
    atlas_data = atlas_img.get_fdata()
    if atlas_data.ndim != 3:
        raise ValueError("分区图应为 3D NIfTI。")
    atlas_labels = np.rint(atlas_data).astype(np.int32)
    if atlas_labels.shape != (X, Y, Z):
        raise ValueError(f"atlas 空间与 func 不一致：atlas {atlas_labels.shape} vs func {(X, Y, Z)}")
    
    func_data = np.where(np.isfinite(func_data), func_data, np.nan)
    func_2d = func_data.reshape(-1, T)
    atlas_1d = atlas_labels.reshape(-1)
    
    # 确定要提取的标签：跳过0（背景），按升序排列
    if labels_keep is None:
        uniq = np.unique(atlas_1d)
        labels_keep = [int(l) for l in uniq if l != 0]
        labels_keep.sort()  # 按数值升序排序
    
    ts_list = []
    roi_sizes = {}
    kept_indices = []
    for lab in labels_keep:
        idx_mask = (atlas_1d == lab)
        nvox = int(idx_mask.sum())
        if nvox == 0:
            continue
        roi_data = func_2d[idx_mask, :]  # (nvox, T)
        if nan_policy == "omit":
            roi_ts = np.nanmean(roi_data, axis=0)
        elif nan_policy == "zero":
            roi_ts = np.mean(np.nan_to_num(roi_data, nan=0.0), axis=0)
        else:
            raise ValueError("nan_policy 仅支持 'omit' 或 'zero'。")
        ts_list.append(roi_ts)
        roi_sizes[lab] = nvox
        kept_indices.append(lab)
    
    if not ts_list:
        raise RuntimeError("在 atlas 中未找到任何与 func 重叠的 ROI 体素。")
    ts = np.vstack(ts_list).T  # (T, n_kept)
    return ts, kept_indices, roi_sizes

# ---- 主程序 ----
def main():
    ap = argparse.ArgumentParser(description="基于 AAL3 脑谱图提取各脑区 BOLD 时序到 CSV")
    ap.add_argument("--func", required=True, help="4D 功能像（MNI 空间），如 func_MNI.nii.gz")
    ap.add_argument("--atlas", default='/ailab/group/medai-share/syDu/Brain_EC/AAL_atlas/AAL3/AAL3v1_1mm.nii.gz',
                    help="3D AAL3 分区图 (1mm)")
    ap.add_argument("--out", required=True, help="输出时序 CSV 路径")
    ap.add_argument("--out_sizes", default=None, help="可选：输出每个 ROI 体素数的 CSV")
    ap.add_argument("--tr", type=float, default=None, help="可选：TR（秒），用于 time_sec 列")
    ap.add_argument("--no_resample", action="store_true", help="atlas 与 func 已同空间时跳过重采样")
    ap.add_argument("--nan_policy", choices=["omit", "zero"], default="omit", help="NaN 处理策略")
    args = ap.parse_args()
    
    # 读入影像
    func_img = load_img(args.func)
    atlas_img = load_img(args.atlas)
    
    # 重采样
    atlas_to_func = atlas_img if args.no_resample else resample_atlas_to_func(atlas_img, func_img)
    
    # 获取 AAL3 中所有非零标签 (1-170) 并排序
    atlas_data = atlas_to_func.get_fdata()
    all_labels = np.unique(np.rint(atlas_data).astype(np.int32))
    aal3_labels = [int(l) for l in all_labels if l != 0]
    aal3_labels.sort()  # 确保顺序为 1,2,3,...,170
    
    # 提取时序
    ts_array, kept_indices, roi_sizes = extract_timeseries(
        func_img=func_img,
        atlas_img=atlas_to_func,
        labels_keep=aal3_labels,  # 按 AAL3 标准顺序 1-170
        nan_policy=args.nan_policy
    )
    
    # 直接使用标签值作为列名 (1,2,3,...,170)
    col_names = [str(idx) for idx in kept_indices]
    
    # 组装 DataFrame：行=T，列=ROI
    ts_df = pd.DataFrame(ts_array, columns=col_names)
    ts_df.index.name = "timepoint"
    if args.tr is not None:
        ts_df.insert(0, "time_sec", ts_df.index.values * float(args.tr))
    
    # 输出时序 CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ts_df.to_csv(args.out, index=False)
    
    # 输出 ROI 体素数（只包含 index 和 n_voxels）
    if args.out_sizes:
        sizes_rows = []
        for idx in kept_indices:
            sizes_rows.append({
                "index": idx,
                "n_voxels": roi_sizes.get(idx, 0)
            })
        sizes_df = pd.DataFrame(sizes_rows)
        sizes_df = sizes_df.sort_values("index")
        sizes_df.to_csv(args.out_sizes, index=False)
    
    # 简短日志
    sys.stderr.write(f"[OK] Saved ROI time series: {args.out}\n")
    if args.out_sizes:
        sys.stderr.write(f"[OK] Saved ROI voxel counts: {args.out_sizes}\n")

if __name__ == "__main__":
    main()