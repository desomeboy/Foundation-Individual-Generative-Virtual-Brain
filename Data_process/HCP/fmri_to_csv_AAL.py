#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批处理 HCP-PTN -> AAL3 ROI (基于预处理好的 .npy 标签) 的 BOLD 时序 (CSV)
--------------------------------------------------------------------------------
输入：
  1) HCP dtseries.nii (空间基)
  2) HCP node_timeseries.txt (时间系数)
  3) AAL3 映射文件 (.npy): 长度为 91282 的数组，值为 0-170

输出：
  out_dir/100206_AAL3_ts.csv
     - 行 = Timepoints (4800)
     - 列 = ROIs (1, 2, 3, ..., 170)

方法：
  Y_roi = T @ A
  其中 A 是 S 在每个 AAL3 ROI 上的均值。
"""

import os
import sys
import csv
import glob
import argparse
import warnings
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed

# 防止 BLAS 线程过度竞争
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def find_subjects(dtseries_dir, node_dir, max_show=20):
    """找出两个文件夹共有的被试 ID"""
    dt_paths = []
    for pat in ("*.dtseries.nii", "*.dtseries.nii.gz"):
        dt_paths.extend(glob.glob(os.path.join(dtseries_dir, pat)))
    dt_ids = {os.path.splitext(os.path.basename(p))[0].replace(".dtseries", "") for p in dt_paths}
    
    tx_paths = glob.glob(os.path.join(node_dir, "*.txt"))
    tx_ids = {os.path.splitext(os.path.basename(p))[0] for p in tx_paths}
    
    common_ids = sorted(dt_ids & tx_ids)
    
    if not common_ids:
        raise RuntimeError("在提供的目录中未找到共同的被试 ID。")
        
    return common_ids

def load_aal3_npy(npy_path):
    """
    读取预处理好的 AAL3 .npy 文件 (shape: 91282,)
    返回：
      roi_names: List[str] ['1', '2', ..., '170']
      roi_indices: List[np.ndarray] 对应每个 label 的顶点索引
      n_grayords: int 总顶点数 (91282)
    """
    print(f"正在加载 AAL3 模板: {npy_path}")
    labels = np.load(npy_path)
    n_grayords = labels.size
    
    # 获取所有唯一的标签值，并排序
    unique_labels = sorted(np.unique(labels))
    
    roi_names = []
    roi_indices = []
    
    for label_val in unique_labels:
        # 0 是背景，跳过
        if int(label_val) == 0:
            continue
            
        # 找出该 label 对应的所有索引
        idx = np.where(labels == label_val)[0]
        
        # 保存索引
        roi_indices.append(idx)
        # 保存名称，直接用数字字符串 '1', '2', ...
        roi_names.append(str(int(label_val)))
        
    if not roi_names:
        raise RuntimeError("在 npy 文件中未找到有效的 ROI 标签（大于 0 的值）。")
        
    return roi_names, roi_indices, n_grayords

def load_S(dtseries_path, n_grayords_expected):
    """读取 S（空间图/基），形状期望为 (300, n_grayords)"""
    img = nib.load(dtseries_path)
    S = img.get_fdata(dtype=np.float32)
    if S.ndim != 2:
        raise ValueError(f"{dtseries_path} 维度异常：{S.shape}")
    
    # 兼容可能的 (n_grayords, 300)
    if S.shape[1] != n_grayords_expected and S.shape[0] == n_grayords_expected:
        S = S.T
    return S 

def load_T(node_txt_path, n_components_expected=300):
    """读取 T（节点时序），期望形状为 (4800, 300)"""
    T = np.loadtxt(node_txt_path, dtype=np.float32)
    if T.ndim != 2:
        raise ValueError(f"{node_txt_path} 维度异常：{T.shape}")
    if T.shape[1] != n_components_expected and T.shape[0] == n_components_expected:
        T = T.T
    return T

def compute_A_S_roi_mean(S, roi_indices):
    """
    对 S 在每个 ROI 上做列均值，返回 A ∈ R^{300×R}
    """
    R = len(roi_indices)
    # S.shape[0] 通常是 300 (components)
    A = np.empty((S.shape[0], R), dtype=np.float32)
    n_grayords = S.shape[1]
    
    for j, idx in enumerate(roi_indices):
        # 确保索引不越界（虽然 npy 和 dtseries 应该都是 91282）
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
    """保存为 CSV"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_names)
        for i in range(data_2d.shape[0]):
            writer.writerow(data_2d[i, :].tolist())

def process_one_subject(sub_id, dtseries_dir, node_dir, roi_indices, roi_names, n_grayords, out_dir, overwrite=False):
    """单被试处理"""
    dt_path = os.path.join(dtseries_dir, f"{sub_id}.dtseries.nii")
    node_path = os.path.join(node_dir, f"{sub_id}.txt")
    out_path = os.path.join(out_dir, f"{sub_id}_AAL3_ts.csv") # 修改了文件名后缀
    
    if (not overwrite) and os.path.exists(out_path):
        return (sub_id, "skipped", "已存在，跳过")
        
    try:
        # 1. 加载 S (300 x 91282)
        S = load_S(dt_path, n_grayords)
        
        # 2. 加载 T (4800 x 300)
        T = load_T(node_path, n_components_expected=S.shape[0])
        
        # 3. 计算 A (ROI均值化的 S) -> (300 x 170)
        A = compute_A_S_roi_mean(S, roi_indices)
        
        # 4. 计算最终 ROI 时序 (4800 x 170)
        Y_roi = T @ A 
        
        # 5. 保存
        save_csv(out_path, roi_names, Y_roi.astype(np.float32))
        
        return (sub_id, "ok", f"保存到 {out_path}")
    except Exception as e:
        return (sub_id, "fail", repr(e))

def main():
    parser = argparse.ArgumentParser(description="HCP-PTN -> AAL3 (npy) ROI BOLD 时序 CSV 批处理")
    
    # 默认路径已更新为你提供的新路径
    parser.add_argument("--dtseries_dir", default='/ailab/group/medai-share/syDu/Brain_EC/HCP/3T_HCP1200_MSMAll_d300_ts2_Z', 
                        help="包含 *.dtseries.nii 的目录")
    parser.add_argument("--node_dir", default='/ailab/group/medai-share/syDu/Brain_EC/HCP/HCP_PTN1200_recon2/node_timeseries/3T_HCP1200_MSMAll_d300_ts2', 
                        help="包含 *.txt 的目录")
    
    # 核心修改：输入改为 npy 文件路径
    parser.add_argument("--atlas_npy", default='/ailab/group/medai-share/syDu/Brain_EC/AAL_atlas/AAL3v1.91k_fs_LR.npy', 
                        help="AAL3 91k .npy 文件路径")
    
    parser.add_argument("--out_dir", default='/ailab/group/medai-share/syDu/Brain_EC/HCP/HCP_AAL3_csv_out', 
                        help="输出 CSV 的目录")
    
    parser.add_argument("--workers", type=int, default=16, help="并行进程数")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖")
    
    args = parser.parse_args()

    # 1. 读取 AAL3 npy
    roi_names, roi_indices, n_grayords = load_aal3_npy(args.atlas_npy)
    print(f"ROI 数量：{len(roi_names)} (Expect: 170)，Grayordinates：{n_grayords}")
    print(f"ROI 名称示例: {roi_names[:5]} ... {roi_names[-5:]}")

    # 2. 查找被试
    ids = find_subjects(args.dtseries_dir, args.node_dir)
    print(f"共 {len(ids)} 个被试；输出目录：{args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    # 3. 并行处理
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

    # 4. 统计
    n_ok = sum(1 for _, s, _ in results if s == "ok")
    n_sk = sum(1 for _, s, _ in results if s == "skipped")
    n_fl = sum(1 for _, s, _ in results if s == "fail")
    print(f"完成：成功 {n_ok}，跳过 {n_sk}，失败 {n_fl}")
    
    if n_fl:
        print("失败明细：")
        for sub_id, s, msg in results:
            if s == "fail":
                print(f"  - {sub_id}: {msg}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()