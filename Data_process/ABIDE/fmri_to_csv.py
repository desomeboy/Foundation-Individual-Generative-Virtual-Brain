#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量化处理 ABIDE 风格目录下的 T1/rs-fMRI，提取各脑区 BOLD 时序为 CSV。
特性：多进程、断点续跑、每例完成即保存、清理中间文件、详细日志、可将中间文件放到 SSD。

用法示例：
python3 batch_fmri_to_csv.py \
  --root /ailab/group/medai-share/syDu/Brain_EC/AUIDE/ABIDE \
  --outdir /ailab/group/medai-share/syDu/Brain_EC/AUIDE/ABIDE_csv_out \
  --atlas-script /ailab/group/medai-share/syDu/Brain_EC/MMP_atlas/The-HCP-MMP1.0-atlas-in-FSL/last_process.py \
  --cache-dir /ailab/user/dusiyuan/code/Brain/EC/cache \
  --cache-pattern "PID({sid})" \
  --workers 6
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# --------------------- 工具函数 ---------------------

def sh(cmd, cwd=None, log_file=None, env=None):
    """运行命令并把 stdout/stderr 追加写入日志；出错抛异常。"""
    cmd_str = " ".join(map(str, cmd))
    with open(log_file, "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n[{ts}] CMD: {cmd_str}\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            check=False,
        )
        f.write(proc.stdout)
        f.flush()
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed (code {proc.returncode}): {cmd_str}")

def find_one(patterns):
    """从多个 glob 模式中找一个存在的文件；返回第一个排序后的文件路径。"""
    for pat in patterns:
        hits = sorted(Path().glob(pat))
        if hits:
            return hits[0]
    return None

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def file_nonempty(p: Path):
    return p.exists() and p.is_file() and p.stat().st_size > 0

def check_tools_exist():
    """简单检查必须的 FSL 工具和 FSLDIR。"""
    required = ["bet", "fast", "flirt", "fnirt", "slicetimer", "mcflirt", "fslmaths", "applywarp"]
    missing = [t for t in required if shutil.which(t) is None]
    fsl_dir = os.environ.get("FSLDIR", "")
    return missing, fsl_dir

# --------------------- 单个受试者流程 ---------------------

def process_subject(subject_dir: Path, outdir: Path, atlas_script: Path,
                    python_exe: str, fsl_dir: str, keep_work: bool,
                    cache_dir: Path | None, cache_pattern: str) -> tuple[str, str]:
    """
    处理单个受试者；返回 (subject_id, status)
    status 取值：'ok' / 'skip' / 'error:<msg>'
    """
    subject_id = subject_dir.name
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{subject_id}.log"

    final_csv = outdir / f"{subject_id}.csv"
    if file_nonempty(final_csv):
        return subject_id, "skip"

    # 选择工作目录：优先 SSD 缓存，否则使用 outdir/_work
    if cache_dir is not None:
        # 支持自定义子目录命名，默认 "PID({sid})"
        subname = cache_pattern.format(sid=subject_id)
        workdir = cache_dir / subname
    else:
        workdir = outdir / "_work" / subject_id
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] ==== Start subject {subject_id} ====\n")
            f.write(f"Workdir: {workdir}\n")
            if cache_dir is not None:
                f.write(f"[INFO] Using SSD cache at {workdir}\n")

        # 1) 寻找 T1 与 rs-fMRI
        t1_path = find_one([
            str(subject_dir / "MP-RAGE" / "*" / "*" / "*.nii*")
        ])
        fmri_path = find_one([
            str(subject_dir / "Resting_State_fMRI" / "*" / "*" / "*.nii*")
        ])
        if not t1_path or not fmri_path:
            raise FileNotFoundError(f"T1 或 fMRI 未找到：T1={t1_path}, fMRI={fmri_path}")

        # 软链到工作目录，避免复制大文件
        t1_link = workdir / "T1_input.nii.gz"
        fmri_link = workdir / "func_input.nii.gz"
        if not t1_link.exists():
            sh(["ln", "-sf", str(t1_path.resolve()), str(t1_link)], log_file=log_file)
        if not fmri_link.exists():
            sh(["ln", "-sf", str(fmri_path.resolve()), str(fmri_link)], log_file=log_file)

        env = os.environ.copy()
        if fsl_dir:
            env["FSLDIR"] = fsl_dir
            fsl_bin = str(Path(fsl_dir) / "bin")
            if fsl_bin not in env.get("PATH", ""):
                env["PATH"] = fsl_bin + os.pathsep + env.get("PATH", "")

        # ------------------- T1 处理 -------------------
        t1_bet_base = workdir / "T1_bet"
        if not file_nonempty(Path(str(t1_bet_base) + ".nii.gz")):
            sh(["bet", "T1_input.nii.gz", "T1_bet", "-R", "-f", "0.3", "-g", "0"],
               cwd=workdir, log_file=log_file, env=env)

        t1_fast_restore = workdir / "T1_fast_restore.nii.gz"
        if not file_nonempty(t1_fast_restore):
            sh(["fast", "-B", "-o", "T1_fast", "T1_input.nii.gz"],
               cwd=workdir, log_file=log_file, env=env)

        t1_to_mni_aff = workdir / "T1_to_MNI_aff.mat"
        if not t1_to_mni_aff.exists():
            sh([
                "flirt",
                "-in", "T1_fast_restore.nii.gz",
                "-ref", f"{fsl_dir}/data/standard/MNI152_T1_1mm_brain",
                "-omat", "T1_to_MNI_aff.mat",
                "-dof", "12"
            ], cwd=workdir, log_file=log_file, env=env)

        t1_to_mni_warp = workdir / "T1_to_MNI_warp.nii.gz"
        t1_to_mni_iout = workdir / "T1_to_MNI.nii.gz"
        if not file_nonempty(t1_to_mni_warp) or not file_nonempty(t1_to_mni_iout):
            sh([
                "fnirt",
                f"--in=T1_input.nii.gz",
                f"--aff=T1_to_MNI_aff.mat",
                f"--ref={fsl_dir}/data/standard/MNI152_T1_1mm",
                "--iout=T1_to_MNI",
                "--cout=T1_to_MNI_warp"
            ], cwd=workdir, log_file=log_file, env=env)

        # ------------------- fMRI 处理 -------------------
        func_stc = workdir / "func_stc.nii.gz"
        if not file_nonempty(func_stc):
            sh(["slicetimer", "-i", "func_input.nii.gz", "-o", "func_stc"],
               cwd=workdir, log_file=log_file, env=env)

        func_mc = workdir / "func_mc.nii.gz"
        if not file_nonempty(func_mc):
            sh(["mcflirt", "-in", "func_stc", "-out", "func_mc", "-plots", "-meanvol"],
               cwd=workdir, log_file=log_file, env=env)

        func_mc_mean = workdir / "func_mc_mean.nii.gz"
        if not file_nonempty(func_mc_mean):
            sh(["fslmaths", "func_mc", "-Tmean", "func_mc_mean"],
               cwd=workdir, log_file=log_file, env=env)

        epi_to_t1_mat = workdir / "EPI_to_T1.mat"
        if not epi_to_t1_mat.exists():
            sh([
                "flirt",
                "-in", "func_mc_mean",
                "-ref", "T1_fast_restore.nii.gz",
                "-omat", "EPI_to_T1.mat",
                "-dof", "6",
                "-cost", "normmi"
            ], cwd=workdir, log_file=log_file, env=env)

        func_mni = workdir / "func_MNI.nii.gz"
        if not file_nonempty(func_mni):
            sh([
                "applywarp",
                "-i", "func_mc.nii.gz",
                "-r", f"{fsl_dir}/data/standard/MNI152_T1_1mm",
                "-o", "func_MNI",
                "--premat=EPI_to_T1.mat",
                "-w", "T1_to_MNI_warp"
            ], cwd=workdir, log_file=log_file, env=env)

        # ------------------- 提取 atlas BOLD → CSV -------------------
        if not file_nonempty(final_csv):
            ensure_parent(final_csv)
            sh([
                python_exe, str(atlas_script),
                "--func", "func_MNI.nii.gz",
                "--out", str(final_csv)
            ], cwd=workdir, log_file=log_file, env=env)

        # ------------------- 清理中间文件 -------------------
        if not keep_work:
            try:
                shutil.rmtree(workdir)
            except Exception as e:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n[WARN] 清理工作目录失败：{e}\n")

        with open(log_file, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] ==== Done subject {subject_id} ====\n")

        return subject_id, "ok"

    except Exception as e:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[ERROR] {e}\n")
        return subject_id, f"error:{e}"

# --------------------- 主程序 ---------------------

def main():
    parser = argparse.ArgumentParser(description="批量 fMRI→CSV 处理（FSL + 多进程 + 断点续跑）")
    parser.add_argument("--root", default='/ailab/group/medai-share/syDu/Brain_EC/ABIDE/ABIDE' , type=Path, help="根目录（包含多个病人子目录）")
    parser.add_argument("--outdir", default='/ailab/group/medai-share/syDu/Brain_EC/ABIDE/ABIDE_AAL3_csv_out', type=Path, help="输出目录（保存 <subject>.csv 与日志）")
    parser.add_argument("--atlas-script", default='/ailab/group/medai-share/syDu/Brain_EC/AAL_atlas/last_process_AAL.py', type=Path,
                        help="提取 atlas BOLD 的 Python 脚本（如 last_process.py）")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="并发进程数（默认半数 CPU）")
    parser.add_argument("--python-exe", type=str, default="python3",
                        help="调用 atlas 脚本的 Python 解释器（默认 python3）")
    parser.add_argument("--keep-work", action="store_true",
                        help="保留 _work/<subject> 中间文件（默认处理成功后会清理）")
    # 新增：SSD 缓存参数
    parser.add_argument("--cache-dir", type='/ailab/user/dusiyuan/code/Brain/EC/cache', default=None,
                        help="SSD 缓存根目录（中间文件写到这里）")
    parser.add_argument("--cache-pattern", type=str, default="PID({sid})",
                        help="SSD 子目录命名模板，{sid} 会被替换为受试者ID。例：'PID({sid})' → 'PID(50002)'")

    args = parser.parse_args()

    missing, fsl_dir = check_tools_exist()
    if missing:
        print(f"[FATAL] 以下命令未找到，请确认 FSL 安装并在 PATH 中：{missing}", file=sys.stderr)
        sys.exit(2)
    if not fsl_dir:
        print("[WARN] 未检测到 $FSLDIR 环境变量；若后续报错请先 export FSLDIR 并将 $FSLDIR/bin 加入 PATH。")

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    subjects = [d for d in sorted(args.root.iterdir()) if d.is_dir()]
    if not subjects:
        print("[FATAL] 根目录下未发现任何子目录。", file=sys.stderr)
        sys.exit(1)

    print(f"发现 {len(subjects)} 个病人目录；并发 {args.workers} 个进程。")
    if args.cache_dir is not None:
        print(f"使用 SSD 缓存目录：{args.cache_dir}，命名模板：{args.cache_pattern}")

    results = {"ok": [], "skip": [], "error": []}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for sd in subjects:
            futs.append(ex.submit(
                process_subject, sd, args.outdir, args.atlas_script,
                args.python_exe, fsl_dir, args.keep_work,
                args.cache_dir, args.cache_pattern
            ))

        for fu in as_completed(futs):
            sid, status = fu.result()
            if status == "ok":
                print(f"[OK] {sid}")
                results["ok"].append(sid)
            elif status == "skip":
                print(f"[SKIP] {sid}（已存在 CSV）")
                results["skip"].append(sid)
            else:
                print(f"[ERROR] {sid} -> {status}")
                results["error"].append((sid, status))

    summary_file = args.outdir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"完成：{len(results['ok'])}\n跳过：{len(results['skip'])}\n失败：{len(results['error'])}\n\n")
        if results["ok"]:
            f.write("OK:\n" + "\n".join(results["ok"]) + "\n\n")
        if results["skip"]:
            f.write("SKIP:\n" + "\n".join(results["skip"]) + "\n\n")
        if results["error"]:
            f.write("ERROR:\n" + "\n".join([f"{sid} -> {msg}" for sid, msg in results["error"]]) + "\n")

    print(f"\n处理完成：OK={len(results['ok'])}, SKIP={len(results['skip'])}, ERROR={len(results['error'])}")
    print(f"汇总见：{summary_file}")

if __name__ == "__main__":
    main()
