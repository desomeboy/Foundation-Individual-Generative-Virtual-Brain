#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从CSV读取 (patient_id, modality, time, nifti_path)，
为同一病人中“同一天”的 MPRAGE 与 Resting_State_fMRI 建立配对，
对每个 rs-fMRI 进行与原脚本一致的预处理，并输出 atlas BOLD CSV。

输出命名：patient_id_time.csv   （time 来自该条 Resting_State_fMRI 的原始 time 字段，
会做安全字符清洗，比如空格、冒号替换成 -）

进度：使用 tqdm 显示总体任务进度（完成/跳过/错误会即时打印 & 记录日志）。
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

# --------------------- 进度条（tqdm 可选） ---------------------
try:
    from tqdm import tqdm
except Exception:
    class tqdm:  # 兜底，tqdm 不在环境中时不报错
        def __init__(self, total=None):
            self.total = total
        def update(self, n=1): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *exc): pass

# --------------------- 工具函数 ---------------------
def sh(cmd: List[str], cwd: Optional[Path] = None, log_file: Optional[Path] = None, env: Optional[dict] = None) -> None:
    """运行命令并把 stdout/stderr 追加写入日志；出错抛异常。"""
    cmd_str = " ".join(map(str, cmd))
    if log_file is None:
        raise RuntimeError("log_file 不能为空")
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
            raise RuntimeError("Command failed (code {}): {}".format(proc.returncode, cmd_str))

def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def file_nonempty(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0

def check_tools_exist() -> Tuple[List[str], str]:
    """简单检查必须的 FSL 工具和 FSLDIR。"""
    required = ["bet", "fast", "flirt", "fnirt", "slicetimer", "mcflirt", "fslmaths", "applywarp"]
    missing = [t for t in required if shutil.which(t) is None]
    fsl_dir = os.environ.get("FSLDIR", "")
    return missing, fsl_dir

def sanitize_filename(s: str) -> str:
    """将任意字符串转换为更安全的文件名。保留字母数字，其余变为-，并压缩重复-。"""
    s = s.strip()
    s = s.replace(":", "-").replace(" ", "-").replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^A-Za-z0-9._-]", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s

def extract_date(s: str) -> str:
    """
    从 time 字符串中提取日期部分用于配对。
    例：'2011-06-02_07_58_50.0' -> '2011-06-02'
    """
    if not s:
        return ""
    # 常见情况：YYYY-MM-DD 开头
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1)
    # 兜底：按 '_' 分割取第一段再看是否像日期
    first = s.split("_")[0]
    m2 = re.match(r"(\d{4}-\d{2}-\d{2})", first)
    if m2:
        return m2.group(1)
    return first  # 不严格，但尽量用首段

# --------------------- 数据结构 ---------------------
class PairJob:
    """
    表示一个要处理的任务：同一病人的一条 rs-fMRI 与（同日）T1 的配对。
    """
    def __init__(self, patient_id: str, fmri_time: str, t1_time: str,
                 fmri_path: Path, t1_path: Path):
        self.patient_id = patient_id
        self.fmri_time = fmri_time  # 输出文件名用这个
        self.t1_time = t1_time
        self.fmri_path = fmri_path
        self.t1_path = t1_path

    @property
    def pair_id(self) -> str:
        # 用于缓存目录命名、日志等的唯一ID
        return f"{self.patient_id}_{sanitize_filename(self.fmri_time)}"

# --------------------- 单个配对的处理流水线 ---------------------
def process_pair(job: PairJob,
                 outdir: Path,
                 atlas_script: Path,
                 python_exe: str,
                 fsl_dir: str,
                 keep_work: bool,
                 cache_dir: Optional[Path],
                 cache_pattern: str) -> Tuple[str, str]:
    """
    处理单个 pair；返回 (pair_id, status)   status: 'ok'/'skip'/'error:<msg>'
    """
    pid = job.patient_id
    pair_id = job.pair_id

    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{pair_id}.log"

    # 输出：patient_id_time.csv （time = fmri 的 time）
    final_csv = outdir / f"{pid}_{sanitize_filename(job.fmri_time)}.csv"
    if file_nonempty(final_csv):
        return pair_id, "skip"

    # 工作目录（优先 SSD 缓存）
    if cache_dir is not None:
        subname = cache_pattern.format(sid=pair_id)
        workdir = cache_dir / subname
    else:
        workdir = outdir / "_work" / pair_id
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] ==== Start pair {pair_id} ====\n")
            f.write(f"Patient: {pid}\n")
            f.write(f"FMRI time: {job.fmri_time}, T1 time: {job.t1_time}\n")
            f.write(f"Workdir: {workdir}\n")
            if cache_dir is not None:
                f.write(f"[INFO] Using SSD cache at {workdir}\n")
            f.write(f"T1: {job.t1_path}\n")
            f.write(f"FMRI: {job.fmri_path}\n")

        # 软链到工作目录，避免复制大文件
        t1_link = workdir / "T1_input.nii.gz"
        fmri_link = workdir / "func_input.nii.gz"
        if not t1_link.exists():
            sh(["ln", "-sf", str(job.t1_path.resolve()), str(t1_link)], log_file=log_file)
        if not fmri_link.exists():
            sh(["ln", "-sf", str(job.fmri_path.resolve()), str(fmri_link)], log_file=log_file)

        env = os.environ.copy()
        if fsl_dir:
            env["FSLDIR"] = fsl_dir
            fsl_bin = str(Path(fsl_dir) / "bin")
            if fsl_bin not in env.get("PATH", ""):
                env["PATH"] = fsl_bin + os.pathsep + env.get("PATH", "")

        # ------------------- T1 处理 -------------------
        t1_bet_base = workdir / "T1_bet"
        if not file_nonempty(Path(str(t1_bet_base) + ".nii.gz")):
            sh(["bet", "T1_input.nii.gz", "T1_bet", "-R", "-f", "0.4", "-g", "0"],
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
        if (not file_nonempty(t1_to_mni_warp)) or (not file_nonempty(t1_to_mni_iout)):
            sh([
                "fnirt",
                "--in=T1_input.nii.gz",
                "--aff=T1_to_MNI_aff.mat",
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
                    f.write("\n[WARN] 清理工作目录失败：{}\n".format(e))

        with open(log_file, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] ==== Done pair {pair_id} ====\n")

        return pair_id, "ok"

    except Exception as e:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n[ERROR] {}\n".format(e))
        return pair_id, "error:{}".format(e)

# --------------------- CSV 读取与配对 ---------------------
def load_pairs_from_csv(csv_path: Path) -> List[PairJob]:
    """
    读取 CSV 并建立配对：
    - 同一 patient_id
    - time 中的“日期部分”（YYYY-MM-DD）相同
    - modality 包含：'MPRAGE' 作为 T1，'Resting_State_fMRI' 作为 fMRI
    若同一天有多条 fMRI，则对每条 fMRI 都建立任务（T1 取当天任一 MPRAGE，优先选择时间最靠近的）。
    """
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 标准化键名（容错大小写/空格）
            row = {k.strip().lower(): v.strip() for k, v in r.items()}
            # 期望字段：patient_id, modality, time, nifti_path
            pid = row.get("patient_id") or row.get("id") or row.get("subject") or ""
            mod = row.get("modality") or ""
            t = row.get("time") or ""
            pth = row.get("nifti_path") or row.get("path") or ""
            if not (pid and mod and t and pth):
                continue
            rows.append({"patient_id": pid, "modality": mod, "time": t, "nifti_path": pth})

    # 按 patient 分组
    from collections import defaultdict
    by_patient: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)

    pair_jobs: List[PairJob] = []

    for pid, items in by_patient.items():
        # 拆成 T1 与 fMRI 列表，同时按日期做索引
        t1_by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        fmri_list: List[Dict[str, Any]] = []
        for it in items:
            date = extract_date(it["time"])
            if it["modality"].lower() in ["mprage", "t1", "mp-rage", "mp_rage"]:
                t1_by_date[date].append(it)
            elif it["modality"] in ["Resting_State_fMRI", "rs-fMRI", "rsfmri", "Resting*"]:
                fmri_list.append(it)
            else:
                # 兼容大小写/其它写法
                if "rest" in it["modality"].lower() and "fmri" in it["modality"].lower():
                    fmri_list.append(it)
                elif "mprage" in it["modality"].lower():
                    t1_by_date[date].append(it)

        # 对每条 fMRI，找同日 T1
        for f in fmri_list:
            f_date = extract_date(f["time"])
            candidates = t1_by_date.get(f_date, [])
            if not candidates:
                # 无同日 T1，跳过
                continue
            # 选择与 fMRI 时间最近的 T1（如果 T1 同日多条）
            def time_to_seconds(x: str) -> int:
                # 尝试解析 'YYYY-MM-DD_HH_MM_SS...'，失败则给一个中性值
                m = re.match(r"(\d{4}-\d{2}-\d{2})[_-](\d{2})[_:](\d{2})[_:](\d{2})", x)
                if not m:
                    return 12*3600  # 中午
                h, mi, se = int(m.group(2)), int(m.group(3)), int(m.group(4))
                return h*3600 + mi*60 + se

            f_sec = time_to_seconds(f["time"])
            cand = min(candidates, key=lambda it: abs(time_to_seconds(it["time"]) - f_sec))
            pair_jobs.append(
                PairJob(
                    patient_id=pid,
                    fmri_time=f["time"],
                    t1_time=cand["time"],
                    fmri_path=Path(f["nifti_path"]),
                    t1_path=Path(cand["nifti_path"]),
                )
            )

    return pair_jobs

# --------------------- 主程序 ---------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="批量 fMRI→CSV 处理（从CSV配对；FSL + 多进程 + 断点续跑 + 可视化进度）")
    parser.add_argument("--csv", type=Path, default='/ailab/group/medai-share/syDu/Brain_EC/PPMI/Process/final_csv.csv', help="清单CSV，包含列：patient_id, modality, time, nifti_path")
    parser.add_argument("--outdir", type=Path, default='/ailab/group/medai-share/syDu/Brain_EC/PPMI/PPMI_AAL3_CSV', help="输出目录（保存 <patient_id>_<time>.csv 与日志）")
    parser.add_argument("--atlas-script", type=Path, default='/ailab/group/medai-share/syDu/Brain_EC/AAL_atlas/last_process_AAL.py', help="提取 atlas BOLD 的脚本（如 last_process.py）")
    parser.add_argument("--workers", type=int, default=16, help="并发进程数（默认半数 CPU）")
    parser.add_argument("--python-exe", type=str, default="python3", help="调用 atlas 脚本的 Python 解释器（默认 python3）")
    parser.add_argument("--keep-work", action="store_true", help="保留 _work/<pair_id> 中间文件（默认成功后清理）")
    parser.add_argument("--cache-dir", type=Path, default='/ailab/user/dusiyuan/code/Brain/EC/cache', help="SSD 缓存根目录（中间文件写到这里）")
    parser.add_argument("--cache-pattern", type=str, default="PID({sid})", help="SSD 子目录命名模板，如 'PID({sid})'")
    args = parser.parse_args()

    missing, fsl_dir = check_tools_exist()
    if missing:
        print("[FATAL] 以下命令未找到，请确认 FSL 安装并在 PATH 中：{}".format(missing), file=sys.stderr)
        sys.exit(2)
    if not fsl_dir:
        print("[WARN] 未检测到 $FSLDIR 环境变量；若后续报错请先 export FSLDIR 并将 $FSLDIR/bin 加入 PATH。")

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("读取并配对中：{}".format(args.csv))
    pair_jobs = load_pairs_from_csv(args.csv)
    if not pair_jobs:
        print("[FATAL] 未找到任何可配对的 (MPRAGE, Resting_State_fMRI)。请检查CSV与字段。", file=sys.stderr)
        sys.exit(1)

    print("发现 {} 个可处理的配对；并发 {} 个进程。".format(len(pair_jobs), args.workers))
    if args.cache_dir is not None:
        print("使用 SSD 缓存目录：{}，命名模板：{}".format(args.cache_dir, args.cache_pattern))

    results: Dict[str, List] = {"ok": [], "skip": [], "error": []}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for job in pair_jobs:
            futs.append(ex.submit(
                process_pair, job, args.outdir, args.atlas_script,
                args.python_exe, fsl_dir, args.keep_work,
                args.cache_dir, args.cache_pattern
            ))

        with tqdm(total=len(futs)) as pbar:
            for fu in as_completed(futs):
                pair_id, status = fu.result()
                if status == "ok":
                    print("[OK] {}".format(pair_id))
                    results["ok"].append(pair_id)
                elif status == "skip":
                    print("[SKIP] {}（已存在 CSV）".format(pair_id))
                    results["skip"].append(pair_id)
                else:
                    print("[ERROR] {} -> {}".format(pair_id, status))
                    results["error"].append((pair_id, status))
                pbar.update(1)

    summary_file = args.outdir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("完成：{}\n跳过：{}\n失败：{}\n\n".format(len(results["ok"]), len(results["skip"]), len(results["error"])))
        if results["ok"]:
            f.write("OK:\n" + "\n".join(results["ok"]) + "\n\n")
        if results["skip"]:
            f.write("SKIP:\n" + "\n".join(results["skip"]) + "\n\n")
        if results["error"]:
            f.write("ERROR:\n" + "\n".join(["{} -> {}".format(sid, msg) for sid, msg in results["error"]]) + "\n")

    print("\n处理完成：OK={}, SKIP={}, ERROR={}".format(len(results["ok"]), len(results["skip"]), len(results["error"])))
    print("汇总见：{}".format(summary_file))

if __name__ == "__main__":
    main()
