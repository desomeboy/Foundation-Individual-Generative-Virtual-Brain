#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read (patient_id, modality, time, nifti_path) from a CSV file,
build pairs of MPRAGE and Resting_State_fMRI acquired on the same day
for the same patient, run the same preprocessing pipeline as the original script
for each rs-fMRI, and output atlas BOLD CSV files.

Output naming: patient_id_time.csv
The `time` value comes from the original Resting_State_fMRI entry and will be
sanitized for safe filenames (for example, spaces and colons will be replaced by '-').
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

# --------------------- Progress bar (tqdm optional) ---------------------
try:
    from tqdm import tqdm
except Exception:
    class tqdm:
        """Fallback class when tqdm is not available."""
        def __init__(self, total=None):
            self.total = total

        def update(self, n=1):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass


# --------------------- Utility functions ---------------------
def sh(
    cmd: List[str],
    cwd: Optional[Path] = None,
    log_file: Optional[Path] = None,
    env: Optional[dict] = None
) -> None:
    """Run a command, append stdout/stderr to the log file, and raise on failure."""
    cmd_str = " ".join(map(str, cmd))
    if log_file is None:
        raise RuntimeError("log_file cannot be None")

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


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def file_nonempty(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0


def check_tools_exist() -> Tuple[List[str], str]:
    """Check whether required FSL tools and FSLDIR are available."""
    required = ["bet", "fast", "flirt", "fnirt", "slicetimer", "mcflirt", "fslmaths", "applywarp"]
    missing = [t for t in required if shutil.which(t) is None]
    fsl_dir = os.environ.get("FSLDIR", "")
    return missing, fsl_dir


def sanitize_filename(s: str) -> str:
    """Convert an arbitrary string into a safer filename."""
    s = s.strip()
    s = s.replace(":", "-").replace(" ", "-").replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^A-Za-z0-9._-]", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s


def extract_date(s: str) -> str:
    """
    Extract the date part from a time string for pairing.

    Example:
        '2011-06-02_07_58_50.0' -> '2011-06-02'
    """
    if not s:
        return ""

    # Common case: starts with YYYY-MM-DD
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1)

    # Fallback: split by '_' and inspect the first segment
    first = s.split("_")[0]
    m2 = re.match(r"(\d{4}-\d{2}-\d{2})", first)
    if m2:
        return m2.group(1)

    return first


# --------------------- Data structure ---------------------
class PairJob:
    """
    Represent a processing task: one rs-fMRI paired with a same-day T1
    for the same patient.
    """
    def __init__(
        self,
        patient_id: str,
        fmri_time: str,
        t1_time: str,
        fmri_path: Path,
        t1_path: Path
    ):
        self.patient_id = patient_id
        self.fmri_time = fmri_time
        self.t1_time = t1_time
        self.fmri_path = fmri_path
        self.t1_path = t1_path

    @property
    def pair_id(self) -> str:
        """Unique ID used for cache directory names, logs, and tracking."""
        return f"{self.patient_id}_{sanitize_filename(self.fmri_time)}"


# --------------------- Processing pipeline for a single pair ---------------------
def process_pair(
    job: PairJob,
    outdir: Path,
    atlas_script: Path,
    python_exe: str,
    fsl_dir: str,
    keep_work: bool,
    cache_dir: Optional[Path],
    cache_pattern: str
) -> Tuple[str, str]:
    """
    Process a single pair.

    Returns:
        (pair_id, status)
        where status is one of:
        - 'ok'
        - 'skip'
        - 'error:<msg>'
    """
    pid = job.patient_id
    pair_id = job.pair_id

    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{pair_id}.log"

    # Final output: patient_id_time.csv
    final_csv = outdir / f"{pid}_{sanitize_filename(job.fmri_time)}.csv"
    if file_nonempty(final_csv):
        return pair_id, "skip"

    # Working directory (prefer SSD cache if provided)
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

        # Create symlinks in the working directory to avoid copying large files
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

        # ------------------- T1 processing -------------------
        t1_bet_base = workdir / "T1_bet"
        if not file_nonempty(Path(str(t1_bet_base) + ".nii.gz")):
            sh(
                ["bet", "T1_input.nii.gz", "T1_bet", "-R", "-f", "0.4", "-g", "0"],
                cwd=workdir,
                log_file=log_file,
                env=env
            )

        t1_fast_restore = workdir / "T1_fast_restore.nii.gz"
        if not file_nonempty(t1_fast_restore):
            sh(
                ["fast", "-B", "-o", "T1_fast", "T1_input.nii.gz"],
                cwd=workdir,
                log_file=log_file,
                env=env
            )

        t1_to_mni_aff = workdir / "T1_to_MNI_aff.mat"
        if not t1_to_mni_aff.exists():
            sh(
                [
                    "flirt",
                    "-in", "T1_fast_restore.nii.gz",
                    "-ref", f"{fsl_dir}/data/standard/MNI152_T1_1mm_brain",
                    "-omat", "T1_to_MNI_aff.mat",
                    "-dof", "12"
                ],
                cwd=workdir,
                log_file=log_file,
                env=env
            )

        t1_to_mni_warp = workdir / "T1_to_MNI_warp.nii.gz"
        t1_to_mni_iout = workdir / "T1_to_MNI.nii.gz"
        if (not file_nonempty(t1_to_mni_warp)) or (not file_nonempty(t1_to_mni_iout)):
            sh(
                [
                    "fnirt",
                    "--in=T1_input.nii.gz",
                    "--aff=T1_to_MNI_aff.mat",
                    f"--ref={fsl_dir}/data/standard/MNI152_T1_1mm",
                    "--iout=T1_to_MNI",
                    "--cout=T1_to_MNI_warp"
                ],
                cwd=workdir,
                log_file=log_file,
                env=env
            )

        # ------------------- fMRI processing -------------------
        func_stc = workdir / "func_stc.nii.gz"
        if not file_nonempty(func_stc):
            sh(
                ["slicetimer", "-i", "func_input.nii.gz", "-o", "func_stc"],
                cwd=workdir,
                log_file=log_file,
                env=env
            )

        func_mc = workdir / "func_mc.nii.gz"
        if not file_nonempty(func_mc):
            sh(
                ["mcflirt", "-in", "func_stc", "-out", "func_mc", "-plots", "-meanvol"],
                cwd=workdir,
                log_file=log_file,
                env=env
            )

        func_mc_mean = workdir / "func_mc_mean.nii.gz"
        if not file_nonempty(func_mc_mean):
            sh(
                ["fslmaths", "func_mc", "-Tmean", "func_mc_mean"],
                cwd=workdir,
                log_file=log_file,
                env=env
            )

        epi_to_t1_mat = workdir / "EPI_to_T1.mat"
        if not epi_to_t1_mat.exists():
            sh(
                [
                    "flirt",
                    "-in", "func_mc_mean",
                    "-ref", "T1_fast_restore.nii.gz",
                    "-omat", "EPI_to_T1.mat",
                    "-dof", "6",
                    "-cost", "normmi"
                ],
                cwd=workdir,
                log_file=log_file,
                env=env
            )

        func_mni = workdir / "func_MNI.nii.gz"
        if not file_nonempty(func_mni):
            sh(
                [
                    "applywarp",
                    "-i", "func_mc.nii.gz",
                    "-r", f"{fsl_dir}/data/standard/MNI152_T1_1mm",
                    "-o", "func_MNI",
                    "--premat=EPI_to_T1.mat",
                    "-w", "T1_to_MNI_warp"
                ],
                cwd=workdir,
                log_file=log_file,
                env=env
            )

        # ------------------- Extract atlas BOLD to CSV -------------------
        if not file_nonempty(final_csv):
            ensure_parent(final_csv)
            sh(
                [
                    python_exe,
                    str(atlas_script),
                    "--func", "func_MNI.nii.gz",
                    "--out", str(final_csv)
                ],
                cwd=workdir,
                log_file=log_file,
                env=env
            )

        # ------------------- Clean up intermediate files -------------------
        if not keep_work:
            try:
                shutil.rmtree(workdir)
            except Exception as e:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n[WARN] Failed to remove working directory: {e}\n")

        with open(log_file, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] ==== Done pair {pair_id} ====\n")

        return pair_id, "ok"

    except Exception as e:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[ERROR] {e}\n")
        return pair_id, f"error:{e}"


# --------------------- CSV loading and pairing ---------------------
def load_pairs_from_csv(csv_path: Path) -> List[PairJob]:
    """
    Read the CSV file and build pairings based on:
    - same patient_id
    - same date extracted from the time field (YYYY-MM-DD)
    - modality containing 'MPRAGE' as T1
    - modality containing 'Resting_State_fMRI' as fMRI

    If multiple fMRI scans exist on the same day, create one job for each fMRI.
    If multiple T1 scans exist on the same day, choose the T1 closest in time.
    """
    rows: List[Dict[str, Any]] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Normalize keys to improve robustness against capitalization and spaces
            row = {k.strip().lower(): v.strip() for k, v in r.items()}

            pid = row.get("patient_id") or row.get("id") or row.get("subject") or ""
            mod = row.get("modality") or ""
            t = row.get("time") or ""
            pth = row.get("nifti_path") or row.get("path") or ""

            if not (pid and mod and t and pth):
                continue

            rows.append(
                {
                    "patient_id": pid,
                    "modality": mod,
                    "time": t,
                    "nifti_path": pth
                }
            )

    from collections import defaultdict
    by_patient: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)

    pair_jobs: List[PairJob] = []

    for pid, items in by_patient.items():
        t1_by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        fmri_list: List[Dict[str, Any]] = []

        for it in items:
            date = extract_date(it["time"])

            if it["modality"].lower() in ["mprage", "t1", "mp-rage", "mp_rage"]:
                t1_by_date[date].append(it)
            elif it["modality"] in ["Resting_State_fMRI", "rs-fMRI", "rsfmri", "Resting*"]:
                fmri_list.append(it)
            else:
                # Additional compatibility for alternative naming styles
                if "rest" in it["modality"].lower() and "fmri" in it["modality"].lower():
                    fmri_list.append(it)
                elif "mprage" in it["modality"].lower():
                    t1_by_date[date].append(it)

        for f in fmri_list:
            f_date = extract_date(f["time"])
            candidates = t1_by_date.get(f_date, [])

            if not candidates:
                continue

            def time_to_seconds(x: str) -> int:
                """
                Try to parse strings such as 'YYYY-MM-DD_HH_MM_SS...'.
                Return a neutral midday value if parsing fails.
                """
                m = re.match(r"(\d{4}-\d{2}-\d{2})[_-](\d{2})[_:](\d{2})[_:](\d{2})", x)
                if not m:
                    return 12 * 3600
                h, mi, se = int(m.group(2)), int(m.group(3)), int(m.group(4))
                return h * 3600 + mi * 60 + se

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


# --------------------- Main program ---------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch fMRI-to-CSV processing from CSV pairings (FSL + multiprocessing + resume support + progress display)"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("final_csv.csv"),
        help="Input CSV file containing columns: patient_id, modality, time, nifti_path"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("PPMI_AAL3_CSV"),
        help="Output directory for <patient_id>_<time>.csv files and logs"
    )
    parser.add_argument(
        "--atlas-script",
        type=Path,
        default=Path("Data_process/last_process_AAL.py"),
        help="Script used to extract atlas BOLD features"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default="python3",
        help="Python executable used to run the atlas script"
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep intermediate files in _work/<pair_id> instead of deleting them after successful completion"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="SSD cache root directory for intermediate files"
    )
    parser.add_argument(
        "--cache-pattern",
        type=str,
        default="PID({sid})",
        help="Naming template for cache subdirectories, for example: 'PID({sid})'"
    )

    args = parser.parse_args()

    missing, fsl_dir = check_tools_exist()
    if missing:
        print(
            f"[FATAL] The following commands were not found. Please make sure FSL is installed and available in PATH: {missing}",
            file=sys.stderr
        )
        sys.exit(2)

    if not fsl_dir:
        print(
            "[WARN] $FSLDIR was not detected. If later steps fail, please export FSLDIR and add $FSLDIR/bin to PATH."
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading and pairing from CSV: {args.csv}")
    pair_jobs = load_pairs_from_csv(args.csv)

    if not pair_jobs:
        print(
            "[FATAL] No valid (MPRAGE, Resting_State_fMRI) pairs were found. Please check the CSV file and field names.",
            file=sys.stderr
        )
        sys.exit(1)

    print(f"Found {len(pair_jobs)} processable pairs; using {args.workers} worker processes.")
    if args.cache_dir is not None:
        print(f"Using SSD cache directory: {args.cache_dir}, naming pattern: {args.cache_pattern}")

    results: Dict[str, List] = {"ok": [], "skip": [], "error": []}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for job in pair_jobs:
            futs.append(
                ex.submit(
                    process_pair,
                    job,
                    args.outdir,
                    args.atlas_script,
                    args.python_exe,
                    fsl_dir,
                    args.keep_work,
                    args.cache_dir,
                    args.cache_pattern
                )
            )

        with tqdm(total=len(futs)) as pbar:
            for fu in as_completed(futs):
                pair_id, status = fu.result()

                if status == "ok":
                    print(f"[OK] {pair_id}")
                    results["ok"].append(pair_id)
                elif status == "skip":
                    print(f"[SKIP] {pair_id} (CSV already exists)")
                    results["skip"].append(pair_id)
                else:
                    print(f"[ERROR] {pair_id} -> {status}")
                    results["error"].append((pair_id, status))

                pbar.update(1)

    summary_file = args.outdir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(
            "Completed: {}\nSkipped: {}\nFailed: {}\n\n".format(
                len(results["ok"]),
                len(results["skip"]),
                len(results["error"])
            )
        )

        if results["ok"]:
            f.write("OK:\n" + "\n".join(results["ok"]) + "\n\n")

        if results["skip"]:
            f.write("SKIP:\n" + "\n".join(results["skip"]) + "\n\n")

        if results["error"]:
            f.write(
                "ERROR:\n" + "\n".join(
                    [f"{sid} -> {msg}" for sid, msg in results["error"]]
                ) + "\n"
            )

    print(
        "\nProcessing finished: OK={}, SKIP={}, ERROR={}".format(
            len(results["ok"]),
            len(results["skip"]),
            len(results["error"])
        )
    )
    print(f"Summary written to: {summary_file}")


if __name__ == "__main__":
    main()