"""
Batch-process T1 and rs-fMRI data under an ABIDE-style directory structure
and extract regional BOLD time series to CSV files.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


def sh(cmd, cwd=None, log_file=None, env=None):
    """Run a command and append stdout/stderr to the log file."""
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
    """Find the first existing file from multiple glob patterns."""
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
    """Check required FSL tools and FSLDIR."""
    required = ["bet", "fast", "flirt", "fnirt", "slicetimer", "mcflirt", "fslmaths", "applywarp"]
    missing = [t for t in required if shutil.which(t) is None]
    fsl_dir = os.environ.get("FSLDIR", "")
    return missing, fsl_dir


def process_subject(
    subject_dir: Path,
    outdir: Path,
    atlas_script: Path,
    python_exe: str,
    fsl_dir: str,
    keep_work: bool,
    cache_dir: Path | None,
    cache_pattern: str,
) -> tuple[str, str]:
    """
    Process one subject and return (subject_id, status).

    Status:
        - "ok"
        - "skip"
        - "error:<msg>"
    """
    subject_id = subject_dir.name
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{subject_id}.log"

    final_csv = outdir / f"{subject_id}.csv"
    if file_nonempty(final_csv):
        return subject_id, "skip"

    if cache_dir is not None:
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
                f.write(f"[INFO] Using cache directory: {workdir}\n")

        t1_path = find_one([
            str(subject_dir / "MP-RAGE" / "*" / "*" / "*.nii*")
        ])
        fmri_path = find_one([
            str(subject_dir / "Resting_State_fMRI" / "*" / "*" / "*.nii*")
        ])
        if not t1_path or not fmri_path:
            raise FileNotFoundError(f"T1 or fMRI not found: T1={t1_path}, fMRI={fmri_path}")

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

        t1_bet_base = workdir / "T1_bet"
        if not file_nonempty(Path(str(t1_bet_base) + ".nii.gz")):
            sh(
                ["bet", "T1_input.nii.gz", "T1_bet", "-R", "-f", "0.3", "-g", "0"],
                cwd=workdir,
                log_file=log_file,
                env=env,
            )

        t1_fast_restore = workdir / "T1_fast_restore.nii.gz"
        if not file_nonempty(t1_fast_restore):
            sh(
                ["fast", "-B", "-o", "T1_fast", "T1_input.nii.gz"],
                cwd=workdir,
                log_file=log_file,
                env=env,
            )

        t1_to_mni_aff = workdir / "T1_to_MNI_aff.mat"
        if not t1_to_mni_aff.exists():
            sh(
                [
                    "flirt",
                    "-in", "T1_fast_restore.nii.gz",
                    "-ref", f"{fsl_dir}/data/standard/MNI152_T1_1mm_brain",
                    "-omat", "T1_to_MNI_aff.mat",
                    "-dof", "12",
                ],
                cwd=workdir,
                log_file=log_file,
                env=env,
            )

        t1_to_mni_warp = workdir / "T1_to_MNI_warp.nii.gz"
        t1_to_mni_iout = workdir / "T1_to_MNI.nii.gz"
        if not file_nonempty(t1_to_mni_warp) or not file_nonempty(t1_to_mni_iout):
            sh(
                [
                    "fnirt",
                    "--in=T1_input.nii.gz",
                    "--aff=T1_to_MNI_aff.mat",
                    f"--ref={fsl_dir}/data/standard/MNI152_T1_1mm",
                    "--iout=T1_to_MNI",
                    "--cout=T1_to_MNI_warp",
                ],
                cwd=workdir,
                log_file=log_file,
                env=env,
            )

        func_stc = workdir / "func_stc.nii.gz"
        if not file_nonempty(func_stc):
            sh(
                ["slicetimer", "-i", "func_input.nii.gz", "-o", "func_stc"],
                cwd=workdir,
                log_file=log_file,
                env=env,
            )

        func_mc = workdir / "func_mc.nii.gz"
        if not file_nonempty(func_mc):
            sh(
                ["mcflirt", "-in", "func_stc", "-out", "func_mc", "-plots", "-meanvol"],
                cwd=workdir,
                log_file=log_file,
                env=env,
            )

        func_mc_mean = workdir / "func_mc_mean.nii.gz"
        if not file_nonempty(func_mc_mean):
            sh(
                ["fslmaths", "func_mc", "-Tmean", "func_mc_mean"],
                cwd=workdir,
                log_file=log_file,
                env=env,
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
                    "-cost", "normmi",
                ],
                cwd=workdir,
                log_file=log_file,
                env=env,
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
                    "-w", "T1_to_MNI_warp",
                ],
                cwd=workdir,
                log_file=log_file,
                env=env,
            )

        if not file_nonempty(final_csv):
            ensure_parent(final_csv)
            sh(
                [
                    python_exe,
                    str(atlas_script),
                    "--func", "func_MNI.nii.gz",
                    "--out", str(final_csv),
                ],
                cwd=workdir,
                log_file=log_file,
                env=env,
            )

        if not keep_work:
            try:
                shutil.rmtree(workdir)
            except Exception as e:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n[WARN] Failed to remove workdir: {e}\n")

        with open(log_file, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] ==== Done subject {subject_id} ====\n")

        return subject_id, "ok"

    except Exception as e:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[ERROR] {e}\n")
        return subject_id, f"error:{e}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch fMRI-to-CSV pipeline with FSL, multiprocessing, and resume support."
    )
    parser.add_argument(
        "--root",
        default=Path("./ABIDE"),
        type=Path,
        help="Root directory containing subject subdirectories.",
    )
    parser.add_argument(
        "--outdir",
        default=Path("./ABIDE_AAL3_csv_out"),
        type=Path,
        help="Output directory for <subject>.csv files and logs.",
    )
    parser.add_argument(
        "--atlas-script",
        default=Path("/Data_process/last_process_AAL.py"),
        type=Path,
        help="Python script used to extract atlas BOLD signals.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Number of worker processes. Default is half of available CPUs.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default="python3",
        help="Python executable used to run the atlas script.",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep intermediate files in the work directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache root directory for intermediate files.",
    )
    parser.add_argument(
        "--cache-pattern",
        type=str,
        default="PID({sid})",
        help="Subdirectory naming template for cache directories.",
    )

    args = parser.parse_args()

    missing, fsl_dir = check_tools_exist()
    if missing:
        print(f"[FATAL] Missing required commands. Please check FSL installation and PATH: {missing}", file=sys.stderr)
        sys.exit(2)
    if not fsl_dir:
        print("[WARN] $FSLDIR is not set. If later steps fail, export FSLDIR and add $FSLDIR/bin to PATH.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    subjects = [d for d in sorted(args.root.iterdir()) if d.is_dir()]
    if not subjects:
        print("[FATAL] No subject subdirectories found under the root directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(subjects)} subject directories; running with {args.workers} workers.")
    if args.cache_dir is not None:
        print(f"Using cache directory: {args.cache_dir}, naming pattern: {args.cache_pattern}")

    results = {"ok": [], "skip": [], "error": []}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for sd in subjects:
            futs.append(
                ex.submit(
                    process_subject,
                    sd,
                    args.outdir,
                    args.atlas_script,
                    args.python_exe,
                    fsl_dir,
                    args.keep_work,
                    args.cache_dir,
                    args.cache_pattern,
                )
            )

        for fu in as_completed(futs):
            sid, status = fu.result()
            if status == "ok":
                print(f"[OK] {sid}")
                results["ok"].append(sid)
            elif status == "skip":
                print(f"[SKIP] {sid} (CSV already exists)")
                results["skip"].append(sid)
            else:
                print(f"[ERROR] {sid} -> {status}")
                results["error"].append((sid, status))

    summary_file = args.outdir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(
            f"Completed: {len(results['ok'])}\n"
            f"Skipped: {len(results['skip'])}\n"
            f"Failed: {len(results['error'])}\n\n"
        )
        if results["ok"]:
            f.write("OK:\n" + "\n".join(results["ok"]) + "\n\n")
        if results["skip"]:
            f.write("SKIP:\n" + "\n".join(results["skip"]) + "\n\n")
        if results["error"]:
            f.write("ERROR:\n" + "\n".join([f"{sid} -> {msg}" for sid, msg in results["error"]]) + "\n")

    print(
        f"\nDone: OK={len(results['ok'])}, "
        f"SKIP={len(results['skip'])}, "
        f"ERROR={len(results['error'])}"
    )
    print(f"Summary written to: {summary_file}")


if __name__ == "__main__":
    main()