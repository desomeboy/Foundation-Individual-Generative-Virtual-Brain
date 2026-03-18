"""
DICOM to NIfTI conversion and CSV post-processing pipeline.

Features:
- Scan a hierarchical DICOM dataset
- Convert every I* series under each time folder using dcm2niix
- Generate a raw CSV index
- Normalize patient_id by keeping digits only
- Keep patients who have fMRI and at least one non-fMRI modality
- Standardize modality names
- Export a final cleaned CSV

Expected dataset layout:
dataset_root/
  PATIENT_ID/
    MODALITY/
      TIME_NAME/
        Ixxxx.../
          *.dcm
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


I_DIR_PATTERN = re.compile(r"^I[\w\-.]+$", re.IGNORECASE)
DCM_SUFFIX = ".dcm"

FMRI_KEYWORDS = ["RESTING_STATE", "bold_rest", "rsfMRI", "fMRI"]

STRICT_RSFMRI_LIST = [
    "rsfMRI_RL",
    "R_L_RESTING_STATE_FMRI_ep2d_fid_basic_bold",
    "rsFMRI_ep2d",
    "ep2d_bold_rest",
    "ep2d_RESTING_STATE",
    "RESTING_STATE_fMRI_FAT_SHIFT_LEFT",
    "rsfMRI",
    "rsfMRI_PA",
    "rsfMRI_R-L",
]

T1_KEYWORDS = ["MPRAGE", "T1"]


@dataclass
class SeriesTask:
    patient_id: str
    modality: str
    time_name: str
    time_dir: Path
    series_dir: Path
    test_name: str


def sanitize(text: str) -> str:
    text = text.strip().replace(" ", "_")
    return re.sub(r"[^\w\-.]+", "_", text)


def extract_digits(text: object) -> str:
    return "".join(re.findall(r"\d", str(text)))


def find_i_dirs(time_dir: Path) -> List[Path]:
    if not time_dir.is_dir():
        return []
    return sorted(
        [p for p in time_dir.iterdir() if p.is_dir() and I_DIR_PATTERN.match(p.name)],
        key=lambda p: p.name,
    )


def has_dcm_files(folder: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() == DCM_SUFFIX for p in folder.rglob("*"))


def safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def discover_tasks(root: Path, overwrite: bool, gz: bool) -> Tuple[List[SeriesTask], int, int]:
    tasks: List[SeriesTask] = []
    skipped = 0
    warnings_count = 0

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    for patient_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        patient_id = patient_dir.name

        for modality_dir in sorted([p for p in patient_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            modality = modality_dir.name

            for time_dir in sorted([p for p in modality_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                time_name = time_dir.name
                i_dirs = find_i_dirs(time_dir)

                if not i_dirs:
                    logging.warning("No I* series directory found: %s", time_dir)
                    skipped += 1
                    warnings_count += 1
                    continue

                for series_dir in i_dirs:
                    if not has_dcm_files(series_dir):
                        logging.warning("No DICOM files found in series directory: %s", series_dir)
                        skipped += 1
                        warnings_count += 1
                        continue

                    test_name = series_dir.name
                    prefix = sanitize(f"{patient_id}_{modality}_{time_name}_{test_name}")

                    if not overwrite:
                        pattern = prefix + ("*.nii.gz" if gz else "*.nii")
                        existing = list(time_dir.glob(pattern))
                        if existing:
                            continue

                    tasks.append(
                        SeriesTask(
                            patient_id=patient_id,
                            modality=modality,
                            time_name=time_name,
                            time_dir=time_dir,
                            series_dir=series_dir,
                            test_name=test_name,
                        )
                    )

    return tasks, skipped, warnings_count


def run_dcm2niix(
    dcm2niix_bin: str,
    series_dir: Path,
    out_dir: Path,
    out_prefix: str,
    gz: bool = False,
) -> Tuple[bool, Optional[Path], Optional[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    z_flag = "y" if gz else "n"

    cmd = [
        dcm2niix_bin,
        "-z",
        z_flag,
        "-b",
        "n",
        "-v",
        "n",
        "-f",
        out_prefix,
        "-o",
        str(out_dir),
        str(series_dir),
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if proc.returncode != 0:
            message = proc.stderr.strip() or proc.stdout.strip() or "Unknown dcm2niix error"
            return False, None, f"dcm2niix failed: {message}"

        produced_nii = sorted(out_dir.glob(out_prefix + "*.nii"))
        produced_niigz = sorted(out_dir.glob(out_prefix + "*.nii.gz"))

        if produced_nii:
            return True, produced_nii[0], None

        if produced_niigz:
            return True, produced_niigz[0], "Generated .nii.gz instead of .nii"

        return False, None, "No NIfTI file was produced"

    except Exception as exc:
        return False, None, f"Exception while running dcm2niix: {exc}"


def worker(task: SeriesTask, dcm2niix_bin: str, gz: bool) -> Tuple[SeriesTask, bool, Optional[Path], Optional[str]]:
    prefix = sanitize(f"{task.patient_id}_{task.modality}_{task.time_name}_{task.test_name}")
    ok, nii_path, msg = run_dcm2niix(
        dcm2niix_bin=dcm2niix_bin,
        series_dir=task.series_dir,
        out_dir=task.time_dir,
        out_prefix=prefix,
        gz=gz,
    )
    return task, ok, nii_path, msg


def collect_existing_nifti_rows(root: Path, gz: bool) -> List[Tuple[str, str, str, str, str]]:
    rows: List[Tuple[str, str, str, str, str]] = []

    for patient_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        pid = patient_dir.name

        for modality_dir in sorted([p for p in patient_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            mod = modality_dir.name

            for time_dir in sorted([p for p in modality_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
                time_name = time_dir.name
                base_prefix = sanitize(f"{pid}_{mod}_{time_name}_")
                pattern = base_prefix + ("*.nii.gz" if gz else "*.nii")

                for nii in sorted(time_dir.glob(pattern), key=lambda p: p.name):
                    filename = nii.name
                    tail = filename[len(base_prefix):]

                    if tail.endswith(".nii.gz"):
                        tail = tail[:-7]
                    elif tail.endswith(".nii"):
                        tail = tail[:-4]

                    test_name = tail.split("_")[0] if "_" in tail else tail
                    rows.append((pid, mod, time_name, test_name, safe_relpath(nii, root)))

    return rows


def write_csv(rows: Iterable[Tuple[str, str, str, str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r[0], r[1], r[2], r[3], r[4]))

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "modality", "time", "test_name", "nifti_path"])
        for row in rows:
            writer.writerow(row)


def postprocess_csv(raw_csv: Path, final_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)

    # Normalize patient_id: keep digits only
    df["patient_id"] = df["patient_id"].apply(extract_digits)

    # Identify fMRI rows
    fmri_pattern = "|".join(FMRI_KEYWORDS)
    df["is_fmri"] = df["modality"].str.contains(fmri_pattern, case=False, na=False)

    # Keep patients who have fMRI and at least one non-fMRI modality
    fmri_patients = df[df["is_fmri"]]["patient_id"].unique()
    valid_patients = []

    for pid in fmri_patients:
        patient_rows = df[df["patient_id"] == pid]
        if patient_rows[~patient_rows["is_fmri"]].shape[0] > 0:
            valid_patients.append(pid)

    df = df[df["patient_id"].isin(valid_patients)].copy()
    df.drop(columns=["is_fmri"], inplace=True)

    # Standardize modalities
    df["keep"] = False
    df["new_modality"] = df["modality"]

    mask_strict_rsfmri = df["modality"].isin(STRICT_RSFMRI_LIST)
    df.loc[mask_strict_rsfmri, "new_modality"] = "Resting_State_fMRI"
    df.loc[mask_strict_rsfmri, "keep"] = True

    t1_pattern = "|".join(T1_KEYWORDS)
    mask_t1 = df["modality"].str.contains(t1_pattern, case=False, na=False)
    df.loc[mask_t1, "new_modality"] = "MPRAGE"
    df.loc[mask_t1, "keep"] = True

    df = df[df["keep"]].copy()
    df["modality"] = df["new_modality"]
    df.drop(columns=["keep", "new_modality"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    final_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(final_csv, index=False)

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DICOM to NIfTI, generate raw CSV, and export a cleaned final CSV."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root directory",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default="",
        help="Output path for the raw CSV index",
    )
    parser.add_argument(
        "--final-csv",
        type=Path,
        default="final_csv.csv",
        help="Output path for the final cleaned CSV",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NIfTI files if present",
    )
    parser.add_argument(
        "--gz",
        action="store_true",
        help="Generate .nii.gz instead of .nii",
    )
    parser.add_argument(
        "--dcm2niix",
        type=str,
        default="dcm2niix",
        help="Path to dcm2niix executable",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    root = args.root.resolve()
    raw_csv = args.raw_csv.resolve()
    final_csv = args.final_csv.resolve()

    tasks, skipped_init, warnings_init = discover_tasks(
        root=root,
        overwrite=args.overwrite,
        gz=args.gz,
    )

    logging.info("Discovered %d conversion tasks", len(tasks))
    logging.info("Initial skipped entries: %d", skipped_init)
    logging.info("Initial warnings: %d", warnings_init)

    results: List[Tuple[str, str, str, str, str]] = []
    success_count = 0
    skipped_count = skipped_init
    warning_count = warnings_init

    if not args.overwrite:
        existing_rows = collect_existing_nifti_rows(root=root, gz=args.gz)
        results.extend(existing_rows)
        logging.info("Collected %d existing NIfTI records", len(existing_rows))

    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(worker, task, args.dcm2niix, args.gz) for task in tasks]

            for future in as_completed(futures):
                task, ok, nii_path, msg = future.result()

                if ok and nii_path is not None:
                    results.append(
                        (
                            task.patient_id,
                            task.modality,
                            task.time_name,
                            task.test_name,
                            safe_relpath(nii_path, root),
                        )
                    )
                    success_count += 1

                    if msg:
                        logging.warning(
                            "Completed with warning: (%s / %s / %s / %s) %s",
                            task.patient_id,
                            task.modality,
                            task.time_name,
                            task.test_name,
                            msg,
                        )
                        warning_count += 1
                else:
                    skipped_count += 1
                    warning_count += 1
                    logging.warning(
                        "Conversion failed: (%s / %s / %s / %s) %s",
                        task.patient_id,
                        task.modality,
                        task.time_name,
                        task.test_name,
                        msg or "",
                    )

    write_csv(results, raw_csv)
    logging.info("Raw CSV written to: %s", raw_csv)

    final_df = postprocess_csv(raw_csv=raw_csv, final_csv=final_csv)
    logging.info("Final CSV written to: %s", final_csv)

    total_patients = final_df["patient_id"].nunique() if not final_df.empty else 0
    total_rows = len(final_df)

    logging.info("Successful conversions: %d", success_count)
    logging.info("Skipped items: %d", skipped_count)
    logging.info("Warnings: %d", warning_count)
    logging.info("Final unique patients: %d", total_patients)
    logging.info("Final rows: %d", total_rows)


if __name__ == "__main__":
    main()