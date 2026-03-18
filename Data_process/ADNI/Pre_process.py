"""
Batch-convert hierarchical DICOM (.dcm) datasets to NIfTI (.nii) and generate a CSV index.

Example directory layout:
root/
  ├─ PATIENT_ID/
  │    ├─ CT/
  │    │   ├─ 2021-01-01/
  │    │   │   └─ Ixxxx.../   # This level should contain exactly one directory starting with "I"
  │    │   │       └─ *.dcm
  │    │   └─ 2021-06-01/ ...
  │    └─ MR/ ...
  └─ PATIENT_ID_2/ ...

Requirements:
- Convert the unique I* series under each "time" directory to .nii using dcm2niix
- Save the output .nii file inside the same "time" directory
- Generate a CSV with columns: patient_id, modality, time, nifti_path
- Warn and skip if a "time" directory contains 0 or more than 1 I* subdirectories
- Support multiprocessing
- Default output is .nii instead of .nii.gz

Dependencies:
- dcm2niix command-line tool must already be installed
- This script invokes dcm2niix via subprocess
"""

import argparse
import csv
import logging
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

I_DIR_PATTERN = re.compile(r"^I[\w\-.]+$", re.IGNORECASE)
DCM_SUFFIX = ".dcm"


@dataclass
class TimeTask:
    patient_id: str
    modality: str
    time_name: str
    time_dir: Path
    series_dir: Path  # Ixxxx... directory


def find_i_dirs(time_dir: Path) -> List[Path]:
    if not time_dir.is_dir():
        return []
    return [p for p in time_dir.iterdir() if p.is_dir() and I_DIR_PATTERN.match(p.name)]


def has_dcm_files(folder: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() == DCM_SUFFIX for p in folder.rglob("*"))


def sanitize(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^\w\-.]+", "_", s)


def discover_tasks(root: Path, overwrite: bool) -> Tuple[List[TimeTask], int, int]:
    tasks: List[TimeTask] = []
    skipped = 0
    warns = 0

    for patient_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        patient_id = patient_dir.name
        for modality_dir in sorted([p for p in patient_dir.iterdir() if p.is_dir()]):
            modality = modality_dir.name
            for time_dir in sorted([p for p in modality_dir.iterdir() if p.is_dir()]):
                time_name = time_dir.name

                i_dirs = find_i_dirs(time_dir)
                if len(i_dirs) == 0:
                    logging.warning("No image directory starting with 'I' found: %s", time_dir)
                    warns += 1
                    skipped += 1
                    continue
                if len(i_dirs) > 1:
                    logging.warning(
                        "Multiple directories starting with 'I' found, using the first one: %s -> %s",
                        time_dir,
                        i_dirs[0].name,
                    )
                    warns += 1

                series_dir = i_dirs[0]

                if not has_dcm_files(series_dir):
                    logging.warning("No .dcm files found under series directory, skipped: %s", series_dir)
                    warns += 1
                    skipped += 1
                    continue

                if not overwrite:
                    existing_nii = list(time_dir.glob("*.nii"))
                    if existing_nii:
                        continue

                tasks.append(TimeTask(patient_id, modality, time_name, time_dir, series_dir))

    return tasks, skipped, warns


def run_dcm2niix(dcm2niix_bin: str, series_dir: Path, out_dir: Path, out_prefix: str, gz: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    z = "y" if gz else "n"
    cmd = [
        dcm2niix_bin,
        "-z",
        z,
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
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode != 0:
            return False, None, f"dcm2niix failed: {proc.stderr.strip() or proc.stdout.strip()}"

        produced_nii = sorted(out_dir.glob(out_prefix + "*.nii"))
        produced_gz = sorted(out_dir.glob(out_prefix + "*.nii.gz"))
        if produced_nii:
            return True, str(produced_nii[0].resolve()), None
        if produced_gz:
            return True, str(produced_gz[0].resolve()), "Generated .nii.gz instead of .nii"
        return False, None, "No generated NIfTI file found"
    except Exception as e:
        return False, None, f"Exception while calling dcm2niix: {e}"


def worker(task: TimeTask, dcm2niix_bin: str, gz: bool):
    prefix = sanitize(f"{task.patient_id}_{task.modality}_{task.time_name}")
    ok, nii_path, msg = run_dcm2niix(dcm2niix_bin, task.series_dir, task.time_dir, prefix, gz=gz)
    return task, ok, nii_path, msg


def main():
    parser = argparse.ArgumentParser(
        description="Batch DICOM-to-NIfTI conversion with multiprocessing and CSV index generation"
    )
    parser.add_argument("--root", type=Path, required=True, help="Dataset root directory")
    parser.add_argument(
        "--csv",
        type=Path,
        required=False,
        default=Path("dataset_index.csv"),
        help="Output CSV path (default: ./dataset_index.csv)",
    )
    parser.add_argument("--workers", type=int, default=32, help="Number of worker processes (default: 32)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite when .nii already exists under the time directory (default: disabled)",
    )
    parser.add_argument("--gz", action="store_true", help="Output .nii.gz instead of .nii (default: disabled)")
    parser.add_argument(
        "--dcm2niix",
        type=str,
        default="dcm2niix",
        help="Path to the dcm2niix executable (default: resolved from PATH)",
    )
    parser.add_argument("--log", type=str, default="INFO", help="Logging level: DEBUG/INFO/WARNING/ERROR")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    tasks, skipped0, warns0 = discover_tasks(args.root, overwrite=args.overwrite)
    logging.info("Discovered %d tasks, skipped %d, warnings %d", len(tasks), skipped0, warns0)

    results = []
    total_success = 0
    warns = warns0
    skipped = skipped0

    if not args.overwrite:
        for patient_dir in sorted([p for p in args.root.iterdir() if p.is_dir()]):
            for modality_dir in sorted([p for p in patient_dir.iterdir() if p.is_dir()]):
                for time_dir in sorted([p for p in modality_dir.iterdir() if p.is_dir()]):
                    existing_nii = list(time_dir.glob("*.nii")) if not args.gz else list(time_dir.glob("*.nii.gz"))
                    for nii in existing_nii:
                        results.append((patient_dir.name, modality_dir.name, time_dir.name, str(nii.resolve())))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, t, args.dcm2niix, args.gz) for t in tasks]
        for fut in as_completed(futures):
            task, ok, nii_path, msg = fut.result()
            if ok and nii_path:
                results.append((task.patient_id, task.modality, task.time_name, nii_path))
                total_success += 1
                if msg:
                    logging.warning("(%s/%s/%s) %s", task.patient_id, task.modality, task.time_name, msg)
                    warns += 1
            else:
                skipped += 1
                warns += 1
                logging.warning(
                    "Conversion failed: (%s/%s/%s) %s",
                    task.patient_id,
                    task.modality,
                    task.time_name,
                    msg or "",
                )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "modality", "time", "nifti_path"])
        for row in sorted(results, key=lambda r: (r[0], r[1], r[2], r[3])):
            writer.writerow(row)

    logging.info("Done. Success: %d, skipped: %d, warnings: %d", total_success, skipped, warns)
    logging.info("CSV index: %s", args.csv.resolve())


if __name__ == "__main__":
    main()