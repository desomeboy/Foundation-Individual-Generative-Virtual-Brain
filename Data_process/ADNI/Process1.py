"""
批量将分层 DICOM (.dcm) 数据集转换为 NIfTI（.nii），并生成 CSV 索引。
目录结构（示例）：
root/
  ├─ PATIENT_ID/
  │    ├─ CT/
  │    │   ├─ 2021-01-01/
  │    │   │   └─ Ixxxx.../   # 该层应只有一个以 "I" 开头的目录
  │    │   │       └─ *.dcm
  │    │   └─ 2021-06-01/ ...
  │    └─ MR/ ...
  └─ PATIENT_ID_2/ ...

需求：
- 将每个“时间”目录下唯一的 I* 序列用 dcm2niix 转为 .nii 并存放在该“时间”目录下
- 生成 CSV：patient_id, modality, time, nifti_path
- 若“时间”目录下发现 0 个或 >1 个 I* 子目录，提醒并跳过
- 支持多进程加速
- 默认输出 .nii（非 .nii.gz）

依赖：
- 已安装 dcm2niix 命令行工具（本脚本会通过子进程调用）

用法：
python dicom2nii_multiproc.py --root /path/to/dataset --csv /path/to/index.csv --workers 8
"""



import argparse
import csv
import logging
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

I_DIR_PATTERN = re.compile(r"^I[\w\-.]+$", re.IGNORECASE)
DCM_SUFFIX = ".dcm"

@dataclass
class TimeTask:
    patient_id: str
    modality: str
    time_name: str
    time_dir: Path
    series_dir: Path  # Ixxxx... 目录


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
                    logging.warning("未找到以 'I' 开头的图像目录：%s", time_dir)
                    warns += 1
                    skipped += 1
                    continue
                if len(i_dirs) > 1:
                    logging.warning("发现多个以 'I' 开头的目录，已选择第一个：%s -> %s", time_dir, i_dirs[0].name)
                    warns += 1
                series_dir = i_dirs[0]

                if not has_dcm_files(series_dir):
                    logging.warning("序列目录下未发现 .dcm 文件，已跳过：%s", series_dir)
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
        dcm2niix_bin, "-z", z, "-b", "n", "-v", "n",
        "-f", out_prefix, "-o", str(out_dir), str(series_dir),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode != 0:
            return False, None, f"dcm2niix 失败：{proc.stderr.strip() or proc.stdout.strip()}"
        produced_nii = sorted(out_dir.glob(out_prefix + "*.nii"))
        produced_gz = sorted(out_dir.glob(out_prefix + "*.nii.gz"))
        if produced_nii:
            return True, str(produced_nii[0].resolve()), None
        if produced_gz:
            return True, str(produced_gz[0].resolve()), "生成了 .nii.gz 而不是 .nii"
        return False, None, "未发现生成的 NIfTI 文件"
    except Exception as e:
        return False, None, f"调用 dcm2niix 异常：{e}"


def worker(task: TimeTask, dcm2niix_bin: str, gz: bool):
    prefix = sanitize(f"{task.patient_id}_{task.modality}_{task.time_name}")
    ok, nii_path, msg = run_dcm2niix(dcm2niix_bin, task.series_dir, task.time_dir, prefix, gz=gz)
    return task, ok, nii_path, msg


def main():
    parser = argparse.ArgumentParser(description="批量 DICOM->NIfTI 转换（多进程，dcm2niix）并生成 CSV 索引")
    parser.add_argument("--root", type=Path, default='/ailab/group/medai-share/syDu/Brain_EC/ADNI2/ADNI', help="数据集根目录")
    parser.add_argument("--csv", type=Path, required=False, default=Path("dataset_index.csv"), help="输出 CSV 路径（默认：./dataset_index.csv）")
    parser.add_argument("--workers", type=int, default=32, help="并行进程数（默认=CPU核数）")
    parser.add_argument("--overwrite", action="store_true", help="若时间目录下已存在 .nii，是否覆盖（默认不覆盖）")
    parser.add_argument("--gz", action="store_true", help="输出 .nii.gz（默认关闭，输出 .nii）")
    parser.add_argument("--dcm2niix", type=str, default="dcm2niix", help="dcm2niix 可执行文件路径（默认已在 PATH 中）")
    parser.add_argument("--log", type=str, default="INFO", help="日志级别：DEBUG/INFO/WARNING/ERROR（默认 INFO）")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    tasks, skipped0, warns0 = discover_tasks(args.root, overwrite=args.overwrite)
    logging.info("发现任务 %d 个，跳过 %d 个，警告 %d 个", len(tasks), skipped0, warns0)

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
                logging.warning("转换失败：(%s/%s/%s) %s", task.patient_id, task.modality, task.time_name, msg or "")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "modality", "time", "nifti_path"])
        for row in sorted(results, key=lambda r: (r[0], r[1], r[2], r[3])):
            writer.writerow(row)

    logging.info("完成。成功: %d，跳过: %d，警告: %d", total_success, skipped, warns)
    logging.info("CSV 索引：%s", args.csv.resolve())


if __name__ == "__main__":
    main()