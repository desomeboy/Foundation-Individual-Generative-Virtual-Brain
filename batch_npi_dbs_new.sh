#!/bin/bash

DBS_INDEX="/ailab/group/medai-share/syDu/ruijin/DBS/AAL3_Index.csv"
SCRIPT="scripts/test_ruijin_npi.py"
# MODEL_PATH="/ailab/user/dusiyuan/code/Brain/EC/EC_results_num_layer_2/lr_5e-05_batch_256_epochs_200_l2_0.001_patience_100_steps_7_dmodel_768/best_model.pth"
MODEL_PATH='/ailab/user/dusiyuan/code/Brain/EC/AAL3/EC_results_num_layer_2/AAL3_lr_5e-05_batch_256_epochs_300_l2_0.0001_patience_100_steps_7_dmodel_256/best_model.pth'

OUTPUT_BASE="/ailab/group/medai-share/syDu/ruijin/DBS/AAL_VTB_compare_ruijinHC"

mkdir -p "$OUTPUT_BASE"

# 使用 Python 安全读取 dbs_index.csv 的 csv_path 列
python3 -c "
import csv, sys
with open('$DBS_INDEX', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        csv_path = row['csv_path'].strip()
        if csv_path:
            print(csv_path)
" | while IFS= read -r csv_path; do
    if [ -z "$csv_path" ]; then
        continue
    fi

    if [ ! -f "$csv_path" ]; then
        echo "ERROR: File not found - '$csv_path'" >&2
        continue
    fi

    # 提取纯文件名（不含 .csv 后缀）
    output_name=$(basename "$csv_path" .csv)
    output_dir="$OUTPUT_BASE/$output_name"

    echo "Processing: $output_name"
    mkdir -p "$output_dir"

    python "$SCRIPT" \
        --patient_csv "$csv_path" \
        --model_path "$MODEL_PATH" \
        --output_dir "$output_dir" \
        --num_layers 2 \
        --num_cross_layers 1 \
        --fine_tune --d_model 256

    echo "Done: $output_name"
    echo "-----------------------------"
done

echo "All done!"