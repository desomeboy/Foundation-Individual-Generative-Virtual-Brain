#!/bin/bash

CSV_DIR="/ailab/group/medai-share/syDu/ruijin/10.10/AAL3_csv"
SCRIPT="scripts/test_ruijin_npi.py"
# MODEL_PATH="/ailab/user/dusiyuan/code/Brain/EC/EC_results_num_layer_2/lr_5e-05_batch_256_epochs_200_l2_0.001_patience_100_steps_7_dmodel_768/best_model.pth"
MODEL_PATH='/ailab/user/dusiyuan/code/Brain/EC/AAL3/EC_results_num_layer_2/AAL3_lr_5e-05_batch_256_epochs_300_l2_0.0001_patience_100_steps_7_dmodel_256/best_model.pth'
OUTPUT_BASE="/ailab/group/medai-share/syDu/ruijin/10.10/AAL_VTB_compare_ruijinHC"

mkdir -p "$OUTPUT_BASE"

for csv in "$CSV_DIR"/*.csv; do
    if [ -f "$csv" ]; then
        patient=$(basename "$csv" .csv)
        echo "Testing patient: $patient"
        python "$SCRIPT" \
            --patient_csv "$csv" \
            --model_path "$MODEL_PATH" \
            --output_dir "$OUTPUT_BASE/$patient" \
            --num_layers 2 \
            --num_cross_layers 1 \
            --fine_tune --d_model 256
        echo "Done: $patient"
        echo "-----------------------------"
    fi
done

echo "All done!"