#!/bin/bash
#held_one_sample
CSV_DIR="/ailab/group/medai-share/syDu/Brain_EC/HCP/held_one_patient"
SCRIPT="scripts/test_ruijin_npi.py"
# MODEL_PATH="/ailab/user/dusiyuan/code/Brain/EC/EC_results_num_layer_2/lr_5e-05_batch_256_epochs_200_l2_0.001_patience_100_steps_7_dmodel_768/best_model.pth"
MODEL_PATH='/ailab/user/dusiyuan/code/Brain/EC/AAL3/EC_results_num_layer_2/AAL3_lr_5e-05_batch_256_epochs_300_l2_0.0001_patience_100_steps_7_dmodel_256/best_model.pth'
OUTPUT_BASE="/ailab/group/medai-share/syDu/Brain_EC/HCP/held_one_VTB"

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
            --fine_tune --d_model 256 --ft_epochs 35 --ft_lr 5e-4 --ft_patience 100 --ft_l2 1e-4
        echo "Done: $patient"
        echo "-----------------------------"
    fi
done

echo "All done!"