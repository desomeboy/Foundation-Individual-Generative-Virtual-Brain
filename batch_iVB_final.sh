#!/bin/bash

#The root directory of the CSV files containing the pre-processed fmri
CSV_DIR="./AAL3_csv"

#Finetunning Script
SCRIPT="scripts/train_iVB.py"

#FVB model path
MODEL_PATH='./AAL3_lr_5e-05_batch_256_epochs_300_l2_0.0001_patience_100_steps_7_dmodel_256/best_model.pth'

#iVB output path
OUTPUT_BASE="./AAL3_VTB"

mkdir -p "$OUTPUT_BASE"

# Array to store failed patients and their commands
declare -a FAILED_RUNS

for csv in "$CSV_DIR"/*.csv; do
    if [ -f "$csv" ]; then
        patient=$(basename "$csv" .csv)
        echo "Testing patient: $patient"
        
        # Build the command as a string for logging
        cmd="python \"$SCRIPT\" \
            --patient_csv \"$csv\" \
            --model_path \"$MODEL_PATH\" \
            --output_dir \"$OUTPUT_BASE/$patient\" \
            --num_layers 2 \
            --num_cross_layers 1 \
            --fine_tune --d_model 256"

        # Execute the command
        if eval "$cmd"; then
            echo "Done: $patient"
        else
            echo "FAILED: $patient"
            FAILED_RUNS+=("$patient|$cmd")
        fi
        echo "-----------------------------"
    fi
done

# Print summary of failures
if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo "All patients processed successfully!"
else
    echo "The following patients failed:"
    for entry in "${FAILED_RUNS[@]}"; do
        patient="${entry%%|*}"
        cmd="${entry#*|}"
        echo "Patient: $patient"
        echo "Command: $cmd"
        echo "------------------------------"
    done
fi

echo "All done!"