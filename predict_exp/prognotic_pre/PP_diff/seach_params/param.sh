#!/bin/bash

# 定义 seed 范围和对应的 GPU
declare -a SEED_RANGES=(
    "1001 1500"
    "1501 2000"
    "2001 2500"
    "2501 3000"
    "3001 3500"
    "3501 4000"
    "4001 4500"
    "4501 5000"
    "5001 5500"
    "5501 6000"
    "6001 6500"
    "6501 7000"
    "7001 7500"
    "7501 8000"
    "8001 8500"
    "8501 9000"
    "9001 9500"
    "9501 9999"
)

# 对应的 GPU 分配（交替使用 0 和 1）
GPUS=(0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1)

# 启动所有任务
for i in "${!SEED_RANGES[@]}"; do
    range=(${SEED_RANGES[i]})
    start=${range[0]}
    end=${range[1]}
    gpu=${GPUS[i]}
    echo "Launching: CUDA_VISIBLE_DEVICES=$gpu python pp_diff2.py --seed_start $start --seed_end $end"
    CUDA_VISIBLE_DEVICES=$gpu python pp_diff2.py --seed_start "$start" --seed_end "$end" &
done

# 等待所有后台任务完成
wait
echo "All jobs completed."