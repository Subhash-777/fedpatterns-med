#!/bin/bash
export FED_EXPERIMENT_NAME="M6-E3-NonIID-ChestMNIST"
export FED_DATASET="ChestMNIST"
export FED_ENABLE_DP="False"
export FED_ENABLE_STRAGGLER="False"

for alpha in 0.5 1.0; do
    export FED_ALPHA="$alpha"
    echo "=== Running B4 ChestMNIST E3 alpha=$alpha Run 1 ==="
    /home/subhash/miniconda3/envs/fedpatterns-med/bin/python fedpatterns/core/app/run_patternized.py
    echo "=== Running B4 ChestMNIST E3 alpha=$alpha Run 2 ==="
    /home/subhash/miniconda3/envs/fedpatterns-med/bin/python fedpatterns/core/app/run_patternized.py
done
