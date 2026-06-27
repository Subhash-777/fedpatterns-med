#!/bin/bash
export FED_EXPERIMENT_NAME="M6-E3-NonIID-ChestMNIST"
export FED_DATASET="ChestMNIST"
export FED_ENABLE_DP="False"
export FED_ENABLE_STRAGGLER="False"

ALPHAS=("0.1" "0.5" "1.0")

for alpha in "${ALPHAS[@]}"; do
    export FED_ALPHA="$alpha"
    echo "=== Running alpha=$alpha ==="

    # B2 (FedProx)
    echo "Running B2 Run 1..."
    /home/subhash/miniconda3/envs/fedpatterns-med/bin/python baselines/b2_vanilla_flower_advanced_strategy/run_b2.py
    echo "Running B2 Run 2..."
    /home/subhash/miniconda3/envs/fedpatterns-med/bin/python baselines/b2_vanilla_flower_advanced_strategy/run_b2.py
done
