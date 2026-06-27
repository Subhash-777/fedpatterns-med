#!/bin/bash
export FED_EXPERIMENT_NAME="M6-E3-NonIID-ChestMNIST"
export FED_DATASET="ChestMNIST"
export FED_ENABLE_DP="False"
export FED_ENABLE_STRAGGLER="False"

ALPHAS=("0.1" "0.5" "1.0")

for alpha in "${ALPHAS[@]}"; do
    export FED_ALPHA="$alpha"
    echo "=== Running alpha=$alpha ==="

    # B1 (FedAvg)
    echo "Running B1 Run 1..."
    /home/subhash/miniconda3/envs/fedpatterns-med/bin/python baselines/b1_vanilla_flower_fedavg/run_b1.py
    echo "Running B1 Run 2..."
    /home/subhash/miniconda3/envs/fedpatterns-med/bin/python baselines/b1_vanilla_flower_fedavg/run_b1.py

    # B2 (FedProx)
    echo "Running B2 Run 1..."
    /home/subhash/miniconda3/envs/fedpatterns-med/bin/python baselines/b2_vanilla_flower_advanced_strategy/run_b2.py
    echo "Running B2 Run 2..."
    /home/subhash/miniconda3/envs/fedpatterns-med/bin/python baselines/b2_vanilla_flower_advanced_strategy/run_b2.py

    # B4 (Patternized - No DP, No Straggler)
    echo "Running B4 Run 1..."
    /home/subhash/miniconda3/envs/fedpatterns-med/bin/python fedpatterns/core/app/run_patternized.py
    echo "Running B4 Run 2..."
    /home/subhash/miniconda3/envs/fedpatterns-med/bin/python fedpatterns/core/app/run_patternized.py
done
