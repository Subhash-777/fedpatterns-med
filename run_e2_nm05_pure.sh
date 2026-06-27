#!/bin/bash
export FED_ENABLE_STRAGGLER="False"
export FED_DATASET="PneumoniaMNIST"
export FED_EXPERIMENT_NAME="M6-E2-DPTradeOff"
export FED_NOISE_MULTIPLIER="0.5"
export MLFLOW_TRACKING_URI="sqlite:////home/subhash/projects/fedpatterns-med/mlflow.db"

# We run nm=0.5 twice to confirm 2-run determinism
for i in 1 2
do
    echo "=== Running nm=0.5 Run $i ==="
    /home/subhash/miniconda3/envs/fedpatterns-med/bin/python fedpatterns/core/app/run_patternized.py
done
