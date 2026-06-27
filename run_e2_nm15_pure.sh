#!/bin/bash
export FED_EXPERIMENT_NAME="M6-E2-DPTradeOff"
export FED_DATASET="PneumoniaMNIST"
export FED_ENABLE_DP="True"
export FED_ENABLE_STRAGGLER="False"
export FED_NOISE_MULTIPLIER="1.5"
export FED_ALPHA="1.0"

echo "=== Running nm=1.5 Run 1 ==="
/home/subhash/miniconda3/envs/fedpatterns-med/bin/python fedpatterns/core/app/run_patternized.py
echo "=== Running nm=1.5 Run 2 ==="
/home/subhash/miniconda3/envs/fedpatterns-med/bin/python fedpatterns/core/app/run_patternized.py
