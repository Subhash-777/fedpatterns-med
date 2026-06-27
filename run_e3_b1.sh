#!/bin/bash
export FED_EXPERIMENT_NAME="M6-E3-NonIID-PneumoniaMNIST"

for alpha in 0.1 0.5 1.0; do
    export FED_ALPHA="$alpha"
    echo "=== Running B1 Pneumonia E3 alpha=$alpha Run 1 ==="
    /home/subhash/miniconda3/envs/fedpatterns-med/bin/python baselines/b1_vanilla_flower_fedavg/run_b1.py --dataset PneumoniaMNIST --num-rounds 3 --num-clients 5 --alpha $alpha
    echo "=== Running B1 Pneumonia E3 alpha=$alpha Run 2 ==="
    /home/subhash/miniconda3/envs/fedpatterns-med/bin/python baselines/b1_vanilla_flower_fedavg/run_b1.py --dataset PneumoniaMNIST --num-rounds 3 --num-clients 5 --alpha $alpha
done
