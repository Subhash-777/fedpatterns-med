"""
baselines/b2_vanilla_flower_advanced_strategy/run_b2.py

Entry point for the B2 FedProx baseline.
"""

import json
import os
import pathlib
import subprocess
import sys

import mlflow
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from fedpatterns.utils.mlflow_utils import setup_mlflow

EXPERIMENT_NAME = "M5-B2-FedProx"
TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlflow.db"
OUTPUT_CSV = PROJECT_ROOT / "output" / "logs" / "b2_results.csv"

def run_simulation(run_id: str) -> None:
    env = os.environ.copy()
    env["MLFLOW_RUN_ID"] = run_id
    env["MLFLOW_TRACKING_URI"] = TRACKING_URI
    
    # Ensure flower-superlink is in PATH
    bin_dir = os.path.dirname(sys.executable)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

    dataset = os.environ.get("FED_DATASET", "PneumoniaMNIST")
    alpha = os.environ.get("FED_ALPHA", "0.5")
    noise_mult = os.environ.get("FED_NOISE_MULTIPLIER", "1.0")
    enable_dp = os.environ.get("FED_ENABLE_DP", "True")
    enable_strag = os.environ.get("FED_ENABLE_STRAGGLER", "True")
    
    run_config_str = (
        f"mlflow-run-id=\"{run_id}\" "
        f"mlflow-tracking-uri=\"{TRACKING_URI}\" "
        f"fed_dataset=\"{dataset}\" "
        f"fed_alpha=\"{alpha}\" "
        f"fed_noise_multiplier=\"{noise_mult}\" "
        f"fed_enable_dp=\"{enable_dp}\" "
        f"fed_enable_straggler=\"{enable_strag}\""
    )

    print(f"[run_b2] Starting flwr simulation (PROJECT_ROOT={PROJECT_ROOT})")
    flwr_bin = os.path.join(os.path.dirname(sys.executable), "flwr")
    result = subprocess.run(
        [
            flwr_bin, "run", "baselines/b2_vanilla_flower_advanced_strategy", "local", "--stream",
            "--run-config", run_config_str,
        ],
        env=env,
        cwd=PROJECT_ROOT,
        check=True,
    )
    return result

def save_run_summary(run_id: str) -> None:
    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    run = client.get_run(run_id)
    
    records = []
    for metric_key in run.data.metrics.keys():
        history = client.get_metric_history(run_id, metric_key)
        for m in history:
            records.append({
                "run_id": run_id,
                "step": m.step,
                "metric": metric_key,
                "value": m.value,
            })
            
    if not records:
        print("[run_b2] No metrics found to save.")
        return
        
    df = pd.DataFrame(records)
    df_pivot = df.pivot_table(
        index=["run_id", "step"], 
        columns="metric", 
        values="value"
    ).reset_index()
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_pivot.to_csv(OUTPUT_CSV, index=False)
    print(f"[run_b2] Results saved to {OUTPUT_CSV}")

def main() -> None:
    experiment_name = os.environ.get("FED_EXPERIMENT_NAME", EXPERIMENT_NAME)
    alpha = os.environ.get("FED_ALPHA", "0.5")

    experiment_id = setup_mlflow(experiment_name=experiment_name, tracking_uri=TRACKING_URI)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"[run_b2] MLflow run started: {run_id}")

        config_meta = {
            "milestone": "M6",
            "baseline_id": "B2",
            "dataset": os.environ.get("FED_DATASET", "PneumoniaMNIST"),
            "strategy": "FedProx",
            "mu": "0.01",
            "seed": "42",
            "alpha": alpha,
        }
        mlflow.set_tags(config_meta)
        mlflow.log_params(config_meta)

        run_simulation(run_id=run_id)
        save_run_summary(run_id=run_id)

    print(f"[run_b2] Done.")

if __name__ == "__main__":
    main()
