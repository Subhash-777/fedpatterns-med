"""
baselines/b3_architectural_pattern_baseline/run_b3.py

Entry point for the B3 Architectural Pattern baseline.
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

EXPERIMENT_NAME = "M5-B3-LoEtAlArchitectural"
TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlflow.db"
OUTPUT_CSV = PROJECT_ROOT / "output" / "logs" / "b3_results.csv"

def run_simulation(run_id: str) -> None:
    env = os.environ.copy()
    env["MLFLOW_RUN_ID"] = run_id
    env["MLFLOW_TRACKING_URI"] = TRACKING_URI

    print(f"[run_b3] Starting flwr simulation (PROJECT_ROOT={PROJECT_ROOT})")
    flwr_bin = os.path.join(os.path.dirname(sys.executable), "flwr")
    result = subprocess.run(
        [
            flwr_bin, "run", "baselines/b3_architectural_pattern_baseline", "local", "--stream",
            "--run-config", f"mlflow-run-id=\"{run_id}\" mlflow-tracking-uri=\"{TRACKING_URI}\"",
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
        print("[run_b3] No metrics found to save.")
        return
        
    df = pd.DataFrame(records)
    df_pivot = df.pivot_table(
        index=["run_id", "step"], 
        columns="metric", 
        values="value"
    ).reset_index()
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_pivot.to_csv(OUTPUT_CSV, index=False)
    print(f"[run_b3] Results saved to {OUTPUT_CSV}")

def main() -> None:
    experiment_id = setup_mlflow(experiment_name=EXPERIMENT_NAME, tracking_uri=TRACKING_URI)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"[run_b3] MLflow run started: {run_id}")

        config_meta = {
            "milestone": "M5",
            "baseline_id": "B3",
            "dataset": "PneumoniaMNIST",
            "strategy": "ArchitecturalFedAvg",
            "seed": "42",
        }
        mlflow.set_tags(config_meta)
        mlflow.log_params(config_meta)

        run_simulation(run_id=run_id)
        save_run_summary(run_id=run_id)

    print(f"[run_b3] Done.")

if __name__ == "__main__":
    main()
