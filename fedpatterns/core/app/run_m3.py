"""
fedpatterns/core/app/run_m3.py

Entry point for the M3 Privacy & Compliance layer.

Responsibilities:
  1. Resolve PROJECT_ROOT via pathlib.
  2. Set up MLflow experiment M3-B4-PrivacyCompliance.
  3. Run Flower simulation with DP, SecureAgg, and Consent logic attached.
  4. Parse radon cc -j across all pattern directories.
  5. Save metrics to m3_results.csv.
"""

import json
import os
import pathlib
import subprocess
import sys

import mlflow
import pandas as pd

# ── Path resolution ────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))  

from fedpatterns.utils.mlflow_utils import setup_mlflow  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "M3-B4-PrivacyCompliance"
TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlflow.db"
OUTPUT_CSV = PROJECT_ROOT / "output" / "logs" / "m3_results.csv"


def run_simulation(run_id: str) -> None:
    env = os.environ.copy()
    env["MLFLOW_RUN_ID"] = run_id
    env["MLFLOW_TRACKING_URI"] = TRACKING_URI

    print(f"[run_m3] Starting flwr simulation (PROJECT_ROOT={PROJECT_ROOT})")
    # By default, run-config enables DP. We can pass enable_dp=True explicitly.
    result = subprocess.run(
        [
            "flwr", "run", ".", "local", "--stream",
            "--run-config", f"mlflow-run-id=\"{run_id}\" mlflow-tracking-uri=\"{TRACKING_URI}\" enable_dp=true",
        ],
        env=env,
        cwd=PROJECT_ROOT,
        check=True,
    )
    return result


def log_cyclomatic_complexity(run_id: str) -> float:
    dirs_to_scan = [
        str(PROJECT_ROOT / "fedpatterns" / "core" / "app"),
        str(PROJECT_ROOT / "fedpatterns" / "patterns" / "mediator"),
        str(PROJECT_ROOT / "fedpatterns" / "patterns" / "strategy"),
        str(PROJECT_ROOT / "fedpatterns" / "patterns" / "decorator"),
        str(PROJECT_ROOT / "fedpatterns" / "patterns" / "proxy"),
        str(PROJECT_ROOT / "fedpatterns" / "patterns" / "observer"),
    ]
    
    result = subprocess.run(
        ["radon", "cc"] + dirs_to_scan + ["-a", "-j"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0 or not result.stdout.strip():
        print(f"[run_m3] Warning: radon returned no output. stderr: {result.stderr}")
        return 0.0

    cc_data = json.loads(result.stdout)
    all_complexities = [
        block["complexity"]
        for blocks in cc_data.values()
        for block in blocks
    ]
    avg_cc = sum(all_complexities) / len(all_complexities) if all_complexities else 0.0
    mlflow.log_metric("cyclomatic_complexity", avg_cc, run_id=run_id)
    print(f"[run_m3] Cyclomatic complexity (avg): {avg_cc:.2f} — logged to MLflow run {run_id}")
    return avg_cc


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
        print("[run_m3] No metrics found to save.")
        return
        
    df = pd.DataFrame(records)
    df_pivot = df.pivot_table(
        index=["run_id", "step"], 
        columns="metric", 
        values="value"
    ).reset_index()
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_pivot.to_csv(OUTPUT_CSV, index=False)
    print(f"[run_m3] Results saved to {OUTPUT_CSV} ({len(df_pivot)} rounds)")


def main() -> None:
    experiment_id = setup_mlflow(experiment_name=EXPERIMENT_NAME, tracking_uri=TRACKING_URI)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"[run_m3] MLflow run started: {run_id}")

        config_meta = {
            "milestone": "M3",
            "baseline_id": "B4",
            "dataset": "PneumoniaMNIST",
            "strategy": "FedAvg",
            "seed": "42",
        }
        mlflow.set_tags(config_meta)
        mlflow.log_params(config_meta)

        run_simulation(run_id=run_id)
        log_cyclomatic_complexity(run_id=run_id)
        save_run_summary(run_id=run_id)

    print(f"[run_m3] Done. View results: mlflow ui --backend-store-uri {TRACKING_URI}")


def cleanup_dp_history() -> None:
    """Wipes DP accountant history to prevent stale state across runs or after crashes."""
    import glob
    for f in glob.glob("/tmp/fedpatterns_dp_history_*.pt"):
        try:
            os.remove(f)
        except OSError:
            pass

import atexit
atexit.register(cleanup_dp_history)

if __name__ == "__main__":
    # Also clean up before starting to handle cases where a previous run crashed hard (e.g. SIGKILL)
    cleanup_dp_history()
    main()
