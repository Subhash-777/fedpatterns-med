"""
fedpatterns/core/app/run_patternized.py

Entry point for the M2 Patternized Architecture baseline (B4).

Responsibilities:
  1. Resolve PROJECT_ROOT via pathlib (directory-agnostic — safe to run from anywhere).
  2. Set up MLflow experiment and open the parent run.
  3. Pass run_id and tracking URI to the flwr subprocess via env vars.
  4. After simulation: parse radon cc -j (JSON) and log cyclomatic_complexity.
  5. Save per-round metrics summary to output/logs/patternized_results.csv.
"""

import json
import os
import pathlib
import subprocess
import sys

import mlflow
import pandas as pd

# ── Path resolution ────────────────────────────────────────────────────────────
# fedpatterns/core/app/run_patternized.py → parents[3] = project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))  # ensure project packages are importable

from fedpatterns.utils.mlflow_utils import setup_mlflow  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "M2-B4-PatternizedFedAvg"
# MLflow 3.14.0 removed file-store support — use SQLite backend
TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlflow.db"
OUTPUT_CSV = PROJECT_ROOT / "output" / "logs" / "patternized_results.csv"


def run_simulation(run_id: str) -> None:
    """Invoke flwr run with the MLflow run_id injected via env vars."""
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

    print(f"[run_patternized] Starting flwr simulation (PROJECT_ROOT={PROJECT_ROOT})")
    flwr_bin = os.path.join(os.path.dirname(sys.executable), "flwr")
    result = subprocess.run(
        [
            flwr_bin, "run", ".", "local", "--stream",
            "--run-config", run_config_str,
        ],
        env=env,
        cwd=PROJECT_ROOT,
        check=True,
    )
    return result


def log_cyclomatic_complexity(run_id: str) -> float:
    """
    Run radon cc with JSON output, compute average cyclomatic complexity,
    and log to MLflow.
    """
    core_app_dir = str(PROJECT_ROOT / "fedpatterns" / "core" / "app")
    mediator_dir = str(PROJECT_ROOT / "fedpatterns" / "patterns" / "mediator")
    strategy_dir = str(PROJECT_ROOT / "fedpatterns" / "patterns" / "strategy")
    
    radon_bin = os.path.join(os.path.dirname(sys.executable), "radon")
    result = subprocess.run(
        [radon_bin, "cc", core_app_dir, mediator_dir, strategy_dir, "-a", "-j"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0 or not result.stdout.strip():
        print(f"[run_patternized] Warning: radon returned no output. stderr: {result.stderr}")
        return 0.0

    cc_data = json.loads(result.stdout)
    all_complexities = [
        block["complexity"]
        for blocks in cc_data.values()
        for block in blocks
    ]
    avg_cc = sum(all_complexities) / len(all_complexities) if all_complexities else 0.0
    mlflow.log_metric("cyclomatic_complexity", avg_cc, run_id=run_id)
    print(f"[run_patternized] Cyclomatic complexity (avg): {avg_cc:.2f} — logged to MLflow run {run_id}")
    return avg_cc


def save_run_summary(run_id: str) -> None:
    """Fetch per-round metric history from MLflow and save to CSV."""
    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    run = client.get_run(run_id)
    
    # Collect all metric history
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
        print("[run_patternized] No metrics found to save.")
        return
        
    df = pd.DataFrame(records)
    # Pivot so each row is a step (round), and columns are metric names
    df_pivot = df.pivot_table(
        index=["run_id", "step"], 
        columns="metric", 
        values="value"
    ).reset_index()
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_pivot.to_csv(OUTPUT_CSV, index=False)
    print(f"[run_patternized] Results saved to {OUTPUT_CSV} ({len(df_pivot)} rounds)")


def main() -> None:
    # Read configuration from environment variables for M6 sweeps
    experiment_name = os.environ.get("FED_EXPERIMENT_NAME", EXPERIMENT_NAME)
    alpha = os.environ.get("FED_ALPHA", "0.5")
    noise_multiplier = os.environ.get("FED_NOISE_MULTIPLIER", "1.0")
    enable_dp = os.environ.get("FED_ENABLE_DP", "True")
    enable_straggler = os.environ.get("FED_ENABLE_STRAGGLER", "True")

    experiment_id = setup_mlflow(experiment_name=experiment_name, tracking_uri=TRACKING_URI)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"[run_patternized] MLflow run started: {run_id}")

        # Tag and log params for the run
        config_meta = {
            "milestone": "M6",
            "baseline_id": "B4",
            "dataset": os.environ.get("FED_DATASET", "PneumoniaMNIST"),
            "strategy": "FedAvg",
            "seed": "42",
            "alpha": alpha,
            "noise_multiplier": noise_multiplier,
            "enable_dp": enable_dp,
            "enable_straggler": enable_straggler,
        }
        mlflow.set_tags(config_meta)
        mlflow.log_params(config_meta)

        # Run the Flower simulation
        run_simulation(run_id=run_id)

        # Log architecture quality metric
        log_cyclomatic_complexity(run_id=run_id)

        # Save summary CSV
        save_run_summary(run_id=run_id)

    print(f"[run_patternized] Done. View results: mlflow ui --backend-store-uri {TRACKING_URI}")


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
    cleanup_dp_history()
    main()
