"""
fedpatterns/utils/mlflow_utils.py

MLflow helpers for FedPatterns-Med.

Key design decisions:
  - setup_mlflow() is idempotent — safe to call multiple times.
  - log_round_metrics() uses mlflow.log_metrics(..., run_id=run_id, step=step)
    with an explicit step argument so per-round charts render correctly in the UI.
  - All logging uses direct run_id injection — no context-manager open/close
    cycle, which avoids RUNNING→FINISHED state flapping across rounds.
"""

import os
from typing import Optional

import mlflow

# Standard tag schema shared across all baselines
TAG_SCHEMA = ["milestone", "baseline_id", "dataset", "strategy", "seed"]


def setup_mlflow(
    experiment_name: str,
    tracking_uri: str = "./mlruns",
) -> str:
    """
    Idempotently set up an MLflow experiment.

    Args:
        experiment_name: Human-readable experiment name (e.g. 'M1-B1-VanillaFedAvg').
        tracking_uri:    Local path or remote URI for the MLflow tracking store.

    Returns:
        experiment_id string.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def log_round_metrics(
    metrics: dict[str, float],
    step: int,
    run_id: Optional[str] = None,
) -> None:
    """
    Log per-round metrics directly to a run, without opening/closing contexts.

    Uses explicit step= so per-round charts render correctly in the MLflow UI.
    Falls back to MLFLOW_RUN_ID env var if run_id is not provided.

    Args:
        metrics: Dict of metric name → value.
        step:    Round number (used as x-axis in MLflow charts).
        run_id:  Optional explicit run ID. Falls back to MLFLOW_RUN_ID env var.
    """
    resolved_run_id = run_id or os.environ.get("MLFLOW_RUN_ID")
    if resolved_run_id is None:
        raise RuntimeError(
            "log_round_metrics: no run_id provided and MLFLOW_RUN_ID env var is not set."
        )
    mlflow.log_metrics(metrics, run_id=resolved_run_id, step=step)


def set_run_tags(tags: dict[str, str], run_id: Optional[str] = None) -> None:
    """
    Set standard tags on a run.

    Args:
        tags:   Dict of tag key → value.
        run_id: Optional explicit run ID. Falls back to MLFLOW_RUN_ID env var.
    """
    resolved_run_id = run_id or os.environ.get("MLFLOW_RUN_ID")
    if resolved_run_id is None:
        raise RuntimeError("set_run_tags: no run_id provided and MLFLOW_RUN_ID env var is not set.")
    with mlflow.start_run(run_id=resolved_run_id):
        mlflow.set_tags(tags)
