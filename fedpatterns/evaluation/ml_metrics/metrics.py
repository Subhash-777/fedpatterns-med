"""
fedpatterns/evaluation/ml_metrics/metrics.py

ML evaluation metrics designed to handle both binary classification
(PneumoniaMNIST, M1) and multi-label classification (NIH ChestX-ray14, M5+).

No binary-only hardcoding: all functions accept a `task` argument.
"""

from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


TaskType = Literal["binary_classification", "multilabel_classification", "multiclass_classification"]


def compute_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: TaskType = "binary_classification",
) -> float:
    """
    Compute accuracy.

    Args:
        y_true: Ground-truth labels. Shape (N,) for binary/multiclass, (N, C) for multilabel.
        y_pred: Predicted labels (thresholded). Same shape as y_true.
        task:   Task type.

    Returns:
        Accuracy score in [0, 1].
    """
    if task == "multilabel_classification":
        # Exact-match (subset) accuracy for multi-label
        return accuracy_score(y_true, y_pred)
    return float(accuracy_score(y_true.ravel(), y_pred.ravel()))


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: TaskType = "binary_classification",
) -> float:
    """
    Compute macro-averaged F1 score.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels (thresholded).
        task:   Task type.

    Returns:
        F1 score in [0, 1].
    """
    average = "macro" if task == "multilabel_classification" else "binary"
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task: TaskType = "binary_classification",
) -> float:
    """
    Compute ROC-AUC.

    Args:
        y_true:  Ground-truth labels.
        y_score: Predicted probabilities/logits (before thresholding).
        task:    Task type.

    Returns:
        AUC score in [0, 1], or 0.0 if only one class present.
    """
    try:
        if task == "multilabel_classification":
            return float(roc_auc_score(y_true, y_score, average="macro", multi_class="ovr"))
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        # Only one class present in y_true (can happen in small partitions)
        return 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    task: TaskType = "binary_classification",
) -> dict[str, float]:
    """
    Compute all ML metrics in one call.

    Returns:
        Dict with keys: accuracy, f1, auc.
    """
    return {
        "accuracy": compute_accuracy(y_true, y_pred, task=task),
        "f1": compute_f1(y_true, y_pred, task=task),
        "auc": compute_auc(y_true, y_score, task=task),
    }
