"""Evaluation metrics for binary POH prediction and probability fitting."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= low) & (y_prob < high if high < 1 else y_prob <= high)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def binary_probability_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    soft_target: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics = {
        "accuracy": float((y_pred == y_true).mean()),
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier_hard": float(np.mean((y_prob - y_true) ** 2)),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    if soft_target is not None:
        soft = np.asarray(soft_target).reshape(-1)
        eps = 1e-8
        metrics["mse_soft"] = float(np.mean((y_prob - soft) ** 2))
        metrics["mae_soft"] = float(np.mean(np.abs(y_prob - soft)))
        metrics["kl_soft"] = float(
            np.mean(
                soft * (np.log(soft + eps) - np.log(y_prob + eps))
                + (1.0 - soft) * (np.log(1.0 - soft + eps) - np.log(1.0 - y_prob + eps))
            )
        )
    return metrics
