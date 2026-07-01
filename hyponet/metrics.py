"""Evaluation metrics for probabilistic POH prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(float).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def binary_probability_metrics(
    y_true,
    y_prob,
    soft_targets=None,
    threshold: float = 0.5,
    n_bins: int = 10,
):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)
    pred = (y_prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    out = {
        "accuracy": float((pred == y_true).mean()),
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier_hard": float(np.mean((y_prob - y_true) ** 2)),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
    }

    if soft_targets is not None:
        soft = np.asarray(soft_targets).astype(float).reshape(-1)
        eps = 1e-8
        p = np.clip(y_prob, eps, 1.0 - eps)
        s = np.clip(soft, eps, 1.0 - eps)
        out.update(
            {
                "mse_soft": float(np.mean((p - s) ** 2)),
                "mae_soft": float(np.mean(np.abs(p - s))),
                "kl_soft": float(
                    np.mean(s * (np.log(s) - np.log(p)) + (1.0 - s) * (np.log(1.0 - s) - np.log(1.0 - p)))
                ),
            }
        )
    return out
