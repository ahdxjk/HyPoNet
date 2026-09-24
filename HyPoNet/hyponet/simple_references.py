"""MAP/trend logistic references for the matched five-minute experiment.

The reference predictors use only two observed, bounded pressure cues from a
preceding 60-second ABP input. Targets are read from the existing *training*
fold; future events are never model inputs. The manuscript does not specify
the logistic optimizer or regularization, so this implementation uses a
deterministic Bernoulli objective with an explicit, configurable L2 penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .monitoring_data import DECISION_COLUMNS, EVENT_COLUMNS, iter_monitoring_windows
from .preprocessing.segmentation import (
    SegmentationConfig,
    calculate_map_from_waveform,
    clean_art_pressure,
    detect_hypotension_intervals,
)


FEATURE_SETS = {
    "map": (0,),
    "trend": (1,),
    "map_trend": (0, 1),
}
SUPERVISION_LABELS = {"hard": "HARD_LABELS", "soft": "LABELS_CONCAVE"}


def _fast_trend_slope(art: np.ndarray, sample_rate: int) -> float:
    """Match the manuscript's complete 5-s mean and 60-s OLS slope in O(N)."""

    art = np.asarray(art, dtype=np.float64).reshape(-1)
    if sample_rate <= 0 or len(art) < 60 * sample_rate:
        return float("nan")
    width = 5 * sample_rate
    finite = np.isfinite(art)
    values = np.where(finite, art, 0.0)
    sums = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    counts = np.r_[0, np.cumsum(finite, dtype=np.int64)]
    smoothed = (sums[width:] - sums[:-width]) / width
    usable = counts[width:] - counts[:-width] == width
    if usable.sum() < 2:
        return float("nan")
    time_minutes = np.flatnonzero(usable) / (sample_rate * 60.0)
    slope, _ = np.polyfit(time_minutes, smoothed[usable], 1)
    return float(np.clip(slope / 30.0, -1.0, 1.0))


def pressure_cues(art: np.ndarray, sample_rate: int = 100) -> np.ndarray:
    """Return [x_MAP, x_trend] using the same formulas as soft target construction."""

    map_value = calculate_map_from_waveform(art, sample_rate)
    normalized_slope = _fast_trend_slope(art, sample_rate)
    if not np.isfinite(map_value) or not np.isfinite(normalized_slope):
        raise ValueError("A pressure reference input has no finite pulse MAP or 5-s-smoothed trend.")
    return np.array([
        np.clip((105.0 - map_value) / 40.0, 0.0, 1.0),
        (1.0 - normalized_slope) / 2.0,
    ], dtype=np.float64)


def load_training_pressure_cues(
    train_dir: Path,
    *,
    sample_rate: int = 100,
    batch_size: int = 128,
    allowed_case_ids: set[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read only the fold's training HDF5 and validate its case membership."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    paths = {name: Path(train_dir) / f"{name}.h5" for name in ("ART", "STATIC", *SUPERVISION_LABELS.values())}
    with h5py.File(paths["ART"], "r") as art_file, \
         h5py.File(paths["STATIC"], "r") as static_file, \
         h5py.File(paths["HARD_LABELS"], "r") as hard_file, \
         h5py.File(paths["LABELS_CONCAVE"], "r") as soft_file:
        art = art_file["ART"]
        static = np.asarray(static_file["STATIC"][:]).reshape(-1)
        hard = np.asarray(hard_file["HARD_LABELS"][:], dtype=np.float64).reshape(-1)
        soft = np.asarray(soft_file["LABELS_CONCAVE"][:], dtype=np.float64).reshape(-1)
        count = len(art)
        if count == 0 or art.ndim != 2 or art.shape[1] != 60 * sample_rate:
            raise ValueError(f"Training ART must have nonempty shape [N, {60 * sample_rate}].")
        if any(len(array) != count for array in (static, hard, soft)):
            raise ValueError("Training ART, STATIC, HARD_LABELS, and LABELS_CONCAVE lengths differ.")
        if allowed_case_ids is not None:
            seen = {f"{int(case_id):04d}" for case_id in static}
            if outside := seen - allowed_case_ids:
                raise ValueError(f"Training HDF5 contains cases outside the train manifest: {sorted(outside)}")
        if not np.isfinite(hard).all() or not np.isin(hard, [0, 1]).all():
            raise ValueError("Training hard labels must be finite binary values.")
        if not np.isfinite(soft).all() or ((soft < 0) | (soft > 1)).any():
            raise ValueError("Training soft labels must be finite in [0, 1].")

        cues = np.empty((count, 2), dtype=np.float64)
        for start in range(0, count, batch_size):
            batch = art[start : start + batch_size]
            cues[start : start + len(batch)] = [pressure_cues(row, sample_rate) for row in batch]
    return cues, {"hard": hard, "soft": soft}


@dataclass(frozen=True)
class LogisticReference:
    name: str
    supervision: str
    coefficients: np.ndarray  # Intercept followed by selected cue weights.
    l2: float

    def __post_init__(self) -> None:
        if self.name not in FEATURE_SETS or self.supervision not in SUPERVISION_LABELS:
            raise ValueError("Unknown pressure reference or supervision.")
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if coefficients.shape != (1 + len(FEATURE_SETS[self.name]),) or not np.isfinite(coefficients).all():
            raise ValueError("Invalid logistic coefficients.")
        object.__setattr__(self, "coefficients", coefficients)

    def predict(self, cues: np.ndarray) -> np.ndarray:
        cues = np.asarray(cues, dtype=np.float64)
        if cues.ndim != 2 or cues.shape[1] != 2 or not np.isfinite(cues).all():
            raise ValueError("Pressure cues must have finite shape [N, 2].")
        columns = cues[:, FEATURE_SETS[self.name]]
        return expit(self.coefficients[0] + columns @ self.coefficients[1:])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "supervision": self.supervision,
            "features": [
                ("x_map", "x_trend")[index] for index in FEATURE_SETS[self.name]
            ],
            "coefficients": self.coefficients.tolist(),
            "l2": self.l2,
        }


def fit_logistic_reference(
    cues: np.ndarray,
    targets: np.ndarray,
    name: str,
    supervision: str,
    *,
    l2: float = 0.0,
) -> LogisticReference:
    """Fit hard BCE or soft Bernoulli KL (equivalent objective up to a constant)."""

    if name not in FEATURE_SETS or supervision not in SUPERVISION_LABELS:
        raise ValueError("Unknown pressure reference or supervision.")
    cues = np.asarray(cues, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if cues.ndim != 2 or cues.shape[1] != 2 or len(cues) != len(targets) or len(targets) == 0:
        raise ValueError("Cues must have nonempty shape [N, 2] matching targets.")
    if not np.isfinite(cues).all() or not np.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
        raise ValueError("Cues and targets must be finite; targets must lie in [0, 1].")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be nonnegative and finite.")
    x = np.column_stack([np.ones(len(cues)), cues[:, FEATURE_SETS[name]]])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = x @ beta
        # Bernoulli cross-entropy differs from KL(target || prediction) only
        # by target entropy, so their minimizers coincide for soft targets.
        loss = np.mean(np.logaddexp(0.0, logits) - targets * logits)
        loss += 0.5 * l2 * np.dot(beta[1:], beta[1:])
        gradient = x.T @ (expit(logits) - targets) / len(targets)
        gradient[1:] += l2 * beta[1:]
        return float(loss), gradient

    initial_mean = float(np.clip(targets.mean(), 1e-6, 1 - 1e-6))
    initial = np.zeros(x.shape[1], dtype=np.float64)
    initial[0] = np.log(initial_mean / (1 - initial_mean))
    result = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"{name}/{supervision} logistic fit failed: {result.message}")
    return LogisticReference(name, supervision, result.x, float(l2))


def fit_all_references(
    cues: np.ndarray, labels: dict[str, np.ndarray], *, l2: float = 0.0
) -> dict[str, LogisticReference]:
    return {
        f"{name}_{supervision}": fit_logistic_reference(cues, labels[supervision], name, supervision, l2=l2)
        for name in FEATURE_SETS
        for supervision in SUPERVISION_LABELS
    }


def score_case_references(
    art: np.ndarray,
    ecg: np.ndarray,
    pleth: np.ndarray,
    models: dict[str, LogisticReference],
    *,
    split: str,
    case_id: str,
    patient_id: str | None = None,
    sample_rate: int = 100,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Score all references on one common 20-s decision grid and event list."""

    if patient_id is None:
        patient_id = case_id
    clean_art = clean_art_pressure(art)
    intervals = detect_hypotension_intervals(clean_art, SegmentationConfig(sample_rate=sample_rate, horizon_minutes=5))
    events = [{
        "split": split, "case_id": case_id, "patient_id": patient_id,
        "onset_s": item["start_idx"] / sample_rate,
    } for item in intervals]
    decisions = {key: [] for key in models}
    for window in iter_monitoring_windows(clean_art, ecg, pleth, intervals, sample_rate=sample_rate):
        base = {
            "split": split,
            "case_id": case_id,
            "patient_id": patient_id,
            "time_s": window.end_sample / sample_rate,
            "ongoing": window.ongoing,
            "input_complete": window.input_complete,
            "future_complete": window.future_complete,
        }
        cues = pressure_cues(window.signals[0], sample_rate)[None, :] if window.signals is not None else None
        for key, model in models.items():
            decisions[key].append({**base, "score": float(model.predict(cues)[0]) if cues is not None else None})
    return decisions, events


__all__ = [
    "DECISION_COLUMNS", "EVENT_COLUMNS", "FEATURE_SETS", "SUPERVISION_LABELS",
    "LogisticReference", "fit_all_references", "fit_logistic_reference",
    "load_training_pressure_cues", "pressure_cues", "score_case_references",
]
