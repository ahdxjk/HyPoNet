"""Matched hard/soft score-dynamics analysis on chronological test scores.

Stable periods are supplied by the caller. The manuscript does not define a
stable-period detector or the precise endpoints for its pre-onset rise, so this
module never infers stability and exposes the response window explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class ValidationTransform:
    sorted_scores: np.ndarray
    mean: float
    std: float


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")


def _boolean_column(values: pd.Series, name: str) -> np.ndarray:
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError(f"{name} requires explicit true/false marks")
    return normalized.map(mapping).to_numpy(dtype=bool)


def prepare_scores(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate val/test score rows on the 20-second decision grid."""

    _require_columns(frame, {"split", "case_id", "time_s", "score"}, name)
    scores = frame.copy()
    if "run" not in scores:
        scores["run"] = scores["fold"] if "fold" in scores else "1"
    scores["run"] = scores["run"].astype(str)
    scores["split"] = scores["split"].astype(str).str.strip().str.lower()
    if not {"val", "test"}.issubset(set(scores["split"])):
        raise ValueError(f"{name} must contain both val and test rows")
    if not scores["split"].isin({"val", "test"}).all():
        raise ValueError(f"{name} contains a split other than val/test")
    scores["case_id"] = scores["case_id"].astype(str)
    if scores["case_id"].str.strip().eq("").any() or scores["case_id"].eq("nan").any():
        raise ValueError(f"{name} contains missing case IDs")
    if "patient_id" not in scores:
        scores["patient_id"] = scores["case_id"]
    else:
        if scores["patient_id"].isna().any() or scores["patient_id"].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} contains missing patient IDs")
        scores["patient_id"] = scores["patient_id"].astype(str)
    scores["time_s"] = pd.to_numeric(scores["time_s"], errors="raise")
    if not np.isfinite(scores["time_s"].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains nonfinite times")
    eligible = np.ones(len(scores), dtype=bool)
    for flag, required_value in (("ongoing", False), ("input_complete", True),
                                 ("future_complete", True)):
        if flag in scores:
            eligible &= _boolean_column(scores[flag], flag) == required_value
    scores = scores.loc[eligible].copy()
    scores["score"] = pd.to_numeric(scores["score"], errors="raise")
    if not np.isfinite(scores["score"].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains nonfinite scores on evaluable rows")
    if not scores["score"].between(0.0, 1.0).all():
        raise ValueError(f"{name} must contain sigmoid scores in [0, 1]")
    if scores.duplicated(["run", "split", "case_id", "time_s"]).any():
        raise ValueError(f"{name} contains duplicate case/time decisions")
    if scores.groupby(["run", "case_id"])["patient_id"].nunique().gt(1).any():
        raise ValueError(f"{name} assigns one case to multiple patients")
    for _, run_scores in scores.groupby("run"):
        val_patients = set(run_scores.loc[run_scores["split"].eq("val"), "patient_id"])
        test_patients = set(run_scores.loc[run_scores["split"].eq("test"), "patient_id"])
        if val_patients & test_patients:
            raise ValueError(f"{name} has patient overlap between val and test")
    if "stable" not in scores:
        raise ValueError(f"{name} requires an explicit stable column for test rows")
    test_mask = scores["split"].eq("test")
    scores.loc[test_mask, "stable"] = _boolean_column(scores.loc[test_mask, "stable"], "stable")
    return scores.sort_values(["run", "split", "case_id", "time_s"]).reset_index(drop=True)


def fit_validation_transform(scores: np.ndarray) -> ValidationTransform:
    """Fit an empirical percentile map and z-score using validation scores."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Validation scores must be a nonempty finite vector")
    std = float(np.std(values, ddof=0))
    if std == 0:
        raise ValueError("Validation scores have zero variance; z-score is undefined")
    return ValidationTransform(np.sort(values), float(np.mean(values)), std)


def transform_scores(scores: np.ndarray, fitted: ValidationTransform) -> dict[str, np.ndarray]:
    """Apply fixed validation transformations to test scores."""

    values = np.asarray(scores, dtype=float)
    return {
        "raw": values.copy(),
        "percentile": np.searchsorted(fitted.sorted_scores, values, side="right") / fitted.sorted_scores.size,
        "standardized": (values - fitted.mean) / fitted.std,
    }


def _transform_each_run(
    aligned: pd.DataFrame,
    source: pd.DataFrame,
    score_column: str,
) -> dict[str, np.ndarray]:
    """Fit each run on its own validation set, then transform its test rows."""

    transformed = {
        scale: np.empty(len(aligned), dtype=float)
        for scale in ("raw", "percentile", "standardized")
    }
    for run, indices in aligned.groupby("run", sort=False).indices.items():
        indices = np.asarray(indices)
        validation = source.loc[
            source["run"].eq(run) & source["split"].eq("val"), "score"
        ].to_numpy(dtype=float)
        fitted = fit_validation_transform(validation)
        scales = transform_scores(aligned.loc[indices, score_column].to_numpy(dtype=float), fitted)
        for scale, values in scales.items():
            transformed[scale][indices] = values
    return transformed


def _aligned_test_scores(hard: pd.DataFrame, soft: pd.DataFrame) -> pd.DataFrame:
    validation_identity = ["run", "case_id", "patient_id", "time_s"]
    hard_val = hard.loc[hard["split"].eq("val"), validation_identity].reset_index(drop=True)
    soft_val = soft.loc[soft["split"].eq("val"), validation_identity].reset_index(drop=True)
    if not hard_val.equals(soft_val):
        raise ValueError("Hard and soft validation decision rows do not match")
    hard_test = hard.loc[hard["split"].eq("test")].reset_index(drop=True)
    soft_test = soft.loc[soft["split"].eq("test")].reset_index(drop=True)
    identity = ["run", "case_id", "patient_id", "time_s", "stable"]
    if not hard_test[identity].equals(soft_test[identity]):
        raise ValueError("Hard and soft test decisions, patients, or stable marks do not match")
    aligned = hard_test[identity].copy()
    aligned["hard_score"] = hard_test["score"].to_numpy(dtype=float)
    aligned["soft_score"] = soft_test["score"].to_numpy(dtype=float)
    return aligned


def _test_events(events: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    _require_columns(events, {"split", "case_id", "onset_s"}, "events")
    selected = events.loc[events["split"].astype(str).str.lower().eq("test")].copy()
    if selected.empty:
        raise ValueError("No test events supplied")
    selected["case_id"] = selected["case_id"].astype(str)
    if "run" not in selected and "fold" in selected:
        selected["run"] = selected["fold"]
    if "run" not in selected:
        run_by_case = aligned[["run", "case_id"]].drop_duplicates()
        if run_by_case.duplicated("case_id").any():
            raise ValueError("Events need run/fold when a case appears in multiple test runs")
        selected["run"] = selected["case_id"].map(run_by_case.set_index("case_id")["run"])
        missing_run = selected["run"].isna()
        if missing_run.any():
            known_runs = aligned["run"].unique()
            if len(known_runs) != 1:
                raise ValueError("Events without test decisions need run/fold in multi-run inputs")
            selected.loc[missing_run, "run"] = known_runs[0]
    selected["run"] = selected["run"].astype(str)
    selected["onset_s"] = pd.to_numeric(selected["onset_s"], errors="raise")
    if not np.isfinite(selected["onset_s"].to_numpy(dtype=float)).all():
        raise ValueError("Event onset times must be finite")
    if selected.duplicated(["run", "case_id", "onset_s"]).any():
        raise ValueError("Duplicate test events")
    case_patients = aligned[["run", "case_id", "patient_id"]].drop_duplicates(["run", "case_id"])
    patient_lookup = case_patients.set_index(["run", "case_id"])["patient_id"]
    event_keys = pd.MultiIndex.from_frame(selected[["run", "case_id"]])
    inferred_patients = pd.Series(patient_lookup.reindex(event_keys).to_numpy(), index=selected.index)
    if "patient_id" in selected:
        known = inferred_patients.notna()
        if not selected.loc[known, "patient_id"].astype(str).eq(inferred_patients.loc[known]).all():
            raise ValueError("Event patient IDs do not match test scores")
        selected["patient_id"] = selected["patient_id"].astype(str)
    else:
        selected["patient_id"] = inferred_patients.fillna(selected["case_id"])
    selected["eligible"] = (
        _boolean_column(selected["eligible"], "eligible")
        if "eligible" in selected else True
    )
    return selected.sort_values(["run", "case_id", "onset_s"]).reset_index(drop=True)


def _case_rolling_sd(
    aligned: pd.DataFrame,
    hard_scores: np.ndarray,
    soft_scores: np.ndarray,
    *,
    step_seconds: float,
    window_predictions: int,
) -> pd.DataFrame:
    rows = []
    for (run, case_id), indices in aligned.groupby(["run", "case_id"], sort=True).indices.items():
        indices = np.asarray(indices)
        times = aligned.loc[indices, "time_s"].to_numpy(dtype=float)
        stable = aligned.loc[indices, "stable"].to_numpy(dtype=bool)
        hard_windows = []
        soft_windows = []
        for start in range(len(indices) - window_predictions + 1):
            stop = start + window_predictions
            if not stable[start:stop].all():
                continue
            if not np.allclose(np.diff(times[start:stop]), step_seconds, rtol=0, atol=1e-6):
                continue
            hard_windows.append(float(np.std(hard_scores[indices[start:stop]], ddof=1)))
            soft_windows.append(float(np.std(soft_scores[indices[start:stop]], ddof=1)))
        if hard_windows:
            rows.append({
                "run": run,
                "case_id": case_id,
                "patient_id": aligned.loc[indices[0], "patient_id"],
                "n_windows": len(hard_windows),
                "hard": float(np.mean(hard_windows)),
                "soft": float(np.mean(soft_windows)),
            })
    if not rows:
        raise ValueError("No case has the requested number of consecutive stable test predictions")
    return pd.DataFrame(rows)


def _spearman_time_score(times: np.ndarray, scores: np.ndarray) -> float:
    if np.all(scores == scores[0]):
        return 0.0
    result = float(spearmanr(times, scores).statistic)
    if not np.isfinite(result):
        raise ValueError("Spearman correlation could not be computed")
    return result


def _event_response(
    aligned: pd.DataFrame,
    events: pd.DataFrame,
    hard_scores: np.ndarray,
    soft_scores: np.ndarray,
    *,
    pre_onset_seconds: float,
    min_lead_seconds: float,
) -> pd.DataFrame:
    rows = []
    indices_by_case = aligned.groupby(["run", "case_id"], sort=False).indices
    for event_index, event in events.iterrows():
        case_indices = np.asarray(indices_by_case.get((event["run"], event["case_id"]), []), dtype=int)
        times = aligned.loc[case_indices, "time_s"].to_numpy(dtype=float)
        onset = float(event["onset_s"])
        in_window = (times >= onset - pre_onset_seconds) & (times < onset)
        if min_lead_seconds > 0:
            in_window &= times <= onset - min_lead_seconds
        selected = case_indices[in_window]
        row = {
            "event_index": int(event_index),
            "run": event["run"],
            "case_id": event["case_id"],
            "patient_id": event["patient_id"],
            "onset_s": onset,
            "n_decisions": int(selected.size),
        }
        if not event["eligible"] or selected.size < 2:
            row["status"] = ("marked_ineligible" if not event["eligible"]
                             else "insufficient_pre_onset_decisions")
            row.update({
                "hard_rise": float("nan"), "soft_rise": float("nan"),
                "hard_spearman": float("nan"), "soft_spearman": float("nan"),
            })
            rows.append(row)
            continue
        selected_times = aligned.loc[selected, "time_s"].to_numpy(dtype=float)
        hard = hard_scores[selected]
        soft = soft_scores[selected]
        row.update({
            "status": "analyzed",
            "hard_rise": float(hard[-1] - hard[0]),
            "soft_rise": float(soft[-1] - soft[0]),
            "hard_spearman": _spearman_time_score(selected_times, hard),
            "soft_spearman": _spearman_time_score(selected_times, soft),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_summary(
    values: pd.DataFrame,
    hard_column: str,
    soft_column: str,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    if values.empty:
        return {
            "hard": float("nan"), "soft": float("nan"),
            "difference": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "n_units": 0, "n_patients": 0,
        }
    hard = values[hard_column].to_numpy(dtype=float)
    soft = values[soft_column].to_numpy(dtype=float)
    patients, inverse = np.unique(values["patient_id"].to_numpy(dtype=str), return_inverse=True)
    if not len(patients):
        raise ValueError("No patients contribute to a requested measure")
    differences = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(patients), size=len(patients))
        weights = np.bincount(sample, minlength=len(patients))[inverse]
        differences[index] = np.average(soft - hard, weights=weights)
    low, high = (np.quantile(differences, [0.025, 0.975])
                 if n_bootstrap else (float("nan"), float("nan")))
    return {
        "hard": float(np.mean(hard)),
        "soft": float(np.mean(soft)),
        "difference": float(np.mean(soft - hard)),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_units": len(values),
        "n_patients": len(patients),
    }


def evaluate_score_dynamics(
    hard_scores: pd.DataFrame,
    soft_scores: pd.DataFrame,
    events: pd.DataFrame,
    *,
    step_seconds: float = 20.0,
    window_predictions: int = 10,
    pre_onset_seconds: float = 300.0,
    min_lead_seconds: float = 0.0,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return summaries, case-level rolling SDs, and event-level responses.

    The response window defaults to the five minutes before onset; rise is
    last minus first available pre-onset score. These endpoints are explicit
    implementation choices, not specified in the manuscript. Every supplied
    eligible test event contributes, including events without an alarm.
    """

    if step_seconds <= 0 or window_predictions < 2 or pre_onset_seconds <= 0:
        raise ValueError("Grid step, rolling length, and pre-onset window must be positive")
    if not 0 <= min_lead_seconds < pre_onset_seconds:
        raise ValueError("min_lead_seconds must be inside the pre-onset window")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be nonnegative")

    hard = prepare_scores(hard_scores, "hard scores")
    soft = prepare_scores(soft_scores, "soft scores")
    aligned = _aligned_test_scores(hard, soft)
    selected_events = _test_events(events, aligned)
    hard_scales = _transform_each_run(aligned, hard, "hard_score")
    soft_scales = _transform_each_run(aligned, soft, "soft_score")

    rng = np.random.default_rng(seed)
    summary_rows = []
    case_rows = []
    event_rows = []
    for scale in ("raw", "percentile", "standardized"):
        cases = _case_rolling_sd(
            aligned, hard_scales[scale], soft_scales[scale],
            step_seconds=step_seconds, window_predictions=window_predictions,
        )
        cases.insert(0, "scale", scale)
        case_rows.append(cases)
        summary_rows.append({
            "scale": scale, "measure": "rolling_sd",
            **_paired_summary(cases, "hard", "soft", n_bootstrap=n_bootstrap, rng=rng),
        })

        responses = _event_response(
            aligned, selected_events, hard_scales[scale], soft_scales[scale],
            pre_onset_seconds=pre_onset_seconds, min_lead_seconds=min_lead_seconds,
        )
        responses.insert(0, "scale", scale)
        event_rows.append(responses)
        analyzed = responses.loc[responses["status"].eq("analyzed")]
        event_counts = {
            "n_events_input": len(responses),
            "n_events_excluded": len(responses) - len(analyzed),
        }
        summary_rows.append({
            "scale": scale, "measure": "pre_onset_rise",
            **event_counts,
            **_paired_summary(
                analyzed, "hard_rise", "soft_rise", n_bootstrap=n_bootstrap, rng=rng,
            ),
        })
        if scale == "raw":
            summary_rows.append({
                "scale": scale, "measure": "time_score_spearman",
                **event_counts,
                **_paired_summary(
                    analyzed, "hard_spearman", "soft_spearman", n_bootstrap=n_bootstrap, rng=rng,
                ),
            })

    return (
        pd.DataFrame(summary_rows),
        pd.concat(case_rows, ignore_index=True),
        pd.concat(event_rows, ignore_index=True),
    )
