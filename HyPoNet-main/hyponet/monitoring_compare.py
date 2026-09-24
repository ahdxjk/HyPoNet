"""Matched hard/soft chronological warning comparison.

Thresholds are selected separately for each model and run on validation data.
The paired patient-cluster bootstrap resamples only test patients, holding the
fitted models and validation thresholds fixed. If patient_id is absent, one
case_id is assumed to represent one patient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from hyponet.monitoring import (
    DEFAULT_BUDGETS,
    DECISION_SECONDS,
    HORIZON_SECONDS,
    MonitoringSplit,
    select_validation_thresholds,
)


_KEY = ["run", "split", "case_id", "time_s"]
_IDENTITY = _KEY + ["patient_id", "ongoing", "input_complete", "future_complete"]
_MEASURES = (
    "timely_sensitivity_pct",
    "coverage_pct",
    "false_alarms_per_hour",
    "lead_median_s",
)


def _required(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    if missing := columns - set(frame):
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _ids(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{name} contains missing IDs")
    result = values.astype(str).str.strip()
    if result.eq("").any() or result.str.lower().isin({"nan", "none"}).any():
        raise ValueError(f"{name} contains empty IDs")
    return result


def _boolean(values: pd.Series, name: str) -> pd.Series:
    mapping = {"true": True, "1": True, "yes": True,
               "false": False, "0": False, "no": False}
    parsed = values.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        raise ValueError(f"{name} must contain explicit true/false values")
    return parsed.astype(bool)


def _run_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if "run" in frame:
        return _ids(frame["run"], f"{name} run")
    if "fold" in frame:
        return _ids(frame["fold"], f"{name} fold")
    return pd.Series("1", index=frame.index, dtype=str)


def _prepare_decisions(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    _required(frame, {"split", "case_id", "time_s", "score", "ongoing",
                      "input_complete", "future_complete"}, name)
    result = frame.copy()
    result["run"] = _run_column(result, name)
    result["split"] = result["split"].astype(str).str.strip().str.lower()
    if not result["split"].isin({"val", "test"}).all():
        raise ValueError(f"{name} splits must be val/test")
    result["case_id"] = _ids(result["case_id"], f"{name} case_id")
    result["patient_id"] = (
        _ids(result["patient_id"], f"{name} patient_id")
        if "patient_id" in result else result["case_id"]
    )
    result["time_s"] = pd.to_numeric(result["time_s"], errors="coerce")
    if not np.isfinite(result["time_s"].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} has nonfinite decision times")
    for flag in ("ongoing", "input_complete", "future_complete"):
        result[flag] = _boolean(result[flag], f"{name} {flag}")
    if result.duplicated(_KEY).any():
        raise ValueError(f"{name} has duplicate run/split/case/time decisions")
    if result.groupby(["run", "case_id"])["patient_id"].nunique().gt(1).any():
        raise ValueError(f"{name} assigns one case to multiple patients")
    for run, part in result.groupby("run"):
        if set(part["split"]) != {"val", "test"}:
            raise ValueError(f"{name} run {run} requires val and test decisions")
        val = set(part.loc[part["split"].eq("val"), "patient_id"])
        test = set(part.loc[part["split"].eq("test"), "patient_id"])
        if val & test:
            raise ValueError(f"{name} run {run} has patient overlap between val and test")
    return result.sort_values(_KEY, kind="stable").reset_index(drop=True)


def _prepare_events(frame: pd.DataFrame, runs: list[str], decisions: pd.DataFrame,
                    name: str) -> pd.DataFrame:
    _required(frame, {"split", "case_id", "onset_s"}, name)
    result = frame.copy()
    if "run" not in result and "fold" not in result and len(runs) != 1:
        raise ValueError(f"{name} needs run/fold when comparing multiple runs")
    result["run"] = _run_column(result, name) if ("run" in result or "fold" in result) else runs[0]
    result["split"] = result["split"].astype(str).str.strip().str.lower()
    if not result["split"].isin({"val", "test"}).all() or not set(result["run"]).issubset(runs):
        raise ValueError(f"{name} contains an unexpected split or run")
    result["case_id"] = _ids(result["case_id"], f"{name} case_id")
    result["onset_s"] = pd.to_numeric(result["onset_s"], errors="coerce")
    if not np.isfinite(result["onset_s"].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} has nonfinite event onsets")
    result["eligible"] = (
        _boolean(result["eligible"], f"{name} eligible")
        if "eligible" in result else True
    )
    if result.duplicated(["run", "split", "case_id", "onset_s"]).any():
        raise ValueError(f"{name} has duplicate events")
    case_patients = decisions[["run", "split", "case_id", "patient_id"]].drop_duplicates()
    lookup = case_patients.set_index(["run", "split", "case_id"])["patient_id"]
    keys = pd.MultiIndex.from_frame(result[["run", "split", "case_id"]])
    inferred = pd.Series(lookup.reindex(keys).to_numpy(), index=result.index)
    if "patient_id" in result:
        result["patient_id"] = _ids(result["patient_id"], f"{name} patient_id")
        known = inferred.notna()
        if not result.loc[known, "patient_id"].eq(inferred.loc[known]).all():
            raise ValueError(f"{name} patient IDs do not match decisions")
    else:
        result["patient_id"] = inferred.fillna(result["case_id"])
    return result.sort_values(["run", "split", "case_id", "onset_s"],
                              kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class _PatientContributions:
    patients: np.ndarray
    event_count: np.ndarray
    detected_count: np.ndarray
    coverage_sum: np.ndarray
    false_count: np.ndarray
    hours: np.ndarray
    event_patient_codes: np.ndarray
    event_leads: np.ndarray


def _contributions(split: MonitoringSplit, threshold: float,
                   decision_patients: np.ndarray, event_patients: np.ndarray,
                   patients: np.ndarray) -> _PatientContributions:
    patient_codes = np.searchsorted(patients, decision_patients)
    event_codes = np.searchsorted(patients, event_patients)
    n_patients = len(patients)
    above = split.valid & (split.scores >= threshold)
    alarms = split._alarm_indices(above)
    alarm_flags = np.zeros(len(split.times), dtype=np.int64)
    alarm_flags[alarms] = 1
    alarm_prefix = np.r_[0, np.cumsum(alarm_flags)]
    detected = split.eligible_events & (
        alarm_prefix[split.timely_end] - alarm_prefix[split.timely_start] > 0
    )
    above_prefix = np.r_[0, np.cumsum(above, dtype=np.int64)]
    event_coverage = np.zeros(len(split.event_onsets), dtype=float)
    eligible = split.eligible_events
    event_coverage[eligible] = (
        above_prefix[split.coverage_end[eligible]]
        - above_prefix[split.coverage_start[eligible]]
    ) / split.coverage_denominator[eligible]

    leads = np.full(len(split.event_onsets), np.nan, dtype=float)
    selected = np.flatnonzero(detected)
    if selected.size:
        first_positions = np.searchsorted(alarms, split.timely_start[selected], side="left")
        leads[selected] = split.event_onsets[selected] - split.times[alarms[first_positions]]

    false = split.next_onset_gap[alarms] > HORIZON_SECONDS
    return _PatientContributions(
        patients=patients,
        event_count=np.bincount(event_codes[eligible], minlength=n_patients),
        detected_count=np.bincount(event_codes[detected], minlength=n_patients),
        coverage_sum=np.bincount(event_codes, weights=event_coverage, minlength=n_patients),
        false_count=np.bincount(patient_codes[alarms[false]], minlength=n_patients),
        hours=np.bincount(patient_codes, weights=split.valid.astype(float),
                          minlength=n_patients) * DECISION_SECONDS / 3600.0,
        event_patient_codes=event_codes,
        event_leads=leads,
    )


def _point_metrics(contribution: _PatientContributions) -> dict[str, float]:
    events = contribution.event_count.sum()
    hours = contribution.hours.sum()
    finite_leads = contribution.event_leads[np.isfinite(contribution.event_leads)]
    return {
        "timely_sensitivity_pct": 100.0 * contribution.detected_count.sum() / events,
        "coverage_pct": 100.0 * contribution.coverage_sum.sum() / events,
        "false_alarms_per_hour": contribution.false_count.sum() / hours,
        "lead_median_s": float(np.median(finite_leads)) if finite_leads.size else float("nan"),
    }


def _bootstrap_metric(contribution: _PatientContributions,
                      weights: np.ndarray, measure: str) -> np.ndarray:
    if measure == "lead_median_s":
        result = np.full(len(weights), np.nan, dtype=float)
        finite = np.isfinite(contribution.event_leads)
        leads = contribution.event_leads[finite]
        codes = contribution.event_patient_codes[finite]
        for i, sampled in enumerate(weights):
            multiplicity = sampled[codes]
            if multiplicity.any():
                result[i] = float(np.median(np.repeat(leads, multiplicity)))
        return result
    if measure == "false_alarms_per_hour":
        numerator = weights @ contribution.false_count
        denominator = weights @ contribution.hours
        scale = 1.0
    elif measure == "timely_sensitivity_pct":
        numerator = weights @ contribution.detected_count
        denominator = weights @ contribution.event_count
        scale = 100.0
    else:
        numerator = weights @ contribution.coverage_sum
        denominator = weights @ contribution.event_count
        scale = 100.0
    result = np.full(len(weights), np.nan, dtype=float)
    np.divide(scale * numerator, denominator, out=result, where=denominator > 0)
    return result


def _bootstrap_intervals(hard: _PatientContributions,
                         soft: _PatientContributions, weights: np.ndarray) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for measure in _MEASURES:
        deltas = _bootstrap_metric(soft, weights, measure) - _bootstrap_metric(hard, weights, measure)
        finite = deltas[np.isfinite(deltas)]
        result[f"difference_{measure}_bootstrap_valid"] = int(finite.size)
        if finite.size:
            result[f"difference_{measure}_ci_low"] = float(np.quantile(finite, 0.025))
            result[f"difference_{measure}_ci_high"] = float(np.quantile(finite, 0.975))
        else:
            result[f"difference_{measure}_ci_low"] = float("nan")
            result[f"difference_{measure}_ci_high"] = float("nan")
    return result


def compare_monitoring(
    hard_decisions: pd.DataFrame,
    soft_decisions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    soft_events: pd.DataFrame | None = None,
    budgets: Iterable[float] = DEFAULT_BUDGETS,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return matched test operating points and soft-minus-hard paired CIs.

    Hard and soft models must have identical decision keys, patient mapping,
    evaluability flags, and event annotations within each run. Thresholds are
    selected independently using only the corresponding validation scores;
    patient resampling affects test metrics only and never refits thresholds.
    """
    budgets = tuple(float(value) for value in budgets)
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("Budgets must be a nonempty list without duplicates")
    if not all(np.isfinite(value) and value >= 0 for value in budgets):
        raise ValueError("Budgets must be nonnegative finite values")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be nonnegative")
    hard = _prepare_decisions(hard_decisions, "hard decisions")
    soft = _prepare_decisions(soft_decisions, "soft decisions")
    if not hard[_IDENTITY].equals(soft[_IDENTITY]):
        raise ValueError("Hard and soft decisions, patients, or evaluability flags do not match")
    runs = sorted(hard["run"].unique())
    common_events = _prepare_events(events, runs, hard, "events")
    if soft_events is not None:
        comparison_events = _prepare_events(soft_events, runs, hard, "soft events")
        columns = ["run", "split", "case_id", "onset_s", "patient_id", "eligible"]
        if not common_events[columns].equals(comparison_events[columns]):
            raise ValueError("Hard and soft event annotations do not match")

    rng = np.random.default_rng(seed)
    rows = []
    for run in runs:
        hard_run = hard.loc[hard["run"].eq(run)]
        soft_run = soft.loc[soft["run"].eq(run)]
        event_run = common_events.loc[common_events["run"].eq(run)]
        h_val, s_val, h_test, s_test = (
            MonitoringSplit.from_frames(model.loc[model["split"].eq(split)],
                                        event_run.loc[event_run["split"].eq(split)], split)
            for model, split in ((hard_run, "val"), (soft_run, "val"),
                                 (hard_run, "test"), (soft_run, "test"))
        )
        if not np.array_equal(h_test.eligible_events, s_test.eligible_events):
            raise ValueError(f"Run {run} has different hard/soft eligible events")
        if h_test.eligible_event_count == 0:
            raise ValueError(f"Run {run} has no eligible test events")
        hard_thresholds = select_validation_thresholds(h_val, budgets)
        soft_thresholds = select_validation_thresholds(s_val, budgets)

        test_decisions = hard_run.loc[hard_run["split"].eq("test")].sort_values(
            ["case_id", "time_s"], kind="stable")
        test_events = event_run.loc[event_run["split"].eq("test")].sort_values(
            ["case_id", "onset_s"], kind="stable")
        patients = np.sort(test_decisions["patient_id"].unique())
        decision_patients = test_decisions["patient_id"].to_numpy(dtype=str)
        event_patients = test_events["patient_id"].to_numpy(dtype=object)
        # Ineligible events without any decisions do not enter the bootstrap.
        unknown = ~np.isin(event_patients, patients)
        if (unknown & h_test.eligible_events).any():
            raise ValueError(f"Run {run} has an eligible event without a test patient")
        if unknown.any():
            event_patients = event_patients.copy()
            event_patients[unknown] = patients[0]
        weights = rng.multinomial(
            len(patients), np.full(len(patients), 1.0 / len(patients)),
            size=n_bootstrap,
        )

        for budget in budgets:
            hard_threshold = hard_thresholds[budget]
            soft_threshold = soft_thresholds[budget]
            hard_val_metrics = h_val.summarize(hard_threshold)
            soft_val_metrics = s_val.summarize(soft_threshold)
            hard_test_metrics = h_test.summarize(hard_threshold)
            soft_test_metrics = s_test.summarize(soft_threshold)
            hard_parts = _contributions(h_test, hard_threshold, decision_patients,
                                        event_patients, patients)
            soft_parts = _contributions(s_test, soft_threshold, decision_patients,
                                        event_patients, patients)
            hard_points = _point_metrics(hard_parts)
            soft_points = _point_metrics(soft_parts)
            for model_points, core_points in ((hard_points, hard_test_metrics),
                                               (soft_points, soft_test_metrics)):
                for measure in _MEASURES:
                    if not np.isclose(model_points[measure], core_points[measure], equal_nan=True):
                        raise AssertionError(f"Patient aggregation disagrees for {measure}")

            row: dict[str, float | int | str] = {
                "run": run,
                "validation_budget_per_hour": budget,
                "hard_threshold": hard_threshold,
                "soft_threshold": soft_threshold,
                "hard_validation_timely_sensitivity_pct": hard_val_metrics["timely_sensitivity_pct"],
                "soft_validation_timely_sensitivity_pct": soft_val_metrics["timely_sensitivity_pct"],
                "hard_validation_false_alarms_per_hour": hard_val_metrics["false_alarms_per_hour"],
                "soft_validation_false_alarms_per_hour": soft_val_metrics["false_alarms_per_hour"],
                "test_evaluable_hours": h_test.evaluable_hours,
                "test_eligible_events": h_test.eligible_event_count,
                "test_patients": len(patients),
                "n_bootstrap": n_bootstrap,
            }
            for measure in _MEASURES:
                row[f"hard_test_{measure}"] = hard_points[measure]
                row[f"soft_test_{measure}"] = soft_points[measure]
                row[f"difference_{measure}"] = soft_points[measure] - hard_points[measure]
            for model_name, metrics in (("hard", hard_test_metrics), ("soft", soft_test_metrics)):
                row[f"{model_name}_test_lead_q1_s"] = metrics["lead_q1_s"]
                row[f"{model_name}_test_lead_q3_s"] = metrics["lead_q3_s"]
                row[f"{model_name}_test_detected_events"] = metrics["detected_events"]
                row[f"{model_name}_test_false_alarms"] = metrics["false_alarms"]
            row.update(_bootstrap_intervals(hard_parts, soft_parts, weights))
            rows.append(row)
    return pd.DataFrame(rows)
