"""Chronological five-minute POH warning evaluation.

Input decision rows have columns:
  split (val/test), case_id, time_s, score, ongoing,
  input_complete, future_complete.
time_s is the END of the preceding 60-second waveform input, in seconds from
the start of that case's recording. Supply one row every 20 seconds, including
rows later excluded by the three Boolean flags. Missing grid rows are treated
as breaks in the threshold-crossing sequence. Scores must be in [0, 1] for
evaluable rows; excluded rows may have missing scores.

Input event rows have split, case_id, onset_s and optional eligible. They must
list adjudicated episode ONSETS, not every low-MAP sample. Event eligibility
requires the optional eligible flag (default true) and at least one evaluable
decision 60-300 seconds before onset.

An alarm is a transition from below to at/above threshold, with a 300-second
refractory period after accepted alarms. A late alarm (<60 seconds before an
onset) does not count as timely, but is not called false; a false alarm has no
observed onset within the next 300 seconds. A single alarm can warn of multiple
closely spaced eligible onsets. These conventions make underspecified edge
cases explicit without using test data during threshold selection.

The repository's sampled HDF5 folds do not retain case/time/onset metadata or
20-second overlapping predictions. Generate chronological scores and these
annotations from original recordings before calling this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # The exact NumPy fallback needs no extra dependency.
    njit = None


DECISION_SECONDS = 20.0
INPUT_SECONDS = 60.0
HORIZON_SECONDS = 300.0
MIN_LEAD_SECONDS = 60.0
REFRACTORY_SECONDS = 300.0
DEFAULT_BUDGETS = (0.25, 0.5, 1.0, 2.0)


def _boolean_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = frame[name]
    if values.isna().any():
        raise ValueError(f"{name} contains missing values.")
    mapping = {
        "true": True, "1": True, "yes": True,
        "false": False, "0": False, "no": False,
    }
    parsed = values.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        raise ValueError(f"{name} must contain only true/false or 1/0.")
    return parsed.to_numpy(dtype=bool)


def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite numeric values.")
    return values


@dataclass
class MonitoringSplit:
    """Precomputed chronological indices for one validation or test split."""

    split: str
    times: np.ndarray
    scores: np.ndarray
    case_codes: np.ndarray
    consecutive: np.ndarray
    valid: np.ndarray
    next_onset_gap: np.ndarray
    event_onsets: np.ndarray
    eligible_events: np.ndarray
    timely_start: np.ndarray
    timely_end: np.ndarray
    coverage_start: np.ndarray
    coverage_end: np.ndarray
    coverage_denominator: np.ndarray

    @property
    def evaluable_hours(self) -> float:
        return float(self.valid.sum() * DECISION_SECONDS / 3600.0)

    @property
    def eligible_event_count(self) -> int:
        return int(self.eligible_events.sum())

    @classmethod
    def from_frames(
        cls, decisions: pd.DataFrame, events: pd.DataFrame, split: str
    ) -> "MonitoringSplit":
        required_decisions = {
            "case_id", "time_s", "score", "ongoing",
            "input_complete", "future_complete",
        }
        required_events = {"case_id", "onset_s"}
        if missing := required_decisions - set(decisions.columns):
            raise ValueError(f"Decision columns missing: {sorted(missing)}")
        if missing := required_events - set(events.columns):
            raise ValueError(f"Event columns missing: {sorted(missing)}")
        if decisions.empty:
            raise ValueError(f"{split} has no decision rows.")
        if decisions["case_id"].isna().any() or events["case_id"].isna().any():
            raise ValueError("case_id must not be missing.")

        decisions = decisions.copy()
        events = events.copy()
        decisions["case_id"] = decisions["case_id"].astype(str)
        events["case_id"] = events["case_id"].astype(str)
        decisions["time_s"] = _numeric_column(decisions, "time_s")
        events["onset_s"] = _numeric_column(events, "onset_s")
        if (decisions["time_s"] < INPUT_SECONDS).any():
            raise ValueError("Decision time_s must be at least 60 s after recording start.")
        if (events["onset_s"] < 0).any():
            raise ValueError("Event onset_s must be nonnegative.")
        if not np.allclose(
            decisions["time_s"].to_numpy() / DECISION_SECONDS,
            np.round(decisions["time_s"].to_numpy() / DECISION_SECONDS),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("Decision time_s must lie on the 20-s grid.")
        if decisions.duplicated(["case_id", "time_s"]).any():
            raise ValueError(f"{split} contains duplicate case/time decisions.")
        if events.duplicated(["case_id", "onset_s"]).any():
            raise ValueError(f"{split} contains duplicate case/onset events.")

        decisions["_ongoing"] = _boolean_column(decisions, "ongoing")
        decisions["_input_complete"] = _boolean_column(decisions, "input_complete")
        decisions["_future_complete"] = _boolean_column(decisions, "future_complete")
        if "eligible" in events:
            events["_curated_eligible"] = _boolean_column(events, "eligible")
        else:
            events["_curated_eligible"] = True
        decisions = decisions.sort_values(["case_id", "time_s"], kind="stable").reset_index(drop=True)
        events = events.sort_values(["case_id", "onset_s"], kind="stable").reset_index(drop=True)

        times = decisions["time_s"].to_numpy(dtype=float)
        scores = pd.to_numeric(decisions["score"], errors="coerce").to_numpy(dtype=float)
        valid = (
            ~decisions["_ongoing"].to_numpy(dtype=bool)
            & decisions["_input_complete"].to_numpy(dtype=bool)
            & decisions["_future_complete"].to_numpy(dtype=bool)
        )
        if not np.isfinite(scores[valid]).all() or ((scores[valid] < 0) | (scores[valid] > 1)).any():
            raise ValueError(f"{split} evaluable scores must be finite and in [0, 1].")
        if not valid.any():
            raise ValueError(f"{split} has no evaluable decisions.")

        case_ids = decisions["case_id"].to_numpy(dtype=str)
        _, case_codes = np.unique(case_ids, return_inverse=True)
        consecutive = np.r_[
            False,
            (case_codes[1:] == case_codes[:-1])
            & np.isclose(times[1:] - times[:-1], DECISION_SECONDS),
        ]
        next_onset_gap = np.full(len(decisions), np.inf, dtype=float)
        event_onsets = events["onset_s"].to_numpy(dtype=float)
        event_eligible = events["_curated_eligible"].to_numpy(dtype=bool).copy()
        timely_start = np.zeros(len(events), dtype=np.int64)
        timely_end = np.zeros(len(events), dtype=np.int64)
        coverage_start = np.zeros(len(events), dtype=np.int64)
        coverage_end = np.zeros(len(events), dtype=np.int64)

        valid_prefix = np.r_[0, np.cumsum(valid, dtype=np.int64)]
        case_slices: dict[str, tuple[int, int]] = {}
        for case_id in np.unique(case_ids):
            left = int(np.searchsorted(case_ids, case_id, side="left"))
            right = int(np.searchsorted(case_ids, case_id, side="right"))
            case_slices[case_id] = (left, right)

        for case_id, (left, right) in case_slices.items():
            in_case = events["case_id"].to_numpy(dtype=str) == case_id
            onsets = event_onsets[in_case]
            if len(onsets):
                position = np.searchsorted(onsets, times[left:right], side="right")
                upcoming = position < len(onsets)
                gaps = np.full(right - left, np.inf, dtype=float)
                gaps[upcoming] = onsets[position[upcoming]] - times[left:right][upcoming]
                next_onset_gap[left:right] = gaps

        for i, (case_id, onset) in enumerate(zip(events["case_id"], event_onsets)):
            if case_id not in case_slices:
                event_eligible[i] = False
                continue
            left, right = case_slices[case_id]
            case_times = times[left:right]
            timely_start[i] = left + np.searchsorted(
                case_times, onset - HORIZON_SECONDS, side="left"
            )
            timely_end[i] = left + np.searchsorted(
                case_times, onset - MIN_LEAD_SECONDS, side="right"
            )
            coverage_start[i] = left + np.searchsorted(
                case_times, onset - HORIZON_SECONDS, side="left"
            )
            coverage_end[i] = left + np.searchsorted(
                case_times, onset, side="left"
            )
            event_eligible[i] &= (
                valid_prefix[timely_end[i]] - valid_prefix[timely_start[i]] > 0
            )
        coverage_denominator = (
            valid_prefix[coverage_end] - valid_prefix[coverage_start]
        )

        return cls(
            split=split,
            times=times,
            scores=scores,
            case_codes=case_codes,
            consecutive=consecutive,
            valid=valid,
            next_onset_gap=next_onset_gap,
            event_onsets=event_onsets,
            eligible_events=event_eligible,
            timely_start=timely_start,
            timely_end=timely_end,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            coverage_denominator=coverage_denominator,
        )

    def _alarm_indices(self, above: np.ndarray) -> np.ndarray:
        previous_above = np.r_[False, above[:-1] & self.consecutive[1:]]
        candidates = np.flatnonzero(above & ~previous_above)
        accepted: list[int] = []
        last_case = -1
        last_time = -np.inf
        for index in candidates:
            case = int(self.case_codes[index])
            if case != last_case:
                last_case = case
                last_time = -np.inf
            time = self.times[index]
            if time - last_time >= REFRACTORY_SECONDS:
                accepted.append(int(index))
                last_time = time
        return np.asarray(accepted, dtype=np.int64)

    def summarize(self, threshold: float, include_coverage: bool = True) -> dict[str, float | int]:
        if not np.isfinite(threshold):
            raise ValueError("Threshold must be finite.")
        above = self.valid & (self.scores >= threshold)
        alarms = self._alarm_indices(above)
        alarm_flags = np.zeros(len(self.times), dtype=np.int8)
        alarm_flags[alarms] = 1
        alarm_prefix = np.r_[0, np.cumsum(alarm_flags, dtype=np.int64)]
        detected = self.eligible_events & (
            alarm_prefix[self.timely_end] - alarm_prefix[self.timely_start] > 0
        )
        false_count = int((self.next_onset_gap[alarms] > HORIZON_SECONDS).sum())
        late_count = int(
            ((self.next_onset_gap[alarms] > 0)
             & (self.next_onset_gap[alarms] < MIN_LEAD_SECONDS)).sum()
        )
        eligible_count = self.eligible_event_count
        sensitivity = (
            100.0 * int(detected.sum()) / eligible_count
            if eligible_count else float("nan")
        )
        result: dict[str, float | int] = {
            "threshold": float(threshold),
            "timely_sensitivity_pct": float(sensitivity),
            "false_alarms_per_hour": false_count / self.evaluable_hours,
            "evaluable_hours": self.evaluable_hours,
            "evaluable_decisions": int(self.valid.sum()),
            "eligible_events": eligible_count,
            "detected_events": int(detected.sum()),
            "alarms": len(alarms),
            "false_alarms": false_count,
            "late_alarms": late_count,
        }
        if not include_coverage:
            return result

        if detected.any():
            selected_events = np.flatnonzero(detected)
            first_positions = np.searchsorted(
                alarms, self.timely_start[selected_events], side="left"
            )
            first_alarms = alarms[first_positions]
            leads = self.event_onsets[selected_events] - self.times[first_alarms]
            q1, median, q3 = np.percentile(leads, [25, 50, 75])
        else:
            q1 = median = q3 = float("nan")
        if eligible_count:
            above_prefix = np.r_[0, np.cumsum(above, dtype=np.int64)]
            numerators = (
                above_prefix[self.coverage_end]
                - above_prefix[self.coverage_start]
            )
            coverage = 100.0 * np.mean(
                numerators[self.eligible_events]
                / self.coverage_denominator[self.eligible_events]
            )
        else:
            coverage = float("nan")
        result.update(
            coverage_pct=float(coverage),
            lead_q1_s=float(q1),
            lead_median_s=float(median),
            lead_q3_s=float(q3),
        )
        return result


if njit is not None:
    @njit(cache=True)
    def _select_thresholds_compiled(
        scores: np.ndarray,
        times: np.ndarray,
        case_codes: np.ndarray,
        consecutive: np.ndarray,
        valid: np.ndarray,
        next_onset_gap: np.ndarray,
        event_eligible: np.ndarray,
        timely_start: np.ndarray,
        timely_end: np.ndarray,
        thresholds: np.ndarray,
        budgets: np.ndarray,
        evaluable_hours: float,
        no_alarm: float,
    ) -> np.ndarray:
        """Exact threshold scan with the same crossing/refractory rules."""
        best = np.full(len(budgets), no_alarm, dtype=np.float64)
        best_detected = np.zeros(len(budgets), dtype=np.int64)
        eligible_count = int(event_eligible.sum())
        alarm_prefix = np.empty(len(scores) + 1, dtype=np.int64)

        for threshold in thresholds:
            alarm_prefix[0] = 0
            false_count = 0
            previous_above = False
            last_case = -1
            last_alarm_time = -np.inf
            for i in range(len(scores)):
                above = valid[i] and scores[i] >= threshold
                crossing = above and (not consecutive[i] or not previous_above)
                accepted = False
                if crossing:
                    if case_codes[i] != last_case:
                        last_case = case_codes[i]
                        last_alarm_time = -np.inf
                    if times[i] - last_alarm_time >= REFRACTORY_SECONDS:
                        accepted = True
                        last_alarm_time = times[i]
                        if next_onset_gap[i] > HORIZON_SECONDS:
                            false_count += 1
                alarm_prefix[i + 1] = alarm_prefix[i] + int(accepted)
                previous_above = above

            detected = 0
            for i in range(len(event_eligible)):
                if event_eligible[i] and (
                    alarm_prefix[timely_end[i]] > alarm_prefix[timely_start[i]]
                ):
                    detected += 1
            false_rate = false_count / evaluable_hours
            all_maxed = True
            for i in range(len(budgets)):
                if false_rate <= budgets[i] + 1e-12 and detected > best_detected[i]:
                    best[i] = threshold
                    best_detected[i] = detected
                if best_detected[i] != eligible_count:
                    all_maxed = False
            if all_maxed:
                break
        return best
else:
    _select_thresholds_compiled = None


def select_validation_thresholds(
    validation: MonitoringSplit,
    budgets: Iterable[float] = DEFAULT_BUDGETS,
) -> dict[float, float]:
    """Exhaustively search observed validation scores, once for all budgets.

    Among thresholds with maximum timely sensitivity and FA/h within budget,
    ties retain the highest threshold. The threshold just above 1 gives the
    feasible no-alarm operating point.
    """
    if validation.eligible_event_count == 0:
        raise ValueError("Validation has no eligible events for threshold selection.")
    budgets = tuple(float(value) for value in budgets)
    if not budgets or not all(np.isfinite(value) and value >= 0 for value in budgets):
        raise ValueError("Budgets must be nonnegative finite values.")
    no_alarm = float(np.nextafter(1.0, np.inf))
    thresholds = np.unique(validation.scores[validation.valid])[::-1]
    if (
        _select_thresholds_compiled is not None
        and len(thresholds) * len(validation.scores) >= 25_000_000
    ):
        selected = _select_thresholds_compiled(
            validation.scores,
            validation.times,
            validation.case_codes,
            validation.consecutive,
            validation.valid,
            validation.next_onset_gap,
            validation.eligible_events,
            validation.timely_start,
            validation.timely_end,
            thresholds,
            np.asarray(budgets, dtype=float),
            validation.evaluable_hours,
            no_alarm,
        )
        return dict(zip(budgets, (float(value) for value in selected)))
    best = {budget: no_alarm for budget in budgets}
    best_sensitivity = {budget: 0.0 for budget in budgets}
    for threshold in thresholds:
        summary = validation.summarize(float(threshold), include_coverage=False)
        sensitivity = float(summary["timely_sensitivity_pct"])
        false_rate = float(summary["false_alarms_per_hour"])
        for budget in budgets:
            if false_rate <= budget + 1e-12 and sensitivity > best_sensitivity[budget]:
                best[budget] = float(threshold)
                best_sensitivity[budget] = sensitivity
        if all(value == 100.0 for value in best_sensitivity.values()):
            break
    return best


def evaluate_monitoring(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    budgets: Iterable[float] = DEFAULT_BUDGETS,
) -> pd.DataFrame:
    """Fit thresholds on validation and report unchanged-threshold test results.

    Call separately for each model and evaluation run. Both tables require a
    split column with val and test rows. They must share the same case/time
    origin within each split.
    """
    if "split" not in decisions or "split" not in events:
        raise ValueError("Both tables require a split column (val/test).")
    if not set(decisions["split"].astype(str)).issubset({"val", "test"}):
        raise ValueError("Decision split values must be val or test.")
    if not set(events["split"].astype(str)).issubset({"val", "test"}):
        raise ValueError("Event split values must be val or test.")
    val_cases = set(decisions.loc[decisions["split"] == "val", "case_id"].astype(str))
    test_cases = set(decisions.loc[decisions["split"] == "test", "case_id"].astype(str))
    if val_cases & test_cases:
        raise ValueError("Validation and test case_ids overlap.")

    prepared = {
        split: MonitoringSplit.from_frames(
            decisions.loc[decisions["split"] == split],
            events.loc[events["split"] == split],
            split,
        )
        for split in ("val", "test")
    }
    if prepared["test"].eligible_event_count == 0:
        raise ValueError("Test has no eligible events.")
    thresholds = select_validation_thresholds(prepared["val"], budgets)
    rows = []
    for budget, threshold in thresholds.items():
        validation = prepared["val"].summarize(threshold)
        test = prepared["test"].summarize(threshold)
        rows.append({
            "validation_budget_per_hour": budget,
            "threshold": threshold,
            "validation_timely_sensitivity_pct": validation["timely_sensitivity_pct"],
            "validation_false_alarms_per_hour": validation["false_alarms_per_hour"],
            "test_timely_sensitivity_pct": test["timely_sensitivity_pct"],
            "test_false_alarms_per_hour": test["false_alarms_per_hour"],
            "test_coverage_pct": test["coverage_pct"],
            "test_lead_q1_s": test["lead_q1_s"],
            "test_lead_median_s": test["lead_median_s"],
            "test_lead_q3_s": test["lead_q3_s"],
            "test_evaluable_hours": test["evaluable_hours"],
            "test_eligible_events": test["eligible_events"],
            "test_detected_events": test["detected_events"],
            "test_false_alarms": test["false_alarms"],
        })
    return pd.DataFrame(rows)
