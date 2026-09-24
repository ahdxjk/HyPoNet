"""Focused checks for chronological threshold selection and event metrics."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from hyponet.monitoring import (
    MonitoringSplit,
    _select_thresholds_compiled,
    evaluate_monitoring,
    select_validation_thresholds,
)


def decisions_for_case(split, case_id, highs, *, invalid=None):
    invalid = invalid or {}
    rows = []
    for time_s in range(60, 400, 20):
        flags = invalid.get(time_s, {})
        rows.append({
            "split": split,
            "case_id": case_id,
            "time_s": time_s,
            "score": highs.get(time_s, 0.1),
            "ongoing": flags.get("ongoing", False),
            "input_complete": flags.get("input_complete", True),
            "future_complete": flags.get("future_complete", True),
        })
    return rows


class MonitoringTests(unittest.TestCase):
    def test_refractory_timely_lead_and_event_averaged_coverage(self):
        decisions = pd.DataFrame(
            decisions_for_case("test", "a", {100: 0.9, 200: 0.9, 380: 0.9})
            + decisions_for_case("test", "b", {})
        )
        events = pd.DataFrame([
            {"split": "test", "case_id": "a", "onset_s": 400},
            {"split": "test", "case_id": "b", "onset_s": 400},
        ])
        split = MonitoringSplit.from_frames(decisions, events, "test")
        result = split.summarize(0.9)
        self.assertEqual(result["alarms"], 1)
        self.assertEqual(result["detected_events"], 1)
        self.assertEqual(result["eligible_events"], 2)
        self.assertEqual(result["timely_sensitivity_pct"], 50.0)
        self.assertEqual(result["lead_median_s"], 300.0)
        self.assertAlmostEqual(result["coverage_pct"], 10.0)
        self.assertEqual(result["false_alarms"], 0)

    def test_ineligible_rows_cannot_create_alarms(self):
        decisions = pd.DataFrame(decisions_for_case(
            "test", "a", {100: 0.99, 120: 0.9},
            invalid={100: {"ongoing": True}},
        ))
        events = pd.DataFrame([{
            "split": "test", "case_id": "a", "onset_s": 400,
        }])
        result = MonitoringSplit.from_frames(decisions, events, "test").summarize(0.9)
        self.assertEqual(result["alarms"], 1)
        self.assertEqual(result["lead_median_s"], 280.0)
        self.assertEqual(result["evaluable_decisions"], 16)

    def test_threshold_is_fitted_only_on_validation(self):
        decisions = pd.DataFrame(
            decisions_for_case("val", "v_event", {100: 0.8})
            + decisions_for_case("val", "v_stable", {100: 0.7})
            + decisions_for_case("test", "t_event", {100: 0.75})
        )
        events = pd.DataFrame([
            {"split": "val", "case_id": "v_event", "onset_s": 400},
            {"split": "test", "case_id": "t_event", "onset_s": 400},
        ])
        result = evaluate_monitoring(decisions, events, budgets=[0.0]).iloc[0]
        self.assertEqual(result["threshold"], 0.8)
        self.assertEqual(result["validation_timely_sensitivity_pct"], 100.0)
        self.assertEqual(result["validation_false_alarms_per_hour"], 0.0)
        self.assertEqual(result["test_timely_sensitivity_pct"], 0.0)
        self.assertEqual(result["test_coverage_pct"], 0.0)

    @unittest.skipIf(_select_thresholds_compiled is None, "Numba is optional")
    def test_compiled_exhaustive_scan_matches_numpy(self):
        decisions = pd.DataFrame(
            decisions_for_case(
                "val", "a", {100: 0.9, 200: 0.8, 340: 0.7},
                invalid={200: {"future_complete": False}},
            )
            + decisions_for_case("val", "b", {100: 0.75})
        )
        events = pd.DataFrame([{
            "split": "val", "case_id": "a", "onset_s": 400,
        }])
        split = MonitoringSplit.from_frames(decisions, events, "val")
        budgets = (0.0, 10.0)
        expected = select_validation_thresholds(split, budgets)
        actual = _select_thresholds_compiled(
            split.scores,
            split.times,
            split.case_codes,
            split.consecutive,
            split.valid,
            split.next_onset_gap,
            split.eligible_events,
            split.timely_start,
            split.timely_end,
            np.unique(split.scores[split.valid])[::-1],
            np.asarray(budgets),
            split.evaluable_hours,
            np.nextafter(1.0, np.inf),
        )
        np.testing.assert_array_equal(actual, [expected[b] for b in budgets])

    @unittest.skipIf(_select_thresholds_compiled is None, "Numba is optional")
    def test_five_thousand_unique_scores_keep_exact_selection(self):
        count = 5000
        decisions = pd.DataFrame({
            "case_id": ["a"] * count,
            "time_s": 60 + 20 * np.arange(count),
            "score": np.random.default_rng(123).random(count),
            "ongoing": [False] * count,
            "input_complete": [True] * count,
            "future_complete": [True] * count,
        })
        events = pd.DataFrame({
            "case_id": ["a", "a"],
            "onset_s": [10000, 25000],
        })
        split = MonitoringSplit.from_frames(decisions, events, "val")
        budgets = (0.0, 0.25, 0.5, 2.0)
        accelerated = select_validation_thresholds(split, budgets)
        with patch("hyponet.monitoring._select_thresholds_compiled", None):
            reference = select_validation_thresholds(split, budgets)
        self.assertEqual(accelerated, reference)


if __name__ == "__main__":
    unittest.main()
