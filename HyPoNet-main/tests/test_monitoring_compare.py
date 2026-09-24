"""Matched chronological monitoring and patient-cluster uncertainty checks."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hyponet.monitoring_compare import compare_monitoring


def _case(split: str, case_id: str, patient_id: str,
          score_at_100: float = 0.1) -> list[dict]:
    return [{
        "split": split,
        "case_id": case_id,
        "patient_id": patient_id,
        "time_s": time_s,
        "score": score_at_100 if time_s == 100 else 0.1,
        "ongoing": False,
        "input_complete": True,
        "future_complete": True,
    } for time_s in range(60, 400, 20)]


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hard = pd.DataFrame(
        _case("val", "v_event", "v1", 0.8)
        + _case("val", "v_stable", "v2", 0.7)
        + _case("test", "t_event_a", "p1", 0.9)
        + _case("test", "t_stable", "p1", 0.8)
        + _case("test", "t_event_b", "p2", 0.7)
    )
    soft = hard.copy()
    soft.loc[soft["case_id"].eq("v_event") & soft["time_s"].eq(100), "score"] = 0.6
    soft.loc[soft["case_id"].eq("v_stable") & soft["time_s"].eq(100), "score"] = 0.5
    soft.loc[soft["case_id"].eq("t_event_a") & soft["time_s"].eq(100), "score"] = 0.7
    soft.loc[soft["case_id"].eq("t_stable") & soft["time_s"].eq(100), "score"] = 0.6
    soft.loc[soft["case_id"].eq("t_event_b") & soft["time_s"].eq(100), "score"] = 0.7
    events = pd.DataFrame([
        {"split": "val", "case_id": "v_event", "patient_id": "v1", "onset_s": 400},
        {"split": "test", "case_id": "t_event_a", "patient_id": "p1", "onset_s": 400},
        {"split": "test", "case_id": "t_event_b", "patient_id": "p2", "onset_s": 400},
    ])
    return hard, soft, events


class MatchedMonitoringTests(unittest.TestCase):
    def test_separate_validation_thresholds_and_paired_test_metrics(self):
        hard, soft, events = _fixture()
        row = compare_monitoring(hard, soft, events, budgets=[0.0],
                                 n_bootstrap=2000, seed=123).iloc[0]
        self.assertEqual(row["hard_threshold"], 0.8)
        self.assertEqual(row["soft_threshold"], 0.6)
        self.assertEqual(row["test_patients"], 2)
        self.assertEqual(row["test_eligible_events"], 2)
        self.assertEqual(row["hard_test_timely_sensitivity_pct"], 50.0)
        self.assertEqual(row["soft_test_timely_sensitivity_pct"], 100.0)
        self.assertEqual(row["difference_timely_sensitivity_pct"], 50.0)
        self.assertAlmostEqual(row["difference_coverage_pct"], 100 / 30)
        self.assertAlmostEqual(row["hard_test_false_alarms_per_hour"], 3600 / (3 * 17 * 20))
        self.assertAlmostEqual(row["soft_test_false_alarms_per_hour"], 3600 / (3 * 17 * 20))
        self.assertEqual(row["hard_test_lead_median_s"], 300.0)
        self.assertEqual(row["soft_test_lead_median_s"], 300.0)
        self.assertEqual(row["difference_timely_sensitivity_pct_ci_low"], 0.0)
        self.assertEqual(row["difference_timely_sensitivity_pct_ci_high"], 100.0)
        self.assertEqual(row["difference_coverage_pct_bootstrap_valid"], 2000)

    def test_test_scores_cannot_change_validation_thresholds(self):
        hard, soft, events = _fixture()
        baseline = compare_monitoring(hard, soft, events, budgets=[0.0],
                                      n_bootstrap=0).iloc[0]
        soft.loc[soft["split"].eq("test"), "score"] = 0.1
        changed = compare_monitoring(hard, soft, events, budgets=[0.0],
                                     n_bootstrap=0).iloc[0]
        self.assertEqual(changed["soft_threshold"], baseline["soft_threshold"])
        self.assertEqual(changed["soft_test_timely_sensitivity_pct"], 0.0)
        self.assertTrue(np.isnan(changed["difference_lead_median_s"]))

    def test_missing_patient_id_uses_case_id_cluster(self):
        hard, soft, events = _fixture()
        hard = hard.drop(columns="patient_id")
        soft = soft.drop(columns="patient_id")
        events = events.drop(columns="patient_id")
        row = compare_monitoring(hard, soft, events, budgets=[0.0],
                                 n_bootstrap=10).iloc[0]
        self.assertEqual(row["test_patients"], 3)

    def test_reject_mismatched_evaluability_or_patient_mapping(self):
        hard, soft, events = _fixture()
        soft.loc[soft.index[0], "future_complete"] = False
        with self.assertRaisesRegex(ValueError, "do not match"):
            compare_monitoring(hard, soft, events, budgets=[0.0], n_bootstrap=0)
        soft = hard.copy()
        soft.loc[soft["case_id"].eq("t_stable"), "patient_id"] = "p3"
        with self.assertRaisesRegex(ValueError, "do not match"):
            compare_monitoring(hard, soft, events, budgets=[0.0], n_bootstrap=0)

    def test_optional_second_event_file_must_match(self):
        hard, soft, events = _fixture()
        second = events.copy()
        second.loc[second["case_id"].eq("t_event_b"), "onset_s"] = 420
        with self.assertRaisesRegex(ValueError, "event annotations do not match"):
            compare_monitoring(hard, soft, events, soft_events=second,
                               budgets=[0.0], n_bootstrap=0)

    def test_multirun_thresholds_are_fitted_separately(self):
        hard, soft, events = _fixture()
        hard["run"] = "A"
        soft["run"] = "A"
        events["run"] = "A"
        h2 = hard.copy()
        s2 = soft.copy()
        e2 = events.copy()
        for frame in (h2, s2, e2):
            frame["run"] = "B"
            frame["case_id"] = "B_" + frame["case_id"]
            frame["patient_id"] = "B_" + frame["patient_id"]
        s2.loc[s2["case_id"].eq("B_v_event") & s2["time_s"].eq(100), "score"] = 0.95
        results = compare_monitoring(pd.concat([hard, h2]), pd.concat([soft, s2]),
                                     pd.concat([events, e2]), budgets=[0.0],
                                     n_bootstrap=0)
        self.assertEqual(results.set_index("run").loc["A", "soft_threshold"], 0.6)
        self.assertEqual(results.set_index("run").loc["B", "soft_threshold"], 0.95)


if __name__ == "__main__":
    unittest.main()
