"""Checks for matched chronological score-dynamics calculations."""

import unittest

import numpy as np
import pandas as pd

from hyponet.score_dynamics import (
    evaluate_score_dynamics,
    fit_validation_transform,
    transform_scores,
)


def example_inputs():
    hard_rows = []
    soft_rows = []
    for time_s, hard, soft in ((-60, 0.0, 0.0), (-40, 0.5, 0.25), (-20, 1.0, 0.5)):
        common = {"split": "val", "case_id": "V", "patient_id": "PV",
                  "time_s": time_s, "stable": ""}
        hard_rows.append({**common, "score": hard})
        soft_rows.append({**common, "score": soft})
    for case_id, patient_id, count in (("A", "P1", 11), ("B", "P2", 10)):
        for index in range(count):
            # Case A's last point is separated by 40 s, so only its first
            # ten scores form a complete rolling window.
            time_s = index * 20 + (20 if case_id == "A" and index == 10 else 0)
            hard = index / 10 if case_id == "A" else 0.2
            soft = hard / 2 if case_id == "A" else 0.2
            common = {"split": "test", "case_id": case_id, "patient_id": patient_id,
                      "time_s": time_s, "stable": True}
            hard_rows.append({**common, "score": hard})
            soft_rows.append({**common, "score": soft})
    events = pd.DataFrame([
        {"split": "test", "case_id": "A", "patient_id": "P1", "onset_s": 240},
        {"split": "test", "case_id": "B", "patient_id": "P2", "onset_s": 220},
    ])
    return pd.DataFrame(hard_rows), pd.DataFrame(soft_rows), events


class ScoreDynamicsTests(unittest.TestCase):
    def test_validation_fitted_scales(self):
        fitted = fit_validation_transform(np.array([0.0, 0.5, 1.0]))
        scaled = transform_scores(np.array([0.25, 0.75]), fitted)
        np.testing.assert_allclose(scaled["percentile"], [1 / 3, 2 / 3])
        np.testing.assert_allclose(scaled["standardized"],
                                   (np.array([0.25, 0.75]) - 0.5) / np.std([0, 0.5, 1]))

    def test_case_then_event_averaging_and_gap_rejection(self):
        hard, soft, events = example_inputs()
        summary, cases, responses = evaluate_score_dynamics(
            hard, soft, events, n_bootstrap=100, seed=7,
        )
        raw_case = cases.loc[cases["scale"].eq("raw")].set_index("case_id")
        self.assertEqual(raw_case.loc["A", "n_windows"], 1)
        self.assertEqual(raw_case.loc["B", "n_windows"], 1)
        expected_case_a_sd = np.std(np.arange(10) / 10, ddof=1)
        self.assertAlmostEqual(raw_case.loc["A", "hard"], expected_case_a_sd)
        raw_rolling = summary.loc[(summary["scale"] == "raw") &
                                  (summary["measure"] == "rolling_sd")].iloc[0]
        self.assertAlmostEqual(raw_rolling["hard"], expected_case_a_sd / 2)
        self.assertAlmostEqual(raw_rolling["soft"], expected_case_a_sd / 4)
        self.assertEqual(raw_rolling["n_units"], 2)

        raw_events = responses.loc[responses["scale"].eq("raw")].set_index("case_id")
        self.assertEqual(len(raw_events), 2)  # No alarm or detection filter.
        self.assertAlmostEqual(raw_events.loc["A", "hard_rise"], 1.0)
        self.assertAlmostEqual(raw_events.loc["A", "soft_rise"], 0.5)
        self.assertAlmostEqual(raw_events.loc["A", "hard_spearman"], 1.0)
        self.assertAlmostEqual(raw_events.loc["B", "hard_spearman"], 0.0)
        raw_rise = summary.loc[(summary["scale"] == "raw") &
                               (summary["measure"] == "pre_onset_rise")].iloc[0]
        self.assertAlmostEqual(raw_rise["hard"], 0.5)
        self.assertAlmostEqual(raw_rise["soft"], 0.25)
        self.assertEqual(raw_rise["n_units"], 2)

    def test_stable_mark_required_and_models_must_align(self):
        hard, soft, events = example_inputs()
        with self.assertRaisesRegex(ValueError, "stable"):
            evaluate_score_dynamics(hard.drop(columns="stable"), soft, events, n_bootstrap=0)
        soft.loc[(soft["split"] == "test") & (soft["case_id"] == "A"), "stable"] = False
        with self.assertRaisesRegex(ValueError, "do not match"):
            evaluate_score_dynamics(hard, soft, events, n_bootstrap=0)

    def test_invalid_grid_rows_and_ineligible_events_are_reported(self):
        hard, soft, events = example_inputs()
        for frame in (hard, soft):
            frame["ongoing"] = False
            frame["input_complete"] = True
            frame["future_complete"] = True
        invalid = {"split": "test", "case_id": "A", "patient_id": "P1",
                   "time_s": 200, "score": np.nan, "stable": "",
                   "ongoing": False, "input_complete": False, "future_complete": True}
        hard.loc[len(hard)] = invalid
        soft.loc[len(soft)] = invalid
        events = pd.concat([
            events.assign(eligible=True),
            pd.DataFrame([{"split": "test", "case_id": "C", "onset_s": 200,
                           "eligible": True}]),
        ], ignore_index=True)
        summary, cases, responses = evaluate_score_dynamics(
            hard, soft, events, n_bootstrap=0,
        )
        raw_rise = summary.loc[(summary["scale"] == "raw") &
                               (summary["measure"] == "pre_onset_rise")].iloc[0]
        self.assertEqual(raw_rise["n_events_input"], 3)
        self.assertEqual(raw_rise["n_events_excluded"], 1)
        self.assertEqual(raw_rise["n_units"], 2)
        excluded = responses.loc[(responses["scale"] == "raw") &
                                 (responses["case_id"] == "C")].iloc[0]
        self.assertEqual(excluded["status"], "insufficient_pre_onset_decisions")
        self.assertEqual(excluded["n_decisions"], 0)
        self.assertEqual(cases.loc[(cases["scale"] == "raw") &
                                   (cases["case_id"] == "A"), "n_windows"].iloc[0], 1)

    def test_validation_decision_sets_must_match(self):
        hard, soft, events = example_inputs()
        soft.loc[(soft["split"] == "val") & (soft["time_s"] == -20), "time_s"] = -10
        with self.assertRaisesRegex(ValueError, "validation decision rows"):
            evaluate_score_dynamics(hard, soft, events, n_bootstrap=0)

    def test_validation_mapping_is_fitted_separately_per_run(self):
        rows = []
        event_rows = []
        for run, case_id, upper in (("1", "A", 1.0), ("2", "B", 0.2)):
            for index, score in enumerate((0.0, upper / 2, upper)):
                rows.append({"run": run, "split": "val", "case_id": f"V{run}",
                             "time_s": index * 20, "score": score, "stable": ""})
            for index in range(10):
                rows.append({"run": run, "split": "test", "case_id": case_id,
                             "time_s": index * 20, "score": index * 0.02,
                             "stable": True})
            event_rows.append({"split": "test", "case_id": case_id, "onset_s": 200})
        frame = pd.DataFrame(rows)
        _, _, responses = evaluate_score_dynamics(
            frame, frame, pd.DataFrame(event_rows), n_bootstrap=0,
        )
        percentile = responses.loc[responses["scale"].eq("percentile")].set_index("run")
        self.assertAlmostEqual(percentile.loc["1", "hard_rise"], 0.0)
        self.assertAlmostEqual(percentile.loc["2", "hard_rise"], 1 / 3)
        self.assertEqual(set(percentile["patient_id"]), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
