"""Focused checks for the manuscript's risk target and event labeling."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from hyponet.preprocessing.segmentation import (
    SegmentationConfig,
    annotate_segment_event_context,
    assign_hard_labels,
    calculate_map_from_waveform,
    detect_hypotension_intervals,
    load_case_list,
    save_case_segments,
)
from hyponet.preprocessing.soft_labels import (
    add_soft_labels_to_segments,
    calculate_trend_slope,
    combined_probability,
    component_probabilities,
)


class SoftTargetTests(unittest.TestCase):
    def test_manuscript_worked_example(self):
        self.assertAlmostEqual(combined_probability(4, 85, 0, 5), 0.594)
        self.assertAlmostEqual(combined_probability(1, 85, 0, 5), 0.834)

    def test_cue_boundaries_and_ongoing_priority(self):
        self.assertEqual(component_probabilities(None, 65, -1, 5),
                         {"time": 0.0, "map": 1.0, "trend": 1.0})
        self.assertEqual(component_probabilities(5, 105, 1, 5),
                         {"time": 0.0, "map": 0.0, "trend": 0.0})
        self.assertEqual(component_probabilities(4, 85, 0, 5, ongoing_event=True)["time"], 1.0)
        self.assertAlmostEqual(combined_probability(4, 85, 0, 5, ongoing_event=True), 0.85)

    def test_five_second_valid_average_and_physical_slope(self):
        rate = 100
        time_minutes = np.arange(60 * rate) / (rate * 60.0)
        falling = 100 - 15 * time_minutes
        rising = 100 + 45 * time_minutes
        self.assertAlmostEqual(calculate_trend_slope(falling, sample_rate=rate), -0.5, places=10)
        self.assertAlmostEqual(calculate_trend_slope(rising, sample_rate=rate), 1.0)


class SegmentEndpointTests(unittest.TestCase):
    def setUp(self):
        self.config = SegmentationConfig(sample_rate=1, horizon_minutes=5)

    @staticmethod
    def segment(start):
        return {
            "art": np.full(60, 85.0),
            "map": 85.0,
            "global_start_idx": start,
            "global_end_idx": start + 60,
        }

    def test_horizon_endpoint_ongoing_and_event_free(self):
        # Onset at 600 s: segment ending at 300 s is exactly H=5 min away.
        segments = [self.segment(start) for start in (240, 300, 540, 600, 720)]
        context = annotate_segment_event_context(
            segments, [{"start_idx": 600, "end_idx": 720}], 1200, self.config,
        )
        labeled = assign_hard_labels(context, self.config)
        self.assertEqual([s["hard_label"] for s in labeled], [1, 1, 1, 1, 0])
        self.assertEqual(labeled[0]["time_to_next_onset_minutes"], 5)
        self.assertFalse(labeled[2]["ongoing_event"])
        self.assertTrue(labeled[3]["ongoing_event"])

        soft = add_soft_labels_to_segments(labeled, 5, sample_rate=1)
        self.assertEqual([key for key in soft[0] if key.startswith("soft_label_")],
                         ["soft_label_concave"])
        self.assertAlmostEqual(soft[0]["soft_label_concave"], 0.45)
        self.assertAlmostEqual(soft[2]["soft_label_concave"], 0.85)
        self.assertAlmostEqual(soft[3]["soft_label_concave"], 0.85)

    def test_incomplete_horizon_excludes_nonongoing_even_with_visible_onset(self):
        segments = [self.segment(420), self.segment(600)]
        context = annotate_segment_event_context(
            segments, [{"start_idx": 600, "end_idx": 720}], 750, self.config,
        )
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]["global_start_idx"], 600)
        self.assertTrue(context[0]["ongoing_event"])
        self.assertFalse(context[0]["future_horizon_complete"])
        self.assertEqual(assign_hard_labels(context, self.config)[0]["hard_label"], 1)

    def test_episode_requires_more_than_one_minute(self):
        rate = 100
        config = SegmentationConfig(sample_rate=rate)

        def pulse_waveform(low_seconds):
            time_seconds = np.arange(150 * rate) / rate
            baseline = np.where(
                (time_seconds >= 20) & (time_seconds < 20 + low_seconds),
                60.0, 100.0,
            )
            return baseline + 10 * np.sin(2 * np.pi * time_seconds)

        self.assertEqual(detect_hypotension_intervals(pulse_waveform(60), config), [])
        episodes = detect_hypotension_intervals(pulse_waveform(61), config)
        self.assertEqual(len(episodes), 1)
        self.assertGreater(episodes[0]["end_idx"] - episodes[0]["start_idx"], 60 * rate)

    def test_segment_map_uses_episode_pulse_estimator(self):
        rate = 100
        seconds = np.arange(60 * rate) / rate
        waveform = 80 + 10 * np.sin(2 * np.pi * seconds)
        self.assertAlmostEqual(calculate_map_from_waveform(waveform, rate), 76.6666667, places=3)

    def test_case_list_accepts_text_and_csv(self):
        with TemporaryDirectory() as directory:
            text_path = Path(directory) / "cases.txt"
            text_path.write_text("0001\n2.vital\n", encoding="utf-8")
            csv_path = Path(directory) / "cases.csv"
            csv_path.write_text("case_id,fold\n0003,1\n", encoding="utf-8")
            self.assertEqual(load_case_list(text_path), {"1", "2"})
            self.assertEqual(load_case_list(csv_path), {"3"})

    def test_empty_case_files_use_fold_compatible_name_and_shape(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_case_segments("0001_extra.vital", 1, [], output_dir)
            self.assertEqual(np.load(output_dir / "0001_ART.npy").shape, (0, 6000))
            self.assertEqual(np.load(output_dir / "0001_ECG.npy").shape, (0, 6000))
            self.assertEqual(np.load(output_dir / "0001_PLETH.npy").shape, (0, 6000))
            self.assertEqual(np.load(output_dir / "0001_LABELS_CONCAVE.npy").shape, (0,))
            self.assertFalse((output_dir / "0001_LABELS_LINEAR.npy").exists())


if __name__ == "__main__":
    unittest.main()
