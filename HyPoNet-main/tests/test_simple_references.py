"""Focused checks for the manuscript's simple pressure references."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from hyponet.preprocessing.soft_labels import calculate_trend_slope
from hyponet.simple_references import (
    FEATURE_SETS,
    LogisticReference,
    _fast_trend_slope,
    fit_all_references,
    load_training_pressure_cues,
    pressure_cues,
    score_case_references,
)
from scripts.run_simple_references import main as run_references


def pulse_wave(seconds: int, sample_rate: int, base: float = 80.0) -> np.ndarray:
    time = np.arange(seconds * sample_rate) / sample_rate
    return (base + 12 * np.sin(2 * np.pi * 1.1 * time)).astype(np.float32)


class SimpleReferenceTests(unittest.TestCase):
    def test_fast_trend_matches_soft_label_formula(self):
        fs = 100
        time = np.arange(60 * fs) / fs
        art = 85 + 8 * np.sin(2 * np.pi * 1.2 * time) - 0.2 * time
        art[1000:1005] = np.nan
        expected = calculate_trend_slope(art, sample_rate=fs)
        self.assertAlmostEqual(_fast_trend_slope(art, fs), expected, places=10)
        cues = pressure_cues(np.nan_to_num(art, nan=85), fs)
        self.assertEqual(cues.shape, (2,))
        self.assertTrue(((cues >= 0) & (cues <= 1)).all())

    def test_six_models_use_only_requested_cues_and_targets(self):
        rng = np.random.default_rng(17)
        cues = rng.uniform(0, 1, size=(240, 2))
        hard = (cues[:, 0] + 0.5 * cues[:, 1] > 0.8).astype(float)
        soft = 0.15 + 0.7 * cues[:, 0] + 0.1 * cues[:, 1]
        models = fit_all_references(cues, {"hard": hard, "soft": soft}, l2=1e-4)
        self.assertEqual(len(models), 6)
        for key, model in models.items():
            self.assertEqual(len(model.coefficients), 1 + len(FEATURE_SETS[model.name]))
            self.assertTrue(np.isfinite(model.predict(cues)).all(), key)
            self.assertTrue(((model.predict(cues) > 0) & (model.predict(cues) < 1)).all(), key)
        self.assertFalse(np.allclose(models["map_hard"].predict(cues), models["map_soft"].predict(cues)))

    def test_training_loader_rejects_cases_outside_train_manifest(self):
        fs = 20
        with tempfile.TemporaryDirectory() as temporary:
            train_dir = Path(temporary)
            arrays = {
                "ART": np.stack([pulse_wave(60, fs, 82), pulse_wave(60, fs, 72)]),
                "STATIC": np.array([[1], [2]], dtype=np.int32),
                "HARD_LABELS": np.array([[0], [1]], dtype=np.int32),
                "LABELS_CONCAVE": np.array([[0.2], [0.8]], dtype=np.float32),
            }
            for name, data in arrays.items():
                with h5py.File(train_dir / f"{name}.h5", "w") as file:
                    file.create_dataset(name, data=data)
            with self.assertRaisesRegex(ValueError, "outside the train manifest"):
                load_training_pressure_cues(train_dir, sample_rate=fs, allowed_case_ids={"0001"})
            cues, labels = load_training_pressure_cues(
                train_dir, sample_rate=fs, allowed_case_ids={"0001", "0002"},
            )
            self.assertEqual(cues.shape, (2, 2))
            np.testing.assert_array_equal(labels["hard"], [0, 1])

    def test_all_references_score_the_same_decision_grid(self):
        fs = 20
        art = pulse_wave(700, fs, 82)
        art[420 * fs : 500 * fs] -= 35
        ecg = pulse_wave(700, fs, 80)
        pleth = pulse_wave(700, fs, 75)
        models = {
            name: LogisticReference(name, "soft", np.ones(1 + len(columns)), 0.0)
            for name, columns in FEATURE_SETS.items()
        }
        decisions, events = score_case_references(
            art, ecg, pleth, models, split="test", case_id="0001", patient_id="p1", sample_rate=fs,
        )
        self.assertGreater(len(events), 0)
        common = [(row["time_s"], row["ongoing"], row["input_complete"], row["future_complete"])
                  for row in decisions["map"]]
        for rows in decisions.values():
            self.assertEqual(common, [(row["time_s"], row["ongoing"], row["input_complete"], row["future_complete"])
                                      for row in rows])
            self.assertTrue(all(np.isfinite(row["score"]) for row in rows if row["input_complete"]))
            self.assertTrue(all(row["patient_id"] == "p1" for row in rows))

    def test_end_to_end_fold_outputs_six_monitoring_results(self):
        fs = 20
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "vital").mkdir()
            (root / "manifests").mkdir()
            train_dir = root / "fold1" / "train"
            train_dir.mkdir(parents=True)
            selected = {"train": ["0001", "0002"], "val": ["0003", "0004"], "test": ["0005", "0006"]}
            for split, case_ids in selected.items():
                (root / "manifests" / f"fold1_{split}_cases.txt").write_text("\n".join(case_ids) + "\n")
                if split != "train":
                    for case_id in case_ids:
                        (root / "vital" / f"{case_id}.vital").touch()
            arrays = {
                "ART": np.stack([pulse_wave(60, fs, base) for base in (85, 80, 75, 68, 82, 72)]),
                "STATIC": np.array([[1], [1], [2], [2], [1], [2]], dtype=np.int32),
                "HARD_LABELS": np.array([[0], [0], [1], [1], [0], [1]], dtype=np.int32),
                "LABELS_CONCAVE": np.array([[0.15], [0.25], [0.8], [0.9], [0.2], [0.75]], dtype=np.float32),
            }
            for name, data in arrays.items():
                with h5py.File(train_dir / f"{name}.h5", "w") as file:
                    file.create_dataset(name, data=data)

            def fake_vital(path: Path, sample_rate: int):
                art = pulse_wave(750, sample_rate, 82)
                if path.stem in {"0003", "0005"}:
                    art[400 * sample_rate : 480 * sample_rate] -= 35
                ecg = pulse_wave(750, sample_rate, 80)
                ppg = pulse_wave(750, sample_rate, 75)
                return art, ecg, ppg

            argv = [
                "run_simple_references.py", "--input-dir", str(root / "vital"),
                "--splits-dir", str(root / "manifests"), "--fold-dir", str(root / "fold1"),
                "--fold", "1", "--output-dir", str(root / "out"), "--sample-rate", str(fs),
                "--l2", "0.0001",
            ]
            with patch("sys.argv", argv), patch("scripts.run_simple_references.load_vital_waveforms", fake_vital):
                run_references()
            result = pd.read_csv(root / "out" / "monitoring_results.csv")
            self.assertEqual(len(result), 6)
            self.assertEqual(set(result["model"]), set(FEATURE_SETS))
            self.assertEqual(set(result["supervision"]), {"hard", "soft"})
            for name in ("map", "trend", "map_trend"):
                for supervision in ("hard", "soft"):
                    self.assertTrue((root / "out" / f"{name}_{supervision}_decisions.csv").exists())


if __name__ == "__main__":
    unittest.main()
