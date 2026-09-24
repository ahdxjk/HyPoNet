"""Focused checks for training-only handcrafted feature preprocessing."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

import h5py
import numpy as np

from hyponet.preprocessing.handcrafted_features import (
    FEATURE_DIM,
    MLFeatureExtractor,
    TrainingFeaturePreprocessor,
    compute_hemodynamic_features,
    extract_features_for_split,
)


class FeaturePreprocessingTests(TestCase):
    def test_validation_uses_training_means_and_scale(self) -> None:
        train = np.zeros((2, FEATURE_DIM), dtype=np.float32)
        train[:, 0] = [0.0, 2.0]
        train[:, 1] = [np.nan, 4.0]
        train[:, 2] = [7.0, 7.0]
        train[:, 3] = [np.inf, 6.0]
        fitted = TrainingFeaturePreprocessor().fit(train)

        validation = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        validation[0, :4] = [10.0, np.nan, 9.0, -np.inf]
        transformed = fitted.transform(validation)

        np.testing.assert_allclose(transformed[0, :4], [9.0, 0.0, 2.0, 0.0])
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_allclose(fitted.column_means[:4], [1.0, 4.0, 7.0, 6.0])

    def test_all_nonfinite_training_column_fails(self) -> None:
        train = np.zeros((2, FEATURE_DIM), dtype=np.float32)
        train[:, 5] = [np.nan, np.inf]
        with self.assertRaisesRegex(ValueError, r"no finite values: \[5\]"):
            TrainingFeaturePreprocessor().fit(train)

    def test_saved_training_statistics_reproduce_inference_transform(self) -> None:
        train = np.zeros((3, FEATURE_DIM), dtype=np.float32)
        train[:, 0] = [0.0, 2.0, 4.0]
        train[:, 1] = [3.0, np.nan, 9.0]
        inference = np.zeros((2, FEATURE_DIM), dtype=np.float32)
        inference[:, :2] = [[10.0, np.nan], [np.inf, 6.0]]
        fitted = TrainingFeaturePreprocessor().fit(train)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "feature_preprocessor.npz"
            fitted.save(path)
            restored = TrainingFeaturePreprocessor.load(path)
            np.testing.assert_allclose(restored.transform(inference), fitted.transform(inference))

    def test_standalone_validation_reads_sibling_train(self) -> None:
        train = np.zeros((2, FEATURE_DIM), dtype=np.float32)
        train[:, 0] = [0.0, 2.0]
        validation = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        validation[:, 0] = 10.0
        with TemporaryDirectory() as temp:
            fold_dir = Path(temp)
            train_dir = fold_dir / "train"
            val_dir = fold_dir / "val"
            train_dir.mkdir()
            val_dir.mkdir()

            def load_raw(split_dir: Path, _static_data: np.ndarray) -> np.ndarray:
                return train if split_dir == train_dir else validation

            with patch("hyponet.preprocessing.handcrafted_features.load_raw_features_for_split", side_effect=load_raw):
                output = extract_features_for_split(val_dir, np.empty((0, 4)))
            with h5py.File(output, "r") as h5f:
                self.assertEqual(h5f["features"].shape, (1, FEATURE_DIM))
                self.assertAlmostEqual(float(h5f["features"][0, 0]), 9.0)

    def test_bmi_is_derived_and_static_values_survive_waveform_failure(self) -> None:
        extractor = MLFeatureExtractor()
        seen = {}

        def abp(_signals: np.ndarray, static: np.ndarray) -> np.ndarray:
            seen["static"] = static
            return np.zeros((len(static), FEATURE_DIM), dtype=np.float32)

        with patch.object(extractor, "extract_abp_features", side_effect=abp), \
             patch.object(extractor, "extract_ecg_features", return_value=np.empty((1, 0))), \
             patch.object(extractor, "extract_ppg_features", return_value=np.empty((1, 0))):
            extractor.extract_features(
                np.zeros((1, 3, 6000), dtype=np.float32),
                np.array([[50, 0, 170, 68, 999]], dtype=np.float32),
            )
        self.assertAlmostEqual(float(seen["static"][0, 4]), 68 / 1.7**2, places=4)

        fallback = compute_hemodynamic_features(np.empty(0), seen["static"][0])
        np.testing.assert_allclose(fallback[9:14], seen["static"][0])


if __name__ == "__main__":
    main()
