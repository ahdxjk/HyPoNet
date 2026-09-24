"""Checks for patient isolation and within-segment waveform repair."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hyponet.preprocessing.folds import (
    assign_patient_folds,
    fill_missing_values,
    load_clinical_table,
    read_fold_manifests,
)
from scripts.score_monitoring import _patient_ids


class PatientFoldTests(unittest.TestCase):
    def test_monitoring_uses_shared_patient_id_for_multiple_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            csv_path = Path(temp) / "clinical.csv"
            pd.DataFrame({"caseid": [1, 2], "patient": ["p1", "p1"]}).to_csv(csv_path, index=False)
            self.assertEqual(_patient_ids(csv_path, "patient"), {"0001": "p1", "0002": "p1"})

    def test_assignment_precedes_segmentation_and_keeps_patients_together(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            records = []
            case_to_patient = {}
            for patient in range(60):
                for recording in range(2):
                    case_id = f"{patient * 2 + recording + 1:04d}"
                    (raw_dir / f"{case_id}.vital").touch()
                    case_to_patient[case_id] = patient
                    records.append({"caseid": case_id, "patient": patient, "age": 40 + (patient % 3) * 10})
            csv_path = root / "clinical.csv"
            pd.DataFrame(records).to_csv(csv_path, index=False)

            manifest_dir = root / "splits"
            assign_patient_folds(csv_path, raw_dir, manifest_dir, patient_id_column="patient")
            folds = read_fold_manifests(manifest_dir)
            self.assertTrue((manifest_dir / "all_cases.txt").exists())
            for splits in folds.values():
                patient_sets = {
                    name: {case_to_patient[case] for case in cases}
                    for name, cases in splits.items()
                }
                self.assertFalse(patient_sets["train"] & patient_sets["val"])
                self.assertFalse(patient_sets["train"] & patient_sets["test"])
                self.assertFalse(patient_sets["val"] & patient_sets["test"])
                self.assertEqual(len(patient_sets["train"] | patient_sets["val"] | patient_sets["test"]), 60)

    def test_missing_waveform_samples_are_filled_within_each_segment(self):
        signal = np.array([[1.0, np.nan, 3.0], [100.0, np.nan, 300.0]], dtype=np.float32)
        filled = fill_missing_values(signal, "ART")
        np.testing.assert_allclose(filled[:, 1], [2.0, 200.0])

    def test_numeric_case_ids_survive_float_csv_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "0001.vital").touch()
            csv_path = root / "clinical.csv"
            csv_path.write_text("caseid,age,patient\n1,50,A\n,40,B\n", encoding="utf-8")
            table = load_clinical_table(csv_path, root, patient_id_column="patient")
            self.assertEqual(table["caseid"].tolist(), ["0001"])


if __name__ == "__main__":
    unittest.main()
