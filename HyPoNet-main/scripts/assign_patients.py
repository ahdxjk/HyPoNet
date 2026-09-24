"""Assign patients to fixed five-fold runs before segment generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hyponet.preprocessing.folds import assign_patient_folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-csv", required=True, type=Path)
    parser.add_argument("--vital-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--patient-id-column", default="caseid")
    parser.add_argument("--expected-patients", type=int, help="Require this many distinct patients in the candidate cohort.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assign_patient_folds(
        args.clinical_csv,
        args.vital_dir,
        args.output_dir,
        args.n_splits,
        args.seed,
        args.patient_id_column,
        args.expected_patients,
    )


if __name__ == "__main__":
    main()
