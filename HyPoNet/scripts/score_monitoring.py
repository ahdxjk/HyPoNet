"""Score chronological 20-second decisions for one five-minute fold.

The output CSVs are consumed by ``scripts/evaluate_monitoring.py``. Decisions
include every complete 60-second grid window, including rows excluded from
alarm evaluation by the ongoing/input/future flags. Scores use observed inputs
only; episode timing is kept in separate retrospective event annotations.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hyponet.monitoring_data import DECISION_COLUMNS, EVENT_COLUMNS, score_case_arrays
from hyponet.preprocessing.handcrafted_features import TrainingFeaturePreprocessor
from hyponet.preprocessing.segmentation import load_case_list, load_vital_waveforms
from hyponet.train import build_model


def _case_id_from_path(path: Path) -> str:
    match = re.match(r"^(\d+)", path.stem)
    if not match:
        raise ValueError(f"Recording filename must begin with a numeric case ID: {path}")
    return f"{int(match.group(1)):04d}"


def _recordings_by_case(input_dir: Path) -> dict[str, Path]:
    recordings: dict[str, Path] = {}
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() != ".vital":
            continue
        case_id = _case_id_from_path(path)
        if case_id in recordings:
            raise ValueError(f"Multiple recordings have case ID {case_id}: {recordings[case_id]}, {path}")
        recordings[case_id] = path
    return recordings


def _patient_ids(clinical_csv: Path | None, patient_id_column: str) -> dict[str, str]:
    """Resolve case-to-patient IDs for patient-cluster comparisons."""

    if clinical_csv is None:
        return {}
    clinical = pd.read_csv(clinical_csv, dtype=str)
    if "caseid" not in clinical or patient_id_column not in clinical:
        raise ValueError(f"Clinical CSV needs caseid and {patient_id_column!r} columns")
    if clinical[["caseid", patient_id_column]].isna().any().any():
        raise ValueError("Clinical CSV has missing case or patient IDs")
    numeric_cases = pd.to_numeric(clinical["caseid"], errors="coerce")
    if numeric_cases.isna().any() or (numeric_cases < 0).any() or (numeric_cases % 1 != 0).any():
        raise ValueError("Clinical caseid values must be nonnegative whole numbers")
    mapping: dict[str, str] = {}
    for case, patient in zip(numeric_cases.astype("int64"), clinical[patient_id_column]):
        case_id = f"{int(case):04d}"
        patient = str(patient).strip()
        if not patient or (case_id in mapping and mapping[case_id] != patient):
            raise ValueError(f"Missing or conflicting patient ID for case {case_id}")
        mapping[case_id] = patient
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Raw VitalDB .vital recordings.")
    parser.add_argument("--splits-dir", required=True, type=Path, help="Patient manifests from assign_patients.py.")
    parser.add_argument("--fold", required=True, type=int, help="One-based fold number for the checkpoint and manifests.")
    parser.add_argument("--fold-dir", required=True, type=Path, help="HDF5 fold directory containing feature_preprocessor.npz.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained best_model.pt for the same fold.")
    parser.add_argument("--static-data", required=True, type=Path, help="Clinical NumPy array indexed by caseid - 1.")
    parser.add_argument("--clinical-csv", type=Path, help="Clinical table with patient IDs when one patient has multiple cases.")
    parser.add_argument("--patient-id-column", default="caseid", help="Patient ID field in --clinical-csv; default caseid.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--feature-preprocessor", type=Path, help="Override the training-fold .npz statistics path.")
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=("val", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=100)
    parser.add_argument("--dwt-level", type=int, default=3)
    parser.add_argument("--device", help="PyTorch device; defaults to CUDA if available, else CPU.")
    args = parser.parse_args()

    if args.fold < 1:
        parser.error("--fold must be positive")
    if args.batch_size < 1 or args.sample_rate < 1:
        parser.error("--batch-size and --sample-rate must be positive")
    if len(set(args.splits)) != len(args.splits):
        parser.error("--splits must not contain duplicates")

    selected: dict[str, set[str]] = {}
    for split in args.splits:
        manifest = args.splits_dir / f"fold{args.fold}_{split}_cases.txt"
        selected[split] = {f"{int(case_id):04d}" for case_id in load_case_list(manifest)}
    if len(selected) == 2 and selected["val"] & selected["test"]:
        raise ValueError("Validation and test manifests overlap")

    recordings = _recordings_by_case(args.input_dir)
    patient_ids = _patient_ids(args.clinical_csv, args.patient_id_column)
    missing = set().union(*selected.values()) - set(recordings)
    if missing:
        raise FileNotFoundError(f"No .vital recording for cases: {sorted(missing)}")
    if args.clinical_csv is not None:
        missing_patients = set().union(*selected.values()) - set(patient_ids)
        if missing_patients:
            raise ValueError(f"No patient ID for cases: {sorted(missing_patients)}")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = build_model(device, wp_level=args.dwt_level)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    stats_path = args.feature_preprocessor or args.fold_dir / "feature_preprocessor.npz"
    preprocessor = TrainingFeaturePreprocessor.load(stats_path)
    static_data = np.load(args.static_data, allow_pickle=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    decision_path = args.output_dir / "decisions.csv"
    event_path = args.output_dir / "events.csv"
    with decision_path.open("w", encoding="utf-8", newline="") as decision_file, \
         event_path.open("w", encoding="utf-8", newline="") as event_file:
        decision_writer = csv.DictWriter(decision_file, fieldnames=DECISION_COLUMNS)
        event_writer = csv.DictWriter(event_file, fieldnames=EVENT_COLUMNS)
        decision_writer.writeheader()
        event_writer.writeheader()

        for split in args.splits:
            for case_id in sorted(selected[split]):
                index = int(case_id) - 1
                if index < 0 or index >= len(static_data):
                    raise IndexError(f"Case {case_id} is outside the static array indexed by caseid - 1")
                art, ecg, pleth = load_vital_waveforms(recordings[case_id], args.sample_rate)
                decisions, events = score_case_arrays(
                    art, ecg, pleth, static_data[index], model, preprocessor,
                    split=split, case_id=case_id, device=device,
                    patient_id=patient_ids.get(case_id, case_id),
                    batch_size=args.batch_size, sample_rate=args.sample_rate,
                )
                decision_writer.writerows(decisions)
                event_writer.writerows(events)
                print(f"{split} {case_id}: {len(decisions)} decisions, {len(events)} events", flush=True)

    print(f"Saved {decision_path} and {event_path}", flush=True)


if __name__ == "__main__":
    main()
