"""Train and evaluate five-minute MAP/trend logistic monitoring references.

For each fold, train the six MAP-only, trend-only, and MAP-plus-trend models
with hard or soft targets from ``foldN/train`` only. Score the same 20-second
validation and test decision grid, fit alarm thresholds on validation only,
and report test performance using the shared monitoring evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from contextlib import ExitStack
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hyponet.monitoring import evaluate_monitoring
from hyponet.preprocessing.segmentation import load_case_list, load_vital_waveforms
from hyponet.simple_references import (
    DECISION_COLUMNS,
    EVENT_COLUMNS,
    fit_all_references,
    load_training_pressure_cues,
    score_case_references,
)


def _case_id_from_path(path: Path) -> str:
    match = re.match(r"^(\d+)", path.stem)
    if not match:
        raise ValueError(f"Recording filename must begin with a numeric case ID: {path}")
    return f"{int(match.group(1)):04d}"


def _case_id_from_value(value: str) -> str:
    try:
        numeric = Decimal(value)
        if not numeric.is_finite() or numeric != numeric.to_integral_value():
            raise InvalidOperation
        number = int(numeric)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"Invalid numeric case ID {value!r}") from error
    if number < 0:
        raise ValueError(f"Invalid numeric case ID {value!r}")
    return f"{number:04d}"


def _recordings_by_case(input_dir: Path) -> dict[str, Path]:
    recordings = {}
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() == ".vital":
            case_id = _case_id_from_path(path)
            if case_id in recordings:
                raise ValueError(f"Multiple .vital files have case ID {case_id}")
            recordings[case_id] = path
    return recordings


def _patient_map(path: Path | None, patient_id_column: str, selected_cases: set[str]) -> dict[str, str]:
    if path is None:
        return {case_id: case_id for case_id in selected_cases}
    clinical = pd.read_csv(path, dtype=str)
    case_column = next((name for name in ("caseid", "case_id") if name in clinical), None)
    if case_column is None or patient_id_column not in clinical:
        raise ValueError(f"Clinical CSV requires a caseid/case_id and {patient_id_column!r} column")
    mapping = {}
    for case_raw, patient_raw in zip(clinical[case_column], clinical[patient_id_column]):
        if pd.isna(case_raw) or pd.isna(patient_raw) or not str(patient_raw).strip():
            continue
        case_id = _case_id_from_value(str(case_raw).strip())
        if case_id in mapping and mapping[case_id] != patient_raw:
            raise ValueError(f"Case {case_id} has conflicting patient IDs")
        mapping[case_id] = str(patient_raw)
    if missing := selected_cases - set(mapping):
        raise ValueError(f"Missing patient IDs for cases: {sorted(missing)}")
    return mapping


def _selected_cases(splits_dir: Path, fold: int) -> dict[str, set[str]]:
    selected = {}
    for split in ("train", "val", "test"):
        manifest = splits_dir / f"fold{fold}_{split}_cases.txt"
        selected[split] = {_case_id_from_value(value) for value in load_case_list(manifest)}
        if not selected[split]:
            raise ValueError(f"Empty {split} manifest: {manifest}")
    if any(selected[a] & selected[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Fold train, val, and test case manifests must be disjoint")
    return selected


def _assert_matched_decisions(reference_path: Path, created_path: Path) -> None:
    """Optional exact eligibility-grid check against HyPoNet decisions."""

    fields = ["split", "case_id", "time_s", "ongoing", "input_complete", "future_complete"]
    existing = pd.read_csv(reference_path, dtype={"case_id": str})
    created = pd.read_csv(created_path, dtype={"case_id": str})
    for frame in (existing, created):
        if missing := set(fields) - set(frame):
            raise ValueError(f"Decision file lacks columns: {sorted(missing)}")
        frame["case_id"] = frame["case_id"].map(lambda value: _case_id_from_value(str(value)))
    existing = existing[fields].sort_values(fields[:3]).reset_index(drop=True)
    created = created[fields].sort_values(fields[:3]).reset_index(drop=True)
    pd.testing.assert_frame_equal(existing, created, check_dtype=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Raw .vital recordings.")
    parser.add_argument("--splits-dir", required=True, type=Path, help="Patient manifests from assign_patients.py.")
    parser.add_argument("--fold-dir", required=True, type=Path, help="One HDF5 fold directory containing train/ART.h5 and labels.")
    parser.add_argument("--fold", required=True, type=int, help="One-based fold number for the manifests.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clinical-csv", type=Path, help="Optional case-to-patient mapping for cluster analysis.")
    parser.add_argument("--patient-id-column", default="caseid")
    parser.add_argument("--matched-decisions", type=Path, help="Optional HyPoNet decisions.csv to verify identical val/test eligibility rows.")
    parser.add_argument("--sample-rate", type=int, default=100)
    parser.add_argument("--training-batch-size", type=int, default=128)
    parser.add_argument("--l2", type=float, default=0.0, help="Optional L2 penalty on slopes; default unpenalized.")
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.5], help="Validation false-alarm budgets per hour.")
    args = parser.parse_args()
    if args.fold < 1 or args.sample_rate <= 0 or args.training_batch_size <= 0:
        parser.error("fold, sample-rate, and training-batch-size must be positive")

    selected = _selected_cases(args.splits_dir, args.fold)
    patient_ids = _patient_map(args.clinical_csv, args.patient_id_column, set().union(*selected.values()))
    val_patients = {patient_ids[case] for case in selected["val"]}
    test_patients = {patient_ids[case] for case in selected["test"]}
    train_patients = {patient_ids[case] for case in selected["train"]}
    if (train_patients & val_patients) or (train_patients & test_patients) or (val_patients & test_patients):
        raise ValueError("Fold manifests share patient IDs across train, val, or test")

    recordings = _recordings_by_case(args.input_dir)
    if missing := (selected["val"] | selected["test"]) - set(recordings):
        raise FileNotFoundError(f"Missing validation/test .vital recordings: {sorted(missing)}")
    cues, labels = load_training_pressure_cues(
        args.fold_dir / "train", sample_rate=args.sample_rate,
        batch_size=args.training_batch_size, allowed_case_ids=selected["train"],
    )
    models = fit_all_references(cues, labels, l2=args.l2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models.json").write_text(
        json.dumps({
            "fold": args.fold,
            "training_segments": len(cues),
            "sample_rate": args.sample_rate,
            "horizon_minutes": 5,
            "step_seconds": 20,
            "optimizer": "L-BFGS-B",
            "training_objective": "mean Bernoulli cross-entropy plus optional half-L2 slope penalty",
            "models": {key: model.to_dict() for key, model in models.items()},
        }, indent=2) + "\n", encoding="utf-8",
    )

    event_path = args.output_dir / "events.csv"
    with ExitStack() as stack:
        event_file = stack.enter_context(event_path.open("w", encoding="utf-8", newline=""))
        event_writer = csv.DictWriter(event_file, fieldnames=EVENT_COLUMNS)
        event_writer.writeheader()
        decision_writers = {}
        decision_paths = {}
        for key in models:
            path = args.output_dir / f"{key}_decisions.csv"
            decision_paths[key] = path
            output = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
            writer = csv.DictWriter(output, fieldnames=DECISION_COLUMNS)
            writer.writeheader()
            decision_writers[key] = writer

        for split in ("val", "test"):
            for case_id in sorted(selected[split]):
                art, ecg, pleth = load_vital_waveforms(recordings[case_id], args.sample_rate)
                decisions, events = score_case_references(
                    art, ecg, pleth, models, split=split, case_id=case_id,
                    patient_id=patient_ids[case_id], sample_rate=args.sample_rate,
                )
                event_writer.writerows(events)
                for key, writer in decision_writers.items():
                    writer.writerows(decisions[key])
                print(f"{split} {case_id}: {len(decisions[next(iter(models))])} decisions, {len(events)} events", flush=True)

    if args.matched_decisions:
        _assert_matched_decisions(args.matched_decisions, next(iter(decision_paths.values())))
    events = pd.read_csv(event_path, dtype={"case_id": str, "patient_id": str})
    results = []
    for key, path in decision_paths.items():
        decisions = pd.read_csv(path, dtype={"case_id": str, "patient_id": str})
        metric = evaluate_monitoring(decisions, events, budgets=args.budgets)
        metric.insert(0, "supervision", models[key].supervision)
        metric.insert(0, "model", models[key].name)
        metric.insert(0, "fold", args.fold)
        results.append(metric)
    results = pd.concat(results, ignore_index=True)
    results.to_csv(args.output_dir / "monitoring_results.csv", index=False)
    print(results.to_string(index=False))
    print(f"Saved models and monitoring results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
