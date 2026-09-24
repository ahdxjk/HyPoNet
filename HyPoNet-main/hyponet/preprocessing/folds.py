"""Build patient-level five-fold HDF5 datasets from per-case NumPy files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


SIGNALS = [
    "ART",
    "ECG",
    "PLETH",
    "STATIC",
    "HARD_LABELS",
    "LABELS_CONCAVE",
]


EXPECTED_FEATURES = {
    "ART": 6000,
    "ECG": 6000,
    "PLETH": 6000,
    "STATIC": 1,
    "HARD_LABELS": 1,
    "LABELS_CONCAVE": 1,
}


def available_case_ids(data_dir: Path, suffix: str = ".npy") -> set[str]:
    """Return case IDs found in raw recordings or generated segment files."""

    ids = set()
    for path in data_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != suffix.lower():
            continue
        match = re.match(r"(\d+)", path.stem)
        if match:
            ids.add(match.group(1).zfill(4))
    return ids


def load_clinical_table(
    clinical_csv: Path,
    cohort_dir: Path,
    source_suffix: str = ".vital",
    patient_id_column: str = "caseid",
) -> pd.DataFrame:
    """Load the candidate cohort before horizon-specific segment eligibility."""

    clinical = pd.read_csv(clinical_csv)
    if patient_id_column not in clinical.columns:
        raise ValueError(f"Clinical table lacks patient ID column {patient_id_column!r}.")
    clinical["age"] = pd.to_numeric(clinical["age"], errors="coerce")
    clinical = clinical.dropna(subset=["age", patient_id_column, "caseid"]).copy()
    numeric_case_ids = pd.to_numeric(clinical["caseid"], errors="coerce")
    if numeric_case_ids.isna().any() or (numeric_case_ids < 0).any() or (numeric_case_ids % 1 != 0).any():
        raise ValueError("Clinical caseid values must be nonnegative whole numbers.")
    clinical["caseid"] = numeric_case_ids.astype("int64").astype(str).str.zfill(4)
    clinical["patient_id"] = clinical[patient_id_column].astype(str)
    present = available_case_ids(cohort_dir, source_suffix)
    clinical = clinical[clinical["caseid"].isin(present)].copy()
    if clinical.empty:
        raise ValueError("No candidate cases matched the clinical table and raw recordings.")

    max_age = int(clinical["age"].max()) + 1
    bins = list(range(0, max_age + 10, 10))
    labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]
    clinical["age_group"] = pd.cut(clinical["age"], bins=bins, labels=labels, right=False)
    clinical = clinical.dropna(subset=["age_group"]).copy()
    return clinical


def create_patient_level_folds(clinical: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> Dict[int, Dict[str, List[str]]]:
    """Create patient-disjoint 80/10/10 runs, stratified by age group."""

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: Dict[int, Dict[str, List[str]]] = {}
    patients = clinical.drop_duplicates("patient_id").reset_index(drop=True)
    x = patients["patient_id"]
    y = patients["age_group"]

    for fold_idx, (train_idx, holdout_idx) in enumerate(skf.split(x, y), start=1):
        train_patients = set(patients.iloc[train_idx]["patient_id"])
        holdout = patients.iloc[holdout_idx]
        stratify = holdout["age_group"] if holdout["age_group"].value_counts().min() >= 2 else None
        val, test = train_test_split(holdout, test_size=0.5, random_state=seed, stratify=stratify)
        patient_sets = {
            "train": train_patients,
            "val": set(val["patient_id"]),
            "test": set(test["patient_id"]),
        }
        folds[fold_idx] = {
            split: clinical.loc[clinical["patient_id"].isin(ids), "caseid"].tolist()
            for split, ids in patient_sets.items()
        }

    return folds


def write_fold_manifests(folds: Dict[int, Dict[str, List[str]]], manifest_dir: Path) -> None:
    """Persist patient assignment before any waveform segments are created."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for fold_idx, splits in folds.items():
        for split_name, case_ids in splits.items():
            (manifest_dir / f"fold{fold_idx}_{split_name}_cases.txt").write_text(
                "\n".join(sorted(case_ids)) + "\n", encoding="utf-8"
            )
    all_ids = sorted({case_id for splits in folds.values() for cases in splits.values() for case_id in cases})
    (manifest_dir / "all_cases.txt").write_text("\n".join(all_ids) + "\n", encoding="utf-8")


def read_fold_manifests(manifest_dir: Path, n_splits: int = 5) -> Dict[int, Dict[str, List[str]]]:
    """Read and validate the fixed assignments for HDF5 aggregation."""
    all_cases_path = manifest_dir / "all_cases.txt"
    all_cases = {line.strip() for line in all_cases_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not all_cases:
        raise ValueError(f"No candidate case IDs in {all_cases_path}.")
    folds: Dict[int, Dict[str, List[str]]] = {}
    for fold_idx in range(1, n_splits + 1):
        splits = {}
        for split_name in ("train", "val", "test"):
            path = manifest_dir / f"fold{fold_idx}_{split_name}_cases.txt"
            splits[split_name] = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(set(splits[a]) & set(splits[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"Case overlap in fold {fold_idx} manifests.")
        if set().union(*map(set, splits.values())) != all_cases:
            raise ValueError(f"Fold {fold_idx} does not cover exactly the candidate cohort.")
        folds[fold_idx] = splits
    return folds


def assign_patient_folds(
    clinical_csv: Path,
    vital_dir: Path,
    manifest_dir: Path,
    n_splits: int = 5,
    seed: int = 42,
    patient_id_column: str = "caseid",
    expected_patients: int | None = None,
) -> None:
    """Fix patient-level partitions before extracting any signal segments."""
    clinical = load_clinical_table(clinical_csv, vital_dir, ".vital", patient_id_column)
    if clinical["caseid"].duplicated().any():
        raise ValueError("Clinical table contains duplicate case IDs.")
    actual_patients = clinical["patient_id"].nunique()
    if expected_patients is not None and actual_patients != expected_patients:
        raise ValueError(
            f"Candidate cohort has {actual_patients} patients; expected {expected_patients}. "
            "Check the manuscript cohort selection before assigning folds."
        )
    folds = create_patient_level_folds(clinical, n_splits=n_splits, seed=seed)
    write_fold_manifests(folds, manifest_dir)


def fill_missing_values(data: np.ndarray, signal: str) -> np.ndarray:
    """Interpolate missing samples within each segment, never across cases."""

    if signal not in {"ART", "ECG", "PLETH"} or data.size == 0:
        return data
    filled = np.asarray(data).copy()
    sample_indices = np.arange(filled.shape[1])
    for row_index, row in enumerate(filled):
        finite = np.isfinite(row)
        if finite.all():
            continue
        if not finite.any():
            raise ValueError(f"{signal} segment {row_index} contains no finite samples.")
        filled[row_index, ~finite] = np.interp(sample_indices[~finite], sample_indices[finite], row[finite])
    return filled


def load_case_signal(data_dir: Path, case_id: str, signal: str) -> np.ndarray:
    path = data_dir / f"{case_id}_{signal}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    if signal in {"STATIC", "HARD_LABELS"} or signal.startswith("LABELS_"):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
    else:
        expected = EXPECTED_FEATURES[signal]
        if data.size == 0:
            return np.empty((0, expected), dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != expected:
            raise ValueError(f"{path} expected shape [N, {expected}], got {data.shape}")
    return data


def merge_split_to_hdf5(data_dir: Path, case_ids: Iterable[str], output_dir: Path, signals: Iterable[str] = SIGNALS) -> None:
    """Merge selected case-level NumPy files into one HDF5 file per signal."""

    output_dir.mkdir(parents=True, exist_ok=True)
    all_data = {signal: [] for signal in signals}

    for case_id in sorted(case_ids):
        case_data = {}
        lengths = []
        for signal in signals:
            data = load_case_signal(data_dir, case_id, signal)
            case_data[signal] = data
            lengths.append(data.shape[0])

        if len(set(lengths)) != 1:
            raise ValueError(f"Case {case_id} has inconsistent sample counts {lengths}.")
        if lengths[0] == 0:
            continue

        for signal in signals:
            all_data[signal].append(case_data[signal])

    for signal, chunks in all_data.items():
        if not chunks:
            raise ValueError(f"No eligible segments for {signal} in {output_dir}.")
        merged = np.concatenate(chunks, axis=0)
        merged = fill_missing_values(merged, signal)
        with h5py.File(output_dir / f"{signal}.h5", "w") as h5f:
            h5f.create_dataset(signal, data=merged, dtype=merged.dtype)


def build_five_fold_dataset(
    data_dir: Path,
    manifest_dir: Path,
    output_root: Path,
    n_splits: int = 5,
    signals: Iterable[str] = SIGNALS,
) -> None:
    """Create HDF5 splits from assignments made before segment generation."""

    folds = read_fold_manifests(manifest_dir, n_splits=n_splits)

    output_root.mkdir(parents=True, exist_ok=True)
    for fold_idx, split_cases in folds.items():
        for split_name, case_ids in split_cases.items():
            merge_split_to_hdf5(data_dir, case_ids, output_root / f"fold{fold_idx}" / split_name, signals)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build patient-level HDF5 folds for HyPo-Net.")
    parser.add_argument("--segments-dir", required=True, type=Path, help="Directory containing per-case .npy segment files.")
    parser.add_argument("--splits-dir", required=True, type=Path, help="Patient fold manifests created before segmentation.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output root for fold directories.")
    parser.add_argument("--n-splits", type=int, default=5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    build_five_fold_dataset(args.segments_dir, args.splits_dir, args.output_root, args.n_splits)


if __name__ == "__main__":
    main()
