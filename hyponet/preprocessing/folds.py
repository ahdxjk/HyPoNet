"""Build patient-level five-fold HDF5 datasets from per-case NumPy files."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.model_selection import StratifiedKFold, train_test_split


SIGNALS = [
    "ART",
    "ECG",
    "PLETH",
    "STATIC",
    "HARD_LABELS",
    "LABELS_LINEAR",
    "LABELS_CONCAVE",
    "LABELS_CONVEX",
    "LABELS_SIGMOID",
    "LABELS_SINE",
    "LABELS_COSINE",
    "LABELS_REVERSE_Z_131",
    "LABELS_REVERSE_Z_212",
]


EXPECTED_FEATURES = {
    "ART": 6000,
    "ECG": 6000,
    "PLETH": 6000,
    "STATIC": 1,
    "HARD_LABELS": 1,
    "LABELS_LINEAR": 1,
    "LABELS_CONCAVE": 1,
    "LABELS_CONVEX": 1,
    "LABELS_SIGMOID": 1,
    "LABELS_SINE": 1,
    "LABELS_COSINE": 1,
    "LABELS_REVERSE_Z_131": 1,
    "LABELS_REVERSE_Z_212": 1,
}


def available_case_ids(data_dir: Path) -> set[str]:
    """Return case IDs that have at least one generated NumPy file."""

    ids = set()
    for path in data_dir.glob("*.npy"):
        match = re.match(r"(\d+)_", path.name)
        if match:
            ids.add(match.group(1).zfill(4))
    return ids


def load_clinical_table(clinical_csv: Path, data_dir: Path) -> pd.DataFrame:
    """Load clinical metadata and keep only cases present in ``data_dir``."""

    clinical = pd.read_csv(clinical_csv)
    clinical["age"] = pd.to_numeric(clinical["age"], errors="coerce")
    clinical = clinical.dropna(subset=["age"]).copy()
    clinical["caseid"] = clinical["caseid"].astype(str).str.zfill(4)
    present = available_case_ids(data_dir)
    clinical = clinical[clinical["caseid"].isin(present)].copy()

    max_age = int(clinical["age"].max()) + 1
    bins = list(range(0, max_age + 10, 10))
    labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]
    clinical["age_group"] = pd.cut(clinical["age"], bins=bins, labels=labels, right=False)
    clinical = clinical.dropna(subset=["age_group"]).copy()
    return clinical


def create_patient_level_folds(clinical: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> Dict[int, Dict[str, List[str]]]:
    """Create 8:1:1 train/validation/test folds stratified by age group."""

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: Dict[int, Dict[str, List[str]]] = {}
    x = clinical["caseid"]
    y = clinical["age_group"]

    for fold_idx, (train_idx, holdout_idx) in enumerate(skf.split(x, y), start=1):
        train_cases = clinical.iloc[train_idx]["caseid"].astype(str).tolist()
        holdout = clinical.iloc[holdout_idx]
        stratify = holdout["age_group"] if holdout["age_group"].value_counts().min() >= 2 else None
        val, test = train_test_split(holdout, test_size=0.5, random_state=seed, stratify=stratify)
        folds[fold_idx] = {
            "train": train_cases,
            "val": val["caseid"].astype(str).tolist(),
            "test": test["caseid"].astype(str).tolist(),
        }

    return folds


def fill_missing_values(data: np.ndarray, signal: str) -> np.ndarray:
    """Fill missing waveform rows by linear interpolation across samples."""

    if signal not in {"ART", "ECG", "PLETH"} or data.size == 0:
        return data
    non_nan_mask = ~np.isnan(data).any(axis=1)
    if not non_nan_mask.any():
        return np.zeros_like(data)
    if non_nan_mask.all():
        return data

    x_valid = np.where(non_nan_mask)[0]
    y_valid = data[non_nan_mask]
    filled = np.zeros_like(data)
    for column in range(data.shape[1]):
        f = interpolate.interp1d(x_valid, y_valid[:, column], kind="linear", bounds_error=False, fill_value="extrapolate")
        filled[:, column] = f(np.arange(len(data)))
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
        try:
            for signal in signals:
                data = load_case_signal(data_dir, case_id, signal)
                case_data[signal] = data
                lengths.append(data.shape[0])
        except FileNotFoundError:
            continue

        if len(set(lengths)) != 1:
            print(f"Skipping case {case_id}: inconsistent sample counts {lengths}")
            continue

        for signal in signals:
            all_data[signal].append(case_data[signal])

    for signal, chunks in all_data.items():
        if not chunks:
            print(f"No data collected for {signal} in {output_dir}")
            continue
        merged = np.concatenate(chunks, axis=0)
        merged = fill_missing_values(merged, signal)
        with h5py.File(output_dir / f"{signal}.h5", "w") as h5f:
            h5f.create_dataset(signal, data=merged, dtype=merged.dtype)


def build_five_fold_dataset(
    data_dir: Path,
    clinical_csv: Path,
    output_root: Path,
    n_splits: int = 5,
    seed: int = 42,
    signals: Iterable[str] = SIGNALS,
) -> None:
    """Create fold1...fold5/train|val|test directories with HDF5 files."""

    clinical = load_clinical_table(clinical_csv, data_dir)
    folds = create_patient_level_folds(clinical, n_splits=n_splits, seed=seed)

    output_root.mkdir(parents=True, exist_ok=True)
    for fold_idx, split_cases in folds.items():
        for split_name, case_ids in split_cases.items():
            merge_split_to_hdf5(data_dir, case_ids, output_root / f"fold{fold_idx}" / split_name, signals)

        for split_name, case_ids in split_cases.items():
            with open(output_root / f"fold{fold_idx}_{split_name}_cases.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(case_ids))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build patient-level HDF5 folds for HyPo-Net.")
    parser.add_argument("--segments-dir", required=True, type=Path, help="Directory containing per-case .npy segment files.")
    parser.add_argument("--clinical-csv", required=True, type=Path, help="CSV with at least caseid and age columns.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output root for fold directories.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    build_five_fold_dataset(args.segments_dir, args.clinical_csv, args.output_root, args.n_splits, args.seed)


if __name__ == "__main__":
    main()
