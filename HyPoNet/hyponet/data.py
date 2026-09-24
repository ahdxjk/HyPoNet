"""HDF5 dataset loader for HyPo-Net training."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def read_h5_dataset(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as h5f:
        return h5f[key][:]


class HyPoNetH5Dataset(Dataset):
    """Dataset for one train/validation/test split.

    Required files:
    ART.h5, ECG.h5, PLETH.h5, HARD_LABELS.h5, LABELS_CONCAVE.h5,
    and extracted_features.h5.
    """

    def __init__(self, split_dir: str | Path, soft_label_key: str = "LABELS_CONCAVE"):
        self.split_dir = Path(split_dir)
        self.art = read_h5_dataset(self.split_dir / "ART.h5", "ART").astype(np.float32)
        self.ecg = read_h5_dataset(self.split_dir / "ECG.h5", "ECG").astype(np.float32)
        self.pleth = read_h5_dataset(self.split_dir / "PLETH.h5", "PLETH").astype(np.float32)
        self.features = read_h5_dataset(self.split_dir / "extracted_features.h5", "features").astype(np.float32)
        self.hard_labels = read_h5_dataset(self.split_dir / "HARD_LABELS.h5", "HARD_LABELS").astype(np.float32).reshape(-1, 1)
        self.soft_labels = read_h5_dataset(self.split_dir / f"{soft_label_key}.h5", soft_label_key).astype(np.float32).reshape(-1, 1)

        lengths = {len(self.art), len(self.ecg), len(self.pleth), len(self.features), len(self.hard_labels), len(self.soft_labels)}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent sample counts in {self.split_dir}: {lengths}")

    def __len__(self) -> int:
        return len(self.art)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        signals = np.stack([self.art[index], self.ecg[index], self.pleth[index]], axis=0)
        return {
            "signals": torch.from_numpy(signals).float(),
            "features": torch.from_numpy(self.features[index]).float(),
            "hard_label": torch.from_numpy(self.hard_labels[index]).float(),
            "soft_label": torch.from_numpy(self.soft_labels[index]).float(),
        }
