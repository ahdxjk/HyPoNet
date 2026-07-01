"""HDF5 data loading utilities for HyPo-Net.

Expected split layout:

    fold1/
      train/
        ART.h5, ECG.h5, PLETH.h5
        extracted_features.h5
        HARD_LABELS.h5
        LABELS_CONCAVE.h5
      val/
      test/
"""

from __future__ import annotations

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


def read_h5_array(path: Path, key: str | None = None):
    with h5py.File(path, "r") as f:
        if key is None:
            key = next(iter(f.keys()))
        return f[key][:]


class HypotensionH5Dataset(Dataset):
    """Dataset for 60-s waveform segments and soft labels."""

    def __init__(
        self,
        split_dir,
        soft_label_file: str = "LABELS_CONCAVE.h5",
        soft_label_key: str = "LABELS_CONCAVE",
    ):
        split_dir = Path(split_dir)
        self.art = torch.tensor(read_h5_array(split_dir / "ART.h5", "ART"), dtype=torch.float32)
        self.ecg = torch.tensor(read_h5_array(split_dir / "ECG.h5", "ECG"), dtype=torch.float32)
        self.pleth = torch.tensor(read_h5_array(split_dir / "PLETH.h5", "PLETH"), dtype=torch.float32)
        self.features = torch.tensor(read_h5_array(split_dir / "extracted_features.h5"), dtype=torch.float32)
        self.hard = torch.tensor(read_h5_array(split_dir / "HARD_LABELS.h5", "HARD_LABELS"), dtype=torch.float32).view(-1)
        self.soft = torch.tensor(read_h5_array(split_dir / soft_label_file, soft_label_key), dtype=torch.float32).view(-1)

        n = len(self.hard)
        for name, value in {
            "ART": self.art,
            "ECG": self.ecg,
            "PLETH": self.pleth,
            "features": self.features,
            "soft labels": self.soft,
        }.items():
            if len(value) != n:
                raise ValueError(f"{name} has {len(value)} samples, expected {n}.")

    def __len__(self):
        return len(self.hard)

    def __getitem__(self, index):
        signals = torch.stack(
            [self.art[index], self.ecg[index], self.pleth[index]],
            dim=0,
        )
        return {
            "signals": signals,
            "features": self.features[index],
            "hard_label": self.hard[index],
            "soft_label": self.soft[index],
        }
