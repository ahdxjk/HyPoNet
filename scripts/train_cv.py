"""Run HyPo-Net five-fold training."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hyponet.train import fit_fold, write_rows


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HyPo-Net on patient-level folds.")
    parser.add_argument("--data-root", required=True, type=Path, help="Root containing fold1...fold5.")
    parser.add_argument("--output-dir", default=Path("outputs/hyponet_cv"), type=Path)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--wp-level", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold in args.folds:
        print(f"\n===== fold {fold} =====", flush=True)
        metrics = fit_fold(
            args.data_root / f"fold{fold}",
            args.output_dir / f"fold{fold}",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            alpha=args.alpha,
            patience=args.patience,
            wp_level=args.wp_level,
            num_workers=args.num_workers,
        )
        metrics["fold"] = fold
        rows.append(metrics)

    write_rows(args.output_dir / "fold_results.csv", rows)
    summary_path = args.output_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        for key in rows[0]:
            if key == "fold":
                continue
            values = np.asarray([row[key] for row in rows], dtype=float)
            writer.writerow([key, np.nanmean(values), np.nanstd(values, ddof=1)])
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
