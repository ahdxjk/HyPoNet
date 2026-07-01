"""Five-fold HyPo-Net training utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import HyPoNetH5Dataset
from .losses import ProbabilityFittingLoss
from .metrics import binary_probability_metrics
from .models import HyPoNet


def build_model(device: torch.device, wp_level: int = 3) -> torch.nn.Module:
    model = HyPoNet(
        seq_len=6000,
        patch_len=64,
        stride=32,
        d_model=1024,
        n_heads=8,
        dropout=0.0,
        wp_level=wp_level,
        input_channels=3,
    )
    return model.to(device)


def run_epoch(model, loader, criterion, optimizer=None, device=None):
    training = optimizer is not None
    model.train(training)
    losses, probs, hard, soft = [], [], [], []

    for batch in loader:
        signals = batch["signals"].to(device)
        features = batch["features"].to(device)
        hard_label = batch["hard_label"].to(device)
        soft_label = batch["soft_label"].to(device)

        if training:
            optimizer.zero_grad()
        logits = model(signals, features)
        loss = criterion(logits, soft_label, hard_label)
        if training:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        hard.append(hard_label.detach().cpu().numpy())
        soft.append(soft_label.detach().cpu().numpy())

    y_prob = np.concatenate(probs).reshape(-1)
    y_hard = np.concatenate(hard).reshape(-1)
    y_soft = np.concatenate(soft).reshape(-1)
    metrics = binary_probability_metrics(y_hard, y_prob, y_soft)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def fit_fold(
    fold_dir: Path,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 2e-5,
    alpha: float = 1.0,
    patience: int = 10,
    wp_level: int = 3,
    num_workers: int = 0,
    device: str | None = None,
) -> dict[str, float]:
    device_obj = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = HyPoNetH5Dataset(fold_dir / "train")
    val_ds = HyPoNetH5Dataset(fold_dir / "val")
    test_ds = HyPoNetH5Dataset(fold_dir / "test")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    model = build_model(device_obj, wp_level=wp_level)
    criterion = ProbabilityFittingLoss(alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_recall = -np.inf
    best_path = output_dir / "best_model.pt"
    stale_epochs = 0

    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device_obj)
        val_metrics = run_epoch(model, val_loader, criterion, None, device_obj)
        print(
            f"epoch={epoch + 1} train_loss={train_metrics['loss']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} val_recall={val_metrics['recall']:.4f}",
            flush=True,
        )
        if val_metrics["recall"] > best_recall:
            best_recall = val_metrics["recall"]
            torch.save(model.state_dict(), best_path)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
        scheduler.step()

    model.load_state_dict(torch.load(best_path, map_location=device_obj))
    return run_epoch(model, test_loader, criterion, None, device_obj)


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
