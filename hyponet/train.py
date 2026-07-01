"""Training and evaluation helpers."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import HypotensionH5Dataset
from .losses import ProbabilityFittingLoss
from .metrics import binary_probability_metrics
from .models import HyPoNet


def build_model(device, wp_level: int = 3):
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


def run_epoch(model, loader, criterion, optimizer=None, device="cuda"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_batches = 0
    probs, hard, soft = [], [], []

    for batch in loader:
        signals = batch["signals"].to(device)
        features = batch["features"].to(device)
        hard_label = batch["hard_label"].to(device).view(-1, 1)
        soft_label = batch["soft_label"].to(device).view(-1, 1)

        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(signals, features)
        loss = criterion(logits, soft_label, hard_label)
        if training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        hard.append(hard_label.detach().cpu().numpy())
        soft.append(soft_label.detach().cpu().numpy())

    y_prob = np.concatenate(probs).reshape(-1)
    y_true = np.concatenate(hard).reshape(-1)
    y_soft = np.concatenate(soft).reshape(-1)
    metrics = binary_probability_metrics(y_true, y_prob, y_soft)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def fit_fold(
    fold_dir,
    output_dir,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 2e-5,
    alpha: float = 1.0,
    patience: int = 10,
    wp_level: int = 3,
    num_workers: int = 0,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fold_dir = Path(fold_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = HypotensionH5Dataset(fold_dir / "train")
    val_ds = HypotensionH5Dataset(fold_dir / "val")
    test_ds = HypotensionH5Dataset(fold_dir / "test")
    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": device.startswith("cuda")}
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    model = build_model(device, wp_level=wp_level)
    criterion = ProbabilityFittingLoss(alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_score = -np.inf
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()

        score = val_metrics["recall"]
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        test_metrics = run_epoch(model, test_loader, criterion, None, device)

    torch.save({"model_state_dict": model.state_dict(), "metrics": test_metrics}, output_dir / "best_model.pt")
    return test_metrics
