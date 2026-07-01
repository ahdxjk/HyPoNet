"""Loss functions for probability fitting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbabilityFittingLoss(nn.Module):
    """Hybrid loss used for soft probability fitting.

    alpha=1.0 reproduces the KL-only probability-fitting setting. Smaller
    alpha values mix in BCE against the hard binary label.
    """

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, soft_label: torch.Tensor, hard_label: torch.Tensor) -> torch.Tensor:
        soft_label = soft_label.float().view_as(logits)
        hard_label = hard_label.float().view_as(logits)
        prob = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        soft = soft_label.clamp(self.eps, 1.0 - self.eps)
        kl = soft * (torch.log(soft) - torch.log(prob)) + (1.0 - soft) * (
            torch.log(1.0 - soft) - torch.log(1.0 - prob)
        )
        kl = kl.mean()
        if self.alpha >= 1.0:
            return kl
        bce = self.bce(logits, hard_label)
        return self.alpha * kl + (1.0 - self.alpha) * bce
