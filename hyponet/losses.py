"""Loss functions used by HyPo-Net."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProbabilityFittingLoss(nn.Module):
    """Hybrid loss for soft probability fitting.

    The manuscript setting uses a soft target probability ``P_i`` and a binary
    hard label ``y_i``. ``alpha=1`` corresponds to pure Bernoulli KL divergence
    between the soft target and the predicted probability. Smaller values mix
    in BCE against the hard label.
    """

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        soft_targets: torch.Tensor,
        hard_targets: torch.Tensor,
    ) -> torch.Tensor:
        soft_targets = soft_targets.float().view_as(logits)
        hard_targets = hard_targets.float().view_as(logits)
        prob = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        soft = soft_targets.clamp(self.eps, 1.0 - self.eps)

        kl = soft * (torch.log(soft) - torch.log(prob))
        kl = kl + (1.0 - soft) * (torch.log(1.0 - soft) - torch.log(1.0 - prob))
        kl = kl.mean()

        if self.alpha >= 1.0:
            return kl
        bce = self.bce(logits, hard_targets)
        return self.alpha * kl + (1.0 - self.alpha) * bce
