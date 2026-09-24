"""Loss functions used for the manuscript's supervision comparison."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbabilityFittingLoss(nn.Module):
    """Mean Bernoulli KL divergence from soft targets to predicted logits.

    The target entropy is retained so the reported loss is KL rather than the
    equivalent soft-target cross-entropy. Hard labels are not part of this
    objective.
    """

    def forward(self, logits: torch.Tensor, soft_label: torch.Tensor) -> torch.Tensor:
        target = soft_label.float().view_as(logits)
        if not torch.isfinite(target).all() or torch.any((target < 0) | (target > 1)):
            raise ValueError("Soft targets must be finite values in [0, 1].")

        cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        # 0 log 0 is zero. Clamp only the logarithm to preserve that identity.
        complement = 1.0 - target
        entropy_terms = target * torch.log(target.clamp_min(1e-12))
        entropy_terms = entropy_terms + complement * torch.log(complement.clamp_min(1e-12))
        return (cross_entropy + entropy_terms).mean()
