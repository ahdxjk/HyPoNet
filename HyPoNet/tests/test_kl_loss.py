"""Checks for the paper's KL-only supervision objective."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from hyponet.losses import ProbabilityFittingLoss
from hyponet.metrics import binary_probability_metrics


class KLLossTests(unittest.TestCase):
    def test_bernoulli_kl_with_boundary_targets_and_gradient(self):
        logits = torch.tensor([[0.0], [0.0], [0.0]], requires_grad=True)
        targets = torch.tensor([[0.0], [0.25], [1.0]])
        loss = ProbabilityFittingLoss()(logits, targets)
        expected = F.binary_cross_entropy_with_logits(logits, targets)
        expected += (0.25 * torch.log(torch.tensor(0.25)) + 0.75 * torch.log(torch.tensor(0.75))) / 3
        self.assertTrue(torch.allclose(loss, expected, atol=1e-7))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        reported = binary_probability_metrics(
            [0, 1, 0], torch.sigmoid(logits.detach()).numpy(), targets.numpy()
        )["kl_soft"]
        self.assertAlmostEqual(reported, loss.item(), places=6)


if __name__ == "__main__":
    unittest.main()
