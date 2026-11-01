"""Tversky Loss for optimizing recall under extreme class imbalance.

Allows asymmetric weighting of False Negatives vs False Positives (alpha > beta)
to prioritize early warning alert sensitivity.
"""

import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    """Tversky index loss operating on predicted probabilities or logits."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1e-6,
        from_logits: bool = True,
    ):
        """
        Args:
            alpha: Weight assigned to False Negatives (missed deforestation).
            beta: Weight assigned to False Positives (false alerts).
                   alpha + beta should typically equal 1.0.
            smooth: Smoothing constant to avoid division by zero.
            from_logits: Whether input is raw logits or probabilities in [0, 1].
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits

    def compute_tversky_index(
        self, probs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Tversky Index scalar or per-sample value."""
        probs = probs.view(-1)
        targets = targets.view(-1)

        tp = (probs * targets).sum()
        fn = ((1.0 - probs) * targets).sum()
        fp = (probs * (1.0 - targets)).sum()

        ti = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return ti

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().view_as(inputs)
        probs = torch.sigmoid(inputs) if self.from_logits else inputs
        ti = self.compute_tversky_index(probs, targets)
        return 1.0 - ti
