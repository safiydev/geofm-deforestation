"""Focal Loss implementation for severe class imbalance in deforestation detection.

Down-weights easy background pixels (vast stretches of intact forest) to concentrate
gradients on rare, subtle deforestation events and logging road incursions.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Numerically stable Binary Focal Loss operating on raw logits."""

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Weighting factor for the rare positive deforestation class (0.0 to 1.0).
            gamma: Focusing parameter gamma >= 0. Higher values penalize easy examples.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted raw logits of shape (B, 1, H, W) or (B, H, W).
            targets: Binary ground truth labels (0 or 1) of matching shape.

        Returns:
            Computed focal loss.
        """
        targets = targets.float().view_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
