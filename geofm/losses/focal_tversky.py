"""Focal Tversky Loss and Compound Deforestation Loss.

Applies exponential focusing to the Tversky Index to force the network to learn
hard-to-detect subtle disturbances such as narrow logging corridors and selective logging gaps.
"""

from typing import Optional
import torch
import torch.nn as nn
from geofm.losses.focal import BinaryFocalLoss
from geofm.losses.tversky import TverskyLoss


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss: (1 - TverskyIndex)^gamma."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.33,
        smooth: float = 1e-6,
        from_logits: bool = True,
    ):
        """
        Args:
            alpha: FN penalty weight.
            beta: FP penalty weight.
            gamma: Focusing exponent (typically 1.0 to 2.0).
            smooth: Numerical stabilizer.
            from_logits: Whether input is logits.
        """
        super().__init__()
        self.tversky = TverskyLoss(
            alpha=alpha, beta=beta, smooth=smooth, from_logits=from_logits
        )
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tl = self.tversky(inputs, targets)
        # Loss = (1 - TI)^gamma
        return torch.pow(tl, self.gamma)


class CompoundDeforestationLoss(nn.Module):
    """Production compound loss combining Binary Focal Loss and Focal Tversky Loss."""

    def __init__(
        self,
        focal_weight: float = 0.5,
        tversky_weight: float = 0.5,
        alpha: float = 0.75,
        beta: float = 0.25,
        gamma_focal: float = 2.0,
        gamma_tversky: float = 1.33,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

        self.focal_loss = BinaryFocalLoss(alpha=alpha, gamma=gamma_focal)
        self.tversky_loss = FocalTverskyLoss(
            alpha=alpha, beta=beta, gamma=gamma_tversky, from_logits=True
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_focal = self.focal_loss(logits, targets)
        loss_tversky = self.tversky_loss(logits, targets)
        return (self.focal_weight * loss_focal) + (self.tversky_weight * loss_tversky)
