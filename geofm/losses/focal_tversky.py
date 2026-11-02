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
    """Focal Tversky Loss: (1 - TverskyIndex)^gamma.

    Penalizes missed deforestation heavier than false alarms when FN weight > FP weight.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.33,
        smooth: float = 1e-6,
        from_logits: bool = True,
        fn_weight: Optional[float] = None,
        fp_weight: Optional[float] = None,
    ):
        """
        Args:
            alpha: FN penalty weight (or FP weight if literature convention alpha=0.3, beta=0.7).
            beta: FP penalty weight (or FN weight if literature convention alpha=0.3, beta=0.7).
            gamma: Focusing exponent (typically 1.0 to 2.0).
            smooth: Numerical stabilizer.
            from_logits: Whether input is logits.
            fn_weight: Explicit weight assigned to False Negatives (missed deforestation).
            fp_weight: Explicit weight assigned to False Positives (false alarms).
        """
        super().__init__()
        self.tversky = TverskyLoss(
            alpha=alpha,
            beta=beta,
            smooth=smooth,
            from_logits=from_logits,
            fn_weight=fn_weight,
            fp_weight=fp_weight,
        )
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tl = self.tversky(inputs, targets)
        # Loss = (1 - TI)^gamma
        return torch.pow(tl, self.gamma)


class CompoundDeforestationLoss(nn.Module):
    """Production compound loss combining Binary Focal Loss and Focal Tversky Loss.

    By default, sets Tversky parameters so missed deforestation (FN) is penalized
    heavier (~2.33x) than false positives (FP), while Focal loss focuses gradient
    on rare canopy change pixels.
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        tversky_weight: float = 0.5,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma_focal: float = 2.0,
        gamma_tversky: float = 1.33,
        focal_alpha: float = 0.75,
        fn_weight: Optional[float] = None,
        fp_weight: Optional[float] = None,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

        self.focal_loss = BinaryFocalLoss(alpha=focal_alpha, gamma=gamma_focal)
        self.tversky_loss = FocalTverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma_tversky,
            from_logits=True,
            fn_weight=fn_weight,
            fp_weight=fp_weight,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_focal = self.focal_loss(logits, targets)
        loss_tversky = self.tversky_loss(logits, targets)
        return (self.focal_weight * loss_focal) + (self.tversky_weight * loss_tversky)
