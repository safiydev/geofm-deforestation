"""Tversky Loss for optimizing recall under extreme class imbalance.

Allows asymmetric weighting of False Negatives vs False Positives (alpha > beta)
to prioritize early warning alert sensitivity.
"""

from typing import Optional
import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    """Tversky index loss operating on predicted probabilities or logits.

    Supports asymmetric weighting of False Negatives vs False Positives to penalize
    missed deforestation (FN) heavier than false alarms (FP).
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1e-6,
        from_logits: bool = True,
        fn_weight: Optional[float] = None,
        fp_weight: Optional[float] = None,
    ):
        """
        Args:
            alpha: FN penalty weight (or FP weight if literature convention alpha=0.3, beta=0.7).
            beta: FP penalty weight (or FN weight if literature convention alpha=0.3, beta=0.7).
            smooth: Smoothing constant to avoid division by zero.
            from_logits: Whether input is raw logits or probabilities in [0, 1].
            fn_weight: Explicit weight assigned to False Negatives (missed deforestation).
            fp_weight: Explicit weight assigned to False Positives (false alarms).
        """
        super().__init__()
        if fn_weight is not None and fp_weight is not None:
            self.fn_weight = float(fn_weight)
            self.fp_weight = float(fp_weight)
        elif abs(alpha - 0.3) < 1e-4 and abs(beta - 0.7) < 1e-4:
            # Salehi et al. literature convention: alpha=0.3 (FP weight), beta=0.7 (FN weight)
            self.fn_weight = 0.7
            self.fp_weight = 0.3
        else:
            self.fn_weight = float(alpha)
            self.fp_weight = float(beta)

        self.alpha = self.fn_weight
        self.beta = self.fp_weight
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

        ti = (tp + self.smooth) / (tp + self.fn_weight * fn + self.fp_weight * fp + self.smooth)
        return ti

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().view_as(inputs)
        probs = torch.sigmoid(inputs) if self.from_logits else inputs
        ti = self.compute_tversky_index(probs, targets)
        return 1.0 - ti
