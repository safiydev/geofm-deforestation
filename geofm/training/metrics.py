"""Deforestation evaluation metrics and confusion matrix calculations.

Computes IoU, F1-Score, Precision, Recall, Specificity, and confusion matrices
optimized for heavily skewed class distributions in early deforestation alerts.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


@dataclass
class MetricReport:
    """Comprehensive evaluation metrics report."""
    iou: float
    f1_score: float
    precision: float
    recall: float
    specificity: float
    overall_accuracy: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    threshold: float

    def to_dict(self) -> Dict[str, Union[float, int]]:
        return {
            "iou": round(self.iou, 4),
            "f1_score": round(self.f1_score, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "specificity": round(self.specificity, 4),
            "overall_accuracy": round(self.overall_accuracy, 4),
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "tn": self.true_negatives,
            "threshold": self.threshold,
        }

    def summary(self) -> str:
        return (
            f"--- Deforestation Evaluation Report (Threshold: {self.threshold:.2f}) ---\n"
            f"  IoU (Jaccard):    {self.iou * 100:.2f}%\n"
            f"  F1-Score (Dice):  {self.f1_score * 100:.2f}%\n"
            f"  Precision:        {self.precision * 100:.2f}%\n"
            f"  Recall:           {self.recall * 100:.2f}%\n"
            f"  Specificity:      {self.specificity * 100:.2f}%\n"
            f"  Overall Accuracy: {self.overall_accuracy * 100:.2f}%\n"
            f"  Confusion Matrix: [TP: {self.true_positives:,}, FP: {self.false_positives:,}, "
            f"FN: {self.false_negatives:,}, TN: {self.true_negatives:,}]"
        )


class DeforestationMetrics:
    """Accumulates predictions and targets across batches and computes metrics."""

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Update running confusion matrix counters.

        Args:
            predictions: Predicted logits or probabilities (B, 1, H, W) or (B, H, W).
            targets: Binary ground truth masks (0 or 1).
        """
        if isinstance(predictions, torch.Tensor):
            preds = predictions.detach().cpu().numpy()
        else:
            preds = np.asarray(predictions)

        if isinstance(targets, torch.Tensor):
            targs = targets.detach().cpu().numpy()
        else:
            targs = np.asarray(targets)

        # Apply sigmoid if values outside [0, 1]
        if np.min(preds) < 0.0 or np.max(preds) > 1.0:
            preds = 1.0 / (1.0 + np.exp(-preds))

        bin_preds = (preds >= self.threshold).astype(bool).ravel()
        bin_targs = (targs >= 0.5).astype(bool).ravel()

        self.tp += int(np.sum(bin_preds & bin_targs))
        self.fp += int(np.sum(bin_preds & (~bin_targs)))
        self.fn += int(np.sum((~bin_preds) & bin_targs))
        self.tn += int(np.sum((~bin_preds) & (~bin_targs)))

    def compute(self) -> MetricReport:
        """Compute final metrics based on accumulated counts."""
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        total = tp + fp + fn + tn

        iou = (tp + self.eps) / (tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        f1 = (2.0 * precision * recall) / (precision + recall + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)
        accuracy = (tp + tn) / max(1, total)

        return MetricReport(
            iou=float(iou),
            f1_score=float(f1),
            precision=float(precision),
            recall=float(recall),
            specificity=float(specificity),
            overall_accuracy=float(accuracy),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            threshold=self.threshold,
        )

    @classmethod
    def evaluate_array(
        cls, preds: np.ndarray, targets: np.ndarray, threshold: float = 0.5
    ) -> MetricReport:
        """Convenience method to evaluate single or multi-image arrays directly."""
        m = cls(threshold=threshold)
        m.update(preds, targets)
        return m.compute()
