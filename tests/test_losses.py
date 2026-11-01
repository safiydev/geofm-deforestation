"""Tests for class-imbalanced deforestation loss functions."""

import torch
import pytest

from geofm.losses.focal import BinaryFocalLoss
from geofm.losses.tversky import TverskyLoss
from geofm.losses.focal_tversky import FocalTverskyLoss, CompoundDeforestationLoss


def test_binary_focal_loss():
    focal = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    logits = torch.randn(4, 1, 32, 32, requires_grad=True)
    # Severe class imbalance: mostly 0, few 1s
    targets = torch.zeros(4, 1, 32, 32)
    targets[:, :, 5:10, 5:10] = 1.0

    loss = focal(logits, targets)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None


def test_tversky_loss_asymmetry():
    # Tversky loss with alpha=0.8 (high FN penalty) and beta=0.2
    tversky = TverskyLoss(alpha=0.8, beta=0.2, from_logits=True)

    # Perfect prediction
    targets = torch.zeros(1, 1, 10, 10)
    targets[0, 0, :5, :5] = 1.0
    perfect_logits = (targets * 10.0) - 5.0  # high confidence match
    loss_perf = tversky(perfect_logits, targets)
    assert loss_perf.item() < 0.05

    # Case A: False Negatives (model predicts zeros everywhere, misses all targets)
    fn_logits = torch.full((1, 1, 10, 10), -10.0)
    loss_fn = tversky(fn_logits, targets)

    # Case B: False Positives (model predicts ones everywhere, false alarms)
    fp_logits = torch.full((1, 1, 10, 10), 10.0)
    # If alpha > beta, missing targets (FN) should be penalized heavily
    assert loss_fn.item() > 0.5


def test_compound_loss_backward():
    compound = CompoundDeforestationLoss(focal_weight=0.5, tversky_weight=0.5)
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 16, 16)).float()

    loss = compound(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(loss)
