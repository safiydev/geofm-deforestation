"""Tests for spatial block cross-validation, evaluation metrics, and trainer."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from geofm.models.siamese import SiameseChangeDetector
from geofm.training.dataset import (
    GeoSample,
    GeospatialDeforestationDataset,
    SpatialBlockSplitter,
)
from geofm.training.metrics import DeforestationMetrics
from geofm.training.trainer import DeforestationTrainer
from geofm.utils.synthetic_data import generate_synthetic_dataset


def test_spatial_block_cross_validation_split():
    # 20 samples spread across varying latitudes and longitudes
    samples = generate_synthetic_dataset(num_samples=20, height=32, width=32, coord_step=0.2)
    splitter = SpatialBlockSplitter(
        grid_size_deg=0.2, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42
    )

    train_idx, val_idx, test_idx = splitter.split(samples)
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0

    # Ensure completely disjoint sets
    s_train = set(train_idx)
    s_val = set(val_idx)
    s_test = set(test_idx)
    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)

    # Ensure no block ID is shared between train and validation (preventing spatial autocorrelation)
    train_blocks = {samples[i].block_id for i in train_idx}
    val_blocks = {samples[i].block_id for i in val_idx}
    assert train_blocks.isdisjoint(val_blocks)

    # Test spatial buffer margin enforcement: verify training samples are outside buffer zone of val samples
    splitter_with_buffer = SpatialBlockSplitter(
        grid_size_deg=0.1, buffer_size_deg=0.08, train_ratio=0.5, val_ratio=0.5, test_ratio=0.0, random_seed=42
    )
    t_idx, v_idx, _ = splitter_with_buffer.split(samples)
    if t_idx and v_idx:
        buf_sq = 0.08**2
        val_pts = [(samples[j].lon, samples[j].lat) for j in v_idx]
        for i in t_idx:
            s_pt = (samples[i].lon, samples[i].lat)
            for e_pt in val_pts:
                dist_sq = (s_pt[0] - e_pt[0])**2 + (s_pt[1] - e_pt[1])**2
                assert dist_sq >= buf_sq - 1e-6


def test_trainer_type_hints():
    import typing
    hints = typing.get_type_hints(DeforestationTrainer._forward_step)
    assert "return" in hints


def test_deforestation_metrics_computation():
    metrics = DeforestationMetrics(threshold=0.5)

    # Ground truth: 10 pixels positive (deforested), 90 pixels negative (forest)
    targets = np.zeros((1, 10, 10))
    targets[0, 0, :10] = 1.0

    # Predictions: 8 true positives, 2 false positives, 2 false negatives, 88 true negatives
    preds = np.zeros((1, 10, 10))
    preds[0, 0, :8] = 0.9  # TP: 8
    preds[0, 1, :2] = 0.9  # FP: 2
    # Missed [0, 0, 8:10] -> FN: 2

    metrics.update(preds, targets)
    report = metrics.compute()

    assert report.true_positives == 8
    assert report.false_positives == 2
    assert report.false_negatives == 2
    assert report.true_negatives == 88

    # IoU = 8 / (8 + 2 + 2) = 8 / 12 = 0.6667
    assert np.isclose(report.iou, 8.0 / 12.0, atol=1e-3)
    # Precision = 8 / 10 = 0.8
    assert np.isclose(report.precision, 0.8, atol=1e-3)
    # Recall = 8 / 10 = 0.8
    assert np.isclose(report.recall, 0.8, atol=1e-3)
    # F1 = 0.8
    assert np.isclose(report.f1_score, 0.8, atol=1e-3)


def test_trainer_single_epoch_run(tmp_path):
    samples = generate_synthetic_dataset(num_samples=4, height=32, width=32)
    dataset = GeospatialDeforestationDataset(samples)
    train_loader = DataLoader(Subset(dataset, [0, 1]), batch_size=2)
    val_loader = DataLoader(Subset(dataset, [2, 3]), batch_size=2)

    model = SiameseChangeDetector(in_channels=4, features=[8, 16, 32, 64])
    trainer = DeforestationTrainer(
        model=model,
        checkpoint_dir=str(tmp_path),
        model_type="siamese",
    )

    history = trainer.fit(train_loader, val_loader, epochs=1)
    assert len(history) == 1
    assert "val_iou" in history[0]
    assert (tmp_path / "best_model.pt").exists()
