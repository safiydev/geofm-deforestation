"""Unit tests for bitemporal data augmentation, spectral indices, and class balance filtering."""

import numpy as np
import torch
import pytest

from geofm.preprocessing.spectral_indices import (
    compute_ndvi,
    compute_nbr,
    compute_spectral_differential_stack,
)
from geofm.training.augmentation import BitemporalAugmentor
from geofm.training.dataset import (
    GeoSample,
    GeospatialDeforestationDataset,
    filter_class_balance,
    ClassBalancedSampler,
)


def test_spectral_indices_numpy():
    # Shape: (6, 32, 32) -> B, G, R, NIR, SWIR1, SWIR2
    t1 = np.zeros((6, 32, 32), dtype=np.float32)
    # Healthy forest: low Red (0.04), high NIR (0.50), low SWIR2 (0.06)
    t1[2] = 0.04
    t1[3] = 0.50
    t1[5] = 0.06

    ndvi = compute_ndvi(t1, red_idx=2, nir_idx=3)
    nbr = compute_nbr(t1, nir_idx=3, swir2_idx=5)

    assert ndvi.shape == (1, 32, 32)
    assert nbr.shape == (1, 32, 32)
    assert np.all(ndvi > 0.8)  # (0.5 - 0.04)/(0.5 + 0.04) ~ 0.85
    assert np.all(nbr > 0.75)


def test_spectral_indices_torch():
    t1 = torch.zeros(2, 6, 16, 16)
    t1[:, 2] = 0.05
    t1[:, 3] = 0.45
    t1[:, 5] = 0.08

    ndvi = compute_ndvi(t1, red_idx=2, nir_idx=3)
    assert ndvi.shape == (2, 1, 16, 16)
    assert torch.all(ndvi > 0.7)


def test_spectral_differential_stack():
    # T1: healthy forest (NIR 0.50, Red 0.04, SWIR2 0.06)
    t1 = np.zeros((6, 16, 16), dtype=np.float32)
    t1[2] = 0.04
    t1[3] = 0.50
    t1[5] = 0.06

    # T2: deforested bare soil (NIR 0.15, Red 0.18, SWIR2 0.25)
    t2 = np.zeros((6, 16, 16), dtype=np.float32)
    t2[2] = 0.18
    t2[3] = 0.15
    t2[5] = 0.25

    t1_stack, t2_stack = compute_spectral_differential_stack(t1, t2)
    # Original 6 bands + NDVI + NBR = 8 channels
    assert t1_stack.shape == (8, 16, 16)
    assert t2_stack.shape == (8, 16, 16)


def test_bitemporal_augmentor_d4():
    augmentor = BitemporalAugmentor(
        enable_flips=True,
        enable_rot90=True,
        enable_radiometric_jitter=True,
        p=1.0,  # force application
    )

    sample = {
        "sample_id": "test_chip",
        "t1": torch.rand(4, 32, 32),
        "t2": torch.rand(4, 32, 32),
        "target": torch.zeros(1, 32, 32),
        "cloud_mask": torch.zeros(1, 32, 32),
    }
    # Put a specific marker pixel in target
    sample["target"][0, 0, 0] = 1.0

    aug_sample = augmentor(sample)
    assert aug_sample["t1"].shape == (4, 32, 32)
    assert aug_sample["t2"].shape == (4, 32, 32)
    assert aug_sample["target"].shape == (1, 32, 32)
    assert aug_sample["target"].sum() == 1.0  # target binary integrity preserved


def test_filter_class_balance():
    # Create 10 dummy samples: 5 with >5% deforested pixels, 5 with 0%
    samples = []
    for i in range(10):
        tgt = np.zeros((1, 20, 20), dtype=np.float32)
        if i < 5:
            tgt[0, :5, :5] = 1.0  # 25/400 = 6.25% deforested
        sample = GeoSample(
            sample_id=f"chip_{i}",
            t1_optical=np.zeros((4, 20, 20), dtype=np.float32),
            t2_optical=np.zeros((4, 20, 20), dtype=np.float32),
            target_mask=tgt,
        )
        samples.append(sample)

    dataset = GeospatialDeforestationDataset(samples)
    # Filter with target 70% positive
    selected = filter_class_balance(dataset, target_pos_ratio=0.70, min_pos_pct=5.0)
    assert len(selected) > 0
    # Positives must be included
    pos_in_selected = sum(1 for idx in selected if idx < 5)
    assert pos_in_selected == 5
