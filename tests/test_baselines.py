"""Tests for classical time-series change detection baselines (CCDC, BFAST, NDVI Delta)."""

import numpy as np
import pytest

from geofm.baselines.ccdc import CCDCDirectDetector
from geofm.baselines.bfast import BFASTMonitor
from geofm.baselines.ndvi_delta import NDVIDeltaBaseline


def test_ccdc_detects_abrupt_step_deforestation():
    detector = CCDCDirectDetector(consecutive_anomalies=3, threshold_factor=3.0)

    # 30 observations across 2 years
    times = np.linspace(0.0, 730.0, 30)
    omega = 2.0 * np.pi / 365.25
    # Stable baseline: seasonal sine wave with mean 0.70 (intact forest canopy)
    baseline = 0.70 + 0.10 * np.sin(omega * times)
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.02, size=30)
    series = baseline + noise

    # Before change: stable
    res_stable = detector.fit_and_detect(times, series, min_init_points=12)
    assert res_stable.break_detected == False

    # Introduce severe deforestation drop starting at index 18
    series_disturbed = np.copy(series)
    series_disturbed[18:] -= 0.40  # Massive drop down to 0.30

    res_break = detector.fit_and_detect(times, series_disturbed, min_init_points=12)
    assert res_break.break_detected == True
    assert res_break.break_index is not None
    assert res_break.break_index >= 18


def test_bfast_ols_mosum_monitoring():
    monitor = BFASTMonitor(bandwidth_fraction=0.3, critical_value=1.36)

    # 25 observations
    series = np.ones(25) * 0.75
    # Deforestation at index 16
    series[16:] = 0.20

    result = monitor.detect_break(series, history_length=12)
    assert result.break_detected == True
    assert result.break_index is not None
    assert result.break_index >= 16


def test_ndvi_delta_baseline():
    baseline = NDVIDeltaBaseline(delta_threshold=-0.25, post_ndvi_max=0.45)
    pre_ndvi = np.array([[0.80, 0.75]])
    post_ndvi = np.array([[0.20, 0.72]])  # Pixel 0 deforested, pixel 1 stable

    res = baseline.detect_change(pre_ndvi, post_ndvi)
    assert res["change_mask"][0, 0] == True
    assert res["change_mask"][0, 1] == False
