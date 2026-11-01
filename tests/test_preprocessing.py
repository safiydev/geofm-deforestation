"""Tests for optical, SAR, COG, and STAC preprocessing modules."""

from datetime import datetime, timezone
import numpy as np
import pytest

from geofm.preprocessing.optical import OpticalPreprocessor
from geofm.preprocessing.sar import SARPreprocessor
from geofm.preprocessing.cog_reader import COGWindowReader, GeoTransform, Window
from geofm.preprocessing.stac import STACManager


def test_optical_normalization_and_indices():
    prep = OpticalPreprocessor(sensor_type="sentinel2")
    # Sentinel-2 DN scaling by 10000
    raw_s2 = np.array([0, 5000, 10000, 12000], dtype=np.float32)
    norm = prep.normalize(raw_s2)
    assert np.allclose(norm, [0.0, 0.5, 1.0, 1.0])

    # NDVI
    red = np.array([[0.05, 0.20]])
    nir = np.array([[0.50, 0.15]])
    ndvi = prep.compute_ndvi(red, nir)
    assert ndvi[0, 0] > 0.7  # Healthy vegetation
    assert ndvi[0, 1] < 0.0  # Bare ground or water

    # EVI
    blue = np.array([[0.02, 0.05]])
    evi = prep.compute_evi(blue, red, nir)
    assert evi.shape == (1, 2)
    assert evi[0, 0] > 0.5


def test_optical_cloud_and_shadow_masking():
    prep = OpticalPreprocessor(cloud_threshold=0.3, shadow_threshold=0.1)
    # Bright cloud pixel
    blue = np.array([[0.45, 0.02]])
    green = np.array([[0.45, 0.05]])
    red = np.array([[0.45, 0.03]])
    nir = np.array([[0.40, 0.04]])  # Second pixel is dark shadow

    c_mask, s_mask = prep.detect_cloud_and_shadow_mask(blue, green, red, nir)
    assert c_mask[0, 0] == True
    assert c_mask[0, 1] == False
    assert s_mask[0, 1] == True
    assert s_mask[0, 0] == False


def test_sar_decibels_and_speckle_filters():
    sar_prep = SARPreprocessor(num_looks=4.4)

    # Decibel conversion
    lin = np.array([0.01, 0.1, 1.0])
    db = sar_prep.to_decibels(lin)
    assert np.allclose(db, [-20.0, -10.0, 0.0], atol=1e-3)
    assert np.allclose(sar_prep.to_linear(db), lin, atol=1e-3)

    # Lee filter
    rng = np.random.default_rng(42)
    clean = np.ones((32, 32), dtype=np.float32) * 0.1
    # Add multiplicative speckle noise
    noisy = clean * rng.exponential(1.0, (32, 32))
    filtered = sar_prep.lee_filter(noisy, window_size=5)

    assert filtered.shape == (32, 32)
    # Filter should reduce variance while preserving mean
    assert np.var(filtered) < np.var(noisy)
    assert np.isclose(np.mean(filtered), np.mean(clean), atol=0.03)

    # Gamma MAP filter
    gamma_filt = sar_prep.gamma_map_filter(noisy, window_size=5)
    assert gamma_filt.shape == (32, 32)
    assert np.var(gamma_filt) < np.var(noisy)


def test_sar_hampel_filter():
    sar_prep = SARPreprocessor()
    # 15 time steps
    series = np.ones((15, 8, 8), dtype=np.float32) * 0.05
    # Insert extreme moisture/rainfall spike at t=7
    series[7, 3, 3] = 0.50

    filtered = sar_prep.hampel_temporal_filter(series, window_size=5, n_sigmas=3.0)
    # The spike at t=7 should be replaced with local median (~0.05)
    assert filtered[7, 3, 3] < 0.10

    # 1D series test
    series_1d = np.array([0.05, 0.05, 0.06, 0.90, 0.05, 0.05, 0.04])
    filt_1d = sar_prep.hampel_temporal_filter(series_1d, window_size=5, n_sigmas=3.0)
    assert filt_1d[3] < 0.10


def test_cog_window_reader_and_chipping():
    data = np.arange(1, 10001, dtype=np.float32).reshape(1, 100, 100)
    transform = GeoTransform(west=-60.0, north=-3.0, pixel_width=0.001, pixel_height=-0.001)
    reader = COGWindowReader(data, transform=transform)

    # Read window
    win = Window(col_off=10, row_off=20, width=15, height=25)
    chip = reader.read_window(win)
    assert chip.shape == (1, 25, 15)

    # Coordinate transform
    geo_x, geo_y = transform.pixel_to_geo(10, 20)
    assert np.isclose(geo_x, -60.0 + 10 * 0.001)
    assert np.isclose(geo_y, -3.0 - 20 * 0.001)

    # Chipping generator
    chips = list(reader.generate_chips(chip_size=32, stride=32))
    assert len(chips) > 0
    w_first, arr_first = chips[0]
    assert arr_first.shape == (1, 32, 32)


def test_stac_manager():
    stac = STACManager()
    s2_item = stac.create_sentinel2_item(
        item_id="test-s2-001",
        bbox=[-60.0, -3.0, -59.0, -2.0],
        dt=datetime(2023, 6, 1, tzinfo=timezone.utc),
        asset_hrefs={"B02": "http://example.com/B02.tif", "B04": "http://example.com/B04.tif"},
    )
    assert s2_item.id == "test-s2-001"
    assert "B02" in s2_item.assets

    s1_item = stac.create_sentinel1_item(
        item_id="test-s1-001",
        bbox=[-60.0, -3.0, -59.0, -2.0],
        dt=datetime(2023, 6, 2, tzinfo=timezone.utc),
        asset_hrefs={"vv": "http://example.com/vv.tif", "vh": "http://example.com/vh.tif"},
    )
    assert s1_item.id == "test-s1-001"

    # Query catalog
    items = [s2_item, s1_item]
    matched = stac.query_catalog(items, platform="sentinel-2")
    assert len(matched) == 1
    assert matched[0].id == "test-s2-001"

    # Spatial query inside bbox
    matched_box = stac.query_catalog(items, bbox=[-59.5, -2.5, -59.2, -2.2])
    assert len(matched_box) == 2

    # Query with offset-naive start_date and end_date (must not raise TypeError)
    matched_naive = stac.query_catalog(
        items,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
    )
    assert len(matched_naive) == 2
