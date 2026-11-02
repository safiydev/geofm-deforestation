"""Unit tests for Google Earth Engine (GEE) data source pipeline."""

from __future__ import annotations

import os
import shutil
import tempfile
import pytest

from geofm.data_sources.earth_engine import (
    generate_hotspot_grid_bboxes,
    get_gee_export_script_template,
    download_and_cache_gee_dataset,
    DEFORESTATION_HOTSPOTS,
)
from geofm.training.dataset import LazyGeospatialDataset


def test_hotspot_grid_bboxes():
    bboxes = generate_hotspot_grid_bboxes(num_chips=50, chip_size_deg=0.0256, seed=42)
    assert len(bboxes) == 50
    for w, s, e, n in bboxes:
        assert w < e
        assert s < n
        # Width should roughly equal chip_size_deg
        assert abs((e - w) - 0.0256) < 1e-4


def test_gee_export_script_template():
    script = get_gee_export_script_template(
        roi_bbox=(-63.0, -9.5, -62.5, -9.0),
        t1_date_range=("2023-06-01", "2023-09-30"),
        t2_date_range=("2024-06-01", "2024-09-30"),
    )
    assert "ee.Initialize" in script
    assert "COPERNICUS/S2_SR_HARMONIZED" in script
    assert "UMD/hansen/global_forest_change" in script
    assert "B2" in script and "B8" in script and "B11" in script and "B12" in script


def test_download_and_cache_gee_fallback():
    tmp_dir = tempfile.mkdtemp(prefix="test_gee_cache_")
    try:
        manifest_path, total = download_and_cache_gee_dataset(
            output_dir=tmp_dir,
            max_chips=4,
            chip_size=64,
            show_progress=False,
        )
        assert os.path.exists(manifest_path)
        assert total == 4

        dataset = LazyGeospatialDataset(cache_dir=tmp_dir, preload_cache=False)
        assert len(dataset) == 4
        sample = dataset[0]
        # 6 multi-spectral bands: Blue, Green, Red, NIR, SWIR-1, SWIR-2
        assert sample["t1"].shape == (6, 64, 64)
        assert sample["t2"].shape == (6, 64, 64)
        assert sample["target"].shape[-2:] == (64, 64)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
