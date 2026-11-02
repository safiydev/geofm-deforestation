"""Data source connectors and adapters for forest change detection."""

from __future__ import annotations

from geofm.data_sources.torchgeo_adapter import TorchGeoBitemporalWrapper
from geofm.data_sources.earth_engine import (
    init_earth_engine,
    download_and_cache_gee_dataset,
    get_gee_export_script_template,
    DEFORESTATION_HOTSPOTS,
)
from geofm.data_sources.stac_loader import STACDeforestationLoader

__all__ = [
    "TorchGeoBitemporalWrapper",
    "init_earth_engine",
    "download_and_cache_gee_dataset",
    "get_gee_export_script_template",
    "STACDeforestationLoader",
    "DEFORESTATION_HOTSPOTS",
]
