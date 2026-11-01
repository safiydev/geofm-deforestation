"""Training and evaluation modules for deforestation detection."""

from geofm.training.dataset import (
    GeoSample,
    GeospatialDeforestationDataset,
    DeforestationDataset,
    SpatialBlockSplitter,
    validate_sample,
    cache_dataset_to_npz,
    load_dataset_from_npz,
)
from geofm.training.metrics import DeforestationMetrics
from geofm.training.trainer import DeforestationTrainer

__all__ = [
    "GeoSample",
    "GeospatialDeforestationDataset",
    "DeforestationDataset",
    "SpatialBlockSplitter",
    "validate_sample",
    "cache_dataset_to_npz",
    "load_dataset_from_npz",
    "DeforestationMetrics",
    "DeforestationTrainer",
]
