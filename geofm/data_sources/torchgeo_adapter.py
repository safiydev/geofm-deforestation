"""TorchGeo Dataset Adapter and Dynamic Patch Sampler.

Integrates TorchGeo (pip install torchgeo) with GeoFM deforestation detection pipelines.
Handles multi-spectral bands (B02, B03, B04, B08, B11, B12), coordinate reference systems (CRS),
and dynamic geospatial patch sampling (RandomGeoSampler and GridGeoSampler) without manual pre-cropping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import torchgeo
    from torchgeo.datasets import Sentinel2, BoundingBox
    from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
    HAS_TORCHGEO = True
except ImportError:
    HAS_TORCHGEO = False


class TorchGeoBitemporalWrapper(Dataset):
    """Wraps TorchGeo geospatial datasets into GeoFM (T1, T2, Target) bitemporal samples."""

    def __init__(
        self,
        t1_dataset: Any,
        t2_dataset: Any,
        sampler: Optional[Any] = None,
        patch_size: int = 256,
        bands: Optional[List[str]] = None,
    ):
        """
        Args:
            t1_dataset: TorchGeo raster dataset for pre-event acquisition.
            t2_dataset: TorchGeo raster dataset for post-event acquisition.
            sampler: GeoSampler (RandomGeoSampler or GridGeoSampler).
            patch_size: Spatial tile size in pixels (default: 256).
            bands: Multi-spectral band names (e.g. ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']).
        """
        self.t1_dataset = t1_dataset
        self.t2_dataset = t2_dataset
        self.sampler = sampler
        self.patch_size = patch_size
        self.bands = bands or ["B02", "B03", "B04", "B08", "B11", "B12"]

    def __len__(self) -> int:
        if self.sampler is not None:
            return len(self.sampler)
        return len(self.t1_dataset) if hasattr(self.t1_dataset, "__len__") else 1000

    def __getitem__(self, query_or_idx: Any) -> Dict[str, torch.Tensor]:
        """Fetch bitemporal chip using geospatial bounding box or sample index."""
        if not HAS_TORCHGEO:
            raise ImportError(
                "TorchGeo is required for this adapter. Install with: pip install torchgeo"
            )

        if isinstance(query_or_idx, int) and self.sampler is not None:
            bbox = list(self.sampler)[query_or_idx]
        else:
            bbox = query_or_idx

        sample_t1 = self.t1_dataset[bbox]
        sample_t2 = self.t2_dataset[bbox]

        t1_tensor = sample_t1["image"].float() / 10000.0  # normalize Sentinel-2 DN to [0, 1]
        t2_tensor = sample_t2["image"].float() / 10000.0

        # Target mask if available in sample_t2, else dummy zero mask
        if "mask" in sample_t2:
            target_mask = sample_t2["mask"].float()
        else:
            target_mask = torch.zeros(1, self.patch_size, self.patch_size, dtype=torch.float32)

        return {
            "sample_id": f"torchgeo_{getattr(bbox, 'mint', 0)}_{getattr(bbox, 'minx', 0):.4f}",
            "t1": t1_tensor,
            "t2": t2_tensor,
            "target": target_mask,
            "lon": torch.tensor(getattr(bbox, "minx", 0.0), dtype=torch.float32),
            "lat": torch.tensor(getattr(bbox, "miny", 0.0), dtype=torch.float32),
            "block_id": torch.tensor(0, dtype=torch.long),
        }
