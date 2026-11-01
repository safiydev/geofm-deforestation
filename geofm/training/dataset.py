"""Geospatial Dataset and Spatial Block Cross-Validation.

Prevents spatial autocorrelation and data leakage between training, validation,
and test sets by partitioning tiles into geographically isolated spatial blocks
with spatial buffer margins (TorchGeo-style spatial partitioning).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


@dataclass
class GeoSample:
    """A single geospatial training chip with coordinates and metadata."""
    sample_id: str
    t1_optical: np.ndarray  # (C, H, W)
    t2_optical: np.ndarray  # (C, H, W)
    target_mask: np.ndarray  # (1, H, W) binary change mask
    sar_data: Optional[np.ndarray] = None  # (C_sar, H, W)
    cloud_mask: Optional[np.ndarray] = None  # (1, H, W)
    lon: float = 0.0
    lat: float = 0.0
    block_id: int = 0


class GeospatialDeforestationDataset(Dataset):
    """PyTorch Dataset for bitemporal and multimodal deforestation chips."""

    def __init__(self, samples: List[GeoSample], transform: Optional[any] = None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        item = {
            "sample_id": s.sample_id,
            "t1": torch.from_numpy(s.t1_optical).float(),
            "t2": torch.from_numpy(s.t2_optical).float(),
            "target": torch.from_numpy(s.target_mask).float(),
            "lon": torch.tensor(s.lon, dtype=torch.float32),
            "lat": torch.tensor(s.lat, dtype=torch.float32),
            "block_id": torch.tensor(s.block_id, dtype=torch.long),
        }

        if s.sar_data is not None:
            item["sar"] = torch.from_numpy(s.sar_data).float()

        if s.cloud_mask is not None:
            item["cloud_mask"] = torch.from_numpy(s.cloud_mask).float()

        return item


class SpatialBlockSplitter:
    """Partitions geospatial coordinates into spatially buffered training/validation blocks.

    In Earth observation, neighboring pixels and adjacent image tiles exhibit high
    spatial autocorrelation. Random pixel/tile splitting leads to severe optimistic
    evaluation bias. This splitter assigns discrete spatial grid cells to splits
    and enforces a buffer zone where boundary blocks are discarded.
    """

    def __init__(
        self,
        grid_size_deg: float = 0.1,  # ~11km grid blocks
        buffer_size_deg: float = 0.02,  # Buffer margin to prevent edge leakage
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        self.grid_size = grid_size_deg
        self.buffer_size = buffer_size_deg
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.rng = np.random.default_rng(random_seed)

    def assign_blocks(
        self, samples: List[GeoSample]
    ) -> Tuple[Dict[int, List[int]], Dict[int, Tuple[float, float]]]:
        """Assign each sample to a 2D integer spatial block ID based on (lon, lat).

        Returns:
            (block_to_indices, block_centroids)
        """
        block_to_indices: Dict[int, List[int]] = {}
        block_centroids: Dict[int, Tuple[float, float]] = {}

        for idx, s in enumerate(samples):
            # Compute discrete grid indices
            col = int(np.floor(s.lon / self.grid_size))
            row = int(np.floor(s.lat / self.grid_size))
            # Cantor-like unique 2D grid hash
            block_id = int((col * 73856093) ^ (row * 19349663)) % 1000000
            s.block_id = block_id

            if block_id not in block_to_indices:
                block_to_indices[block_id] = []
                block_centroids[block_id] = (
                    (col + 0.5) * self.grid_size,
                    (row + 0.5) * self.grid_size,
                )
            block_to_indices[block_id].append(idx)

        return block_to_indices, block_centroids

    def split(
        self, samples: List[GeoSample]
    ) -> Tuple[List[int], List[int], List[int]]:
        """Split sample indices into train, validation, and test sets with buffer isolation.

        Samples within buffer_size_deg of a differing split are filtered out to prevent
        spatial autocorrelation and edge leakage.

        Returns:
            (train_indices, val_indices, test_indices)
        """
        block_to_indices, block_centroids = self.assign_blocks(samples)
        unique_blocks = list(block_to_indices.keys())
        self.rng.shuffle(unique_blocks)

        n_blocks = len(unique_blocks)
        n_train = max(1, int(n_blocks * self.train_ratio))
        n_val = max(1, int(n_blocks * self.val_ratio))

        train_blocks = set(unique_blocks[:n_train])
        val_blocks = set(unique_blocks[n_train : n_train + n_val])
        test_blocks = set(unique_blocks[n_train + n_val :])

        raw_train_idx, val_idx, test_idx = [], [], []

        for b_id, indices in block_to_indices.items():
            if b_id in train_blocks:
                raw_train_idx.extend(indices)
            elif b_id in val_blocks:
                val_idx.extend(indices)
            else:
                test_idx.extend(indices)

        # Enforce spatial buffer margin: remove train samples too close to val/test sets
        if self.buffer_size > 0.0 and (val_idx or test_idx):
            eval_coords = [
                (samples[i].lon, samples[i].lat) for i in (val_idx + test_idx)
            ]
            buf_sq = self.buffer_size**2
            train_idx = []

            for i in raw_train_idx:
                s_lon, s_lat = samples[i].lon, samples[i].lat
                # Check distance to nearest evaluation sample
                is_buffered = any(
                    ((s_lon - e_lon) ** 2 + (s_lat - e_lat) ** 2) < buf_sq
                    for e_lon, e_lat in eval_coords
                )
                if not is_buffered:
                    train_idx.append(i)

            # If aggressive buffer removed all train samples, fallback to raw_train_idx
            if not train_idx and raw_train_idx:
                train_idx = raw_train_idx
        else:
            train_idx = raw_train_idx

        return train_idx, val_idx, test_idx
