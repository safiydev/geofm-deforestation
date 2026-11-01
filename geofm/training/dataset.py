"""Geospatial Dataset and Spatial Block Cross-Validation.

Prevents spatial autocorrelation and data leakage between training, validation,
and test sets by partitioning tiles into geographically isolated spatial blocks
with spatial buffer margins (TorchGeo-style spatial partitioning).
Includes resilient pre-flight verification and offline NPZ chip caching.
"""

from dataclasses import asdict, dataclass
import gc
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

logger = logging.getLogger("geofm.training.dataset")


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


def validate_sample(sample: GeoSample) -> Tuple[bool, str]:
    """Pre-flight validation for a geospatial sample to prevent training crashes."""
    if not isinstance(sample, GeoSample):
        return False, f"Expected GeoSample instance, got {type(sample)}"

    # 1. Check array presence
    if sample.t1_optical is None or sample.t2_optical is None or sample.target_mask is None:
        return False, "Missing required array (t1_optical, t2_optical, or target_mask)"

    # 2. Check dimensions
    if sample.t1_optical.ndim != 3 or sample.t2_optical.ndim != 3:
        return False, f"Optical arrays must be 3D (C, H, W), got t1:{sample.t1_optical.shape}, t2:{sample.t2_optical.shape}"

    if sample.t1_optical.shape != sample.t2_optical.shape:
        return False, f"Shape mismatch between t1 {sample.t1_optical.shape} and t2 {sample.t2_optical.shape}"

    # 3. Check spatial dimensions with target mask
    _, h, w = sample.t1_optical.shape
    if sample.target_mask.shape[-2:] != (h, w):
        return False, f"Mask spatial shape {sample.target_mask.shape} does not match optical spatial shape ({h}, {w})"

    # 4. Check for NaNs or Infs
    if np.isnan(sample.t1_optical).any() or np.isnan(sample.t2_optical).any():
        return False, "Optical arrays contain NaN values"

    if np.isinf(sample.t1_optical).any() or np.isinf(sample.t2_optical).any():
        return False, "Optical arrays contain Infinite values"

    return True, "Valid"


class GeospatialDeforestationDataset(Dataset):
    """PyTorch Dataset for bitemporal and multimodal deforestation chips.

    Includes automatic pre-flight data sanitization to filter out corrupt or
    dimension-mismatched satellite tiles before training.
    """

    def __init__(
        self,
        samples: List[GeoSample],
        transform: Optional[Any] = None,
        preflight_validate: bool = True,
    ):
        self.transform = transform
        self.invalid_samples_count = 0

        if preflight_validate:
            valid_samples: List[GeoSample] = []
            for s in samples:
                is_valid, msg = validate_sample(s)
                if is_valid:
                    valid_samples.append(s)
                else:
                    self.invalid_samples_count += 1
                    logger.warning(f"Skipping corrupt sample '{getattr(s, 'sample_id', 'unknown')}': {msg}")

            if not valid_samples and samples:
                logger.error("All samples failed pre-flight validation. Falling back to raw list.")
                self.samples = samples
            else:
                self.samples = valid_samples
        else:
            self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        # Ensure non-negative finite float32
        t1_arr = np.nan_to_num(s.t1_optical, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        t2_arr = np.nan_to_num(s.t2_optical, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        tgt_arr = np.nan_to_num(s.target_mask, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

        if tgt_arr.ndim == 2:
            tgt_arr = tgt_arr[np.newaxis, ...]

        item = {
            "sample_id": s.sample_id,
            "t1": torch.from_numpy(t1_arr),
            "t2": torch.from_numpy(t2_arr),
            "target": torch.from_numpy(tgt_arr),
            "lon": torch.tensor(s.lon, dtype=torch.float32),
            "lat": torch.tensor(s.lat, dtype=torch.float32),
            "block_id": torch.tensor(s.block_id, dtype=torch.long),
        }

        if s.sar_data is not None:
            sar_arr = np.nan_to_num(s.sar_data, nan=0.0, posinf=0.0, neginf=-30.0).astype(np.float32)
            item["sar"] = torch.from_numpy(sar_arr)

        if s.cloud_mask is not None:
            c_arr = np.nan_to_num(s.cloud_mask, nan=0.0).astype(np.float32)
            if c_arr.ndim == 2:
                c_arr = c_arr[np.newaxis, ...]
            item["cloud_mask"] = torch.from_numpy(c_arr)

        return item


# Backwards-compatibility alias
DeforestationDataset = GeospatialDeforestationDataset


class LazyGeospatialDataset(Dataset):
    """Zero-RAM disk-backed PyTorch Dataset for large-scale geospatial chip collections.

    Instead of holding uncompressed multispectral satellite imagery in host RAM (which
    crashes on 12.7 GB Google Colab when scaling past 1,000 chips), this dataset loads chips
    on-demand in `__getitem__` directly from cached .npz files on disk. Memory footprint
    remains strictly bounded (<50 MB) regardless of whether the dataset contains 100 or
    100,000 chips.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        file_paths: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        transform: Optional[Any] = None,
    ):
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.samples_meta: List[Dict[str, Any]] = []

        if metadata is not None:
            self.samples_meta = list(metadata)
        elif self.cache_dir and (self.cache_dir / "manifest.jsonl").exists():
            # Fast O(1) metadata streaming from manifest
            manifest_path = self.cache_dir / "manifest.jsonl"
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        if "file_path" in record and not os.path.isabs(record["file_path"]):
                            record["file_path"] = str(self.cache_dir / record["file_path"])
                        self.samples_meta.append(record)
        elif self.cache_dir and (self.cache_dir / "manifest.json").exists():
            manifest_path = self.cache_dir / "manifest.json"
            with open(manifest_path, "r", encoding="utf-8") as f:
                records = json.load(f)
                for record in records:
                    if "file_path" in record and not os.path.isabs(record["file_path"]):
                        record["file_path"] = str(self.cache_dir / record["file_path"])
                    self.samples_meta.append(record)
        elif file_paths is not None:
            for p in file_paths:
                p_str = str(p)
                stem = Path(p_str).stem
                self.samples_meta.append({
                    "sample_id": stem,
                    "file_path": p_str,
                    "lon": 0.0,
                    "lat": 0.0,
                    "block_id": 0,
                })
        elif self.cache_dir and self.cache_dir.is_dir():
            # Discover .npz files directly from disk without loading full arrays
            npz_files = sorted(list(self.cache_dir.glob("*.npz")))
            for p in npz_files:
                p_str = str(p)
                stem = p.stem
                lon, lat, block_id = 0.0, 0.0, 0
                try:
                    with np.load(p_str) as data:
                        if "lon" in data:
                            lon = float(data["lon"])
                        if "lat" in data:
                            lat = float(data["lat"])
                        if "block_id" in data:
                            block_id = int(data["block_id"])
                        if "sample_id" in data:
                            stem = str(data["sample_id"])
                except Exception as e:
                    logger.warning(f"Failed to read header of {p_str}: {e}")
                self.samples_meta.append({
                    "sample_id": stem,
                    "file_path": p_str,
                    "lon": lon,
                    "lat": lat,
                    "block_id": block_id,
                })
        else:
            logger.warning("Initialized empty LazyGeospatialDataset.")

        logger.info(f"Initialized LazyGeospatialDataset with {len(self.samples_meta)} chips (Zero-RAM mode)")

    def __len__(self) -> int:
        return len(self.samples_meta)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.samples_meta[idx]
        file_path = meta["file_path"]

        with np.load(file_path) as data:
            t1_arr = np.nan_to_num(data["t1_optical"], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
            t2_arr = np.nan_to_num(data["t2_optical"], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
            tgt_arr = np.nan_to_num(data["target_mask"], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

            if tgt_arr.ndim == 2:
                tgt_arr = tgt_arr[np.newaxis, ...]

            sample_id = str(meta.get("sample_id", data.get("sample_id", f"chip_{idx:05d}")))
            lon = float(meta.get("lon", data.get("lon", 0.0)))
            lat = float(meta.get("lat", data.get("lat", 0.0)))
            block_id = int(meta.get("block_id", data.get("block_id", 0)))

            item = {
                "sample_id": sample_id,
                "t1": torch.from_numpy(t1_arr),
                "t2": torch.from_numpy(t2_arr),
                "target": torch.from_numpy(tgt_arr),
                "lon": torch.tensor(lon, dtype=torch.float32),
                "lat": torch.tensor(lat, dtype=torch.float32),
                "block_id": torch.tensor(block_id, dtype=torch.long),
            }

            if "sar_data" in data and data["sar_data"] is not None:
                sar_arr = np.nan_to_num(data["sar_data"], nan=0.0, posinf=0.0, neginf=-30.0).astype(np.float32)
                item["sar"] = torch.from_numpy(sar_arr)

            if "cloud_mask" in data and data["cloud_mask"] is not None:
                c_arr = np.nan_to_num(data["cloud_mask"], nan=0.0).astype(np.float32)
                if c_arr.ndim == 2:
                    c_arr = c_arr[np.newaxis, ...]
                item["cloud_mask"] = torch.from_numpy(c_arr)

            if self.transform is not None:
                item = self.transform(item)

            return item

    def get_coordinates(self) -> List[Tuple[float, float]]:
        """Return list of (lon, lat) tuples without loading image arrays into RAM."""
        return [(float(m.get("lon", 0.0)), float(m.get("lat", 0.0))) for m in self.samples_meta]

    def get_sample_ids(self) -> List[str]:
        """Return list of sample IDs without loading image arrays into RAM."""
        return [str(m.get("sample_id", f"chip_{i}")) for i, m in enumerate(self.samples_meta)]


def cache_dataset_to_npz(samples: List[GeoSample], output_dir: str) -> List[str]:
    """Serialize and cache geospatial training chips to local .npz files for high-speed offline training.

    Also writes a streaming manifest.jsonl for instant O(1) LazyGeospatialDataset loading.

    Args:
        samples: List of GeoSample objects.
        output_dir: Directory where .npz chips will be saved.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []
    manifest_path = os.path.join(output_dir, "manifest.jsonl")

    with open(manifest_path, "w", encoding="utf-8") as f_man:
        for idx, s in enumerate(samples):
            chip_id = s.sample_id or f"chip_{idx:05d}"
            file_name = f"{chip_id}.npz"
            file_path = os.path.join(output_dir, file_name)

            data_dict: Dict[str, Any] = {
                "sample_id": chip_id,
                "t1_optical": s.t1_optical,
                "t2_optical": s.t2_optical,
                "target_mask": s.target_mask,
                "lon": s.lon,
                "lat": s.lat,
                "block_id": s.block_id,
            }
            if s.sar_data is not None:
                data_dict["sar_data"] = s.sar_data
            if s.cloud_mask is not None:
                data_dict["cloud_mask"] = s.cloud_mask

            np.savez_compressed(file_path, **data_dict)
            saved_paths.append(file_path)

            manifest_record = {
                "sample_id": chip_id,
                "file_path": file_name,
                "lon": float(s.lon),
                "lat": float(s.lat),
                "block_id": int(s.block_id),
            }
            f_man.write(json.dumps(manifest_record) + "\n")

    logger.info(f"Cached {len(saved_paths)} geospatial chips and manifest to {output_dir}")
    return saved_paths


def load_dataset_from_npz(
    cache_dir: str,
    lazy: bool = False,
) -> Union[List[GeoSample], LazyGeospatialDataset]:
    """Load cached .npz geospatial chips from directory.

    Args:
        cache_dir: Directory containing .npz files.
        lazy: If True, returns a zero-RAM LazyGeospatialDataset for large collections (10,000+ chips).
              If False (default), returns in-memory List[GeoSample] for backward compatibility.

    Returns:
        LazyGeospatialDataset (if lazy=True) or List of GeoSample objects (if lazy=False).
    """
    if lazy:
        return LazyGeospatialDataset(cache_dir=cache_dir)

    path = Path(cache_dir)
    npz_files = sorted(list(path.glob("*.npz")))
    samples: List[GeoSample] = []

    for p in npz_files:
        with np.load(p) as data:
            s = GeoSample(
                sample_id=str(data["sample_id"]),
                t1_optical=data["t1_optical"],
                t2_optical=data["t2_optical"],
                target_mask=data["target_mask"],
                sar_data=data["sar_data"] if "sar_data" in data else None,
                cloud_mask=data["cloud_mask"] if "cloud_mask" in data else None,
                lon=float(data["lon"]) if "lon" in data else 0.0,
                lat=float(data["lat"]) if "lat" in data else 0.0,
                block_id=int(data["block_id"]) if "block_id" in data else 0,
            )
            samples.append(s)

    logger.info(f"Loaded {len(samples)} cached chips from {cache_dir}")
    return samples


def _extract_coord(item: Any) -> Tuple[float, float]:
    """Helper to extract (lon, lat) float coordinates from any representation."""
    if isinstance(item, GeoSample):
        return item.lon, item.lat
    if isinstance(item, dict):
        return float(item.get("lon", 0.0)), float(item.get("lat", 0.0))
    if isinstance(item, (tuple, list)):
        return float(item[0]), float(item[1])
    return float(getattr(item, "lon", 0.0)), float(getattr(item, "lat", 0.0))


class SpatialBlockSplitter:
    """Partitions geospatial coordinates into spatially buffered training/validation blocks.

    In Earth observation, neighboring pixels and adjacent image tiles exhibit high
    spatial autocorrelation. Random pixel/tile splitting leads to severe optimistic
    evaluation bias. This splitter assigns discrete spatial grid cells to splits
    and enforces a buffer zone where boundary blocks are discarded.

    Compatible with in-memory List[GeoSample], zero-RAM LazyGeospatialDataset, or coordinate lists.
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
        self,
        samples: Union[List[GeoSample], LazyGeospatialDataset, List[Dict[str, Any]], List[Tuple[float, float]]],
    ) -> Tuple[Dict[int, List[int]], Dict[int, Tuple[float, float]]]:
        """Assign each sample to a 2D integer spatial block ID based on (lon, lat).

        Returns:
            (block_to_indices, block_centroids)
        """
        block_to_indices: Dict[int, List[int]] = {}
        block_centroids: Dict[int, Tuple[float, float]] = {}

        if hasattr(samples, "samples_meta"):
            items: Union[List[Any], LazyGeospatialDataset] = samples.samples_meta
        else:
            items = samples

        for idx, s in enumerate(items):
            s_lon, s_lat = _extract_coord(s)

            # Compute discrete grid indices
            col = int(np.floor(s_lon / self.grid_size))
            row = int(np.floor(s_lat / self.grid_size))
            # Cantor-like unique 2D grid hash
            block_id = int((col * 73856093) ^ (row * 19349663)) % 1000000

            # Store block_id back if object is mutable
            if isinstance(s, GeoSample):
                s.block_id = block_id
            elif isinstance(s, dict):
                s["block_id"] = block_id
            elif hasattr(s, "block_id"):
                try:
                    s.block_id = block_id
                except Exception:
                    pass

            if block_id not in block_to_indices:
                block_to_indices[block_id] = []
                block_centroids[block_id] = (
                    (col + 0.5) * self.grid_size,
                    (row + 0.5) * self.grid_size,
                )
            block_to_indices[block_id].append(idx)

        return block_to_indices, block_centroids

    def split(
        self,
        samples: Union[List[GeoSample], LazyGeospatialDataset, List[Dict[str, Any]], List[Tuple[float, float]]],
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
            if hasattr(samples, "samples_meta"):
                items: Union[List[Any], LazyGeospatialDataset] = samples.samples_meta
            else:
                items = samples

            eval_coords = [_extract_coord(items[i]) for i in (val_idx + test_idx)]
            buf_sq = self.buffer_size**2
            train_idx = []

            for i in raw_train_idx:
                s_lon, s_lat = _extract_coord(items[i])
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
