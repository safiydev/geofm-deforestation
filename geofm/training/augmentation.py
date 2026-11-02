"""Bitemporal geospatial data augmentation for satellite change detection.

Implements D4 dihedral group transformations (random 90/180/270 degree rotations,
horizontal flips, and vertical flips) applied synchronously across pre-event (T1),
post-event (T2), target masks, cloud masks, and SAR backscatter.
Applies mild bitemporal radiometric jitter (gain and offset) independently to T1 and T2
to simulate atmospheric haze and sensor viewing angle differences.
Effectively expands training dataset diversity by 8x with zero storage overhead.

Supports both native NumPy/PyTorch transformations and Albumentations pipelines.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


class BitemporalAugmentor:
    """Synchronous bitemporal augmentor supporting D4 dihedral symmetry and radiometric jitter."""

    def __init__(
        self,
        enable_flips: bool = True,
        enable_rot90: bool = True,
        enable_radiometric_jitter: bool = True,
        jitter_gain_range: Tuple[float, float] = (0.95, 1.05),
        jitter_offset_range: Tuple[float, float] = (-0.02, 0.02),
        use_albumentations: bool = False,
        p: float = 0.5,
    ):
        """
        Args:
            enable_flips: Random horizontal and vertical flips.
            enable_rot90: Random 90, 180, 270 degree rotations.
            enable_radiometric_jitter: Bitemporal gain and offset jitter.
            jitter_gain_range: Multiplicative gain factor bounds [min, max].
            jitter_offset_range: Additive offset bounds [min, max].
            use_albumentations: Prefer albumentations pipeline if installed.
            p: Probability of applying each individual augmentation.
        """
        self.enable_flips = enable_flips
        self.enable_rot90 = enable_rot90
        self.enable_radiometric_jitter = enable_radiometric_jitter
        self.gain_min, self.gain_max = jitter_gain_range
        self.offset_min, self.offset_max = jitter_offset_range
        self.p = p
        self.use_albumentations = use_albumentations and HAS_ALBUMENTATIONS

        if self.use_albumentations:
            self._alb_pipeline = A.Compose(
                [
                    A.HorizontalFlip(p=p),
                    A.VerticalFlip(p=p),
                    A.RandomRotate90(p=p),
                ],
                additional_targets={"t2": "image", "target": "mask"},
            )

    def _apply_geometric_numpy(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        target: np.ndarray,
        cloud_mask: Optional[np.ndarray] = None,
        sar: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Apply D4 dihedral transformations identically to all spatial channels."""
        # Random k * 90 degree rotation
        if self.enable_rot90:
            k = int(np.random.randint(0, 4))
            if k > 0:
                t1 = np.rot90(t1, k, axes=(-2, -1)).copy()
                t2 = np.rot90(t2, k, axes=(-2, -1)).copy()
                target = np.rot90(target, k, axes=(-2, -1)).copy()
                if cloud_mask is not None:
                    cloud_mask = np.rot90(cloud_mask, k, axes=(-2, -1)).copy()
                if sar is not None:
                    sar = np.rot90(sar, k, axes=(-2, -1)).copy()

        # Random horizontal flip
        if self.enable_flips and np.random.rand() < self.p:
            t1 = np.flip(t1, axis=-1).copy()
            t2 = np.flip(t2, axis=-1).copy()
            target = np.flip(target, axis=-1).copy()
            if cloud_mask is not None:
                cloud_mask = np.flip(cloud_mask, axis=-1).copy()
            if sar is not None:
                sar = np.flip(sar, axis=-1).copy()

        # Random vertical flip
        if self.enable_flips and np.random.rand() < self.p:
            t1 = np.flip(t1, axis=-2).copy()
            t2 = np.flip(t2, axis=-2).copy()
            target = np.flip(target, axis=-2).copy()
            if cloud_mask is not None:
                cloud_mask = np.flip(cloud_mask, axis=-2).copy()
            if sar is not None:
                sar = np.flip(sar, axis=-2).copy()

        return t1, t2, target, cloud_mask, sar

    def _apply_radiometric_jitter(self, arr: np.ndarray) -> np.ndarray:
        """Simulate atmospheric haze / sensor gain variations."""
        gain = np.random.uniform(self.gain_min, self.gain_max)
        offset = np.random.uniform(self.offset_min, self.offset_max)
        jittered = arr * gain + offset
        return np.clip(jittered, 0.0, 1.0)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentations to sample dictionary.

        Args:
            sample: Dict containing at least 't1', 't2', 'target'.

        Returns:
            Augmented sample dictionary.
        """
        is_torch = isinstance(sample.get("t1"), torch.Tensor)

        # Convert to numpy for processing
        t1 = sample["t1"].cpu().numpy() if is_torch else np.array(sample["t1"])
        t2 = sample["t2"].cpu().numpy() if is_torch else np.array(sample["t2"])
        target = sample["target"].cpu().numpy() if is_torch else np.array(sample["target"])

        cloud_mask = None
        if "cloud_mask" in sample and sample["cloud_mask"] is not None:
            cloud_mask = sample["cloud_mask"].cpu().numpy() if is_torch else np.array(sample["cloud_mask"])

        sar = None
        if "sar" in sample and sample["sar"] is not None:
            sar = sample["sar"].cpu().numpy() if is_torch else np.array(sample["sar"])

        # Geometric transforms
        t1, t2, target, cloud_mask, sar = self._apply_geometric_numpy(t1, t2, target, cloud_mask, sar)

        # Radiometric jitter (independent between T1 and T2)
        if self.enable_radiometric_jitter and np.random.rand() < self.p:
            t1 = self._apply_radiometric_jitter(t1)
            t2 = self._apply_radiometric_jitter(t2)

        out = dict(sample)
        if is_torch:
            out["t1"] = torch.from_numpy(t1.astype(np.float32))
            out["t2"] = torch.from_numpy(t2.astype(np.float32))
            out["target"] = torch.from_numpy(target.astype(np.float32))
            if cloud_mask is not None:
                out["cloud_mask"] = torch.from_numpy(cloud_mask.astype(np.float32))
            if sar is not None:
                out["sar"] = torch.from_numpy(sar.astype(np.float32))
        else:
            out["t1"] = t1.astype(np.float32)
            out["t2"] = t2.astype(np.float32)
            out["target"] = target.astype(np.float32)
            if cloud_mask is not None:
                out["cloud_mask"] = cloud_mask.astype(np.float32)
            if sar is not None:
                out["sar"] = sar.astype(np.float32)

        return out
