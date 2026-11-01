"""Bitemporal NDVI Differencing baseline for fast deforestation screening."""

from typing import Dict, Tuple
import numpy as np


class NDVIDeltaBaseline:
    """Detects deforestation via direct differencing of pre and post NDVI."""

    def __init__(self, delta_threshold: float = -0.25, post_ndvi_max: float = 0.45):
        """
        Args:
            delta_threshold: Minimum drop in NDVI to trigger deforestation flag (e.g. -0.25).
            post_ndvi_max: Maximum post-disturbance NDVI to confirm persistent clear-cut
                           (prevents false triggers on seasonal deciduous canopy shedding).
        """
        self.delta_threshold = delta_threshold
        self.post_ndvi_max = post_ndvi_max

    def detect_change(
        self, pre_ndvi: np.ndarray, post_ndvi: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute delta NDVI and binary change mask.

        Args:
            pre_ndvi: Pre-disturbance NDVI array (H, W).
            post_ndvi: Post-disturbance NDVI array (H, W).

        Returns:
            Dictionary with delta_ndvi and change_mask.
        """
        delta = post_ndvi - pre_ndvi
        # Condition 1: Drop exceeds threshold
        # Condition 2: Post value is genuinely low (bare soil or scrub)
        change_mask = (delta <= self.delta_threshold) & (post_ndvi <= self.post_ndvi_max)

        return {
            "delta_ndvi": delta,
            "change_mask": change_mask,
        }
