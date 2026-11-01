"""Optical imagery preprocessing for Sentinel-2 and Planet NICFI.

Implements normalization, vegetation indices (NDVI, EVI, NDWI),
and cloud/shadow masking algorithms.
"""

from typing import Dict, Optional, Tuple
import numpy as np


class OpticalPreprocessor:
    """Preprocessor for multispectral optical satellite imagery."""

    def __init__(
        self,
        sensor_type: str = "sentinel2",
        cloud_threshold: float = 0.25,
        shadow_threshold: float = 0.12,
        eps: float = 1e-7,
    ):
        """
        Args:
            sensor_type: 'sentinel2' or 'planet_nicfi'.
            cloud_threshold: Reflectance threshold in blue/green for cloud detection.
            shadow_threshold: Reflectance threshold in NIR/SWIR for shadow detection.
            eps: Epsilon to avoid division by zero.
        """
        self.sensor_type = sensor_type.lower()
        self.cloud_threshold = cloud_threshold
        self.shadow_threshold = shadow_threshold
        self.eps = eps

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize digital numbers to surface reflectance [0, 1].

        Sentinel-2 L2A typically uses a scale factor of 10000.
        Planet NICFI typically uses 8-bit [0, 255] or 12-bit [0, 4095].
        """
        data = data.astype(np.float32)
        if self.sensor_type == "sentinel2":
            # If values exceed 1.0, scale by 10000 (standard ESA Copernicus scaling)
            if np.max(data) > 1.0:
                normalized = np.clip(data / 10000.0, 0.0, 1.0)
            else:
                normalized = np.clip(data, 0.0, 1.0)
        elif self.sensor_type == "planet_nicfi":
            if np.max(data) > 255.0:
                normalized = np.clip(data / 4095.0, 0.0, 1.0)
            elif np.max(data) > 1.0:
                normalized = np.clip(data / 255.0, 0.0, 1.0)
            else:
                normalized = np.clip(data, 0.0, 1.0)
        else:
            max_val = np.max(data)
            normalized = data / max_val if max_val > 0 else data
        return normalized

    def compute_ndvi(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Compute Normalized Difference Vegetation Index (NDVI).

        NDVI = (NIR - Red) / (NIR + Red + eps)
        Values range from -1.0 to 1.0. Dense tropical forest typically > 0.7.
        Deforested / bare soil typically < 0.3.
        """
        numerator = nir - red
        denominator = nir + red + self.eps
        ndvi = numerator / denominator
        return np.clip(ndvi, -1.0, 1.0)

    def compute_evi(
        self,
        blue: np.ndarray,
        red: np.ndarray,
        nir: np.ndarray,
        g: float = 2.5,
        c1: float = 6.0,
        c2: float = 7.5,
        l: float = 1.0,
    ) -> np.ndarray:
        """Compute Enhanced Vegetation Index (EVI).

        EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L + eps)
        Optimized for high biomass regions to reduce canopy background saturation.
        """
        denom = nir + (c1 * red) - (c2 * blue) + l + self.eps
        evi = g * (nir - red) / denom
        return np.clip(evi, -1.0, 1.5)

    def compute_ndwi(self, green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Compute Normalized Difference Water Index (NDWI) for surface water/moisture."""
        denom = green + nir + self.eps
        return np.clip((green - nir) / denom, -1.0, 1.0)

    def detect_cloud_and_shadow_mask(
        self,
        blue: np.ndarray,
        green: np.ndarray,
        red: np.ndarray,
        nir: np.ndarray,
        swir: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect cloud and cloud shadow pixels.

        Returns:
            cloud_mask: Boolean mask where True indicates cloud.
            shadow_mask: Boolean mask where True indicates cloud shadow.
        """
        # Clouds: Bright in visible, high reflectance across all optical bands
        brightness = (blue + green + red) / 3.0
        cloud_mask = (blue > self.cloud_threshold) & (brightness > self.cloud_threshold)

        # Shadows: Abrupt drop in NIR and SWIR, adjacent to bright clouds
        if swir is not None:
            shadow_cand = (nir < self.shadow_threshold) & (swir < self.shadow_threshold)
        else:
            shadow_cand = (nir < self.shadow_threshold) & (brightness < self.shadow_threshold)

        shadow_mask = shadow_cand & (~cloud_mask)
        return cloud_mask, shadow_mask

    def process_multispectral_cube(
        self,
        bands: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Process multispectral cube into normalized bands, indices, and quality masks.

        Args:
            bands: Dictionary containing 'blue', 'green', 'red', 'nir', and optional 'swir1'.

        Returns:
            Dictionary containing normalized channels, NDVI, EVI, and validity mask.
        """
        norm_bands = {k: self.normalize(v) for k, v in bands.items()}
        red = norm_bands["red"]
        nir = norm_bands["nir"]
        blue = norm_bands.get("blue", np.zeros_like(red))
        green = norm_bands.get("green", np.zeros_like(red))
        swir = norm_bands.get("swir1", None)

        ndvi = self.compute_ndvi(red, nir)
        evi = self.compute_evi(blue, red, nir)
        cloud_mask, shadow_mask = self.detect_cloud_and_shadow_mask(
            blue, green, red, nir, swir
        )
        valid_mask = ~(cloud_mask | shadow_mask)

        return {
            "normalized_bands": norm_bands,
            "ndvi": ndvi,
            "evi": evi,
            "cloud_mask": cloud_mask,
            "shadow_mask": shadow_mask,
            "valid_mask": valid_mask,
        }
