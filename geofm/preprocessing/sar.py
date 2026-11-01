"""Synthetic Aperture Radar (SAR) preprocessing for Sentinel-1 C-band data.

Implements radiometric calibration (amplitude to dB), VH/VV cross-ratio computation,
adaptive speckle filtering (Lee filter and Gamma MAP filter), and temporal
outlier/moisture mitigation via Hampel filtering.
"""

from typing import Dict, Tuple
import numpy as np


class SARPreprocessor:
    """Preprocessor for Synthetic Aperture Radar (SAR) Sentinel-1 data."""

    def __init__(self, num_looks: float = 4.4, eps: float = 1e-7):
        """
        Args:
            num_looks: Equivalent Number of Looks (ENL) for Sentinel-1 GRD IW (~4.4).
            eps: Numerical stability constant.
        """
        self.num_looks = num_looks
        self.eps = eps

    def to_decibels(self, intensity: np.ndarray) -> np.ndarray:
        """Convert linear intensity to decibels (dB).

        dB = 10 * log10(intensity + eps)
        """
        intensity_pos = np.clip(intensity, self.eps, None)
        return 10.0 * np.log10(intensity_pos)

    def to_linear(self, db: np.ndarray) -> np.ndarray:
        """Convert decibels (dB) back to linear intensity.

        intensity = 10^(dB / 10)
        """
        return np.power(10.0, db / 10.0)

    def compute_vh_vv_ratio(
        self, vh: np.ndarray, vv: np.ndarray, in_db: bool = False
    ) -> np.ndarray:
        """Compute the cross-polarization ratio (VH / VV).

        The ratio VH/VV mitigates topographic variations and highlights
        canopy volume scattering vs bare surface scattering.

        Args:
            vh: VH channel (linear intensity or dB).
            vv: VV channel (linear intensity or dB).
            in_db: Whether inputs are already in decibels.

        Returns:
            Ratio in linear scale (or difference if in dB).
        """
        if in_db:
            return vh - vv
        vv_safe = np.where(vv <= self.eps, self.eps, vv)
        return np.clip(vh / vv_safe, 0.0, 10.0)

    def lee_filter(
        self, image: np.ndarray, window_size: int = 5
    ) -> np.ndarray:
        """Apply adaptive Lee speckle filter to SAR linear intensity image.

        The Lee filter balances speckle noise reduction in homogeneous areas
        with preservation of edges, logging roads, and clear-cut boundaries.

        Args:
            image: 2D array of linear intensity.
            window_size: Odd integer for kernel size (default 5).

        Returns:
            Filtered 2D intensity array.
        """
        if window_size % 2 == 0:
            window_size += 1
        pad = window_size // 2

        padded = np.pad(image, pad, mode="reflect")
        filtered = np.zeros_like(image, dtype=np.float32)

        # Theoretical noise variance for multi-look intensity: sigma_v^2 = 1 / ENL
        noise_var = 1.0 / self.num_looks

        # Efficient sliding box sum via 2D convolution or cumsum
        h, w = image.shape
        # Vectorized window view using numpy sliding_window_view if available or explicit looping
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, (window_size, window_size)
        )
        # windows shape: (h, w, window_size, window_size)
        local_mean = np.mean(windows, axis=(-2, -1))
        local_var = np.var(windows, axis=(-2, -1))

        # Weight W = max(0, (local_var - noise_var * local_mean^2) / (local_var + eps))
        numerator = local_var - (noise_var * (local_mean ** 2))
        denominator = local_var + self.eps
        weight = np.clip(numerator / denominator, 0.0, 1.0)

        # Output: mean + weight * (center_pixel - mean)
        filtered = local_mean + weight * (image - local_mean)
        return np.clip(filtered, 0.0, None)

    def gamma_map_filter(
        self, image: np.ndarray, window_size: int = 5
    ) -> np.ndarray:
        """Apply Gamma Maximum A Posteriori (MAP) speckle filter.

        Assumes Gamma distributed radar cross section and multiplicative noise.

        Args:
            image: 2D array of linear intensity.
            window_size: Odd filter window dimension.

        Returns:
            MAP filtered intensity array.
        """
        if window_size % 2 == 0:
            window_size += 1
        pad = window_size // 2
        padded = np.pad(image, pad, mode="reflect")

        windows = np.lib.stride_tricks.sliding_window_view(
            padded, (window_size, window_size)
        )
        local_mean = np.mean(windows, axis=(-2, -1))
        local_var = np.var(windows, axis=(-2, -1))

        enl = self.num_looks
        # Coefficient of variation
        ci = np.sqrt(np.clip(local_var, 0, None)) / (local_mean + self.eps)
        cu = 1.0 / np.sqrt(enl)
        cmax = np.sqrt(2.0) * cu

        filtered = np.copy(image)
        # Case 1: Homogeneous (ci <= cu) -> pure averaging
        homog_mask = ci <= cu
        filtered[homog_mask] = local_mean[homog_mask]

        # Case 2: Extreme heterogenous / point target (ci >= cmax) -> keep original
        # Case 3: Heterogeneous area (cu < ci < cmax) -> Gamma MAP solution
        map_mask = (ci > cu) & (ci < cmax)
        if np.any(map_mask):
            b = (1.0 + cu**2) / (ci[map_mask] ** 2 - cu**2 + self.eps)
            alpha = (b + 1.0) / (enl * (cu**2) + self.eps)
            term1 = (alpha - enl - 1.0) * local_mean[map_mask]
            disc = (term1**2) + 4.0 * alpha * enl * image[map_mask] * local_mean[map_mask]
            filtered[map_mask] = (term1 + np.sqrt(np.clip(disc, 0, None))) / (
                2.0 * alpha + self.eps
            )

        return np.clip(filtered, 0.0, None)

    def hampel_temporal_filter(
        self, series: np.ndarray, window_size: int = 5, n_sigmas: float = 3.0
    ) -> np.ndarray:
        """Apply Hampel filter along the temporal axis to remove moisture anomalies.

        Significant rainfall can cause temporary spikes in backscatter due to
        dielectric changes in wet foliage or soil. The Hampel filter uses median
        and Median Absolute Deviation (MAD) to replace episodic spikes.

        Args:
            series: 3D or 1D array where axis 0 is time (T, ...).
            window_size: Temporal sliding window length (odd).
            n_sigmas: Number of standard deviations threshold (default 3.0).

        Returns:
            Filtered series with soil-moisture outliers suppressed.
        """
        t_len = series.shape[0]
        filtered = np.copy(series)
        pad = window_size // 2

        # 1.4826 is normal consistency factor for median absolute deviation
        c = 1.4826

        for t in range(t_len):
            t_min = max(0, t - pad)
            t_max = min(t_len, t + pad + 1)
            window_slice = series[t_min:t_max]
            med = np.median(window_slice, axis=0)
            mad = c * np.median(np.abs(window_slice - med), axis=0)
            threshold = n_sigmas * mad

            diff = np.abs(series[t] - med)
            outlier_mask = diff > threshold
            if series.ndim == 1:
                if bool(outlier_mask):
                    filtered[t] = med
            else:
                filtered[t] = np.where(outlier_mask, med, filtered[t])

        return filtered

    def process_sentinel1_pair(
        self, vv: np.ndarray, vh: np.ndarray, filter_type: str = "lee"
    ) -> Dict[str, np.ndarray]:
        """Process VV and VH polarizations into filtered, calibrated feature stack.

        Args:
            vv: Raw VV intensity array (H, W).
            vh: Raw VH intensity array (H, W).
            filter_type: 'lee' or 'gamma_map'.

        Returns:
            Dictionary with filtered channels in linear and dB scale, plus VH/VV ratio.
        """
        # Ensure non-negative linear intensity
        vv_lin = np.clip(vv.astype(np.float32), 0.0, None)
        vh_lin = np.clip(vh.astype(np.float32), 0.0, None)

        if filter_type == "gamma_map":
            vv_filt = self.gamma_map_filter(vv_lin)
            vh_filt = self.gamma_map_filter(vh_lin)
        else:
            vv_filt = self.lee_filter(vv_lin)
            vh_filt = self.lee_filter(vh_lin)

        vv_db = self.to_decibels(vv_filt)
        vh_db = self.to_decibels(vh_filt)
        ratio_db = self.compute_vh_vv_ratio(vh_db, vv_db, in_db=True)
        ratio_lin = self.compute_vh_vv_ratio(vh_filt, vv_filt, in_db=False)

        return {
            "vv_filtered": vv_filt,
            "vh_filtered": vh_filt,
            "vv_db": vv_db,
            "vh_db": vh_db,
            "ratio_db": ratio_db,
            "ratio_linear": ratio_lin,
            "stacked_features": np.stack([vv_db, vh_db, ratio_db], axis=0),
        }
