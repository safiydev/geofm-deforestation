"""Continuous Change Detection and Classification (CCDC) harmonic regression baseline.

Tracks continuous pixel spectral trajectories using multi-harmonic regression models
(capturing intra-annual phenology and multi-year trends) and identifies structural breaks
when consecutive observations systematically deviate beyond the model error bounds.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CCDCBreakResult:
    """Result of CCDC break detection for a single pixel or trajectory."""
    break_detected: bool
    break_index: Optional[int]
    break_time: Optional[float]
    rmse: float
    model_coefficients: Optional[np.ndarray]


class CCDCDirectDetector:
    """CCDC-style multi-harmonic regression change detector."""

    def __init__(
        self,
        consecutive_anomalies: int = 3,
        threshold_factor: float = 3.0,
        num_harmonics: int = 1,
        include_trend: bool = False,
        min_rmse: float = 0.02,
    ):
        """
        Args:
            consecutive_anomalies: Number of consecutive outliers required to confirm break.
            threshold_factor: Multiplier for RMSE to define the anomaly boundary (typically 3.0).
            num_harmonics: Number of harmonic pairs (1 or 2) for seasonal modeling.
            include_trend: Whether to fit a linear slope term (only advised for multi-year data).
            min_rmse: Minimum noise floor to prevent division/false alarms on clean data.
        """
        self.consecutive_anomalies = consecutive_anomalies
        self.threshold_factor = threshold_factor
        self.num_harmonics = num_harmonics
        self.include_trend = include_trend
        self.min_rmse = min_rmse

    def _build_design_matrix(
        self, times: np.ndarray, include_trend: Optional[bool] = None
    ) -> np.ndarray:
        """Construct design matrix X with intercept, optional slope, and sine/cosine harmonics."""
        omega = 2.0 * np.pi / 365.25
        cols = [np.ones_like(times)]
        inc_t = self.include_trend if include_trend is None else include_trend
        if inc_t:
            cols.append(times)
        for h in range(1, self.num_harmonics + 1):
            cols.append(np.cos(h * omega * times))
            cols.append(np.sin(h * omega * times))
        return np.column_stack(cols)

    def fit_and_detect(
        self, times: np.ndarray, values: np.ndarray, min_init_points: int = 8
    ) -> CCDCBreakResult:
        """Fit baseline on initial observations and monitor remaining series for break.

        Args:
            times: 1D array of acquisition timestamps (e.g. day of year or decimal year).
            values: 1D array of spectral values (e.g. NDVI or NIR reflectance).
            min_init_points: Minimum number of historical observations to train the baseline.

        Returns:
            CCDCBreakResult containing break flag, break index, and model error.
        """
        n = len(values)
        if n < min_init_points + self.consecutive_anomalies:
            return CCDCBreakResult(
                break_detected=False, break_index=None, break_time=None, rmse=0.0, model_coefficients=None
            )

        # 1. Fit historical baseline model on stable initialization window
        init_times = times[:min_init_points]
        init_vals = values[:min_init_points]

        X_init = self._build_design_matrix(init_times)
        coeffs, _, _, _ = np.linalg.lstsq(X_init, init_vals, rcond=None)

        pred_init = X_init @ coeffs
        residuals_init = init_vals - pred_init
        rmse = float(np.sqrt(np.mean(residuals_init ** 2)))
        rmse = max(rmse, self.min_rmse)

        # 2. Sequentially monitor incoming observations
        X_all = self._build_design_matrix(times)
        pred_all = X_all @ coeffs
        residuals = values - pred_all

        consec_count = 0
        break_idx = None

        for i in range(min_init_points, n):
            # For deforestation detection, we monitor negative deviations (abrupt loss of vegetation)
            if residuals[i] < -self.threshold_factor * rmse:
                consec_count += 1
                if consec_count >= self.consecutive_anomalies:
                    break_idx = i - self.consecutive_anomalies + 1
                    return CCDCBreakResult(
                        break_detected=True,
                        break_index=break_idx,
                        break_time=float(times[break_idx]),
                        rmse=rmse,
                        model_coefficients=coeffs,
                    )
            else:
                consec_count = 0

        return CCDCBreakResult(
            break_detected=False,
            break_index=None,
            break_time=None,
            rmse=rmse,
            model_coefficients=coeffs,
        )

    def detect_raster_cube(
        self, times: np.ndarray, cube: np.ndarray, min_init_points: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run CCDC change detection across spatial raster stack (T, H, W).

        Returns:
            change_mask: 2D boolean array (H, W) where True = structural break detected.
            change_time: 2D float array (H, W) recording the timestamp of disturbance.
        """
        t_len, h, w = cube.shape
        change_mask = np.zeros((h, w), dtype=bool)
        change_times = np.full((h, w), np.nan, dtype=np.float32)

        for r in range(h):
            for c in range(w):
                res = self.fit_and_detect(
                    times, cube[:, r, c], min_init_points=min_init_points
                )
                if res.break_detected:
                    change_mask[r, c] = True
                    change_times[r, c] = res.break_time

        return change_mask, change_times
