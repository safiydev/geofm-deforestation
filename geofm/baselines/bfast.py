"""BFAST-style OLS-MOSUM (Moving Sum of Residuals) structural break detector.

Identifies abrupt shifts in time-series trajectories by computing moving sums
of regression residuals against empirical boundary processes.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class BFASTResult:
    break_detected: bool
    break_index: Optional[int]
    mosum_values: np.ndarray
    boundary: float


class BFASTMonitor:
    """OLS-MOSUM structural break monitor for satellite time series."""

    def __init__(
        self,
        bandwidth_fraction: float = 0.25,
        critical_value: float = 1.36,  # Standard 95% significance level boundary
    ):
        """
        Args:
            bandwidth_fraction: Window length as fraction of historical reference period.
            critical_value: Critical value boundary for MOSUM test (1.36 corresponds to alpha=0.05).
        """
        self.bandwidth_fraction = bandwidth_fraction
        self.critical_value = critical_value

    def detect_break(
        self, series: np.ndarray, history_length: int
    ) -> BFASTResult:
        """Run OLS-MOSUM monitoring on a time-series vector.

        Args:
            series: 1D array of observations (e.g. NDVI or SAR backscatter).
            history_length: Number of points in the stable reference history period.

        Returns:
            BFASTResult with break detection flag, break location, and MOSUM path.
        """
        n = len(series)
        m = history_length
        if m < 4 or n <= m:
            return BFASTResult(False, None, np.array([]), self.critical_value)

        # 1. Fit historical linear trend: y = a + b * t
        t_hist = np.arange(m, dtype=np.float64)
        A = np.column_stack([np.ones(m), t_hist])
        coeffs, _, _, _ = np.linalg.lstsq(A, series[:m], rcond=None)

        # Compute reference standard deviation
        residuals_hist = series[:m] - (A @ coeffs)
        sigma = np.std(residuals_hist)
        sigma = max(sigma, 1e-4)

        # 2. Predict full series and compute residuals
        t_all = np.arange(n, dtype=np.float64)
        A_all = np.column_stack([np.ones(n), t_all])
        residuals = series - (A_all @ coeffs)

        # Bandwidth h
        h = max(2, int(m * self.bandwidth_fraction))

        # 3. Compute MOSUM process for monitoring period [m, n)
        mosum = np.zeros(n, dtype=np.float64)
        break_detected = False
        break_index = None

        for t in range(h - 1, n):
            # Sum residuals in sliding window [t - h + 1, t]
            window_res = residuals[t - h + 1 : t + 1]
            # Empirical MOSUM normalization: (1 / (sigma * sqrt(m))) * sum(res)
            mo_val = np.sum(window_res) / (sigma * np.sqrt(m))
            mosum[t] = mo_val

            # In monitoring period (t >= m), check boundary crossing
            if t >= m and not break_detected:
                # For deforestation, we test for negative drops in vegetation
                if mo_val < -self.critical_value:
                    break_detected = True
                    break_index = t

        return BFASTResult(
            break_detected=break_detected,
            break_index=break_index,
            mosum_values=mosum,
            boundary=self.critical_value,
        )
