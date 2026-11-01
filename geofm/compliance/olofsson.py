"""Olofsson et al. (2014) Area-Adjusted Accuracy Assessment and 95% Confidence Intervals.

Implements the international standard for estimating unbiased land-change areas,
error-adjusted overall accuracy, user's accuracy, producer's accuracy,
and standard errors from stratified random sampling.
Reference: Olofsson et al. (2014) "Good practices for estimating area and assessing accuracy
of land change", Remote Sensing of Environment, 148, 42-57.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class OlofssonReport:
    """Statistical accuracy report with area-adjusted estimates and 95% confidence intervals."""
    overall_accuracy: float
    oa_standard_error: float
    oa_ci_95: Tuple[float, float]
    users_accuracy: Dict[str, float]
    producers_accuracy: Dict[str, float]
    pa_standard_errors: Dict[str, float]
    pa_ci_95: Dict[str, Tuple[float, float]]
    estimated_area_proportions: Dict[str, float]
    estimated_areas_ha: Dict[str, float]
    area_margins_of_error_ha: Dict[str, float]
    class_names: List[str]

    def summary(self) -> str:
        lines = [
            "=================================================================",
            "  Olofsson et al. (2014) Area-Adjusted Accuracy Assessment",
            "=================================================================",
            f"Overall Accuracy (Area-Adjusted): {self.overall_accuracy * 100:.2f}% "
            f"+/- {1.96 * self.oa_standard_error * 100:.2f}% (95% CI: [{self.oa_ci_95[0]*100:.2f}%, {self.oa_ci_95[1]*100:.2f}%])",
            "",
            "Class-specific Accuracy & Unbiased Area Estimates:",
        ]
        for c in self.class_names:
            ua = self.users_accuracy.get(c, 0.0) * 100
            pa = self.producers_accuracy.get(c, 0.0) * 100
            pa_se = self.pa_standard_errors.get(c, 0.0) * 100
            area = self.estimated_areas_ha.get(c, 0.0)
            moe = self.area_margins_of_error_ha.get(c, 0.0)
            lines.append(
                f"  [{c}]:\n"
                f"    User's Accuracy (Precision):     {ua:.2f}%\n"
                f"    Producer's Accuracy (Recall):    {pa:.2f}% +/- {1.96 * pa_se:.2f}%\n"
                f"    Unbiased Estimated Area:         {area:,.1f} ha +/- {moe:,.1f} ha (95% CI)"
            )
        lines.append("=================================================================")
        return "\n".join(lines)


class OlofssonAssessor:
    """Area-adjusted accuracy assessment calculator using stratified random sampling."""

    def __init__(self, class_names: Optional[List[str]] = None):
        if class_names is None:
            self.class_names = ["Deforestation", "Stable Forest"]
        else:
            self.class_names = class_names

    def compute(
        self,
        sample_matrix: np.ndarray,
        strata_areas_ha: np.ndarray,
    ) -> OlofssonReport:
        """Compute area-adjusted accuracy and confidence intervals.

        Args:
            sample_matrix: 2D array of sample counts n_ij (shape K x K).
                           Rows (i) = Mapped/Predicted class.
                           Columns (j) = Reference/Ground truth class.
            strata_areas_ha: 1D array of total mapped area per stratum A_i (shape K).

        Returns:
            OlofssonReport containing area-adjusted metrics, standard errors, and 95% CIs.
        """
        k = len(self.class_names)
        n = np.asarray(sample_matrix, dtype=np.float64)
        a = np.asarray(strata_areas_ha, dtype=np.float64)

        total_area = np.sum(a)
        # Weight of each stratum: W_i = A_i / A_tot
        w = a / total_area

        # Row totals: n_i_dot
        n_row = np.sum(n, axis=1)
        # Avoid division by zero
        n_row_safe = np.where(n_row == 0, 1.0, n_row)

        # 1. Error matrix of estimated area proportions: p_ij = W_i * (n_ij / n_i_dot)
        p = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            p[i, :] = w[i] * (n[i, :] / n_row_safe[i])

        # 2. Area-adjusted overall accuracy: OA = sum(p_ii)
        oa = float(np.sum(np.diag(p)))

        # Standard error of overall accuracy:
        # S(OA) = sqrt( sum_i [ W_i^2 * (n_ii / n_i.) * (1 - n_ii / n_i.) / (n_i. - 1) ] )
        var_oa = 0.0
        for i in range(k):
            if n_row[i] > 1:
                u_i = n[i, i] / n_row[i]
                var_oa += (w[i] ** 2) * (u_i * (1.0 - u_i)) / (n_row[i] - 1.0)
        se_oa = float(np.sqrt(max(0.0, var_oa)))
        oa_ci_95 = (max(0.0, oa - 1.96 * se_oa), min(1.0, oa + 1.96 * se_oa))

        # 3. User's Accuracy (Precision): UA_i = p_ii / p_i. = n_ii / n_i.
        users_acc = {}
        for i, c in enumerate(self.class_names):
            ua_val = n[i, i] / n_row_safe[i] if n_row[i] > 0 else 0.0
            users_acc[c] = float(ua_val)

        # 4. Producer's Accuracy (Recall): PA_j = p_jj / p_.j
        p_col = np.sum(p, axis=0)
        p_col_safe = np.where(p_col == 0, 1e-9, p_col)

        producers_acc = {}
        pa_std_errors = {}
        pa_ci_95 = {}

        for j, c in enumerate(self.class_names):
            pa_val = float(p[j, j] / p_col_safe[j])
            producers_acc[c] = pa_val

            # Standard error of PA using equation (7) from Olofsson et al. (2014):
            # Var(PA_j) approx = (1 / p_.j^2) * [ W_j^2 * (1 - PA_j)^2 * u_j*(1-u_j)/(n_j.-1)
            #   + PA_j^2 * sum_{i != j} ( W_i^2 * (n_ij / n_i.)*(1 - n_ij/n_i.) / (n_i. - 1) ) ]
            var_term = 0.0
            for i in range(k):
                if n_row[i] > 1:
                    prop_ij = n[i, j] / n_row[i]
                    comp = prop_ij * (1.0 - prop_ij) / (n_row[i] - 1.0)
                    if i == j:
                        var_term += (w[i] ** 2) * ((1.0 - pa_val) ** 2) * comp
                    else:
                        var_term += (w[i] ** 2) * (pa_val ** 2) * comp

            var_pa = var_term / (p_col_safe[j] ** 2)
            se_pa = float(np.sqrt(max(0.0, var_pa)))
            pa_std_errors[c] = se_pa
            pa_ci_95[c] = (max(0.0, pa_val - 1.96 * se_pa), min(1.0, pa_val + 1.96 * se_pa))

        # 5. Unbiased Estimated Area of each class: A_j = A_tot * p_.j
        est_proportions = {}
        est_areas = {}
        moe_areas = {}

        for j, c in enumerate(self.class_names):
            prop_j = float(p_col[j])
            est_proportions[c] = prop_j

            # Variance of estimated area proportion: Var(p_.j) = sum_i [ W_i^2 * (n_ij / n_i.) * (1 - n_ij/n_i.) / (n_i. - 1) ]
            var_prop = 0.0
            for i in range(k):
                if n_row[i] > 1:
                    prop_ij = n[i, j] / n_row[i]
                    var_prop += (w[i] ** 2) * (prop_ij * (1.0 - prop_ij)) / (n_row[i] - 1.0)
            se_prop = np.sqrt(max(0.0, var_prop))

            area_j = prop_j * total_area
            moe_j = 1.96 * se_prop * total_area  # 95% margin of error in hectares

            est_areas[c] = float(area_j)
            moe_areas[c] = float(moe_j)

        return OlofssonReport(
            overall_accuracy=oa,
            oa_standard_error=se_oa,
            oa_ci_95=oa_ci_95,
            users_accuracy=users_acc,
            producers_accuracy=producers_acc,
            pa_standard_errors=pa_std_errors,
            pa_ci_95=pa_ci_95,
            estimated_area_proportions=est_proportions,
            estimated_areas_ha=est_areas,
            area_margins_of_error_ha=moe_areas,
            class_names=self.class_names,
        )
