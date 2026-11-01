"""Due Diligence Statement (DDS) Compliance Auditor for EUDR.

Cross-references supply chain production plots with AI deforestation detections
to produce mathematically verifiable, legally auditable compliance verdicts.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from geofm.compliance.eudr import (
    EUDR_CUTOFF_DATE,
    EUDR_REGULATED_COMMODITIES,
    EUDRValidator,
)
from geofm.postprocessing.geometry_utils import (
    is_point_in_polygon,
    polygons_intersect,
)


@dataclass
class PlotAuditResult:
    plot_id: str
    status: str  # 'COMPLIANT', 'NON_COMPLIANT', 'INVALID_GEOMETRY'
    area_ha: float
    violations: List[str]
    alert_overlaps: List[Dict[str, Any]]


@dataclass
class DDSAuditReport:
    statement_id: str
    operator_name: str
    commodity: str
    country_of_production: str
    overall_status: str  # 'COMPLIANT' or 'NON_COMPLIANT' or 'REJECTED'
    risk_score: float  # 0.0 (clean) to 1.0 (severe violation)
    plot_results: List[PlotAuditResult]
    audit_timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement_id": self.statement_id,
            "operator_name": self.operator_name,
            "commodity": self.commodity,
            "country": self.country_of_production,
            "overall_status": self.overall_status,
            "risk_score": round(self.risk_score, 3),
            "audit_timestamp": self.audit_timestamp,
            "total_plots": len(self.plot_results),
            "plots": [
                {
                    "plot_id": p.plot_id,
                    "status": p.status,
                    "area_ha": round(p.area_ha, 2),
                    "violations": p.violations,
                    "alert_overlaps": p.alert_overlaps,
                }
                for p in self.plot_results
            ],
        }


class DDSAuditor:
    """Audits supplier Due Diligence Statements against AI deforestation alerts."""

    def __init__(self, validator: Optional[EUDRValidator] = None):
        self.validator = validator if validator is not None else EUDRValidator()

    def audit_statement(
        self,
        statement_id: str,
        operator_name: str,
        commodity: str,
        country: str,
        plots: List[Dict[str, Any]],
        deforestation_alerts: List[Dict[str, Any]],
    ) -> DDSAuditReport:
        """Audit a complete Due Diligence Statement against detected deforestation alerts.

        Args:
            statement_id: Unique identifier for the DDS filing.
            operator_name: Company or cooperative submitting statement.
            commodity: Commodity type (e.g., 'cattle', 'cocoa', 'coffee', 'soya', etc.).
            country: ISO country code or name.
            plots: List of plot dictionaries with keys:
                   'plot_id', 'geometry_type' ('point'|'polygon'), 'coordinates', 'declared_area_ha'.
            deforestation_alerts: List of alert dictionaries with keys:
                   'alert_id', 'geometry', 'detection_date' (ISO string or datetime), 'area_ha'.

        Returns:
            DDSAuditReport containing overall status and per-plot findings.
        """
        commodity_clean = commodity.lower().strip()
        plot_results: List[PlotAuditResult] = []
        has_invalid = False
        has_violations = False

        # Validate commodity against EUDR Annex I scope
        comm_valid, comm_msg = self.validator.validate_commodity(commodity_clean)
        if not comm_valid:
            has_invalid = True

        for plot in plots:
            p_id = plot.get("plot_id", "unknown-plot")
            g_type = plot.get("geometry_type", "polygon")
            coords = plot.get("coordinates", [])
            dec_area = plot.get("declared_area_ha")

            # 1. Validate geometry rules
            is_valid, v_msg, calc_area = self.validator.validate_plot_geometry(
                g_type, coords, declared_area_ha=dec_area
            )

            if not is_valid:
                has_invalid = True
                plot_results.append(
                    PlotAuditResult(
                        plot_id=p_id,
                        status="INVALID_GEOMETRY",
                        area_ha=calc_area,
                        violations=[v_msg],
                        alert_overlaps=[],
                    )
                )
                continue

            # Unwrap plot coords if nested Polygon ring [[ [lon, lat], ... ]]
            plot_poly_coords = coords
            if (
                g_type.lower() == "polygon"
                and len(plot_poly_coords) > 0
                and isinstance(plot_poly_coords[0], (list, tuple))
                and len(plot_poly_coords[0]) > 0
                and isinstance(plot_poly_coords[0][0], (list, tuple))
            ):
                plot_poly_coords = plot_poly_coords[0]

            # 2. Check spatial intersection with deforestation alerts
            plot_violations = []
            if not comm_valid:
                plot_violations.append(comm_msg)

            overlaps = []

            for alert in deforestation_alerts:
                alert_geom = alert.get("geometry", {})
                alert_type = alert_geom.get("type", "Polygon")
                alert_rings = []

                if alert_type == "Polygon":
                    raw_c = alert_geom.get("coordinates", [])
                    if raw_c and len(raw_c[0]) >= 3:
                        alert_rings.append(raw_c[0])
                elif alert_type == "MultiPolygon":
                    for poly_coords in alert_geom.get("coordinates", []):
                        if poly_coords and len(poly_coords[0]) >= 3:
                            alert_rings.append(poly_coords[0])

                if not alert_rings:
                    continue

                # Intersection test against any ring
                intersected = False
                for ring in alert_rings:
                    if g_type.lower() == "point":
                        if is_point_in_polygon(coords, ring):
                            intersected = True
                            break
                    else:
                        if polygons_intersect(plot_poly_coords, ring):
                            intersected = True
                            break

                if intersected:
                    # Parse detection date
                    dt_raw = alert.get("detection_date")
                    if isinstance(dt_raw, str):
                        try:
                            det_dt = datetime.fromisoformat(dt_raw)
                        except Exception:
                            det_dt = datetime.now(timezone.utc)
                    elif isinstance(dt_raw, datetime):
                        det_dt = dt_raw
                    else:
                        det_dt = datetime.now(timezone.utc)

                    is_violation, timing_msg = self.validator.verify_deforestation_timing(det_dt)
                    overlap_info = {
                        "alert_id": alert.get("alert_id"),
                        "detection_date": det_dt.isoformat(),
                        "alert_area_ha": alert.get("area_ha", 0.0),
                        "violates_cutoff": is_violation,
                        "details": timing_msg,
                    }
                    overlaps.append(overlap_info)

                    if is_violation:
                        plot_violations.append(
                            f"Post-2020 Deforestation Alert [{alert.get('alert_id')}] detected on plot: {timing_msg}"
                        )

            if plot_violations:
                has_violations = True
                status = "NON_COMPLIANT"
            else:
                status = "COMPLIANT"

            plot_results.append(
                PlotAuditResult(
                    plot_id=p_id,
                    status=status,
                    area_ha=calc_area,
                    violations=plot_violations,
                    alert_overlaps=overlaps,
                )
            )

        # Overall Status
        if not comm_valid:
            overall_status = "REJECTED_UNREGULATED_COMMODITY"
            risk_score = 1.0
        elif has_invalid:
            overall_status = "REJECTED_INVALID_GEOMETRY"
            risk_score = 1.0
        elif has_violations:
            overall_status = "NON_COMPLIANT"
            risk_score = 1.0
        else:
            overall_status = "COMPLIANT"
            risk_score = 0.0

        return DDSAuditReport(
            statement_id=statement_id,
            operator_name=operator_name,
            commodity=commodity_clean,
            country_of_production=country,
            overall_status=overall_status,
            risk_score=risk_score,
            plot_results=plot_results,
            audit_timestamp=datetime.now(timezone.utc).isoformat(),
        )
