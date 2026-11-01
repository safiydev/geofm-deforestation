"""Tests for EUDR compliance verification, DDS auditor, and Olofsson accuracy assessment."""

from datetime import datetime, timezone
import numpy as np
import pytest

from geofm.compliance.eudr import EUDRValidator, EUDR_CUTOFF_DATE
from geofm.compliance.dds_auditor import DDSAuditor
from geofm.compliance.olofsson import OlofssonAssessor


def test_eudr_validator_rules():
    val = EUDRValidator()

    # Rule 1: Point <= 4 ha is valid
    ok, msg, area = val.validate_plot_geometry(
        "point", [-60.123456, -3.123456], declared_area_ha=2.5
    )
    assert ok == True
    assert area == 2.5

    # Rule 2: Point > 4 ha is strictly rejected under EUDR
    ok, msg, area = val.validate_plot_geometry(
        "point", [-60.123456, -3.123456], declared_area_ha=5.5
    )
    assert ok == False
    assert "exceeds 4 ha threshold" in msg

    # Rule 3: Unclosed polygon is rejected
    unclosed_coords = [
        [-60.0, -3.0],
        [-59.9, -3.0],
        [-59.9, -2.9],
        [-60.0, -2.9],  # Missing closing point [-60.0, -3.0]
    ]
    ok, msg, area = val.validate_plot_geometry("polygon", unclosed_coords)
    assert ok == False
    assert "not closed" in msg

    # Rule 4: Temporal cutoff check
    # Post-2020 disturbance is a violation
    violation, v_msg = val.verify_deforestation_timing(datetime(2022, 5, 10, tzinfo=timezone.utc))
    assert violation == True
    assert "EUDR VIOLATION" in v_msg

    # Pre-2020 disturbance is compliant (historical)
    violation_old, v_msg_old = val.verify_deforestation_timing(datetime(2019, 11, 20, tzinfo=timezone.utc))
    assert violation_old == False


def test_dds_auditor_compliance_check():
    auditor = DDSAuditor()

    plots = [
        {
            "plot_id": "PLOT-OK-01",
            "geometry_type": "point",
            "coordinates": [-60.123456, -3.123456],
            "declared_area_ha": 3.0,
        },
        {
            "plot_id": "PLOT-DEF-02",
            "geometry_type": "polygon",
            "coordinates": [
                [-59.500000, -2.500000],
                [-59.490000, -2.500000],
                [-59.490000, -2.490000],
                [-59.500000, -2.490000],
                [-59.500000, -2.500000],
            ],
        },
    ]

    # Deforestation alert overlapping PLOT-DEF-02 in 2023
    alerts = [
        {
            "alert_id": "ALERT-001",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-59.498000, -2.498000],
                    [-59.492000, -2.498000],
                    [-59.492000, -2.492000],
                    [-59.498000, -2.492000],
                    [-59.498000, -2.498000],
                ]],
            },
            "detection_date": "2023-04-12T00:00:00Z",
            "area_ha": 8.0,
        }
    ]

    report = auditor.audit_statement(
        statement_id="DDS-2023-COCOA-099",
        operator_name="Amazonian Agroforestry Ltd",
        commodity="cocoa",
        country="BR",
        plots=plots,
        deforestation_alerts=alerts,
    )

    assert report.overall_status == "NON_COMPLIANT"
    assert report.plot_results[0].status == "COMPLIANT"
    assert report.plot_results[1].status == "NON_COMPLIANT"
    assert len(report.plot_results[1].violations) > 0


def test_olofsson_accuracy_assessment():
    assessor = OlofssonAssessor(class_names=["Deforestation", "Forest"])

    # Sample error matrix n_ij (2x2):
    # Rows: Mapped [Deforestation, Forest]
    # Cols: Reference [Deforestation, Forest]
    # Stratum 1 (Mapped Deforestation): 80 samples true deforest, 20 samples forest
    # Stratum 2 (Mapped Forest): 10 samples true deforest, 890 samples forest
    sample_matrix = np.array([
        [80, 20],
        [10, 890],
    ])

    # Mapped areas: Deforestation is rare (5,000 ha out of 100,000 ha total = 5%)
    strata_areas = np.array([5000.0, 95000.0])

    report = assessor.compute(sample_matrix, strata_areas)

    assert 0.90 < report.overall_accuracy < 1.0
    assert report.oa_standard_error > 0.0
    assert report.oa_ci_95[0] < report.overall_accuracy < report.oa_ci_95[1]

    # User's accuracy for Deforestation = 80 / 100 = 80%
    assert np.isclose(report.users_accuracy["Deforestation"], 0.80)

    # Unbiased estimated true area of deforestation
    est_deforest_ha = report.estimated_areas_ha["Deforestation"]
    assert est_deforest_ha > 0.0
    assert report.area_margins_of_error_ha["Deforestation"] > 0.0


def test_eudr_commodity_scope_validation():
    val = EUDRValidator()
    # Regulated commodities
    for comm in ["cattle", "cocoa", "coffee", "oil palm", "rubber", "soya", "wood"]:
        ok, msg = val.validate_commodity(comm)
        assert ok is True
        assert "covered under EUDR Annex I" in msg

    # Unregulated commodities
    for unreg in ["electronics", "gold", "lithium", "wheat"]:
        ok, msg = val.validate_commodity(unreg)
        assert ok is False
        assert "outside the scope of EUDR" in msg


def test_dds_auditor_multipolygon_alert_overlap():
    auditor = DDSAuditor()
    plots = [
        {
            "plot_id": "PLOT-MULTI-TEST",
            "geometry_type": "polygon",
            "coordinates": [
                [
                    [-60.0, -3.0],
                    [-59.99, -3.0],
                    [-59.99, -2.99],
                    [-60.0, -2.99],
                    [-60.0, -3.0],
                ]
            ],
        }
    ]

    # MultiPolygon alert structure
    multipoly_alert = [
        {
            "alert_id": "ALERT-MP-101",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [
                            [-60.005, -3.005],
                            [-59.995, -3.005],
                            [-59.995, -2.995],
                            [-60.005, -2.995],
                            [-60.005, -3.005],
                        ]
                    ]
                ],
            },
            "detection_date": "2023-09-01T10:00:00Z",
            "area_ha": 15.0,
        }
    ]

    report = auditor.audit_statement(
        statement_id="DDS-MP-TEST",
        operator_name="EcoLogistics",
        commodity="soya",
        country="BR",
        plots=plots,
        deforestation_alerts=multipoly_alert,
    )
    assert report.overall_status == "NON_COMPLIANT"
    assert report.plot_results[0].status == "NON_COMPLIANT"
    assert len(report.plot_results[0].alert_overlaps) == 1


def test_dds_auditor_unregulated_commodity_rejection():
    auditor = DDSAuditor()
    report = auditor.audit_statement(
        statement_id="DDS-PLASTIC-TEST",
        operator_name="Generic Manufacturer",
        commodity="plastic",
        country="DE",
        plots=[],
        deforestation_alerts=[],
    )
    assert report.overall_status == "REJECTED_UNREGULATED_COMMODITY"
    assert report.risk_score == 1.0
