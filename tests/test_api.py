"""Tests for FastAPI REST endpoints and webhook alerting service."""

from fastapi.testclient import TestClient
import pytest

from geofm.api.app import app
from geofm.api.webhooks import webhook_dispatcher


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "torch_version" in data


def test_detect_endpoint(client):
    req_body = {
        "threshold": 0.5,
        "mmu_ha": 0.1,
        "model_type": "siamese",
        "use_synthetic_data": True,
        "synthetic_deforestation_fraction": 0.08,
    }
    resp = client.post("/api/v1/detect", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_type"] == "siamese"
    assert "geojson" in data
    assert data["geojson"]["type"] == "FeatureCollection"
    assert data["total_alerts"] >= 0


def test_eudr_audit_endpoint(client):
    req_body = {
        "statement_id": "DDS-API-TEST-001",
        "operator_name": "Sustainable Global Trade Inc.",
        "commodity": "coffee",
        "country": "CO",
        "plots": [
            {
                "plot_id": "PLOT-API-1",
                "geometry_type": "point",
                "coordinates": [-75.123456, 4.123456],
                "declared_area_ha": 2.0,
            }
        ],
    }
    resp = client.post("/api/v1/eudr/audit", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["statement_id"] == "DDS-API-TEST-001"
    assert data["overall_status"] == "COMPLIANT"
    assert data["risk_score"] == 0.0


def test_stac_search_endpoint(client):
    req_body = {
        "platform": "sentinel-2",
    }
    resp = client.post("/api/v1/stac/search", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_matched"] >= 1


def test_webhook_register_and_dispatch(client):
    # Register webhook
    reg_body = {
        "url": "http://mock-webhook-receiver.local/alerts",
        "secret": "top-secret-signing-key",
        "events": ["deforestation.detected"],
        "min_alert_area_ha": 0.1,
    }
    resp = client.post("/api/v1/webhooks/register", json=reg_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ACTIVE"

    # Dispatch event
    deliveries = webhook_dispatcher.dispatch(
        event_type="deforestation.detected",
        payload_data={"total_deforested_ha": 2.5, "alert_id": "ALT-TEST-99"},
    )
    assert len(deliveries) >= 1
    assert deliveries[0]["success"] == True


def test_detect_endpoint_multipolygon(client):
    req_body = {
        "threshold": 0.5,
        "mmu_ha": 0.05,
        "model_type": "siamese",
        "use_synthetic_data": True,
        "synthetic_deforestation_fraction": 0.08,
        "as_multipolygon": True,
    }
    resp = client.post("/api/v1/detect", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["geojson"]["type"] == "Feature"
    assert data["geojson"]["geometry"]["type"] == "MultiPolygon"
    assert data["multipolygon_feature"] is not None


def test_stac_search_endpoint_with_naive_dates(client):
    req_body = {
        "platform": "sentinel-2",
        "start_date": "2023-01-01T00:00:00",
        "end_date": "2023-12-31T23:59:59",
    }
    resp = client.post("/api/v1/stac/search", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_matched"] >= 1


def test_eudr_audit_endpoint_unregulated_commodity(client):
    req_body = {
        "statement_id": "DDS-API-UNREG-01",
        "operator_name": "Consumer Tech Corp",
        "commodity": "electronics",
        "country": "DE",
        "plots": [
            {
                "plot_id": "PLOT-ELEC-1",
                "geometry_type": "point",
                "coordinates": [10.123456, 50.123456],
                "declared_area_ha": 1.5,
            }
        ],
    }
    resp = client.post("/api/v1/eudr/audit", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["overall_status"] == "REJECTED_UNREGULATED_COMMODITY"
    assert data["risk_score"] == 1.0
