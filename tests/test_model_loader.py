"""Unit and integration tests for GeoFM standalone model loader and web app."""

import io
import json
import numpy as np
import pytest
import torch
from starlette.testclient import TestClient
from model_loader import app, engine


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "GeoFM" in response.text
    assert "Model Loader" in response.text


def test_system_status(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "device" in data
    assert "has_loaded_model" in data
    assert data["has_loaded_model"] is True
    assert data["model_info"]["model_type"] == "siamese"


def test_checkpoints_list(client):
    response = client.get("/api/checkpoints")
    assert response.status_code == 200
    data = response.json()
    assert "checkpoints" in data
    assert len(data["checkpoints"]) >= 1
    # Check best_model.pt is present
    names = [c["filename"] for c in data["checkpoints"]]
    assert "best_model.pt" in names


def test_samples_list(client):
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) > 0


def test_synthetic_inference(client):
    response = client.post("/api/predict/synthetic", data={"threshold": 0.50, "anomaly_size": 40})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    result = data["result"]
    assert "probability_b64" in result
    assert "binary_mask_b64" in result
    assert "difference_b64" in result
    assert "statistics" in result
    assert result["statistics"]["threshold"] == 0.50
    assert result["statistics"]["total_pixels"] == 256 * 256
    assert "alerts_geojson" in result
    assert result["alerts_geojson"]["type"] == "FeatureCollection"


def test_rethreshold_without_rerun(client):
    # Ensure active prediction first
    client.post("/api/predict/synthetic", data={"threshold": 0.50})
    
    # Fast re-thresholding
    response = client.post("/api/predict/rethreshold", data={"threshold": 0.75})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["threshold"] == 0.75
    assert "binary_mask_b64" in data
    assert "statistics" in data
    assert data["statistics"]["threshold"] == 0.75


def test_cached_sample_inference(client):
    samples_res = client.get("/api/samples").json()
    first_chip = samples_res["samples"][0]["filename"]
    
    response = client.post("/api/predict/sample", data={"filename": first_chip, "threshold": 0.50})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "t1_image_b64" in data["result"]
    assert "probability_b64" in data["result"]
    assert data["result"]["statistics"]["area_ha"] >= 0.0


def test_load_local_checkpoint(client):
    response = client.post(
        "/api/model/load-local",
        data={"path": "checkpoints/best_model.pt", "arch_override": "auto"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["model_info"]["model_type"] == "siamese"
    assert data["model_info"]["has_lora"] is True
    assert data["model_info"]["in_channels"] == 4


def test_upload_custom_checkpoint(client):
    # Create small dummy state dict
    from geofm.models.siamese import SiameseChangeDetector
    dummy_model = SiameseChangeDetector(in_channels=4, features=[8, 16, 32, 64])
    buf = io.BytesIO()
    torch.save({"model_type": "siamese", "model_state_dict": dummy_model.state_dict(), "epoch": 99}, buf)
    buf.seek(0)
    
    response = client.post(
        "/api/model/upload",
        files={"file": ("test_custom_model.pt", buf, "application/octet-stream")},
        data={"arch_override": "siamese"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["model_info"]["model_type"] == "siamese"
    assert data["model_info"]["metadata"]["epoch"] == 99
