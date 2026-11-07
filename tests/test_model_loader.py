"""Unit and integration tests for GeoFM standalone model loader and web app."""

import io
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from starlette.testclient import TestClient

from geofm.models.siamese import SiameseChangeDetector
from geofm.models.lora import apply_lora_to_model
from model_loader import app, engine


@pytest.fixture(scope="module")
def client():
    # Ensure test directories and artifacts exist even in a fresh CI environment
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("data/chips_cache", exist_ok=True)

    test_chk = Path("checkpoints/test_fixture_model.pt")
    if not test_chk.exists():
        dummy_model = SiameseChangeDetector(in_channels=4, features=[16, 32, 64, 128])
        dummy_model, _ = apply_lora_to_model(dummy_model, rank=8, alpha=16.0)
        torch.save({
            "model_type": "siamese",
            "model_state_dict": dummy_model.state_dict(),
            "epoch": 5,
            "best_val_iou": 0.72
        }, str(test_chk))

    test_chip = Path("data/chips_cache/test_chip_fixture.npz")
    if not test_chip.exists():
        np.savez(
            test_chip,
            sample_id="test_01",
            t1_optical=np.ones((4, 64, 64), dtype=np.float32),
            t2_optical=np.ones((4, 64, 64), dtype=np.float32) * 0.5,
            target_mask=np.zeros((1, 64, 64), dtype=np.float32),
            lon=-62.5,
            lat=-8.5,
            deforested_pct=5.2
        )

    # Ensure engine has model loaded
    engine.inspect_and_load_checkpoint(test_chk)

    c = TestClient(app)
    yield c

    # Teardown
    if test_chk.exists():
        try:
            test_chk.unlink()
        except Exception:
            pass
    if test_chip.exists():
        try:
            test_chip.unlink()
        except Exception:
            pass


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


def test_samples_list(client):
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) >= 1


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
    # Ensure test model exists for loading
    test_path = "checkpoints/test_load_temp.pt"
    dummy = SiameseChangeDetector(in_channels=4, features=[16, 32, 64, 128])
    torch.save({"model_type": "siamese", "model_state_dict": dummy.state_dict()}, test_path)

    try:
        response = client.post(
            "/api/model/load-local",
            data={"path": test_path, "arch_override": "auto"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_info"]["model_type"] == "siamese"
        assert data["model_info"]["in_channels"] == 4
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)


def test_upload_custom_checkpoint(client):
    # Create small dummy state dict
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


def test_upload_dataset_zip(client):
    import zipfile
    
    # Create in-memory zip with 2 dummy npz chips
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w') as zf:
        c1 = io.BytesIO()
        np.savez(c1, sample_id="zip_01", t1_optical=np.ones((4, 32, 32), dtype=np.float32), t2_optical=np.ones((4, 32, 32), dtype=np.float32)*0.8, lon=-62.1, lat=-8.1, deforested_pct=8.5)
        zf.writestr("chip_zip_test_01.npz", c1.getvalue())

        c2 = io.BytesIO()
        np.savez(c2, sample_id="zip_02", t1_optical=np.ones((4, 32, 32), dtype=np.float32), t2_optical=np.ones((4, 32, 32), dtype=np.float32)*0.2, lon=-62.2, lat=-8.2, deforested_pct=15.0)
        zf.writestr("subfolder/chip_zip_test_02.npz", c2.getvalue())

    zip_buf.seek(0)
    response = client.post(
        "/api/dataset/upload-zip",
        files={"file": ("geofm_test_dataset.zip", zip_buf, "application/zip")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["unpacked_count"] == 2
    assert "chip_zip_test_01.npz" in [s["filename"] for s in data["samples"]]

    # Cleanup unpacked test chips
    for name in ["chip_zip_test_01.npz", "chip_zip_test_02.npz"]:
        p = Path("data/chips_cache") / name
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass
