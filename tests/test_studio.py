"""Unit and integration tests for GeoFM Web Training Studio, Logger, and EUDR Suite."""

import json
import os
import pytest
from starlette.testclient import TestClient

from geofm.studio.core import (
    CheckpointManager,
    StudioComplianceManager,
    StudioInferenceManager,
    StudioSTACManager,
    StudioTrainingManager,
    array_to_png_base64,
    render_ndvi_colormap,
    render_optical_rgb,
    render_optical_cir,
    render_sar_composite,
    render_probability_heatmap,
)
from geofm.studio.streamlit_app import run_streamlit_app
from geofm.studio.web_app import studio_fastapi_app
from geofm.training.trainer import DeforestationTrainer
from geofm.models.siamese import SiameseChangeDetector
from geofm.training.dataset import GeospatialDeforestationDataset
from geofm.utils.synthetic_data import generate_synthetic_dataset
from torch.utils.data import DataLoader, Subset


client = TestClient(studio_fastapi_app)


# =============================================================================
# Unit Tests: Image Rendering Utilities
# =============================================================================

def test_image_rendering_utilities():
    import numpy as np

    # Test array to base64
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    b64 = array_to_png_base64(rgb)
    assert b64.startswith("data:image/png;base64,")

    # Test optical 4-band RGB render
    opt = np.random.uniform(0.0, 1.0, (4, 32, 32)).astype(np.float32)
    b64_rgb = render_optical_rgb(opt)
    assert b64_rgb.startswith("data:image/png;base64,")

    # Test CIR
    b64_cir = render_optical_cir(opt)
    assert b64_cir.startswith("data:image/png;base64,")

    # Test NDVI colormap
    ndvi = np.random.uniform(-0.5, 0.8, (32, 32))
    b64_ndvi = render_ndvi_colormap(ndvi)
    assert b64_ndvi.startswith("data:image/png;base64,")

    # Test SAR composite
    sar_vv = np.random.exponential(0.1, (32, 32))
    sar_vh = np.random.exponential(0.05, (32, 32))
    b64_sar = render_sar_composite(sar_vv, sar_vh)
    assert b64_sar.startswith("data:image/png;base64,")

    # Test probability heatmap
    probs = np.random.uniform(0.0, 1.0, (32, 32))
    b64_prob = render_probability_heatmap(probs)
    assert b64_prob.startswith("data:image/png;base64,")


# =============================================================================
# Unit Tests: Training Logger & JSON Export
# =============================================================================

def test_training_logger_and_json_export(tmp_path):
    samples = generate_synthetic_dataset(num_samples=4, height=32, width=32)
    dataset = GeospatialDeforestationDataset(samples)
    train_loader = DataLoader(Subset(dataset, [0, 1]), batch_size=2)
    val_loader = DataLoader(Subset(dataset, [2, 3]), batch_size=2)

    model = SiameseChangeDetector(in_channels=4, features=[8, 16, 32, 64])

    callbacks_fired = []

    def on_epoch(ep, rec):
        callbacks_fired.append((ep, rec))

    trainer = DeforestationTrainer(
        model=model,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        experiment_name="test_experiment",
        epoch_callback=on_epoch,
        enable_tensorboard=True,
    )

    history = trainer.fit(train_loader, val_loader, epochs=2)

    # Check history structure
    assert len(history) == 2
    assert len(callbacks_fired) == 2
    assert "confusion_matrix" in history[0]
    cm = history[0]["confusion_matrix"]
    assert "tp" in cm and "fp" in cm and "fn" in cm and "tn" in cm

    # Check metrics JSON export
    metrics_file = tmp_path / "logs" / "test_experiment_metrics.json"
    assert metrics_file.exists()

    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_name"] == "test_experiment"
    assert data["total_epochs_recorded"] == 2
    assert "summary" in data
    assert "best_val_iou" in data
    assert len(data["history"]) == 2

    # Check latest_metrics.json in checkpoint_dir
    latest_file = tmp_path / "checkpoints" / "latest_metrics.json"
    assert latest_file.exists()

    # Check close does not raise
    trainer.close()


# =============================================================================
# Unit Tests: Studio Core Managers
# =============================================================================

def test_studio_inference_manager():
    mgr = StudioInferenceManager()
    res = mgr.run_inference(
        deforestation_fraction=0.1,
        cloud_fraction=0.1,
        model_type="siamese",
        threshold=0.5,
        mmu_ha=0.1,
    )

    assert res["model_type"] == "siamese"
    assert "total_alerts" in res
    assert "total_deforested_ha" in res
    assert "layers" in res
    assert "t1_rgb" in res["layers"]
    assert "probability_heatmap" in res["layers"]
    assert "geojson" in res
    assert res["geojson"]["type"] == "FeatureCollection"
    assert "severity_counts" in res


def test_studio_compliance_manager():
    mgr = StudioComplianceManager()
    presets = mgr.get_preset_scenarios()
    assert len(presets) >= 4

    # Test Brazilian soy violation scenario
    sc = presets["Scenario A: Brazilian Soy Farm (Deforestation Violation)"]
    audit = mgr.audit_statement(
        statement_id=sc["statement_id"],
        operator_name=sc["operator_name"],
        commodity=sc["commodity"],
        country=sc["country"],
        plots=sc["plots"],
        alerts=sc["alerts"],
    )
    assert audit["overall_status"] == "NON_COMPLIANT"
    assert audit["risk_score"] > 0.0
    assert len(audit["plots"]) == 1
    assert len(audit["plots"][0]["violations"]) > 0

    # Test Indonesian palm oil compliant smallholder
    sc_clean = presets["Scenario B: Indonesian Palm Oil Smallholder (<=4ha Point - Compliant)"]
    audit_clean = mgr.audit_statement(
        statement_id=sc_clean["statement_id"],
        operator_name=sc_clean["operator_name"],
        commodity=sc_clean["commodity"],
        country=sc_clean["country"],
        plots=sc_clean["plots"],
        alerts=sc_clean["alerts"],
    )
    assert audit_clean["overall_status"] == "COMPLIANT"
    assert audit_clean["risk_score"] == 0.0

    # Test Olofsson area-adjusted accuracy assessment
    olof = mgr.compute_olofsson_accuracy()
    assert olof["overall_accuracy_pct"] > 90.0
    assert "Deforestation" in olof["estimated_areas_ha"]
    assert "summary_text" in olof


def test_studio_stac_manager():
    mgr = StudioSTACManager()
    items = mgr.query_catalog(bbox=[-60.5, -3.5, -59.5, -2.5])
    assert len(items) > 0
    assert "platform" in items[0]
    assert "assets" in items[0]

    # Filter by Sentinel-2
    s2_items = mgr.query_catalog(bbox=[-60.5, -3.5, -59.5, -2.5], platform="sentinel-2")
    assert all(it["platform"] == "sentinel-2" for it in s2_items)


def test_checkpoint_manager(tmp_path):
    mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
    chks = mgr.list_checkpoints()
    assert chks == []


def test_streamlit_app_graceful_run():
    # Calling run_streamlit_app when Streamlit is not installed should exit cleanly without raising
    run_streamlit_app()


# =============================================================================
# Integration Tests: Web Studio REST Endpoints
# =============================================================================

def test_web_studio_get_index():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "GeoFM Web Training Studio" in resp.text
    assert "Leaflet" in resp.text
    assert "Chart.js" in resp.text


def test_web_studio_api_infer():
    payload = {
        "deforestation_fraction": 0.08,
        "cloud_fraction": 0.1,
        "model_type": "siamese",
        "threshold": 0.5,
        "mmu_ha": 0.10,
    }
    resp = client.post("/studio/api/infer", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "total_alerts" in data
    assert "total_deforested_ha" in data
    assert "layers" in data
    assert "probability_heatmap" in data["layers"]


def test_web_studio_api_eudr_presets():
    resp = client.get("/studio/api/eudr/presets")
    assert resp.status_code == 200
    presets = resp.json()
    assert "Scenario A: Brazilian Soy Farm (Deforestation Violation)" in presets


def test_web_studio_api_eudr_audit():
    payload = {
        "statement_id": "TEST-DDS-001",
        "operator_name": "Test Forest Products",
        "commodity": "wood",
        "country": "Brazil",
        "plots": [
            {
                "plot_id": "PLOT-TEST-1",
                "geometry_type": "Point",
                "coordinates": [-60.123456, -3.123456],
                "declared_area_ha": 2.5,
            }
        ],
        "alerts": [],
    }
    resp = client.post("/studio/api/eudr/audit", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["overall_status"] == "COMPLIANT"
    assert data["risk_score"] == 0.0


def test_web_studio_api_olofsson():
    resp = client.post("/studio/api/olofsson")
    assert resp.status_code == 200
    data = resp.json()
    assert "overall_accuracy_pct" in data
    assert "estimated_areas_ha" in data


def test_web_studio_api_stac_search():
    payload = {
        "bbox": [-60.5, -3.5, -59.5, -2.5],
        "platform": "sentinel-2",
        "max_cloud_cover": 20.0,
    }
    resp = client.post("/studio/api/stac/search", json=payload)
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) > 0
    assert items[0]["platform"] == "sentinel-2"


def test_web_studio_api_checkpoints():
    resp = client.get("/studio/api/checkpoints")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_web_studio_api_train():
    payload = {
        "architecture": "siamese",
        "use_lora": True,
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "loss_type": "compound",
        "learning_rate": 1e-3,
        "epochs": 1,
        "batch_size": 2,
        "num_samples": 4,
    }
    resp = client.post("/studio/api/train", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["architecture"] == "siamese"
    assert len(data["history"]) == 1
    assert "val_iou" in data["history"][0]


def test_web_studio_api_train_status():
    resp = client.get("/studio/api/train/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "is_training" in data
    assert "latest_run" in data
    assert "history" in data


def test_trainer_learning_rate_and_csv_export(tmp_path):
    samples = generate_synthetic_dataset(num_samples=4, height=32, width=32)
    dataset = GeospatialDeforestationDataset(samples)
    loader = DataLoader(dataset, batch_size=2)
    model = SiameseChangeDetector(in_channels=4, features=[8, 16, 32, 64])

    custom_lr = 0.0035
    trainer = DeforestationTrainer(
        model=model,
        learning_rate=custom_lr,
        checkpoint_dir=str(tmp_path / "chk"),
        log_dir=str(tmp_path / "logs"),
        experiment_name="test_lr_csv",
    )

    # Check optimizer got configured LR
    assert abs(trainer.optimizer.param_groups[0]["lr"] - custom_lr) < 1e-6

    history = trainer.fit(loader, loader, epochs=1)
    assert len(history) == 1
    assert abs(history[0]["lr"] - custom_lr) < 1e-6

    # Verify CSV export
    csv_file = tmp_path / "logs" / "test_lr_csv_metrics.csv"
    assert csv_file.exists()
    content = csv_file.read_text(encoding="utf-8")
    assert "epoch,train_loss,val_loss,val_iou" in content
    assert str(custom_lr) in content

    # Verify TensorBoard export directory returns path
    tb_dir = trainer.export_tensorboard()
    assert tb_dir is not None
    assert str(tmp_path / "logs") in tb_dir


def test_trainer_fit_zero_epochs(tmp_path):
    samples = generate_synthetic_dataset(num_samples=2, height=32, width=32)
    dataset = GeospatialDeforestationDataset(samples)
    loader = DataLoader(dataset, batch_size=2)
    model = SiameseChangeDetector(in_channels=4, features=[8, 16, 32, 64])

    trainer = DeforestationTrainer(
        model=model,
        checkpoint_dir=str(tmp_path / "chk"),
    )
    history = trainer.fit(loader, loader, epochs=0)
    assert history == []
    summary = trainer.get_metrics_summary()
    assert summary["epochs_completed"] == 0


def test_compliance_manager_with_geojson_feature_collections():
    mgr = StudioComplianceManager()

    # Pass plots as a full GeoJSON FeatureCollection
    plots_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "plot-amazon-01",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-60.015, -3.015],
                            [-59.985, -3.015],
                            [-59.985, -2.985],
                            [-60.015, -2.985],
                            [-60.015, -3.015],
                        ]
                    ],
                },
                "properties": {"plot_id": "PLOT-FC-01", "declared_area_ha": 35.0},
            }
        ],
    }

    # Pass alerts as a full GeoJSON FeatureCollection
    alerts_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "alert-cluster-0042",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-60.005, -3.005],
                            [-59.995, -3.005],
                            [-59.995, -2.995],
                            [-60.005, -2.995],
                            [-60.005, -3.005],
                        ]
                    ],
                },
                "properties": {
                    "cluster_id": 42,
                    "area_ha": 8.5,
                    "detection_date": "2023-09-01T12:00:00Z",
                },
            }
        ],
    }

    report = mgr.audit_statement(
        statement_id="DDS-FC-TEST",
        operator_name="Eco Amazon Forestry",
        commodity="soya",
        country="Brazil",
        plots=plots_fc,
        alerts=alerts_fc,
    )

    assert report["overall_status"] == "NON_COMPLIANT"
    assert report["risk_score"] > 0.0
    plot_res = report["plots"][0]
    assert plot_res["plot_id"] == "plot-amazon-01" or plot_res["plot_id"] == "PLOT-FC-01"
    assert len(plot_res["violations"]) > 0
    assert len(plot_res["alert_overlaps"]) == 1
    assert plot_res["alert_overlaps"][0]["alert_id"] == "alert-cluster-0042"
    assert plot_res["alert_overlaps"][0]["alert_area_ha"] == 8.5


def test_studio_training_multimodal_and_unet(tmp_path):
    mgr = StudioTrainingManager(checkpoint_dir=str(tmp_path / "checkpoints"))

    # Test multimodal architecture with custom learning rate
    res_multi = mgr.train_model(
        architecture="multimodal",
        learning_rate=0.002,
        epochs=1,
        batch_size=2,
        num_samples=4,
    )
    assert res_multi["architecture"] == "multimodal"
    assert abs(res_multi["history"][0]["lr"] - 0.002) < 1e-6

    # Test unet architecture without LoRA
    res_unet = mgr.train_model(
        architecture="unet",
        use_lora=False,
        learning_rate=0.0005,
        epochs=1,
        batch_size=2,
        num_samples=4,
    )
    assert res_unet["architecture"] == "unet"
    assert res_unet["use_lora"] is False
    assert abs(res_unet["history"][0]["lr"] - 0.0005) < 1e-6


def test_web_studio_catalog_validate_endpoint():
    # Valid scheme test
    resp = client.post(
        "/studio/api/catalog/validate",
        json={"url": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "url" in data
    assert "exists" in data

    # Invalid scheme test
    resp_inv = client.post(
        "/studio/api/catalog/validate",
        json={"url": "ftp://bad-url"},
    )
    assert resp_inv.status_code == 200
    assert resp_inv.json()["exists"] is False


def test_web_studio_catalog_heal_endpoint():
    # Auto-heal the user's specific malformed path
    malformed_url = (
        "https://sentinel-cogs.s3.amazonaws.com/sentinel-s2-l2a-cogs/"
        "20/M/PA/2023/11/S2A_20231124/B02.tif"
    )
    resp = client.post("/studio/api/catalog/heal", json={"url": malformed_url})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["healed"] is True
    assert data["mgrs_tile"] == "20MPA"
    assert "B02.tif" in data["verified_band_url"]


# =============================================================================
# New Studio Suite Tests: Benchmark, Inspector, Zoo, Telemetry
# =============================================================================

def test_web_studio_telemetry_endpoint():
    resp = client.get("/studio/api/telemetry")
    assert resp.status_code == 200
    data = resp.json()
    assert "cpu_count" in data
    assert "ram_total_mb" in data
    assert "device_name" in data
    assert "checkpoint_count" in data


def test_web_studio_test_benchmark_endpoint():
    payload = {
        "num_samples": 4,
        "threshold": 0.5,
    }
    resp = client.post("/studio/api/test/benchmark", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "iou" in data
    assert "f1" in data
    assert "confusion_matrix" in data
    assert "tp" in data["confusion_matrix"]
    assert "fp" in data["confusion_matrix"]
    assert "fn" in data["confusion_matrix"]
    assert "tn" in data["confusion_matrix"]


def test_web_studio_test_chip_endpoint():
    payload = {
        "chip_idx": 0,
        "threshold": 0.5,
        "mmu_ha": 0.10,
    }
    resp = client.post("/studio/api/test/chip", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "chip_id" in data
    assert "layers" in data
    assert "t1_rgb" in data["layers"]
    assert "t2_rgb" in data["layers"]
    assert "probability_heatmap" in data["layers"]
    assert "error_diff_map" in data["layers"]
    assert "ground_truth_mask" in data["layers"]
    assert "chip_iou" in data


def test_web_studio_test_thresholds_endpoint():
    payload = {"num_samples": 4}
    resp = client.post("/studio/api/test/thresholds", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 9  # 0.1 to 0.9 in steps of 0.1
    assert "threshold" in data[0]
    assert "iou" in data[0]
    assert "f1" in data[0]


def test_web_studio_chips_generate_and_list(tmp_path):
    from geofm.studio.core import StudioChipManager
    chip_mgr = StudioChipManager(cache_dir=str(tmp_path / "chips"))
    res = chip_mgr.generate_synthetic_chips(num_chips=10, deforest_ratio=0.70)
    assert res["status"] == "success"
    assert res["created_chips"] == 10
    assert res["deforest_chips"] == 7
    assert res["intact_chips"] == 3

    chips_list = chip_mgr.list_chips()
    assert chips_list["total_chips"] == 10
    assert len(chips_list["chips"]) == 10


def test_web_studio_train_stop_endpoint():
    resp = client.post("/studio/api/train/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"


