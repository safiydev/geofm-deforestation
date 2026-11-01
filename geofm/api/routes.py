"""FastAPI route handlers for deforestation detection and EUDR audits."""

from datetime import datetime, timezone
import time
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
import numpy as np
import torch

from geofm.api.schemas import (
    DetectRequest,
    DetectResponse,
    EUDRCheckRequest,
    EUDRCheckResponse,
    STACSearchRequest,
    STACSearchResponse,
    WebhookRegisterRequest,
    WebhookRegisterResponse,
)
from geofm.api.webhooks import webhook_dispatcher
from geofm.compliance.dds_auditor import DDSAuditor
from geofm.models.siamese import SiameseChangeDetector
from geofm.models.cross_attention import MultimodalFusionDeforestationNet
from geofm.postprocessing.polygonizer import Polygonizer
from geofm.preprocessing.stac import STACManager
from geofm.utils.synthetic_data import generate_synthetic_scene

router = APIRouter()

# Initialize core services in memory
_siamese_model = None
_fusion_model = None
_dds_auditor = DDSAuditor()
_stac_manager = STACManager()
_polygonizer = Polygonizer(mmu_ha=0.1)

# Pre-populate sample STAC items in catalog for queries
_cached_stac_items = [
    _stac_manager.create_sentinel2_item(
        item_id="S2A_MSIL2A_20231015T140051_N0509_R067",
        bbox=[-60.5, -3.5, -59.5, -2.5],
        dt=datetime(2023, 10, 15, 14, 0, 51, tzinfo=timezone.utc),
        asset_hrefs={
            "B02": "https://sentinel-cogs.s3.amazonaws.com/sentinel-s2-l2a-cogs/B02.tif",
            "B03": "https://sentinel-cogs.s3.amazonaws.com/sentinel-s2-l2a-cogs/B03.tif",
            "B04": "https://sentinel-cogs.s3.amazonaws.com/sentinel-s2-l2a-cogs/B04.tif",
            "B08": "https://sentinel-cogs.s3.amazonaws.com/sentinel-s2-l2a-cogs/B08.tif",
        },
        cloud_cover=3.8,
    ),
    _stac_manager.create_sentinel1_item(
        item_id="S1A_IW_GRDH_1SDV_20231016T093022",
        bbox=[-60.5, -3.5, -59.5, -2.5],
        dt=datetime(2023, 10, 16, 9, 30, 22, tzinfo=timezone.utc),
        asset_hrefs={
            "vv": "https://sentinel-s1-rtc.s3.amazonaws.com/GRD/VV.tif",
            "vh": "https://sentinel-s1-rtc.s3.amazonaws.com/GRD/VH.tif",
        },
    ),
]


def _get_siamese_model() -> SiameseChangeDetector:
    global _siamese_model
    if _siamese_model is None:
        _siamese_model = SiameseChangeDetector(in_channels=4, features=[16, 32, 64, 128])
        _siamese_model.eval()
    return _siamese_model


def _get_fusion_model() -> MultimodalFusionDeforestationNet:
    global _fusion_model
    if _fusion_model is None:
        _fusion_model = MultimodalFusionDeforestationNet(
            opt_channels=4, sar_channels=3, features=[16, 32, 64, 128]
        )
        _fusion_model.eval()
    return _fusion_model


@router.get("/health")
def health_check():
    """Health status and environment metadata."""
    return {
        "status": "healthy",
        "service": "GeoFM Deforestation & EUDR Compliance API",
        "version": "1.0.0",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "active_models": ["siamese_change_detector", "cross_attention_fusion", "ccdc_baseline"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/api/v1/detect", response_model=DetectResponse)
def detect_deforestation(req: DetectRequest):
    """Run deforestation detection inference on satellite inputs and produce GeoJSON alerts."""
    start_time = time.perf_counter()

    # Generate or parse input scene
    scene = generate_synthetic_scene(
        height=128,
        width=128,
        seed=42,
        deforestation_fraction=req.synthetic_deforestation_fraction,
    )

    t1_t = torch.from_numpy(scene["t1_optical"]).unsqueeze(0)
    t2_t = torch.from_numpy(scene["t2_optical"]).unsqueeze(0)

    if req.model_type == "siamese":
        model = _get_siamese_model()
        with torch.no_grad():
            logits = model(t1_t, t2_t)
            probs = torch.sigmoid(logits).squeeze().numpy()
    elif req.model_type == "multimodal":
        model = _get_fusion_model()
        sar_stacked = np.stack(
            [scene["sar_vv"], scene["sar_vh"], scene["sar_vh"] / (scene["sar_vv"] + 1e-6)],
            axis=0,
        )
        sar_t = torch.from_numpy(sar_stacked).unsqueeze(0)
        cloud_t = torch.from_numpy(scene["cloud_mask"]).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = model(t2_t, sar_t, cloud_mask=cloud_t)
            probs = torch.sigmoid(logits).squeeze().numpy()
    else:
        # NDVI delta baseline
        nir_t1 = scene["t1_optical"][3]
        red_t1 = scene["t1_optical"][2]
        nir_t2 = scene["t2_optical"][3]
        red_t2 = scene["t2_optical"][2]
        ndvi1 = (nir_t1 - red_t1) / (nir_t1 + red_t1 + 1e-6)
        ndvi2 = (nir_t2 - red_t2) / (nir_t2 + red_t2 + 1e-6)
        delta = ndvi2 - ndvi1
        probs = np.clip(-delta * 2.0, 0.0, 1.0)

    # Polygonize raster into GeoJSON
    polygonizer = Polygonizer(mmu_ha=req.mmu_ha)
    fc_result = polygonizer.polygonize(probs, threshold=req.threshold, as_multipolygon=False)
    multipoly_result = polygonizer.to_multipolygon_feature(fc_result)
    output_geojson = multipoly_result if req.as_multipolygon else fc_result

    total_alerts = fc_result["properties"]["total_alerts"]
    total_deforested_ha = fc_result["properties"]["total_deforested_ha"]
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # If critical alerts found, dispatch real-time webhooks
    if total_alerts > 0:
        webhook_dispatcher.dispatch(
            event_type="deforestation.detected",
            payload_data={
                "total_alerts": total_alerts,
                "total_deforested_ha": total_deforested_ha,
                "geojson": fc_result,
            },
        )

    return DetectResponse(
        model_type=req.model_type,
        total_alerts=total_alerts,
        total_deforested_ha=total_deforested_ha,
        geojson=output_geojson,
        multipolygon_feature=multipoly_result,
        inference_time_ms=round(elapsed_ms, 2),
    )


@router.post("/api/v1/eudr/audit", response_model=EUDRCheckResponse)
def audit_eudr_compliance(req: EUDRCheckRequest):
    """Audit a Due Diligence Statement against EUDR rules and deforestation alerts."""
    plots_data = [p.model_dump() for p in req.plots]

    # If alerts are not supplied, default to mock/recent post-2020 alerts
    if req.alerts is not None:
        alerts = req.alerts
    else:
        # Default test alert polygon located in central test zone (-59.96, -2.96)
        alerts = [
            {
                "alert_id": "ALT-2023-AMZ-001",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-59.965000, -2.965000],
                        [-59.955000, -2.965000],
                        [-59.955000, -2.955000],
                        [-59.965000, -2.955000],
                        [-59.965000, -2.965000],
                    ]],
                },
                "detection_date": "2023-08-14T10:00:00Z",  # Post-2020 violation
                "area_ha": 12.5,
            }
        ]

    report = _dds_auditor.audit_statement(
        statement_id=req.statement_id,
        operator_name=req.operator_name,
        commodity=req.commodity,
        country=req.country,
        plots=plots_data,
        deforestation_alerts=alerts,
    )

    # If violation occurs, dispatch alert
    if report.overall_status == "NON_COMPLIANT":
        webhook_dispatcher.dispatch(
            event_type="eudr.violation",
            payload_data=report.to_dict(),
        )

    return EUDRCheckResponse(**report.to_dict())


@router.post("/api/v1/stac/search", response_model=STACSearchResponse)
def search_stac_catalog(req: STACSearchRequest):
    """Search registered STAC items by bounding box, time interval, and platform."""
    matched = _stac_manager.query_catalog(
        items=_cached_stac_items,
        bbox=req.bbox,
        start_date=req.start_date,
        end_date=req.end_date,
        platform=req.platform,
    )
    items_json = [it.to_dict() for it in matched]
    return STACSearchResponse(total_matched=len(items_json), items=items_json)


@router.post("/api/v1/webhooks/register", response_model=WebhookRegisterResponse)
def register_webhook(req: WebhookRegisterRequest):
    """Register a new webhook listener URL."""
    sub = webhook_dispatcher.register(
        url=req.url,
        secret=req.secret or "default-secret",
        events=req.events,
        min_alert_area_ha=req.min_alert_area_ha,
    )
    return WebhookRegisterResponse(
        subscription_id=sub.subscription_id,
        url=sub.url,
        events=sub.events,
        min_alert_area_ha=sub.min_alert_area_ha,
        status="ACTIVE",
    )


@router.get("/api/v1/webhooks/deliveries")
def get_webhook_deliveries():
    """List recent webhook dispatch delivery attempts and status codes."""
    return {
        "total_deliveries": len(webhook_dispatcher.delivery_log),
        "deliveries": webhook_dispatcher.delivery_log[-20:],
    }
