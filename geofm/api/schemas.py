"""Pydantic schemas for the GeoFM Deforestation Detection and EUDR API."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class DetectRequest(BaseModel):
    """Request payload for deforestation detection inference."""
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Probability threshold")
    mmu_ha: float = Field(0.1, ge=0.01, description="Minimum Mapping Unit in hectares")
    model_type: Literal["siamese", "multimodal", "ndvi_delta"] = "siamese"
    # Optional raw matrices or synthetic trigger for testing
    use_synthetic_data: bool = Field(True, description="Use high-fidelity synthetic scene if true")
    synthetic_deforestation_fraction: float = Field(0.08, ge=0.0, le=0.5)
    as_multipolygon: bool = Field(False, description="Return unified MultiPolygon Feature instead of FeatureCollection")


class DetectResponse(BaseModel):
    """Detection results including summary statistics and GeoJSON polygons."""
    model_type: str
    total_alerts: int
    total_deforested_ha: float
    geojson: Dict[str, Any]
    multipolygon_feature: Optional[Dict[str, Any]] = None
    inference_time_ms: float


class DDSPlotItem(BaseModel):
    plot_id: str
    geometry_type: Literal["point", "polygon"]
    coordinates: Union[List[float], List[List[float]]]
    declared_area_ha: Optional[float] = None


class EUDRCheckRequest(BaseModel):
    statement_id: str
    operator_name: str
    commodity: str = Field(..., description="Regulated commodity (e.g. cattle, cocoa, coffee, soya)")
    country: str = Field(..., description="Country of production")
    plots: List[DDSPlotItem]
    # Deforestation alerts can be supplied or queried automatically from system
    alerts: Optional[List[Dict[str, Any]]] = None


class DDSPlotAuditItem(BaseModel):
    plot_id: str
    status: str
    area_ha: float
    violations: List[str]
    alert_overlaps: List[Dict[str, Any]]


class EUDRCheckResponse(BaseModel):
    statement_id: str
    operator_name: str
    commodity: str
    country: str
    overall_status: str
    risk_score: float
    audit_timestamp: str
    total_plots: int
    plots: List[DDSPlotAuditItem]


class STACSearchRequest(BaseModel):
    bbox: Optional[List[float]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    platform: Optional[str] = None


class STACSearchResponse(BaseModel):
    total_matched: int
    items: List[Dict[str, Any]]


class WebhookRegisterRequest(BaseModel):
    url: str
    secret: Optional[str] = "secret-token"
    events: List[str] = ["deforestation.detected", "eudr.violation"]
    min_alert_area_ha: float = 0.5


class WebhookRegisterResponse(BaseModel):
    subscription_id: str
    url: str
    events: List[str]
    min_alert_area_ha: float
    status: str
