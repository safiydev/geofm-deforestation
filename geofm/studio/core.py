"""Core business logic, inference orchestration, and training managers for the GeoFM Studio."""

from datetime import datetime, timezone
import io
import base64
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image

from geofm.compliance.dds_auditor import DDSAuditor, DDSAuditReport
from geofm.compliance.eudr import EUDRValidator, EUDR_CUTOFF_DATE, EUDR_REGULATED_COMMODITIES
from geofm.compliance.olofsson import OlofssonAssessor, OlofssonReport
from geofm.losses.focal import BinaryFocalLoss
from geofm.losses.focal_tversky import CompoundDeforestationLoss, FocalTverskyLoss
from geofm.losses.tversky import TverskyLoss
from geofm.models.cross_attention import MultimodalFusionDeforestationNet
from geofm.models.lora import apply_lora_to_model
from geofm.models.siamese import SiameseChangeDetector
from geofm.models.unet import UNet
from geofm.preprocessing.cog_reader import GeoTransform
from geofm.postprocessing.polygonizer import Polygonizer
from geofm.preprocessing.optical import OpticalPreprocessor
from geofm.preprocessing.sar import SARPreprocessor
from geofm.preprocessing.stac import STACManager
from geofm.preprocessing.catalog_validator import CatalogValidator
from geofm.training.dataset import GeospatialDeforestationDataset, SpatialBlockSplitter
from geofm.training.trainer import DeforestationTrainer
from geofm.utils.synthetic_data import generate_synthetic_dataset, generate_synthetic_scene


# =============================================================================
# Image Visualization & Base64 Data URI Utilities
# =============================================================================

def array_to_png_base64(arr: np.ndarray) -> str:
    """Encode an RGB uint8 array (H, W, 3) or grayscale (H, W) to base64 PNG data URI."""
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    elif arr.ndim == 3 and arr.shape[2] == 4:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    else:
        raise ValueError(f"Unsupported array shape for image conversion: {arr.shape}")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def render_optical_rgb(opt_4band: np.ndarray) -> str:
    """Render Sentinel-2 True Color RGB (Red=B4, Green=B3, Blue=B2) as base64 PNG."""
    # opt_4band is (4, H, W): [Blue, Green, Red, NIR]
    b = opt_4band[0]
    g = opt_4band[1]
    r = opt_4band[2]

    # Normalize each channel with 2-98% percentile stretch for satellite realism
    rgb = np.stack([r, g, b], axis=-1)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb_stretched = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
    rgb_uint8 = (rgb_stretched * 255).astype(np.uint8)
    return array_to_png_base64(rgb_uint8)


def render_optical_cir(opt_4band: np.ndarray) -> str:
    """Render Color Infrared (CIR False Color: NIR=B8, Red=B4, Green=B3)."""
    # opt_4band is (4, H, W): [Blue, Green, Red, NIR]
    nir = opt_4band[3]
    r = opt_4band[2]
    g = opt_4band[1]

    cir = np.stack([nir, r, g], axis=-1)
    p2, p98 = np.percentile(cir, (2, 98))
    cir_stretched = np.clip((cir - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
    cir_uint8 = (cir_stretched * 255).astype(np.uint8)
    return array_to_png_base64(cir_uint8)


def render_ndvi_colormap(ndvi: np.ndarray) -> str:
    """Render NDVI raster (-1.0 to 1.0) with an Earth Observation vegetation colormap."""
    # Colormap: low NDVI = brown/red (bare soil), mid = yellow, high = vibrant dark green
    h, w = ndvi.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Normalize roughly from 0.0 to 0.8
    norm = np.clip((ndvi - 0.0) / 0.8, 0.0, 1.0)

    for r in range(h):
        for c in range(w):
            val = norm[r, c]
            if val < 0.25:  # Non-vegetated / deforested / bare soil: Red/Brown
                t = val / 0.25
                rgb[r, c] = [int(180 + t * 40), int(50 + t * 50), 30]
            elif val < 0.5:  # Sparse / clearing: Yellowish-Orange
                t = (val - 0.25) / 0.25
                rgb[r, c] = [int(220 - t * 30), int(120 + t * 80), 30]
            elif val < 0.75:  # Regrowth / secondary: Lime Green
                t = (val - 0.5) / 0.25
                rgb[r, c] = [int(140 - t * 100), int(180 + t * 40), int(30 + t * 20)]
            else:  # Dense canopy / intact rainforest: Deep Forest Green
                t = (val - 0.75) / 0.25
                rgb[r, c] = [int(40 - t * 25), int(140 - t * 30), int(30 - t * 10)]

    return array_to_png_base64(rgb)


def render_sar_composite(sar_vv: np.ndarray, sar_vh: np.ndarray) -> str:
    """Render Sentinel-1 SAR dual-polarization false color (R=VV dB, G=VH dB, B=Ratio dB)."""
    sar_prep = SARPreprocessor()
    vv_db = sar_prep.to_decibels(sar_vv)
    vh_db = sar_prep.to_decibels(sar_vh)
    ratio_db = vh_db - vv_db

    # Standard Sentinel-1 dynamic ranges: VV [-22, -2], VH [-28, -8], Ratio [-15, -2]
    def _norm(ch, vmin, vmax):
        return np.clip((ch - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)

    r = _norm(vv_db, -22, -2)
    g = _norm(vh_db, -28, -8)
    b = _norm(ratio_db, -15, -2)

    comp = np.stack([r, g, b], axis=-1)
    comp_uint8 = (comp * 255).astype(np.uint8)
    return array_to_png_base64(comp_uint8)


def render_probability_heatmap(prob_map: np.ndarray) -> str:
    """Render continuous probability heatmap from dark blue (0.0 intact) to hot red (1.0 clear-cut)."""
    h, w = prob_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for r in range(h):
        for c in range(w):
            p = float(np.clip(prob_map[r, c], 0.0, 1.0))
            if p < 0.3:  # Low / Forest: Dark Blue/Green
                rgb[r, c] = [int(20 + p * 100), int(40 + p * 150), int(120 - p * 100)]
            elif p < 0.6:  # Moderate: Yellow / Amber
                t = (p - 0.3) / 0.3
                rgb[r, c] = [int(200 + t * 40), int(160 + t * 40), 20]
            else:  # High / Critical Deforestation: Vivid Red
                t = (p - 0.6) / 0.4
                rgb[r, c] = [int(220 + t * 35), int(60 * (1 - t)), int(20 * (1 - t))]

    return array_to_png_base64(rgb)


# =============================================================================
# Tab 1: AI Model Training Studio Manager
# =============================================================================

class StudioTrainingManager:
    """Manages interactive training sessions, live loss/IoU curves, and checkpoint tracking."""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.latest_run_summary: Dict[str, Any] = {}
        self.latest_history: List[Dict[str, Any]] = []
        self.is_training: bool = False

    def train_model(
        self,
        architecture: str = "siamese",
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        loss_type: str = "compound",
        learning_rate: float = 1e-3,
        epochs: int = 3,
        batch_size: int = 4,
        num_samples: int = 16,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute a full training run on synthetic geospatial tiles with real-time callbacks."""
        self.is_training = True
        device = torch.device("cpu")

        # 1. Generate Dataset & Spatial Split
        samples = generate_synthetic_dataset(num_samples=num_samples, height=64, width=64)
        splitter = SpatialBlockSplitter(grid_size_deg=0.1, train_ratio=0.75, val_ratio=0.25, test_ratio=0.0)
        train_idx, val_idx, _ = splitter.split(samples)

        dataset = GeospatialDeforestationDataset(samples)
        train_sub = torch.utils.data.Subset(dataset, train_idx)
        val_sub = torch.utils.data.Subset(dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_sub, batch_size=batch_size, shuffle=False)

        # 2. Model Initialization
        if architecture == "siamese":
            model = SiameseChangeDetector(in_channels=4, features=[16, 32, 64, 128])
            model_type_str = "siamese"
        elif architecture == "multimodal":
            model = MultimodalFusionDeforestationNet(opt_channels=4, sar_channels=3, features=[16, 32, 64, 128])
            model_type_str = "multimodal"
        else:
            model = UNet(in_channels=4, num_classes=1, features=[16, 32, 64, 128], use_vit_bottleneck=True)
            model_type_str = "unet"

        lora_stats: Optional[Dict[str, Any]] = None
        if use_lora:
            model, lora_stats = apply_lora_to_model(model, rank=lora_rank, alpha=lora_alpha)

        # 3. Loss Function
        if loss_type == "compound":
            criterion = CompoundDeforestationLoss(alpha=0.75, beta=0.25)
        elif loss_type == "focal_tversky":
            criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)
        elif loss_type == "tversky":
            criterion = TverskyLoss(alpha=0.7, beta=0.3)
        else:
            criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)

        # 4. Trainer Callback Bridge
        history_records: List[Dict[str, Any]] = []

        def _epoch_hook(ep: int, rec: Dict[str, Any]):
            history_records.append(rec)
            if progress_callback is not None:
                progress_callback(ep, epochs, rec)

        trainer = DeforestationTrainer(
            model=model,
            criterion=criterion,
            learning_rate=learning_rate,
            checkpoint_dir=self.checkpoint_dir,
            model_type=model_type_str,
            epoch_callback=_epoch_hook,
            enable_tensorboard=True,
        )

        # 5. Fit
        trainer.fit(train_loader, val_loader, epochs=epochs)
        summary = trainer.get_metrics_summary()

        self.latest_history = history_records
        self.latest_run_summary = {
            "architecture": architecture,
            "use_lora": use_lora,
            "lora_stats": lora_stats,
            "loss_type": loss_type,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "best_epoch": trainer.best_epoch,
            "best_val_iou": trainer.best_val_iou,
            "summary": summary,
            "history": history_records,
            "checkpoint_dir": os.path.abspath(self.checkpoint_dir),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.is_training = False
        return self.latest_run_summary


# =============================================================================
# Tab 2: Satellite Inference & Interactive Map Manager
# =============================================================================

class StudioInferenceManager:
    """Orchestrates bitemporal satellite data ingestion, deep learning inference, and polygonization."""

    def __init__(self):
        self.optical_prep = OpticalPreprocessor(sensor_type="sentinel2")
        self.sar_prep = SARPreprocessor(num_looks=4.4)
        self.models_cache: Dict[str, Any] = {}

    def _get_model(self, model_type: str):
        if model_type not in self.models_cache:
            if model_type == "siamese":
                m = SiameseChangeDetector(in_channels=4, features=[16, 32, 64, 128])
            elif model_type == "multimodal":
                m = MultimodalFusionDeforestationNet(opt_channels=4, sar_channels=3, features=[16, 32, 64, 128])
            else:
                m = UNet(in_channels=4, num_classes=1, features=[16, 32, 64, 128])
            m.eval()
            self.models_cache[model_type] = m
        return self.models_cache[model_type]

    def run_inference(
        self,
        deforestation_fraction: float = 0.08,
        cloud_fraction: float = 0.15,
        model_type: str = "siamese",
        threshold: float = 0.5,
        mmu_ha: float = 0.10,
        west: float = -60.05,
        north: float = -3.05,
    ) -> Dict[str, Any]:
        """Generate bitemporal scene, run neural inference, and extract polygon alerts."""
        t_start = time.perf_counter()

        # 1. Synthesize Scene
        scene = generate_synthetic_scene(
            height=128,
            width=128,
            seed=42,
            deforestation_fraction=deforestation_fraction,
            cloud_fraction=cloud_fraction,
        )

        opt_t1 = scene["t1_optical"]
        opt_t2 = scene["t2_optical"]
        sar_vv = scene["sar_vv"]
        sar_vh = scene["sar_vh"]

        # Optical Indices
        ndvi_t1 = self.optical_prep.compute_ndvi(opt_t1[2], opt_t1[3])
        ndvi_t2 = self.optical_prep.compute_ndvi(opt_t2[2], opt_t2[3])
        delta_ndvi = ndvi_t2 - ndvi_t1
        cloud_mask, shadow_mask = self.optical_prep.detect_cloud_and_shadow_mask(
            opt_t2[0], opt_t2[1], opt_t2[2], opt_t2[3]
        )

        # SAR Speckle Filtering
        sar_processed = self.sar_prep.process_sentinel1_pair(sar_vv, sar_vh, filter_type="lee")

        # 2. Neural Forward Pass
        if model_type == "siamese":
            model = self._get_model("siamese")
            t1_t = torch.from_numpy(opt_t1).unsqueeze(0)
            t2_t = torch.from_numpy(opt_t2).unsqueeze(0)
            with torch.no_grad():
                logits = model(t1_t, t2_t)
                probs = torch.sigmoid(logits).squeeze().numpy()
        elif model_type == "multimodal":
            model = self._get_model("multimodal")
            t2_t = torch.from_numpy(opt_t2).unsqueeze(0)
            sar_stacked = np.stack(
                [sar_processed["filtered_vv"], sar_processed["filtered_vh"], sar_processed["ratio_db"]],
                axis=0,
            )
            sar_t = torch.from_numpy(sar_stacked).unsqueeze(0).float()
            cloud_t = torch.from_numpy(cloud_mask).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(t2_t, sar_t, cloud_mask=cloud_t)
                probs = torch.sigmoid(logits).squeeze().numpy()
        elif model_type == "unet":
            model = self._get_model("unet")
            t2_t = torch.from_numpy(opt_t2).unsqueeze(0)
            with torch.no_grad():
                logits = model(t2_t)
                probs = torch.sigmoid(logits).squeeze().numpy()
        else:
            # Baseline NDVI delta
            probs = np.clip(-delta_ndvi * 2.0, 0.0, 1.0)

        # 3. Polygonization & MMU Filtering
        transform = GeoTransform(west=west, north=north, pixel_width=0.0001, pixel_height=-0.0001)
        polygonizer = Polygonizer(mmu_ha=mmu_ha, simplify_tolerance_deg=0.00005)
        geojson_fc = polygonizer.polygonize(probs, threshold=threshold, transform=transform)
        multipoly_feature = polygonizer.to_multipolygon_feature(geojson_fc)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # 4. Render Visual Layers as Base64 Data URIs
        img_t1_rgb = render_optical_rgb(opt_t1)
        img_t2_rgb = render_optical_rgb(opt_t2)
        img_t1_cir = render_optical_cir(opt_t1)
        img_t2_cir = render_optical_cir(opt_t2)
        img_ndvi_t1 = render_ndvi_colormap(ndvi_t1)
        img_ndvi_t2 = render_ndvi_colormap(ndvi_t2)
        img_sar = render_sar_composite(sar_vv, sar_vh)
        img_prob = render_probability_heatmap(probs)

        # Statistics & Severity breakdown
        features = geojson_fc.get("features", [])
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0}
        for f in features:
            lvl = f.get("properties", {}).get("alert_level", "MEDIUM")
            severity_counts[lvl] = severity_counts.get(lvl, 0) + 1

        return {
            "model_type": model_type,
            "threshold": threshold,
            "mmu_ha": mmu_ha,
            "inference_time_ms": round(elapsed_ms, 2),
            "total_alerts": geojson_fc["properties"]["total_alerts"],
            "total_deforested_ha": geojson_fc["properties"]["total_deforested_ha"],
            "severity_counts": severity_counts,
            "geojson": geojson_fc,
            "multipolygon": multipoly_feature,
            "mean_ndvi_baseline": float(np.mean(ndvi_t1)),
            "mean_ndvi_disturbed": float(np.mean(ndvi_t2)),
            "cloud_pixels": int(np.sum(cloud_mask)),
            "cloud_percentage": float(np.mean(cloud_mask) * 100),
            "layers": {
                "t1_rgb": img_t1_rgb,
                "t2_rgb": img_t2_rgb,
                "t1_cir": img_t1_cir,
                "t2_cir": img_t2_cir,
                "ndvi_t1": img_ndvi_t1,
                "ndvi_t2": img_ndvi_t2,
                "sar_composite": img_sar,
                "probability_heatmap": img_prob,
            },
        }


# =============================================================================
# Tab 3: EUDR Due Diligence & Regulatory Compliance Manager
# =============================================================================

class StudioComplianceManager:
    """Executes Due Diligence Statement (DDS) compliance audits and Olofsson accuracy verification."""

    def __init__(self):
        self.auditor = DDSAuditor()
        self.validator = EUDRValidator()
        self.olofsson = OlofssonAssessor()

    @staticmethod
    def get_preset_scenarios() -> Dict[str, Dict[str, Any]]:
        """Return predefined EUDR farm plot test cases covering compliance, violations, and boundary rules."""
        return {
            "Scenario A: Brazilian Soy Farm (Deforestation Violation)": {
                "statement_id": "DDS-2024-BRAZIL-SOYA-001",
                "operator_name": "Agroflorestal Rio Negro S.A.",
                "commodity": "soya",
                "country": "Brazil",
                "plots": [
                    {
                        "plot_id": "PLOT-MT-001",
                        "geometry_type": "Polygon",
                        "coordinates": [
                            [-60.015000, -3.015000],
                            [-59.985000, -3.015000],
                            [-59.985000, -2.985000],
                            [-60.015000, -2.985000],
                            [-60.015000, -3.015000],
                        ],
                        "declared_area_ha": 35.0,
                    }
                ],
                "alerts": [
                    {
                        "alert_id": "ALT-2023-AMZ-001",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-60.005000, -3.005000],
                                    [-59.995000, -3.005000],
                                    [-59.995000, -2.995000],
                                    [-60.005000, -2.995000],
                                    [-60.005000, -3.005000],
                                ]
                            ],
                        },
                        "detection_date": "2023-08-14T10:00:00Z",  # Post-2020 violation
                        "area_ha": 12.5,
                    }
                ],
            },
            "Scenario B: Indonesian Palm Oil Smallholder (<=4ha Point - Compliant)": {
                "statement_id": "DDS-2024-IDN-PALM-002",
                "operator_name": "Koperasi Petani Sawit Lestari",
                "commodity": "oil palm",
                "country": "Indonesia",
                "plots": [
                    {
                        "plot_id": "PLOT-SUM-002",
                        "geometry_type": "Point",
                        "coordinates": [101.456789, 0.543210],
                        "declared_area_ha": 3.2,
                    }
                ],
                "alerts": [],
            },
            "Scenario C: Ghanaian Cocoa Cooperative (>4ha Polygon - Compliant)": {
                "statement_id": "DDS-2024-GHA-COCOA-003",
                "operator_name": "Ashanti Green Cocoa Growers",
                "commodity": "cocoa",
                "country": "Ghana",
                "plots": [
                    {
                        "plot_id": "PLOT-ASH-003",
                        "geometry_type": "Polygon",
                        "coordinates": [
                            [-1.625000, 6.715000],
                            [-1.615000, 6.715000],
                            [-1.615000, 6.725000],
                            [-1.625000, 6.725000],
                            [-1.625000, 6.715000],
                        ],
                        "declared_area_ha": 6.8,
                    }
                ],
                "alerts": [],
            },
            "Scenario D: Paraguayan Cattle Ranch (>4ha Point - Mandate Violation)": {
                "statement_id": "DDS-2024-PRY-BEEF-004",
                "operator_name": "Estancia Chaco Ganadera",
                "commodity": "cattle",
                "country": "Paraguay",
                "plots": [
                    {
                        "plot_id": "PLOT-CHA-004",
                        "geometry_type": "Point",
                        "coordinates": [-59.789123, -22.123456],
                        "declared_area_ha": 18.5,  # Exceeds 4 ha with Point coordinates!
                    }
                ],
                "alerts": [],
            },
            "Scenario E: Peruvian Coffee Farm (Low Coordinate Precision - Rejection)": {
                "statement_id": "DDS-2024-PER-COFFEE-005",
                "operator_name": "Valle Sagrado Coffee Coop",
                "commodity": "coffee",
                "country": "Peru",
                "plots": [
                    {
                        "plot_id": "PLOT-CUZ-005",
                        "geometry_type": "Point",
                        "coordinates": [-71.95, -13.52],  # Only 2 decimals! Requires 6
                        "declared_area_ha": 2.5,
                    }
                ],
                "alerts": [],
            },
        }

    def audit_statement(
        self,
        statement_id: str,
        operator_name: str,
        commodity: str,
        country: str,
        plots: List[Dict[str, Any]],
        alerts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive EUDR legal and spatial due diligence audit."""
        alerts_list = alerts or []
        report: DDSAuditReport = self.auditor.audit_statement(
            statement_id=statement_id,
            operator_name=operator_name,
            commodity=commodity,
            country=country,
            plots=plots,
            deforestation_alerts=alerts_list,
        )
        return report.to_dict()

    def compute_olofsson_accuracy(
        self,
        sample_matrix: Optional[np.ndarray] = None,
        strata_areas_ha: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute Olofsson et al. (2014) area-adjusted stratified accuracy assessment."""
        if sample_matrix is None:
            # Standard stratified sampling matrix: [ [n_def_def, n_def_for], [n_for_def, n_for_for] ]
            sample_matrix = np.array([[85, 15], [8, 392]])
        if strata_areas_ha is None:
            # Strata areas: Stratum 1 (Mapped Deforestation) = 1,250 ha, Stratum 2 (Mapped Forest) = 18,750 ha
            strata_areas_ha = np.array([1250.0, 18750.0])

        report: OlofssonReport = self.olofsson.compute(sample_matrix, strata_areas_ha)

        return {
            "overall_accuracy_pct": round(report.overall_accuracy * 100, 2),
            "oa_margin_of_error_pct": round(1.96 * report.oa_standard_error * 100, 2),
            "oa_ci_95": [round(report.oa_ci_95[0] * 100, 2), round(report.oa_ci_95[1] * 100, 2)],
            "users_accuracy_pct": {k: round(v * 100, 2) for k, v in report.users_accuracy.items()},
            "producers_accuracy_pct": {k: round(v * 100, 2) for k, v in report.producers_accuracy.items()},
            "pa_margin_of_error_pct": {k: round(1.96 * v * 100, 2) for k, v in report.pa_standard_errors.items()},
            "estimated_areas_ha": {k: round(v, 2) for k, v in report.estimated_areas_ha.items()},
            "area_margins_of_error_ha": {k: round(v, 2) for k, v in report.area_margins_of_error_ha.items()},
            "summary_text": report.summary(),
        }


# =============================================================================
# Tab 4: STAC Satellite Catalog Explorer Manager
# =============================================================================

class StudioSTACManager:
    """Manages queries to Sentinel-2 and Sentinel-1 STAC catalogs with asset inspection."""

    def __init__(self):
        self.stac_manager = STACManager()
        self.validator = CatalogValidator()
        self.cached_items = self._init_sample_items()

    def _init_sample_items(self) -> List[Any]:
        return [
            self.stac_manager.create_sentinel2_item(
                item_id="S2B_MSIL2A_20231015T140051_N0509_R067_T20MPA",
                bbox=[-60.5, -3.5, -59.5, -2.5],
                dt=datetime(2023, 10, 15, 14, 0, 51, tzinfo=timezone.utc),
                asset_hrefs={
                    "B02": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/20/M/PA/2023/10/S2B_20MPA_20231015_0_L2A/B02.tif",
                    "B03": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/20/M/PA/2023/10/S2B_20MPA_20231015_0_L2A/B03.tif",
                    "B04": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/20/M/PA/2023/10/S2B_20MPA_20231015_0_L2A/B04.tif",
                    "B08": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/20/M/PA/2023/10/S2B_20MPA_20231015_0_L2A/B08.tif",
                },
                cloud_cover=3.8,
            ),
            self.stac_manager.create_sentinel2_item(
                item_id="S2A_MSIL2A_20231121T140049_N0509_R067_T20MPA",
                bbox=[-60.5, -3.5, -59.5, -2.5],
                dt=datetime(2023, 11, 21, 14, 0, 49, tzinfo=timezone.utc),
                asset_hrefs={
                    "B02": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/20/M/PA/2023/11/S2A_20MPA_20231121_0_L2A/B02.tif",
                    "B03": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/20/M/PA/2023/11/S2A_20MPA_20231121_0_L2A/B03.tif",
                    "B04": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/20/M/PA/2023/11/S2A_20MPA_20231121_0_L2A/B04.tif",
                    "B08": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/20/M/PA/2023/11/S2A_20MPA_20231121_0_L2A/B08.tif",
                },
                cloud_cover=14.2,
            ),
            self.stac_manager.create_sentinel1_item(
                item_id="S1A_IW_GRDH_1SDV_20231016T093022_RTC",
                bbox=[-60.5, -3.5, -59.5, -2.5],
                dt=datetime(2023, 10, 16, 9, 30, 22, tzinfo=timezone.utc),
                asset_hrefs={
                    "vv": "https://sentinel-s1-rtc.s3.amazonaws.com/GRD/2023/10/16/S1A_20231016/VV.tif",
                    "vh": "https://sentinel-s1-rtc.s3.amazonaws.com/GRD/2023/10/16/S1A_20231016/VH.tif",
                },
            ),
            self.stac_manager.create_sentinel1_item(
                item_id="S1A_IW_GRDH_1SDV_20231121T093021_RTC",
                bbox=[-60.5, -3.5, -59.5, -2.5],
                dt=datetime(2023, 11, 21, 9, 30, 21, tzinfo=timezone.utc),
                asset_hrefs={
                    "vv": "https://sentinel-s1-rtc.s3.amazonaws.com/GRD/2023/11/21/S1A_20231121/VV.tif",
                    "vh": "https://sentinel-s1-rtc.s3.amazonaws.com/GRD/2023/11/21/S1A_20231121/VH.tif",
                },
            ),
        ]

    def validate_url(self, url: str) -> Dict[str, Any]:
        """Validate COG URL via rapid HTTP HEAD request."""
        exists, code, msg = self.validator.validate_cog_url(url)
        return {"url": url, "exists": exists, "status_code": code, "message": msg}

    def heal_url(self, url: str) -> Dict[str, Any]:
        """Auto-heal malformed or non-existent Sentinel-2 S3 URL using Earth Search STAC."""
        return self.validator.auto_heal_sentinel2_url(url)

    def query_catalog(
        self,
        bbox: List[float],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        platform: Optional[str] = None,
        max_cloud_cover: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Query STAC items matching spatial bounding box and temporal filters."""
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

        matched = self.stac_manager.query_catalog(
            items=self.cached_items,
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            platform=platform,
        )

        if max_cloud_cover is not None:
            matched = [
                it for it in matched
                if it.properties.get("eo:cloud_cover") is None
                or it.properties.get("eo:cloud_cover") <= max_cloud_cover
            ]

        return [
            {
                "id": it.id,
                "platform": it.properties.get("platform", "sentinel-2"),
                "datetime": it.datetime.isoformat() if it.datetime else "N/A",
                "cloud_cover": it.properties.get("eo:cloud_cover"),
                "bbox": it.bbox,
                "assets": {
                    k: {
                        "href": v.href,
                        "title": v.title or k,
                        "media_type": v.media_type,
                    }
                    for k, v in it.assets.items()
                },
            }
            for it in matched
        ]


# =============================================================================
# Checkpoint & Artifact Manager
# =============================================================================

class CheckpointManager:
    """Inspects and manages saved deep learning model checkpoints."""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoint files with sizes and metadata."""
        if not os.path.exists(self.checkpoint_dir):
            return []

        results = []
        for fname in sorted(os.listdir(self.checkpoint_dir)):
            if fname.endswith(".pt") or fname.endswith(".pth"):
                fpath = os.path.join(self.checkpoint_dir, fname)
                stat = os.stat(fpath)
                info = {
                    "filename": fname,
                    "filepath": os.path.abspath(fpath),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "epoch": "N/A",
                    "best_val_iou": "N/A",
                }
                # Attempt to read PyTorch metadata dictionary
                try:
                    chk = torch.load(fpath, map_location="cpu", weights_only=False)
                    if isinstance(chk, dict):
                        info["epoch"] = chk.get("epoch", "N/A")
                        metrics = chk.get("metrics", {})
                        if isinstance(metrics, dict):
                            val_iou = metrics.get("iou", metrics.get("val_iou"))
                            if val_iou is not None:
                                info["best_val_iou"] = val_iou
                        if info["best_val_iou"] == "N/A" and isinstance(chk.get("history"), list) and chk["history"]:
                            info["best_val_iou"] = chk["history"][-1].get("val_iou", "N/A")
                except Exception:
                    pass
                results.append(info)

        return results
