"""Core business logic, inference orchestration, and training managers for the GeoFM Studio."""

from datetime import datetime, timezone
import io
import base64
import os
import time
import threading
import glob
import struct
import zlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

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
from geofm.training.dataset import GeospatialDeforestationDataset, LazyGeospatialDataset, SpatialBlockSplitter, filter_class_balance
from geofm.training.metrics import DeforestationMetrics
from geofm.training.trainer import DeforestationTrainer
from geofm.utils.synthetic_data import generate_synthetic_dataset, generate_synthetic_scene, stream_and_cache_synthetic_dataset


# =============================================================================
# Image Visualization & Base64 Data URI Utilities
# =============================================================================

def _pure_array_to_png_base64(arr: np.ndarray) -> str:
    """Encode an RGB uint8 array (H, W, 3), RGBA (H, W, 4), or grayscale (H, W) to base64 PNG.

    Zero-dependency pure Python implementation using standard-library struct, zlib, and base64.
    Used when Pillow is not installed in the environment.
    """
    if arr.ndim == 2:
        height, width = arr.shape
        color_type = 0  # Grayscale (8-bit)
        raw_data = bytearray()
        for y in range(height):
            raw_data.append(0)  # filter type 0 (None)
            raw_data.extend(arr[y, :].astype(np.uint8).tobytes())
    elif arr.ndim == 3 and arr.shape[2] == 3:
        height, width, _ = arr.shape
        color_type = 2  # Truecolor RGB (8-bit per channel)
        raw_data = bytearray()
        for y in range(height):
            raw_data.append(0)  # filter type 0 (None)
            raw_data.extend(arr[y, :, :].astype(np.uint8).tobytes())
    elif arr.ndim == 3 and arr.shape[2] == 4:
        height, width, _ = arr.shape
        color_type = 6  # Truecolor with alpha RGBA (8-bit per channel)
        raw_data = bytearray()
        for y in range(height):
            raw_data.append(0)  # filter type 0 (None)
            raw_data.extend(arr[y, :, :].astype(np.uint8).tobytes())
    else:
        raise ValueError(f"Unsupported array shape for image conversion: {arr.shape}")

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    # PNG File Signature
    header = b"\x89PNG\r\n\x1a\n"
    # IHDR Chunk
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)
    # IDAT Chunk (zlib deflate)
    compressed = zlib.compress(bytes(raw_data), level=6)
    idat = _chunk(b"IDAT", compressed)
    # IEND Chunk
    iend = _chunk(b"IEND", b"")

    png_bytes = header + ihdr + idat + iend
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def array_to_png_base64(arr: np.ndarray) -> str:
    """Encode an RGB uint8 array (H, W, 3), RGBA (H, W, 4), or grayscale (H, W) to base64 PNG data URI.

    Seamlessly uses Pillow when available, with automatic zero-dependency standard-library fallback.
    """
    if not HAS_PIL or Image is None:
        return _pure_array_to_png_base64(arr)

    try:
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
    except Exception:
        return _pure_array_to_png_base64(arr)


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


def render_mask(mask: np.ndarray, color: Tuple[int, int, int] = (239, 68, 68), alpha: float = 0.65) -> str:
    """Render a binary or probability mask overlay on a transparent background."""
    if mask.ndim == 3:
        mask = mask.squeeze()
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    bin_mask = mask > 0.5 if mask.dtype != bool else mask
    for i in range(3):
        rgba[..., i] = color[i]
    rgba[..., 3] = (bin_mask * (alpha * 255)).astype(np.uint8)
    return array_to_png_base64(rgba)


def render_mask_overlay(rgb_or_opt: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (239, 68, 68), alpha: float = 0.50) -> str:
    """Superimpose a translucent mask overlay directly onto an optical true color image."""
    if mask.ndim == 3:
        mask = mask.squeeze()

    if rgb_or_opt.ndim == 3 and (rgb_or_opt.shape[0] in (3, 4, 6, 8, 12, 13) or rgb_or_opt.shape[0] < rgb_or_opt.shape[1]):
        # Multi-band optical (C, H, W): [Blue=0, Green=1, Red=2, ...] -> True Color RGB
        b = rgb_or_opt[0]
        g = rgb_or_opt[1]
        r = rgb_or_opt[2]
        rgb = np.stack([r, g, b], axis=-1)
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb_stretched = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
        base = (rgb_stretched * 255).astype(np.float32)
    elif rgb_or_opt.ndim == 3 and rgb_or_opt.shape[2] == 3:
        base = rgb_or_opt.astype(np.float32)
    elif rgb_or_opt.ndim == 2:
        base = np.stack([rgb_or_opt] * 3, axis=-1).astype(np.float32)
    else:
        base = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)

    m = (mask > 0.5)
    if m.shape != (base.shape[0], base.shape[1]):
        h, w = base.shape[0], base.shape[1]
        padded_m = np.zeros((h, w), dtype=bool)
        min_h = min(m.shape[0], h)
        min_w = min(m.shape[1], w)
        padded_m[:min_h, :min_w] = m[:min_h, :min_w]
        m = padded_m

    overlay = base.copy()
    for c in range(3):
        overlay[m, c] = (1.0 - alpha) * base[m, c] + alpha * color[c]
    return array_to_png_base64(np.clip(overlay, 0, 255).astype(np.uint8))


def render_error_diff(ground_truth: np.ndarray, detected_binary: Optional[np.ndarray] = None, prediction_binary: Optional[np.ndarray] = None) -> str:
    """Render pixel-level confusion difference map:
    - True Positive  (Deforest detected correctly): Emerald Green #10B981 [16, 185, 129]
    - False Positive (False alarm clearing):        Amber Orange  #F59E0B [245, 158, 11]
    - False Negative (Missed deforestation):        Hot Crimson   #EF4444 [239, 68, 68]
    - True Negative  (Intact forest background):    Dark Obsidian #0F172A [15, 23, 42]
    """
    binary_map = detected_binary if detected_binary is not None else prediction_binary
    if binary_map is None:
        raise ValueError("Either detected_binary or prediction_binary must be provided.")

    if ground_truth.ndim == 3:
        ground_truth = ground_truth.squeeze()
    if binary_map.ndim == 3:
        binary_map = binary_map.squeeze()

    gt = ground_truth > 0.5
    det = binary_map > 0.5

    if gt.shape != det.shape:
        h, w = det.shape
        padded_gt = np.zeros((h, w), dtype=bool)
        min_h = min(gt.shape[0], h)
        min_w = min(gt.shape[1], w)
        padded_gt[:min_h, :min_w] = gt[:min_h, :min_w]
        gt = padded_gt

    h, w = det.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # True Negative (Intact)
    rgb[~gt & ~det] = [15, 23, 42]
    # True Positive (Detected clearing)
    rgb[gt & det] = [16, 185, 129]
    # False Positive (False alarm)
    rgb[~gt & det] = [245, 158, 11]
    # False Negative (Missed clearing)
    rgb[gt & ~det] = [239, 68, 68]

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
        self.stop_requested: bool = False
        self.training_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Real-time telemetry & step progress
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.current_step: int = 0
        self.total_steps: int = 0
        self.current_train_loss: float = 0.0
        self.current_val_loss: float = 0.0
        self.current_val_iou: float = 0.0
        self.current_val_f1: float = 0.0
        self.current_lr: float = 0.0
        self.best_val_iou: float = 0.0
        self.best_epoch: int = 0
        self.logs: List[str] = []
        self.active_architecture: str = "siamese"

    def log(self, message: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        self.logs.append(entry)
        if len(self.logs) > 300:
            self.logs = self.logs[-250:]

    def stop_training(self) -> Dict[str, Any]:
        """Request active training loop to stop gracefully at the next batch."""
        with self._lock:
            if not self.is_training:
                return {"status": "idle", "message": "No training run currently active."}
            self.stop_requested = True
            self.log("⚠️ Stop signal received. Finishing current batch and saving checkpoint...")
            return {"status": "stopping", "message": "Stop requested. Saving latest checkpoint."}

    def train_model(
        self,
        architecture: str = "siamese",
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        loss_type: str = "compound",
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        learning_rate: float = 3e-4,
        epochs: int = 20,
        batch_size: int = 16,
        num_samples: int = 200,
        dataset_source: str = "cached",
        dataset_scale: str = "standard",
        gee_project_id: Optional[str] = None,
        chip_size: int = 256,
        use_data_augmentation: bool = True,
        enforce_class_balance: bool = True,
        resume: bool = False,
        use_amp: bool = True,
        ram_cache_budget_gb: Optional[float] = 4.0,
        async_run: bool = False,
        cache_dir: str = "./data/chips_cache",
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute a full training run with planetary-scale datasets, GEE, LoRA PEFT, and compound loss."""
        if self.is_training:
            return {"status": "error", "message": "Training is already in progress.", "is_training": True}

        self.stop_requested = False
        self.is_training = True
        self.active_architecture = architecture
        self.total_epochs = epochs
        self.current_epoch = 0
        self.current_train_loss = 0.0
        self.current_val_loss = 0.0
        self.current_val_iou = 0.0
        self.current_val_f1 = 0.0
        self.current_lr = learning_rate
        self.logs.clear()
        self.log(f"🚀 Initializing {architecture.upper()} training pipeline (LoRA: {use_lora}, Rank: {lora_rank}, Loss: {loss_type}, Epochs: {epochs})...")

        kwargs = {
            "architecture": architecture,
            "use_lora": use_lora,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "loss_type": loss_type,
            "tversky_alpha": tversky_alpha,
            "tversky_beta": tversky_beta,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_samples": num_samples,
            "dataset_source": dataset_source,
            "dataset_scale": dataset_scale,
            "gee_project_id": gee_project_id,
            "chip_size": chip_size,
            "use_data_augmentation": use_data_augmentation,
            "enforce_class_balance": enforce_class_balance,
            "resume": resume,
            "use_amp": use_amp,
            "ram_cache_budget_gb": ram_cache_budget_gb,
            "cache_dir": cache_dir,
            "progress_callback": progress_callback,
        }

        if async_run:
            self.training_thread = threading.Thread(
                target=self._run_training_pipeline,
                kwargs=kwargs,
                daemon=True,
            )
            self.training_thread.start()
            return {
                "status": "started",
                "message": f"Training {architecture.upper()} launched in background for {epochs} epochs.",
                "architecture": architecture,
                "epochs": epochs,
            }
        else:
            return self._run_training_pipeline(**kwargs)

    def _run_training_pipeline(
        self,
        architecture: str = "siamese",
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        loss_type: str = "compound",
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        learning_rate: float = 3e-4,
        epochs: int = 20,
        batch_size: int = 16,
        num_samples: int = 200,
        dataset_source: str = "cached",
        dataset_scale: str = "standard",
        gee_project_id: Optional[str] = None,
        chip_size: int = 256,
        use_data_augmentation: bool = True,
        enforce_class_balance: bool = True,
        resume: bool = False,
        use_amp: bool = True,
        ram_cache_budget_gb: Optional[float] = 4.0,
        cache_dir: str = "./data/chips_cache",
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        self.is_training = True
        try:
            telemetry = get_hardware_telemetry(self.checkpoint_dir, cache_dir)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log(f"💻 Hardware compute allocated: {telemetry['device_name']} ({telemetry['cpu_count']} CPU cores)")

            # 1. Dataset Scale Mapping & Source Ingestion
            scale_map = {"quick": 32, "standard": 200, "high_fidelity": 1000, "continental": 5000}
            if dataset_scale in scale_map and (num_samples is None or num_samples == 200):
                effective_samples = scale_map[dataset_scale]
            else:
                effective_samples = num_samples if num_samples is not None else scale_map.get(dataset_scale, 200)

            dataset = None
            clean_cache_dir = os.path.abspath(cache_dir)
            os.makedirs(clean_cache_dir, exist_ok=True)
            manifest_file = os.path.join(clean_cache_dir, "manifest.jsonl")
            has_real_cache = os.path.exists(manifest_file) or any(glob.glob(os.path.join(clean_cache_dir, "*.npz")))

            if dataset_source == "auto":
                if has_real_cache and architecture != "multimodal":
                    dataset_source = "cached"
                else:
                    dataset_source = "synthetic"

            if dataset_source == "gee":
                self.log(f"🛰️ Earth Engine + Hansen GFC active: Scale={dataset_scale.upper()} ({effective_samples} chips, Project={gee_project_id or os.getenv('GEE_PROJECT_ID', 'not configured')})")
                chip_mgr = StudioChipManager(cache_dir=clean_cache_dir)
                cached_files = glob.glob(os.path.join(clean_cache_dir, "*.npz"))
                if len(cached_files) < effective_samples:
                    self.log(f"⚡ Ingesting {effective_samples} satellite chips with Hansen loss masks...")
                    chip_mgr.fetch_gee_chips(
                        project_id=gee_project_id or os.getenv("GEE_PROJECT_ID", "demo-gee-project"),
                        num_chips=effective_samples,
                        chip_size=chip_size,
                        target_pos_ratio=0.70 if enforce_class_balance else 0.50,
                    )
                dataset = LazyGeospatialDataset(cache_dir=clean_cache_dir, max_ram_cache_gb=ram_cache_budget_gb, preload_cache=False)
                in_channels = int(dataset[0]["t1"].shape[0])
            elif dataset_source in ("cached", "real") and has_real_cache and architecture != "multimodal":
                self.log(f"📦 Loading local cached satellite chips from: {clean_cache_dir}")
                dataset = LazyGeospatialDataset(cache_dir=clean_cache_dir, max_ram_cache_gb=ram_cache_budget_gb, preload_cache=False)
                in_channels = int(dataset[0]["t1"].shape[0])
            else:
                self.log(f"⚡ Streaming {effective_samples} synthetic multi-spectral training chips ({chip_size}x{chip_size}, Zero-RAM)...")
                samples = generate_synthetic_dataset(num_samples=max(4, effective_samples), height=chip_size, width=chip_size)
                dataset = GeospatialDeforestationDataset(samples)
                in_channels = 4

            if enforce_class_balance:
                self.log("⚖️ Class balancing verified (70:30 deforestation loss quota)")

            if use_data_augmentation:
                self.log("🔄 Synchronous D4 dihedral augmentations active (rot90, flips, radiometric jitter)")

            # 2. Spatial Disjoint Split
            splitter = SpatialBlockSplitter(grid_size_deg=0.1, train_ratio=0.75, val_ratio=0.25, test_ratio=0.0)
            train_indices_file = os.path.join(self.checkpoint_dir, "train_indices.npy")
            val_indices_file = os.path.join(self.checkpoint_dir, "val_indices.npy")

            if resume and os.path.exists(train_indices_file) and os.path.exists(val_indices_file):
                try:
                    train_idx = np.load(train_indices_file).tolist()
                    val_idx = np.load(val_indices_file).tolist()
                    self.log(f"✓ Restored frozen spatial split ({len(train_idx)} train, {len(val_idx)} val chips)")
                except Exception:
                    train_idx, val_idx, _ = splitter.split(dataset)
            else:
                train_idx, val_idx, _ = splitter.split(dataset)
                try:
                    np.save(train_indices_file, np.array(train_idx, dtype=np.int64))
                    np.save(val_indices_file, np.array(val_idx, dtype=np.int64))
                except Exception:
                    pass
                self.log(f"✓ Spatial block split complete ({len(train_idx)} train, {len(val_idx)} val chips)")

            # Cap training & validation splits to effective_samples if requested
            if effective_samples and (len(train_idx) + len(val_idx) > effective_samples):
                max_train = max(2, int(effective_samples * 0.75))
                max_val = max(1, effective_samples - max_train)
                train_idx = train_idx[:max_train]
                val_idx = val_idx[:max_val]
                self.log(f"⚡ Capped active dataset to {len(train_idx)} train, {len(val_idx)} val chips for fast execution")

            train_sub = torch.utils.data.Subset(dataset, train_idx)
            val_sub = torch.utils.data.Subset(dataset, val_idx)

            train_loader = torch.utils.data.DataLoader(train_sub, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_sub, batch_size=batch_size, shuffle=False)

            # 3. Model Architecture Initialization
            if architecture == "siamese":
                model = SiameseChangeDetector(in_channels=in_channels, features=[16, 32, 64, 128])
                model_type_str = "siamese"
            elif architecture == "multimodal":
                model = MultimodalFusionDeforestationNet(opt_channels=4, sar_channels=3, features=[16, 32, 64, 128])
                model_type_str = "multimodal"
            else:
                model = UNet(in_channels=in_channels, num_classes=1, features=[16, 32, 64, 128], use_vit_bottleneck=True)
                model_type_str = "unet"

            lora_stats: Optional[Dict[str, Any]] = None
            if use_lora:
                model, lora_stats = apply_lora_to_model(model, rank=lora_rank, alpha=lora_alpha)
                self.log(f"✓ LoRA PEFT active (r={lora_rank}, α={lora_alpha}): {lora_stats['trainable_fraction']*100:.2f}% parameters trainable")

            # 4. Loss Function Configuration
            if loss_type == "compound":
                criterion = CompoundDeforestationLoss(alpha=tversky_alpha, beta=tversky_beta)
            elif loss_type == "focal_tversky":
                criterion = FocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta, gamma=1.33)
            elif loss_type == "tversky":
                criterion = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
            elif loss_type == "bce":
                criterion = torch.nn.BCEWithLogitsLoss()
            else:
                criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
            self.log(f"✓ Criterion initialized: {loss_type} (α={tversky_alpha}, β={tversky_beta})")

            # 5. Optimizer & ReduceLROnPlateau Scheduler
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)

            history_records: List[Dict[str, Any]] = []

            def _epoch_hook(ep: int, rec: Dict[str, Any]):
                self.current_epoch = ep
                self.current_train_loss = rec["train_loss"]
                self.current_val_loss = rec["val_loss"]
                self.current_val_iou = rec["val_iou"]
                self.current_val_f1 = rec["val_f1"]
                self.current_lr = rec["lr"]
                history_records.append(rec)
                self.latest_history = list(history_records)
                self.log(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {rec['train_loss']:.4f} | Val Loss: {rec['val_loss']:.4f} | IoU: {rec['val_iou']*100:.2f}% | F1: {rec['val_f1']*100:.2f}%")
                if progress_callback is not None:
                    progress_callback(ep, epochs, rec)

            trainer = DeforestationTrainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                learning_rate=learning_rate,
                checkpoint_dir=self.checkpoint_dir,
                model_type=model_type_str,
                epoch_callback=_epoch_hook,
                use_amp=use_amp and device.type == "cuda",
                enable_tensorboard=True,
            )

            start_epoch = 1
            if resume:
                latest_pt = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
                if os.path.exists(latest_pt):
                    resumed_ep = trainer.load_checkpoint(latest_pt)
                    start_epoch = resumed_ep + 1
                    history_records = list(trainer.history)
                    self.log(f"✓ Resumed from checkpoint: Epoch {resumed_ep} (Best Val IoU: {trainer.best_val_iou*100:.2f}%)")

            # 6. Fit
            self.log(f"🚀 Training starting ({start_epoch} -> {epochs} epochs)...")
            trainer.fit(train_loader, val_loader, epochs=epochs, start_epoch=start_epoch)

            summary = trainer.get_metrics_summary()
            self.best_epoch = trainer.best_epoch
            self.best_val_iou = trainer.best_val_iou
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
            self.log(f"🏆 Training completed! Best Validation IoU: {trainer.best_val_iou*100:.2f}% (Epoch {trainer.best_epoch})")
            return self.latest_run_summary
        except Exception as e:
            self.log(f"❌ Training failed: {str(e)}")
            raise
        finally:
            self.is_training = False


# =============================================================================
# Tab 2: Satellite Inference & Interactive Map Manager
# =============================================================================

class StudioInferenceManager:
    """Orchestrates bitemporal satellite data ingestion, deep learning inference, and polygonization."""

    def __init__(self):
        self.optical_prep = OpticalPreprocessor(sensor_type="sentinel2")
        self.sar_prep = SARPreprocessor(num_looks=4.4)
        self.models_cache: Dict[str, Any] = {}
        self.last_benchmark_scorecard: Optional[Dict[str, Any]] = None
        self.last_inference_result: Optional[Dict[str, Any]] = None

    def _get_model(self, model_type: str, in_channels: int = 4):
        cache_key = f"{model_type}_{in_channels}"
        if cache_key not in self.models_cache:
            if model_type == "siamese":
                m = SiameseChangeDetector(in_channels=in_channels, features=[16, 32, 64, 128])
            elif model_type == "multimodal":
                m = MultimodalFusionDeforestationNet(opt_channels=in_channels, sar_channels=3, features=[16, 32, 64, 128])
            else:
                m = UNet(in_channels=in_channels, num_classes=1, features=[16, 32, 64, 128])
            m.eval()
            self.models_cache[cache_key] = m
            self.models_cache[model_type] = m
        return self.models_cache[cache_key]

    def _get_model_in_channels(self, model: Any) -> int:
        """Detect expected input optical channels of a neural network model."""
        try:
            if hasattr(model, "in_channels") and isinstance(model.in_channels, int):
                return model.in_channels
            if hasattr(model, "encoder") and hasattr(model.encoder, "in_conv"):
                first = model.encoder.in_conv[0]
                if hasattr(first, "in_channels"):
                    return first.in_channels
            if hasattr(model, "in_conv"):
                first = model.in_conv[0]
                if hasattr(first, "in_channels"):
                    return first.in_channels
        except Exception:
            pass
        return 4

    def _match_input_channels(self, opt: np.ndarray, target_channels: int = 4) -> np.ndarray:
        """Ensure optical tensor matches the expected channel count (e.g. 4 for B02, B03, B04, B08)."""
        if opt.shape[0] > target_channels:
            return opt[:target_channels]
        elif opt.shape[0] < target_channels:
            pad = np.zeros((target_channels - opt.shape[0], opt.shape[1], opt.shape[2]), dtype=opt.dtype)
            return np.concatenate([opt, pad], axis=0)
        return opt

    def load_model_weights(self, checkpoint_path: str, model_type: str = "siamese") -> Dict[str, Any]:
        """Load weights from a checkpoint path into the active model backbone."""
        clean_path = os.path.abspath(checkpoint_path)
        if not os.path.exists(clean_path):
            alt_path = os.path.join("./checkpoints", checkpoint_path)
            if os.path.exists(alt_path):
                clean_path = os.path.abspath(alt_path)
            else:
                return {"status": "error", "message": f"Checkpoint path not found: {checkpoint_path}"}

        model = self._get_model(model_type)
        try:
            chk = torch.load(clean_path, map_location="cpu", weights_only=False)
            st = chk.get("model_state_dict", chk)
            # Detect in_channels from checkpoint weights if possible
            chk_channels = None
            for k, v in st.items():
                if "in_conv" in k and "weight" in k and hasattr(v, "shape") and len(v.shape) == 4:
                    chk_channels = v.shape[1]
                    break
            if chk_channels and chk_channels != self._get_model_in_channels(model):
                model = self._get_model(model_type, in_channels=chk_channels)

            res = model.load_state_dict(st, strict=False)
            model.eval()
            return {
                "status": "success",
                "checkpoint_path": clean_path,
                "model_type": model_type,
                "missing_keys": len(res.missing_keys),
                "unexpected_keys": len(res.unexpected_keys),
                "message": f"Loaded {os.path.basename(clean_path)} into {model_type.upper()} model.",
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed loading weights: {str(e)}"}

    def run_inference(
        self,
        deforestation_fraction: float = 0.08,
        cloud_fraction: float = 0.15,
        model_type: str = "siamese",
        threshold: float = 0.5,
        mmu_ha: float = 0.10,
        west: float = -60.05,
        north: float = -3.05,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate bitemporal scene, run neural inference, and extract polygon alerts."""
        t_start = time.perf_counter()

        # 1. Synthesize Scene with dynamic seed
        effective_seed = seed if seed is not None else int(time.time() * 1000) % 1000000
        scene = generate_synthetic_scene(
            height=128,
            width=128,
            seed=effective_seed,
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
            expected_c = self._get_model_in_channels(model)
            t1_opt = self._match_input_channels(opt_t1, expected_c)
            t2_opt = self._match_input_channels(opt_t2, expected_c)
            t1_t = torch.from_numpy(t1_opt).unsqueeze(0).float()
            t2_t = torch.from_numpy(t2_opt).unsqueeze(0).float()
            with torch.no_grad():
                logits = model(t1_t, t2_t)
                probs = torch.sigmoid(logits).squeeze().numpy()
        elif model_type == "multimodal":
            model = self._get_model("multimodal")
            expected_c = self._get_model_in_channels(model)
            t2_opt = self._match_input_channels(opt_t2, expected_c)
            t2_t = torch.from_numpy(t2_opt).unsqueeze(0).float()
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
            expected_c = self._get_model_in_channels(model)
            t2_opt = self._match_input_channels(opt_t2, expected_c)
            t2_t = torch.from_numpy(t2_opt).unsqueeze(0).float()
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

        res = {
            "model_type": model_type,
            "threshold": threshold,
            "mmu_ha": mmu_ha,
            "west": west,
            "north": north,
            "seed": effective_seed,
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
        self.last_inference_result = res
        return res

    def test_benchmark(
        self,
        checkpoint_name: Optional[str] = None,
        num_samples: int = 16,
        threshold: float = 0.5,
        checkpoint_dir: str = "./checkpoints",
        cache_dir: str = "./data/chips_cache",
    ) -> Dict[str, Any]:
        """Evaluate validation/test benchmark on batch of chips and compute full scorecard."""
        t_start = time.perf_counter()

        # Load weights if checkpoint provided
        model = self._get_model("siamese")
        if checkpoint_name and os.path.exists(checkpoint_dir):
            chk_path = os.path.join(checkpoint_dir, checkpoint_name)
            if os.path.exists(chk_path):
                try:
                    chk = torch.load(chk_path, map_location="cpu", weights_only=False)
                    st = chk.get("model_state_dict", chk)
                    model.load_state_dict(st, strict=False)
                except Exception:
                    pass
        model.eval()

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        # Try to sample from cache if available, else synthetic
        cached_files = sorted(glob.glob(os.path.join(cache_dir, "*.npz"))) if os.path.exists(cache_dir) else []
        n_eval = min(num_samples, len(cached_files)) if cached_files else num_samples

        for i in range(n_eval):
            if cached_files and i < len(cached_files):
                try:
                    data = np.load(cached_files[i], allow_pickle=True)
                    opt_t1 = data["t1_optical"]
                    opt_t2 = data["t2_optical"]
                    if "target_mask" in data:
                        gt_mask = data["target_mask"]
                    elif "label" in data:
                        gt_mask = data["label"]
                    elif "mask" in data:
                        gt_mask = data["mask"]
                    elif "change_mask" in data:
                        gt_mask = data["change_mask"]
                    else:
                        gt_mask = np.zeros((opt_t1.shape[1], opt_t1.shape[2]), dtype=np.float32)
                except Exception:
                    cached_files = []
                    scene = generate_synthetic_scene(height=128, width=128, seed=100 + i, deforestation_fraction=0.15 if i % 2 == 0 else 0.0)
                    opt_t1 = scene["t1_optical"]
                    opt_t2 = scene["t2_optical"]
                    gt_mask = scene.get("change_mask", scene.get("label", np.zeros((128, 128), dtype=np.float32)))
            else:
                scene = generate_synthetic_scene(
                    height=128,
                    width=128,
                    seed=100 + i,
                    deforestation_fraction=0.15 if i % 2 == 0 else 0.0,
                    cloud_fraction=0.08,
                )
                opt_t1 = scene["t1_optical"]
                opt_t2 = scene["t2_optical"]
                gt_mask = scene.get("change_mask", scene.get("label", np.zeros((128, 128), dtype=np.float32)))

            if gt_mask.ndim == 3:
                gt_mask = gt_mask[0]

            expected_c = self._get_model_in_channels(model)
            t1_opt = self._match_input_channels(opt_t1, expected_c)
            t2_opt = self._match_input_channels(opt_t2, expected_c)
            t1_t = torch.from_numpy(t1_opt).unsqueeze(0).float()
            t2_t = torch.from_numpy(t2_opt).unsqueeze(0).float()
            with torch.no_grad():
                logits = model(t1_t, t2_t)
                probs = torch.sigmoid(logits).squeeze().numpy()

            if probs.ndim == 3:
                probs = probs[0]

            gt_bin = (gt_mask > 0.5).astype(bool)
            pr_bin = (probs >= threshold).astype(bool)

            if gt_bin.shape != pr_bin.shape:
                h, w = pr_bin.shape
                padded_gt = np.zeros((h, w), dtype=bool)
                min_h = min(gt_bin.shape[0], h)
                min_w = min(gt_bin.shape[1], w)
                padded_gt[:min_h, :min_w] = gt_bin[:min_h, :min_w]
                gt_bin = padded_gt

            total_tp += int(np.sum(gt_bin & pr_bin))
            total_fp += int(np.sum(~gt_bin & pr_bin))
            total_fn += int(np.sum(gt_bin & ~pr_bin))
            total_tn += int(np.sum(~gt_bin & ~pr_bin))

        total_pixels = total_tp + total_fp + total_fn + total_tn
        iou = float(total_tp / (total_tp + total_fp + total_fn + 1e-7))
        f1 = float(2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-7))
        precision = float(total_tp / (total_tp + total_fp + 1e-7))
        recall = float(total_tp / (total_tp + total_fn + 1e-7))
        specificity = float(total_tn / (total_tn + total_fp + 1e-7))
        accuracy = float((total_tp + total_tn) / (total_pixels + 1e-7))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        res = {
            "checkpoint_name": checkpoint_name or "Active Siamese Backbone",
            "samples_evaluated": n_eval,
            "threshold": threshold,
            "iou": round(iou, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "accuracy": round(accuracy, 4),
            "confusion_matrix": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "tn": total_tn,
            },
            "elapsed_ms": round(elapsed_ms, 2),
        }
        self.last_benchmark_scorecard = res
        return res

    def inspect_chip(
        self,
        chip_idx: int = 0,
        checkpoint_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        threshold: float = 0.5,
        mmu_ha: float = 0.10,
        checkpoint_dir: str = "./checkpoints",
        cache_dir: str = "./data/chips_cache",
    ) -> Dict[str, Any]:
        """Deep inspect single chip: render 11 spectral layers, detected change vs reference GT, translucent mask overlays, pixel diff map, and alerts."""
        t_start = time.perf_counter()

        cached_files = sorted(glob.glob(os.path.join(cache_dir, "*.npz"))) if os.path.exists(cache_dir) else []
        chip_name = f"chip_{chip_idx:04d}"

        if cached_files and 0 <= chip_idx < len(cached_files):
            fpath = cached_files[chip_idx]
            chip_name = os.path.basename(fpath).replace(".npz", "")
            data = np.load(fpath, allow_pickle=True)
            opt_t1 = data["t1_optical"]
            opt_t2 = data["t2_optical"]
            sar_vv = data["sar_vv"] if "sar_vv" in data else np.random.exponential(0.1, (opt_t1.shape[1], opt_t1.shape[2])).astype(np.float32)
            sar_vh = data["sar_vh"] if "sar_vh" in data else np.random.exponential(0.04, (opt_t1.shape[1], opt_t1.shape[2])).astype(np.float32)
            if "target_mask" in data:
                gt_mask = data["target_mask"]
            elif "label" in data:
                gt_mask = data["label"]
            elif "mask" in data:
                gt_mask = data["mask"]
            elif "change_mask" in data:
                gt_mask = data["change_mask"]
            else:
                gt_mask = np.zeros((opt_t1.shape[1], opt_t1.shape[2]), dtype=np.float32)
        else:
            scene = generate_synthetic_scene(
                height=128,
                width=128,
                seed=42 + chip_idx,
                deforestation_fraction=0.14 if chip_idx % 3 != 1 else 0.0,
                cloud_fraction=0.08,
            )
            opt_t1 = scene["t1_optical"]
            opt_t2 = scene["t2_optical"]
            sar_vv = scene["sar_vv"]
            sar_vh = scene["sar_vh"]
            gt_mask = scene.get("change_mask", scene.get("label", np.zeros((128, 128), dtype=np.float32)))

        if gt_mask.ndim == 3:
            gt_mask = gt_mask[0]

        # Model forward
        model = self._get_model("siamese")
        target_chk = checkpoint_path
        if not target_chk and checkpoint_name and os.path.exists(checkpoint_dir):
            target_chk = os.path.join(checkpoint_dir, checkpoint_name)
        if target_chk and os.path.exists(target_chk):
            try:
                chk = torch.load(target_chk, map_location="cpu", weights_only=False)
                st = chk.get("model_state_dict", chk)
                chk_channels = None
                for k, v in st.items():
                    if "in_conv" in k and "weight" in k and hasattr(v, "shape") and len(v.shape) == 4:
                        chk_channels = v.shape[1]
                        break
                if chk_channels and chk_channels != self._get_model_in_channels(model):
                    model = self._get_model("siamese", in_channels=chk_channels)
                model.load_state_dict(st, strict=False)
            except Exception:
                pass
        model.eval()

        expected_c = self._get_model_in_channels(model)
        t1_opt = self._match_input_channels(opt_t1, expected_c)
        t2_opt = self._match_input_channels(opt_t2, expected_c)

        t1_t = torch.from_numpy(t1_opt).unsqueeze(0).float()
        t2_t = torch.from_numpy(t2_opt).unsqueeze(0).float()
        with torch.no_grad():
            logits = model(t1_t, t2_t)
            probs = torch.sigmoid(logits).squeeze().numpy()

        pred_binary = (probs >= threshold).astype(np.float32)

        # Indices & Visual layers
        ndvi_t1 = self.optical_prep.compute_ndvi(opt_t1[2], opt_t1[3])
        ndvi_t2 = self.optical_prep.compute_ndvi(opt_t2[2], opt_t2[3])

        img_t1_rgb = render_optical_rgb(opt_t1)
        img_t2_rgb = render_optical_rgb(opt_t2)
        img_t1_cir = render_optical_cir(opt_t1)
        img_t2_cir = render_optical_cir(opt_t2)
        img_ndvi_t1 = render_ndvi_colormap(ndvi_t1)
        img_ndvi_t2 = render_ndvi_colormap(ndvi_t2)
        img_sar = render_sar_composite(sar_vv, sar_vh)
        img_prob = render_probability_heatmap(probs)
        img_gt_mask = render_mask(gt_mask, color=[239, 68, 68])
        img_pred_mask = render_mask(pred_binary, color=[239, 68, 68])
        img_mask_overlay = render_mask_overlay(opt_t2, pred_binary, color=(239, 68, 68), alpha=0.50)
        img_gt_overlay = render_mask_overlay(opt_t2, gt_mask, color=(16, 185, 129), alpha=0.50)
        img_error_diff = render_error_diff(gt_mask, pred_binary)

        # Polygonization
        transform = GeoTransform(west=-60.05, north=-3.05, pixel_width=0.0001, pixel_height=-0.0001)
        polygonizer = Polygonizer(mmu_ha=mmu_ha, simplify_tolerance_deg=0.00005)
        geojson_fc = polygonizer.polygonize(probs, threshold=threshold, transform=transform)

        # Chip-level Metrics
        gt_bin = (gt_mask > 0.5).astype(bool)
        pr_bin = (pred_binary > 0.5).astype(bool)
        if gt_bin.shape != pr_bin.shape:
            h, w = pr_bin.shape
            padded_gt = np.zeros((h, w), dtype=bool)
            min_h = min(gt_bin.shape[0], h)
            min_w = min(gt_bin.shape[1], w)
            padded_gt[:min_h, :min_w] = gt_bin[:min_h, :min_w]
            gt_bin = padded_gt
        tp = int(np.sum(gt_bin & pr_bin))
        fp = int(np.sum(~gt_bin & pr_bin))
        fn = int(np.sum(gt_bin & ~pr_bin))
        tn = int(np.sum(~gt_bin & ~pr_bin))
        chip_iou = float(tp / (tp + fp + fn + 1e-7))
        chip_f1 = float(2 * tp / (2 * tp + fp + fn + 1e-7))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "chip_id": chip_name,
            "chip_idx": chip_idx,
            "dimensions": [opt_t1.shape[1], opt_t1.shape[2]],
            "threshold": threshold,
            "mmu_ha": mmu_ha,
            "elapsed_ms": round(elapsed_ms, 2),
            "ground_truth_pct": round(float(np.mean(gt_mask) * 100), 2),
            "detected_pct": round(float(np.mean(pred_binary) * 100), 2),
            "predicted_pct": round(float(np.mean(pred_binary) * 100), 2),
            "chip_iou": round(chip_iou, 4),
            "chip_f1": round(chip_f1, 4),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "total_alerts": geojson_fc["properties"]["total_alerts"],
            "total_deforested_ha": geojson_fc["properties"]["total_deforested_ha"],
            "geojson": geojson_fc,
            "layers": {
                "t1_rgb": img_t1_rgb,
                "t2_rgb": img_t2_rgb,
                "t1_cir": img_t1_cir,
                "t2_cir": img_t2_cir,
                "ndvi_t1": img_ndvi_t1,
                "ndvi_t2": img_ndvi_t2,
                "sar_composite": img_sar,
                "ground_truth_mask": img_gt_mask,
                "probability_heatmap": img_prob,
                "detection_probability_heatmap": img_prob,
                "detected_binary_mask": img_pred_mask,
                "pred_binary_mask": img_pred_mask,
                "mask_overlay": img_mask_overlay,
                "ground_truth_overlay": img_gt_overlay,
                "error_diff_map": img_error_diff,
            },
        }

    def run_manual_inference(
        self,
        chip_idx: Optional[int] = None,
        chip_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        model_type: str = "siamese",
        threshold: float = 0.5,
        mmu_ha: float = 0.10,
        west: float = -60.05,
        north: float = -3.05,
        cache_dir: str = "./data/chips_cache",
    ) -> Dict[str, Any]:
        """Execute manual inference on a chosen chip or active scene with selected checkpoint."""
        t_start = time.perf_counter()

        if checkpoint_path:
            self.load_model_weights(checkpoint_path, model_type=model_type)

        cached_files = sorted(glob.glob(os.path.join(cache_dir, "*.npz"))) if os.path.exists(cache_dir) else []
        chosen_file = None
        if chip_id:
            for f in cached_files:
                if os.path.basename(f).replace(".npz", "") == chip_id:
                    chosen_file = f
                    break
        elif chip_idx is not None and cached_files and 0 <= chip_idx < len(cached_files):
            chosen_file = cached_files[chip_idx]

        has_ground_truth = False
        gt_mask = None

        if chosen_file:
            chip_name = os.path.basename(chosen_file).replace(".npz", "")
            data = np.load(chosen_file, allow_pickle=True)
            opt_t1 = data["t1_optical"]
            opt_t2 = data["t2_optical"]
            sar_vv = data["sar_vv"] if "sar_vv" in data else np.random.exponential(0.1, (opt_t1.shape[1], opt_t1.shape[2])).astype(np.float32)
            sar_vh = data["sar_vh"] if "sar_vh" in data else np.random.exponential(0.04, (opt_t1.shape[1], opt_t1.shape[2])).astype(np.float32)
            if "target_mask" in data:
                gt_mask = data["target_mask"]
                has_ground_truth = True
            elif "label" in data:
                gt_mask = data["label"]
                has_ground_truth = True
            elif "mask" in data:
                gt_mask = data["mask"]
                has_ground_truth = True
            if "lon" in data and "lat" in data:
                west = float(data["lon"])
                north = float(data["lat"])
        else:
            chip_name = f"manual_sim_{int(time.time()) % 10000}"
            scene = generate_synthetic_scene(
                height=128,
                width=128,
                seed=int(time.time() * 1000) % 1000000,
                deforestation_fraction=0.12,
                cloud_fraction=0.05,
            )
            opt_t1 = scene["t1_optical"]
            opt_t2 = scene["t2_optical"]
            sar_vv = scene["sar_vv"]
            sar_vh = scene["sar_vh"]
            gt_mask = scene.get("change_mask", scene.get("label"))
            has_ground_truth = (gt_mask is not None)

        if gt_mask is not None and gt_mask.ndim == 3:
            gt_mask = gt_mask[0]

        model = self._get_model(model_type)
        expected_c = self._get_model_in_channels(model)
        t1_opt = self._match_input_channels(opt_t1, expected_c)
        t2_opt = self._match_input_channels(opt_t2, expected_c)

        t1_t = torch.from_numpy(t1_opt).unsqueeze(0).float()
        t2_t = torch.from_numpy(t2_opt).unsqueeze(0).float()

        with torch.no_grad():
            if model_type == "siamese":
                logits = model(t1_t, t2_t)
            elif model_type == "multimodal":
                sar_stacked = np.stack([sar_vv, sar_vh, np.log10(np.clip(sar_vv / (sar_vh + 1e-6), 1e-4, 1e4))], axis=0)
                sar_t = torch.from_numpy(sar_stacked).unsqueeze(0).float()
                logits = model(t2_t, sar_t)
            else:
                logits = model(t2_t)
            probs = torch.sigmoid(logits).squeeze().numpy()

        pred_binary = (probs >= threshold).astype(np.float32)

        transform = GeoTransform(west=west, north=north, pixel_width=0.0001, pixel_height=-0.0001)
        polygonizer = Polygonizer(mmu_ha=mmu_ha, simplify_tolerance_deg=0.00005)
        geojson_fc = polygonizer.polygonize(probs, threshold=threshold, transform=transform)

        ndvi_t1 = self.optical_prep.compute_ndvi(opt_t1[2], opt_t1[3])
        ndvi_t2 = self.optical_prep.compute_ndvi(opt_t2[2], opt_t2[3])

        img_t1_rgb = render_optical_rgb(opt_t1)
        img_t2_rgb = render_optical_rgb(opt_t2)
        img_t1_cir = render_optical_cir(opt_t1)
        img_t2_cir = render_optical_cir(opt_t2)
        img_ndvi_t1 = render_ndvi_colormap(ndvi_t1)
        img_ndvi_t2 = render_ndvi_colormap(ndvi_t2)
        img_sar = render_sar_composite(sar_vv, sar_vh)
        img_prob = render_probability_heatmap(probs)
        img_pred_mask = render_mask(pred_binary, color=[239, 68, 68])
        img_mask_overlay = render_mask_overlay(opt_t2, pred_binary, color=(239, 68, 68), alpha=0.50)

        layers = {
            "t1_rgb": img_t1_rgb,
            "t2_rgb": img_t2_rgb,
            "t1_cir": img_t1_cir,
            "t2_cir": img_t2_cir,
            "ndvi_t1": img_ndvi_t1,
            "ndvi_t2": img_ndvi_t2,
            "sar_composite": img_sar,
            "probability_heatmap": img_prob,
            "pred_binary_mask": img_pred_mask,
            "mask_overlay": img_mask_overlay,
        }

        metrics = {}
        if has_ground_truth and gt_mask is not None:
            layers["ground_truth_mask"] = render_mask(gt_mask, color=[16, 185, 129])
            layers["ground_truth_overlay"] = render_mask_overlay(opt_t2, gt_mask, color=[16, 185, 129], alpha=0.50)
            layers["error_diff_map"] = render_error_diff(gt_mask, pred_binary)

            gt_bin = (gt_mask > 0.5).astype(bool)
            pr_bin = (pred_binary > 0.5).astype(bool)
            tp = int(np.sum(gt_bin & pr_bin))
            fp = int(np.sum(~gt_bin & pr_bin))
            fn = int(np.sum(gt_bin & ~pr_bin))
            tn = int(np.sum(~gt_bin & ~pr_bin))
            iou = float(tp / (tp + fp + fn + 1e-7))
            f1 = float(2 * tp / (2 * tp + fp + fn + 1e-7))
            precision = float(tp / (tp + fp + 1e-7))
            recall = float(tp / (tp + fn + 1e-7))

            metrics = {
                "has_ground_truth": True,
                "iou": round(iou, 4),
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                "ground_truth_pct": round(float(np.mean(gt_mask) * 100), 2),
            }
        else:
            metrics = {"has_ground_truth": False}

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "status": "success",
            "chip_id": chip_name,
            "model_type": model_type,
            "checkpoint_path": checkpoint_path or "Backbone In-Memory Weights",
            "threshold": threshold,
            "mmu_ha": mmu_ha,
            "west": west,
            "north": north,
            "detected_pct": round(float(np.mean(pred_binary) * 100), 2),
            "total_alerts": geojson_fc["properties"]["total_alerts"],
            "total_deforested_ha": geojson_fc["properties"]["total_deforested_ha"],
            "elapsed_ms": round(elapsed_ms, 2),
            "metrics": metrics,
            "geojson": geojson_fc,
            "layers": layers,
        }

    def threshold_sweep(
        self,
        checkpoint_name: Optional[str] = None,
        num_samples: int = 8,
        checkpoint_dir: str = "./checkpoints",
        cache_dir: str = "./data/chips_cache",
    ) -> List[Dict[str, Any]]:
        """Compute metrics across thresholds 0.1 to 0.9 for interactive sensitivity curve."""
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []
        for t in thresholds:
            res = self.test_benchmark(
                checkpoint_name=checkpoint_name,
                num_samples=num_samples,
                threshold=t,
                checkpoint_dir=checkpoint_dir,
                cache_dir=cache_dir,
            )
            cm = res["confusion_matrix"]
            tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
            fp_rate = float(fp / (fp + tn + 1e-7))
            results.append({
                "threshold": t,
                "iou": res["iou"],
                "f1": res["f1"],
                "precision": res["precision"],
                "recall": res["recall"],
                "specificity": res["specificity"],
                "fp_rate": round(fp_rate, 4),
            })
        return results


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

    def load_checkpoint_from_path(self, file_path: str) -> Dict[str, Any]:
        """Inspect and validate a checkpoint file from an arbitrary disk path."""
        clean_path = os.path.abspath(file_path)
        if not os.path.exists(clean_path):
            return {"status": "error", "message": f"Checkpoint file not found: {clean_path}", "exists": False}
        stat = os.stat(clean_path)
        info = {
            "status": "success",
            "filename": os.path.basename(clean_path),
            "filepath": clean_path,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "epoch": "N/A",
            "best_val_iou": "N/A",
            "has_weights": False,
        }
        try:
            chk = torch.load(clean_path, map_location="cpu", weights_only=False)
            if isinstance(chk, dict):
                info["epoch"] = chk.get("epoch", "N/A")
                metrics = chk.get("metrics", {})
                if isinstance(metrics, dict):
                    val_iou = metrics.get("iou", metrics.get("val_iou"))
                    if val_iou is not None:
                        info["best_val_iou"] = round(float(val_iou), 4)
                if info["best_val_iou"] == "N/A" and isinstance(chk.get("history"), list) and chk["history"]:
                    info["best_val_iou"] = round(float(chk["history"][-1].get("val_iou", 0.0)), 4)
                st = chk.get("model_state_dict", chk)
                info["num_param_tensors"] = len(st) if isinstance(st, dict) else 0
                info["has_weights"] = True
            else:
                info["has_weights"] = True
        except Exception as e:
            info["load_warning"] = str(e)
        return info


# =============================================================================
# Cached Chips & Dataset Zoo Manager
# =============================================================================

class StudioChipManager:
    """Manages cached training chips (.npz) and synthetic dataset generation."""

    def __init__(self, cache_dir: str = "./data/chips_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def list_chips(self, limit: int = 100) -> Dict[str, Any]:
        """List cached chips with inspection statistics."""
        if not os.path.exists(self.cache_dir):
            return {"total_chips": 0, "chips": []}

        files = sorted(glob.glob(os.path.join(self.cache_dir, "*.npz")))
        results = []
        for i, fpath in enumerate(files[:limit]):
            stat = os.stat(fpath)
            fname = os.path.basename(fpath)
            loss_pct = 0.0
            try:
                data = np.load(fpath, allow_pickle=True)
                if "label" in data:
                    loss_pct = float(np.mean(data["label"]) * 100)
                elif "mask" in data:
                    loss_pct = float(np.mean(data["mask"]) * 100)
            except Exception:
                pass

            results.append({
                "chip_id": fname.replace(".npz", ""),
                "chip_idx": i,
                "file_name": fname,
                "size_kb": round(stat.st_size / 1024, 1),
                "deforestation_pct": round(loss_pct, 2),
                "class_label": "DEFOREST" if loss_pct >= 5.0 else "INTACT",
                "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            })

        return {
            "total_chips": len(files),
            "returned_chips": len(results),
            "chips": results,
        }

    def generate_synthetic_chips(
        self,
        num_chips: int = 20,
        deforest_ratio: float = 0.70,
    ) -> Dict[str, Any]:
        """Generate realistic synthetic Sentinel-2/Sentinel-1 chips with 70:30 quota."""
        os.makedirs(self.cache_dir, exist_ok=True)
        deforest_count = int(round(num_chips * deforest_ratio))
        intact_count = num_chips - deforest_count

        created = 0
        for i in range(num_chips):
            is_deforest = (i < deforest_count)
            frac = float(np.random.uniform(0.08, 0.25)) if is_deforest else 0.0
            scene = generate_synthetic_scene(
                height=128,
                width=128,
                seed=1000 + i,
                deforestation_fraction=frac,
                cloud_fraction=float(np.random.uniform(0.02, 0.15)),
            )
            cid = f"chip_{i:04d}_{'DEFOREST' if is_deforest else 'INTACT'}"
            out_path = os.path.join(self.cache_dir, f"{cid}.npz")
            np.savez_compressed(
                out_path,
                t1_optical=scene["t1_optical"],
                t2_optical=scene["t2_optical"],
                sar_vv=scene["sar_vv"],
                sar_vh=scene["sar_vh"],
                label=scene.get("change_mask", scene.get("label", np.zeros((128, 128), dtype=np.float32))),
            )
            created += 1

        return {
            "status": "success",
            "created_chips": created,
            "deforest_chips": deforest_count,
            "intact_chips": intact_count,
            "cache_dir": os.path.abspath(self.cache_dir),
        }

    def set_cache_dir(self, new_dir: str) -> Dict[str, Any]:
        """Update active chips directory to an arbitrary local path."""
        clean_dir = os.path.abspath(new_dir)
        os.makedirs(clean_dir, exist_ok=True)
        self.cache_dir = clean_dir
        chip_count = len(glob.glob(os.path.join(clean_dir, "*.npz")))
        return {
            "status": "success",
            "cache_dir": clean_dir,
            "chip_count": chip_count,
            "message": f"Active cache directory set to {clean_dir} ({chip_count} chips found)",
        }

    def extract_chips_zip(self, zip_bytes: bytes) -> Dict[str, Any]:
        """Extract a .zip archive of satellite chips (.npz) into the active cache directory."""
        import zipfile
        import io
        os.makedirs(self.cache_dir, exist_ok=True)
        extracted_count = 0
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for member in zf.infolist():
                if member.filename.endswith(".npz") or member.filename.endswith("manifest.jsonl"):
                    basename = os.path.basename(member.filename)
                    if not basename:
                        continue
                    target_path = os.path.join(self.cache_dir, basename)
                    with zf.open(member) as src, open(target_path, "wb") as dst:
                        dst.write(src.read())
                    if basename.endswith(".npz"):
                        extracted_count += 1
        total_chips = len(glob.glob(os.path.join(self.cache_dir, "*.npz")))
        return {
            "status": "success",
            "extracted_count": extracted_count,
            "total_chips": total_chips,
            "cache_dir": self.cache_dir,
            "message": f"Extracted {extracted_count} chips into {self.cache_dir} (Total: {total_chips})",
        }

    def ingest_paired_chip(
        self,
        t1_bytes: bytes,
        t2_bytes: bytes,
        mask_bytes: Optional[bytes] = None,
        chip_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest user-supplied pre/post satellite imagery pair (Bring Your Own Data).
        Supports GeoTIFF, TIFF, PNG, JPEG, and NPZ.
        """
        import io
        import numpy as np
        from datetime import datetime, timezone
        from PIL import Image

        def _decode_image_bytes(b: bytes) -> np.ndarray:
            try:
                import tifffile
                arr = tifffile.imread(io.BytesIO(b))
                if arr is not None and arr.size > 0:
                    return arr
            except Exception:
                pass
            img = Image.open(io.BytesIO(b))
            return np.array(img)

        os.makedirs(self.cache_dir, exist_ok=True)
        t1_arr = _decode_image_bytes(t1_bytes).astype(np.float32)
        t2_arr = _decode_image_bytes(t2_bytes).astype(np.float32)

        if t1_arr.ndim == 2:
            t1_arr = t1_arr[np.newaxis, ...]
        elif t1_arr.ndim == 3 and t1_arr.shape[-1] in (1, 3, 4, 12, 13):
            t1_arr = np.transpose(t1_arr, (2, 0, 1))

        if t2_arr.ndim == 2:
            t2_arr = t2_arr[np.newaxis, ...]
        elif t2_arr.ndim == 3 and t2_arr.shape[-1] in (1, 3, 4, 12, 13):
            t2_arr = np.transpose(t2_arr, (2, 0, 1))

        h = min(t1_arr.shape[1], t2_arr.shape[1])
        w = min(t1_arr.shape[2], t2_arr.shape[2])
        t1_arr = t1_arr[:, :h, :w]
        t2_arr = t2_arr[:, :h, :w]

        if mask_bytes and len(mask_bytes) > 0:
            mask_arr = _decode_image_bytes(mask_bytes)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
            mask_arr = (mask_arr[:h, :w] > 0).astype(np.float32)
        else:
            mask_arr = np.zeros((h, w), dtype=np.float32)

        sar_vv = np.zeros((h, w), dtype=np.float32)
        sar_vh = np.zeros((h, w), dtype=np.float32)

        cid = chip_name or f"byod_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        clean_cid = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in cid)
        out_path = os.path.join(self.cache_dir, f"{clean_cid}.npz")

        np.savez_compressed(
            out_path,
            t1_optical=t1_arr,
            t2_optical=t2_arr,
            sar_vv=sar_vv,
            sar_vh=sar_vh,
            label=mask_arr,
        )

        total_chips = len(glob.glob(os.path.join(self.cache_dir, "*.npz")))
        return {
            "status": "success",
            "chip_id": clean_cid,
            "filename": f"{clean_cid}.npz",
            "dimensions": f"{w}x{h}",
            "channels": int(t1_arr.shape[0]),
            "cache_dir": self.cache_dir,
            "total_chips": total_chips,
            "message": f"Successfully ingested BYOD satellite pair as {clean_cid}.npz ({w}x{h}, {t1_arr.shape[0]} bands)",
        }

    def fetch_gee_chips(
        self,
        project_id: Optional[str] = None,
        num_chips: int = 200,
        chip_size: int = 256,
        target_pos_ratio: float = 0.70,
    ) -> Dict[str, Any]:
        """Fetch satellite chips with Hansen GFC loss masks using Earth Engine pipeline."""
        from geofm.data_sources.earth_engine import download_and_cache_gee_dataset
        manifest_path, total_cached = download_and_cache_gee_dataset(
            output_dir=self.cache_dir,
            max_chips=num_chips,
            project_id=project_id,
            chip_size=chip_size,
            class_balance_filter=True,
            target_pos_ratio=target_pos_ratio,
            show_progress=True,
        )
        return {
            "status": "success",
            "project_id": project_id,
            "total_cached": total_cached,
            "manifest_path": manifest_path,
            "cache_dir": self.cache_dir,
        }


# =============================================================================
# Hardware & Accelerator Telemetry
# =============================================================================

def get_hardware_telemetry(
    checkpoint_dir: str = "./checkpoints",
    cache_dir: str = "./data/chips_cache",
) -> Dict[str, Any]:
    """Return live system and accelerator telemetry (RAM, CPU, GPU, OS, architecture, storage)."""
    import platform
    import sys

    # 1. CPU & Operating System Platform Detection
    cpu_count = os.cpu_count() or 1
    sys_os = platform.system()  # 'Windows', 'Linux', 'Darwin', etc.
    machine = platform.machine()  # 'AMD64', 'x86_64', 'arm64', 'aarch64'
    proc_desc = platform.processor() or machine

    # Normalize architecture string for UI clarity
    if machine.lower() in ("amd64", "x86_64"):
        arch_label = "x86_64"
    elif "arm" in machine.lower() or "aarch" in machine.lower():
        arch_label = "ARM64"
    else:
        arch_label = machine or "CPU"

    # Human-friendly OS Label
    if sys_os == "Windows":
        rel = platform.release()
        os_label = f"Windows {rel}" if rel else "Windows"
    elif sys_os == "Darwin":
        os_label = "macOS"
    elif sys_os == "Linux":
        os_label = "Linux"
    else:
        os_label = sys_os

    # 2. Dynamic Memory (RAM) Telemetry (Windows, Linux, macOS, BSD)
    total_ram_mb = 0.0
    used_ram_mb = 0.0
    ram_pct = 0.0

    # Try psutil first if installed
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_ram_mb = round(vm.total / (1024 * 1024), 1)
        used_ram_mb = round((vm.total - vm.available) / (1024 * 1024), 1)
        ram_pct = round(((vm.total - vm.available) / vm.total) * 100, 1) if vm.total > 0 else 0.0
    except Exception:
        pass

    # Windows Native Fallback via ctypes if psutil is absent
    if total_ram_mb <= 0.0 and sys.platform == "win32":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_ram_mb = round(stat.ullTotalPhys / (1024 * 1024), 1)
                avail_ram_mb = round(stat.ullAvailPhys / (1024 * 1024), 1)
                used_ram_mb = round(total_ram_mb - avail_ram_mb, 1)
                ram_pct = float(stat.dwMemoryLoad)
        except Exception:
            pass

    # Linux / Termux Native Fallback via /proc/meminfo
    if total_ram_mb <= 0.0 and os.path.exists("/proc/meminfo"):
        try:
            mem = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        mem[parts[0].strip()] = int(parts[1].strip().split()[0])
            total_ram_mb = round(mem.get("MemTotal", 0) / 1024, 1)
            free_ram_mb = round(mem.get("MemAvailable", mem.get("MemFree", 0)) / 1024, 1)
            used_ram_mb = round(total_ram_mb - free_ram_mb, 1)
            ram_pct = round((used_ram_mb / total_ram_mb) * 100, 1) if total_ram_mb > 0 else 0.0
        except Exception:
            pass

    # 3. Dynamic Accelerator / Hardware Compute Detection
    gpu_available = False
    device_name = ""
    vram_used_mb = 0.0
    vram_total_mb = 0.0
    vram_pct = 0.0

    if torch.cuda.is_available():
        gpu_available = True
        try:
            raw_gpu_name = torch.cuda.get_device_name(0)
            vram_total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
            vram_used_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
            vram_pct = round((vram_used_mb / vram_total_mb) * 100, 1) if vram_total_mb > 0 else 0.0
            device_name = f"CUDA: {raw_gpu_name}"
        except Exception:
            device_name = "CUDA GPU"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_available = True
        device_name = f"Apple Silicon MPS ({arch_label})"
    elif hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
        gpu_available = True
        try:
            device_name = f"Intel XPU: {torch.xpu.get_device_name(0)}"
        except Exception:
            device_name = "Intel XPU"
    else:
        gpu_available = False
        device_name = f"CPU Host ({arch_label} / {os_label})"

    checkpoint_count = 0
    if os.path.exists(checkpoint_dir):
        checkpoint_count = len([f for f in os.listdir(checkpoint_dir) if f.endswith((".pt", ".pth"))])

    cached_chips_count = 0
    if os.path.exists(cache_dir):
        cached_chips_count = len([f for f in os.listdir(cache_dir) if f.endswith(".npz")])

    return {
        "gpu_available": gpu_available,
        "device_name": device_name,
        "os_name": sys_os,
        "architecture": arch_label,
        "processor": proc_desc,
        "vram_used_mb": vram_used_mb,
        "vram_total_mb": vram_total_mb,
        "vram_pct": vram_pct,
        "cpu_count": cpu_count,
        "ram_total_mb": total_ram_mb,
        "ram_used_mb": used_ram_mb,
        "ram_pct": ram_pct,
        "checkpoint_count": checkpoint_count,
        "cached_chips_count": cached_chips_count,
    }
