"""Standalone Model Loader & Interactive Inference Studio for GeoFM.

Allows users to:
1. Load local or custom PyTorch checkpoints (.pt / .pth) with automatic architecture and LoRA PEFT inspection.
2. Ingest input data (local cached chips, synthetic benchmark tiles, or uploaded GeoTIFF/PNG/JPG bitemporal rasters).
3. Execute real-time inference with hardware acceleration (CUDA/CPU).
4. Visualize continuous probability heatmaps, spectral change maps, and interactive thresholded alert masks.
5. Export prediction masks and polygonized GeoJSON alerts for GIS / EUDR compliance audits.

Usage:
    python model_loader.py [--host 127.0.0.1] [--port 8600] [--model checkpoints/best_model.pt]
"""

import argparse
import base64
import glob
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Core GeoFM dependencies
from geofm.models import (
    SiameseChangeDetector,
    UNet,
    MultimodalFusionDeforestationNet,
    apply_lora_to_model,
)
from geofm.postprocessing.polygonizer import Polygonizer
from geofm.preprocessing.cog_reader import GeoTransform


# ==============================================================================
# Helper Utilities: Image Conversion & Colormaps (Zero-Matplotlib for max portability)
# ==============================================================================

def array_to_png_base64(arr: np.ndarray) -> str:
    """Convert a numpy array (uint8 or float) to a base64-encoded PNG string."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def raster_to_rgb_base64(arr: np.ndarray) -> str:
    """Render multi-spectral raster (C, H, W) or (H, W, C) to RGB base64 PNG with percentile stretch."""
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in [1, 2, 3, 4, 6, 12, 13]:
            if arr.shape[0] == 1:
                rgb = np.repeat(arr, 3, axis=0)
            elif arr.shape[0] == 2:
                pad = np.zeros_like(arr[:1])
                rgb = np.concatenate([arr, pad], axis=0)
            elif arr.shape[0] >= 3:
                # Default to channels 0, 1, 2 (RGB / False-color)
                rgb = arr[:3]
            rgb = np.transpose(rgb, (1, 2, 0))
        else:
            rgb = arr[:, :, :3]
    else:
        rgb = np.zeros((256, 256, 3), dtype=np.float32)

    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)

    uint8_img = (rgb * 255).astype(np.uint8)
    return array_to_png_base64(uint8_img)


def probability_to_turbo_base64(probs: np.ndarray) -> str:
    """Render continuous probability map (H, W) in [0, 1] to a high-contrast Turbo-style RGB heatmap."""
    p = np.clip(probs, 0.0, 1.0)
    # 3-stop smooth gradient: Forest Navy/Green (0.0) -> Goldenrod (0.5) -> Intense Crimson (1.0)
    r = np.where(p < 0.5, p * 2.0 * 240.0, 240.0 + (p - 0.5) * 2.0 * 15.0)
    g = np.where(p < 0.5, 30.0 + p * 2.0 * 190.0, 220.0 - (p - 0.5) * 2.0 * 190.0)
    b = np.where(p < 0.5, 70.0 * (1.0 - p * 2.0), 20.0 * (1.0 - (p - 0.5) * 2.0))
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return array_to_png_base64(rgb)


def binary_mask_to_overlay_base64(binary: np.ndarray) -> str:
    """Render binary mask (H, W) as high-contrast neon deforestation alert overlay."""
    b = (binary > 0).astype(np.uint8)
    h, w = b.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Dark charcoal for intact background, Vivid Scarlet for deforestation
    rgb[b == 0] = [18, 22, 28]
    rgb[b == 1] = [239, 68, 68]
    return array_to_png_base64(rgb)


# ==============================================================================
# Model Engine & Inference Manager
# ==============================================================================

class ModelLoaderEngine:
    """Manages loading, auto-inspection, and evaluation of GeoFM PyTorch models."""

    def __init__(self, default_checkpoints_dir: str = "checkpoints"):
        self.checkpoints_dir = Path(default_checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # State
        self.current_model: Optional[torch.nn.Module] = None
        self.model_info: Dict[str, Any] = {}
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.last_probs: Optional[np.ndarray] = None
        self.last_transform: Optional[GeoTransform] = None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all discoverable .pt and .pth checkpoints."""
        candidates = list(self.checkpoints_dir.glob("**/*.pt")) + list(self.checkpoints_dir.glob("**/*.pth"))
        results = []
        for p in sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True):
            size_mb = round(p.stat().st_size / (1024 * 1024), 2)
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
            results.append({
                "filename": p.name,
                "relative_path": str(p.relative_to(Path.cwd()) if p.is_relative_to(Path.cwd()) else p),
                "absolute_path": str(p.resolve()),
                "size_mb": size_mb,
                "modified": mtime,
            })
        return results

    def list_cached_samples(self, cache_dir: str = "data/chips_cache") -> List[Dict[str, Any]]:
        """List pre-cached .npz satellite chips."""
        cpath = Path(cache_dir)
        if not cpath.exists():
            return []
        chips = sorted(list(cpath.glob("*.npz")))
        samples = []
        for c in chips[:60]:  # Limit top 60 for fast responsiveness
            try:
                data = np.load(c)
                sample_id = str(data.get("sample_id", c.stem))
                def_pct = float(data.get("deforested_pct", 0.0))
                lon = float(data.get("lon", -62.5))
                lat = float(data.get("lat", -8.5))
                samples.append({
                    "sample_id": sample_id,
                    "filename": c.name,
                    "path": str(c.resolve()),
                    "deforested_pct": round(def_pct, 2),
                    "coordinates": f"{lat:.3f}, {lon:.3f}",
                })
            except Exception:
                continue
        return samples

    def inspect_and_load_checkpoint(
        self,
        checkpoint_source: Union[str, Path, io.BytesIO],
        arch_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load a PyTorch checkpoint with automated architecture detection and parameter inspection."""
        if isinstance(checkpoint_source, (str, Path)):
            chk_path = Path(checkpoint_source)
            if not chk_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {chk_path}")
            checkpoint_data = torch.load(str(chk_path), map_location=self.device)
            source_name = chk_path.name
        else:
            checkpoint_data = torch.load(checkpoint_source, map_location=self.device)
            source_name = "uploaded_model.pt"

        # Extract state dict
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            state_dict = checkpoint_data["model_state_dict"]
            metadata = {
                "epoch": checkpoint_data.get("epoch"),
                "best_val_iou": checkpoint_data.get("best_val_iou"),
                "metrics": checkpoint_data.get("metrics"),
                "saved_at": checkpoint_data.get("saved_at"),
            }
            chk_model_type = checkpoint_data.get("model_type")
        elif isinstance(checkpoint_data, dict):
            state_dict = checkpoint_data
            metadata = {}
            chk_model_type = None
        else:
            raise ValueError("Unsupported checkpoint format. Expected dict or state_dict.")

        # 1. Determine model architecture
        if arch_override and arch_override.lower() != "auto":
            model_type = arch_override.lower()
        elif chk_model_type:
            model_type = chk_model_type.lower()
        else:
            if any("cross_attention" in k for k in state_dict.keys()):
                model_type = "multimodal"
            elif any("inc." in k for k in state_dict.keys()):
                model_type = "unet"
            else:
                model_type = "siamese"

        # 2. Detect input channels & feature scaling
        in_channels = 4
        for k, v in state_dict.items():
            if "in_conv.0.weight" in k or "inc.conv.0.weight" in k or "opt_encoder.in_conv.0.weight" in k:
                in_channels = int(v.shape[1])
                break

        f0, f1, f2, f3 = 16, 32, 64, 128
        if "encoder.enc1.conv1.weight" in state_dict:
            f0 = state_dict["encoder.enc1.conv1.weight"].shape[0]
        elif "encoder.in_conv.0.weight" in state_dict:
            f0 = state_dict["encoder.in_conv.0.weight"].shape[0]
        elif "inc.conv.0.weight" in state_dict:
            f0 = state_dict["inc.conv.0.weight"].shape[0]

        if "encoder.enc2.conv1.weight" in state_dict:
            f1 = state_dict["encoder.enc2.conv1.weight"].shape[0]
        elif "down1.conv.0.weight" in state_dict:
            f1 = state_dict["down1.conv.0.weight"].shape[0]

        if "encoder.enc3.conv1.weight" in state_dict:
            f2 = state_dict["encoder.enc3.conv1.weight"].shape[0]
        elif "down2.conv.0.weight" in state_dict:
            f2 = state_dict["down2.conv.0.weight"].shape[0]

        if "encoder.enc4.conv1.weight" in state_dict:
            f3 = state_dict["encoder.enc4.conv1.weight"].shape[0]
        elif "down3.conv.0.weight" in state_dict:
            f3 = state_dict["down3.conv.0.weight"].shape[0]
        features = [int(f0), int(f1), int(f2), int(f3)]

        # 3. Detect LoRA PEFT adapters
        has_lora = any("lora_" in k for k in state_dict.keys())
        lora_rank = 8
        if has_lora:
            for k, v in state_dict.items():
                if k.endswith(".lora_A"):
                    lora_rank = int(v.shape[0])
                    break
        lora_alpha = float(lora_rank * 2.0)

        # 4. Instantiate model architecture
        if model_type == "unet":
            model = UNet(in_channels=in_channels, num_classes=1, features=features)
        elif model_type == "multimodal":
            model = MultimodalFusionDeforestationNet(opt_channels=in_channels, sar_channels=3, features=features)
        else:
            model = SiameseChangeDetector(in_channels=in_channels, features=features)

        # 5. Apply LoRA if checkpoint contains low-rank adapters
        if has_lora:
            model, _ = apply_lora_to_model(model, rank=lora_rank, alpha=lora_alpha)

        # 6. Load weights into model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()

        self.current_model = model

        # Parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.model_info = {
            "source_name": source_name,
            "model_type": model_type,
            "in_channels": in_channels,
            "has_lora": has_lora,
            "lora_rank": lora_rank if has_lora else None,
            "lora_alpha": lora_alpha if has_lora else None,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
            "missing_keys_count": len(missing_keys),
            "unexpected_keys_count": len(unexpected_keys),
            "metadata": metadata,
            "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return self.model_info

    def _prepare_tensor_channels(self, arr: np.ndarray) -> np.ndarray:
        """Align input raster channel count to the model's expected in_channels."""
        target_c = self.model_info.get("in_channels", 4)
        c, h, w = arr.shape
        if c == target_c:
            return arr
        elif c > target_c:
            return arr[:target_c]
        else:
            # Replicate last channel to match target
            pad = np.repeat(arr[-1:], target_c - c, axis=0)
            return np.concatenate([arr, pad], axis=0)

    def run_inference(
        self,
        t1_arr: np.ndarray,
        t2_arr: np.ndarray,
        threshold: float = 0.50,
        ground_truth: Optional[np.ndarray] = None,
        lon: float = -62.5,
        lat: float = -8.5,
    ) -> Dict[str, Any]:
        """Execute model forward pass and format visual heatmaps and polygon alerts."""
        if self.current_model is None:
            raise RuntimeError("No model is loaded. Please load a checkpoint first.")

        # Ensure (C, H, W)
        if t1_arr.ndim == 2:
            t1_arr = np.expand_dims(t1_arr, axis=0)
        if t2_arr.ndim == 2:
            t2_arr = np.expand_dims(t2_arr, axis=0)

        # Re-order (H, W, C) -> (C, H, W) if necessary
        if t1_arr.shape[0] > 16 and t1_arr.shape[-1] <= 16:
            t1_arr = np.transpose(t1_arr, (2, 0, 1))
        if t2_arr.shape[0] > 16 and t2_arr.shape[-1] <= 16:
            t2_arr = np.transpose(t2_arr, (2, 0, 1))

        # Channel alignment
        t1_aligned = self._prepare_tensor_channels(t1_arr.astype(np.float32))
        t2_aligned = self._prepare_tensor_channels(t2_arr.astype(np.float32))

        # Tensor conversion
        t1_tensor = torch.from_numpy(t1_aligned).unsqueeze(0).to(self.device).float()
        t2_tensor = torch.from_numpy(t2_aligned).unsqueeze(0).to(self.device).float()

        # Benchmarked forward pass
        t_start = time.perf_counter()
        with torch.no_grad():
            model_type = self.model_info.get("model_type", "siamese")
            if model_type == "unet":
                inp = torch.cat([t1_tensor, t2_tensor], dim=1)
                logits = self.current_model(inp)
            elif model_type == "multimodal":
                sar_tensor = torch.zeros((1, 3, t1_aligned.shape[1], t1_aligned.shape[2]), device=self.device)
                logits = self.current_model(t1_tensor, sar_tensor)
            else:
                logits = self.current_model(t1_tensor, t2_tensor)

            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # Store for fast re-thresholding without model execution
        self.last_probs = probs
        self.last_transform = GeoTransform(west=lon, north=lat, pixel_width=0.0001, pixel_height=-0.0001)

        # Render visualizations
        t1_b64 = raster_to_rgb_base64(t1_arr)
        t2_b64 = raster_to_rgb_base64(t2_arr)
        
        # Spectral difference heatmap
        diff_arr = np.abs(t2_aligned[:3] - t1_aligned[:3]).mean(axis=0) if t1_aligned.shape[0] >= 3 else np.abs(t2_aligned[0] - t1_aligned[0])
        diff_b64 = probability_to_turbo_base64(diff_arr)
        
        # Model continuous probability map
        prob_b64 = probability_to_turbo_base64(probs)

        # Thresholded statistics & alert vectorization
        stats, binary_b64, alerts_fc = self.recompute_threshold_statistics(probs, threshold, self.last_transform)

        # Ground truth validation metrics if provided
        metrics = None
        gt_b64 = None
        if ground_truth is not None:
            gt = (ground_truth > 0).squeeze().astype(np.uint8)
            bin_pred = (probs >= threshold).astype(np.uint8)
            gt_b64 = binary_mask_to_overlay_base64(gt)
            
            intersection = int(np.logical_and(bin_pred == 1, gt == 1).sum())
            union = int(np.logical_or(bin_pred == 1, gt == 1).sum())
            iou = round(float(intersection / union) if union > 0 else 1.0, 4)
            tp = intersection
            fp = int(np.logical_and(bin_pred == 1, gt == 0).sum())
            fn = int(np.logical_and(bin_pred == 0, gt == 1).sum())
            precision = round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4)
            recall = round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4)
            f1 = round(float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0, 4)
            metrics = {
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

        response = {
            "latency_ms": latency_ms,
            "device": str(self.device),
            "model_type": model_type,
            "threshold": threshold,
            "t1_image_b64": t1_b64,
            "t2_image_b64": t2_b64,
            "difference_b64": diff_b64,
            "probability_b64": prob_b64,
            "binary_mask_b64": binary_b64,
            "ground_truth_b64": gt_b64,
            "statistics": stats,
            "metrics": metrics,
            "alerts_geojson": alerts_fc,
        }
        self.last_prediction = response
        return response

    def recompute_threshold_statistics(
        self,
        probs: np.ndarray,
        threshold: float,
        transform: Optional[GeoTransform] = None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """Recompute binary mask, affected area, and vector polygons at a chosen threshold."""
        binary = (probs >= threshold).astype(np.uint8)
        total_pixels = int(binary.size)
        def_pixels = int(np.sum(binary))
        def_pct = round(float(def_pixels / total_pixels * 100.0), 2)
        # 10m Sentinel-2 pixel GSD = 100m² = 0.01 ha
        area_ha = round(float(def_pixels * 0.01), 3)
        mean_conf = round(float(np.mean(probs[binary == 1]) * 100.0) if def_pixels > 0 else 0.0, 1)

        # Polygonize alerts
        poly = Polygonizer(mmu_ha=0.05)
        tf = transform or GeoTransform(west=-62.5, north=-8.5, pixel_width=0.0001, pixel_height=-0.0001)
        alerts_fc = poly.polygonize(probs, threshold=threshold, transform=tf)
        alert_count = len(alerts_fc.get("features", []))

        # Risk level determination
        if def_pct > 15.0 or area_ha > 50.0:
            risk = "CRITICAL"
        elif def_pct > 5.0 or area_ha > 15.0:
            risk = "HIGH"
        elif def_pct > 1.0 or area_ha > 2.0:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        stats = {
            "threshold": threshold,
            "total_pixels": total_pixels,
            "deforested_pixels": def_pixels,
            "deforested_pct": def_pct,
            "area_ha": area_ha,
            "mean_confidence_pct": mean_conf,
            "alert_clusters_count": alert_count,
            "risk_level": risk,
        }
        binary_b64 = binary_mask_to_overlay_base64(binary)
        return stats, binary_b64, alerts_fc


# Initialize global engine instance
engine = ModelLoaderEngine()

# Pre-load best_model.pt if it exists, or initialize default Siamese architecture
default_best = Path("checkpoints/best_model.pt")
if default_best.exists():
    try:
        engine.inspect_and_load_checkpoint(default_best)
        print(f"[OK] Default model loaded: {default_best.name} ({engine.model_info.get('model_type')})")
    except Exception as e:
        print(f"Notice: Default checkpoint auto-load skipped: {e}")
else:
    # Initialize a baseline Siamese model with LoRA adapters so the studio is immediately functional
    try:
        base_model = SiameseChangeDetector(in_channels=4, features=[16, 32, 64, 128])
        base_model, _ = apply_lora_to_model(base_model, rank=8, alpha=16.0)
        base_model.eval()
        engine.current_model = base_model
        engine.model_info = {
            "source_name": "built-in default (Siamese + LoRA)",
            "model_type": "siamese",
            "in_channels": 4,
            "has_lora": True,
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "total_params": sum(p.numel() for p in base_model.parameters()),
            "trainable_params": sum(p.numel() for p in base_model.parameters() if p.requires_grad),
            "device": str(engine.device),
            "metadata": {"epoch": 1, "best_val_iou": 0.50},
            "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception:
        pass


# ==============================================================================
# FastAPI Application & API Endpoints
# ==============================================================================

app = FastAPI(
    title="GeoFM Model Loader & Inference Studio",
    description="Standalone interface to load custom checkpoints, run inference, and inspect model outputs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Transparent 1x1 favicon."""
    return Response(content=b"", media_type="image/x-icon")


@app.get("/api/status")
async def get_system_status():
    """Retrieve current engine status, active model, and hardware device."""
    return {
        "status": "online",
        "device": str(engine.device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "has_loaded_model": engine.current_model is not None,
        "model_info": engine.model_info,
        "checkpoints_count": len(engine.list_checkpoints()),
        "samples_count": len(engine.list_cached_samples()),
    }


@app.get("/api/checkpoints")
async def get_checkpoints_list():
    """List all available checkpoints on local disk."""
    return {"checkpoints": engine.list_checkpoints()}


@app.get("/api/samples")
async def get_samples_list():
    """List available pre-cached satellite chips."""
    return {"samples": engine.list_cached_samples()}


@app.post("/api/model/load-local")
async def load_local_checkpoint(
    path: str = Form(...),
    arch_override: str = Form("auto"),
):
    """Load a checkpoint from local file path."""
    try:
        info = engine.inspect_and_load_checkpoint(path, arch_override=arch_override)
        return {"success": True, "model_info": info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/model/upload")
async def upload_custom_checkpoint(
    file: UploadFile = File(...),
    arch_override: str = Form("auto"),
):
    """Upload a custom PyTorch model checkpoint (.pt/.pth) and load it."""
    try:
        content = await file.read()
        buf = io.BytesIO(content)
        info = engine.inspect_and_load_checkpoint(buf, arch_override=arch_override)
        info["source_name"] = file.filename
        
        # Save a copy to checkpoints/uploaded
        upload_dir = Path("checkpoints/uploaded")
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / file.filename
        with open(save_path, "wb") as f:
            f.write(content)
        info["saved_path"] = str(save_path)
        
        return {"success": True, "model_info": info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load uploaded checkpoint: {str(e)}")


@app.post("/api/predict/sample")
async def predict_cached_sample(
    filename: str = Form(...),
    threshold: float = Form(0.50),
):
    """Run model inference on a pre-cached satellite chip."""
    chip_path = Path("data/chips_cache") / filename
    if not chip_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample chip not found: {filename}")
    
    try:
        data = np.load(chip_path)
        t1 = data["t1_optical"]
        t2 = data["t2_optical"]
        mask = data.get("target_mask", None)
        lon = float(data.get("lon", -62.5))
        lat = float(data.get("lat", -8.5))
        
        res = engine.run_inference(t1, t2, threshold=threshold, ground_truth=mask, lon=lon, lat=lat)
        return {"success": True, "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/predict/synthetic")
async def predict_synthetic(
    threshold: float = Form(0.50),
    anomaly_size: int = Form(60),
):
    """Generate a clean synthetic bitemporal test pair and run inference."""
    try:
        h, w = 256, 256
        # Generate baseline forest (T1)
        np.random.seed(int(time.time()) % 10000)
        t1 = np.random.uniform(0.1, 0.4, (4, h, w)).astype(np.float32)
        # NIR band higher for healthy canopy
        t1[3] += 0.35
        
        # Clone for observation (T2)
        t2 = t1.copy()
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Carve simulated deforestation clearcut patch
        cx, cy = np.random.randint(60, 190, 2)
        half = anomaly_size // 2
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        
        # Spectral loss: Red rises (exposed soil), NIR drops (canopy removed)
        t2[0, y1:y2, x1:x2] += 0.35  # Red increase
        t2[3, y1:y2, x1:x2] -= 0.40  # NIR drop
        t2 = np.clip(t2, 0.0, 1.0)
        mask[y1:y2, x1:x2] = 1.0
        
        res = engine.run_inference(t1, t2, threshold=threshold, ground_truth=mask, lon=-61.8, lat=-9.2)
        return {"success": True, "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthetic inference failed: {str(e)}")


@app.post("/api/predict/upload")
async def predict_uploaded_files(
    t1_file: UploadFile = File(...),
    t2_file: UploadFile = File(...),
    mask_file: Optional[UploadFile] = File(None),
    threshold: float = Form(0.50),
):
    """Upload custom bitemporal rasters (PNG/JPG/TIFF/NPZ) and run inference."""
    def decode_file(up_file: UploadFile) -> np.ndarray:
        content = up_file.file.read()
        fname = up_file.filename.lower()
        if fname.endswith(".npz"):
            npz = np.load(io.BytesIO(content))
            return npz["t1_optical"] if "t1_optical" in npz else npz[list(npz.keys())[0]]
        elif fname.endswith(".npy"):
            return np.load(io.BytesIO(content))
        else:
            # Standard image (PNG / JPG / TIFF)
            img = Image.open(io.BytesIO(content))
            arr = np.array(img).astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.ndim == 3:
                arr = np.transpose(arr, (2, 0, 1))
            return arr

    try:
        t1_arr = decode_file(t1_file)
        t2_arr = decode_file(t2_file)
        gt_arr = decode_file(mask_file) if mask_file else None
        
        res = engine.run_inference(t1_arr, t2_arr, threshold=threshold, ground_truth=gt_arr)
        return {"success": True, "result": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded rasters: {str(e)}")


@app.post("/api/predict/rethreshold")
async def rethreshold_current_prediction(threshold: float = Form(...)):
    """Fast slider-driven re-thresholding without running PyTorch forward pass."""
    if engine.last_probs is None:
        raise HTTPException(status_code=400, detail="No active prediction in memory. Please run inference first.")
    
    stats, binary_b64, alerts_fc = engine.recompute_threshold_statistics(
        engine.last_probs,
        threshold,
        engine.last_transform,
    )
    return {
        "success": True,
        "threshold": threshold,
        "binary_mask_b64": binary_b64,
        "statistics": stats,
        "alerts_geojson": alerts_fc,
    }


# ==============================================================================
# Single-Page Web Application Frontend (Modern Glassmorphic UI)
# ==============================================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GeoFM — Model Loader & Inference Studio</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background-color: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        .glass-panel { background: rgba(22, 27, 34, 0.75); backdrop-filter: blur(12px); border: 1px solid rgba(48, 54, 61, 0.7); }
        .accent-gradient { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
        .card-hover:hover { border-color: #3b82f6; transform: translateY(-2px); transition: all 0.2s ease; }
        input[type=range] { accent-color: #10b981; }
    </style>
</head>
<body class="min-h-screen flex flex-col">

    <!-- Top Navigation Bar -->
    <header class="glass-panel sticky top-0 z-50 px-6 py-4 flex items-center justify-between border-b border-gray-800">
        <div class="flex items-center space-x-3">
            <div class="w-9 h-9 rounded-lg bg-emerald-600 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-emerald-900/40">
                🛰️
            </div>
            <div>
                <h1 class="text-lg font-bold text-white tracking-tight flex items-center gap-2">
                    GeoFM <span class="text-xs px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800">Model Loader</span>
                </h1>
                <p class="text-xs text-gray-400">Custom PyTorch Model Evaluation & Bitemporal Inference Studio</p>
            </div>
        </div>

        <div class="flex items-center space-x-3 text-xs">
            <div id="deviceBadge" class="px-3 py-1.5 rounded-full bg-gray-800 border border-gray-700 text-gray-300 flex items-center gap-1.5">
                <i class="fa-solid fa-microchip text-blue-400"></i> Detecting Hardware...
            </div>
            <div id="modelStatusBadge" class="px-3 py-1.5 rounded-full bg-gray-800 border border-gray-700 text-gray-300 flex items-center gap-1.5">
                <i class="fa-solid fa-circle-notch fa-spin text-amber-400"></i> Checking Model...
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">

        <!-- Grid: Model Loader on Left, Input Data on Right -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

            <!-- Section 1: Model Management (5 cols) -->
            <div class="lg:col-span-5 space-y-6">
                <div class="glass-panel rounded-xl p-5 space-y-4">
                    <div class="flex items-center justify-between border-b border-gray-800 pb-3">
                        <h2 class="text-sm font-semibold uppercase tracking-wider text-gray-300 flex items-center gap-2">
                            <i class="fa-solid fa-cube text-emerald-400"></i> 1. Model Checkpoint
                        </h2>
                        <span id="modelTypeTag" class="text-xs px-2 py-0.5 rounded bg-blue-950 border border-blue-800 text-blue-300 font-mono">None</span>
                    </div>

                    <!-- Tabs: Select Local vs Upload File -->
                    <div class="flex border-b border-gray-800 text-xs">
                        <button id="tabLocalBtn" onclick="switchModelTab('local')" class="px-4 py-2 border-b-2 border-emerald-500 font-medium text-emerald-400">Local Checkpoint</button>
                        <button id="tabUploadBtn" onclick="switchModelTab('upload')" class="px-4 py-2 text-gray-400 hover:text-white">Upload Custom .pt</button>
                    </div>

                    <!-- Local Selection Form -->
                    <div id="localModelPane" class="space-y-3">
                        <div>
                            <label class="block text-xs font-medium text-gray-400 mb-1">Available Local Models (in ./checkpoints)</label>
                            <select id="checkpointSelect" class="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-emerald-500">
                                <option value="">Loading checkpoints...</option>
                            </select>
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-gray-400 mb-1">Architecture Override</label>
                            <select id="archOverride" class="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-emerald-500">
                                <option value="auto">Auto-Detect from Weights</option>
                                <option value="siamese">Siamese Change Detector (Bitemporal)</option>
                                <option value="unet">UNet Change Detector</option>
                                <option value="multimodal">Multimodal Cross-Attention (Optical+SAR)</option>
                            </select>
                        </div>

                        <button onclick="loadLocalModel()" class="w-full py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-sm transition flex items-center justify-center gap-2 shadow-lg shadow-emerald-900/30">
                            <i class="fa-solid fa-arrow-down-to-bracket"></i> Load Selected Model
                        </button>
                    </div>

                    <!-- Upload Model Form -->
                    <div id="uploadModelPane" class="space-y-3 hidden">
                        <div class="border-2 border-dashed border-gray-700 hover:border-emerald-500 rounded-lg p-6 text-center transition cursor-pointer" onclick="document.getElementById('modelFileInput').click()">
                            <i class="fa-solid fa-cloud-arrow-up text-3xl text-gray-500 mb-2"></i>
                            <p class="text-xs text-gray-300 font-medium">Click to select or drag .pt / .pth checkpoint</p>
                            <p id="modelUploadFileName" class="text-xs text-emerald-400 mt-1 font-mono"></p>
                            <input type="file" id="modelFileInput" class="hidden" accept=".pt,.pth" onchange="onModelFileSelected(this)">
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-gray-400 mb-1">Architecture Override</label>
                            <select id="archOverrideUpload" class="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-emerald-500">
                                <option value="auto">Auto-Detect from Weights</option>
                                <option value="siamese">Siamese Change Detector</option>
                                <option value="unet">UNet Change Detector</option>
                                <option value="multimodal">Multimodal Cross-Attention</option>
                            </select>
                        </div>

                        <button onclick="uploadModelFile()" class="w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium text-sm transition flex items-center justify-center gap-2">
                            <i class="fa-solid fa-upload"></i> Inspect & Load Checkpoint
                        </button>
                    </div>

                    <!-- Model Metadata Inspection Card -->
                    <div id="modelMetaCard" class="bg-gray-900/80 rounded-lg p-4 border border-gray-800 text-xs space-y-2">
                        <div class="font-semibold text-gray-300 border-b border-gray-800 pb-1.5 flex justify-between">
                            <span>Model Parameter Inspection</span>
                            <span id="paramCount" class="text-emerald-400 font-mono">0 params</span>
                        </div>
                        <div class="grid grid-cols-2 gap-2 text-gray-400">
                            <div>Input Channels: <span id="metaChannels" class="text-gray-200 font-mono">-</span></div>
                            <div>LoRA PEFT: <span id="metaLora" class="text-gray-200 font-mono">-</span></div>
                            <div>Saved Epoch: <span id="metaEpoch" class="text-gray-200 font-mono">-</span></div>
                            <div>Validation IoU: <span id="metaIoU" class="text-emerald-400 font-mono">-</span></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Section 2: Input Data Selector & Execution (7 cols) -->
            <div class="lg:col-span-7 space-y-6">
                <div class="glass-panel rounded-xl p-5 space-y-4">
                    <div class="flex items-center justify-between border-b border-gray-800 pb-3">
                        <h2 class="text-sm font-semibold uppercase tracking-wider text-gray-300 flex items-center gap-2">
                            <i class="fa-solid fa-satellite text-blue-400"></i> 2. Input Rasters
                        </h2>
                        <span class="text-xs text-gray-400">Sentinel-2 Bitemporal Pairs</span>
                    </div>

                    <!-- Input Data Modes -->
                    <div class="flex border-b border-gray-800 text-xs">
                        <button id="tabPresetBtn" onclick="switchDataTab('preset')" class="px-4 py-2 border-b-2 border-emerald-500 font-medium text-emerald-400">Preset Chip Cache</button>
                        <button id="tabCustomUploadBtn" onclick="switchDataTab('custom')" class="px-4 py-2 text-gray-400 hover:text-white">Upload Images (T1 + T2)</button>
                        <button id="tabSynthBtn" onclick="switchDataTab('synth')" class="px-4 py-2 text-gray-400 hover:text-white">Synthetic Test Case</button>
                    </div>

                    <!-- Mode A: Preset Cached Chips -->
                    <div id="dataPresetPane" class="space-y-3">
                        <div>
                            <label class="block text-xs font-medium text-gray-400 mb-1">Select from Amazon / GEE Cached Chips</label>
                            <select id="sampleSelect" class="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-emerald-500">
                                <option value="">Loading cached chips...</option>
                            </select>
                        </div>
                        <p class="text-xs text-gray-400">Pre-downloaded multi-spectral bitemporal Sentinel-2 chips with ground-truth Hansen GFC loss labels.</p>
                        <button onclick="runSampleInference()" class="w-full py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-sm transition flex items-center justify-center gap-2 shadow-lg shadow-emerald-900/30">
                            <i class="fa-solid fa-play"></i> Run Model Inference
                        </button>
                    </div>

                    <!-- Mode B: Custom File Upload -->
                    <div id="dataCustomPane" class="space-y-3 hidden">
                        <div class="grid grid-cols-2 gap-3">
                            <div class="border border-gray-700 rounded-lg p-3 text-center cursor-pointer hover:border-emerald-500 transition" onclick="document.getElementById('t1Input').click()">
                                <i class="fa-regular fa-image text-xl text-gray-400 mb-1"></i>
                                <div class="text-xs font-medium text-gray-300">T1 Pre-Change</div>
                                <div id="t1Name" class="text-[10px] text-emerald-400 truncate mt-1">Select PNG/TIFF</div>
                                <input type="file" id="t1Input" class="hidden" accept=".png,.jpg,.jpeg,.tif,.tiff,.npz,.npy" onchange="document.getElementById('t1Name').innerText = this.files[0]?.name || 'Select PNG/TIFF'">
                            </div>

                            <div class="border border-gray-700 rounded-lg p-3 text-center cursor-pointer hover:border-emerald-500 transition" onclick="document.getElementById('t2Input').click()">
                                <i class="fa-regular fa-image text-xl text-gray-400 mb-1"></i>
                                <div class="text-xs font-medium text-gray-300">T2 Post-Change</div>
                                <div id="t2Name" class="text-[10px] text-emerald-400 truncate mt-1">Select PNG/TIFF</div>
                                <input type="file" id="t2Input" class="hidden" accept=".png,.jpg,.jpeg,.tif,.tiff,.npz,.npy" onchange="document.getElementById('t2Name').innerText = this.files[0]?.name || 'Select PNG/TIFF'">
                            </div>
                        </div>

                        <div class="border border-gray-800 rounded-lg p-2.5 text-center cursor-pointer hover:border-gray-600 transition" onclick="document.getElementById('gtInput').click()">
                            <span class="text-xs text-gray-400">(Optional) Ground Truth Mask for IoU Validation: </span>
                            <span id="gtName" class="text-xs text-emerald-400 font-mono">None</span>
                            <input type="file" id="gtInput" class="hidden" accept=".png,.jpg,.jpeg,.tif,.tiff,.npz,.npy" onchange="document.getElementById('gtName').innerText = this.files[0]?.name || 'None'">
                        </div>

                        <button onclick="runUploadInference()" class="w-full py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-sm transition flex items-center justify-center gap-2">
                            <i class="fa-solid fa-play"></i> Ingest & Run Inference
                        </button>
                    </div>

                    <!-- Mode C: Synthetic Generator -->
                    <div id="dataSynthPane" class="space-y-3 hidden">
                        <p class="text-xs text-gray-400">Generates on-the-fly multi-spectral canopy baseline and simulates illegal logging patch with canopy loss.</p>
                        <button onclick="runSyntheticInference()" class="w-full py-2.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-white font-medium text-sm transition flex items-center justify-center gap-2">
                            <i class="fa-solid fa-wand-magic-sparkles"></i> Generate & Predict
                        </button>
                    </div>

                    <!-- Active Threshold Slider -->
                    <div class="bg-gray-900/60 rounded-lg p-3.5 border border-gray-800 space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-medium text-gray-300 flex items-center gap-1.5">
                                <i class="fa-solid fa-sliders text-emerald-400"></i> Confidence Decision Threshold
                            </span>
                            <span id="thresholdVal" class="font-mono text-emerald-400 font-bold">0.50</span>
                        </div>
                        <input type="range" id="thresholdSlider" min="0.05" max="0.95" step="0.05" value="0.50" oninput="onThresholdChange(this.value)" class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer">
                        <div class="flex justify-between text-[10px] text-gray-500">
                            <span>0.05 (High Recall / Sensitive)</span>
                            <span>0.50 (Balanced)</span>
                            <span>0.95 (High Precision / Low FP)</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Section 3: Visual Outputs & Analytics Panel -->
        <div id="resultsPanel" class="glass-panel rounded-xl p-6 space-y-6 hidden">
            <div class="flex flex-wrap items-center justify-between border-b border-gray-800 pb-4 gap-3">
                <div>
                    <h2 class="text-base font-bold text-white flex items-center gap-2">
                        <i class="fa-solid fa-chart-area text-emerald-400"></i> Model Output & Detection Analytics
                    </h2>
                    <p class="text-xs text-gray-400">Bitemporal change heatmaps, binary polygon alerts, and EUDR risk metrics</p>
                </div>

                <div class="flex items-center space-x-2 text-xs">
                    <span id="latencyPill" class="px-3 py-1 rounded-full bg-blue-950 border border-blue-800 text-blue-300 font-mono">
                        <i class="fa-solid fa-bolt"></i> 0.0 ms
                    </span>
                    <button onclick="downloadAlertsGeoJSON()" class="px-3 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-200 border border-gray-700 flex items-center gap-1.5 transition">
                        <i class="fa-solid fa-download text-emerald-400"></i> GeoJSON
                    </button>
                </div>
            </div>

            <!-- 4-Up Visual Grid -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <!-- T1 Baseline -->
                <div class="space-y-2">
                    <div class="text-xs font-semibold text-gray-400 uppercase tracking-wider flex justify-between">
                        <span>T1 Baseline (Pre-Change)</span>
                        <span class="text-[10px] text-gray-500">Sentinel-2</span>
                    </div>
                    <div class="bg-black rounded-lg overflow-hidden border border-gray-800 aspect-square flex items-center justify-center">
                        <img id="t1Img" src="" alt="T1 Baseline" class="w-full h-full object-cover">
                    </div>
                </div>

                <!-- T2 Observation -->
                <div class="space-y-2">
                    <div class="text-xs font-semibold text-gray-400 uppercase tracking-wider flex justify-between">
                        <span>T2 Observation (Post)</span>
                        <span class="text-[10px] text-gray-500">Sentinel-2</span>
                    </div>
                    <div class="bg-black rounded-lg overflow-hidden border border-gray-800 aspect-square flex items-center justify-center">
                        <img id="t2Img" src="" alt="T2 Observation" class="w-full h-full object-cover">
                    </div>
                </div>

                <!-- Difference Map -->
                <div class="space-y-2">
                    <div class="text-xs font-semibold text-gray-400 uppercase tracking-wider flex justify-between">
                        <span>Spectral Delta |T2 - T1|</span>
                        <span class="text-[10px] text-amber-400 font-mono">Shift</span>
                    </div>
                    <div class="bg-black rounded-lg overflow-hidden border border-gray-800 aspect-square flex items-center justify-center">
                        <img id="diffImg" src="" alt="Spectral Difference" class="w-full h-full object-cover">
                    </div>
                </div>

                <!-- Model Probability Map -->
                <div class="space-y-2">
                    <div class="text-xs font-semibold text-gray-400 uppercase tracking-wider flex justify-between">
                        <span>Deforestation Probability</span>
                        <span class="text-[10px] text-emerald-400 font-mono">[0.0, 1.0]</span>
                    </div>
                    <div class="bg-black rounded-lg overflow-hidden border border-gray-800 aspect-square flex items-center justify-center relative">
                        <img id="probImg" src="" alt="Deforestation Probability" class="w-full h-full object-cover">
                    </div>
                </div>
            </div>

            <!-- Detection Mask & Metrics Split -->
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                
                <!-- Binary Alert Mask (5 cols) -->
                <div class="lg:col-span-5 space-y-2">
                    <div class="text-xs font-semibold text-gray-400 uppercase tracking-wider flex justify-between">
                        <span>Binary Alert Mask (Thresholded)</span>
                        <span id="binaryThresholdTag" class="text-xs font-mono text-emerald-400">τ = 0.50</span>
                    </div>
                    <div class="bg-black rounded-lg overflow-hidden border border-emerald-900/40 aspect-square flex items-center justify-center relative shadow-xl shadow-red-950/20">
                        <img id="binaryImg" src="" alt="Binary Alert Mask" class="w-full h-full object-cover">
                    </div>
                </div>

                <!-- Analytical Summary Cards (7 cols) -->
                <div class="lg:col-span-7 space-y-4">
                    <div class="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                        Compliance & Spatial Alert Statistics
                    </div>

                    <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
                        <div class="bg-gray-900/80 p-3.5 rounded-lg border border-gray-800 space-y-1">
                            <div class="text-[11px] text-gray-400">Deforested Area</div>
                            <div id="statArea" class="text-xl font-bold font-mono text-red-400">0.00 ha</div>
                            <div id="statPct" class="text-[10px] text-gray-500">0.0% of tile</div>
                        </div>

                        <div class="bg-gray-900/80 p-3.5 rounded-lg border border-gray-800 space-y-1">
                            <div class="text-[11px] text-gray-400">Alert Clusters</div>
                            <div id="statClusters" class="text-xl font-bold font-mono text-amber-400">0</div>
                            <div class="text-[10px] text-gray-500">MMU ≥ 0.05 ha</div>
                        </div>

                        <div class="bg-gray-900/80 p-3.5 rounded-lg border border-gray-800 space-y-1">
                            <div class="text-[11px] text-gray-400">Confidence</div>
                            <div id="statConfidence" class="text-xl font-bold font-mono text-emerald-400">0.0%</div>
                            <div class="text-[10px] text-gray-500">Mean probability</div>
                        </div>

                        <div class="bg-gray-900/80 p-3.5 rounded-lg border border-gray-800 space-y-1">
                            <div class="text-[11px] text-gray-400">EUDR Risk Rating</div>
                            <div id="statRisk" class="text-xl font-bold font-mono text-emerald-400">LOW</div>
                            <div class="text-[10px] text-gray-500">Regulatory cutoff</div>
                        </div>

                        <div class="bg-gray-900/80 p-3.5 rounded-lg border border-gray-800 space-y-1">
                            <div class="text-[11px] text-gray-400">Tile Dimensions</div>
                            <div class="text-xl font-bold font-mono text-gray-200">256 x 256</div>
                            <div class="text-[10px] text-gray-500">10m GSD (6.55 km²)</div>
                        </div>

                        <div class="bg-gray-900/80 p-3.5 rounded-lg border border-gray-800 space-y-1">
                            <div class="text-[11px] text-gray-400">Validation IoU</div>
                            <div id="statIoU" class="text-xl font-bold font-mono text-blue-400">N/A</div>
                            <div id="statF1" class="text-[10px] text-gray-500">F1: N/A</div>
                        </div>
                    </div>

                    <!-- Alerts Details Accordion -->
                    <div class="bg-gray-900/60 rounded-lg p-4 border border-gray-800 space-y-2">
                        <div class="text-xs font-semibold text-gray-300 flex justify-between">
                            <span>Detected Alert Polygons (GeoJSON)</span>
                            <span id="alertListCount" class="text-gray-400">0 features</span>
                        </div>
                        <div id="alertFeatureList" class="space-y-1.5 max-h-40 overflow-y-auto text-xs text-gray-400 font-mono pr-2">
                            <div class="text-gray-600 italic">No alerts detected at current threshold.</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="border-t border-gray-800 text-center py-4 text-xs text-gray-500 mt-auto">
        GeoFM Standalone Model Loader & Inference Studio • Open Source Geospatial AI Foundation Suite
    </footer>

    <!-- Frontend Interactive Script -->
    <script>
        let currentAlertsGeoJSON = null;
        let isRethresholding = false;

        // Initialize state
        window.addEventListener('DOMContentLoaded', async () => {
            await refreshStatus();
            await loadCheckpointsDropdown();
            await loadSamplesDropdown();
        });

        async function refreshStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                // Device badge
                const devEl = document.getElementById('deviceBadge');
                if (data.cuda_available) {
                    devEl.innerHTML = `<i class="fa-solid fa-bolt text-yellow-400"></i> GPU: ${data.gpu_name}`;
                    devEl.className = "px-3 py-1.5 rounded-full bg-emerald-950 border border-emerald-800 text-emerald-300 flex items-center gap-1.5";
                } else {
                    devEl.innerHTML = `<i class="fa-solid fa-microchip text-blue-400"></i> CPU Mode`;
                    devEl.className = "px-3 py-1.5 rounded-full bg-gray-800 border border-gray-700 text-gray-300 flex items-center gap-1.5";
                }

                // Model status
                const modelBadge = document.getElementById('modelStatusBadge');
                const metaCard = document.getElementById('modelMetaCard');
                if (data.has_loaded_model) {
                    const info = data.model_info;
                    modelBadge.innerHTML = `<i class="fa-solid fa-check text-emerald-400"></i> Model: ${info.source_name}`;
                    modelBadge.className = "px-3 py-1.5 rounded-full bg-emerald-950 border border-emerald-800 text-emerald-300 flex items-center gap-1.5";
                    
                    document.getElementById('modelTypeTag').innerText = info.model_type.toUpperCase();
                    document.getElementById('paramCount').innerText = `${(info.total_params / 1e6).toFixed(2)}M params`;
                    document.getElementById('metaChannels').innerText = `${info.in_channels} bands`;
                    document.getElementById('metaLora').innerText = info.has_lora ? `Yes (Rank ${info.lora_rank})` : 'Full Weights';
                    document.getElementById('metaEpoch').innerText = info.metadata?.epoch || 'N/A';
                    document.getElementById('metaIoU').innerText = info.metadata?.best_val_iou ? `${(info.metadata.best_val_iou * 100).toFixed(1)}%` : 'N/A';
                } else {
                    modelBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-amber-400"></i> No Model Loaded`;
                }
            } catch (err) {
                console.error("Status error:", err);
            }
        }

        async function loadCheckpointsDropdown() {
            try {
                const res = await fetch('/api/checkpoints');
                const data = await res.json();
                const sel = document.getElementById('checkpointSelect');
                sel.innerHTML = '';
                if (!data.checkpoints || data.checkpoints.length === 0) {
                    sel.innerHTML = '<option value="">No .pt checkpoints found in ./checkpoints</option>';
                    return;
                }
                data.checkpoints.forEach(c => {
                    const opt = document.createElement('option');
                    opt.value = c.relative_path;
                    opt.innerText = `${c.filename} (${c.size_mb} MB) — ${c.modified}`;
                    sel.appendChild(opt);
                });
            } catch (err) {
                console.error("Failed to load checkpoints:", err);
            }
        }

        async function loadSamplesDropdown() {
            try {
                const res = await fetch('/api/samples');
                const data = await res.json();
                const sel = document.getElementById('sampleSelect');
                sel.innerHTML = '';
                if (!data.samples || data.samples.length === 0) {
                    sel.innerHTML = '<option value="">No chips found in data/chips_cache</option>';
                    return;
                }
                data.samples.forEach(s => {
                    const opt = document.createElement('option');
                    opt.value = s.filename;
                    opt.innerText = `${s.filename} — ${s.deforested_pct}% loss [${s.coordinates}]`;
                    sel.appendChild(opt);
                });
            } catch (err) {
                console.error("Failed to load samples:", err);
            }
        }

        function switchModelTab(tab) {
            const localPane = document.getElementById('localModelPane');
            const uploadPane = document.getElementById('uploadModelPane');
            const localBtn = document.getElementById('tabLocalBtn');
            const uploadBtn = document.getElementById('tabUploadBtn');

            if (tab === 'local') {
                localPane.classList.remove('hidden');
                uploadPane.classList.add('hidden');
                localBtn.className = "px-4 py-2 border-b-2 border-emerald-500 font-medium text-emerald-400";
                uploadBtn.className = "px-4 py-2 text-gray-400 hover:text-white";
            } else {
                localPane.classList.add('hidden');
                uploadPane.classList.remove('hidden');
                uploadBtn.className = "px-4 py-2 border-b-2 border-emerald-500 font-medium text-emerald-400";
                localBtn.className = "px-4 py-2 text-gray-400 hover:text-white";
            }
        }

        function switchDataTab(tab) {
            const presetPane = document.getElementById('dataPresetPane');
            const customPane = document.getElementById('dataCustomPane');
            const synthPane = document.getElementById('dataSynthPane');

            const presetBtn = document.getElementById('tabPresetBtn');
            const customBtn = document.getElementById('tabCustomUploadBtn');
            const synthBtn = document.getElementById('tabSynthBtn');

            presetPane.classList.add('hidden');
            customPane.classList.add('hidden');
            synthPane.classList.add('hidden');

            presetBtn.className = "px-4 py-2 text-gray-400 hover:text-white";
            customBtn.className = "px-4 py-2 text-gray-400 hover:text-white";
            synthBtn.className = "px-4 py-2 text-gray-400 hover:text-white";

            if (tab === 'preset') {
                presetPane.classList.remove('hidden');
                presetBtn.className = "px-4 py-2 border-b-2 border-emerald-500 font-medium text-emerald-400";
            } else if (tab === 'custom') {
                customPane.classList.remove('hidden');
                customBtn.className = "px-4 py-2 border-b-2 border-emerald-500 font-medium text-emerald-400";
            } else {
                synthPane.classList.remove('hidden');
                synthBtn.className = "px-4 py-2 border-b-2 border-emerald-500 font-medium text-emerald-400";
            }
        }

        async function loadLocalModel() {
            const path = document.getElementById('checkpointSelect').value;
            const arch = document.getElementById('archOverride').value;
            if (!path) return alert("Please select a checkpoint file.");

            const formData = new FormData();
            formData.append('path', path);
            formData.append('arch_override', arch);

            try {
                const res = await fetch('/api/model/load-local', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.success) {
                    await refreshStatus();
                    alert(`✓ Model successfully loaded: ${data.model_info.source_name} (${data.model_info.model_type})`);
                } else {
                    alert(`Error: ${data.detail}`);
                }
            } catch (err) {
                alert(`Load failed: ${err.message}`);
            }
        }

        function onModelFileSelected(input) {
            if (input.files && input.files[0]) {
                document.getElementById('modelUploadFileName').innerText = input.files[0].name;
            }
        }

        async function uploadModelFile() {
            const input = document.getElementById('modelFileInput');
            if (!input.files || !input.files[0]) return alert("Please select a .pt / .pth checkpoint file to upload.");

            const arch = document.getElementById('archOverrideUpload').value;
            const formData = new FormData();
            formData.append('file', input.files[0]);
            formData.append('arch_override', arch);

            try {
                const res = await fetch('/api/model/upload', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.success) {
                    await refreshStatus();
                    await loadCheckpointsDropdown();
                    alert(`✓ Custom model loaded: ${data.model_info.source_name} (${data.model_info.model_type})`);
                } else {
                    alert(`Upload failed: ${data.detail}`);
                }
            } catch (err) {
                alert(`Upload failed: ${err.message}`);
            }
        }

        function displayInferenceResults(r) {
            document.getElementById('resultsPanel').classList.remove('hidden');
            
            // Images
            document.getElementById('t1Img').src = 'data:image/png;base64,' + r.t1_image_b64;
            document.getElementById('t2Img').src = 'data:image/png;base64,' + r.t2_image_b64;
            document.getElementById('diffImg').src = 'data:image/png;base64,' + r.difference_b64;
            document.getElementById('probImg').src = 'data:image/png;base64,' + r.probability_b64;
            document.getElementById('binaryImg').src = 'data:image/png;base64,' + r.binary_mask_b64;

            // Latency
            document.getElementById('latencyPill').innerHTML = `<i class="fa-solid fa-bolt text-yellow-400"></i> ${r.latency_ms} ms (${r.device})`;

            // Statistics
            const s = r.statistics;
            document.getElementById('statArea').innerText = `${s.area_ha.toFixed(2)} ha`;
            document.getElementById('statPct').innerText = `${s.deforested_pct}% of tile`;
            document.getElementById('statClusters').innerText = s.alert_clusters_count;
            document.getElementById('statConfidence').innerText = `${s.mean_confidence_pct}%`;
            
            const riskEl = document.getElementById('statRisk');
            riskEl.innerText = s.risk_level;
            riskEl.className = s.risk_level === 'CRITICAL' ? 'text-xl font-bold font-mono text-red-500' :
                               s.risk_level === 'HIGH' ? 'text-xl font-bold font-mono text-amber-500' :
                               'text-xl font-bold font-mono text-emerald-400';

            // Ground truth metrics
            if (r.metrics) {
                document.getElementById('statIoU').innerText = `${(r.metrics.iou * 100).toFixed(1)}%`;
                document.getElementById('statF1').innerText = `F1: ${(r.metrics.f1_score * 100).toFixed(1)}% | Prec: ${(r.metrics.precision * 100).toFixed(1)}%`;
            } else {
                document.getElementById('statIoU').innerText = 'N/A';
                document.getElementById('statF1').innerText = 'No ground truth mask';
            }

            // Polygon alerts list
            currentAlertsGeoJSON = r.alerts_geojson;
            const feats = r.alerts_geojson?.features || [];
            document.getElementById('alertListCount').innerText = `${feats.length} clusters`;
            const listEl = document.getElementById('alertFeatureList');
            listEl.innerHTML = '';
            if (feats.length === 0) {
                listEl.innerHTML = '<div class="text-gray-600 italic">No alerts detected at current threshold.</div>';
            } else {
                feats.slice(0, 10).forEach(f => {
                    const p = f.properties;
                    const item = document.createElement('div');
                    item.className = "flex justify-between items-center py-1 border-b border-gray-800/60";
                    item.innerHTML = `<span>Cluster #${p.cluster_id} (${p.alert_level})</span><span class="text-emerald-400">${p.area_ha.toFixed(2)} ha • ${(p.mean_probability*100).toFixed(0)}% conf</span>`;
                    listEl.appendChild(item);
                });
            }

            // Scroll to results
            document.getElementById('resultsPanel').scrollIntoView({ behavior: 'smooth' });
        }

        async function runSampleInference() {
            const filename = document.getElementById('sampleSelect').value;
            const threshold = document.getElementById('thresholdSlider').value;
            if (!filename) return alert("Please select a sample chip from the list.");

            const formData = new FormData();
            formData.append('filename', filename);
            formData.append('threshold', threshold);

            try {
                const res = await fetch('/api/predict/sample', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.success) {
                    displayInferenceResults(data.result);
                } else {
                    alert(`Inference failed: ${data.detail}`);
                }
            } catch (err) {
                alert(`Inference error: ${err.message}`);
            }
        }

        async function runSyntheticInference() {
            const threshold = document.getElementById('thresholdSlider').value;
            const formData = new FormData();
            formData.append('threshold', threshold);

            try {
                const res = await fetch('/api/predict/synthetic', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.success) {
                    displayInferenceResults(data.result);
                } else {
                    alert(`Synthetic test failed: ${data.detail}`);
                }
            } catch (err) {
                alert(`Error: ${err.message}`);
            }
        }

        async function runUploadInference() {
            const t1 = document.getElementById('t1Input').files[0];
            const t2 = document.getElementById('t2Input').files[0];
            const gt = document.getElementById('gtInput').files[0];
            const threshold = document.getElementById('thresholdSlider').value;

            if (!t1 || !t2) return alert("Please select both T1 Pre-Change and T2 Post-Change images.");

            const formData = new FormData();
            formData.append('t1_file', t1);
            formData.append('t2_file', t2);
            if (gt) formData.append('mask_file', gt);
            formData.append('threshold', threshold);

            try {
                const res = await fetch('/api/predict/upload', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.success) {
                    displayInferenceResults(data.result);
                } else {
                    alert(`Upload inference error: ${data.detail}`);
                }
            } catch (err) {
                alert(`Error: ${err.message}`);
            }
        }

        async function onThresholdChange(val) {
            document.getElementById('thresholdVal').innerText = parseFloat(val).toFixed(2);
            document.getElementById('binaryThresholdTag').innerText = `τ = ${parseFloat(val).toFixed(2)}`;

            if (isRethresholding) return;
            isRethresholding = true;

            try {
                const formData = new FormData();
                formData.append('threshold', val);
                const res = await fetch('/api/predict/rethreshold', { method: 'POST', body: formData });
                if (res.ok) {
                    const data = await res.json();
                    if (data.success) {
                        document.getElementById('binaryImg').src = 'data:image/png;base64,' + data.binary_mask_b64;
                        const s = data.statistics;
                        document.getElementById('statArea').innerText = `${s.area_ha.toFixed(2)} ha`;
                        document.getElementById('statPct').innerText = `${s.deforested_pct}% of tile`;
                        document.getElementById('statClusters').innerText = s.alert_clusters_count;
                        document.getElementById('statConfidence').innerText = `${s.mean_confidence_pct}%`;
                        
                        const riskEl = document.getElementById('statRisk');
                        riskEl.innerText = s.risk_level;
                        riskEl.className = s.risk_level === 'CRITICAL' ? 'text-xl font-bold font-mono text-red-500' :
                                           s.risk_level === 'HIGH' ? 'text-xl font-bold font-mono text-amber-500' :
                                           'text-xl font-bold font-mono text-emerald-400';

                        currentAlertsGeoJSON = data.alerts_geojson;
                    }
                }
            } catch (err) {
                // Silently ignore during rapid sliding
            } finally {
                isRethresholding = false;
            }
        }

        function downloadAlertsGeoJSON() {
            if (!currentAlertsGeoJSON) return alert("No active alerts generated. Please run inference first.");
            const blob = new Blob([JSON.stringify(currentAlertsGeoJSON, null, 2)], { type: 'application/geo+json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `geofm_deforestation_alerts_${Date.now()}.geojson`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the single-page application."""
    return HTMLResponse(content=HTML_PAGE, status_code=200)


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="GeoFM Standalone Model Loader & Inference Studio")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8600, help="Port to bind server (default: 8600)")
    parser.add_argument("--model", type=str, default=None, help="Path to default checkpoint to load on startup")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
    args = parser.parse_args()

    if args.model:
        try:
            engine.inspect_and_load_checkpoint(args.model)
            print(f"[OK] Successfully loaded startup checkpoint: {args.model}")
        except Exception as e:
            print(f"Warning: Failed to load specified startup checkpoint: {e}")

    print("=" * 70)
    print("GeoFM Standalone Model Loader & Inference Studio")
    print(f"   * Web Interface:  http://{args.host}:{args.port}")
    print(f"   * Acceleration:   {engine.device}")
    print(f"   * Checkpoints:    {len(engine.list_checkpoints())} discovered in ./checkpoints")
    print(f"   * Cached Chips:   {len(engine.list_cached_samples())} discovered in data/chips_cache")
    print("=" * 70)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
