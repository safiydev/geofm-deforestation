"""Synthetic multimodal geospatial data generator for tests and demonstrations.

Generates realistic bitemporal Sentinel-2 (optical multi-spectral RGB+NIR+SWIR),
Sentinel-1 (SAR VV+VH), cloud masks, and deforestation ground-truth change masks.
Supports 6-band multispectral simulation (B02, B03, B04, B08, B11, B12),
differential spectral indices (ΔNDVI, ΔNBR), and 70/30 class-balance filtering.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from geofm.training.dataset import GeoSample
from geofm.preprocessing.spectral_indices import compute_spectral_differential_stack


def generate_synthetic_scene(
    height: int = 128,
    width: int = 128,
    seed: int = 42,
    deforestation_fraction: float = 0.08,
    cloud_fraction: float = 0.15,
    num_bands: int = 4,
    append_indices: bool = False,
    is_hard_negative: bool = False,
) -> Dict[str, np.ndarray]:
    """Generate a realistic bitemporal optical + SAR satellite scene pair.

    Args:
        height: Spatial height in pixels.
        width: Spatial width in pixels.
        seed: Random generator seed.
        deforestation_fraction: Target area fraction for deforestation (>0.05).
        cloud_fraction: Area fraction for cloud occlusion on T2.
        num_bands: Number of raw spectral channels (4 for RGB+NIR, 6 for RGB+NIR+SWIR1+SWIR2).
        append_indices: Whether to append pre-calculated ΔNDVI and ΔNBR channels.
        is_hard_negative: If True, produces 0% deforestation with cloud/phenological noise.

    Returns:
        dict containing:
          - t1_optical: (C, H, W) pre-disturbance optical reflectance
          - t2_optical: (C, H, W) post-disturbance with deforested clear-cuts
          - sar_vv: (H, W) post-disturbance VV intensity
          - sar_vh: (H, W) post-disturbance VH intensity
          - cloud_mask: (H, W) boolean cloud mask
          - change_mask: (H, W) boolean deforestation mask
          - deforested_pct: float percentage of chip deforested
    """
    rng = np.random.default_rng(seed)

    # 1. Base intact forest optical reflectance (Sentinel-2 L2A scaled 0-1)
    # Intact tropical forest: low Blue (~0.03), medium Green (~0.06), low Red (~0.04), high NIR (~0.48)
    base_blue = rng.normal(0.03, 0.005, (height, width))
    base_green = rng.normal(0.06, 0.008, (height, width))
    base_red = rng.normal(0.04, 0.006, (height, width))
    base_nir = rng.normal(0.48, 0.04, (height, width))

    bands_t1 = [base_blue, base_green, base_red, base_nir]

    if num_bands >= 6:
        # Healthy moist forest canopy absorbs SWIR strongly: SWIR-1 (~0.10), SWIR-2 (~0.06)
        base_swir1 = rng.normal(0.10, 0.015, (height, width))
        base_swir2 = rng.normal(0.06, 0.010, (height, width))
        bands_t1.extend([base_swir1, base_swir2])

    t1_opt = np.stack(bands_t1, axis=0).clip(0.0, 1.0)
    t2_opt = np.copy(t1_opt)

    # 2. Deforestation patterns (clear-cut blocks and linear logging corridors)
    change_mask = np.zeros((height, width), dtype=bool)

    if not is_hard_negative:
        # Deforestation clear-cut polygon (enforce >5% deforested pixels)
        cut_h = max(int(height * 0.25), 8)
        cut_w = max(int(width * 0.25), 8)
        r0 = int(height * 0.30)
        c0 = int(width * 0.30)
        change_mask[r0 : r0 + cut_h, c0 : c0 + cut_w] = True

        # Linear logging road corridor
        road_r = int(height * 0.68)
        road_width = max(2, int(height * 0.025))
        change_mask[road_r : road_r + road_width, : int(width * 0.85)] = True

        # In deforested areas:
        # Red increases (~0.18 bare dry soil)
        # NIR drops sharply (~0.14 loss of mesophyll scattering)
        n_changed = int(np.sum(change_mask))
        t2_opt[2, change_mask] = rng.normal(0.18, 0.02, n_changed).clip(0.05, 0.5)
        t2_opt[3, change_mask] = rng.normal(0.14, 0.02, n_changed).clip(0.05, 0.5)

        if num_bands >= 6:
            # Deforested dry soil and slash reflect massive amounts of SWIR:
            # SWIR-1 jumps from 0.10 -> 0.38
            # SWIR-2 jumps from 0.06 -> 0.26
            t2_opt[4, change_mask] = rng.normal(0.38, 0.03, n_changed).clip(0.15, 0.7)
            t2_opt[5, change_mask] = rng.normal(0.26, 0.02, n_changed).clip(0.10, 0.6)
    else:
        # Hard Negative: Add subtle seasonal phenology / dry grass variation without tree loss
        # NIR drops slightly by 0.03, Red rises by 0.01 uniformly across canopy
        t2_opt[2] = np.clip(t2_opt[2] + rng.normal(0.01, 0.003, (height, width)), 0.0, 1.0)
        t2_opt[3] = np.clip(t2_opt[3] - rng.normal(0.02, 0.005, (height, width)), 0.0, 1.0)

    # 3. SAR backscatter (Sentinel-1 C-band linear intensity)
    sar_vv = rng.exponential(0.12, (height, width))
    sar_vh = rng.exponential(0.05, (height, width))

    if not is_hard_negative and np.any(change_mask):
        n_changed = int(np.sum(change_mask))
        sar_vh[change_mask] = rng.exponential(0.008, n_changed)
        sar_vv[change_mask] = rng.exponential(0.04, n_changed)

    # 4. Cloud occlusion on T2 optical
    cloud_mask = np.zeros((height, width), dtype=bool)
    if cloud_fraction > 0:
        cr, cc = int(height * 0.1), int(width * 0.1)
        ch = max(1, int(height * cloud_fraction))
        cw = max(1, int(width * cloud_fraction))
        cloud_mask[cr : cr + ch, cc : cc + cw] = True
        n_clouds = int(np.sum(cloud_mask))
        for b_i in range(len(t2_opt)):
            t2_opt[b_i, cloud_mask] = rng.normal(0.65, 0.05, n_clouds).clip(0.4, 1.0)

    # 5. Optionally append pre-calculated ΔNDVI and ΔNBR
    if append_indices:
        t1_opt, t2_opt = compute_spectral_differential_stack(
            t1_opt, t2_opt, red_idx=2, nir_idx=3, swir2_idx=5 if num_bands >= 6 else None
        )

    deforested_pct = float(np.mean(change_mask) * 100.0)

    return {
        "t1_optical": t1_opt.astype(np.float32),
        "t2_optical": t2_opt.astype(np.float32),
        "sar_vv": sar_vv.astype(np.float32),
        "sar_vh": sar_vh.astype(np.float32),
        "cloud_mask": cloud_mask,
        "change_mask": change_mask,
        "deforested_pct": deforested_pct,
    }


def generate_synthetic_dataset(
    num_samples: int = 16,
    height: int = 64,
    width: int = 64,
    start_lon: float = -60.0,
    start_lat: float = -3.0,
    coord_step: float = 0.05,
    num_bands: int = 4,
    append_indices: bool = False,
) -> List[GeoSample]:
    """Generate a collection of GeoSamples for spatial cross-validation tests."""
    samples = []
    for i in range(num_samples):
        lon = start_lon + (i % 4) * coord_step
        lat = start_lat + (i // 4) * coord_step
        scene = generate_synthetic_scene(
            height=height,
            width=width,
            seed=100 + i,
            num_bands=num_bands,
            append_indices=append_indices,
        )

        sar_stacked = np.stack(
            [scene["sar_vv"], scene["sar_vh"], scene["sar_vh"] / (scene["sar_vv"] + 1e-6)],
            axis=0,
        )

        sample = GeoSample(
            sample_id=f"chip_{i:04d}",
            t1_optical=scene["t1_optical"],
            t2_optical=scene["t2_optical"],
            target_mask=scene["change_mask"][np.newaxis, ...].astype(np.float32),
            sar_data=sar_stacked,
            cloud_mask=scene["cloud_mask"][np.newaxis, ...].astype(np.float32),
            lon=lon,
            lat=lat,
        )
        samples.append(sample)

    return samples


def stream_and_cache_synthetic_dataset(
    output_dir: str,
    num_samples: int = 16,
    height: int = 64,
    width: int = 64,
    start_lon: float = -60.0,
    start_lat: float = -3.0,
    coord_step: float = 0.05,
    batch_flush: int = 50,
    show_progress: bool = True,
    num_bands: int = 4,
    append_indices: bool = False,
    positive_ratio: float = 0.70,
) -> Tuple[str, int]:
    """Stream-generate synthetic geospatial chips directly to disk cache with ZERO RAM accumulation.

    Enforces class-balance sampling: by default, 70% of chips contain >5% deforested pixels
    (clear cuts + logging corridors), while 30% are pure unchanged/cloud hard-negatives to
    prevent false alarms on unperturbed forest and cloud shadows.

    Supports 4-band (RGB+NIR) and 6-band (RGB+NIR+SWIR1+SWIR2) multispectral generation,
    plus pre-calculated spectral difference channels (ΔNDVI, ΔNBR).

    Args:
        output_dir: Target directory where .npz chips and manifest.jsonl are stored.
        num_samples: Total number of paired chips to generate (e.g. 100, 1000, 10000).
        height: Spatial height of each satellite chip.
        width: Spatial width of each satellite chip.
        start_lon: Starting longitude coordinate.
        start_lat: Starting latitude coordinate.
        coord_step: Spatial distance increment between consecutive tiles.
        batch_flush: Number of chips generated before triggering garbage collection.
        show_progress: Whether to print progress milestones.
        num_bands: Spectral channels (4 or 6).
        append_indices: Whether to append ΔNDVI and ΔNBR.
        positive_ratio: Target proportion of chips containing >5% deforested pixels (default: 0.70).

    Returns:
        (manifest_path, total_chips_saved)
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.jsonl")

    num_positive = int(num_samples * positive_ratio)

    with open(manifest_path, "w", encoding="utf-8") as f_out:
        for i in range(num_samples):
            lon = float(start_lon + (i % 25) * coord_step)
            lat = float(start_lat + (i // 25) * coord_step)

            # Class balance filter: 70% positive (>5% deforestation), 30% hard negative (0% deforestation)
            is_negative = (i >= num_positive)

            scene = generate_synthetic_scene(
                height=height,
                width=width,
                seed=100 + i,
                num_bands=num_bands,
                append_indices=append_indices,
                is_hard_negative=is_negative,
            )

            sar_stacked = np.stack(
                [scene["sar_vv"], scene["sar_vh"], scene["sar_vh"] / (scene["sar_vv"] + 1e-6)],
                axis=0,
            ).astype(np.float32)

            chip_id = f"chip_{i:05d}"
            file_name = f"{chip_id}.npz"
            file_path = os.path.join(output_dir, file_name)

            np.savez_compressed(
                file_path,
                sample_id=chip_id,
                t1_optical=scene["t1_optical"],
                t2_optical=scene["t2_optical"],
                target_mask=scene["change_mask"][np.newaxis, ...].astype(np.float32),
                sar_data=sar_stacked,
                cloud_mask=scene["cloud_mask"][np.newaxis, ...].astype(np.float32),
                lon=lon,
                lat=lat,
                block_id=0,
                deforested_pct=scene["deforested_pct"],
            )

            record = {
                "sample_id": chip_id,
                "file_path": file_name,
                "lon": lon,
                "lat": lat,
                "block_id": 0,
                "deforested_pct": scene["deforested_pct"],
            }
            f_out.write(json.dumps(record) + "\n")

            del scene
            del sar_stacked

            if (i + 1) % batch_flush == 0:
                f_out.flush()
                gc.collect()

            if show_progress and ((i + 1) % max(1, num_samples // 10) == 0 or (i + 1) == num_samples):
                pct = ((i + 1) / num_samples) * 100
                print(f"  [Disk Streaming] {i + 1}/{num_samples} chips cached ({pct:.0f}%) -> {output_dir}")

    return manifest_path, num_samples
