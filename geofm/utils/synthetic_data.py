"""Synthetic multimodal geospatial data generator for tests and demonstrations.

Generates realistic bitemporal Sentinel-2 (optical RGB+NIR), Sentinel-1 (SAR VV+VH),
cloud masks, and deforestation ground-truth change masks.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from geofm.training.dataset import GeoSample


def generate_synthetic_scene(
    height: int = 128,
    width: int = 128,
    seed: int = 42,
    deforestation_fraction: float = 0.05,
    cloud_fraction: float = 0.15,
) -> Dict[str, np.ndarray]:
    """Generate a realistic bitemporal optical + SAR satellite scene pair.

    Returns:
        dict containing:
          - t1_optical: (4, H, W) [Blue, Green, Red, NIR] pre-disturbance
          - t2_optical: (4, H, W) post-disturbance with deforested clear-cuts
          - sar_vv: (H, W) post-disturbance VV intensity
          - sar_vh: (H, W) post-disturbance VH intensity
          - cloud_mask: (H, W) boolean cloud mask
          - change_mask: (H, W) boolean deforestation mask
    """
    rng = np.random.default_rng(seed)

    # 1. Base intact forest optical reflectance (Sentinel-2 L2A scaled 0-1)
    # Intact tropical forest: low Blue (~0.03), low Red (~0.04), medium Green (~0.06), high NIR (~0.45)
    base_blue = rng.normal(0.03, 0.005, (height, width))
    base_green = rng.normal(0.06, 0.008, (height, width))
    base_red = rng.normal(0.04, 0.006, (height, width))
    base_nir = rng.normal(0.45, 0.04, (height, width))

    t1_opt = np.stack([base_blue, base_green, base_red, base_nir], axis=0).clip(0.0, 1.0)
    t2_opt = np.copy(t1_opt)

    # 2. Deforestation patterns (clear-cut blocks and linear logging roads)
    change_mask = np.zeros((height, width), dtype=bool)

    # Rectangular clear-cut
    cut_h, cut_w = int(height * 0.25), int(width * 0.25)
    r0, c0 = int(height * 0.35), int(width * 0.35)
    change_mask[r0 : r0 + cut_h, c0 : c0 + cut_w] = True

    # Linear logging road
    road_r = int(height * 0.65)
    change_mask[road_r : road_r + 3, : int(width * 0.8)] = True

    # In deforested areas: Red increases (~0.18), NIR drops sharply (~0.15 bare soil)
    t2_opt[2, change_mask] = rng.normal(0.18, 0.02, np.sum(change_mask)).clip(0.05, 0.5)
    t2_opt[3, change_mask] = rng.normal(0.15, 0.02, np.sum(change_mask)).clip(0.05, 0.5)

    # 3. SAR backscatter (Sentinel-1 C-band linear intensity)
    # Intact forest: high volume scattering in VH (~0.05 linear, -13 dB) and VV (~0.12 linear, -9 dB)
    # Deforested: surface scattering, severe drop in VH (~0.008 linear, -21 dB)
    sar_vv = rng.exponential(0.12, (height, width))
    sar_vh = rng.exponential(0.05, (height, width))

    # Deforested drop
    sar_vh[change_mask] = rng.exponential(0.008, np.sum(change_mask))
    sar_vv[change_mask] = rng.exponential(0.04, np.sum(change_mask))

    # 4. Cloud occlusion on T2 optical
    cloud_mask = np.zeros((height, width), dtype=bool)
    if cloud_fraction > 0:
        cr, cc = int(height * 0.1), int(width * 0.1)
        ch, cw = int(height * cloud_fraction), int(width * cloud_fraction)
        cloud_mask[cr : cr + ch, cc : cc + cw] = True
        # Bright cloud saturation
        t2_opt[0, cloud_mask] = rng.normal(0.65, 0.05, np.sum(cloud_mask)).clip(0.4, 1.0)
        t2_opt[1, cloud_mask] = rng.normal(0.65, 0.05, np.sum(cloud_mask)).clip(0.4, 1.0)
        t2_opt[2, cloud_mask] = rng.normal(0.65, 0.05, np.sum(cloud_mask)).clip(0.4, 1.0)
        t2_opt[3, cloud_mask] = rng.normal(0.60, 0.05, np.sum(cloud_mask)).clip(0.4, 1.0)

    return {
        "t1_optical": t1_opt.astype(np.float32),
        "t2_optical": t2_opt.astype(np.float32),
        "sar_vv": sar_vv.astype(np.float32),
        "sar_vh": sar_vh.astype(np.float32),
        "cloud_mask": cloud_mask,
        "change_mask": change_mask,
    }


def generate_synthetic_dataset(
    num_samples: int = 16,
    height: int = 64,
    width: int = 64,
    start_lon: float = -60.0,
    start_lat: float = -3.0,
    coord_step: float = 0.05,
) -> List[GeoSample]:
    """Generate a collection of GeoSamples for spatial cross-validation tests."""
    samples = []
    for i in range(num_samples):
        lon = start_lon + (i % 4) * coord_step
        lat = start_lat + (i // 4) * coord_step
        scene = generate_synthetic_scene(height=height, width=width, seed=100 + i)

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
