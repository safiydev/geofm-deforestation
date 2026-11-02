"""Spectral indices and multi-spectral band processing for deforestation detection.

Healthy forest canopies exhibit massive Near-Infrared (NIR, Band 8) scattering from
spongy mesophyll leaf structures and strong Short-Wave Infrared (SWIR, Bands 11 & 12)
absorption due to canopy water content. Deforested and clear-cut land does the reverse,
producing steep drops in NIR and sharp increases in SWIR.

Appending differential spectral indices (ΔNDVI, ΔNBR) directly feeds physical
canopy disturbance signatures into neural encoder backbones.
"""

from typing import Union, Tuple
import numpy as np
import torch


def compute_ndvi(
    optical: Union[np.ndarray, torch.Tensor],
    red_idx: int = 2,
    nir_idx: int = 3,
    eps: float = 1e-6,
) -> Union[np.ndarray, torch.Tensor]:
    """Compute Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red + eps)

    Args:
        optical: Multi-spectral array of shape (C, H, W) or (B, C, H, W).
        red_idx: Channel index of Red band (Sentinel-2 B04).
        nir_idx: Channel index of NIR band (Sentinel-2 B08).
        eps: Small epsilon to prevent division by zero.

    Returns:
        NDVI array or tensor of shape (1, H, W) or (B, 1, H, W) bounded in [-1.0, 1.0].
    """
    if isinstance(optical, torch.Tensor):
        red = optical[:, red_idx : red_idx + 1] if optical.ndim == 4 else optical[red_idx : red_idx + 1]
        nir = optical[:, nir_idx : nir_idx + 1] if optical.ndim == 4 else optical[nir_idx : nir_idx + 1]
        ndvi = (nir - red) / (nir + red + eps)
        return torch.clamp(ndvi, -1.0, 1.0)
    else:
        red = optical[:, red_idx : red_idx + 1] if optical.ndim == 4 else optical[red_idx : red_idx + 1]
        nir = optical[:, nir_idx : nir_idx + 1] if optical.ndim == 4 else optical[nir_idx : nir_idx + 1]
        ndvi = (nir - red) / (nir + red + eps)
        return np.clip(ndvi, -1.0, 1.0)


def compute_nbr(
    optical: Union[np.ndarray, torch.Tensor],
    nir_idx: int = 3,
    swir2_idx: int = 5,
    eps: float = 1e-6,
) -> Union[np.ndarray, torch.Tensor]:
    """Compute Normalized Burn Ratio (NBR).

    NBR = (NIR - SWIR2) / (NIR + SWIR2 + eps)
    High positive values indicate healthy moist canopy. Burn scars and bare slash
    produce negative NBR values.

    Args:
        optical: Multi-spectral array with >= 6 bands (B, G, R, NIR, SWIR1, SWIR2).
        nir_idx: Channel index of NIR band (Sentinel-2 B08).
        swir2_idx: Channel index of SWIR-2 band (Sentinel-2 B12).
        eps: Small epsilon to prevent division by zero.

    Returns:
        NBR array or tensor of shape (1, H, W) or (B, 1, H, W) bounded in [-1.0, 1.0].
    """
    if isinstance(optical, torch.Tensor):
        nir = optical[:, nir_idx : nir_idx + 1] if optical.ndim == 4 else optical[nir_idx : nir_idx + 1]
        swir = optical[:, swir2_idx : swir2_idx + 1] if optical.ndim == 4 else optical[swir2_idx : swir2_idx + 1]
        nbr = (nir - swir) / (nir + swir + eps)
        return torch.clamp(nbr, -1.0, 1.0)
    else:
        nir = optical[:, nir_idx : nir_idx + 1] if optical.ndim == 4 else optical[nir_idx : nir_idx + 1]
        swir = optical[:, swir2_idx : swir2_idx + 1] if optical.ndim == 4 else optical[swir2_idx : swir2_idx + 1]
        nbr = (nir - swir) / (nir + swir + eps)
        return np.clip(nbr, -1.0, 1.0)


def compute_spectral_differential_stack(
    t1_optical: Union[np.ndarray, torch.Tensor],
    t2_optical: Union[np.ndarray, torch.Tensor],
    red_idx: int = 2,
    nir_idx: int = 3,
    swir2_idx: Optional[int] = 5,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Append ΔNDVI and ΔNBR difference channels to bitemporal optical stacks.

    Returns enriched (C + N_indices, H, W) stacks for T1 and T2.
    """
    is_torch = isinstance(t1_optical, torch.Tensor)
    num_channels = t1_optical.shape[1] if t1_optical.ndim == 4 else t1_optical.shape[0]

    ndvi_t1 = compute_ndvi(t1_optical, red_idx=red_idx, nir_idx=min(nir_idx, num_channels - 1))
    ndvi_t2 = compute_ndvi(t2_optical, red_idx=red_idx, nir_idx=min(nir_idx, num_channels - 1))
    delta_ndvi = ndvi_t2 - ndvi_t1

    indices_t1 = [ndvi_t1]
    indices_t2 = [ndvi_t2]

    if swir2_idx is not None and num_channels > swir2_idx:
        nbr_t1 = compute_nbr(t1_optical, nir_idx=nir_idx, swir2_idx=swir2_idx)
        nbr_t2 = compute_nbr(t2_optical, nir_idx=nir_idx, swir2_idx=swir2_idx)
        indices_t1.append(nbr_t1)
        indices_t2.append(nbr_t2)

    cat_dim = 1 if (is_torch and t1_optical.ndim == 4) or (not is_torch and t1_optical.ndim == 4) else 0

    if is_torch:
        t1_out = torch.cat([t1_optical] + indices_t1, dim=cat_dim)
        t2_out = torch.cat([t2_optical] + indices_t2, dim=cat_dim)
    else:
        t1_out = np.concatenate([t1_optical] + indices_t1, axis=cat_dim)
        t2_out = np.concatenate([t2_optical] + indices_t2, axis=cat_dim)

    return t1_out, t2_out
