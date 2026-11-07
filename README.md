# GeoFM: Geospatial AI for Early Deforestation Detection & EUDR Compliance

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-06B6D4.svg)](https://safiydev.github.io/geofm-deforestation/)
[![CI](https://github.com/safiydev/geofm-deforestation/actions/workflows/ci.yml/badge.svg)](https://github.com/safiydev/geofm-deforestation/actions)
[![Tests](https://img.shields.io/badge/Tests-95%20passed-10B981.svg)](tests/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safiydev/geofm-deforestation/blob/main/notebooks/geofm_training.ipynb)

A production-grade, multimodal Geospatial Artificial Intelligence system engineered for **bitemporal change detection** of tropical deforestation, Vision Transformer (ViT) spatial attention, Parameter-Efficient Fine-Tuning (PEFT/LoRA), and European Union Deforestation Regulation (EUDR, Regulation (EU) 2023/1115) compliance auditing. Rather than forecasting hypothetical future events, GeoFM compares paired satellite acquisitions across two timestamps ($T_1$ baseline and $T_2$ observation) to detect verified forest canopy disturbances, logging roads, and clear-cuts as they occur.

---

## Architectural Overview

```
                                  +------------------------------------+
                                  |     Multimodal Satellite Feeds     |
                                  |  - Sentinel-2 (Copernicus Optical) |
                                  |  - Planet NICFI (4.77m Optical)    |
                                  |  - Sentinel-1 (C-Band SAR GRD IW)  |
                                  +-----------------+------------------+
                                                    |
                                                    v
+-------------------------------------------------------------------------------------------------------+
|                                1. PHYSICAL SIGNAL PREPROCESSING & STAC                                |
|  - Optical: Surface Reflectance Normalization (10,000 scale), NDVI/EVI/NDWI Indices                   |
|  - Optical Cloud/Shadow Masking (Blue band saturation & NIR/SWIR shadow projection)                   |
|  - SAR Radiometric Calibration: Amplitude -> Decibels (dB = 10 * log10(I))                            |
|  - Adaptive Speckle Filtering: Multi-look Lee Filter & Gamma MAP Filter (ENL=4.4)                     |
|  - Dielectric Soil Moisture Outlier Mitigation: Temporal Hampel Filter (3 * MAD)                      |
|  - Cloud-Optimized GeoTIFF (COG) Windowed Reader & PySTAC Catalog Generator (EO, SAR, Proj)           |
+---------------------------------------------------+---------------------------------------------------+
                                                    |
                                                    v
+-------------------------------------------------------------------------------------------------------+
|                                  2. DEEP NEURAL ARCHITECTURES & PEFT                                  |
|  - Siamese Bitemporal Change Detector: Multi-scale differential feature fusion (|T2 - T1|)            |
|  - U-Net Backbone: ResNet Residual blocks with Vision Transformer (ViT) global attention bottleneck  |
|  - Prithvi-CAFE Hybrid Cross-Attention: Bidirectional Optical <-> SAR attention with cloud gating    |
|  - LoRA (Low-Rank Adaptation): Freezes base backbone, adapts W_q and W_v projections (rank=8, alpha=16) |
+---------------------------------------------------+---------------------------------------------------+
                                                    |
                                                    v
+-------------------------------------------------------------------------------------------------------+
|                                3. LOSS FUNCTIONS & SPATIAL TRAINING                                   |
|  - Compound Deforestation Loss: Binary Focal Loss (gamma=2.0) + Focal Tversky Loss (alpha=0.75)       |
|  - Asymmetric Tversky Index: Penalizes False Negatives (alpha > beta) to maximize deforestation recall|
|  - Spatial Block Cross-Validation: Spatial grid partitioning with buffer margins (TorchGeo style)    |
|  - TerraTorch / Lightning-style Trainer: Gradient clipping, CosineAnnealingLR, early stopping, IoU   |
+---------------------------------------------------+---------------------------------------------------+
                                                    |
                                                    v
+-------------------------------------------------------------------------------------------------------+
|                                4. BASELINES & POST-PROCESSING ENGINE                                  |
|  - Classical Baselines: CCDC (Continuous Change Detection & Classification), BFAST-style OLS-MOSUM    |
|  - Bitemporal NDVI Differencing with post-canopy thresholding                                         |
|  - Raster-to-Vector Polygonizer: 8-connected component clustering                                    |
|  - Minimum Mapping Unit (MMU) Filtering: Suppresses noise below 0.1 - 0.2 hectares                    |
|  - Douglas-Peucker Simplification: Geodesic polygon simplification preserving topological closure     |
|  - GeoJSON MultiPolygon Export: Computes spherical excess geodesic area (WGS84 ellipsoid) in ha       |
+---------------------------------------------------+---------------------------------------------------+
                                                    |
                                                    v
+-------------------------------------------------------------------------------------------------------+
|                              5. REGULATORY COMPLIANCE & STATISTICAL AUDIT                             |
|  - EUDR (EU 2023/1115) Mandates:                                                                      |
|      1. Regulatory Cutoff Date: December 31, 2020 (post-2020 disturbances trigger violations)         |
|      2. Geolocation Rules: Point coordinates permitted for <=4 ha; closed continuous polygons for >4ha|
|      3. Decimal Precision Verification: Mandates >= 6 decimal places (~0.1m ground precision)         |
|      4. Annex I Commodity Verification: Cattle, Cocoa, Coffee, Oil Palm, Rubber, Soya, Wood           |
|  - Due Diligence Statement (DDS) Auditor: Spatial intersection of production plots with AI alerts     |
|  - Olofsson et al. (2014) Area-Adjusted Accuracy Assessment: Stratified random sampling error matrix,  |
|    unbiased true deforestation area estimation, and 95% Confidence Intervals (Z=1.96)                 |
+---------------------------------------------------+---------------------------------------------------+
                                                    |
                                                    v
+-------------------------------------------------------------------------------------------------------+
|                               6. PRODUCTION REST API & WEBHOOK ALERTING                               |
|  - FastAPI Microservice: Endpoints for /detect, /eudr/audit, /stac/search, /webhooks                  |
|  - HMAC-SHA256 Signed Webhook Dispatcher: Real-time alert notifications to environmental authorities  |
+-------------------------------------------------------------------------------------------------------+
```

---

## Installation

```bash
git clone https://github.com/safiydev/geofm-deforestation.git
cd geofm-deforestation
pip install -e .
```

### Core Dependencies
- Python >= 3.9
- PyTorch >= 2.0
- NumPy >= 1.22
- FastAPI >= 0.100
- PySTAC >= 1.8
- Requests >= 2.28

---

## Quickstart

### 1. Run the Full End-to-End Pipeline Demonstration
```bash
python demo.py
```

### 2. Train / Fine-Tune with LoRA PEFT
```bash
python geofm/train_cli.py \
    --model-type siamese \
    --use-lora \
    --lora-rank 8 \
    --loss compound \
    --epochs 5 \
    --batch-size 4
```

### 3. Launch the Production FastAPI Service
```bash
uvicorn geofm.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Launch the Interactive Web Studio & EUDR Suite (Proof-of-Concept)
> [!IMPORTANT]
> **Proof-of-Concept Notice**: The interactive Web Studio (`studio.py`) is strictly an **experimental, untested proof-of-concept**. **If anyone wants to try out, test, or evaluate this project, running the Google Colab notebooks is strongly recommended.**
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safiydev/geofm-deforestation/blob/main/notebooks/geofm_training.ipynb)

```bash
python studio.py --port 8501
# Open http://localhost:8501 in your browser (experimental PoC)
```

### 5. Launch the Standalone Model Loader & Inference Studio
Load custom PyTorch checkpoints (`.pt` / `.pth`), auto-inspect architecture and LoRA PEFT adapters, test with pre-cached chips or custom image pairs, and view real-time heatmaps with interactive thresholding:

```bash
python model_loader.py --port 8600
# Open http://localhost:8600 in your browser
```
- **Custom Model Loading**: Upload any PyTorch `.pt` / `.pth` checkpoint or select from local `./checkpoints/`. Auto-detects architecture (`siamese`, `unet`, `multimodal`), input channels, and LoRA rank.
- **Multiple Input Sources**: Ingest pre-cached satellite chips, upload custom bitemporal rasters (T1 + T2 in PNG/TIFF/NPZ), or generate synthetic canopy test scenes.
- **Live Output Visualizer**: Inspect T1 baseline, T2 observation, spectral delta shift, continuous probability heatmaps, and thresholded binary masks.
- **Dynamic Decision Slider**: Adjust detection threshold (0.05 to 0.95) with instant (<5ms) re-polygonization and deforestation area recalculation.
- **GIS Export**: 1-click download of polygonized Deforestation Alert GeoJSONs.

### 6. Run the Automated Test Suite (104 Tests)
```bash
pytest tests/ -v
```

### 6. Cloud GPU Foundation Model Training (Google Colab)
> **Primary Recommended Method**: **If you want to test, evaluate, or train this project, using the Google Colab notebook is strongly recommended.** 
> 
> Heavy planetary-scale foundation model training (5,000+ multi-spectral satellite chips) and checkpoint generation was executed on cloud GPU accelerators (NVIDIA A100 / T4) using our reproducible [Google Colab Training Notebook](notebooks/geofm_training.ipynb) with automated Earth Engine ingestion. 
> 
> The local Web Studio (`python studio.py --port 8501`) is solely an **experimental proof-of-concept and untested** interface for Bring-Your-Own-Data (BYOD) prototyping and EUDR compliance visualization.
> 
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safiydev/geofm-deforestation/blob/main/notebooks/geofm_training.ipynb)

---

## Detailed Module Documentation

### 1. Signal Preprocessing & Sensor Physics (`geofm/preprocessing/`)
- **Optical (`optical.py`)**:
  - Normalized Difference Vegetation Index: $\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + \epsilon}$
  - Enhanced Vegetation Index: $\text{EVI} = 2.5 \cdot \frac{\text{NIR} - \text{Red}}{\text{NIR} + 6 \cdot \text{Red} - 7.5 \cdot \text{Blue} + 1}$
  - Cloud & Shadow Mask: Identifies high visible reflectance clouds and projects dark shadow regions in NIR/SWIR.
- **SAR (`sar.py`)**:
  - Sentinel-1 C-band ($5.405\text{ GHz}, \lambda \approx 5.5\text{ cm}$) dual-polarization ($\text{VV}, \text{VH}$).
  - Intact tropical forest exhibits strong volume scattering in cross-polarized $\text{VH}$ ($\approx -13\text{ dB}$). Clear-cutting shifts the scattering mechanism to bare ground surface scattering, lowering $\text{VH}$ to $\le -21\text{ dB}$.
  - **Lee Filter**:
    $$\hat{I} = \bar{I} + W(I - \bar{I}), \quad W = \max\left(0, \frac{\sigma_I^2 - \sigma_v^2 \bar{I}^2}{\sigma_I^2}\right)$$
    where noise variance $\sigma_v^2 = 1/\text{ENL}$ ($\text{ENL} = 4.4$ for Sentinel-1 GRD IW).
  - **Temporal Hampel Outlier Filter**: Computes sliding temporal median and Median Absolute Deviation ($\text{MAD}$) to remove episodic false positives caused by heavy tropical rainfall / dielectric soil moisture changes.
- **COG Windowed Reader (`cog_reader.py`)**: Supports bounding box coordinate queries and `Window(col_off, row_off, width, height)` extraction with affine geotransforms.
- **STAC Engine (`stac.py`)**: Generates PySTAC items adhering to SpatioTemporal Asset Catalog specifications with `eo`, `sar`, and `projection` extensions.

### 2. Model Architectures & Parameter-Efficient Fine-Tuning (`geofm/models/`)
- **Siamese Change Detector (`siamese.py`)**: Dual-branch shared-weight encoder taking pre-disturbance ($T_1$) and post-disturbance ($T_2$) scenes, extracting multi-scale feature hierarchies $[f_{T_1}, f_{T_2}, |f_{T_2} - f_{T_1}|]$, decoded to binary disturbance masks.
- **Vision Transformer Bottleneck (`unet.py`)**: Self-attention bottleneck capturing long-range contextual spatial dependencies (e.g. tracking narrow linear logging corridors).
- **Hybrid Cross-Attention Fusion (`cross_attention.py`)**: Bidirectional cross-attention between Optical and SAR tokens with an adaptive cloud gate. When optical pixels are occluded by cloud cover, the gate suppresses optical features and relies on SAR structural backscatter.
- **LoRA Adapter (`lora.py`)**: Injects trainable rank-$r$ decomposition matrices into attention projection layers ($W_q, W_v$):
  $$W' = W_0 + \frac{\alpha}{r} (B \times A)$$
  Freezes over 99% of backbone parameters, enabling rapid domain adaptation to new forest biomes.

### 3. Loss Functions & Class Imbalance (`geofm/losses/`)
- **Binary Focal Loss (`focal.py`)**:
  $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  Down-weights vast expanses of stable forest background ($\gamma = 2.0$) to focus learning on rare deforestation pixels.
- **Tversky Loss (`tversky.py`)**:
  $$\text{TI} = \frac{\text{TP} + \epsilon}{\text{TP} + \alpha \text{FN} + \beta \text{FP} + \epsilon}$$
  Asymmetric tuning ($\alpha=0.75, \beta=0.25$) heavily penalizes False Negatives to maximize alert recall.
- **Compound Deforestation Loss (`focal_tversky.py`)**: Combines Focal Loss and Focal Tversky Loss.

### 4. Classical Time-Series Baselines (`geofm/baselines/`)
- **CCDC (`ccdc.py`)**: Fits multi-harmonic regression models ($y(t) = c_0 + a_1 \cos(\omega t) + b_1 \sin(\omega t)$) to historical observations. Confirms structural breaks when consecutive observations deviate beyond $3 \times \text{RMSE}$.
- **BFAST-style OLS-MOSUM (`bfast.py`)**: Computes moving sums of ordinary least squares residuals against a 95% critical boundary ($\lambda = 1.36$).
- **NDVI Differencing (`ndvi_delta.py`)**: Screenings for rapid canopy degradation ($\Delta\text{NDVI} < -0.25$).

### 5. Vector Polygonization & Geometry (`geofm/postprocessing/`)
- **Connected Components**: Groups 8-connected positive alert pixels.
- **Minimum Mapping Unit (MMU)**: Automatically eliminates spurious isolated noise patches below 0.10 - 0.20 hectares.
- **Douglas-Peucker Simplification (`douglas_peucker.py`)**: Simplifies polygon boundaries to reduce vertex density while guaranteeing ring closure.
- **Geodesic Spherical Area (`geometry_utils.py`)**: Computes precise polygon areas in hectares using WGS84 authalic sphere formulas.

### 6. Regulatory Compliance & Statistical Validation (`geofm/compliance/`)
- **EUDR Mandate Verification (`eudr.py`)**:
  - Baseline cutoff: December 31, 2020 (Regulation EU 2023/1115).
  - Geolocation format mandate: Plots $> 4\text{ ha}$ must be submitted as closed continuous polygons; point coordinates are restricted to plots $\le 4\text{ ha}$.
  - Coordinate precision: Validates $\ge 6$ decimal places.
- **Due Diligence Statement (DDS) Auditor (`dds_auditor.py`)**: Evaluates supply chain declarations against AI alert polygons, outputting `COMPLIANT`, `NON_COMPLIANT`, or `REJECTED` verdicts.
- **Olofsson et al. (2014) Accuracy Assessment (`olofsson.py`)**:
  - Area-weighted error adjustment matrix: $\hat{p}_{ij} = W_i \frac{n_{ij}}{n_{i\cdot}}$
  - Area-adjusted Overall Accuracy ($\hat{\text{OA}}$), Producer's Accuracy, and User's Accuracy.
  - Standard errors and 95% Confidence Intervals ($Z=1.96$).
  - Unbiased estimation of true deforestation area ($\hat{A}_{\text{deforest}} = A_{\text{tot}} \cdot \hat{p}_{\cdot \text{deforest}}$).

### 7. REST API & Webhooks (`geofm/api/`)
- `GET /health`: Health and device telemetry.
- `POST /api/v1/detect`: Runs model inference, returns GeoJSON alert polygons.
- `POST /api/v1/eudr/audit`: Audits DDS plot filings against EUDR mandates.
- `POST /api/v1/stac/search`: Queries STAC catalog metadata.
- `POST /api/v1/webhooks/register`: Subscribes external systems to alert streams with HMAC-SHA256 signature verification.

### 8. Interactive Web Training Studio & EUDR Suite (`studio.py` / `geofm/studio/`)
High-performance, dark glassmorphic control room dashboard engineered to completely replace Google Colab workflows:
- **Tab 1: AI Model Training Studio**: Select architecture (Siamese Change Detector, Multimodal Sentinel-1/2, UNet-ResNet), configure LoRA PEFT ($r=2..64, \alpha=4..128$), select class-imbalanced compound loss, tune learning rate/epochs/batch size, trigger background asynchronous training runs with live dual-axis Chart.js curves (Loss vs IoU), and stream real-time monospace terminal logs.
- **Tab 2: Model Testing & Benchmarking**: Evaluate validation test batches with in-memory weights or saved checkpoints, render $2 \times 2$ pixel-level Confusion Matrix scorecards (True Positives, False Positives, False Negatives, True Negatives), and compute full threshold sensitivity trade-off sweep curves ($0.1 \le \tau \le 0.9$).
- **Tab 3: Single-Chip Deep Inspector**: Inspect all 9 multi-spectral layers (T1/T2 RGB, CIR, NDVI, SAR Radar Composite, Hansen GT Mask, AI Deforestation Probability Map, and Pixel Error Confusion Diff Map: Green=TP, Orange=FP, Red=FN, Slate=TN).
- **Tab 4: Geospatial Alert Map**: Full interactive Leaflet map with CartoDB Dark Matter satellite basemap, Minimum Mapping Unit (MMU) filtering ($0.02 - 1.0\text{ ha}$), severity-colored vector polygons (CRITICAL, HIGH, MEDIUM), and GeoJSON alert download.
- **Tab 5: EUDR Due Diligence & Regulatory Compliance**: Upload or test farm plots against legal mandates under Regulation (EU) 2023/1115 (Dec 31, 2020 cutoff, $\le 4\,\text{ha}$ point vs $> 4\,\text{ha}$ polygon, $\ge 6$ decimal precision, Annex I commodities), compute Olofsson et al. (2014) area-adjusted accuracy with 95% confidence intervals, and generate compliance audit verdicts.
- **Tab 6: Checkpoints & Dataset Zoo**: Browse saved `.pt` weights, direct browser file downloads via `/studio/api/download/{filename}`, hot-swap active inference backbones, and generate balanced 70:30 synthetic chips.

#### Launching the Studio:
```bash
# Standalone high-performance web dashboard (FastAPI + Leaflet + Chart.js):
python studio.py --port 8501 --host 0.0.0.0

# Open http://localhost:8501 in your browser
```

---

### 9. Catalog Sanitation & STAC Auto-Healing (`geofm/preprocessing/catalog_validator.py`)
- **HTTP HEAD Pre-flight Verification**: Rapidly tests remote COG asset existence without downloading full arrays.
- **S3 Granule Syntax Auto-Healing**: Detects malformed paths (e.g. missing MGRS tile or date offsets like `S2A_20231124`) and automatically queries the Earth Search STAC catalog to resolve the nearest valid satellite pass with working COG URLs.
- **Offline NPZ Chip Caching (`geofm/training/dataset.py`)**: `cache_dataset_to_npz()` and `load_dataset_from_npz()` serialize satellite chips to local storage, decoupling GPU training from remote HTTP/S3 network failures.

---

## Verification & Testing Suite

All 87 unit and integration tests execute with a 100% pass rate:
```bash
pytest tests/ -v
```
Output:
```
======================= 87 passed in 46.22s ========================
```
