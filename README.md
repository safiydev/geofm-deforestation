# GeoFM: Geospatial AI for Early Deforestation Detection & EUDR Compliance

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, multimodal Geospatial Artificial Intelligence system engineered for near-real-time tropical deforestation early warning, Vision Transformer (ViT) change detection, Parameter-Efficient Fine-Tuning (PEFT/LoRA), and European Union Deforestation Regulation (EUDR, Regulation (EU) 2023/1115) compliance auditing.

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
git clone https://github.com/example/geofm-deforestation.git
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
Self-contained interactive visual suite with dual runtime support (Streamlit & Standalone FastAPI + Leaflet.js + Chart.js):
- **Tab 1: AI Model Training Studio**: Select architecture (Siamese Change Detector, U-Net ViT Hybrid, Multimodal Cross-Attention), configure LoRA PEFT ($r=2..64, \alpha=4..128$), select class-imbalanced compound loss, tune learning rate/epochs, trigger training runs with real-time loss & IoU curves, and inspect saved checkpoint metadata.
- **Tab 2: Satellite Inference & Interactive Map**: Generate bitemporal satellite scenes ($T_1$ vs $T_2$, Sentinel-2 RGB/NIR, NDVI, Sentinel-1 SAR VV/VH), run neural inference with configurable probability threshold and Minimum Mapping Unit (MMU ha), and plot detected deforestation clusters on interactive Leaflet maps with severity levels (Critical, High, Medium).
- **Tab 3: EUDR Due Diligence & Regulatory Compliance**: Upload or test farm plots against legal mandates under Regulation (EU) 2023/1115 (Dec 31, 2020 cutoff, $\le 4\,\text{ha}$ point vs $> 4\,\text{ha}$ polygon, $\ge 6$ decimal precision, Annex I commodities), intersect with AI alerts, compute Olofsson et al. (2014) area-adjusted accuracy with 95% confidence intervals, and generate compliance audit verdicts.
- **Tab 4: STAC Satellite Catalog Explorer**: Query Sentinel-2 and Sentinel-1 STAC items across Amazon, Congo, and Borneo biomes by bounding box, date range, and cloud cover.

#### Launching the Studio:
```bash
# Standalone high-performance web dashboard (FastAPI + Leaflet + Chart.js):
python studio.py --port 8501

# Or with Streamlit (if installed):
streamlit run studio.py
```

---

## Verification & Testing Suite

All 62 unit and integration tests execute with a 100% pass rate:
```bash
pytest tests/ -v
```
Output:
```
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_detect_endpoint PASSED
tests/test_api.py::test_eudr_audit_endpoint PASSED
tests/test_api.py::test_stac_search_endpoint PASSED
tests/test_api.py::test_webhook_register_and_dispatch PASSED
tests/test_api.py::test_detect_endpoint_multipolygon PASSED
tests/test_api.py::test_stac_search_endpoint_with_naive_dates PASSED
tests/test_api.py::test_eudr_audit_endpoint_unregulated_commodity PASSED
tests/test_baselines.py::test_ccdc_detects_abrupt_step_deforestation PASSED
tests/test_baselines.py::test_bfast_ols_mosum_monitoring PASSED
tests/test_baselines.py::test_ndvi_delta_baseline PASSED
tests/test_compliance.py::test_eudr_validator_rules PASSED
tests/test_compliance.py::test_dds_auditor_compliance_check PASSED
tests/test_compliance.py::test_olofsson_accuracy_assessment PASSED
tests/test_compliance.py::test_eudr_commodity_scope_validation PASSED
tests/test_compliance.py::test_dds_auditor_multipolygon_alert_overlap PASSED
tests/test_compliance.py::test_dds_auditor_unregulated_commodity_rejection PASSED
tests/test_losses.py::test_binary_focal_loss PASSED
tests/test_losses.py::test_tversky_loss_asymmetry PASSED
tests/test_losses.py::test_compound_loss_backward PASSED
tests/test_models.py::test_unet_forward_and_shapes PASSED
tests/test_models.py::test_siamese_change_detector PASSED
tests/test_models.py::test_cross_attention_multimodal_fusion PASSED
tests/test_models.py::test_lora_adaptation_and_merge PASSED
tests/test_postprocessing.py::test_polygon_area_and_point_in_polygon PASSED
tests/test_postprocessing.py::test_polygon_intersection PASSED
tests/test_postprocessing.py::test_douglas_peucker_simplification PASSED
tests/test_postprocessing.py::test_polygonizer_mmu_filter PASSED
tests/test_postprocessing.py::test_polygon_intersection_crossing_without_vertices_inside PASSED
tests/test_postprocessing.py::test_is_point_in_polygon_unclosed_triangle PASSED
tests/test_postprocessing.py::test_closed_polygon_simplification_all_collinear_edges PASSED
tests/test_postprocessing.py::test_polygonizer_l_shape_and_multipolygon PASSED
tests/test_preprocessing.py::test_optical_normalization_and_indices PASSED
tests/test_preprocessing.py::test_optical_cloud_and_shadow_masking PASSED
tests/test_preprocessing.py::test_sar_decibels_and_speckle_filters PASSED
tests/test_preprocessing.py::test_sar_hampel_filter PASSED
tests/test_preprocessing.py::test_cog_window_reader_and_chipping PASSED
tests/test_preprocessing.py::test_stac_manager PASSED
tests/test_studio.py::test_image_rendering_utilities PASSED
tests/test_studio.py::test_training_logger_and_json_export PASSED
tests/test_studio.py::test_studio_inference_manager PASSED
tests/test_studio.py::test_studio_compliance_manager PASSED
tests/test_studio.py::test_studio_stac_manager PASSED
tests/test_studio.py::test_checkpoint_manager PASSED
tests/test_studio.py::test_streamlit_app_graceful_run PASSED
tests/test_studio.py::test_web_studio_get_index PASSED
tests/test_studio.py::test_web_studio_api_infer PASSED
tests/test_studio.py::test_web_studio_api_eudr_presets PASSED
tests/test_studio.py::test_web_studio_api_eudr_audit PASSED
tests/test_studio.py::test_web_studio_api_olofsson PASSED
tests/test_studio.py::test_web_studio_api_stac_search PASSED
tests/test_studio.py::test_web_studio_api_checkpoints PASSED
tests/test_studio.py::test_web_studio_api_train PASSED
tests/test_studio.py::test_web_studio_api_train_status PASSED
tests/test_studio.py::test_trainer_learning_rate_and_csv_export PASSED
tests/test_studio.py::test_trainer_fit_zero_epochs PASSED
tests/test_studio.py::test_compliance_manager_with_geojson_feature_collections PASSED
tests/test_studio.py::test_studio_training_multimodal_and_unet PASSED
tests/test_training.py::test_spatial_block_cross_validation_split PASSED
tests/test_training.py::test_trainer_type_hints PASSED
tests/test_training.py::test_deforestation_metrics_computation PASSED
tests/test_training.py::test_trainer_single_epoch_run PASSED
======================= 62 passed in 13.65s ========================
```
