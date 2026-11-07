# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

1. **Remote Sensing Scientists & Geospatial ML Engineers**: Specialists training, fine-tuning (LoRA PEFT), and benchmarking foundation change detection models across bitemporal optical (Sentinel-2) and synthetic aperture radar (Sentinel-1 SAR) constellations.
2. **Sustainability Officers & Commodity Supply-Chain Analysts**: Corporate and agricultural auditors assessing production plots (cattle, cocoa, coffee, oil palm, rubber, soya, wood) for deforestation-free verification under EUDR (Regulation EU 2023/1115), requiring one-click polygon ingestion and automated compliance reporting without coding.
3. **Conservation Organizations & Environmental Monitors**: Field analysts investigating canopy clearings, logging roads, and illegal forest encroachments across tropical frontier hotspots (Amazon, Congo Basin, Borneo, Cerrado, Gran Chaco).

## Product Purpose

GeoFM detects physical bitemporal forest canopy disturbance and subtle deforestation changes between two satellite acquisitions ($T_1$ pre-disturbance baseline vs. $T_2$ post-disturbance observation). It exists to replace slow, error-prone manual GIS analysis with automated deep foundation model inference and legally defensible EUDR Due Diligence Statements (DDS).

**Success Metric**: Operator simplicity — enabling non-GIS analysts to upload boundary polygons or satellite chips and instantly receive pixel-accurate visual overlays, verified alert vectors, and statutory compliance ratings with 95% confidence intervals.

## Positioning

Unlike generic computer vision frameworks or speculative risk-scoring tools that attempt to guess future deforestation, GeoFM focuses strictly on empirical change detection:
- Ingests true multi-spectral Sentinel-2 bands paired with speckle-filtered Sentinel-1 dual-polarization SAR to penetrate cloud cover.
- Validates detections against reference ground truth from Hansen Global Forest Change (UMD GFC 2024).
- Directly enforces EU Regulation 2023/1115 statutory rules (December 31, 2020 cutoff date; 4-hectare polygon threshold; point-to-polygon validation).
- Quantifies land change using Olofsson et al. (2014) stratified area-adjusted error matrices with 95% confidence intervals.

## Operating Context

- **Primary Evaluation & Testing**: **Google Colab Notebooks are strongly recommended** for testing, evaluating, and training this project with dedicated cloud GPU acceleration (`notebooks/geofm_training.ipynb`).
- **Web Studio Status**: The interactive Web Studio (`http://127.0.0.1:8501`) is solely an **experimental proof-of-concept and untested**; if anyone wants to evaluate or test this project, the Google Colab notebooks are strongly recommended.
- **Data Ingestion**: Hybrid architecture supporting direct streaming from Google Earth Engine, custom local folder scanning, browser drag-and-drop ZIP ingest, and zero-RAM synthetic streaming.
- **Output Formats**: Interactive Leaflet maps, Chart.js loss/IoU convergence curves, GeoJSON alert vectors, audit PDF/JSON statements, and PNG spectral overlays.

## Capabilities and Constraints

### Capabilities
- **16-Parameter Planetary Training Suite**: Configurable architecture (`siamese`, `multimodal`, `unet`), dataset source (`gee`, `real`, `cached`, `synthetic`), scale (`continental`, `high_fidelity`, `standard`, `quick`), chip dimensions, loss functions (`compound`, `focal_tversky`, `tversky`, `focal`, `bce`), LoRA rank/alpha, 70:30 class balance quotas, and mixed precision AMP.
- **Interactive Satellite Chip & Mask Viewer**: Dynamic opacity slider (0% to 100%) superimposing translucent red deforestation masks onto true-color $T_2$ optical imagery.
- **12-Layer Spectral Gallery**: True Color RGB ($T_1$, $T_2$), Color Infrared CIR ($T_1$, $T_2$), NDVI vegetation indices ($T_1$, $T_2$), Sentinel-1 SAR composite, Hansen GFC mask, AI detection probability heatmap, binary change mask, and confusion difference map.
- **Model Checkpoint Management**: Storage zoo dropdown and manual text/browser file loader for any local `.pt` or `.pth` weights file.
- **Standalone Model Loader & Inference Studio**: Lightweight local web application (`python model_loader.py --port 8600`) to load custom PyTorch checkpoints (`.pt`), auto-inspect LoRA adapters/weights, test on pre-cached chips or custom image pairs, and view real-time probability heatmaps with instant threshold sliders.
- **Automated Geometry Extraction**: Douglas-Peucker polygon simplification with Minimum Mapping Unit (MMU) area thresholds.
- **EUDR Regulatory Suite**: Automated Due Diligence Statement validation, cutoff date checking, risk scoring, and Olofsson area-adjusted verification.

### Constraints
- Strictly bitemporal change detection (comparing past vs. recent physical state), never future prediction.
- Pure zero-dependency client architecture (vanilla JavaScript, Leaflet, and Chart.js without heavyweight npm toolchains).
- Standard-library zero-dependency fallbacks for image encoding (pure Python DEFLATE/PNG encoder) to guarantee operation without Pillow or native C libraries.

## Brand Commitments

- **Name**: GeoFM (Geospatial Foundation Model for Deforestation Detection & EUDR Compliance).
- **Voice & Tone**: Rigorous remote sensing precision, authoritative regulatory diligence, and accessible operator clarity.
- **Color Identity**:
  - Deep Obsidian (`#070B12`, `#0F172A`): Deep space and satellite backdrop.
  - Emerald Green (`#10B981`): Intact canopy, compliant status, high vegetation vigor.
  - Hot Crimson (`#EF4444`): Canopy disturbance alerts, deforestation violations, high loss.
  - Electric Cyan (`#06B6D4`): Satellite telemetry, optical spectral layers, primary actions.
  - Amber Gold (`#F59E0B`): Moderate risk, cloud warnings, validation losses.

## Evidence on Hand

- Live integration with Google Earth Engine (`COPERNICUS/S2_SR_HARMONIZED` and `UMD/hansen/global_forest_change_2024_v1_12`).
- Benchmark evaluation on real Sentinel-2 bi-temporal datasets (OSCD benchmark).
- 94 passed unit and integration tests across data sources, neural networks, loss functions, polygonization, and web APIs.
- Formal implementation of the Olofsson et al. (2014) Good Practices for Estimating Area and Assessing Accuracy equation set.

## Product Principles

1. **Empirical Ground Truth**: Detect actual physical canopy alterations between two satellite dates; never fabricate predictive claims without observable pixel differential.
2. **Operator Simplicity**: Enable users without GIS or programming experience to upload boundaries, inspect translucent overlays, and produce actionable compliance verdicts in one click.
3. **Statutory Defensibility**: Ensure every polygon alert is accompanied by area metrics in hectares, severity classifications, and confidence bounds compliant with EUDR audits.
4. **Cross-Platform Resilience**: Run reliably across consumer Windows PCs, Linux servers, and macOS laptops with dynamic hardware detection and automatic band-channel alignment.

## Accessibility & Inclusion

- Dark theme optimized for prolonged geospatial inspection with WCAG AAA contrast ratios.
- Multi-modal status representation: color coding is paired with clear textual labels and distinct badge icons so state is never conveyed by color alone.
- Scalable typography with monospace formatting for coordinates, bounding boxes, and hectare metrics.
