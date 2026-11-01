"""Standalone FastAPI Interactive Web Studio for GeoFM Deforestation Detection & EUDR Suite.

Serves an interactive HTML5/CSS/JavaScript dashboard powered by Leaflet.js maps
and Chart.js analytics when Streamlit is not installed, or when running headless/server mode.
"""

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from geofm.compliance.eudr import EUDR_REGULATED_COMMODITIES
from geofm.studio.core import (
    CheckpointManager,
    StudioComplianceManager,
    StudioInferenceManager,
    StudioSTACManager,
    StudioTrainingManager,
)


# =============================================================================
# Request / Response Schemas
# =============================================================================

class TrainRequest(BaseModel):
    architecture: str = Field(default="siamese", description="siamese, unet, or multimodal")
    use_lora: bool = Field(default=True)
    lora_rank: int = Field(default=8)
    lora_alpha: float = Field(default=16.0)
    loss_type: str = Field(default="compound")
    learning_rate: float = Field(default=1e-3)
    epochs: int = Field(default=3)
    batch_size: int = Field(default=4)
    num_samples: int = Field(default=16)


class InferRequest(BaseModel):
    deforestation_fraction: float = Field(default=0.08)
    cloud_fraction: float = Field(default=0.15)
    model_type: str = Field(default="siamese")
    threshold: float = Field(default=0.5)
    mmu_ha: float = Field(default=0.10)
    west: float = Field(default=-60.05)
    north: float = Field(default=-3.05)


class AUDITRequest(BaseModel):
    statement_id: str
    operator_name: str
    commodity: str
    country: str
    plots: List[Dict[str, Any]]
    alerts: Optional[List[Dict[str, Any]]] = None


class STACQueryRequest(BaseModel):
    bbox: List[float] = Field(default=[-60.5, -3.5, -59.5, -2.5])
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    platform: Optional[str] = None
    max_cloud_cover: Optional[float] = None


# =============================================================================
# HTML / CSS / JS Single Page App Template
# =============================================================================

STUDIO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GeoFM Web Training Studio & EUDR Compliance Suite</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --primary: #1b4332;
      --primary-light: #2d6a4f;
      --accent: #52b788;
      --bg: #f8f9fa;
      --card-bg: #ffffff;
      --text: #212529;
      --text-muted: #6c757d;
      --border: #dee2e6;
      --success: #2d6a4f;
      --danger: #d90429;
      --warning: #f77f00;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
    header { background: linear-gradient(135deg, #1b4332 0%, #081c15 100%); color: white; padding: 1.2rem 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.15); display: flex; justify-content: space-between; align-items: center; }
    .brand { display: flex; align-items: center; gap: 0.8rem; }
    .brand h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.5px; }
    .brand p { font-size: 0.85rem; color: #b7e4c7; }
    .header-badges { display: flex; gap: 0.5rem; }
    .badge { font-size: 0.75rem; padding: 4px 10px; border-radius: 12px; font-weight: 600; text-transform: uppercase; }
    .badge-primary { background: #52b788; color: #081c15; }
    .badge-secondary { background: rgba(255,255,255,0.2); color: white; }

    nav.tabs-nav { background: white; border-bottom: 1px solid var(--border); display: flex; padding: 0 2rem; gap: 1rem; overflow-x: auto; }
    .tab-btn { background: none; border: none; padding: 1rem 1.2rem; font-size: 0.95rem; font-weight: 600; color: var(--text-muted); cursor: pointer; border-bottom: 3px solid transparent; transition: all 0.2s ease; display: flex; align-items: center; gap: 0.5rem; white-space: nowrap; }
    .tab-btn:hover { color: var(--primary-light); }
    .tab-btn.active { color: var(--primary); border-bottom-color: var(--primary); }

    main { flex: 1; padding: 2rem; max-width: 1600px; margin: 0 auto; width: 100%; }
    .tab-content { display: none; }
    .tab-content.active { display: block; animation: fadeIn 0.25s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

    .grid-2 { display: grid; grid-template-columns: 360px 1fr; gap: 1.5rem; }
    @media (max-width: 1024px) { .grid-2 { grid-template-columns: 1fr; } }

    .card { background: white; border: 1px solid var(--border); border-radius: 10px; padding: 1.5rem; box-shadow: 0 2px 6px rgba(0,0,0,0.04); margin-bottom: 1.5rem; }
    .card-title { font-size: 1.15rem; font-weight: 700; color: var(--primary); margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; }

    .form-group { margin-bottom: 1rem; }
    .form-group label { display: block; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.35rem; color: #495057; }
    .form-control, select, textarea { width: 100%; padding: 0.55rem 0.75rem; border: 1px solid var(--border); border-radius: 6px; font-size: 0.9rem; transition: border-color 0.15s; }
    .form-control:focus, select:focus, textarea:focus { border-color: var(--accent); outline: none; }
    .checkbox-group { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; }

    .btn { display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem; padding: 0.65rem 1.25rem; font-size: 0.9rem; font-weight: 600; border-radius: 6px; border: none; cursor: pointer; transition: all 0.2s; }
    .btn-primary { background: var(--primary); color: white; width: 100%; }
    .btn-primary:hover { background: var(--primary-light); }
    .btn-secondary { background: #e9ecef; color: var(--text); }
    .btn-secondary:hover { background: #dee2e6; }

    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
    .metric-box { background: #f8f9fa; border-left: 4px solid var(--accent); border-radius: 6px; padding: 1rem; }
    .metric-label { font-size: 0.8rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: var(--primary); margin-top: 0.25rem; }

    .images-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
    .image-card { border: 1px solid var(--border); border-radius: 6px; overflow: hidden; background: white; text-align: center; }
    .image-card img { width: 100%; height: 180px; object-fit: cover; display: block; }
    .image-card .caption { padding: 0.5rem; font-size: 0.8rem; font-weight: 600; color: #495057; background: #f8f9fa; }

    #leafletMap { height: 420px; width: 100%; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 1.5rem; z-index: 1; }

    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th, td { padding: 0.65rem 0.85rem; text-align: left; border-bottom: 1px solid var(--border); }
    th { background: #f8f9fa; font-weight: 600; color: var(--primary); }
    tr:hover td { background: #f8f9fa; }

    .badge-status { padding: 4px 10px; border-radius: 12px; font-weight: 700; font-size: 0.8rem; }
    .badge-status.COMPLIANT { background: #d8f3dc; color: #1b4332; }
    .badge-status.NON_COMPLIANT { background: #ffe5d9; color: #d90429; }
    .badge-status.REJECTED { background: #fff3cd; color: #856404; }

    .progress-bar-container { width: 100%; background-color: #e9ecef; border-radius: 4px; height: 8px; margin: 0.5rem 0 1rem; overflow: hidden; display: none; }
    .progress-bar { height: 100%; background-color: var(--accent); width: 0%; transition: width 0.3s; }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <span style="font-size: 2rem;">🛰️</span>
      <div>
        <h1>GeoFM Web Training Studio & EUDR Suite</h1>
        <p>Interactive Foundation Model Fine-Tuning & Satellite Compliance Engine</p>
      </div>
    </div>
    <div class="header-badges">
      <span class="badge badge-primary">FastAPI Engine</span>
      <span class="badge badge-secondary">PyTorch 2.x</span>
      <span class="badge badge-secondary">EUDR 2023/1115</span>
    </div>
  </header>

  <nav class="tabs-nav">
    <button class="tab-btn active" onclick="switchTab('tab1', this)">🏋️ 1. AI Training Studio</button>
    <button class="tab-btn" onclick="switchTab('tab2', this)">🛰️ 2. Satellite Inference & Map</button>
    <button class="tab-btn" onclick="switchTab('tab3', this)">⚖️ 3. EUDR Due Diligence Audit</button>
    <button class="tab-btn" onclick="switchTab('tab4', this)">🗂️ 4. STAC Catalog Explorer</button>
  </nav>

  <main>
    <!-- TAB 1: AI MODEL TRAINING STUDIO -->
    <div id="tab1" class="tab-content active">
      <div class="grid-2">
        <div>
          <div class="card">
            <h3 class="card-title">Training Configuration</h3>
            <div class="form-group">
              <label>Model Architecture</label>
              <select id="trainArch">
                <option value="siamese">Siamese Bitemporal Change Detector</option>
                <option value="unet">U-Net ViT Hybrid (ResNet + Transformer)</option>
                <option value="multimodal">Multimodal Cross-Attention (Optical + SAR)</option>
              </select>
            </div>
            <div class="checkbox-group">
              <input type="checkbox" id="trainUseLora" checked>
              <label for="trainUseLora" style="margin:0; font-weight:600;">Inject LoRA PEFT Adapters</label>
            </div>
            <div style="display:flex; gap:0.5rem;">
              <div class="form-group" style="flex:1;">
                <label>LoRA Rank (r)</label>
                <input type="number" id="trainLoraRank" class="form-control" value="8" min="2" max="64">
              </div>
              <div class="form-group" style="flex:1;">
                <label>LoRA Alpha (α)</label>
                <input type="number" id="trainLoraAlpha" class="form-control" value="16.0" min="4" max="128">
              </div>
            </div>
            <div class="form-group">
              <label>Loss Function</label>
              <select id="trainLoss">
                <option value="compound">Compound Loss (Focal + Tversky)</option>
                <option value="focal_tversky">Focal Tversky Loss (Asymmetric)</option>
                <option value="focal">Binary Focal Loss (Hard Negatives)</option>
                <option value="tversky">Tversky Loss</option>
              </select>
            </div>
            <div style="display:flex; gap:0.5rem;">
              <div class="form-group" style="flex:1;">
                <label>Epochs</label>
                <input type="number" id="trainEpochs" class="form-control" value="3" min="1" max="20">
              </div>
              <div class="form-group" style="flex:1;">
                <label>Batch Size</label>
                <select id="trainBatch">
                  <option value="2">2</option>
                  <option value="4" selected>4</option>
                  <option value="8">8</option>
                </select>
              </div>
            </div>
            <div class="form-group">
              <label>Learning Rate</label>
              <select id="trainLr">
                <option value="0.0001">1e-4</option>
                <option value="0.0005">5e-4</option>
                <option value="0.001" selected>1e-3</option>
                <option value="0.005">5e-3</option>
              </select>
            </div>
            <button class="btn btn-primary" id="btnStartTrain" onclick="startTraining()">🚀 Start Training Run</button>
            <div class="progress-bar-container" id="trainProgBar">
              <div class="progress-bar" id="trainProgFill"></div>
            </div>
          </div>
        </div>

        <div>
          <div class="card">
            <h3 class="card-title">Live Loss & Validation IoU Curves</h3>
            <div style="height: 280px; position: relative;">
              <canvas id="trainingChart"></canvas>
            </div>
          </div>

          <div class="card">
            <h3 class="card-title">Epoch History & Confusion Matrix</h3>
            <div style="overflow-x: auto;">
              <table id="epochTable">
                <thead>
                  <tr>
                    <th>Epoch</th><th>Train Loss</th><th>Val Loss</th><th>Val IoU</th><th>Val F1</th><th>Precision</th><th>Recall</th><th>TP / FP / FN</th>
                  </tr>
                </thead>
                <tbody id="epochTableBody">
                  <tr><td colspan="8" style="text-align:center; color:#6c757d;">No training runs executed yet.</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          <div class="card">
            <h3 class="card-title">Saved Model Checkpoints</h3>
            <div style="overflow-x: auto;">
              <table id="checkpointTable">
                <thead>
                  <tr>
                    <th>Checkpoint File</th><th>Epoch</th><th>Best IoU</th><th>Size</th><th>Last Modified</th>
                  </tr>
                </thead>
                <tbody id="checkpointBody">
                  <tr><td colspan="5" style="text-align:center;">Loading checkpoints...</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: SATELLITE INFERENCE & MAP -->
    <div id="tab2" class="tab-content">
      <div class="grid-2">
        <div>
          <div class="card">
            <h3 class="card-title">Inference Pipeline Settings</h3>
            <div class="form-group">
              <label>Deforestation Extent (Fraction)</label>
              <input type="range" id="inferDefFrac" class="form-control" min="0.02" max="0.25" step="0.02" value="0.08" oninput="document.getElementById('lblDefFrac').innerText = this.value">
              <span id="lblDefFrac" style="font-size:0.8rem; color:var(--text-muted);">0.08</span>
            </div>
            <div class="form-group">
              <label>Cloud Occlusion (Fraction)</label>
              <input type="range" id="inferCloudFrac" class="form-control" min="0.0" max="0.3" step="0.05" value="0.15" oninput="document.getElementById('lblCloudFrac').innerText = this.value">
              <span id="lblCloudFrac" style="font-size:0.8rem; color:var(--text-muted);">0.15</span>
            </div>
            <div class="form-group">
              <label>Model Architecture</label>
              <select id="inferModel">
                <option value="siamese" selected>Siamese Bitemporal Network</option>
                <option value="multimodal">Multimodal Cross-Attention (Opt+SAR)</option>
                <option value="unet">U-Net ViT Disturbance</option>
                <option value="ndvi_delta">Baseline: Bitemporal ΔNDVI</option>
              </select>
            </div>
            <div class="form-group">
              <label>Probability Threshold</label>
              <input type="number" id="inferThreshold" class="form-control" value="0.5" min="0.1" max="0.9" step="0.05">
            </div>
            <div class="form-group">
              <label>Minimum Mapping Unit (MMU in ha)</label>
              <input type="number" id="inferMmu" class="form-control" value="0.10" min="0.05" max="1.0" step="0.05">
            </div>
            <button class="btn btn-primary" onclick="runInference()">🛰️ Run Satellite Inference</button>
          </div>
        </div>

        <div>
          <div class="metric-grid">
            <div class="metric-box">
              <div class="metric-label">Alert Clusters</div>
              <div class="metric-val" id="metricAlerts">0</div>
            </div>
            <div class="metric-box">
              <div class="metric-label">Deforested Area</div>
              <div class="metric-val" id="metricDefHa">0.00 ha</div>
            </div>
            <div class="metric-box">
              <div class="metric-label">Inference Latency</div>
              <div class="metric-val" id="metricLatency">0.0 ms</div>
            </div>
            <div class="metric-box">
              <div class="metric-label">Cloud Occlusion</div>
              <div class="metric-val" id="metricCloud">0.0%</div>
            </div>
          </div>

          <div class="card">
            <h3 class="card-title">Interactive Geospatial Alert Map</h3>
            <div id="leafletMap"></div>
          </div>

          <div class="card">
            <h3 class="card-title">Multi-Spectral Bands & Feature Layers</h3>
            <div class="images-grid" id="layerGrid">
              <div class="image-card"><div style="height:180px; background:#eee; display:flex; align-items:center; justify-content:center;">Click Run Inference</div><div class="caption">Baseline T1 RGB</div></div>
              <div class="image-card"><div style="height:180px; background:#eee; display:flex; align-items:center; justify-content:center;">Click Run Inference</div><div class="caption">Disturbed T2 RGB</div></div>
              <div class="image-card"><div style="height:180px; background:#eee; display:flex; align-items:center; justify-content:center;">Click Run Inference</div><div class="caption">Sentinel-1 SAR False Color</div></div>
              <div class="image-card"><div style="height:180px; background:#eee; display:flex; align-items:center; justify-content:center;">Click Run Inference</div><div class="caption">Deforestation Probability</div></div>
            </div>
          </div>

          <div class="card">
            <h3 class="card-title">Detected Deforestation Polygons</h3>
            <div style="overflow-x: auto;">
              <table id="alertsTable">
                <thead>
                  <tr><th>Cluster ID</th><th>Area (ha)</th><th>Alert Severity</th><th>Centroid Coordinates</th></tr>
                </thead>
                <tbody id="alertsBody">
                  <tr><td colspan="4" style="text-align:center;">Run inference to view alert polygons.</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: EUDR DUE DILIGENCE AUDIT -->
    <div id="tab3" class="tab-content">
      <div class="grid-2">
        <div>
          <div class="card">
            <h3 class="card-title">Due Diligence Statement (DDS)</h3>
            <div class="form-group">
              <label>Preset Test Scenarios</label>
              <select id="eudrPreset" onchange="loadPresetScenario()">
                <option value="">-- Choose a Preset --</option>
              </select>
            </div>
            <div class="form-group">
              <label>Statement ID</label>
              <input type="text" id="eudrStmtId" class="form-control" value="DDS-2024-BRAZIL-SOYA-001">
            </div>
            <div class="form-group">
              <label>Operator Name</label>
              <input type="text" id="eudrOperator" class="form-control" value="Agroflorestal Rio Negro S.A.">
            </div>
            <div class="form-group">
              <label>Commodity (EUDR Annex I)</label>
              <select id="eudrCommodity">
                <option value="cattle">Cattle</option>
                <option value="cocoa">Cocoa</option>
                <option value="coffee">Coffee</option>
                <option value="oil palm">Oil Palm</option>
                <option value="rubber">Rubber</option>
                <option value="soya" selected>Soya</option>
                <option value="wood">Wood</option>
                <option value="wheat">Wheat (Non-Regulated Test)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Country of Production</label>
              <input type="text" id="eudrCountry" class="form-control" value="Brazil">
            </div>
            <button class="btn btn-primary" onclick="runEudrAudit()">⚖️ Audit EUDR Compliance</button>
          </div>
        </div>

        <div>
          <div class="card" id="auditVerdictCard" style="display:none;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
              <div>
                <span style="font-size:0.9rem; color:var(--text-muted); font-weight:600;">AUDIT VERDICT:</span>
                <span id="verdictBadge" class="badge-status COMPLIANT" style="font-size:1.1rem; margin-left:0.5rem;">COMPLIANT</span>
              </div>
              <div>
                <span style="font-size:0.9rem; color:var(--text-muted); font-weight:600;">RISK SCORE:</span>
                <span id="verdictRisk" style="font-size:1.2rem; font-weight:700; margin-left:0.3rem;">0.00</span>
              </div>
            </div>
            <div id="plotsAuditContainer"></div>
          </div>

          <div class="card">
            <h3 class="card-title">Plot Geolocation Specifications (JSON)</h3>
            <div class="form-group">
              <textarea id="eudrPlotsJson" rows="6"></textarea>
            </div>
            <h3 class="card-title" style="margin-top:1.5rem;">Deforestation Alerts for Intersection (JSON)</h3>
            <div class="form-group">
              <textarea id="eudrAlertsJson" rows="4"></textarea>
              <button class="btn btn-secondary" type="button" onclick="importLatestAlertsToEudr()" style="margin-top:0.5rem; font-size:0.8rem; padding:0.4rem 0.8rem; width:auto;">⚡ Import AI Alerts from Tab 2 Inference</button>
            </div>
          </div>

          <div class="card">
            <h3 class="card-title">📐 Olofsson et al. (2014) Area-Adjusted Accuracy Assessment</h3>
            <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:1rem;">
              Calculates unbiased change area estimates and 95% confidence intervals from stratified error matrices.
            </p>
            <div class="metric-grid">
              <div class="metric-box">
                <div class="metric-label">Area-Adjusted Overall Accuracy</div>
                <div class="metric-val" id="olofOA">97.2% ± 0.8%</div>
              </div>
              <div class="metric-box">
                <div class="metric-label">Unbiased Deforestation Area</div>
                <div class="metric-val" id="olofArea">1,215 ha ± 94 ha</div>
              </div>
            </div>
            <button class="btn btn-secondary" onclick="calculateOlofsson()">Recompute Olofsson Accuracy</button>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: STAC CATALOG EXPLORER -->
    <div id="tab4" class="tab-content">
      <div class="grid-2">
        <div>
          <div class="card">
            <h3 class="card-title">Catalog Search Filters</h3>
            <div class="form-group">
              <label>Region Preset</label>
              <select id="stacPreset" onchange="applyStacPreset()">
                <option value="amazon" selected>Amazon Basin (Manaus)</option>
                <option value="congo">Congo Basin (Salonga)</option>
                <option value="borneo">Borneo Tropical Peatland</option>
              </select>
            </div>
            <div class="form-group">
              <label>Bounding Box [W, S, E, N]</label>
              <input type="text" id="stacBbox" class="form-control" value="[-60.5, -3.5, -59.5, -2.5]">
            </div>
            <div class="form-group">
              <label>Satellite Platform</label>
              <select id="stacPlatform">
                <option value="">All Platforms</option>
                <option value="sentinel-2">Sentinel-2 (Optical L2A)</option>
                <option value="sentinel-1">Sentinel-1 (SAR GRD)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Max Cloud Cover (%)</label>
              <input type="number" id="stacCloud" class="form-control" value="30" min="0" max="100">
            </div>
            <button class="btn btn-primary" onclick="searchSTAC()">🔍 Search STAC Items</button>
          </div>
        </div>

        <div>
          <div class="card">
            <h3 class="card-title">Matched Cloud-Optimized GeoTIFF (COG) Items</h3>
            <div id="stacResultsContainer">
              <p style="color:var(--text-muted); font-size:0.9rem;">Click "Search STAC Items" to query the catalog.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </main>

  <script>
    // Tab switching
    function switchTab(tabId, el) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      el.classList.add('active');
      document.getElementById(tabId).classList.add('active');
      if (tabId === 'tab2' && map) {
        setTimeout(() => {
          map.invalidateSize();
          if (alertGeoJsonLayer && alertGeoJsonLayer.getBounds && alertGeoJsonLayer.getBounds().isValid()) {
            map.fitBounds(alertGeoJsonLayer.getBounds(), { padding: [20, 20] });
          }
        }, 200);
      }
    }

    // Chart.js Training Graph
    let trainChart = null;
    function initChart() {
      const ctx = document.getElementById('trainingChart').getContext('2d');
      trainChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: [1, 2, 3],
          datasets: [
            { label: 'Train Loss', data: [0.65, 0.42, 0.28], borderColor: '#d90429', tension: 0.2 },
            { label: 'Val Loss', data: [0.61, 0.39, 0.25], borderColor: '#f77f00', tension: 0.2 },
            { label: 'Val IoU (%)', data: [45.2, 72.8, 86.4], borderColor: '#2d6a4f', tension: 0.2, yAxisID: 'y1' }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { title: { display: true, text: 'Loss' } },
            y1: { position: 'right', title: { display: true, text: 'IoU (%)' }, grid: { drawOnChartArea: false } }
          }
        }
      });
    }

    // Leaflet Interactive Map
    let map = null;
    let alertGeoJsonLayer = null;
    function initMap() {
      map = L.map('leafletMap').setView([-3.05, -60.05], 11);
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
        attribution: '© OpenStreetMap contributors | GeoFM Sentinel Alert Layer'
      }).addTo(map);
    }

    // Preset Scenarios Cache
    let presetScenarios = {};

    async function loadPresets() {
      try {
        const resp = await fetch('/studio/api/eudr/presets');
        presetScenarios = await resp.json();
        const sel = document.getElementById('eudrPreset');
        sel.innerHTML = '<option value="">-- Choose a Preset Scenario --</option>';
        for (const [name, sc] of Object.entries(presetScenarios)) {
          const opt = document.createElement('option');
          opt.value = name;
          opt.innerText = name;
          sel.appendChild(opt);
        }
        if (Object.keys(presetScenarios).length > 0) {
          sel.selectedIndex = 1;
          loadPresetScenario();
        }
      } catch (err) {
        console.error('Failed to load presets:', err);
      }
    }

    function loadPresetScenario() {
      const name = document.getElementById('eudrPreset').value;
      if (!name || !presetScenarios[name]) return;
      const sc = presetScenarios[name];
      document.getElementById('eudrStmtId').value = sc.statement_id;
      document.getElementById('eudrOperator').value = sc.operator_name;
      document.getElementById('eudrCommodity').value = sc.commodity;
      document.getElementById('eudrCountry').value = sc.country;
      document.getElementById('eudrPlotsJson').value = JSON.stringify(sc.plots, null, 2);
      document.getElementById('eudrAlertsJson').value = JSON.stringify(sc.alerts, null, 2);
    }

    async function loadCheckpoints() {
      try {
        const resp = await fetch('/studio/api/checkpoints');
        const chks = await resp.json();
        const tbody = document.getElementById('checkpointBody');
        tbody.innerHTML = '';
        if (chks.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;">No checkpoint files found.</td></tr>';
          return;
        }
        for (const c of chks) {
          const tr = document.createElement('tr');
          const iouStr = typeof c.best_val_iou === 'number' ? (c.best_val_iou * 100).toFixed(2) + '%' : c.best_val_iou;
          tr.innerHTML = `<td><b>${c.filename}</b></td><td>${c.epoch}</td><td>${iouStr}</td><td>${c.size_mb} MB</td><td>${c.modified}</td>`;
          tbody.appendChild(tr);
        }
      } catch (err) {
        console.error('Failed to load checkpoints:', err);
      }
    }

    // Training Execution
    async function startTraining() {
      const btn = document.getElementById('btnStartTrain');
      btn.disabled = true;
      btn.innerText = '⏳ Training in Progress...';
      const prog = document.getElementById('trainProgBar');
      prog.style.display = 'block';

      const payload = {
        architecture: document.getElementById('trainArch').value,
        use_lora: document.getElementById('trainUseLora').checked,
        lora_rank: parseInt(document.getElementById('trainLoraRank').value),
        lora_alpha: parseFloat(document.getElementById('trainLoraAlpha').value),
        loss_type: document.getElementById('trainLoss').value,
        learning_rate: parseFloat(document.getElementById('trainLr').value),
        epochs: parseInt(document.getElementById('trainEpochs').value),
        batch_size: parseInt(document.getElementById('trainBatch').value),
        num_samples: 16
      };

      try {
        const resp = await fetch('/studio/api/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await resp.json();

        // Update Chart
        const hist = data.history || [];
        const labels = hist.map(h => 'Epoch ' + h.epoch);
        const trainLosses = hist.map(h => h.train_loss);
        const valLosses = hist.map(h => h.val_loss);
        const valIous = hist.map(h => (h.val_iou * 100).toFixed(2));

        trainChart.data.labels = labels;
        trainChart.data.datasets[0].data = trainLosses;
        trainChart.data.datasets[1].data = valLosses;
        trainChart.data.datasets[2].data = valIous;
        trainChart.update();

        // Update Table
        const tbody = document.getElementById('epochTableBody');
        tbody.innerHTML = '';
        for (const h of hist) {
          const cm = h.confusion_matrix || {};
          const tr = document.createElement('tr');
          tr.innerHTML = `<td>${h.epoch}</td><td>${h.train_loss}</td><td>${h.val_loss}</td><td>${(h.val_iou*100).toFixed(2)}%</td><td>${(h.val_f1*100).toFixed(2)}%</td><td>${(h.val_precision*100).toFixed(2)}%</td><td>${(h.val_recall*100).toFixed(2)}%</td><td>${cm.tp || 0} / ${cm.fp || 0} / ${cm.fn || 0}</td>`;
          tbody.appendChild(tr);
        }

        loadCheckpoints();
      } catch (err) {
        alert('Training failed: ' + err);
      } finally {
        btn.disabled = false;
        btn.innerText = '🚀 Start Training Run';
        prog.style.display = 'none';
      }
    }

    // Satellite Inference Execution
    async function runInference() {
      const payload = {
        deforestation_fraction: parseFloat(document.getElementById('inferDefFrac').value),
        cloud_fraction: parseFloat(document.getElementById('inferCloudFrac').value),
        model_type: document.getElementById('inferModel').value,
        threshold: parseFloat(document.getElementById('inferThreshold').value),
        mmu_ha: parseFloat(document.getElementById('inferMmu').value)
      };

      try {
        const resp = await fetch('/studio/api/infer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await resp.json();

        // Metrics
        document.getElementById('metricAlerts').innerText = data.total_alerts;
        document.getElementById('metricDefHa').innerText = data.total_deforested_ha.toFixed(2) + ' ha';
        document.getElementById('metricLatency').innerText = data.inference_time_ms + ' ms';
        document.getElementById('metricCloud').innerText = data.cloud_percentage.toFixed(1) + '%';

        // Layers
        const layers = data.layers;
        window.latestInferenceGeoJson = data.geojson;
        const grid = document.getElementById('layerGrid');
        grid.innerHTML = `
          <div class="image-card"><img src="${layers.t1_rgb}"><div class="caption">Baseline T1 (True Color RGB)</div></div>
          <div class="image-card"><img src="${layers.t1_cir}"><div class="caption">Baseline T1 CIR (Color Infrared)</div></div>
          <div class="image-card"><img src="${layers.ndvi_t1}"><div class="caption">Baseline T1 NDVI (Vegetation)</div></div>
          <div class="image-card"><img src="${layers.t2_rgb}"><div class="caption">Disturbed T2 (True Color RGB)</div></div>
          <div class="image-card"><img src="${layers.t2_cir}"><div class="caption">T2 Color Infrared (CIR B8-B4-B3)</div></div>
          <div class="image-card"><img src="${layers.ndvi_t2}"><div class="caption">Disturbed T2 NDVI (Disturbance)</div></div>
          <div class="image-card"><img src="${layers.sar_composite}"><div class="caption">Sentinel-1 SAR False Color</div></div>
          <div class="image-card"><img src="${layers.probability_heatmap}"><div class="caption">Deforestation Probability Map</div></div>
        `;

        // Render GeoJSON Alerts on Leaflet Map
        if (alertGeoJsonLayer) {
          map.removeLayer(alertGeoJsonLayer);
        }

        const geojson = data.geojson;
        alertGeoJsonLayer = L.geoJSON(geojson, {
          style: function(feat) {
            const lvl = feat.properties.alert_level;
            const color = lvl === 'CRITICAL' ? '#d90429' : lvl === 'HIGH' ? '#f77f00' : '#fcbf49';
            return { color: color, weight: 2, fillColor: color, fillOpacity: 0.5 };
          },
          onEachFeature: function(feat, layer) {
            const p = feat.properties;
            layer.bindPopup(`<b>Alert ${p.cluster_id}</b><br>Severity: <b>${p.alert_level}</b><br>Area: ${p.area_ha} ha`);
          }
        }).addTo(map);

        if (geojson.features && geojson.features.length > 0 && alertGeoJsonLayer.getBounds && alertGeoJsonLayer.getBounds().isValid() && map.getSize().x > 0) {
          map.fitBounds(alertGeoJsonLayer.getBounds(), { padding: [20, 20] });
        }

        // Table
        const tbody = document.getElementById('alertsBody');
        tbody.innerHTML = '';
        for (const f of geojson.features || []) {
          const p = f.properties;
          const tr = document.createElement('tr');
          tr.innerHTML = `<td><b>${p.cluster_id}</b></td><td>${p.area_ha} ha</td><td><span class="badge" style="background:#d90429; color:white;">${p.alert_level}</span></td><td>${p.centroid ? p.centroid.join(', ') : 'N/A'}</td>`;
          tbody.appendChild(tr);
        }
      } catch (err) {
        alert('Inference failed: ' + err);
      }
    }

    function importLatestAlertsToEudr() {
      if (window.latestInferenceGeoJson) {
        document.getElementById('eudrAlertsJson').value = JSON.stringify(window.latestInferenceGeoJson, null, 2);
      } else {
        alert('No inference alerts available yet. Run satellite inference in Tab 2 first.');
      }
    }

    // EUDR Due Diligence Audit
    async function runEudrAudit() {
      let plots, alerts;
      try {
        plots = JSON.parse(document.getElementById('eudrPlotsJson').value);
        alerts = JSON.parse(document.getElementById('eudrAlertsJson').value);
      } catch (err) {
        alert('Invalid JSON in plot or alert fields: ' + err);
        return;
      }

      const payload = {
        statement_id: document.getElementById('eudrStmtId').value,
        operator_name: document.getElementById('eudrOperator').value,
        commodity: document.getElementById('eudrCommodity').value,
        country: document.getElementById('eudrCountry').value,
        plots: plots,
        alerts: alerts
      };

      try {
        const resp = await fetch('/studio/api/eudr/audit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await resp.json();

        const card = document.getElementById('auditVerdictCard');
        card.style.display = 'block';

        const badge = document.getElementById('verdictBadge');
        badge.innerText = data.overall_status;
        badge.className = 'badge-status ' + data.overall_status;

        document.getElementById('verdictRisk').innerText = data.risk_score.toFixed(2);

        const container = document.getElementById('plotsAuditContainer');
        container.innerHTML = '';
        for (const p of data.plots || []) {
          const div = document.createElement('div');
          div.style.cssText = 'border:1px solid #dee2e6; border-radius:6px; padding:0.75rem; margin-top:0.75rem; background:#f8f9fa;';
          let violationsHtml = '<span style="color:#2d6a4f; font-weight:600;">✓ No Violations Detected</span>';
          if (p.violations && p.violations.length > 0) {
            violationsHtml = '<ul style="color:#d90429; margin-left:1.2rem; margin-top:0.25rem;">' + p.violations.map(v => `<li>${v}</li>`).join('') + '</ul>';
          }
          div.innerHTML = `
            <div style="display:flex; justify-content:space-between; font-weight:700;">
              <span>Plot: ${p.plot_id} (${p.area_ha} ha)</span>
              <span class="badge-status ${p.status}">${p.status}</span>
            </div>
            <div style="margin-top:0.5rem; font-size:0.85rem;">${violationsHtml}</div>
          `;
          container.appendChild(div);
        }
      } catch (err) {
        alert('Audit failed: ' + err);
      }
    }

    // Olofsson Assessment
    async function calculateOlofsson() {
      try {
        const resp = await fetch('/studio/api/olofsson', { method: 'POST' });
        const data = await resp.json();
        document.getElementById('olofOA').innerText = `${data.overall_accuracy_pct}% ± ${data.oa_margin_of_error_pct}%`;
        const defArea = data.estimated_areas_ha['Deforestation'] || 0;
        const defMoe = data.area_margins_of_error_ha['Deforestation'] || 0;
        document.getElementById('olofArea').innerText = `${defArea.toLocaleString()} ha ± ${defMoe.toLocaleString()} ha`;
      } catch (err) {
        console.error('Olofsson failed:', err);
      }
    }

    // STAC Search
    function applyStacPreset() {
      const p = document.getElementById('stacPreset').value;
      if (p === 'amazon') document.getElementById('stacBbox').value = '[-60.5, -3.5, -59.5, -2.5]';
      if (p === 'congo') document.getElementById('stacBbox').value = '[20.0, -2.5, 21.0, -1.5]';
      if (p === 'borneo') document.getElementById('stacBbox').value = '[112.5, -2.5, 113.5, -1.5]';
    }

    async function searchSTAC() {
      let bbox;
      try {
        bbox = JSON.parse(document.getElementById('stacBbox').value);
      } catch (e) {
        alert('Invalid bounding box format. Use [min_lon, min_lat, max_lon, max_lat]');
        return;
      }

      const payload = {
        bbox: bbox,
        platform: document.getElementById('stacPlatform').value || null,
        max_cloud_cover: parseFloat(document.getElementById('stacCloud').value)
      };

      try {
        const resp = await fetch('/studio/api/stac/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const items = await resp.json();
        const container = document.getElementById('stacResultsContainer');
        container.innerHTML = '';
        if (items.length === 0) {
          container.innerHTML = '<p style="color:var(--text-muted);">No matching STAC items found.</p>';
          return;
        }
        for (const it of items) {
          const div = document.createElement('div');
          div.style.cssText = 'border:1px solid #dee2e6; border-radius:6px; padding:1rem; margin-bottom:1rem; background:white;';
          let assetsHtml = Object.entries(it.assets).map(([k, v]) => `<li><b>${k}</b>: <a href="${v.href}" target="_blank">${v.title}</a> (${v.media_type})</li>`).join('');
          div.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
              <h4 style="color:var(--primary); font-size:0.95rem;">🛰️ ${it.id}</h4>
              <span class="badge badge-primary">${it.platform}</span>
            </div>
            <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:0.5rem;">Acquisition: <b>${it.datetime}</b> | Cloud: <b>${it.cloud_cover !== null ? it.cloud_cover + '%' : 'N/A'}</b></p>
            <p style="font-size:0.8rem; margin-bottom:0.5rem;">Bounding Box: <code>${JSON.stringify(it.bbox)}</code></p>
            <details style="font-size:0.85rem;"><summary style="cursor:pointer; font-weight:600;">COG Assets (${Object.keys(it.assets).length})</summary><ul style="margin-left:1.2rem; margin-top:0.35rem;">${assetsHtml}</ul></details>
          `;
          container.appendChild(div);
        }
      } catch (err) {
        alert('STAC query failed: ' + err);
      }
    }

    // Startup initialization
    window.addEventListener('DOMContentLoaded', () => {
      initChart();
      initMap();
      loadPresets();
      loadCheckpoints();
      // Run initial inference to populate layers
      runInference();
    });
  </script>
</body>
</html>
"""


# =============================================================================
# FastAPI Application Factory
# =============================================================================

def create_studio_app() -> FastAPI:
    """Create FastAPI application instance for the Web Training Studio."""
    app = FastAPI(
        title="GeoFM Web Training Studio & EUDR Compliance Suite",
        description="Interactive Studio for Geospatial AI Deforestation Detection & EUDR Regulatory Audits",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize Managers
    training_mgr = StudioTrainingManager()
    inference_mgr = StudioInferenceManager()
    compliance_mgr = StudioComplianceManager()
    stac_mgr = StudioSTACManager()
    checkpoint_mgr = CheckpointManager()

    # --- HTML Studio UI Route ---
    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(content=STUDIO_HTML, status_code=200)

    # --- Studio Training API Endpoints ---
    @app.post("/studio/api/train")
    def train_endpoint(req: TrainRequest):
        summary = training_mgr.train_model(
            architecture=req.architecture,
            use_lora=req.use_lora,
            lora_rank=req.lora_rank,
            lora_alpha=req.lora_alpha,
            loss_type=req.loss_type,
            learning_rate=req.learning_rate,
            epochs=req.epochs,
            batch_size=req.batch_size,
            num_samples=req.num_samples,
        )
        return summary

    @app.get("/studio/api/train/status")
    def train_status():
        return {
            "is_training": training_mgr.is_training,
            "latest_run": training_mgr.latest_run_summary,
            "history": training_mgr.latest_history,
        }

    # --- Satellite Inference API Endpoints ---
    @app.post("/studio/api/infer")
    def infer_endpoint(req: InferRequest):
        result = inference_mgr.run_inference(
            deforestation_fraction=req.deforestation_fraction,
            cloud_fraction=req.cloud_fraction,
            model_type=req.model_type,
            threshold=req.threshold,
            mmu_ha=req.mmu_ha,
            west=req.west,
            north=req.north,
        )
        return result

    # --- EUDR Compliance API Endpoints ---
    @app.get("/studio/api/eudr/presets")
    def eudr_presets():
        return compliance_mgr.get_preset_scenarios()

    @app.post("/studio/api/eudr/audit")
    def eudr_audit_endpoint(req: AUDITRequest):
        report = compliance_mgr.audit_statement(
            statement_id=req.statement_id,
            operator_name=req.operator_name,
            commodity=req.commodity,
            country=req.country,
            plots=req.plots,
            alerts=req.alerts,
        )
        return report

    @app.post("/studio/api/olofsson")
    def olofsson_endpoint():
        return compliance_mgr.compute_olofsson_accuracy()

    # --- STAC Catalog API Endpoints ---
    @app.post("/studio/api/stac/search")
    def stac_search_endpoint(req: STACQueryRequest):
        items = stac_mgr.query_catalog(
            bbox=req.bbox,
            start_date=req.start_date,
            end_date=req.end_date,
            platform=req.platform,
            max_cloud_cover=req.max_cloud_cover,
        )
        return items

    # --- Checkpoints API Endpoints ---
    @app.get("/studio/api/checkpoints")
    def checkpoints_endpoint():
        return checkpoint_mgr.list_checkpoints()

    return app


studio_fastapi_app = create_studio_app()
