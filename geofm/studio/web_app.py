"""Standalone FastAPI Interactive Web Studio for GeoFM Deforestation Detection & EUDR Suite.

Serves an interactive HTML5/CSS/JavaScript dashboard powered by Leaflet.js maps
and Chart.js analytics when Streamlit is not installed, or when running headless/server mode.
"""

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from pydantic import BaseModel, Field
import torch
import numpy as np

from geofm.compliance.eudr import EUDR_REGULATED_COMMODITIES
from geofm.studio.core import (
    CheckpointManager,
    StudioComplianceManager,
    StudioInferenceManager,
    StudioSTACManager,
    StudioTrainingManager,
    StudioChipManager,
    get_hardware_telemetry,
)


# =============================================================================
# Request / Response Schemas
# =============================================================================

class TrainRequest(BaseModel):
    architecture: str = Field(default="siamese", description="siamese, unet, or multimodal")
    use_lora: bool = Field(default=True)
    lora_rank: int = Field(default=16)
    lora_alpha: float = Field(default=32.0)
    loss_type: str = Field(default="compound")
    tversky_alpha: float = Field(default=0.3)
    tversky_beta: float = Field(default=0.7)
    learning_rate: float = Field(default=0.0003)
    epochs: int = Field(default=20)
    batch_size: int = Field(default=16)
    num_samples: int = Field(default=200)
    dataset_source: str = Field(default="cached", description="cached (BYOD), real, or synthetic")
    dataset_scale: str = Field(default="continental")
    gee_project_id: Optional[str] = Field(default=None)
    chip_size: int = Field(default=256)
    use_data_augmentation: bool = Field(default=True)
    enforce_class_balance: bool = Field(default=True)
    use_amp: bool = Field(default=True)
    ram_cache_budget_gb: Optional[float] = Field(default=4.0)
    async_run: bool = Field(default=False)
    resume: bool = Field(default=False)
    cache_dir: str = Field(default="./data/chips_cache")


class TestBenchmarkRequest(BaseModel):
    checkpoint_name: Optional[str] = None
    num_samples: int = Field(default=16)
    threshold: float = Field(default=0.5)


class InspectChipRequest(BaseModel):
    chip_idx: int = Field(default=0)
    checkpoint_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    threshold: float = Field(default=0.5)
    mmu_ha: float = Field(default=0.10)


class ManualInferRequest(BaseModel):
    chip_idx: Optional[int] = Field(default=None)
    chip_id: Optional[str] = Field(default=None)
    checkpoint_path: Optional[str] = Field(default=None)
    model_type: str = Field(default="siamese")
    threshold: float = Field(default=0.5)
    mmu_ha: float = Field(default=0.10)
    west: float = Field(default=-60.05)
    north: float = Field(default=-3.05)


class SetFolderRequest(BaseModel):
    folder_path: str


class FetchGEERequest(BaseModel):
    project_id: str = Field(default="")
    num_chips: int = Field(default=200)
    chip_size: int = Field(default=256)
    target_pos_ratio: float = Field(default=0.70)


class LoadPathCheckpointRequest(BaseModel):
    file_path: str
    model_type: str = Field(default="siamese")


class ThresholdSweepRequest(BaseModel):
    checkpoint_name: Optional[str] = None
    num_samples: int = Field(default=8)


class GenerateChipsRequest(BaseModel):
    num_chips: int = Field(default=20)
    deforest_ratio: float = Field(default=0.70)


class LoadCheckpointRequest(BaseModel):
    filename: str


class InferRequest(BaseModel):
    deforestation_fraction: float = Field(default=0.08)
    cloud_fraction: float = Field(default=0.15)
    model_type: str = Field(default="siamese")
    threshold: float = Field(default=0.5)
    mmu_ha: float = Field(default=0.10)
    west: float = Field(default=-60.05)
    north: float = Field(default=-3.05)
    seed: Optional[int] = Field(default=None)


class OlofssonRequest(BaseModel):
    source: str = Field(default="reference")  # "reference", "benchmark", "custom"
    tp: Optional[int] = Field(default=None)
    fp: Optional[int] = Field(default=None)
    fn: Optional[int] = Field(default=None)
    tn: Optional[int] = Field(default=None)
    stratum_deforest_ha: Optional[float] = Field(default=1250.0)
    stratum_forest_ha: Optional[float] = Field(default=18750.0)


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


class CatalogValidateRequest(BaseModel):
    url: str


class CatalogHealRequest(BaseModel):
    url: str


# =============================================================================
# HTML / CSS / JS Single Page App Template
# =============================================================================

STUDIO_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GeoFM Web Training Studio & EUDR Compliance Suite</title>
  <!-- Powered by Leaflet Maps and Chart.js Analytics -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0B111A;
      --card-bg: #131B26;
      --card-elevated: #1B2433;
      --text: #F8FAFC;
      --text-muted: #94A3B8;
      --border: rgba(255, 255, 255, 0.08);
      --border-focus: #06B6D4;
      --accent-emerald: #10B981;
      --accent-cyan: #06B6D4;
      --accent-crimson: #EF4444;
      --accent-amber: #F59E0B;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: 'Fira Code', 'JetBrains Mono', ui-monospace, monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      -webkit-font-smoothing: antialiased;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.16); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.28); }
    ::selection { background: rgba(6, 182, 212, 0.35); color: #F8FAFC; }

    /* Header & Telemetry */
    header {
      background: #0E1622;
      border-bottom: 1px solid var(--border);
      padding: 0.85rem 1.75rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
    }
    .brand-wrap { display: flex; align-items: center; gap: 0.75rem; }
    .brand-badge {
      width: 32px;
      height: 32px;
      border-radius: 8px;
      background: rgba(6, 182, 212, 0.15);
      border: 1px solid rgba(6, 182, 212, 0.3);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--accent-cyan);
    }
    .brand-title { font-size: 1.15rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; }
    .brand-desc { font-size: 0.75rem; color: var(--text-muted); }

    .telemetry-strip { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
    .tele-pill {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 4px 10px;
      font-size: 0.72rem;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
      color: var(--text-muted);
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
    }
    .tele-pill strong { color: var(--text); font-weight: 600; }
    .status-pulse {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent-emerald);
      box-shadow: 0 0 8px var(--accent-emerald);
      animation: pulseAnim 2s infinite ease-in-out;
    }
    .status-pulse.busy {
      background: var(--accent-amber);
      box-shadow: 0 0 8px var(--accent-amber);
    }
    @keyframes pulseAnim { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.85); } }

    /* Segmented Navigation */
    nav.tabs-nav {
      background: #0E1622;
      border-bottom: 1px solid var(--border);
      display: flex;
      padding: 0 1.75rem;
      gap: 0.5rem;
      overflow-x: auto;
    }
    .tab-btn {
      background: transparent;
      border: none;
      padding: 0.85rem 1rem;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      white-space: nowrap;
    }
    .tab-btn svg { width: 16px; height: 16px; stroke-width: 1.75; }
    .tab-btn:hover { color: var(--text); }
    .tab-btn.active {
      color: var(--accent-cyan);
      border-bottom-color: var(--accent-cyan);
      background: rgba(6, 182, 212, 0.04);
    }

    /* Main Container */
    main { flex: 1; padding: 1.5rem 1.75rem; max-width: 1680px; margin: 0 auto; width: 100%; }
    .tab-content { display: none; }
    .tab-content.active { display: block; animation: fadeIn 0.2s ease-out; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

    /* Layout Grids */
    .grid-2 { display: grid; grid-template-columns: 340px 1fr; gap: 1.25rem; }
    .grid-2-equal { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
    @media (max-width: 1100px) {
      .grid-2, .grid-2-equal { grid-template-columns: 1fr; }
    }

    /* Cards */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      margin-bottom: 1.25rem;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .card-title {
      font-size: 0.98rem;
      font-weight: 600;
      color: #FFFFFF;
      margin-bottom: 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      letter-spacing: -0.01em;
    }

    /* Form Inputs */
    .form-group { margin-bottom: 0.85rem; }
    .form-group label {
      display: block;
      font-size: 0.78rem;
      font-weight: 500;
      color: var(--text-muted);
      margin-bottom: 0.35rem;
    }
    .form-control, select, textarea {
      width: 100%;
      background: #0B111A;
      color: var(--text);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 7px;
      padding: 0.55rem 0.75rem;
      font-size: 0.85rem;
      font-family: var(--font-sans);
      transition: border-color 0.15s;
    }
    .form-control:focus, select:focus, textarea:focus {
      border-color: var(--border-focus);
      outline: none;
      box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.15);
    }
    .checkbox-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin: 0.65rem 0;
      font-size: 0.82rem;
      color: var(--text);
    }
    .checkbox-row input[type="checkbox"] {
      accent-color: var(--accent-cyan);
      width: 15px;
      height: 15px;
      cursor: pointer;
    }

    /* Buttons */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      padding: 0.55rem 1.1rem;
      font-size: 0.84rem;
      font-weight: 600;
      border-radius: 7px;
      border: 1px solid transparent;
      cursor: pointer;
      transition: all 0.15s ease;
      text-decoration: none;
    }
    .btn svg { width: 15px; height: 15px; stroke-width: 1.75; }
    .btn-emerald { background: var(--accent-emerald); color: #0B111A; }
    .btn-emerald:hover { background: #059669; }
    .btn-cyan { background: var(--accent-cyan); color: #0B111A; }
    .btn-cyan:hover { background: #0891B2; }
    .btn-crimson { background: var(--accent-crimson); color: #FFFFFF; }
    .btn-crimson:hover { background: #DC2626; }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.05);
      border-color: var(--border);
      color: var(--text);
    }
    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.10);
      border-color: rgba(255, 255, 255, 0.2);
    }
    .btn-block { width: 100%; }

    /* Scorecards & Metrics */
    .metrics-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
      margin-bottom: 1.25rem;
    }
    .metric-cell {
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem 1rem;
    }
    .metric-cell .label {
      font-size: 0.70rem;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
      letter-spacing: 0.03em;
    }
    .metric-cell .val {
      font-size: 1.35rem;
      font-weight: 700;
      color: #FFFFFF;
      margin-top: 0.2rem;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
    }
    .metric-cell.emerald .val { color: var(--accent-emerald); }
    .metric-cell.cyan .val { color: var(--accent-cyan); }
    .metric-cell.crimson .val { color: var(--accent-crimson); }
    .metric-cell.amber .val { color: var(--accent-amber); }

    /* Confusion Matrix 2x2 */
    .cm-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 6px;
      margin-bottom: 1rem;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
    }
    .cm-cell {
      background: #0E1622;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.85rem;
      text-align: center;
    }
    .cm-cell .cm-count { font-size: 1.15rem; font-weight: 700; margin-top: 0.2rem; }
    .cm-cell.tp { border-color: rgba(16, 185, 129, 0.35); background: rgba(16, 185, 129, 0.05); }
    .cm-cell.tp .cm-count { color: var(--accent-emerald); }
    .cm-cell.fp { border-color: rgba(245, 158, 11, 0.35); background: rgba(245, 158, 11, 0.05); }
    .cm-cell.fp .cm-count { color: var(--accent-amber); }
    .cm-cell.fn { border-color: rgba(239, 68, 68, 0.35); background: rgba(239, 68, 68, 0.05); }
    .cm-cell.fn .cm-count { color: var(--accent-crimson); }
    .cm-cell.tn { border-color: var(--border); }
    .cm-cell.tn .cm-count { color: var(--text-muted); }

    /* Terminal Console */
    .terminal-box {
      background: #070B10;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.85rem 1rem;
      font-family: var(--font-mono);
      font-size: 0.76rem;
      color: #A7F3D0;
      height: 220px;
      overflow-y: auto;
      white-space: pre-wrap;
      line-height: 1.5;
    }
    .terminal-line { margin-bottom: 2px; }
    .terminal-line.warn { color: var(--accent-amber); }
    .terminal-line.err { color: var(--accent-crimson); }
    .terminal-line.info { color: var(--accent-cyan); }

    /* Spectral Layers Grid */
    .layers-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
      gap: 1rem;
      margin-top: 1rem;
    }
    .layer-card {
      background: #0E1622;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }
    .layer-card .layer-header {
      padding: 0.45rem 0.65rem;
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
    }
    .layer-card img {
      width: 100%;
      height: 170px;
      object-fit: cover;
      display: block;
      background: #05080C;
    }

    /* Tables */
    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
    }
    .data-table th {
      text-align: left;
      padding: 0.6rem 0.75rem;
      border-bottom: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      font-weight: 600;
    }
    .data-table td {
      padding: 0.65rem 0.75rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
    }
    .data-table tr:hover td { background: rgba(255, 255, 255, 0.02); }

    /* Map Viewport */
    #map {
      height: 480px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #0B111A;
    }

    /* Status Badges */
    .badge-pill {
      font-size: 0.70rem;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      display: inline-block;
    }
    .badge-compliant { background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald); border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-violation { background: rgba(239, 68, 68, 0.2); color: var(--accent-crimson); border: 1px solid rgba(239, 68, 68, 0.4); }

    /* Dual Viewport & Deforestation Overlay */
    .viewport-overlay-box {
      position: relative;
      width: 100%;
      aspect-ratio: 1 / 1;
      background: #05080C;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .overlay-base-img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .overlay-mask-img {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      pointer-events: none;
      transition: opacity 0.05s ease;
    }
    .overlay-badge {
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(11, 17, 26, 0.85);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 3px 8px;
      font-size: 0.70rem;
      font-family: var(--font-mono);
      color: var(--text);
      z-index: 10;
    }
    .slider-control-card {
      background: var(--card-elevated);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.85rem 1rem;
      margin-bottom: 1rem;
    }
    .form-range {
      width: 100%;
      cursor: pointer;
      accent-color: var(--accent-crimson);
    }
    .config-grid-4 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1rem;
    }
    .dropzone-box {
      border: 2px dashed rgba(6, 182, 212, 0.4);
      background: rgba(6, 182, 212, 0.03);
      border-radius: 8px;
      padding: 1.25rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .dropzone-box:hover {
      border-color: var(--accent-cyan);
      background: rgba(6, 182, 212, 0.08);
    }
  </style>
</head>
<body>

  <!-- Top App Header -->
  <header>
    <div class="brand-wrap">
      <div class="brand-badge">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
      </div>
      <div>
        <div class="brand-title">GeoFM Web Training Studio</div>
        <div class="brand-desc">Bitemporal Satellite Change Detection & EUDR Compliance Platform</div>
      </div>
    </div>
    <div class="telemetry-strip" id="telemetryStrip">
      <div class="tele-pill"><span class="status-pulse" id="trainingStatusDot"></span> <span id="teleTrainStatus">IDLE</span></div>
      <div class="tele-pill">DEVICE: <strong id="teleDevice">Detecting...</strong></div>
      <div class="tele-pill">RAM: <strong id="teleRam">-- / --</strong></div>
      <div class="tele-pill">CHIPS: <strong id="teleChips">--</strong></div>
      <div class="tele-pill">CHECKPOINTS: <strong id="teleCheckpoints">--</strong></div>
    </div>
  </header>

  <!-- Segmented Navigation (Author SVG Icons) -->
  <nav class="tabs-nav">
    <button class="tab-btn active" onclick="switchTab('training')">
      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>
      Training Studio
    </button>
    <button class="tab-btn" onclick="switchTab('inspector')">
      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><line x1="22" y1="12" x2="18" y2="12"></line><line x1="6" y1="12" x2="2" y2="12"></line><line x1="12" y1="6" x2="12" y2="2"></line><line x1="12" y1="22" x2="12" y2="18"></line></svg>
      Chip & Mask Viewer
    </button>
    <button class="tab-btn" onclick="switchTab('manual-infer')">
      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
      Model Inference Studio
    </button>
    <button class="tab-btn" onclick="switchTab('input-loader')">
      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
      Dataset & Input Loader
    </button>
    <button class="tab-btn" onclick="switchTab('benchmark')">
      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
      Benchmark & Scorecard
    </button>
    <button class="tab-btn" onclick="switchTab('map')">
      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"></polygon><line x1="8" y1="2" x2="8" y2="18"></line><line x1="16" y1="6" x2="16" y2="22"></line></svg>
      Geospatial Map
    </button>
    <button class="tab-btn" onclick="switchTab('eudr')">
      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
      EUDR Due Diligence
    </button>
    <button class="tab-btn" onclick="switchTab('zoo')">
      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>
      Checkpoints & Zoo
    </button>
    <a href="https://colab.research.google.com/github/safiydev/geofm-deforestation/blob/main/notebooks/geofm_training.ipynb" target="_blank" rel="noopener noreferrer" class="tab-btn" style="text-decoration:none; color:var(--accent-cyan); margin-left:auto; display:inline-flex; align-items:center; gap:0.4rem;">
      <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" style="width:14px; height:14px;"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
      Open Colab Pipeline ↗
    </a>
  </nav>

  <!-- Main Content Areas -->
  <main>

    <!-- TAB 1: AI TRAINING STUDIO -->
    <div id="tab-training" class="tab-content active">
      <!-- Colab Foundation Training Callout Banner -->
      <div style="background:linear-gradient(90deg, rgba(6,182,212,0.12), rgba(16,185,129,0.08)); border:1px solid rgba(6,182,212,0.3); border-radius:8px; padding:0.85rem 1.15rem; margin-bottom:1.25rem; display:flex; align-items:center; justify-content:space-between; gap:1.25rem; box-shadow:0 4px 16px rgba(0,0,0,0.2);">
        <div>
          <div style="display:flex; align-items:center; gap:0.5rem;">
            <span style="font-size:0.72rem; font-weight:800; background:rgba(6,182,212,0.2); color:var(--accent-cyan); padding:2px 7px; border-radius:4px; text-transform:uppercase; letter-spacing:0.05em;">Cloud Training Architecture</span>
            <span style="font-size:0.85rem; font-weight:700; color:var(--text);">Foundation Model Trained via Google Colab GPUs</span>
          </div>
          <div style="font-size:0.76rem; color:var(--text-muted); margin-top:0.3rem; line-height:1.45;">
            Our planetary-scale foundation weights were pre-trained on cloud GPU accelerators (NVIDIA A100 / T4) using our reproducible Google Colab pipeline with automated Google Earth Engine ingestion. This local Web Studio is engineered for edge inference, rapid LoRA adaptation, and <strong>Bring-Your-Own-Data (BYOD)</strong> validation.
          </div>
        </div>
        <a href="https://colab.research.google.com/github/safiydev/geofm-deforestation/blob/main/notebooks/geofm_training.ipynb" target="_blank" rel="noopener noreferrer" style="display:inline-flex; align-items:center; gap:0.4rem; padding:8px 14px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.2); border-radius:6px; color:#fff; text-decoration:none; font-size:0.75rem; font-weight:600; white-space:nowrap; transition:all 0.15s ease;">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height:19px;">
        </a>
      </div>
      <div class="grid-2">
        <div>
          <div class="card">
            <div class="card-title">
              <span>Planetary Training Configuration</span>
              <span class="badge-pill badge-compliant" style="font-size:0.68rem;">16 Hyperparameters</span>
            </div>

            <!-- Group 1: Dataset Ingestion & Scale -->
            <div style="font-size:0.75rem; font-weight:700; color:var(--accent-cyan); text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.5rem;">
              1. Dataset Ingestion & Spatial Scale
            </div>
            <div class="form-group">
              <label>Dataset Source (DATASET_SOURCE)</label>
              <select id="trainDatasetSource">
                <option value="cached" selected>Bring Your Own Data: Local Cache & Custom Uploads</option>
                <option value="real">Real Satellite Benchmark (OSCD100 Sentinel-2 Bitemporal)</option>
                <option value="synthetic">Offline Synthetic Multi-Spectral (Zero-RAM Streaming)</option>
              </select>
            </div>
            <div class="grid-2-equal">
              <div class="form-group">
                <label>Dataset Scale (DATASET_SCALE)</label>
                <select id="trainDatasetScale">
                  <option value="standard" selected>Standard (200 chips)</option>
                  <option value="high_fidelity">High-Fidelity (1,000 chips)</option>
                  <option value="continental">Continental Scale (5,000 chips)</option>
                  <option value="quick">Quick Test (32 chips)</option>
                </select>
              </div>
              <div class="form-group">
                <label>Chip Size (CHIP_SIZE)</label>
                <select id="trainChipSize">
                  <option value="256" selected>256 x 256 px (Native 10m Sentinel-2)</option>
                  <option value="128">128 x 128 px (Standard)</option>
                  <option value="64">64 x 64 px (Fast Prototype)</option>
                </select>
              </div>
            </div>
            <div class="form-group">
              <label>Active Dataset Directory (DATA_DIR)</label>
              <input type="text" id="trainDataDir" class="form-control" value="./data/chips_cache" placeholder="e.g. ./data/chips_cache or /path/to/my_data">
            </div>

            <!-- Group 2: Model Architecture & LoRA PEFT -->
            <div style="font-size:0.75rem; font-weight:700; color:var(--accent-cyan); text-transform:uppercase; letter-spacing:0.04em; margin:1rem 0 0.5rem 0;">
              2. Neural Backbone & LoRA Adaptation
            </div>
            <div class="form-group">
              <label>Model Architecture (MODEL_ARCHITECTURE)</label>
              <select id="trainArch">
                <option value="siamese" selected>Siamese Bitemporal Change Detector (T1 Baseline vs T2 Observation)</option>
                <option value="multimodal">Multimodal Bitemporal Change Net (Sentinel-1 SAR + Sentinel-2 Optical)</option>
                <option value="unet">UNet Bitemporal Change Baseline (ResNet-18 Backbone)</option>
              </select>
            </div>
            <div class="checkbox-row">
              <input type="checkbox" id="trainUseLora" checked>
              <label for="trainUseLora">Enable LoRA PEFT (USE_LORA)</label>
            </div>
            <div class="grid-2-equal">
              <div class="form-group">
                <label>LoRA Rank (LORA_RANK)</label>
                <input type="number" id="trainLoraRank" class="form-control" value="16" min="2" max="64">
              </div>
              <div class="form-group">
                <label>LoRA Alpha (LORA_ALPHA)</label>
                <input type="number" id="trainLoraAlpha" class="form-control" value="32.0" min="4" max="128">
              </div>
            </div>

            <!-- Group 3: Loss Function & Hyperparameters -->
            <div style="font-size:0.75rem; font-weight:700; color:var(--accent-cyan); text-transform:uppercase; letter-spacing:0.04em; margin:1rem 0 0.5rem 0;">
              3. Objective Loss & Penalties
            </div>
            <div class="form-group">
              <label>Loss Type (LOSS_TYPE)</label>
              <select id="trainLoss">
                <option value="compound" selected>Compound Change Detection Loss (Focal + Tversky)</option>
                <option value="focal_tversky">Focal Tversky Loss</option>
                <option value="tversky">Tversky Loss</option>
                <option value="focal">Binary Focal Loss (γ = 2.0)</option>
                <option value="bce">Standard Binary Cross Entropy</option>
              </select>
            </div>
            <div class="grid-2-equal">
              <div class="form-group">
                <label>Tversky Alpha (TVERSKY_ALPHA)</label>
                <input type="number" id="trainTverskyAlpha" class="form-control" value="0.3" step="0.05" min="0.05" max="0.95">
              </div>
              <div class="form-group">
                <label>Tversky Beta (TVERSKY_BETA)</label>
                <input type="number" id="trainTverskyBeta" class="form-control" value="0.7" step="0.05" min="0.05" max="0.95">
              </div>
            </div>

            <!-- Group 4: Optimization & Acceleration -->
            <div style="font-size:0.75rem; font-weight:700; color:var(--accent-cyan); text-transform:uppercase; letter-spacing:0.04em; margin:1rem 0 0.5rem 0;">
              4. Optimization & Acceleration
            </div>
            <div class="grid-2-equal">
              <div class="form-group">
                <label>Epochs (EPOCHS)</label>
                <input type="number" id="trainEpochs" class="form-control" value="20" min="1" max="100">
              </div>
              <div class="form-group">
                <label>Batch Size (BATCH_SIZE)</label>
                <input type="number" id="trainBatch" class="form-control" value="16" min="1" max="64">
              </div>
            </div>
            <div class="grid-2-equal">
              <div class="form-group">
                <label>Learning Rate (LEARNING_RATE)</label>
                <input type="number" id="trainLR" class="form-control" value="0.0003" step="0.00005">
              </div>
              <div class="form-group">
                <label>RAM Budget GB (RAM_CACHE_BUDGET_GB)</label>
                <input type="number" id="trainRamBudget" class="form-control" value="4.0" step="0.5" min="0.5" max="64.0">
              </div>
            </div>

            <div class="checkbox-row">
              <input type="checkbox" id="trainAugmentation" checked>
              <label for="trainAugmentation">Synchronous D4 Dihedral Data Augmentation (USE_DATA_AUGMENTATION)</label>
            </div>
            <div class="checkbox-row">
              <input type="checkbox" id="trainClassBalance" checked>
              <label for="trainClassBalance">Enforce 70:30 Deforestation Quota (ENFORCE_CLASS_BALANCE)</label>
            </div>
            <div class="checkbox-row">
              <input type="checkbox" id="trainAmp" checked>
              <label for="trainAmp">Automatic Mixed Precision FP16 (USE_AMP)</label>
            </div>
            <div class="checkbox-row">
              <input type="checkbox" id="trainResume" checked>
              <label for="trainResume">Resume from latest checkpoint</label>
            </div>

            <div style="display: flex; gap: 0.5rem; margin-top: 1.25rem;">
              <button class="btn btn-emerald" style="flex:1;" onclick="startTraining()">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                Start Training
              </button>
              <button class="btn btn-crimson" onclick="stopTraining()">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"></rect></svg>
                Stop
              </button>
            </div>
          </div>
        </div>

        <div>
          <div class="metrics-row">
            <div class="metric-cell cyan">
              <div class="label">Current Epoch</div>
              <div class="val" id="dispEpoch">0 / 0</div>
            </div>
            <div class="metric-cell crimson">
              <div class="label">Train Loss</div>
              <div class="val" id="dispTrainLoss">--</div>
            </div>
            <div class="metric-cell amber">
              <div class="label">Val Loss</div>
              <div class="val" id="dispValLoss">--</div>
            </div>
            <div class="metric-cell emerald">
              <div class="label">Val IoU</div>
              <div class="val" id="dispValIoU">--</div>
            </div>
            <div class="metric-cell">
              <div class="label">Learning Rate</div>
              <div class="val" id="dispLR">--</div>
            </div>
          </div>

          <div class="card">
            <div class="card-title">Live Loss & Metrics Convergence</div>
            <div style="height: 250px;">
              <canvas id="trainingChart"></canvas>
            </div>
          </div>

          <div class="card">
            <div class="card-title">Training Log Stream</div>
            <div class="terminal-box" id="trainTerminal">Studio standby. Ready to train foundation model.</div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: MODEL TESTING & BENCHMARKING -->
    <div id="tab-benchmark" class="tab-content">
      <div class="grid-2">
        <div>
          <div class="card">
            <div class="card-title">Change Detection Benchmark Parameters</div>
            <div class="form-group">
              <label>Select Model Checkpoint</label>
              <select id="benchCheckpoint">
                <option value="">Active Siamese Change Detection Backbone (In-Memory)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Validation Samples Count</label>
              <input type="number" id="benchSamples" class="form-control" value="16" min="4" max="100">
            </div>
            <div class="form-group">
              <label>Change Detection Threshold: <strong id="benchThreshVal" style="color:var(--accent-cyan);">0.50</strong></label>
              <input type="range" id="benchThreshold" min="0.10" max="0.90" step="0.05" value="0.50" oninput="document.getElementById('benchThreshVal').innerText = parseFloat(this.value).toFixed(2)">
            </div>
            <button class="btn btn-cyan btn-block" onclick="runBenchmark()">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path></svg>
              Execute Benchmark
            </button>
            <div style="margin-top: 0.75rem;">
              <button class="btn btn-secondary btn-block" onclick="runThresholdSweep()">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                Compute Detection Threshold Sweep (0.1 - 0.9)
              </button>
            </div>
          </div>
        </div>

        <div>
          <div class="card">
            <div class="card-title">Bitemporal Change Confusion Scorecard</div>
            <table class="cm-table">
              <tr>
                <td></td>
                <td style="text-align:center; font-size:0.75rem; color:var(--text-muted); font-weight:600;">ACTUAL DEFOREST</td>
                <td style="text-align:center; font-size:0.75rem; color:var(--text-muted); font-weight:600;">ACTUAL INTACT</td>
              </tr>
              <tr>
                <td style="font-size:0.75rem; color:var(--text-muted); font-weight:600; vertical-align:middle;">DETECTED DEFOREST</td>
                <td class="cm-cell tp">
                  <div style="font-size:0.70rem; color:var(--text-muted);">TRUE POSITIVE</div>
                  <div class="cm-count" id="cmTP">--</div>
                </td>
                <td class="cm-cell fp">
                  <div style="font-size:0.70rem; color:var(--text-muted);">FALSE POSITIVE</div>
                  <div class="cm-count" id="cmFP">--</div>
                </td>
              </tr>
              <tr>
                <td style="font-size:0.75rem; color:var(--text-muted); font-weight:600; vertical-align:middle;">DETECTED INTACT</td>
                <td class="cm-cell fn">
                  <div style="font-size:0.70rem; color:var(--text-muted);">FALSE NEGATIVE</div>
                  <div class="cm-count" id="cmFN">--</div>
                </td>
                <td class="cm-cell tn">
                  <div style="font-size:0.70rem; color:var(--text-muted);">TRUE NEGATIVE</div>
                  <div class="cm-count" id="cmTN">--</div>
                </td>
              </tr>
            </table>

            <div class="metrics-row" style="margin-top: 1rem;">
              <div class="metric-cell emerald">
                <div class="label">IoU Score</div>
                <div class="val" id="bmIoU">--</div>
              </div>
              <div class="metric-cell cyan">
                <div class="label">F1 / Dice</div>
                <div class="val" id="bmF1">--</div>
              </div>
              <div class="metric-cell">
                <div class="label">Precision</div>
                <div class="val" id="bmPrecision">--</div>
              </div>
              <div class="metric-cell">
                <div class="label">Recall</div>
                <div class="val" id="bmRecall">--</div>
              </div>
              <div class="metric-cell">
                <div class="label">Specificity</div>
                <div class="val" id="bmSpecificity">--</div>
              </div>
              <div class="metric-cell">
                <div class="label">Accuracy</div>
                <div class="val" id="bmAccuracy">--</div>
              </div>
            </div>
          </div>

          <div class="card" id="sweepCard" style="display: none;">
            <div class="card-title">Threshold Sensitivity & Trade-Off Sweep Curve</div>
            <div style="height: 240px;">
              <canvas id="sweepChart"></canvas>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: INTERACTIVE SATELLITE CHIP & DEFORESTATION MASK VIEWER -->
    <div id="tab-inspector" class="tab-content">
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
          <div style="display:flex; align-items:center; gap:0.6rem; flex-wrap:wrap;">
            <div class="card-title" style="margin-bottom:0;">Satellite Chip & Deforestation Mask Viewer: <span id="inspChipTitle" style="color:var(--accent-cyan); font-family:var(--font-mono);">chip_0000</span></div>
            <div class="tele-pill">REFERENCE GT: <strong id="inspGtPct" style="color:var(--accent-crimson);">--%</strong></div>
            <div class="tele-pill">DETECTED: <strong id="inspPredPct" style="color:var(--accent-amber);">--%</strong></div>
            <div class="tele-pill">CHANGE IoU: <strong id="inspIoU" style="color:var(--accent-emerald);">--</strong></div>
            <div class="tele-pill">ALERTS: <strong id="inspAlertsCount" style="color:var(--accent-cyan);">--</strong></div>
            <div class="tele-pill">AREA: <strong id="inspAreaHa" style="color:var(--accent-amber);">-- ha</strong></div>
          </div>
          <div style="display:flex; align-items:center; gap:0.5rem;">
            <input type="number" id="inspChipIdx" class="form-control" style="width: 80px;" value="0" min="0">
            <button class="btn btn-secondary" onclick="stepChip(-1)">Prev</button>
            <button class="btn btn-secondary" onclick="stepChip(1)">Next</button>
            <button class="btn btn-cyan" onclick="inspectCurrentChip()">Inspect Chip</button>
          </div>
        </div>
      </div>

      <!-- Interactive Deforestation Mask Overlay Viewport with Opacity Slider -->
      <div class="card">
        <div class="card-title">
          <span>Interactive Deforestation Mask Overlay</span>
          <span style="font-size:0.75rem; color:var(--text-muted); font-weight:400;">Sentinel-2 Optical (T2) with Dynamic Change Mask Blending</span>
        </div>

        <div class="slider-control-card">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.5rem;">
            <div style="font-weight:600; font-size:0.85rem; color:var(--text);">
              Deforestation Mask Opacity: <strong id="maskOpacityDisplay" style="color:var(--accent-crimson); font-family:var(--font-mono); font-size:0.95rem;">50%</strong>
            </div>
            <div style="display:flex; gap:0.4rem; align-items:center;">
              <span style="font-size:0.72rem; color:var(--text-muted); margin-right:0.25rem;">Quick Presets:</span>
              <button class="btn btn-secondary" style="padding:2px 8px; font-size:0.72rem;" onclick="setMaskOpacity(0)">0% (Optical)</button>
              <button class="btn btn-secondary" style="padding:2px 8px; font-size:0.72rem;" onclick="setMaskOpacity(35)">35%</button>
              <button class="btn btn-secondary" style="padding:2px 8px; font-size:0.72rem;" onclick="setMaskOpacity(50)">50%</button>
              <button class="btn btn-secondary" style="padding:2px 8px; font-size:0.72rem;" onclick="setMaskOpacity(75)">75%</button>
              <button class="btn btn-secondary" style="padding:2px 8px; font-size:0.72rem;" onclick="setMaskOpacity(100)">100% (Solid)</button>
            </div>
          </div>
          <input type="range" id="maskOpacitySlider" min="0" max="100" value="50" class="form-range" oninput="onOpacitySliderChange(this.value)">
        </div>

        <!-- Bitemporal Split Comparison -->
        <div class="grid-2-equal" style="gap:1.25rem; margin-top:1rem;">
          <div>
            <div style="font-size:0.78rem; font-weight:600; color:var(--accent-cyan); margin-bottom:0.4rem; display:flex; justify-content:space-between;">
              <span>Pre-Disturbance Baseline (T1 Sentinel-2 RGB)</span>
              <span style="color:var(--text-muted); font-size:0.70rem;">Acquisition T1</span>
            </div>
            <div class="viewport-overlay-box">
              <div class="overlay-badge">T1 OPTICAL BASELINE</div>
              <img id="inspBaseT1Img" src="" alt="T1 True Color Optical" class="overlay-base-img">
            </div>
          </div>
          <div>
            <div style="font-size:0.78rem; font-weight:600; color:var(--accent-crimson); margin-bottom:0.4rem; display:flex; justify-content:space-between;">
              <span>Post-Disturbance Observation (T2 Optical + Live Red Deforestation Mask)</span>
              <span style="color:var(--text-muted); font-size:0.70rem;">Acquisition T2</span>
            </div>
            <div class="viewport-overlay-box" id="viewportOverlayBox">
              <div class="overlay-badge" style="color:var(--accent-crimson);">T2 + DEFORESTATION OVERLAY</div>
              <img id="inspBaseT2Img" src="" alt="T2 True Color Optical" class="overlay-base-img">
              <img id="inspOverlayMaskImg" src="" alt="Deforestation Mask" class="overlay-mask-img" style="opacity: 0.50;">
            </div>
          </div>
        </div>
      </div>

      <!-- Multi-Spectral 11-Layer Gallery -->
      <div class="card">
        <div class="card-title">
          <span>Complete Multi-Spectral & Synthetic Aperture Radar (SAR) Layers</span>
          <span style="font-size:0.75rem; color:var(--text-muted); font-weight:400;">Optical, Vegetation Indices, Dual-Pol SAR & Ground Truth</span>
        </div>
        <div class="layers-grid" id="layersContainer">
          <!-- Populated dynamically -->
        </div>
      </div>
    </div>

    <!-- TAB 3: MODEL PICKER & MANUAL INFERENCE STUDIO -->
    <div id="tab-manual-infer" class="tab-content">
      <div class="grid-2">
        <div>
          <!-- Model Selection Card -->
          <div class="card">
            <div class="card-title">1. Model Checkpoint Picker</div>
            <div class="form-group">
              <label>Model Architecture</label>
              <select id="manualModelArch" class="form-control">
                <option value="siamese" selected>Siamese Bitemporal Change Detector</option>
                <option value="multimodal">Multimodal Bitemporal Change Net (SAR + Optical)</option>
                <option value="unet">UNet Bitemporal Baseline</option>
              </select>
            </div>
            <div class="form-group">
              <label>Select Checkpoint from Storage Zoo</label>
              <select id="manualModelSelect" class="form-control" onchange="onManualModelDropdownChange()">
                <option value="">Active In-Memory Backbone Weights</option>
              </select>
            </div>
            <div style="text-align:center; font-size:0.72rem; color:var(--text-muted); margin:0.5rem 0;">— OR LOAD FROM DISK —</div>
            <div class="form-group">
              <label>Manual Model File Path (.pt / .pth)</label>
              <div style="display:flex; gap:0.4rem;">
                <input type="text" id="manualModelPath" class="form-control" placeholder="e.g. ./checkpoints/best_model.pt or /path/to/model.pt">
                <input type="file" id="manualModelFileInput" accept=".pt,.pth" style="display:none;" onchange="handleManualModelFileChosen(this)">
                <button class="btn btn-secondary" onclick="document.getElementById('manualModelFileInput').click()" title="Browse local disk">Browse</button>
              </div>
            </div>
            <button class="btn btn-cyan btn-block" onclick="loadManualModelFromPath()">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
              Hot-Load Model Weights
            </button>
            <div id="manualModelStatus" style="font-size:0.72rem; margin-top:0.5rem; color:var(--text-muted); font-family:var(--font-mono); word-break:break-all;"></div>
          </div>

          <!-- Manual Inference Input Card -->
          <div class="card">
            <div class="card-title">2. Input Data & Inference Controls</div>
            <div class="form-group">
              <label>Satellite Chip Input</label>
              <select id="manualChipSourceSelect" class="form-control">
                <option value="sim_deforest">Dynamic Disturbed Scene (12% Deforestation)</option>
                <option value="sim_intact">Dynamic Intact Canopy (0% Deforestation)</option>
                <option value="active_chip_0">Active Cache: Chip 0</option>
                <option value="active_chip_1">Active Cache: Chip 1</option>
                <option value="active_chip_2">Active Cache: Chip 2</option>
              </select>
            </div>
            <div class="form-group">
              <label>Detection Threshold: <strong id="manualThreshVal" style="color:var(--accent-cyan);">0.50</strong></label>
              <input type="range" id="manualThreshold" min="0.10" max="0.90" step="0.05" value="0.50" oninput="document.getElementById('manualThreshVal').innerText = parseFloat(this.value).toFixed(2)">
            </div>
            <div class="form-group">
              <label>Minimum Mapping Unit (MMU): <strong id="manualMmuVal" style="color:var(--accent-emerald);">0.10 ha</strong></label>
              <input type="range" id="manualMmu" min="0.02" max="1.0" step="0.02" value="0.10" oninput="document.getElementById('manualMmuVal').innerText = parseFloat(this.value).toFixed(2) + ' ha'">
            </div>
            <button class="btn btn-emerald btn-block" onclick="runManualInference()">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
              Run Inference
            </button>
          </div>
        </div>

        <!-- Output Dashboard -->
        <div>
          <!-- Scorecard -->
          <div class="card">
            <div class="card-title">
              <span>Inference Scorecard & Performance Metrics</span>
              <span id="manualInferStatusBadge" class="badge-pill badge-compliant">STANDBY</span>
            </div>
            <div class="metrics-row">
              <div class="metric-cell emerald">
                <div class="label">Detection IoU</div>
                <div class="val" id="manualIoU">--</div>
              </div>
              <div class="metric-cell cyan">
                <div class="label">F1 / Dice</div>
                <div class="val" id="manualF1">--</div>
              </div>
              <div class="metric-cell crimson">
                <div class="label">Detected Area</div>
                <div class="val" id="manualDetectedArea">-- ha</div>
              </div>
              <div class="metric-cell amber">
                <div class="label">Alert Polygons</div>
                <div class="val" id="manualAlertsCount">--</div>
              </div>
              <div class="metric-cell">
                <div class="label">Latency</div>
                <div class="val" id="manualLatency">-- ms</div>
              </div>
            </div>
          </div>

          <!-- Interactive Viewports -->
          <div class="card">
            <div class="card-title">Interactive Visual Outputs</div>
            <div class="grid-2-equal" style="gap:1rem;">
              <div>
                <div style="font-size:0.75rem; color:var(--text-muted); font-weight:600; margin-bottom:0.3rem;">T1 PRE-DISTURBANCE RGB</div>
                <div class="viewport-overlay-box">
                  <img id="manualT1Img" src="" alt="T1 Optical" class="overlay-base-img">
                </div>
              </div>
              <div>
                <div style="font-size:0.75rem; color:var(--text-muted); font-weight:600; margin-bottom:0.3rem;">T2 OBSERVATION + RED MASK</div>
                <div class="viewport-overlay-box">
                  <img id="manualT2Img" src="" alt="T2 Optical" class="overlay-base-img">
                  <img id="manualMaskImg" src="" alt="Mask" class="overlay-mask-img" style="opacity:0.55;">
                </div>
              </div>
            </div>
            <div class="grid-2-equal" style="gap:1rem; margin-top:1rem;">
              <div>
                <div style="font-size:0.75rem; color:var(--text-muted); font-weight:600; margin-bottom:0.3rem;">PROBABILITY HEATMAP</div>
                <div class="viewport-overlay-box">
                  <img id="manualHeatmapImg" src="" alt="Heatmap" class="overlay-base-img">
                </div>
              </div>
              <div>
                <div style="font-size:0.75rem; color:var(--text-muted); font-weight:600; margin-bottom:0.3rem;">BINARY DETECTION MASK</div>
                <div class="viewport-overlay-box">
                  <img id="manualBinaryImg" src="" alt="Binary Mask" class="overlay-base-img">
                </div>
              </div>
            </div>
          </div>

          <!-- Alerts Table & Export -->
          <div class="card">
            <div class="card-title">
              <span>Extracted Deforestation Polygons (GeoJSON)</span>
              <button class="btn btn-secondary" style="padding:3px 8px; font-size:0.72rem;" onclick="exportManualGeoJSON()">Export GeoJSON</button>
            </div>
            <table class="data-table">
              <thead>
                <tr>
                  <th>Polygon ID</th>
                  <th>Severity</th>
                  <th>Area (ha)</th>
                  <th>Mean Prob</th>
                  <th>Centroid</th>
                </tr>
              </thead>
              <tbody id="manualAlertsTableBody">
                <tr><td colspan="5" style="color:var(--text-muted); text-align:center;">Run inference to extract polygon alerts.</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: DATASET & INPUT LOADER -->
    <div id="tab-input-loader" class="tab-content">
      <div class="config-grid-4">
        <!-- Ingestion Source 1: Bring Your Own Data (Paired Imagery Upload) -->
        <div class="card">
          <div class="card-title">1. Bring Your Own Satellite Scenes</div>
          <p style="font-size:0.72rem; color:var(--text-muted); margin-bottom:0.75rem; line-height:1.4;">
            Upload pre/post satellite imagery (.tif, .png, .jpg) or raster chips to ingest directly into your local training cache.
          </p>
          <div class="form-group">
            <label>Pre-Disturbance Scene (T1 Baseline)</label>
            <input type="file" id="byodT1File" class="form-control" accept=".tif,.tiff,.png,.jpg,.jpeg">
          </div>
          <div class="form-group">
            <label>Post-Disturbance Scene (T2 Observation)</label>
            <input type="file" id="byodT2File" class="form-control" accept=".tif,.tiff,.png,.jpg,.jpeg">
          </div>
          <div class="form-group">
            <label>Ground Truth Mask (Optional)</label>
            <input type="file" id="byodMaskFile" class="form-control" accept=".tif,.tiff,.png,.jpg,.jpeg">
          </div>
          <button class="btn btn-cyan btn-block" onclick="uploadBYODScenes()">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
            Ingest Scene Pair into Cache
          </button>
          <div id="byodUploadStatus" style="font-size:0.72rem; margin-top:0.5rem; color:var(--text-muted); font-family:var(--font-mono); word-break:break-all;"></div>
        </div>

        <!-- Ingestion Source 2: Custom Local Folder Scanner -->
        <div class="card">
          <div class="card-title">2. Local Folder Scanner</div>
          <div class="form-group">
            <label>Enter Chips Directory Path</label>
            <input type="text" id="loaderLocalFolder" class="form-control" placeholder="e.g. ./data/chips_cache or /path/to/chips" value="./data/chips_cache">
          </div>
          <p style="font-size:0.72rem; color:var(--text-muted); margin-bottom:0.85rem; line-height:1.4;">
            Point the studio to any folder on your hard drive containing <code>.npz</code> multi-spectral chips or <code>manifest.jsonl</code>.
          </p>
          <button class="btn btn-secondary btn-block" onclick="setLocalFolder()">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>
            Set Active Dataset Directory
          </button>
          <div id="loaderFolderStatus" style="font-size:0.72rem; margin-top:0.5rem; color:var(--text-muted); font-family:var(--font-mono);"></div>
        </div>

        <!-- Ingestion Source 3: ZIP Archive Ingestor -->
        <div class="card">
          <div class="card-title">3. Upload Satellite ZIP Archive</div>
          <div class="dropzone-box" onclick="document.getElementById('loaderZipFile').click()">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" style="width:32px; height:32px; stroke-width:1.5; color:var(--accent-cyan); margin-bottom:0.5rem;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
            <div style="font-size:0.80rem; font-weight:600; color:var(--text);">Select .zip archive of satellite chips</div>
            <div style="font-size:0.70rem; color:var(--text-muted); margin-top:0.25rem;">Extracts .npz chips directly into active cache</div>
          </div>
          <input type="file" id="loaderZipFile" accept=".zip" style="display:none;" onchange="uploadZipFile(this)">
          <div id="loaderZipStatus" style="font-size:0.72rem; margin-top:0.5rem; color:var(--text-muted); font-family:var(--font-mono);"></div>
        </div>

        <!-- Ingestion Source 4: Synthetic Multi-Spectral Generator -->
        <div class="card">
          <div class="card-title">4. Synthetic Generator</div>
          <div class="form-group">
            <label>Number of Chips to Synthesize</label>
            <input type="number" id="loaderSynthCount" class="form-control" value="20" min="4" max="200">
          </div>
          <p style="font-size:0.72rem; color:var(--text-muted); margin-bottom:0.85rem; line-height:1.4;">
            Instantly stream realistic Sentinel-2 and Sentinel-1 multi-spectral chips with 70:30 class balance.
          </p>
          <button class="btn btn-emerald btn-block" onclick="generateSyntheticChipsFromLoader()">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
            Generate Synthetic Chips
          </button>
        </div>
      </div>

      <!-- Active Dataset Inspector Table -->
      <div class="card" style="margin-top:1.25rem;">
        <div class="card-title">
          <span>Active Cached Chips Dataset</span>
          <div style="display:flex; gap:0.5rem;">
            <button class="btn btn-secondary" style="padding:3px 8px; font-size:0.72rem;" onclick="refreshDatasetTable()">Refresh Dataset</button>
          </div>
        </div>
        <table class="data-table">
          <thead>
            <tr>
              <th>Chip ID</th>
              <th>File Size</th>
              <th>Deforestation Area %</th>
              <th>Class Label Quota</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="loaderChipsTableBody">
            <tr><td colspan="5" style="color:var(--text-muted); text-align:center;">Scanning cached chips...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 4: GEOSPATIAL ALERT MAP -->
    <div id="tab-map" class="tab-content">
      <div class="grid-2">
        <div>
          <div class="card">
            <div class="card-title">Geospatial Bitemporal Change Simulation</div>
            <div class="form-group">
              <label>Target Deforestation Hotspot / Region</label>
              <select id="mapHotspotSelect" onchange="onHotspotChange()">
                <option value="amazon_arc" selected>Amazon Arc of Deforestation (Manaus, Brazil) [-3.05, -60.05]</option>
                <option value="congo_basin">Congo Basin Rainforest (DR Congo) [0.50, 18.30]</option>
                <option value="borneo_kalimantan">Borneo Peatland Frontier (Kalimantan, Indonesia) [-2.20, 113.90]</option>
                <option value="cerrado_matopiba">Cerrado MATOPIBA Soy Belt (Brazil) [-10.10, -45.50]</option>
                <option value="gran_chaco">Gran Chaco Dry Tropical Forest (Paraguay/Argentina) [-24.30, -62.10]</option>
                <option value="custom">Custom Geographic Coordinates (Click on Map)</option>
              </select>
            </div>
            <div class="grid-2-equal">
              <div class="form-group">
                <label>Longitude (West/East)</label>
                <input type="number" step="0.001" id="mapWest" class="form-control" value="-60.05">
              </div>
              <div class="form-group">
                <label>Latitude (North/South)</label>
                <input type="number" step="0.001" id="mapNorth" class="form-control" value="-3.05">
              </div>
            </div>
            <div class="checkbox-row" style="margin-bottom:0.75rem;">
              <input type="checkbox" id="mapRandomSeed" checked>
              <label for="mapRandomSeed">Dynamic Bitemporal Scene (Randomize Satellite Passes)</label>
            </div>
            <div class="form-group">
              <label>Change Detection Engine</label>
              <select id="mapModelType">
                <option value="siamese" selected>Bitemporal Siamese Change Detector</option>
                <option value="multimodal">Sentinel-1/2 Multimodal Change Net</option>
                <option value="unet">UNet Bitemporal Change Baseline</option>
                <option value="baseline_ndvi">ΔNDVI Spectral Differencing Heuristic</option>
              </select>
            </div>
            <div class="form-group">
              <label>Deforestation Fraction: <strong id="mapDefVal" style="color:var(--accent-crimson);">0.08</strong></label>
              <input type="range" id="mapDef" min="0.01" max="0.30" step="0.01" value="0.08" oninput="document.getElementById('mapDefVal').innerText = parseFloat(this.value).toFixed(2)">
            </div>
            <div class="form-group">
              <label>Cloud Contamination: <strong id="mapCloudVal" style="color:var(--text-muted);">0.15</strong></label>
              <input type="range" id="mapCloud" min="0.0" max="0.50" step="0.05" value="0.15" oninput="document.getElementById('mapCloudVal').innerText = parseFloat(this.value).toFixed(2)">
            </div>
            <div class="form-group">
              <label>Minimum Mapping Unit (MMU Filter): <strong id="mapMmuVal" style="color:var(--accent-cyan);">0.10 ha</strong></label>
              <input type="range" id="mapMmu" min="0.02" max="1.0" step="0.02" value="0.10" oninput="document.getElementById('mapMmuVal').innerText = parseFloat(this.value).toFixed(2) + ' ha'">
            </div>
            <button class="btn btn-emerald btn-block" onclick="runGeospatialInference()">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"></polygon></svg>
              Run Geospatial Detection
            </button>
            <div style="margin-top:0.65rem;">
              <button class="btn btn-secondary btn-block" onclick="exportGeoJSON()">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                Export Alerts GeoJSON
              </button>
            </div>
          </div>

          <div class="metrics-row">
            <div class="metric-cell crimson">
              <div class="label">Total Alerts</div>
              <div class="val" id="mapAlertsCount">--</div>
            </div>
            <div class="metric-cell amber">
              <div class="label">Deforested Area</div>
              <div class="val" id="mapDefArea">-- ha</div>
            </div>
          </div>
        </div>

        <div>
          <div class="card" style="padding:0.75rem;">
            <div id="map"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 5: EUDR DUE DILIGENCE AUDIT -->
    <div id="tab-eudr" class="tab-content">
      <div class="grid-2">
        <div>
          <div class="card">
            <div class="card-title">Due Diligence Audit (EUDR)</div>
            <div style="margin-bottom:0.75rem;">
              <button class="btn btn-cyan btn-block" onclick="importActiveAlertsToEUDR()">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"></polygon></svg>
                Import Detected Alerts from Map into Audit
              </button>
            </div>
            <div class="form-group">
              <label>Regulatory Preset Scenarios</label>
              <select id="eudrPresetSelect" onchange="applyPresetScenario()">
                <option value="">-- Choose Standard Farm Case --</option>
              </select>
            </div>
            <div class="form-group">
              <label>DDS Statement ID</label>
              <input type="text" id="eudrStatementId" class="form-control" value="DDS-2024-TEST-001">
            </div>
            <div class="form-group">
              <label>Operator Legal Entity</label>
              <input type="text" id="eudrOperator" class="form-control" value="AgriCommodities International">
            </div>
            <div class="grid-2-equal">
              <div class="form-group">
                <label>Annex I Commodity</label>
                <select id="eudrCommodity">
                  <option value="soya" selected>Soya</option>
                  <option value="oil palm">Oil Palm</option>
                  <option value="cattle">Cattle (Beef/Leather)</option>
                  <option value="cocoa">Cocoa</option>
                  <option value="coffee">Coffee</option>
                  <option value="rubber">Rubber</option>
                  <option value="wood">Wood / Timber</option>
                </select>
              </div>
              <div class="form-group">
                <label>Country of Production</label>
                <input type="text" id="eudrCountry" class="form-control" value="Brazil">
              </div>
            </div>
            <button class="btn btn-emerald btn-block" onclick="executeEUDRAudit()">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
              Execute Due Diligence Audit
            </button>
            <div class="grid-2-equal" style="margin-top:0.75rem;">
              <div class="form-group" style="margin-bottom:0;">
                <label>Olofsson Sample Source</label>
                <select id="olofSource" class="form-control" style="font-size:0.75rem;">
                  <option value="reference" selected>Reference Benchmark (Olofsson 2014)</option>
                  <option value="benchmark">Live Model Benchmark Matrix</option>
                </select>
              </div>
              <div class="form-group" style="margin-bottom:0;">
                <label>Stratum Forest Area (ha)</label>
                <input type="number" id="olofTotalArea" class="form-control" value="18750" style="font-size:0.75rem;">
              </div>
            </div>
            <div style="margin-top:0.5rem;">
              <button class="btn btn-secondary btn-block" onclick="executeOlofsson()">
                Compute Olofsson Stratified Accuracy (95% CI)
              </button>
            </div>
          </div>
        </div>

        <div>
          <div class="card">
            <div class="card-title">
              Audit Assessment Result
              <span id="eudrStatusBadge" class="badge-pill badge-compliant">STANDBY</span>
            </div>
            <div class="metrics-row">
              <div class="metric-cell" id="eudrRiskCell">
                <div class="label">Compliance Risk Score</div>
                <div class="val" id="eudrRiskScore">0.00</div>
              </div>
              <div class="metric-cell">
                <div class="label">Plots Assessed</div>
                <div class="val" id="eudrPlotsCount">0</div>
              </div>
              <div class="metric-cell">
                <div class="label">Cutoff Date</div>
                <div class="val" style="font-size:0.95rem; margin-top:0.4rem;">2020-12-31</div>
              </div>
            </div>
            <div id="eudrViolationBox" style="font-size:0.85rem; line-height:1.6; color:var(--text-muted);">
              Select a scenario or audit statement to evaluate compliance against EU Regulation 2023/1115.
            </div>
          </div>

          <div class="card" id="olofssonCard" style="display:none;">
            <div class="card-title">Olofsson et al. (2014) Area-Adjusted Verification</div>
            <div class="metrics-row">
              <div class="metric-cell emerald">
                <div class="label">Overall Accuracy</div>
                <div class="val" id="olofOA">--%</div>
              </div>
              <div class="metric-cell">
                <div class="label">Margin of Error (95% CI)</div>
                <div class="val" id="olofME">±--%</div>
              </div>
              <div class="metric-cell cyan">
                <div class="label">Deforestation Area</div>
                <div class="val" id="olofArea">-- ha</div>
              </div>
            </div>
            <div id="olofSummary" style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted); white-space:pre-wrap;"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 6: CHECKPOINTS & DATASET ZOO -->
    <div id="tab-zoo" class="tab-content">
      <div class="grid-2-equal">
        <div class="card">
          <div class="card-title">
            Model Checkpoints
            <button class="btn btn-secondary" style="padding:3px 8px; font-size:0.72rem;" onclick="loadCheckpoints()">Refresh</button>
          </div>
          <table class="data-table">
            <thead>
              <tr>
                <th>Checkpoint File</th>
                <th>Size (MB)</th>
                <th>Epoch</th>
                <th>Val IoU</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody id="checkpointsTableBody">
              <tr><td colspan="5" style="color:var(--text-muted); text-align:center;">Scanning checkpoints directory...</td></tr>
            </tbody>
          </table>
        </div>

        <div class="card">
          <div class="card-title">
            Cached Chips Zoo
            <button class="btn btn-cyan" style="padding:3px 8px; font-size:0.72rem;" onclick="generateSyntheticChips()">Generate 70:30 Chips</button>
          </div>
          <table class="data-table">
            <thead>
              <tr>
                <th>Chip ID</th>
                <th>Size</th>
                <th>Loss %</th>
                <th>Class Quota</th>
              </tr>
            </thead>
            <tbody id="chipsTableBody">
              <tr><td colspan="4" style="color:var(--text-muted); text-align:center;">No cached chips in ./data/chips_cache</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

  </main>

  <!-- Studio Javascript Engine -->
  <script>
    let trainingChart = null;
    let sweepChart = null;
    let map = null;
    let mapAlertsLayer = null;
    let activeGeoJSON = null;
    let eudrPresetsData = {};
    let isTrainingActive = false;
    let trainingPollInterval = null;

    // Tab Navigation
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
      if (activeBtn) activeBtn.classList.add('active');
      const target = document.getElementById('tab-' + tabId);
      if (target) target.classList.add('active');
      if (tabId === 'map' && map) {
        setTimeout(() => map.invalidateSize(), 150);
      }
    }

    // Chart Initializations
    function initCharts() {
      const tCtx = document.getElementById('trainingChart').getContext('2d');
      trainingChart = new Chart(tCtx, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            { label: 'Train Loss', data: [], borderColor: '#EF4444', backgroundColor: 'transparent', tension: 0.3, yAxisID: 'y' },
            { label: 'Val Loss', data: [], borderColor: '#F59E0B', backgroundColor: 'transparent', tension: 0.3, yAxisID: 'y' },
            { label: 'Val IoU', data: [], borderColor: '#10B981', backgroundColor: 'transparent', tension: 0.3, yAxisID: 'y1' }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94A3B8', font: { family: 'Fira Code' } } },
            y: { position: 'left', grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94A3B8', font: { family: 'Fira Code' } } },
            y1: { position: 'right', min: 0, max: 1, grid: { drawOnChartArea: false }, ticks: { color: '#10B981', font: { family: 'Fira Code' } } }
          },
          plugins: {
            legend: { labels: { color: '#F8FAFC', font: { family: 'Inter', size: 11 } } }
          }
        }
      });
    }

    // Hotspot Regions & Map Initialization
    const HOTSPOTS = {
      amazon_arc: [-3.05, -60.05, 11],
      congo_basin: [0.50, 18.30, 11],
      borneo_kalimantan: [-2.20, 113.90, 11],
      cerrado_matopiba: [-10.10, -45.50, 11],
      gran_chaco: [-24.30, -62.10, 11]
    };

    function onHotspotChange() {
      const key = document.getElementById('mapHotspotSelect').value;
      if (HOTSPOTS[key]) {
        const [lat, lng, zoom] = HOTSPOTS[key];
        document.getElementById('mapNorth').value = lat.toFixed(4);
        document.getElementById('mapWest').value = lng.toFixed(4);
        if (map) map.flyTo([lat, lng], zoom);
      }
    }

    function initMap() {
      map = L.map('map', { zoomControl: false }).setView([-3.05, -60.05], 11);
      L.control.zoom({ position: 'topright' }).addTo(map);
      // Dark Basemap (CartoDB Dark Matter with OSM Fallback)
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; CartoDB & OpenStreetMap contributors',
        maxZoom: 19
      }).addTo(map);

      map.on('click', function(e) {
        document.getElementById('mapNorth').value = e.latlng.lat.toFixed(4);
        document.getElementById('mapWest').value = e.latlng.lng.toFixed(4);
        document.getElementById('mapHotspotSelect').value = 'custom';
      });
    }

    // Telemetry Polling (Dynamic RAM / Host Device)
    async function updateTelemetry() {
      try {
        const resp = await fetch('/studio/api/telemetry');
        const data = await resp.json();
        document.getElementById('teleDevice').innerText = data.device_name;
        if (data.ram_total_mb > 1024) {
          const usedGb = (data.ram_used_mb / 1024).toFixed(1);
          const totalGb = (data.ram_total_mb / 1024).toFixed(1);
          document.getElementById('teleRam').innerText = `${usedGb}GB / ${totalGb}GB (${data.ram_pct}%)`;
        } else if (data.ram_total_mb > 0) {
          document.getElementById('teleRam').innerText = `${data.ram_used_mb}MB / ${data.ram_total_mb}MB (${data.ram_pct}%)`;
        } else {
          document.getElementById('teleRam').innerText = `${data.cpu_count} Cores Active`;
        }
        document.getElementById('teleChips').innerText = data.cached_chips_count;
        document.getElementById('teleCheckpoints').innerText = data.checkpoint_count;
      } catch (e) {
        console.warn('Telemetry update error:', e);
      }
    }

    // Training Actions (All 16 Training Configuration Options)
    async function startTraining() {
      const payload = {
        dataset_source: document.getElementById('trainDatasetSource') ? document.getElementById('trainDatasetSource').value : 'cached',
        dataset_scale: document.getElementById('trainDatasetScale') ? document.getElementById('trainDatasetScale').value : 'standard',
        cache_dir: document.getElementById('trainDataDir') ? document.getElementById('trainDataDir').value : './data/chips_cache',
        chip_size: parseInt(document.getElementById('trainChipSize') ? document.getElementById('trainChipSize').value : 256),
        architecture: document.getElementById('trainArch').value,
        use_lora: document.getElementById('trainUseLora').checked,
        lora_rank: parseInt(document.getElementById('trainLoraRank').value),
        lora_alpha: parseFloat(document.getElementById('trainLoraAlpha').value),
        loss_type: document.getElementById('trainLoss').value,
        tversky_alpha: parseFloat(document.getElementById('trainTverskyAlpha') ? document.getElementById('trainTverskyAlpha').value : 0.3),
        tversky_beta: parseFloat(document.getElementById('trainTverskyBeta') ? document.getElementById('trainTverskyBeta').value : 0.7),
        epochs: parseInt(document.getElementById('trainEpochs').value),
        batch_size: parseInt(document.getElementById('trainBatch').value),
        learning_rate: parseFloat(document.getElementById('trainLR').value),
        ram_cache_budget_gb: parseFloat(document.getElementById('trainRamBudget') ? document.getElementById('trainRamBudget').value : 4.0),
        use_data_augmentation: document.getElementById('trainAugmentation') ? document.getElementById('trainAugmentation').checked : true,
        enforce_class_balance: document.getElementById('trainClassBalance') ? document.getElementById('trainClassBalance').checked : true,
        use_amp: document.getElementById('trainAmp') ? document.getElementById('trainAmp').checked : true,
        resume: document.getElementById('trainResume').checked,
        async_run: true
      };

      try {
        const resp = await fetch('/studio/api/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const res = await resp.json();
        isTrainingActive = true;
        document.getElementById('trainingStatusDot').className = 'status-pulse busy';
        document.getElementById('teleTrainStatus').innerText = 'TRAINING';
        appendTerminal(`[INFO] Planetary training initiated with ${payload.architecture} architecture on ${payload.dataset_source} dataset (${payload.dataset_scale} scale).`);
        if (trainingPollInterval) clearInterval(trainingPollInterval);
        trainingPollInterval = setInterval(pollTrainingStatus, 1000);
      } catch (err) {
        appendTerminal(`[ERROR] Failed to start training: ${err}`);
      }
    }

    async function stopTraining() {
      try {
        const resp = await fetch('/studio/api/train/stop', { method: 'POST' });
        appendTerminal('[WARN] Stop signal delivered to background training loop.');
      } catch (err) {
        appendTerminal(`[ERROR] Failed to stop: ${err}`);
      }
    }

    async function pollTrainingStatus() {
      try {
        const resp = await fetch('/studio/api/train/status');
        const data = await resp.json();
        if (data.is_training) {
          document.getElementById('dispEpoch').innerText = `${data.current_epoch} / ${data.total_epochs}`;
          if (data.current_train_loss !== null) document.getElementById('dispTrainLoss').innerText = data.current_train_loss.toFixed(4);
          if (data.current_val_loss !== null) document.getElementById('dispValLoss').innerText = data.current_val_loss.toFixed(4);
          if (data.current_val_iou !== null) document.getElementById('dispValIoU').innerText = data.current_val_iou.toFixed(4);
          if (data.current_lr !== null) document.getElementById('dispLR').innerText = data.current_lr.toExponential(2);
        } else if (isTrainingActive) {
          isTrainingActive = false;
          clearInterval(trainingPollInterval);
          document.getElementById('trainingStatusDot').className = 'status-pulse';
          document.getElementById('teleTrainStatus').innerText = 'IDLE';
          appendTerminal('[INFO] Training loop completed successfully.');
          loadCheckpoints();
        }

        // Render logs
        if (data.logs && data.logs.length > 0) {
          const t = document.getElementById('trainTerminal');
          t.innerHTML = data.logs.map(l => `<div class="terminal-line">${escapeHtml(l)}</div>`).join('');
          t.scrollTop = t.scrollHeight;
        }

        // Update Chart
        if (data.history && data.history.length > 0) {
          trainingChart.data.labels = data.history.map(h => `Ep ${h.epoch}`);
          trainingChart.data.datasets[0].data = data.history.map(h => h.train_loss);
          trainingChart.data.datasets[1].data = data.history.map(h => h.val_loss);
          trainingChart.data.datasets[2].data = data.history.map(h => h.val_iou);
          trainingChart.update();
        }
      } catch (err) {
        console.warn('Poll error:', err);
      }
    }

    function appendTerminal(msg) {
      const t = document.getElementById('trainTerminal');
      const time = new Date().toISOString().substring(11, 19);
      t.innerHTML += `<div class="terminal-line">[${time}] ${escapeHtml(msg)}</div>`;
      t.scrollTop = t.scrollHeight;
    }

    // Benchmark Actions
    async function runBenchmark() {
      const payload = {
        checkpoint_name: document.getElementById('benchCheckpoint').value || null,
        num_samples: parseInt(document.getElementById('benchSamples').value),
        threshold: parseFloat(document.getElementById('benchThreshold').value)
      };

      try {
        const resp = await fetch('/studio/api/test/benchmark', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const d = await resp.json();
        document.getElementById('cmTP').innerText = d.confusion_matrix.tp.toLocaleString();
        document.getElementById('cmFP').innerText = d.confusion_matrix.fp.toLocaleString();
        document.getElementById('cmFN').innerText = d.confusion_matrix.fn.toLocaleString();
        document.getElementById('cmTN').innerText = d.confusion_matrix.tn.toLocaleString();

        document.getElementById('bmIoU').innerText = (d.iou * 100).toFixed(1) + '%';
        document.getElementById('bmF1').innerText = (d.f1 * 100).toFixed(1) + '%';
        document.getElementById('bmPrecision').innerText = (d.precision * 100).toFixed(1) + '%';
        document.getElementById('bmRecall').innerText = (d.recall * 100).toFixed(1) + '%';
        document.getElementById('bmSpecificity').innerText = (d.specificity * 100).toFixed(1) + '%';
        document.getElementById('bmAccuracy').innerText = (d.accuracy * 100).toFixed(1) + '%';
      } catch (err) {
        alert('Benchmark failed: ' + err);
      }
    }

    async function runThresholdSweep() {
      const payload = {
        checkpoint_name: document.getElementById('benchCheckpoint').value || null,
        num_samples: 8
      };

      try {
        const resp = await fetch('/studio/api/test/thresholds', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const sweepData = await resp.json();
        document.getElementById('sweepCard').style.display = 'block';

        if (!sweepChart) {
          const sCtx = document.getElementById('sweepChart').getContext('2d');
          sweepChart = new Chart(sCtx, {
            type: 'line',
            data: {
              labels: sweepData.map(s => s.threshold.toFixed(2)),
              datasets: [
                { label: 'IoU', data: sweepData.map(s => s.iou), borderColor: '#10B981', tension: 0.3 },
                { label: 'F1 Score', data: sweepData.map(s => s.f1), borderColor: '#06B6D4', tension: 0.3 },
                { label: 'Precision', data: sweepData.map(s => s.precision), borderColor: '#A78BFA', tension: 0.3 },
                { label: 'Recall', data: sweepData.map(s => s.recall), borderColor: '#F59E0B', tension: 0.3 }
              ]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                x: { title: { display: true, text: 'Threshold (τ)', color: '#94A3B8' }, ticks: { color: '#94A3B8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { min: 0, max: 1, ticks: { color: '#94A3B8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
              },
              plugins: { legend: { labels: { color: '#F8FAFC' } } }
            }
          });
        } else {
          sweepChart.data.labels = sweepData.map(s => s.threshold.toFixed(2));
          sweepChart.data.datasets[0].data = sweepData.map(s => s.iou);
          sweepChart.data.datasets[1].data = sweepData.map(s => s.f1);
          sweepChart.data.datasets[2].data = sweepData.map(s => s.precision);
          sweepChart.data.datasets[3].data = sweepData.map(s => s.recall);
          sweepChart.update();
        }
      } catch (err) {
        alert('Sweep failed: ' + err);
      }
    }

    // Dynamic Mask Opacity Controls
    function setMaskOpacity(val) {
      const slider = document.getElementById('maskOpacitySlider');
      if (slider) {
        slider.value = val;
        onOpacitySliderChange(val);
      }
    }

    function onOpacitySliderChange(val) {
      const display = document.getElementById('maskOpacityDisplay');
      if (display) display.innerText = val + '%';
      const maskImg = document.getElementById('inspOverlayMaskImg');
      if (maskImg) {
        maskImg.style.opacity = (parseFloat(val) / 100).toFixed(2);
      }
    }

    // Chip Deep Inspector & Multi-Spectral Layer Gallery
    async function inspectCurrentChip() {
      const idx = parseInt(document.getElementById('inspChipIdx').value) || 0;
      try {
        const resp = await fetch('/studio/api/test/chip', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ chip_idx: idx, threshold: 0.5, mmu_ha: 0.10 })
        });
        const d = await resp.json();
        document.getElementById('inspChipTitle').innerText = d.chip_id;
        document.getElementById('inspGtPct').innerText = d.ground_truth_pct + '%';
        document.getElementById('inspPredPct').innerText = (d.detected_pct !== undefined ? d.detected_pct : d.predicted_pct) + '%';
        document.getElementById('inspIoU').innerText = (d.chip_iou * 100).toFixed(1) + '%';
        if (document.getElementById('inspAlertsCount')) {
          document.getElementById('inspAlertsCount').innerText = d.total_alerts !== undefined ? d.total_alerts : '--';
        }
        if (document.getElementById('inspAreaHa')) {
          document.getElementById('inspAreaHa').innerText = (d.total_deforested_ha !== undefined ? d.total_deforested_ha.toFixed(2) : '--') + ' ha';
        }

        // Set dual viewport overlay images
        const t1El = document.getElementById('inspBaseT1Img');
        const t2El = document.getElementById('inspBaseT2Img');
        const maskEl = document.getElementById('inspOverlayMaskImg');
        if (t1El && d.layers.t1_rgb) t1El.src = d.layers.t1_rgb;
        if (t2El && d.layers.t2_rgb) t2El.src = d.layers.t2_rgb;
        if (maskEl) {
          maskEl.src = d.layers.pred_binary_mask || d.layers.detected_binary_mask || d.layers.mask_overlay;
          const currentOpacity = document.getElementById('maskOpacitySlider') ? (document.getElementById('maskOpacitySlider').value / 100) : 0.50;
          maskEl.style.opacity = currentOpacity;
        }

        const layers = [
          { title: 'T1 S-2 Baseline Optical (RGB)', src: d.layers.t1_rgb },
          { title: 'T2 S-2 Post-Acquisition Optical (RGB)', src: d.layers.t2_rgb },
          { title: 'Translucent Red Deforestation Overlay', src: d.layers.mask_overlay },
          { title: 'Ground Truth Reference Mask (Hansen GFC)', src: d.layers.ground_truth_mask },
          { title: 'AI Deforestation Detection Probability', src: d.layers.probability_heatmap },
          { title: 'Detected Binary Change Mask', src: d.layers.pred_binary_mask || d.layers.detected_binary_mask },
          { title: 'Bitemporal Change Confusion Diff Map', src: d.layers.error_diff_map },
          { title: 'T1 Baseline Color Infrared (CIR)', src: d.layers.t1_cir },
          { title: 'T2 Observation Color Infrared (CIR)', src: d.layers.t2_cir },
          { title: 'T1 Baseline NDVI Canopy Index', src: d.layers.ndvi_t1 },
          { title: 'T2 Observation NDVI Canopy Index', src: d.layers.ndvi_t2 },
          { title: 'Sentinel-1 SAR Radar Composite', src: d.layers.sar_composite }
        ];

        const container = document.getElementById('layersContainer');
        if (container) {
          container.innerHTML = layers.map(l => `
            <div class="layer-card">
              <div class="layer-header">
                <span>${l.title}</span>
              </div>
              <img src="${l.src}" alt="${l.title}">
            </div>
          `).join('');
        }
      } catch (err) {
        console.error('Inspect error:', err);
      }
    }

    function stepChip(delta) {
      const el = document.getElementById('inspChipIdx');
      let val = Math.max(0, (parseInt(el.value) || 0) + delta);
      el.value = val;
      inspectCurrentChip();
    }

    // Geospatial Inference
    async function runGeospatialInference() {
      const isRandom = document.getElementById('mapRandomSeed') ? document.getElementById('mapRandomSeed').checked : true;
      const westVal = parseFloat(document.getElementById('mapWest') ? document.getElementById('mapWest').value : -60.05);
      const northVal = parseFloat(document.getElementById('mapNorth') ? document.getElementById('mapNorth').value : -3.05);

      const payload = {
        model_type: document.getElementById('mapModelType').value,
        deforestation_fraction: parseFloat(document.getElementById('mapDef').value),
        cloud_fraction: parseFloat(document.getElementById('mapCloud').value),
        threshold: 0.5,
        mmu_ha: parseFloat(document.getElementById('mapMmu').value),
        west: isNaN(westVal) ? -60.05 : westVal,
        north: isNaN(northVal) ? -3.05 : northVal,
        seed: isRandom ? null : 42
      };

      try {
        const resp = await fetch('/studio/api/infer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await resp.json();
        document.getElementById('mapAlertsCount').innerText = data.total_alerts;
        document.getElementById('mapDefArea').innerText = data.total_deforested_ha.toFixed(2) + ' ha';

        activeGeoJSON = data.geojson;
        if (mapAlertsLayer) map.removeLayer(mapAlertsLayer);

        mapAlertsLayer = L.geoJSON(data.geojson, {
          style: function(feat) {
            const lvl = feat.properties.alert_level || 'MEDIUM';
            const col = lvl === 'CRITICAL' ? '#EF4444' : (lvl === 'HIGH' ? '#F59E0B' : '#06B6D4');
            return { color: col, weight: 2, fillOpacity: 0.45, fillColor: col };
          },
          onEachFeature: function(feat, layer) {
            const p = feat.properties;
            layer.bindPopup(`
              <div style="font-family:var(--font-sans); color:#0B111A; font-size:12px;">
                <strong style="color:#EF4444;">${p.alert_level || 'ALERT'}</strong><br>
                Area: <b>${p.area_ha.toFixed(2)} ha</b><br>
                Confidence: <b>${(p.confidence * 100).toFixed(0)}%</b><br>
                Date: ${p.detection_date || 'N/A'}
              </div>
            `);
          }
        }).addTo(map);

        if (data.geojson.features && data.geojson.features.length > 0) {
          map.fitBounds(mapAlertsLayer.getBounds(), { padding: [30, 30] });
        }
      } catch (err) {
        alert('Geospatial inference failed: ' + err);
      }
    }

    function exportGeoJSON() {
      if (!activeGeoJSON) {
        alert('Run geospatial inference first to generate alert polygons.');
        return;
      }
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(activeGeoJSON, null, 2));
      const a = document.createElement('a');
      a.setAttribute('href', dataStr);
      a.setAttribute('download', 'geofm_deforestation_alerts.geojson');
      document.body.appendChild(a);
      a.click();
      a.remove();
    }

    // EUDR Audit
    async function loadPresets() {
      try {
        const resp = await fetch('/studio/api/eudr/presets');
        eudrPresetsData = await resp.json();
        const sel = document.getElementById('eudrPresetSelect');
        sel.innerHTML = '<option value="">-- Choose Standard Farm Case --</option>';
        Object.keys(eudrPresetsData).forEach(k => {
          sel.innerHTML += `<option value="${escapeHtml(k)}">${escapeHtml(k)}</option>`;
        });
      } catch (e) {
        console.warn('Failed loading presets:', e);
      }
    }

    let liveImportedDDS = null;

    function importActiveAlertsToEUDR() {
      if (!activeGeoJSON || !activeGeoJSON.features || activeGeoJSON.features.length === 0) {
        alert('Run Geospatial Detection in the Map tab first to generate alert polygons.');
        return;
      }
      document.getElementById('eudrPresetSelect').value = '';
      const stmtId = 'DDS-LIVE-' + Date.now().toString().slice(-5);
      document.getElementById('eudrStatementId').value = stmtId;
      document.getElementById('eudrOperator').value = 'Audited Frontier Producer S.A.';

      const ddsAlerts = activeGeoJSON.features.map((f, i) => ({
        alert_id: f.properties.alert_id || `ALT-${i+1}`,
        geometry: f.geometry,
        detection_date: f.properties.detection_date || '2023-08-15T12:00:00Z',
        area_ha: f.properties.area_ha || 2.5
      }));

      const firstGeom = activeGeoJSON.features[0].geometry;
      const plotCoords = firstGeom.type === 'Polygon'
        ? firstGeom.coordinates
        : [[[-60.05, -3.05], [-60.01, -3.05], [-60.01, -3.01], [-60.05, -3.01], [-60.05, -3.05]]];

      const plot = {
        plot_id: 'PLOT-AUDITED-01',
        geometry_type: 'Polygon',
        coordinates: plotCoords,
        declared_area_ha: parseFloat(document.getElementById('mapDefArea').innerText || 10.0) * 1.5
      };

      liveImportedDDS = {
        statement_id: stmtId,
        operator_name: 'Audited Frontier Producer S.A.',
        commodity: document.getElementById('eudrCommodity').value || 'soya',
        country: document.getElementById('eudrCountry').value || 'Brazil',
        plots: [plot],
        alerts: ddsAlerts
      };

      alert(`Imported ${ddsAlerts.length} detected deforestation polygons into Due Diligence audit! Press 'Execute Due Diligence Audit' to evaluate.`);
    }

    function applyPresetScenario() {
      liveImportedDDS = null;
      const key = document.getElementById('eudrPresetSelect').value;
      if (!key || !eudrPresetsData[key]) return;
      const sc = eudrPresetsData[key];
      document.getElementById('eudrStatementId').value = sc.statement_id;
      document.getElementById('eudrOperator').value = sc.operator_name;
      document.getElementById('eudrCommodity').value = sc.commodity;
      document.getElementById('eudrCountry').value = sc.country;
    }

    async function executeEUDRAudit(customPayload = null) {
      const key = document.getElementById('eudrPresetSelect').value;
      const sc = customPayload || liveImportedDDS || (key && eudrPresetsData[key] ? eudrPresetsData[key] : {
        statement_id: document.getElementById('eudrStatementId').value,
        operator_name: document.getElementById('eudrOperator').value,
        commodity: document.getElementById('eudrCommodity').value,
        country: document.getElementById('eudrCountry').value,
        plots: [{ plot_id: 'PLOT-LIVE-01', geometry_type: 'Point', coordinates: [parseFloat(document.getElementById('mapWest').value || -60.05), parseFloat(document.getElementById('mapNorth').value || -3.05)], declared_area_ha: 2.5 }],
        alerts: []
      });

      try {
        const resp = await fetch('/studio/api/eudr/audit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(sc)
        });
        const rep = await resp.json();
        const badge = document.getElementById('eudrStatusBadge');
        if (rep.overall_status === 'COMPLIANT') {
          badge.className = 'badge-pill badge-compliant';
          badge.innerText = 'COMPLIANT (APPROVED)';
        } else {
          badge.className = 'badge-pill badge-violation';
          badge.innerText = 'NON-COMPLIANT (FLAGGED)';
        }
        document.getElementById('eudrRiskScore').innerText = rep.risk_score.toFixed(2);
        document.getElementById('eudrPlotsCount').innerText = rep.plots.length;

        let html = '';
        rep.plots.forEach(p => {
          if (p.violations && p.violations.length > 0) {
            html += `<div style="color:var(--accent-crimson); margin-bottom:6px;"><b>Violation [${p.plot_id}]:</b> ${p.violations.join(', ')}</div>`;
          } else {
            html += `<div style="color:var(--accent-emerald); margin-bottom:6px;"><b>Compliant [${p.plot_id}]:</b> Passed statutory EUDR criteria (≤4ha point rule, post-2020 zero-deforestation).</div>`;
          }
        });
        document.getElementById('eudrViolationBox').innerHTML = html;
      } catch (err) {
        alert('Audit execution error: ' + err);
      }
    }

    async function executeOlofsson() {
      try {
        const source = document.getElementById('olofSource') ? document.getElementById('olofSource').value : 'reference';
        const forestArea = parseFloat(document.getElementById('olofTotalArea') ? document.getElementById('olofTotalArea').value : 18750) || 18750;
        const payload = {
          source: source,
          stratum_forest_ha: forestArea,
          stratum_deforest_ha: forestArea * 0.08
        };
        const resp = await fetch('/studio/api/olofsson', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const o = await resp.json();
        document.getElementById('olofssonCard').style.display = 'block';
        document.getElementById('olofOA').innerText = o.overall_accuracy_pct + '%';
        document.getElementById('olofME').innerText = '±' + o.oa_margin_of_error_pct + '%';
        document.getElementById('olofArea').innerText = o.estimated_areas_ha['Deforestation'] + ' ha';
        document.getElementById('olofSummary').innerText = o.summary_text;
      } catch (err) {
        alert('Olofsson assessment failed: ' + err);
      }
    }

    // Checkpoints & Zoo
    async function loadCheckpoints() {
      try {
        const resp = await fetch('/studio/api/checkpoints');
        const chks = await resp.json();
        const tbody = document.getElementById('checkpointsTableBody');
        const benchSel = document.getElementById('benchCheckpoint');
        const manualSel = document.getElementById('manualModelSelect');

        if (benchSel) benchSel.innerHTML = '<option value="">Active Siamese Backbone (In-Memory)</option>';
        if (manualSel) manualSel.innerHTML = '<option value="">Active In-Memory Backbone Weights</option>';
        if (chks.length === 0) {
          if (tbody) tbody.innerHTML = '<tr><td colspan="5" style="color:var(--text-muted); text-align:center;">No .pt checkpoints found.</td></tr>';
          return;
        }

        if (tbody) {
          tbody.innerHTML = chks.map(c => `
            <tr>
              <td><strong>${escapeHtml(c.filename)}</strong></td>
              <td>${c.size_mb} MB</td>
              <td>${c.epoch}</td>
              <td style="color:var(--accent-emerald);">${typeof c.best_val_iou === 'number' ? (c.best_val_iou * 100).toFixed(1) + '%' : c.best_val_iou}</td>
              <td>
                <a href="/studio/api/download/${encodeURIComponent(c.filename)}" class="btn btn-secondary" style="padding:2px 7px; font-size:0.70rem;">Download</a>
                <button class="btn btn-cyan" style="padding:2px 7px; font-size:0.70rem;" onclick="loadCheckpointForInference('${escapeHtml(c.filename)}')">Load</button>
              </td>
            </tr>
          `).join('');
        }

        chks.forEach(c => {
          if (benchSel) benchSel.innerHTML += `<option value="${escapeHtml(c.filename)}">${escapeHtml(c.filename)} (${c.size_mb} MB)</option>`;
          if (manualSel) manualSel.innerHTML += `<option value="${escapeHtml(c.filename)}">${escapeHtml(c.filename)} (${c.size_mb} MB)</option>`;
        });
      } catch (err) {
        console.warn('Failed loading checkpoints:', err);
      }
    }

    async function loadCheckpointForInference(fname) {
      try {
        const resp = await fetch('/studio/api/checkpoints/load', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: fname })
        });
        const res = await resp.json();
        alert(res.message || 'Checkpoint loaded into active inference backbone.');
      } catch (err) {
        alert('Failed loading checkpoint: ' + err);
      }
    }

    async function loadChipsZoo() {
      try {
        const resp = await fetch('/studio/api/chips');
        const data = await resp.json();
        const tbody = document.getElementById('chipsTableBody');
        if (!tbody) return;
        if (!data.chips || data.chips.length === 0) {
          tbody.innerHTML = '<tr><td colspan="4" style="color:var(--text-muted); text-align:center;">No chips cached. Click "Generate 70:30 Chips" above.</td></tr>';
          return;
        }
        tbody.innerHTML = data.chips.slice(0, 15).map(c => `
          <tr>
            <td><code>${escapeHtml(c.chip_id)}</code></td>
            <td>${c.size_kb} KB</td>
            <td style="color:${c.deforestation_pct > 0 ? '#EF4444' : '#10B981'};">${c.deforestation_pct}%</td>
            <td><span class="badge-pill ${c.class_label === 'DEFOREST' ? 'badge-violation' : 'badge-compliant'}">${c.class_label}</span></td>
          </tr>
        `).join('');
      } catch (err) {
        console.warn('Failed loading chips zoo:', err);
      }
    }

    async function generateSyntheticChips() {
      try {
        const resp = await fetch('/studio/api/chips/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ num_chips: 20, deforest_ratio: 0.70 })
        });
        const data = await resp.json();
        alert(`Successfully generated ${data.created_chips} chips! (${data.deforest_chips} DEFOREST, ${data.intact_chips} INTACT enforcing 70:30 quota)`);
        loadChipsZoo();
        refreshDatasetTable();
        updateTelemetry();
      } catch (err) {
        alert('Chip generation error: ' + err);
      }
    }

    // =========================================================================
    // Tab 3: Model Picker & Manual Inference Studio JavaScript
    // =========================================================================
    let manualActiveGeoJSON = null;

    function onManualModelDropdownChange() {
      const sel = document.getElementById('manualModelSelect');
      const pathInput = document.getElementById('manualModelPath');
      if (sel && sel.value) {
        if (pathInput) pathInput.value = sel.value;
      }
    }

    function handleManualModelFileChosen(input) {
      if (input.files && input.files[0]) {
        const file = input.files[0];
        const pathInput = document.getElementById('manualModelPath');
        if (pathInput) pathInput.value = file.name;
        const statusBox = document.getElementById('manualModelStatus');
        if (statusBox) statusBox.innerHTML = `<span style="color:var(--accent-cyan);">Selected local file: ${escapeHtml(file.name)}</span>`;
      }
    }

    async function loadManualModelFromPath() {
      const pathVal = document.getElementById('manualModelPath') ? document.getElementById('manualModelPath').value.trim() : '';
      const archVal = document.getElementById('manualModelArch') ? document.getElementById('manualModelArch').value : 'siamese';
      const statusBox = document.getElementById('manualModelStatus');
      if (!pathVal) {
        if (statusBox) statusBox.innerHTML = '<span style="color:var(--accent-amber);">Please enter a model file path or select a checkpoint.</span>';
        return;
      }
      if (statusBox) statusBox.innerHTML = '<span>Loading model weights from path...</span>';
      try {
        const resp = await fetch('/studio/api/checkpoints/load-path', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ file_path: pathVal, model_type: archVal })
        });
        const res = await resp.json();
        if (statusBox) {
          if (res.load_result && res.load_result.status === 'success') {
            statusBox.innerHTML = `<span style="color:var(--accent-emerald);">✓ ${escapeHtml(res.load_result.message)}</span>`;
          } else {
            statusBox.innerHTML = `<span style="color:var(--accent-crimson);">${escapeHtml(res.load_result ? res.load_result.message : 'Error')}</span>`;
          }
        }
      } catch (e) {
        if (statusBox) statusBox.innerHTML = `<span style="color:var(--accent-crimson);">Failed loading model: ${escapeHtml(String(e))}</span>`;
      }
    }

    async function runManualInference() {
      const chipSource = document.getElementById('manualChipSourceSelect') ? document.getElementById('manualChipSourceSelect').value : 'sim_deforest';
      const threshold = parseFloat(document.getElementById('manualThreshold') ? document.getElementById('manualThreshold').value : 0.5);
      const mmu = parseFloat(document.getElementById('manualMmu') ? document.getElementById('manualMmu').value : 0.10);
      const modelPath = document.getElementById('manualModelPath') ? document.getElementById('manualModelPath').value.trim() : null;
      const modelArch = document.getElementById('manualModelArch') ? document.getElementById('manualModelArch').value : 'siamese';

      let chipIdx = null;
      let chipId = null;
      if (chipSource.startsWith('active_chip_')) {
        chipIdx = parseInt(chipSource.replace('active_chip_', '')) || 0;
      } else if (chipSource.startsWith('chip_')) {
        chipId = chipSource;
      } else {
        chipId = chipSource;
      }

      const payload = {
        chip_idx: chipIdx,
        chip_id: chipId,
        checkpoint_path: modelPath || null,
        model_type: modelArch,
        threshold: threshold,
        mmu_ha: mmu,
        west: -60.05,
        north: -3.05
      };

      try {
        const resp = await fetch('/studio/api/infer/manual', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const d = await resp.json();
        manualActiveGeoJSON = d.geojson;

        const badge = document.getElementById('manualInferStatusBadge');
        if (badge) {
          badge.className = 'badge-pill badge-compliant';
          badge.innerText = 'INFERENCE COMPLETE';
        }

        if (d.metrics && d.metrics.has_ground_truth) {
          document.getElementById('manualIoU').innerText = (d.metrics.iou * 100).toFixed(1) + '%';
          document.getElementById('manualF1').innerText = (d.metrics.f1 * 100).toFixed(1) + '%';
        } else {
          document.getElementById('manualIoU').innerText = 'N/A (Unlabelled)';
          document.getElementById('manualF1').innerText = 'N/A (Unlabelled)';
        }

        document.getElementById('manualDetectedArea').innerText = (d.total_deforested_ha !== undefined ? d.total_deforested_ha.toFixed(2) : '0.00') + ' ha';
        document.getElementById('manualAlertsCount').innerText = d.total_alerts !== undefined ? d.total_alerts : 0;
        document.getElementById('manualLatency').innerText = (d.elapsed_ms !== undefined ? d.elapsed_ms : 0) + ' ms';

        // Render images
        const t1 = document.getElementById('manualT1Img');
        const t2 = document.getElementById('manualT2Img');
        const mask = document.getElementById('manualMaskImg');
        const heat = document.getElementById('manualHeatmapImg');
        const bin = document.getElementById('manualBinaryImg');

        if (t1 && d.layers.t1_rgb) t1.src = d.layers.t1_rgb;
        if (t2 && d.layers.t2_rgb) t2.src = d.layers.t2_rgb;
        if (mask && (d.layers.mask_overlay_red_trans || d.layers.pred_binary_mask)) {
          mask.src = d.layers.mask_overlay_red_trans || d.layers.pred_binary_mask;
        }
        if (heat && d.layers.probability_heatmap) heat.src = d.layers.probability_heatmap;
        if (bin && (d.layers.pred_binary_mask || d.layers.detected_binary_mask)) {
          bin.src = d.layers.pred_binary_mask || d.layers.detected_binary_mask;
        }

        // Render Polygons Table
        const tbody = document.getElementById('manualAlertsTableBody');
        if (tbody) {
          if (!d.geojson || !d.geojson.features || d.geojson.features.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="color:var(--text-muted); text-align:center;">No deforestation polygons detected above MMU threshold.</td></tr>';
          } else {
            tbody.innerHTML = d.geojson.features.map((f, i) => {
              const p = f.properties || {};
              const lvl = p.alert_level || 'MEDIUM';
              const col = lvl === 'CRITICAL' ? 'var(--accent-crimson)' : (lvl === 'HIGH' ? 'var(--accent-amber)' : 'var(--accent-cyan)');
              const coords = f.geometry && f.geometry.coordinates && f.geometry.coordinates[0] && f.geometry.coordinates[0][0]
                ? `${f.geometry.coordinates[0][0][1].toFixed(4)}, ${f.geometry.coordinates[0][0][0].toFixed(4)}`
                : '--';
              return `
                <tr>
                  <td><code>${escapeHtml(p.alert_id || 'POLY-' + (i+1))}</code></td>
                  <td><span class="badge-pill" style="color:${col}; border-color:${col};">${lvl}</span></td>
                  <td><strong>${(p.area_ha || 0).toFixed(2)} ha</strong></td>
                  <td>${((p.confidence || 0) * 100).toFixed(0)}%</td>
                  <td><code>${coords}</code></td>
                </tr>
              `;
            }).join('');
          }
        }
      } catch (err) {
        alert('Manual inference failed: ' + err);
      }
    }

    function exportManualGeoJSON() {
      if (!manualActiveGeoJSON) {
        alert('Run inference first to generate alert polygons.');
        return;
      }
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(manualActiveGeoJSON, null, 2));
      const a = document.createElement('a');
      a.setAttribute('href', dataStr);
      a.setAttribute('download', 'geofm_manual_inference_alerts.geojson');
      document.body.appendChild(a);
      a.click();
      a.remove();
    }

    // =========================================================================
    // Tab 4: Dataset & Input Loader JavaScript
    async function uploadBYODScenes() {
      const t1Input = document.getElementById('byodT1File');
      const t2Input = document.getElementById('byodT2File');
      const maskInput = document.getElementById('byodMaskFile');
      const statusBox = document.getElementById('byodUploadStatus');

      if (!t1Input.files || t1Input.files.length === 0 || !t2Input.files || t2Input.files.length === 0) {
        if (statusBox) statusBox.innerHTML = '<span style="color:var(--accent-coral);">Please select both T1 (pre) and T2 (post) image files.</span>';
        return;
      }

      const formData = new FormData();
      formData.append('t1', t1Input.files[0]);
      formData.append('t2', t2Input.files[0]);
      if (maskInput.files && maskInput.files.length > 0) {
        formData.append('mask', maskInput.files[0]);
      }

      if (statusBox) statusBox.innerHTML = '<span style="color:var(--accent-cyan);">Ingesting BYOD satellite pair into cache...</span>';

      try {
        const resp = await fetch('/studio/api/dataset/upload-pair', {
          method: 'POST',
          body: formData,
        });
        const data = await resp.json();
        if (data.status === 'success') {
          if (statusBox) statusBox.innerHTML = `<span style="color:var(--accent-emerald);">✓ Ingested: ${data.filename} (${data.dimensions}, ${data.channels} bands). Total chips: ${data.total_chips}</span>`;
          refreshDatasetTable();
        } else {
          if (statusBox) statusBox.innerHTML = `<span style="color:var(--accent-coral);">Error: ${data.detail || data.message || 'Ingestion failed'}</span>`;
        }
      } catch (err) {
        if (statusBox) statusBox.innerHTML = `<span style="color:var(--accent-coral);">Upload error: ${err.message}</span>`;
      }
    }

    // =========================================================================
    async function fetchChipsFromGEE() {
      const projId = document.getElementById('loaderGeeProjectId') ? document.getElementById('loaderGeeProjectId').value.trim() : '';
      const vol = parseInt(document.getElementById('loaderGeeVolume') ? document.getElementById('loaderGeeVolume').value : 200);
      const chipSz = parseInt(document.getElementById('loaderGeeChipSize') ? document.getElementById('loaderGeeChipSize').value : 256);
      const statusBox = document.getElementById('loaderGeeStatus');

      if (statusBox) statusBox.innerHTML = '<span style="color:var(--accent-cyan);">Fetching chips from Earth Engine...</span>';
      try {
        const resp = await fetch('/studio/api/dataset/fetch-gee', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ project_id: projId, num_chips: vol, chip_size: chipSz, target_pos_ratio: 0.70 })
        });
        const res = await resp.json();
        if (statusBox) {
          statusBox.innerHTML = `<span style="color:var(--accent-emerald);">✓ ${escapeHtml(res.message || 'Fetched chips')} (${res.cached_chips_count} in cache)</span>`;
        }
        refreshDatasetTable();
        updateTelemetry();
      } catch (e) {
        if (statusBox) statusBox.innerHTML = `<span style="color:var(--accent-crimson);">Failed: ${escapeHtml(String(e))}</span>`;
      }
    }

    async function setLocalFolder() {
      const folderPath = document.getElementById('loaderLocalFolder') ? document.getElementById('loaderLocalFolder').value.trim() : '';
      const statusBox = document.getElementById('loaderFolderStatus');
      if (!folderPath) {
        if (statusBox) statusBox.innerHTML = '<span style="color:var(--accent-amber);">Enter a directory path</span>';
        return;
      }
      if (statusBox) statusBox.innerHTML = '<span>Scanning folder...</span>';
      try {
        const resp = await fetch('/studio/api/dataset/set-folder', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ folder_path: folderPath })
        });
        const res = await resp.json();
        if (statusBox) {
          if (res.status === 'success') {
            statusBox.innerHTML = `<span style="color:var(--accent-emerald);">✓ ${escapeHtml(res.message)} (${res.chips_count} chips found)</span>`;
          } else {
            statusBox.innerHTML = `<span style="color:var(--accent-crimson);">${escapeHtml(res.message)}</span>`;
          }
        }
        refreshDatasetTable();
        updateTelemetry();
      } catch (e) {
        if (statusBox) statusBox.innerHTML = `<span style="color:var(--accent-crimson);">Failed: ${escapeHtml(String(e))}</span>`;
      }
    }

    async function uploadZipFile(input) {
      if (!input.files || !input.files[0]) return;
      const file = input.files[0];
      const statusBox = document.getElementById('loaderZipStatus');
      if (statusBox) statusBox.innerHTML = '<span style="color:var(--accent-cyan);">Uploading & extracting ZIP archive...</span>';

      const formData = new FormData();
      formData.append('file', file);

      try {
        const resp = await fetch('/studio/api/dataset/upload-zip', {
          method: 'POST',
          body: formData
        });
        const res = await resp.json();
        if (statusBox) {
          if (res.status === 'success') {
            statusBox.innerHTML = `<span style="color:var(--accent-emerald);">✓ Extracted ${res.extracted_count} chips into active cache!</span>`;
          } else {
            statusBox.innerHTML = `<span style="color:var(--accent-crimson);">${escapeHtml(res.message)}</span>`;
          }
        }
        refreshDatasetTable();
        updateTelemetry();
      } catch (e) {
        if (statusBox) statusBox.innerHTML = `<span style="color:var(--accent-crimson);">Upload failed: ${escapeHtml(String(e))}</span>`;
      }
    }

    async function generateSyntheticChipsFromLoader() {
      const count = parseInt(document.getElementById('loaderSynthCount') ? document.getElementById('loaderSynthCount').value : 20);
      try {
        const resp = await fetch('/studio/api/chips/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ num_chips: count, deforest_ratio: 0.70 })
        });
        const data = await resp.json();
        alert(`Successfully generated ${data.created_chips} synthetic chips with 70:30 class balance!`);
        refreshDatasetTable();
        updateTelemetry();
      } catch (e) {
        alert('Synthetic generation failed: ' + e);
      }
    }

    async function refreshDatasetTable() {
      try {
        const resp = await fetch('/studio/api/chips?limit=50');
        const data = await resp.json();
        const tbody = document.getElementById('loaderChipsTableBody');
        const manualChipSel = document.getElementById('manualChipSourceSelect');

        if (manualChipSel) {
          let optionsHtml = `
            <option value="sim_deforest">Dynamic Disturbed Scene (12% Deforestation)</option>
            <option value="sim_intact">Dynamic Intact Canopy (0% Deforestation)</option>
          `;
          if (data.chips && data.chips.length > 0) {
            data.chips.slice(0, 30).forEach((c, idx) => {
              optionsHtml += `<option value="active_chip_${idx}">Active Cache: ${escapeHtml(c.chip_id)} (${c.deforestation_pct}%)</option>`;
            });
          }
          manualChipSel.innerHTML = optionsHtml;
        }

        if (!tbody) return;
        if (!data.chips || data.chips.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5" style="color:var(--text-muted); text-align:center;">No chips cached in active dataset directory. Use any loader card above.</td></tr>';
          return;
        }

        tbody.innerHTML = data.chips.map((c, idx) => `
          <tr>
            <td><code>${escapeHtml(c.chip_id)}</code></td>
            <td>${c.size_kb} KB</td>
            <td style="color:${c.deforestation_pct > 0 ? 'var(--accent-crimson)' : 'var(--accent-emerald)'}; font-weight:600;">${c.deforestation_pct}%</td>
            <td><span class="badge-pill ${c.class_label === 'DEFOREST' ? 'badge-violation' : 'badge-compliant'}">${c.class_label}</span></td>
            <td>
              <button class="btn btn-cyan" style="padding:2px 7px; font-size:0.70rem;" onclick="jumpToChipInViewer(${idx})">View in Viewer</button>
            </td>
          </tr>
        `).join('');
      } catch (err) {
        console.warn('Dataset table refresh failed:', err);
      }
    }

    function jumpToChipInViewer(idx) {
      switchTab('inspector');
      const el = document.getElementById('inspChipIdx');
      if (el) el.value = idx;
      inspectCurrentChip();
    }

    function escapeHtml(str) {
      if (!str) return '';
      return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // Startup
    window.addEventListener('DOMContentLoaded', () => {
      initCharts();
      initMap();
      updateTelemetry();
      loadPresets();
      loadCheckpoints();
      loadChipsZoo();
      refreshDatasetTable();
      inspectCurrentChip();
      runGeospatialInference();
      setInterval(updateTelemetry, 5000);
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
        version="2.0.0",
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
    chip_mgr = StudioChipManager()

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
            tversky_alpha=req.tversky_alpha,
            tversky_beta=req.tversky_beta,
            learning_rate=req.learning_rate,
            epochs=req.epochs,
            batch_size=req.batch_size,
            num_samples=req.num_samples,
            dataset_source=req.dataset_source,
            dataset_scale=req.dataset_scale,
            gee_project_id=req.gee_project_id,
            chip_size=req.chip_size,
            use_data_augmentation=req.use_data_augmentation,
            enforce_class_balance=req.enforce_class_balance,
            use_amp=req.use_amp,
            ram_cache_budget_gb=req.ram_cache_budget_gb,
            async_run=req.async_run,
            resume=req.resume,
            cache_dir=req.cache_dir,
        )
        return summary

    @app.post("/studio/api/train/stop")
    def train_stop_endpoint():
        training_mgr.stop_training()
        return {"status": "stopped", "message": "Stop signal delivered to background trainer"}

    @app.get("/studio/api/train/status")
    def train_status():
        return {
            "is_training": training_mgr.is_training,
            "current_epoch": training_mgr.current_epoch,
            "total_epochs": training_mgr.total_epochs,
            "current_train_loss": training_mgr.current_train_loss,
            "current_val_loss": training_mgr.current_val_loss,
            "current_val_iou": training_mgr.current_val_iou,
            "current_val_f1": training_mgr.current_val_f1,
            "current_lr": training_mgr.current_lr,
            "logs": training_mgr.logs,
            "history": training_mgr.latest_history,
            "latest_run": training_mgr.latest_run_summary,
        }

    # --- Satellite Inference & Benchmarking API Endpoints ---
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
            seed=req.seed,
        )
        return result

    @app.post("/studio/api/infer/manual")
    def infer_manual_endpoint(req: ManualInferRequest):
        return inference_mgr.run_manual_inference(
            chip_idx=req.chip_idx,
            chip_id=req.chip_id,
            checkpoint_path=req.checkpoint_path,
            model_type=req.model_type,
            threshold=req.threshold,
            mmu_ha=req.mmu_ha,
            west=req.west,
            north=req.north,
            cache_dir=chip_mgr.cache_dir,
        )

    @app.post("/studio/api/test/benchmark")
    def test_benchmark_endpoint(req: TestBenchmarkRequest):
        return inference_mgr.test_benchmark(
            checkpoint_name=req.checkpoint_name,
            num_samples=req.num_samples,
            threshold=req.threshold,
        )

    @app.post("/studio/api/test/chip")
    def test_chip_endpoint(req: InspectChipRequest):
        return inference_mgr.inspect_chip(
            chip_idx=req.chip_idx,
            checkpoint_name=req.checkpoint_name,
            checkpoint_path=req.checkpoint_path,
            threshold=req.threshold,
            mmu_ha=req.mmu_ha,
        )

    @app.post("/studio/api/test/thresholds")
    def test_thresholds_endpoint(req: ThresholdSweepRequest):
        return inference_mgr.threshold_sweep(
            checkpoint_name=req.checkpoint_name,
            num_samples=req.num_samples,
        )

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
    def olofsson_endpoint(req: Optional[OlofssonRequest] = None):
        sample_matrix = None
        strata_areas = None

        if req and req.source == "benchmark" and inference_mgr.last_benchmark_scorecard:
            cm = inference_mgr.last_benchmark_scorecard.get("confusion_matrix", {})
            tp = cm.get("tp", 85)
            fp = cm.get("fp", 15)
            fn = cm.get("fn", 8)
            tn = cm.get("tn", 392)
            sample_matrix = np.array([[max(1, tp), max(1, fp)], [max(1, fn), max(1, tn)]])
            if req.stratum_deforest_ha and req.stratum_forest_ha:
                strata_areas = np.array([float(req.stratum_deforest_ha), float(req.stratum_forest_ha)])

        elif req and req.source == "custom" and req.tp is not None:
            sample_matrix = np.array([[max(1, req.tp), max(1, req.fp or 0)], [max(1, req.fn or 0), max(1, req.tn or 1)]])
            if req.stratum_deforest_ha and req.stratum_forest_ha:
                strata_areas = np.array([float(req.stratum_deforest_ha), float(req.stratum_forest_ha)])

        return compliance_mgr.compute_olofsson_accuracy(sample_matrix=sample_matrix, strata_areas_ha=strata_areas)

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

    @app.post("/studio/api/catalog/validate")
    def catalog_validate_endpoint(req: CatalogValidateRequest):
        return stac_mgr.validate_url(req.url)

    @app.post("/studio/api/catalog/heal")
    def catalog_heal_endpoint(req: CatalogHealRequest):
        return stac_mgr.heal_url(req.url)

    # --- Checkpoints & Zoo API Endpoints ---
    @app.get("/studio/api/checkpoints")
    def checkpoints_endpoint():
        return checkpoint_mgr.list_checkpoints()

    @app.get("/studio/api/download/{filename}")
    def download_checkpoint_endpoint(filename: str):
        safe_name = os.path.basename(filename)
        path = os.path.join(checkpoint_mgr.checkpoint_dir, safe_name)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Checkpoint file not found")
        return FileResponse(path, filename=safe_name, media_type="application/octet-stream")

    @app.post("/studio/api/checkpoints/load")
    def load_checkpoint_endpoint(req: LoadCheckpointRequest):
        safe_name = os.path.basename(req.filename)
        path = os.path.join(checkpoint_mgr.checkpoint_dir, safe_name)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Checkpoint {safe_name} not found")
        try:
            chk = torch.load(path, map_location="cpu", weights_only=False)
            st = chk.get("model_state_dict", chk)
            model = inference_mgr._get_model("siamese")
            model.load_state_dict(st, strict=False)
            return {"status": "success", "message": f"Successfully hot-loaded weights from {safe_name}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed loading checkpoint: {str(e)}")

    @app.post("/studio/api/checkpoints/load-path")
    def load_path_checkpoint_endpoint(req: LoadPathCheckpointRequest):
        load_res = inference_mgr.load_model_weights(req.file_path, model_type=req.model_type)
        chk_info = checkpoint_mgr.load_checkpoint_from_path(req.file_path)
        return {
            "load_result": load_res,
            "checkpoint_info": chk_info,
        }

    # --- Dataset & Input Loader API Endpoints ---
    @app.post("/studio/api/dataset/set-folder")
    def dataset_set_folder_endpoint(req: SetFolderRequest):
        return chip_mgr.set_cache_dir(req.folder_path)

    @app.post("/studio/api/dataset/upload-pair")
    async def dataset_upload_pair_endpoint(
        t1: UploadFile = File(...),
        t2: UploadFile = File(...),
        mask: Optional[UploadFile] = File(None),
        chip_id: Optional[str] = Form(None),
    ):
        t1_bytes = await t1.read()
        t2_bytes = await t2.read()
        mask_bytes = await mask.read() if mask is not None else None
        return chip_mgr.ingest_paired_chip(
            t1_bytes=t1_bytes,
            t2_bytes=t2_bytes,
            mask_bytes=mask_bytes,
            chip_name=chip_id or os.path.splitext(t1.filename or "byod_chip")[0],
        )

    @app.post("/studio/api/dataset/upload-zip")
    async def dataset_upload_zip_endpoint(file: UploadFile = File(...)):
        content = await file.read()
        return chip_mgr.extract_chips_zip(content)

    @app.post("/studio/api/dataset/fetch-gee")
    def dataset_fetch_gee_endpoint(req: FetchGEERequest):
        return chip_mgr.fetch_gee_chips(
            project_id=req.project_id,
            num_chips=req.num_chips,
            chip_size=req.chip_size,
            target_pos_ratio=req.target_pos_ratio,
        )

    @app.get("/studio/api/chips")
    def list_chips_endpoint(limit: int = 100):
        return chip_mgr.list_chips(limit=limit)

    @app.post("/studio/api/chips/generate")
    def generate_chips_endpoint(req: GenerateChipsRequest):
        return chip_mgr.generate_synthetic_chips(num_chips=req.num_chips, deforest_ratio=req.deforest_ratio)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon_endpoint():
        return Response(content=b"", media_type="image/x-icon", status_code=204)

    @app.get("/studio/api/telemetry")
    def telemetry_endpoint():
        return get_hardware_telemetry(
            checkpoint_dir=checkpoint_mgr.checkpoint_dir,
            cache_dir=chip_mgr.cache_dir,
        )

    return app


studio_fastapi_app = create_studio_app()
