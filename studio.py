#!/usr/bin/env python3
"""GeoFM Interactive Web Training Studio & EUDR Compliance Suite Launcher.

Universal launcher supporting:
1. Streamlit Mode:
   $ streamlit run studio.py

2. Standalone High-Performance Web Studio (FastAPI + Leaflet + Chart.js fallback):
   $ python studio.py [--port 8501] [--host 0.0.0.0]
"""

import argparse
import os
import sys

# Ensure UTF-8 output encoding on Windows terminals to prevent charmap UnicodeEncodeErrors
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure geofm is importable regardless of invocation directory or installation status
_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import uvicorn

# Detect if running under Streamlit
def _is_running_in_streamlit() -> bool:
    try:
        import streamlit as st
        # If running via `streamlit run`, streamlit will have an active script context or runtime
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def main():
    if _is_running_in_streamlit():
        # Streamlit runner path
        from geofm.studio.streamlit_app import run_streamlit_app
        run_streamlit_app()
        return

    parser = argparse.ArgumentParser(
        description="Launch GeoFM Interactive Web Training Studio & EUDR Compliance Platform"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve on (default: 8501)")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "fastapi", "streamlit"],
        help="Web server backend: auto, fastapi, or streamlit (default: auto)",
    )
    args = parser.parse_args()

    # If user explicitly requested Streamlit backend
    if args.backend == "streamlit":
        try:
            import streamlit
            print(f"[GeoFM Studio] Launching Streamlit UI on http://{args.host}:{args.port}...")
            os.system(f"streamlit run {os.path.abspath(__file__)} --server.port {args.port} --server.address {args.host}")
            return
        except ImportError:
            print("[GeoFM Studio] Streamlit is not installed in current environment.")
            print("[GeoFM Studio] Falling back seamlessly to standalone FastAPI interactive studio...")

    # Default / Fallback: Fast, self-contained FastAPI Web Studio
    import torch
    from geofm.studio.core import get_hardware_telemetry
    telemetry = get_hardware_telemetry()
    ram_str = f"{telemetry['ram_total_mb'] / 1024:.1f} GB" if telemetry['ram_total_mb'] > 0 else "System Allocated"

    print("=" * 72)
    print("  🛰️  GeoFM INTERACTIVE WEB TRAINING STUDIO & EUDR COMPLIANCE SUITE  ")
    print("=" * 72)
    print(f"  • Interface:       http://{args.host}:{args.port}")
    print(f"  • Host Compute:    {telemetry['device_name']} ({telemetry['cpu_count']} CPU cores, {ram_str} RAM)")
    print(f"  • Backend Engine:  FastAPI + Uvicorn + PyTorch {torch.__version__}")
    print(f"  • Visual Mapping:  Leaflet.js + Sentinel-2/1 Spectral Band Layers")
    print(f"  • Analytics:       Chart.js Live Loss & IoU History")
    print(f"  • EUDR Baseline:   December 31, 2020 Cutoff (Regulation EU 2023/1115)")
    print(f"  • Olofsson Engine: Stratified Area-Adjusted Accuracy & 95% CI")
    print("-" * 72)
    print(f"  Open your browser at: http://{args.host}:{args.port}")
    print("  Press Ctrl+C to stop server.")
    print("=" * 72)

    from geofm.studio.web_app import studio_fastapi_app
    uvicorn.run(studio_fastapi_app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
