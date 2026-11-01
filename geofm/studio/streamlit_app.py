"""Streamlit Web Training Studio & EUDR Compliance Dashboard for GeoFM.

Run via: `streamlit run studio.py` or `streamlit run geofm/studio/streamlit_app.py`
"""

import json
import os
from typing import Any, Dict, List, Optional
import numpy as np

# Safe import of streamlit with helpful instructions if missing
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    st = None
    STREAMLIT_AVAILABLE = False

from geofm.studio.core import (
    CheckpointManager,
    StudioComplianceManager,
    StudioInferenceManager,
    StudioSTACManager,
    StudioTrainingManager,
)


def run_streamlit_app():
    """Main Streamlit application interface."""
    if not STREAMLIT_AVAILABLE:
        print("Error: Streamlit is not installed in the current environment.")
        print("To run the Streamlit studio, install it or run `python studio.py` to use the standalone web studio.")
        return

    st.set_page_config(
        page_title="GeoFM Studio | Deforestation & EUDR Compliance",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1b4332;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #40916c;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            border-left: 4px solid #2d6a4f;
        }
        .compliant-badge {
            background-color: #d8f3dc;
            color: #1b4332;
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: bold;
        }
        .violation-badge {
            background-color: #ffe5d9;
            color: #9d0208;
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">🛰️ GeoFM Web Training Studio & EUDR Suite</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Interactive Foundation Model Fine-Tuning, Multi-Modal Bitemporal Inference, and Legal EUDR Audit Platform</div>',
        unsafe_allow_html=True,
    )

    # Initialize session managers
    if "training_mgr" not in st.session_state:
        st.session_state.training_mgr = StudioTrainingManager()
    if "inference_mgr" not in st.session_state:
        st.session_state.inference_mgr = StudioInferenceManager()
    if "compliance_mgr" not in st.session_state:
        st.session_state.compliance_mgr = StudioComplianceManager()
    if "stac_mgr" not in st.session_state:
        st.session_state.stac_mgr = StudioSTACManager()
    if "checkpoint_mgr" not in st.session_state:
        st.session_state.checkpoint_mgr = CheckpointManager()

    tabs = st.tabs([
        "🏋️ 1. AI Model Training Studio",
        "🛰️ 2. Satellite Inference & Interactive Map",
        "⚖️ 3. EUDR Due Diligence & Regulatory Compliance",
        "🗂️ 4. STAC Satellite Catalog Explorer",
    ])

    # =========================================================================
    # TAB 1: AI Model Training Studio
    # =========================================================================
    with tabs[0]:
        st.header("Geospatial Neural Model Training & Parameter-Efficient Fine-Tuning (PEFT)")
        st.markdown(
            "Configure model backbones, inject LoRA rank decomposition adapters into attention layers, "
            "select class-imbalanced compound losses, and trigger fine-tuning with live metric visualization."
        )

        col_t1, col_t2 = st.columns([1, 2])

        with col_t1:
            st.subheader("Training Parameters")
            arch = st.selectbox(
                "Model Architecture",
                options=["siamese", "unet", "multimodal"],
                format_func=lambda x: {
                    "siamese": "Siamese Bitemporal Change Detector",
                    "unet": "U-Net ViT Hybrid (ResNet + Transformer Bottleneck)",
                    "multimodal": "Multimodal Cross-Attention Fusion (Optical + SAR)",
                }[x],
            )

            use_lora = st.checkbox("Inject LoRA PEFT Adapters", value=True)
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                lora_rank = st.number_input("LoRA Rank ($r$)", min_value=2, max_value=64, value=8, step=2, disabled=not use_lora)
            with col_l2:
                lora_alpha = st.number_input("LoRA Alpha ($\\alpha$)", min_value=4.0, max_value=128.0, value=16.0, step=4.0, disabled=not use_lora)

            loss_choice = st.selectbox(
                "Loss Function",
                options=["compound", "focal_tversky", "focal", "tversky"],
                format_func=lambda x: {
                    "compound": "Compound Deforestation Loss (Focal + Tversky)",
                    "focal_tversky": "Focal Tversky Loss (Asymmetric Boundary Penalty)",
                    "focal": "Binary Focal Loss (Hard Negative Mining)",
                    "tversky": "Tversky Loss (Custom FP/FN Tradeoff)",
                }[x],
            )

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                lr = st.select_slider("Learning Rate", options=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3], value=1e-3)
                epochs = st.number_input("Epochs", min_value=1, max_value=50, value=3)
            with col_p2:
                batch_size = st.selectbox("Batch Size", options=[2, 4, 8, 16], index=1)
                num_samples = st.selectbox("Synthetic Samples", options=[4, 8, 16, 32], index=2)

            start_btn = st.button("🚀 Start Training Run", type="primary", use_container_width=True)

        with col_t2:
            st.subheader("Real-Time Experiment Tracking & Metrics")

            if start_btn:
                progress_bar = st.progress(0.0)
                status_placeholder = st.empty()
                chart_placeholder = st.empty()
                metrics_placeholder = st.empty()

                def _progress_cb(ep: int, total: int, rec: Dict[str, Any]):
                    pct = ep / float(total)
                    progress_bar.progress(pct)
                    status_placeholder.info(
                        f"Epoch {ep}/{total} - Train Loss: {rec['train_loss']:.4f} | "
                        f"Val Loss: {rec['val_loss']:.4f} | Val IoU: {rec['val_iou']*100:.2f}% | Val F1: {rec['val_f1']*100:.2f}%"
                    )

                with st.spinner("Training geospatial neural network on spatial tiles..."):
                    run_summary = st.session_state.training_mgr.train_model(
                        architecture=arch,
                        use_lora=use_lora,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                        loss_type=loss_choice,
                        learning_rate=lr,
                        epochs=int(epochs),
                        batch_size=batch_size,
                        num_samples=num_samples,
                        progress_callback=_progress_cb,
                    )

                st.success(f"🎉 Training run completed! Best Validation IoU: {run_summary['best_val_iou']*100:.2f}%")

                hist = run_summary["history"]
                epochs_x = [h["epoch"] for h in hist]
                train_losses = [h["train_loss"] for h in hist]
                val_losses = [h["val_loss"] for h in hist]
                val_ious = [h["val_iou"] * 100 for h in hist]

                st.line_chart(
                    data={
                        "Train Loss": train_losses,
                        "Val Loss": val_losses,
                    }
                )
                st.line_chart(
                    data={
                        "Validation IoU (%)": val_ious,
                    }
                )

                st.write("### Epoch Performance Summary Table")
                st.dataframe(hist, use_container_width=True)

            elif st.session_state.training_mgr.latest_history:
                st.info("Displaying most recent training run history:")
                st.dataframe(st.session_state.training_mgr.latest_history, use_container_width=True)

            # Saved Checkpoint Inspector
            st.write("---")
            st.subheader("💾 Checkpoint Management & Weights")
            checkpoints = st.session_state.checkpoint_mgr.list_checkpoints()
            if checkpoints:
                st.dataframe(checkpoints, use_container_width=True)
            else:
                st.caption("No saved checkpoints found in `./checkpoints` directory yet.")

    # =========================================================================
    # TAB 2: Satellite Inference & Interactive Map
    # =========================================================================
    with tabs[1]:
        st.header("Satellite Change Detection Inference & Deforestation Polygonization")
        st.markdown(
            "Inspect raw multi-spectral Sentinel-2 bands, Normalized Difference Vegetation Index (NDVI), "
            "Sentinel-1 SAR C-band dual polarization backscatter, and neural deforestation predictions."
        )

        col_s1, col_s2 = st.columns([1, 3])

        with col_s1:
            st.subheader("Scene & Model Settings")
            def_fraction = st.slider("Deforestation Fraction", min_value=0.01, max_value=0.25, value=0.08, step=0.01)
            cloud_fraction = st.slider("Cloud Occlusion Fraction", min_value=0.0, max_value=0.35, value=0.15, step=0.05)

            infer_model = st.selectbox(
                "Inference Engine",
                options=["siamese", "multimodal", "unet", "ndvi_delta"],
                format_func=lambda x: {
                    "siamese": "Siamese Bitemporal Network (T1 vs T2)",
                    "multimodal": "Multimodal Cross-Attention (Optical + SAR)",
                    "unet": "U-Net ViT Semantic Disturbance",
                    "ndvi_delta": "Baseline: Bitemporal ΔNDVI Thresholding",
                }[x],
            )

            threshold = st.slider("Probability Threshold", min_value=0.1, max_value=0.95, value=0.5, step=0.05)
            mmu_ha = st.slider("Minimum Mapping Unit (MMU ha)", min_value=0.05, max_value=1.0, value=0.10, step=0.05)

            run_infer_btn = st.button("🛰️ Run Deforestation Inference", type="primary", use_container_width=True)

        with col_s2:
            if run_infer_btn or "last_infer_result" not in st.session_state:
                with st.spinner("Executing neural change detection and vector polygonization..."):
                    infer_result = st.session_state.inference_mgr.run_inference(
                        deforestation_fraction=def_fraction,
                        cloud_fraction=cloud_fraction,
                        model_type=infer_model,
                        threshold=threshold,
                        mmu_ha=mmu_ha,
                    )
                    st.session_state.last_infer_result = infer_result
            else:
                infer_result = st.session_state.last_infer_result

            # Key metrics header
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Alert Clusters", f"{infer_result['total_alerts']:,}")
            m2.metric("Total Deforested Area", f"{infer_result['total_deforested_ha']:.2f} ha")
            m3.metric("Inference Latency", f"{infer_result['inference_time_ms']:.1f} ms")
            m4.metric("Cloud Occlusion", f"{infer_result['cloud_percentage']:.1f}%")

            # Satellite Layer Visualizer
            st.subheader("Satellite Spectral Bands & Feature Maps")
            img_col1, img_col2, img_col3, img_col4 = st.columns(4)

            layers = infer_result["layers"]
            with img_col1:
                st.image(layers["t1_rgb"], caption="Baseline T1 (Sentinel-2 RGB)", use_container_width=True)
                st.image(layers["ndvi_t1"], caption="Baseline T1 NDVI", use_container_width=True)

            with img_col2:
                st.image(layers["t2_rgb"], caption="Disturbed T2 (Sentinel-2 RGB)", use_container_width=True)
                st.image(layers["ndvi_t2"], caption="Disturbed T2 NDVI", use_container_width=True)

            with img_col3:
                st.image(layers["t2_cir"], caption="T2 Color Infrared (CIR B8-B4-B3)", use_container_width=True)
                st.image(layers["sar_composite"], caption="Sentinel-1 SAR False Color", use_container_width=True)

            with img_col4:
                st.image(layers["probability_heatmap"], caption="Deforestation Heatmap", use_container_width=True)
                st.write("**Severity Summary:**")
                st.write(infer_result["severity_counts"])

            # Alert Polygons & GeoJSON Inspection
            st.subheader("Vector Alert Polygons (GeoJSON)")
            features = infer_result["geojson"].get("features", [])
            if features:
                summary_table = [
                    {
                        "Cluster ID": f.get("properties", {}).get("cluster_id"),
                        "Area (ha)": f.get("properties", {}).get("area_ha"),
                        "Alert Severity": f.get("properties", {}).get("alert_level"),
                        "Centroid (Lon, Lat)": f.get("properties", {}).get("centroid"),
                    }
                    for f in features
                ]
                st.dataframe(summary_table, use_container_width=True)

                with st.expander("Inspect Raw GeoJSON FeatureCollection"):
                    st.json(infer_result["geojson"])
            else:
                st.info("No deforestation clusters detected above the selected MMU threshold.")

    # =========================================================================
    # TAB 3: EUDR Due Diligence & Regulatory Compliance
    # =========================================================================
    with tabs[2]:
        st.header("European Union Deforestation Regulation (EUDR) Legal Due Diligence")
        st.markdown(
            "Verify agricultural farm plots against Regulation (EU) 2023/1115 mandates: "
            "**Cutoff Date (Dec 31, 2020)**, **Plot Size Threshold (<=4 ha Point vs >4 ha Polygon)**, "
            "**6-Decimal Coordinate Precision**, and **Annex I Commodity Scope**."
        )

        col_c1, col_c2 = st.columns([1, 2])

        scenarios = st.session_state.compliance_mgr.get_preset_scenarios()
        selected_scenario_name = col_c1.selectbox("Select Test Due Diligence Scenario", options=list(scenarios.keys()))
        selected_scenario = scenarios[selected_scenario_name]

        with col_c1:
            st.subheader("Due Diligence Statement Details")
            stmt_id = st.text_input("Statement ID", value=selected_scenario["statement_id"])
            operator = st.text_input("Operator Name", value=selected_scenario["operator_name"])
            commodity = st.selectbox(
                "Commodity (EUDR Annex I)",
                options=sorted(list(EUDR_REGULATED_COMMODITIES)) + ["maize (non-EUDR test)"],
                index=sorted(list(EUDR_REGULATED_COMMODITIES)).index(selected_scenario["commodity"]) if selected_scenario["commodity"] in EUDR_REGULATED_COMMODITIES else 0,
            )
            country = st.text_input("Country of Production", value=selected_scenario["country"])

            audit_btn = st.button("⚖️ Run EUDR Compliance Audit", type="primary", use_container_width=True)

        with col_c2:
            st.subheader("Plot Geolocation & Alert Intersection")
            plots_json_str = st.text_area(
                "Plot Geometries (JSON)",
                value=json.dumps(selected_scenario["plots"], indent=2),
                height=180,
            )

            alerts_json_str = st.text_area(
                "Active Deforestation Alerts (JSON)",
                value=json.dumps(selected_scenario["alerts"], indent=2),
                height=140,
            )

            if audit_btn:
                try:
                    plots_obj = json.loads(plots_json_str)
                    alerts_obj = json.loads(alerts_json_str)
                except Exception as e:
                    st.error(f"Invalid JSON format: {e}")
                    plots_obj = None

                if plots_obj is not None:
                    audit_res = st.session_state.compliance_mgr.audit_statement(
                        statement_id=stmt_id,
                        operator_name=operator,
                        commodity=commodity,
                        country=country,
                        plots=plots_obj,
                        alerts=alerts_obj,
                    )

                    status = audit_res["overall_status"]
                    if status == "COMPLIANT":
                        st.markdown(
                            f'<h3>Verdict: <span class="compliant-badge">✅ {status}</span></h3>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<h3>Verdict: <span class="violation-badge">❌ {status}</span></h3>',
                            unsafe_allow_html=True,
                        )

                    st.metric("Risk Score", f"{audit_res['risk_score']:.2f}")

                    st.write("#### Detailed Plot Audit Results")
                    for p in audit_res["plots"]:
                        with st.expander(f"Plot {p['plot_id']} - Status: {p['status']}"):
                            st.write(f"- **Calculated Area:** {p['area_ha']} ha")
                            if p["violations"]:
                                st.error("Violations Detected:")
                                for v in p["violations"]:
                                    st.write(f"  • {v}")
                            else:
                                st.success("No EUDR violations detected on this plot.")

                            if p["alert_overlaps"]:
                                st.warning(f"Overlapping Alerts: {len(p['alert_overlaps'])}")
                                st.json(p["alert_overlaps"])

        st.write("---")
        st.subheader("📐 Olofsson et al. (2014) Area-Adjusted Accuracy Assessment")
        st.markdown(
            "International standard for unbiased land change estimation and 95% confidence intervals "
            "from stratified random sampling error matrices."
        )

        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.write("**Stratified Sample Matrix Counts:**")
            st.write("- Mapped Deforestation & Reference Deforestation: **85**")
            st.write("- Mapped Deforestation & Reference Forest: **15**")
            st.write("- Mapped Forest & Reference Deforestation: **8**")
            st.write("- Mapped Forest & Reference Forest: **392**")
            calc_olof_btn = st.button("Calculate Area-Adjusted Statistics")

        with col_o2:
            olof_report = st.session_state.compliance_mgr.compute_olofsson_accuracy()
            st.metric(
                "Area-Adjusted Overall Accuracy",
                f"{olof_report['overall_accuracy_pct']}% ± {olof_report['oa_margin_of_error_pct']}%",
            )
            def_area = olof_report["estimated_areas_ha"].get("Deforestation", 0.0)
            def_moe = olof_report["area_margins_of_error_ha"].get("Deforestation", 0.0)
            st.metric(
                "Unbiased Deforestation Area",
                f"{def_area:,.1f} ha ± {def_moe:,.1f} ha",
            )
            with st.expander("Full Statistical Confidence Intervals"):
                st.text(olof_report["summary_text"])

    # =========================================================================
    # TAB 4: STAC Satellite Catalog Explorer
    # =========================================================================
    with tabs[3]:
        st.header("SpatioTemporal Asset Catalog (STAC) Satellite Explorer")
        st.markdown(
            "Search and inspect Cloud-Optimized GeoTIFF (COG) assets for Sentinel-2 optical "
            "and Sentinel-1 SAR observations across tropical forest monitoring sites."
        )

        col_stac1, col_stac2 = st.columns([1, 2])

        with col_stac1:
            st.subheader("Catalog Query Filters")
            region_preset = st.selectbox(
                "Geographic Region Preset",
                options=["Amazon Basin (Manaus)", "Congo Basin (Salonga)", "Borneo Peatland (Kalimantan)", "Custom Bounding Box"],
            )

            if region_preset == "Amazon Basin (Manaus)":
                bbox_val = [-60.5, -3.5, -59.5, -2.5]
            elif region_preset == "Congo Basin (Salonga)":
                bbox_val = [20.0, -2.5, 21.0, -1.5]
            elif region_preset == "Borneo Peatland (Kalimantan)":
                bbox_val = [112.5, -2.5, 113.5, -1.5]
            else:
                bbox_val = [-60.5, -3.5, -59.5, -2.5]

            bbox_str = st.text_input("Bounding Box [min_lon, min_lat, max_lon, max_lat]", value=str(bbox_val))
            platform_filter = st.selectbox("Platform Filter", options=["All", "sentinel-2", "sentinel-1"])
            max_cloud = st.slider("Max Cloud Cover (%)", min_value=0, max_value=100, value=25)

            search_stac_btn = st.button("🔍 Search STAC Catalog", type="primary", use_container_width=True)

        with col_stac2:
            st.subheader("Query Results & COG Assets")

            try:
                parsed_bbox = json.loads(bbox_str)
            except Exception:
                parsed_bbox = bbox_val

            plat = None if platform_filter == "All" else platform_filter
            items = st.session_state.stac_mgr.query_catalog(
                bbox=parsed_bbox,
                platform=plat,
                max_cloud_cover=float(max_cloud),
            )

            st.write(f"Found **{len(items)}** matching STAC items:")
            for it in items:
                with st.expander(f"🛰️ {it['id']} ({it['platform']}) - {it['datetime']}"):
                    st.write(f"**Bounding Box:** `{it['bbox']}`")
                    if it["cloud_cover"] is not None:
                        st.write(f"**Cloud Cover:** {it['cloud_cover']:.1f}%")
                    st.write("**COG Assets:**")
                    for k, v in it["assets"].items():
                        st.markdown(f"- **{k}**: [{v['title']}]({v['href']}) (`{v['media_type']}`)")


if __name__ == "__main__":
    run_streamlit_app()
