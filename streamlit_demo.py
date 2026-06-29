"""
streamlit_demo.py
==================
Quick Streamlit demo for RadiologyAI.
No backend server needed — model runs directly in the app process.

Run:
    streamlit run streamlit_demo.py
"""

import io
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title   = "RadiologyAI",
    page_icon    = "🩻",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #07080d;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #0d0f17;
    border-right: 1px solid rgba(255,255,255,.07);
}
.metric-card {
    background: #131622;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-val { font-size: 28px; font-weight: 700; color: #63b3ed; }
.metric-lbl { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: .8px; }
.report-box {
    background: #0d0f17;
    border: 1px solid rgba(99,179,237,.25);
    border-radius: 12px;
    padding: 20px 24px;
    font-size: 15px;
    line-height: 1.8;
    color: #e2e8f0;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 600;
    margin: 2px;
}
.badge-warn { background: rgba(246,173,85,.15); color: #f6ad55; border: 1px solid rgba(246,173,85,.3); }
.badge-ok   { background: rgba(72,187,120,.15);  color: #48bb78; border: 1px solid rgba(72,187,120,.3); }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    beam_size   = st.slider("Beam Size",     min_value=1, max_value=10, value=5)
    max_len     = st.slider("Max Report Length", 30, 200, 120)
    show_gradcam = st.toggle("Show Grad-CAM", value=True)
    st.divider()
    st.markdown("### Model Info")
    st.markdown("""
    - **Encoder**: EfficientNet-B3
    - **Decoder**: Transformer (6L, 8H)
    - **d_model**: 512
    - **Trained on**: OpenI / MIMIC-CXR
    """)
    st.divider()
    st.markdown("### Dataset Links")
    st.markdown("- [OpenI (Kaggle)](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)")
    st.markdown("- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/)")

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# 🩻")
with col_title:
    st.markdown("# RadiologyAI")
    st.markdown("*Automatic Radiology Report Generation — Deep Learning · Vision-Language Model*")
st.divider()


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model …")
def load_engine():
    try:
        from ai_model.inference import RadiologyInferenceEngine
        return RadiologyInferenceEngine.from_config("config/config.yaml")
    except Exception as exc:
        return None, str(exc)


engine = load_engine()


# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown("### 📤 Upload Chest X-Ray")
uploaded = st.file_uploader(
    "Supported: JPG, PNG, DICOM",
    type=["jpg", "jpeg", "png", "dcm"],
    label_visibility="collapsed",
)

ABNORMAL_KW = [
    "effusion","pneumothorax","consolidation","atelectasis","edema",
    "cardiomegaly","opacity","infiltrate","mass","nodule","fracture",
    "pneumonia","congestion","enlarged","abnormal","bilateral",
]

# ── Main area ─────────────────────────────────────────────────────────────────
if uploaded is not None:
    img_bytes = uploaded.read()

    # Load PIL image
    from backend.utils.image_utils import convert_to_pil
    ext = Path(uploaded.name).suffix.lower()
    try:
        pil_img = convert_to_pil(img_bytes, ext)
    except ValueError as e:
        st.error(f"Image error: {e}")
        st.stop()

    col_img, col_report = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("#### 🫁 Image")
        tab_orig, tab_gc = st.tabs(["Original", "Grad-CAM"])
        with tab_orig:
            st.image(pil_img, use_container_width=True, caption=uploaded.name)

    with col_report:
        st.markdown("#### 📋 Generated Report")
        with st.spinner("Running AI inference …"):
            if engine and not isinstance(engine, tuple):
                try:
                    result = engine.predict(pil_img, beam_size=beam_size, generate_gradcam=show_gradcam)
                    report     = result["report"]
                    confidence = result["confidence"]
                    latency    = result["latency_ms"]
                    gradcam_pil = result.get("gradcam_pil")

                    # Grad-CAM tab
                    if gradcam_pil:
                        with tab_gc:
                            st.image(gradcam_pil, use_container_width=True, caption="Grad-CAM Overlay")

                except Exception as exc:
                    st.error(f"Inference failed: {exc}")
                    st.stop()
            else:
                # Demo fallback
                st.warning("⚡ Demo mode — no model checkpoint found. Showing example output.")
                report     = ("The lungs are clear bilaterally. No focal consolidation, "
                              "pleural effusion, or pneumothorax is identified. "
                              "The cardiomediastinal silhouette is within normal limits. "
                              "No acute cardiopulmonary abnormality.")
                confidence = 0.87
                latency    = 342.0
                gradcam_pil = None

        # ── Report display ─────────────────────────────────────────────────
        st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
        st.markdown("")

        # ── Confidence bar ─────────────────────────────────────────────────
        pct = int(confidence * 100)
        color = "#48bb78" if pct >= 70 else "#f6ad55" if pct >= 40 else "#fc8181"
        st.markdown(f"**Model Confidence: {pct}%**")
        st.markdown(
            f'<div style="height:8px;background:#131622;border-radius:4px;overflow:hidden;">'
            f'<div style="width:{pct}%;height:100%;background:{color};border-radius:4px;'
            f'transition:width 1s;"></div></div>',
            unsafe_allow_html=True
        )

        # ── Key findings ───────────────────────────────────────────────────
        st.markdown("**Key Findings**")
        found = [kw for kw in ABNORMAL_KW if kw.lower() in report.lower()]
        if found:
            badges = "".join(f'<span class="badge badge-warn">⚠ {kw}</span>' for kw in found)
        else:
            badges = '<span class="badge badge-ok">✅ No significant abnormalities</span>'
        st.markdown(badges, unsafe_allow_html=True)

        # ── Metrics row ────────────────────────────────────────────────────
        st.markdown("")
        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(
            f'<div class="metric-card"><div class="metric-val">{pct}%</div>'
            f'<div class="metric-lbl">Confidence</div></div>',
            unsafe_allow_html=True
        )
        mc2.markdown(
            f'<div class="metric-card"><div class="metric-val">{latency:.0f}ms</div>'
            f'<div class="metric-lbl">Latency</div></div>',
            unsafe_allow_html=True
        )
        mc3.markdown(
            f'<div class="metric-card"><div class="metric-val">{len(report.split())}</div>'
            f'<div class="metric-lbl">Words</div></div>',
            unsafe_allow_html=True
        )

        # ── Download ───────────────────────────────────────────────────────
        st.download_button(
            "⬇ Download Report (.txt)",
            data     = report,
            file_name= f"radiology_report_{Path(uploaded.name).stem}.txt",
            mime     = "text/plain",
        )

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
        <div style="font-size:80px;margin-bottom:16px;">🩻</div>
        <h3 style="color:#94a3b8;">Upload a chest X-ray to get started</h3>
        <p style="color:#64748b;max-width:500px;margin:0 auto;">
        The AI model will automatically analyse the image and produce
        a professional radiology report with key findings highlighted.
        </p>
    </div>
    """, unsafe_allow_html=True)
