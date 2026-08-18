import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.predictor import PlantDiseasePredictor
from src.disease_info import get_disease_info

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background:
      radial-gradient(circle at 10% 0%, rgba(53, 196, 125, .13), transparent 30%),
      radial-gradient(circle at 90% 15%, rgba(133, 211, 111, .10), transparent 28%),
      #07130e;
    color: #ecfff4;
}
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }

.hero {
    padding: 2.4rem 2.5rem;
    border: 1px solid rgba(145,255,190,.18);
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(16,47,33,.95), rgba(8,26,19,.86));
    box-shadow: 0 25px 80px rgba(0,0,0,.28);
}
.eyebrow {
    color: #79e7a7; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; font-size: .78rem;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.2rem, 5vw, 4.3rem);
    line-height: 1.02; margin: .35rem 0 .8rem;
}
.hero p { color: #b9d5c5; font-size: 1.05rem; max-width: 760px; }
.status {
    display:inline-block; padding:.45rem .75rem; border-radius:999px;
    background:rgba(84,232,139,.11); border:1px solid rgba(84,232,139,.25);
    color:#86f0ad; font-size:.82rem; font-weight:600;
}
.card {
    background: rgba(13, 34, 25, .78);
    border: 1px solid rgba(145,255,190,.12);
    border-radius: 22px;
    padding: 1.25rem;
    box-shadow: 0 16px 45px rgba(0,0,0,.18);
}
.metric {
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 16px; padding: 1rem;
}
.metric .label { color:#8eaa9a; font-size:.8rem; }
.metric .value { font-family:'Space Grotesk'; font-size:1.55rem; font-weight:700; margin-top:.25rem; }
.diagnosis {
    border-radius: 22px; padding: 1.35rem;
    background: linear-gradient(135deg, rgba(40,105,69,.55), rgba(13,37,27,.85));
    border: 1px solid rgba(121,231,167,.22);
}
.diagnosis h2 { font-family:'Space Grotesk'; margin:.2rem 0; }
.badge {
    display:inline-block; padding:.35rem .65rem; border-radius:999px;
    font-size:.76rem; font-weight:700; margin-bottom:.5rem;
}
.badge-ok { background:rgba(85,230,137,.13); color:#8af2ad; }
.badge-warn { background:rgba(255,188,92,.13); color:#ffd18b; }
.small { color:#9eb8aa; font-size:.88rem; }
.section-title { font-family:'Space Grotesk'; font-size:1.35rem; margin:1.8rem 0 .8rem; }
div[data-testid="stFileUploader"] {
    border: 1px dashed rgba(121,231,167,.35);
    border-radius: 20px; padding: .6rem;
    background: rgba(17,47,34,.45);
}
.stButton > button {
    border-radius: 13px; border: 0; padding: .7rem 1.25rem;
    font-weight: 700; background: #65dc92; color: #062014;
}
.stButton > button:hover { background: #83e8a7; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- Model ----------
@st.cache_resource(show_spinner=False)
def load_predictor():
    return PlantDiseasePredictor()

predictor = load_predictor()

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">PlantCare AI · Computer Vision</div>
  <h1>Detect plant diseases<br>with a single leaf photo.</h1>
  <p>Upload a clear leaf image and get an AI-assisted disease classification,
     confidence score, top alternatives, and practical next-step guidance.</p>
  <span class="status">● AI MODEL ONLINE · ResNet50</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">🔬 Leaf diagnosis</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop a leaf image here",
    type=["jpg", "jpeg", "png", "webp"],
    help="For best results, use one clear leaf with good lighting.",
    label_visibility="collapsed",
)

if uploaded:
    image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    col_img, col_result = st.columns([1.05, 1], gap="large")

    with col_img:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded leaf", use_container_width=True)
        st.markdown(
            f'<div class="small">Image · {image.width} × {image.height}px · {uploaded.size/1024:.0f} KB</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if st.button("✨ Analyze leaf", use_container_width=True):
            with st.spinner("Inspecting leaf patterns..."):
                start = time.perf_counter()
                result = predictor.predict(image)
                elapsed = time.perf_counter() - start
            st.session_state["result"] = result
            st.session_state["elapsed"] = elapsed

    result = st.session_state.get("result")
    if result:
        label = result["label"]
        confidence = result["confidence"]
        info = get_disease_info(label)
        healthy = info["healthy"]

        with col_result:
            badge_cls = "badge-ok" if healthy else "badge-warn"
            badge_text = "HEALTHY SIGNAL" if healthy else "DISEASE SIGNAL"
            st.markdown(f"""
            <div class="diagnosis">
              <span class="badge {badge_cls}">{badge_text}</span>
              <h2>{info["display_name"]}</h2>
              <p class="small">{info["crop"]} · {info["category"]}</p>
              <p>{info["description"]}</p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Confidence", f"{confidence*100:.1f}%")
            with c2:
                st.metric("Inference", f"{st.session_state.get('elapsed', 0)*1000:.0f} ms")

        st.markdown('<div class="section-title">📊 Prediction confidence</div>', unsafe_allow_html=True)
        top = result["top_k"]
        chart_data = {x["label"]: x["confidence"] for x in top}
        st.bar_chart(chart_data, horizontal=True)

        st.markdown('<div class="section-title">🌱 What this means</div>', unsafe_allow_html=True)
        info_cols = st.columns(3)
        with info_cols[0]:
            st.markdown('<div class="card"><b>👀 Symptoms</b><br><span class="small">' +
                        "<br>• ".join([""] + info["symptoms"]) + '</span></div>', unsafe_allow_html=True)
        with info_cols[1]:
            st.markdown('<div class="card"><b>🛠 Recommended actions</b><br><span class="small">' +
                        "<br>• ".join([""] + info["actions"]) + '</span></div>', unsafe_allow_html=True)
        with info_cols[2]:
            st.markdown('<div class="card"><b>🛡 Prevention</b><br><span class="small">' +
                        "<br>• ".join([""] + info["prevention"]) + '</span></div>', unsafe_allow_html=True)

        st.caption("AI output is informational and should be confirmed by an agronomist or plant pathologist before treatment decisions.")

else:
    st.markdown("""
    <div class="card" style="text-align:center; padding:3rem 1rem;">
      <div style="font-size:3.2rem;">🌿</div>
      <h2 style="font-family:'Space Grotesk';">Your leaf is waiting.</h2>
      <p class="small">Upload a JPG, PNG or WEBP image to start an AI diagnosis.</p>
      <p class="small">Best results: one leaf, centered, sharp focus, natural lighting.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-title">⚡ Model snapshot</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
for col, label, value in [
    (m1, "Architecture", "ResNet50"),
    (m2, "Classes", "38"),
    (m3, "Input", "224 × 224"),
    (m4, "Dataset", "PlantVillage"),
]:
    with col:
        st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

with st.expander("About the model & dataset"):
    st.write(
        "This demo uses a ResNet50 image-classification checkpoint fine-tuned for "
        "PlantVillage plant-disease classes. The upstream model card reports 38 classes "
        "and 95%+ accuracy on its PlantVillage test set. This project does not represent "
        "that benchmark as an independently reproduced result."
    )
    st.write(
        "PlantVillage is an open-access dataset containing more than 54,000 leaf images "
        "across 14 crop species and 38 crop-disease/healthy classes."
    )
