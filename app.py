# ============================================================
# EMBERSIGHT - FIRE & SMOKE DETECTION
# YOLOv8n + Streamlit
# Revised UI/UX + Live Camera
# (Fixed: all HTML blocks flattened to avoid Markdown code-block bug)
# ============================================================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import pandas as pd
import io
import time
import av

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase,
)


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EmberSight | Fire & Smoke Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================
# NOTE: CSS inside <style> is NOT affected by the markdown
# code-block issue (that only affects the *outer* indentation
# of the string passed to st.markdown). Kept as-is, but the
# whole block is still emitted through a single unsafe_allow_html
# call, which is fine for <style> tags.

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800;900&display=swap');

:root {
    --ink: #111318;
    --muted: #6b7078;
    --muted-2: #9a9fa8;

    --line: #e9e9e5;
    --line-soft: #f0f0ec;

    --paper: #f7f6f2;
    --card: #ffffff;

    --fire: #e95b2f;
    --fire-deep: #c8431c;

    --smoke: #53606d;
    --smoke-soft: #8a94a1;

    --good: #1f9d63;

    --shadow-sm: 0 2px 10px rgba(17,19,24,.04);
    --shadow-md: 0 10px 30px rgba(17,19,24,.06);

    --radius-lg: 24px;
    --radius-md: 16px;
    --radius-sm: 11px;
}

/* ======================================================
   GLOBAL
   ====================================================== */

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 88% 0%, rgba(233,91,47,.12), transparent 32%),
        radial-gradient(circle at 0% 85%, rgba(83,96,109,.09), transparent 30%),
        var(--paper);
    color: var(--ink);
}

.block-container {
    max-width: 1180px;
    padding-top: 1.6rem;
    padding-bottom: 4rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-thumb {
    background: #d7d7d1;
    border-radius: 20px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

/* ======================================================
   HERO
   ====================================================== */

.hero {
    border: 1px solid var(--line);
    background: rgba(255,255,255,.74);
    backdrop-filter: blur(14px);
    border-radius: var(--radius-lg);
    padding: 36px 40px;
    position: relative;
    overflow: hidden;
    margin-bottom: 22px;
    box-shadow: var(--shadow-md);
}

.hero:after {
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    right: -90px;
    top: -120px;
    border-radius: 50%;
    border: 1px solid rgba(233,91,47,.22);
    box-shadow:
        0 0 0 30px rgba(233,91,47,.04),
        0 0 0 62px rgba(233,91,47,.025);
}

.hero-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
}

.kicker {
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--fire);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.kicker .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--fire);
    box-shadow: 0 0 0 4px rgba(233,91,47,.18);
}

.hero h1 {
    font-size: clamp(2rem, 5vw, 4rem);
    line-height: .98;
    letter-spacing: -.055em;
    margin: 0;
    font-weight: 900;
}

.hero p {
    color: var(--muted);
    max-width: 640px;
    font-size: 1rem;
    line-height: 1.7;
    margin-top: 14px;
}

.badge-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 16px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 7px 12px;
    border-radius: 999px;
    background: #111318;
    color: white;
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
    letter-spacing: .02em;
}

.badge.ghost {
    background: transparent;
    border: 1px solid var(--line);
    color: var(--muted);
}

/* ======================================================
   CARD
   ====================================================== */

.card {
    border: 1px solid var(--line);
    background: rgba(255,255,255,.86);
    border-radius: var(--radius-lg);
    padding: 22px 24px;
    height: 100%;
    box-shadow: var(--shadow-sm);
    transition: box-shadow .2s ease, border-color .2s ease;
}

.card:hover {
    box-shadow: var(--shadow-md);
    border-color: #ddddd7;
}

.step-tag {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 7px;
    background: var(--ink);
    color: white;
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
    margin-right: 8px;
}

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--muted);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
}

.card-title {
    font-size: 1.15rem;
    font-weight: 800;
    margin-bottom: 4px;
    letter-spacing: -.01em;
}

.card-sub {
    color: var(--muted);
    font-size: .86rem;
    margin-bottom: 4px;
    line-height: 1.55;
}

/* ======================================================
   METRIC
   ====================================================== */

.metric {
    border: 1px solid var(--line);
    background: #fff;
    border-radius: var(--radius-md);
    padding: 16px 17px;
    min-height: 96px;
    position: relative;
    overflow: hidden;
}

.metric.accent-fire {
    border-color: rgba(233,91,47,.35);
    background: linear-gradient(180deg, rgba(233,91,47,.06), #fff 60%);
}

.metric.accent-smoke {
    border-color: rgba(83,96,109,.30);
    background: linear-gradient(180deg, rgba(83,96,109,.06), #fff 60%);
}

.metric .label {
    color: var(--muted);
    font-size: .70rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    display: flex;
    align-items: center;
    gap: 6px;
}

.metric .value {
    font-size: 1.7rem;
    font-weight: 800;
    margin-top: 6px;
    letter-spacing: -.02em;
}

.metric .sub-value {
    font-size: .72rem;
    color: var(--muted-2);
    margin-top: 2px;
}

/* ======================================================
   DETECTION PILL
   ====================================================== */

.det-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 7px 12px;
    margin: 3px 6px 3px 0;
    background: white;
    font-size: .78rem;
    font-weight: 600;
    box-shadow: var(--shadow-sm);
}

.dot-fire, .dot-smoke {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
}

.dot-fire {
    background: var(--fire);
}

.dot-smoke {
    background: var(--smoke);
}

/* ======================================================
   EMPTY STATE
   ====================================================== */

.empty-state {
    border: 1.5px dashed var(--line);
    border-radius: var(--radius-md);
    padding: 30px 20px;
    text-align: center;
    color: var(--muted);
    font-size: .88rem;
    background: #fbfbf8;
}

.empty-state .big {
    font-size: 1.6rem;
    margin-bottom: 6px;
}

/* ======================================================
   FILE UPLOADER
   ====================================================== */

[data-testid="stFileUploader"] {
    width: 100%;
}

[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed #d5d5cf !important;
    border-radius: 16px !important;
    background: #ffffff !important;
    min-height: 82px !important;
    padding: 14px 18px !important;
    transition: border-color .2s ease, background .2s ease, box-shadow .2s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #e95b2f !important;
    background: #fffaf7 !important;
    box-shadow: 0 4px 16px rgba(233,91,47,.07) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #6b7078 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #6b7078 !important;
    font-family: 'Manrope', sans-serif !important;
}

/* ======================================================
   UPLOAD BUTTON
   ====================================================== */

[data-testid="stFileUploaderDropzone"] button {
    background: #111318 !important;
    color: #ffffff !important;
    border: 1px solid #111318 !important;
    border-radius: 10px !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    padding: 7px 15px !important;
    min-height: 38px !important;
    transition: background .15s ease, border-color .15s ease, transform .12s ease;
}

[data-testid="stFileUploaderDropzone"] button span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: #e95b2f !important;
    border-color: #e95b2f !important;
    transform: translateY(-1px);
}

/* ======================================================
   FILE SIZE TEXT
   ====================================================== */

[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #9a9fa8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .68rem !important;
}

/* ======================================================
   UPLOAD ICON
   ====================================================== */

[data-testid="stFileUploaderDropzone"] svg {
    color: #e95b2f !important;
}

/* ======================================================
   FILE YANG SUDAH DIPILIH
   ====================================================== */

[data-testid="stFileUploaderFile"] {
    background: #fffaf7 !important;
    border: 1px solid #f0d5c9 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploaderFile"] span {
    color: #111318 !important;
}

/* ======================================================
   BUTTON
   ====================================================== */

.stButton > button, .stDownloadButton > button {
    border-radius: 13px;
    border: 1px solid #111318;
    background: #111318;
    color: white;
    font-weight: 700;
    padding: .65rem 1.2rem;
    transition: transform .12s ease, background .15s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--fire-deep);
    border-color: var(--fire-deep);
    transform: translateY(-1px);
}

.stButton > button:active, .stDownloadButton > button:active {
    transform: translateY(0px);
}

/* ======================================================
   SIDEBAR
   ====================================================== */

section[data-testid="stSidebar"] {
    background: #fcfcfa;
    border-right: 1px solid var(--line);
}

.side-title {
    font-weight: 800;
    font-size: 1.02rem;
    margin-bottom: 2px;
}

.side-caption {
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    color: var(--muted);
    margin: 2px 0;
}

/* ======================================================
   CAMERA
   ====================================================== */

.camera-card {
    border: 1px solid var(--line);
    background: rgba(255,255,255,.86);
    border-radius: var(--radius-lg);
    padding: 24px;
    box-shadow: var(--shadow-sm);
}

.camera-info {
    border: 1px solid var(--line);
    background: #fbfbf8;
    border-radius: var(--radius-md);
    padding: 15px 17px;
    color: var(--muted);
    line-height: 1.7;
    margin: 15px 0;
}

.camera-live {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(31,157,99,.10);
    border: 1px solid rgba(31,157,99,.25);
    color: var(--good);
    font-size: .75rem;
    font-weight: 700;
}

/* ======================================================
   FOOTER
   ====================================================== */

.footer-note {
    text-align: center;
    color: #9a9fa8;
    font-family: 'DM Mono', monospace;
    font-size: .65rem;
    letter-spacing: .08em;
    margin-top: 32px;
}

/* ======================================================
   TABS - SELALU MERAH
   ====================================================== */

[data-baseweb="tab-list"] {
    gap: 8px !important;
}

[data-baseweb="tab-list"] [data-baseweb="tab"] {
    color: #e95b2f !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}

[data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="false"] {
    color: #e95b2f !important;
    opacity: 1 !important;
}

[data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"] {
    color: #e95b2f !important;
    opacity: 1 !important;
}

[data-baseweb="tab-list"] [data-baseweb="tab"]:hover {
    color: #e95b2f !important;
    opacity: 1 !important;
}

[data-baseweb="tab-list"] [data-baseweb="tab"] p {
    color: #e95b2f !important;
    -webkit-text-fill-color: #e95b2f !important;
    opacity: 1 !important;
    font-weight: 800 !important;
}

[data-baseweb="tab-list"] [data-baseweb="tab"] span {
    color: #e95b2f !important;
    -webkit-text-fill-color: #e95b2f !important;
    opacity: 1 !important;
}

[data-baseweb="tab-highlight"] {
    background-color: #e95b2f !important;
    height: 2px !important;
}

/* ======================================================
   RADIO CAMERA - MERAH
   ====================================================== */

div[data-testid="stRadio"] {
    color: #e95b2f !important;
}

div[data-testid="stRadio"] label {
    color: #e95b2f !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {
    color: #e95b2f !important;
    -webkit-text-fill-color: #e95b2f !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

div[data-testid="stRadio"] label:hover {
    color: #e95b2f !important;
}

div[data-testid="stRadio"] label:has(input:checked) {
    color: #e95b2f !important;
    font-weight: 800 !important;
}

div[data-testid="stRadio"] label:has(input:checked) p,
div[data-testid="stRadio"] label:has(input:checked) span {
    color: #e95b2f !important;
    -webkit-text-fill-color: #e95b2f !important;
}

div[data-testid="stRadio"] label div[role="radio"] {
    border-color: #e95b2f !important;
}

div[data-testid="stRadio"] label div[role="radio"][aria-checked="true"] {
    background-color: #e95b2f !important;
    border-color: #e95b2f !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "best.pt"


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ============================================================
# 5. HEADER
# ============================================================
# NOTE: this block is the one that broke in the screenshot.
# Every line below is flush-left (no nested indentation), so
# Markdown will never interpret it as a code block.

st.markdown("""
<div class="hero">
<div class="hero-top">
<div>
<div class="kicker">
<span class="dot"></span>
Computer Vision / YOLOv8
</div>
<h1>EmberSight<span style="color:#e95b2f;">.</span></h1>
<p>Visual detection interface for identifying <b>Fire</b> and <b>Smoke</b> from uploaded images or live camera, powered by a custom-trained YOLOv8 model.</p>
<div class="badge-row">
<span class="badge">🔥 MODEL · YOLOv8n / 640px</span>
<span class="badge ghost">Classes · Fire, Smoke</span>
<span class="badge ghost">JPG · PNG · WEBP</span>
<span class="badge ghost">📷 Live Camera</span>
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 6. CHECK MODEL
# ============================================================

if not model_loaded:
    st.error("Model `best.pt` belum ditemukan atau gagal dimuat.")
    st.code(model_error)
    st.stop()


# ============================================================
# 7. SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="side-title">⚙️ Detection Controls</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="side-caption">Tune inference before running</div>',
        unsafe_allow_html=True
    )

    st.write("")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Lower = more detections (may include false positives).",
    )

    image_size = st.select_slider(
        "Inference image size",
        options=[320, 480, 640, 768, 960],
        value=640,
        help="Larger sizes can improve accuracy on small objects but run slower.",
    )

    st.divider()

    st.markdown(
        '<div class="side-caption">MODEL</div>',
        unsafe_allow_html=True
    )

    st.caption("best.pt · YOLOv8n")

    st.markdown(
        '<div class="side-caption">CLASSES</div>',
        unsafe_allow_html=True
    )

    st.caption("🔥 Fire   ·   💨 Smoke")

    st.markdown(
        '<div class="side-caption">INPUT FORMATS</div>',
        unsafe_allow_html=True
    )

    st.caption("JPG · JPEG · PNG · WEBP")


# ============================================================
# 8. DETECTION MODE
# ============================================================

tab_upload, tab_camera = st.tabs(["📤 Upload Gambar", "📷 Kamera Live"])


# ============================================================
# 9. UPLOAD IMAGE
# ============================================================

with tab_upload:

    left, right = st.columns([1, 1], gap="large")

    # ========================================================
    # LEFT - INPUT
    # ========================================================

    with left:

        st.markdown("""
<div class="card">
<div class="section-label"><span class="step-tag">1</span>INPUT</div>
<div class="card-title">Drop an image</div>
<div class="card-sub">Upload a fire/smoke scene and run the trained detector.</div>
</div>
""", unsafe_allow_html=True)

        st.write("")

        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file:

            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Original image",
                use_container_width=True
            )

        else:

            st.markdown("""
<div class="empty-state">
<div class="big">🖼️</div>
No image yet — drag and drop or browse a file above.
</div>
""", unsafe_allow_html=True)

    # ========================================================
    # RIGHT - SETTINGS
    # ========================================================

    with right:

        st.markdown("""
<div class="card">
<div class="section-label"><span class="step-tag">2</span>INFERENCE</div>
<div class="card-title">Detection settings</div>
<div class="card-sub">Adjust confidence and image size from the sidebar, then run inference.</div>
</div>
""", unsafe_allow_html=True)

        st.write("")

        if uploaded_file:

            size_kb = len(uploaded_file.getbuffer()) / 1024

            m1, m2 = st.columns(2)

            with m1:

                file_display = uploaded_file.name[:20]

                if len(uploaded_file.name) > 20:
                    file_display += "…"

                st.markdown(f"""
<div class="metric">
<div class="label">Selected file</div>
<div class="value" style="font-size:1rem;">{file_display}</div>
<div class="sub-value">{size_kb:.1f} KB</div>
</div>
""", unsafe_allow_html=True)

            with m2:

                st.markdown(f"""
<div class="metric">
<div class="label">Settings</div>
<div class="value" style="font-size:1rem;">conf {confidence:.2f}</div>
<div class="sub-value">imgsz {image_size}px</div>
</div>
""", unsafe_allow_html=True)

            st.write("")

            run_detection = st.button(
                "Run Detection  →",
                use_container_width=True
            )

        else:

            st.markdown("""
<div class="empty-state">
<div class="big">⏳</div>
Upload an image first to enable detection.
</div>
""", unsafe_allow_html=True)

            run_detection = False

    # ========================================================
    # 10. IMAGE INFERENCE
    # ========================================================

    if uploaded_file and run_detection:

        suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        try:

            with st.status("Running YOLOv8 inference...", expanded=False) as status:

                t0 = time.time()

                results = model.predict(
                    source=temp_path,
                    conf=confidence,
                    imgsz=image_size,
                    verbose=False
                )

                elapsed = time.time() - t0

                result = results[0]

                status.update(
                    label=f"Inference complete in {elapsed:.2f}s",
                    state="complete"
                )

            # ------------------------------------------------
            # ANNOTATED IMAGE
            # ------------------------------------------------

            annotated = result.plot()
            annotated_rgb = annotated[:, :, ::-1]

            # ------------------------------------------------
            # DETECTION DATA
            # ------------------------------------------------

            boxes = result.boxes
            rows = []

            if boxes is not None:

                for box in boxes:

                    class_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()

                    rows.append({
                        "Class": result.names[class_id],
                        "Confidence": f"{conf * 100:.2f}%",
                        "X1": round(xyxy[0], 1),
                        "Y1": round(xyxy[1], 1),
                        "X2": round(xyxy[2], 1),
                        "Y2": round(xyxy[3], 1),
                    })

            # =================================================
            # RESULT HEADER
            # =================================================

            st.markdown("---")

            st.markdown("""
<div class="section-label"><span class="step-tag">3</span>RESULT</div>
<div class="card-title">Detection output</div>
""", unsafe_allow_html=True)

            st.write("")

            result_col, info_col = st.columns([1.65, 1], gap="large")

            # =================================================
            # RESULT IMAGE
            # =================================================

            with result_col:

                st.image(
                    annotated_rgb,
                    caption="YOLOv8 detection result",
                    use_container_width=True
                )

            # =================================================
            # RESULT INFO
            # =================================================

            with info_col:

                fire_count = sum(1 for x in rows if x["Class"].lower() == "fire")
                smoke_count = sum(1 for x in rows if x["Class"].lower() == "smoke")
                total = len(rows)

                c1, c2 = st.columns(2)

                with c1:

                    st.markdown(f"""
<div class="metric accent-fire">
<div class="label">🔥 Fire</div>
<div class="value">{fire_count}</div>
</div>
""", unsafe_allow_html=True)

                with c2:

                    st.markdown(f"""
<div class="metric accent-smoke">
<div class="label">💨 Smoke</div>
<div class="value">{smoke_count}</div>
</div>
""", unsafe_allow_html=True)

                st.write("")

                st.markdown(f"""
<div class="metric">
<div class="label">Total detections</div>
<div class="value">{total}</div>
<div class="sub-value">at conf ≥ {confidence:.2f} · {image_size}px</div>
</div>
""", unsafe_allow_html=True)

                st.write("")

                if rows:

                    pills = ""

                    for row in rows:

                        cls = row["Class"].lower()
                        dot = "dot-fire" if cls == "fire" else "dot-smoke"

                        pills += (
                            f'<span class="det-pill">'
                            f'<span class="{dot}"></span>'
                            f'{row["Class"]} · {row["Confidence"]}'
                            f'</span>'
                        )

                    st.markdown(pills, unsafe_allow_html=True)

            # =================================================
            # TABLE + DOWNLOAD
            # =================================================

            st.markdown("---")

            if rows:

                st.markdown("### Detection details")

                df = pd.DataFrame(rows)

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # IMAGE DOWNLOAD
                # ------------------------------------------------

                output_image = Image.fromarray(annotated_rgb)

                img_buffer = io.BytesIO()

                output_image.save(img_buffer, format="JPEG", quality=95)

                # ------------------------------------------------
                # CSV DOWNLOAD
                # ------------------------------------------------

                csv_buffer = io.StringIO()

                df.to_csv(csv_buffer, index=False)

                dl1, dl2 = st.columns(2)

                with dl1:

                    st.download_button(
                        "⬇ Download annotated image",
                        data=img_buffer.getvalue(),
                        file_name=(
                            "detected_"
                            + uploaded_file.name.rsplit(".", 1)[0]
                            + ".jpg"
                        ),
                        mime="image/jpeg",
                        use_container_width=True,
                    )

                with dl2:

                    st.download_button(
                        "⬇ Download detections (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=(
                            "detections_"
                            + uploaded_file.name.rsplit(".", 1)[0]
                            + ".csv"
                        ),
                        mime="text/csv",
                        use_container_width=True,
                    )

            else:

                st.markdown("""
<div class="empty-state">
<div class="big">✅</div>
No Fire or Smoke detected at the selected confidence threshold.
<br><br>
Try lowering the confidence slider in the sidebar.
</div>
""", unsafe_allow_html=True)

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)


# ============================================================
# 11. LIVE CAMERA PROCESSOR
# ============================================================

class FireSmokeProcessor(VideoProcessorBase):

    def recv(self, frame):

        # ----------------------------------------------------
        # GET CAMERA FRAME
        # ----------------------------------------------------

        img = frame.to_ndarray(format="bgr24")

        # ----------------------------------------------------
        # YOLO INFERENCE
        # ----------------------------------------------------

        results = model.predict(
            source=img,
            imgsz=640,
            conf=confidence,
            verbose=False
        )

        result = results[0]

        # ----------------------------------------------------
        # DRAW BOUNDING BOX
        # ----------------------------------------------------

        annotated = result.plot()

        # ----------------------------------------------------
        # RETURN FRAME
        # ----------------------------------------------------

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ============================================================
# 12. LIVE CAMERA
# ============================================================

with tab_camera:

    st.markdown("""
<div class="camera-card">
<div class="section-label"><span class="step-tag">C</span>LIVE CAMERA</div>
<div class="card-title">📷 Fire & Smoke Live Detection</div>
<div class="card-sub">Gunakan kamera perangkat untuk mendeteksi api dan asap secara langsung menggunakan YOLOv8n.</div>
</div>
""", unsafe_allow_html=True)

    st.write("")

    # ========================================================
    # CAMERA SELECTION
    # ========================================================

    st.markdown("### 📷 Pilih Kamera")

    camera_type = st.radio(
        "Sumber kamera",
        ["📱 Kamera Depan", "📷 Kamera Belakang"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # ========================================================
    # CAMERA FACING MODE
    # ========================================================

    if camera_type == "📱 Kamera Depan":
        facing_mode = "user"
        camera_name = "Kamera Depan"
    else:
        facing_mode = "environment"
        camera_name = "Kamera Belakang"

    # ========================================================
    # CAMERA INFORMATION
    # ========================================================

    st.markdown(f"""
<div class="camera-info">
📷 <b>Kamera:</b> {camera_name}
<br>
🎯 <b>Confidence:</b> {confidence:.0%}
<br>
🖼️ <b>Inference Size:</b> 640 × 640
<br><br>
▶️ Tekan <b>START</b> untuk memulai kamera.
<br>
⏹️ Tekan <b>STOP</b> untuk menghentikan kamera.
</div>
""", unsafe_allow_html=True)

    # ========================================================
    # WEBRTC CAMERA
    # ========================================================

    webrtc_ctx = webrtc_streamer(
        key="embersight-camera-" + facing_mode,
        mode=WebRtcMode.SENDRECV,

        # ----------------------------------------------------
        # CAMERA CONSTRAINTS
        # ----------------------------------------------------

        media_stream_constraints={
            "video": {
                "facingMode": facing_mode,
                "width": {"ideal": 1280},
                "height": {"ideal": 720}
            },
            "audio": False
        },

        # ----------------------------------------------------
        # VIDEO PROCESSOR
        # ----------------------------------------------------

        video_processor_factory=FireSmokeProcessor,

        async_processing=True,

        # ----------------------------------------------------
        # STUN SERVER
        # ----------------------------------------------------

        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        }
    )

    # ========================================================
    # CAMERA STATUS
    # ========================================================

    if webrtc_ctx.state.playing:

        st.markdown(
            '<div class="camera-live">🟢 CAMERA LIVE</div>',
            unsafe_allow_html=True
        )

        st.caption(
            f"Deteksi aktif · {camera_name} · Confidence {confidence:.0%}"
        )

    else:

        st.info(
            "⚪ Kamera belum aktif. Tekan tombol START untuk memulai deteksi."
        )


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown(
    '<div class="footer-note">EMBERSIGHT · FIRE & SMOKE DETECTION · YOLOv8</div>',
    unsafe_allow_html=True,
)