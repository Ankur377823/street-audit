"""
Streamlit Roadside Audit App — v5 Enhanced UI
NIT Calicut | Automated Roadside Hazard Audit System

Logic: street_audit_v5.py
  - Pinhole camera similar-triangles model for longitudinal pole distance
  - Percentile-based robust mask extremes
  - Dynamic class resolution from model.names
  - Depth-gated canopy check
  - NEAR POLE hazard reason
  - targets always initialized (no NameError)
"""

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NIT Calicut — Roadside Audit",
    page_icon="🛣️",
    layout="wide",
)

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root theme ── */
:root {
    --bg-deep:      #0a0d14;
    --bg-card:      #111520;
    --bg-glass:     rgba(17, 21, 32, 0.85);
    --accent-lime:  #c6f135;
    --accent-cyan:  #38f0d4;
    --accent-red:   #ff4757;
    --accent-amber: #ffb142;
    --border:       rgba(198, 241, 53, 0.18);
    --text-primary: #eef1f8;
    --text-muted:   #7a8299;
    --font-display: 'Syne', sans-serif;
    --font-body:    'DM Sans', sans-serif;
}

/* ── Global overrides ── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg-deep) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(198,241,53,0.07) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 90% 80%, rgba(56,240,212,0.05) 0%, transparent 60%),
        var(--bg-deep);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    color: var(--accent-lime) !important;
    letter-spacing: -0.02em;
}

/* ── Main title ── */
h1 {
    font-family: var(--font-display) !important;
    font-weight: 800 !important;
    font-size: 2.4rem !important;
    letter-spacing: -0.04em !important;
    background: linear-gradient(90deg, var(--accent-lime) 0%, var(--accent-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0 !important;
}

/* ── Sub-headings ── */
h2, h3, h4 {
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
}

/* ── Caption / helper text ── */
.stCaption, [data-testid="stCaptionContainer"] p {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(198, 241, 53, 0.12);
}

[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-lime), var(--accent-cyan));
}

[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: var(--accent-lime) !important;
    line-height: 1.1 !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent-lime) !important;
}

.stSlider [data-baseweb="slider"] [data-testid="stTickBar"] {
    color: var(--text-muted) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-lime) !important;
}

[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
}

[data-testid="stFileUploadDropzone"] p {
    color: var(--text-muted) !important;
}

/* ── Buttons ── */
.stDownloadButton > button, .stButton > button {
    background: linear-gradient(135deg, var(--accent-lime) 0%, var(--accent-cyan) 100%) !important;
    color: #0a0d14 !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    transition: opacity 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease !important;
    box-shadow: 0 0 20px rgba(198, 241, 53, 0.25) !important;
}

.stDownloadButton > button:hover, .stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(198, 241, 53, 0.4) !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border) !important;
    margin: 1rem 0 !important;
}

/* ── Image container ── */
[data-testid="stImage"] {
    border-radius: 16px !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
    box-shadow: 0 0 40px rgba(56, 240, 212, 0.06) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: var(--accent-lime) !important;
}

/* ── Info / alert boxes ── */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-muted) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

/* ── Custom component boxes ── */
.hazard-box {
    background: rgba(255, 71, 87, 0.08);
    border-left: 3px solid var(--accent-red);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-family: var(--font-body);
    font-size: 0.85rem;
    color: #ffaab2;
    display: flex;
    align-items: center;
    gap: 8px;
}

.safe-box {
    background: rgba(198, 241, 53, 0.06);
    border-left: 3px solid var(--accent-lime);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-family: var(--font-body);
    font-size: 0.85rem;
    color: #d6f57d;
}

.info-box {
    background: rgba(56, 240, 212, 0.06);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-family: var(--font-body);
    font-size: 0.85rem;
    color: #a0f5ea;
    line-height: 1.6;
}

/* Hero badge */
.hero-badge {
    display: inline-block;
    background: rgba(198, 241, 53, 0.1);
    border: 1px solid rgba(198, 241, 53, 0.3);
    color: var(--accent-lime);
    font-family: var(--font-display);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 99px;
    margin-bottom: 10px;
}

/* How-it-works cards */
.how-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 20px;
    height: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.how-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(198, 241, 53, 0.08);
}

.how-card-icon {
    font-size: 2rem;
    margin-bottom: 12px;
}

.how-card-title {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 1rem;
    color: var(--accent-lime);
    margin-bottom: 8px;
}

.how-card-body {
    font-family: var(--font-body);
    font-size: 0.84rem;
    color: var(--text-muted);
    line-height: 1.65;
}

/* Stats subheader */
.section-label {
    font-family: var(--font-display);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Sidebar meta tag */
.meta-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(198, 241, 53, 0.07);
    border: 1px solid rgba(198, 241, 53, 0.15);
    border-radius: 8px;
    padding: 5px 10px;
    font-size: 0.77rem;
    color: var(--text-muted);
    margin-bottom: 6px;
    width: 100%;
}

/* Override Streamlit markdown text */
p, li {
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)

# ── HERO HEADER ───────────────────────────────────────────────────────────────

st.markdown('<div class="hero-badge">⚡ Final Year Project · NIT Calicut</div>', unsafe_allow_html=True)
st.title("Roadside Hazard Audit System")
st.markdown(
    '<p style="color:#7a8299;font-family:\'DM Sans\',sans-serif;font-size:1rem;margin-top:-6px;">'
    'AI-powered pole &amp; tree hazard detection · YOLOv9-seg · Pinhole depth model'
    '</p>',
    unsafe_allow_html=True,
)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

MODEL_PATH                  = 'best.pt'
REAL_POLE_HEIGHT_M          = 8.0
VEHICLE_SPEED_KMH           = 35.0
FRAME_RATE_FPS              = 4.0
CAMERA_VFOV_DEG             = 69.0
POLE_PROXIMITY_THRESHOLD_PX = 300
CANOPY_ROAD_LEFT_RATIO      = 0.55
CANOPY_ROAD_RIGHT_RATIO     = 0.45
SIDE_BUFFER_PX              = 80

METRES_PER_FRAME = (VEHICLE_SPEED_KMH * 1000 / 3600) / FRAME_RATE_FPS

# ─── HELPERS ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def get_class_indices(model):
    names    = {v.lower(): k for k, v in model.names.items()}
    pole_cls = names.get('pole', 0)
    tree_cls = names.get('tree', 1)
    return pole_cls, tree_cls


def focal_length_px(image_h, vfov_deg):
    vfov_rad = np.radians(vfov_deg)
    return (image_h / 2.0) / np.tan(vfov_rad / 2.0)


def robust_mask_extremes(pts):
    top_y    = np.percentile(pts[:, 1], 2)
    base_y   = np.percentile(pts[:, 1], 98)
    top_idx  = np.argmin(np.abs(pts[:, 1] - top_y))
    base_idx = np.argmin(np.abs(pts[:, 1] - base_y))
    top      = tuple(pts[top_idx].astype(int))
    base     = tuple(pts[base_idx].astype(int))
    px_h     = abs(base_y - top_y)
    return top, base, px_h


def compute_depth_weight(base_y, horizon_y, image_h):
    denom = max(1, base_y - horizon_y)
    return ((image_h - horizon_y) / denom) ** 2.15


def distance_to_pole_m(px_h, focal_px):
    if px_h < 5:
        return None
    return (REAL_POLE_HEIGHT_M * focal_px) / px_h


def pole_to_pole_span_m(p_near, p_far, focal_px):
    d_near = distance_to_pole_m(p_near['px_h'], focal_px)
    d_far  = distance_to_pole_m(p_far['px_h'],  focal_px)
    if d_near is None or d_far is None:
        return None, None, None
    return abs(d_far - d_near), d_near, d_far


# ─── CORE AUDIT FUNCTION ──────────────────────────────────────────────────────

def run_audit(img, model, vfov_deg, pole_prox_px):
    h, w, _  = img.shape
    results  = model(img)[0]
    POLE_CLS, TREE_CLS = get_class_indices(model)

    focal_px  = focal_length_px(h, vfov_deg)
    horizon_y = h * 0.47

    all_poles, all_trees = [], []
    targets              = []
    hazards_log          = []

    if results.masks is not None:
        for i, mask in enumerate(results.masks.xy):
            cls  = int(results.boxes.cls[i])
            bbox = results.boxes.xyxy[i].cpu().numpy().astype(int)
            pts  = mask.astype(np.float32)
            if len(pts) < 3:
                continue

            top, base, px_h = robust_mask_extremes(pts)
            depth_w = compute_depth_weight(base[1], horizon_y, h)

            data = {
                'base'   : base,
                'top'    : top,
                'px_h'   : px_h,
                'depth_w': depth_w,
                'mask'   : pts.astype(np.int32),
                'bbox'   : bbox,
            }

            if cls == POLE_CLS:
                all_poles.append(data)
            elif cls == TREE_CLS:
                all_trees.append(data)

    if all_poles:
        closest_p = max(all_poles, key=lambda x: x['base'][1])
        is_left   = closest_p['base'][0] < w / 2

        if is_left:
            targets = [p for p in all_poles if p['base'][0] < (w / 2 + SIDE_BUFFER_PX)]
        else:
            targets = [p for p in all_poles if p['base'][0] > (w / 2 - SIDE_BUFFER_PX)]

        targets.sort(key=lambda x: x['px_h'], reverse=True)

    pole_distances = []

    for i, p in enumerate(targets):
        d_m = distance_to_pole_m(p['px_h'], focal_px)
        dist_label = f"({d_m:.0f}m away)" if d_m else ""

        cv2.rectangle(img,
                      (p['bbox'][0], p['bbox'][1]),
                      (p['bbox'][2], p['bbox'][3]),
                      (255, 120, 0), 2)
        cv2.putText(img, f"POLE P{i+1} {dist_label}",
                    (p['bbox'][0], p['bbox'][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 0), 2)

        if i < len(targets) - 1:
            p_near, p_far = targets[i], targets[i + 1]
            span_m, d_near_m, d_far_m = pole_to_pole_span_m(p_near, p_far, focal_px)

            if span_m is not None:
                pole_distances.append((f"P{i+1}→P{i+2}", span_m, d_near_m, d_far_m))

                cv2.line(img, p_near['base'], p_far['base'], (0, 255, 255), 3)
                mid = (
                    (p_near['base'][0] + p_far['base'][0]) // 2,
                    (p_near['base'][1] + p_far['base'][1]) // 2,
                )
                label = f"{span_m:.1f}m span"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
                cv2.rectangle(img,
                              (mid[0] - 6, mid[1] - th - 16),
                              (mid[0] + tw + 6, mid[1] + 4),
                              (0, 0, 0), -1)
                cv2.putText(img, label,
                            (mid[0], mid[1] - 12),
                            cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255), 2)

    for i, tree in enumerate(all_trees):
        is_hazard = False
        reasons   = []

        if targets:
            cp = min(targets,
                     key=lambda p: np.linalg.norm(
                         np.array(p['base']) - np.array(tree['base'])
                     ))

            if tree['top'][1] < cp['top'][1]:
                is_hazard = True
                reasons.append("HEIGHT")

            dist_to_pole = np.linalg.norm(
                np.array(cp['base']) - np.array(tree['base'])
            )
            if dist_to_pole < pole_prox_px:
                is_hazard = True
                reasons.append("NEAR POLE")

        x_min = np.min(tree['mask'][:, 0])
        x_max = np.max(tree['mask'][:, 0])
        if (tree['base'][1] > horizon_y
                and x_min < (w * CANOPY_ROAD_LEFT_RATIO)
                and x_max > (w * CANOPY_ROAD_RIGHT_RATIO)):
            is_hazard = True
            reasons.append("CANOPY")

        if is_hazard:
            color = (0, 0, 255)
            label = f"HAZARD: {' & '.join(reasons)}"
            hazards_log.append({'tree': f"T{i+1}", 'reasons': reasons})
        else:
            color = (0, 200, 0)
            label = f"TREE T{i+1} (SAFE)"

        cv2.rectangle(img,
                      (tree['bbox'][0], tree['bbox'][1]),
                      (tree['bbox'][2], tree['bbox'][3]),
                      color, 2)
        cv2.putText(img, label,
                    (tree['bbox'][0], tree['bbox'][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.line(img, (0, int(horizon_y)), (w, int(horizon_y)),
             (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, "horizon", (5, int(horizon_y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (820, 88), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    status = (f"NIT CALICUT AUDIT  |  POLES: {len(targets)}  |  "
              f"TREES: {len(all_trees)}  |  HAZARDS: {len(hazards_log)}  |  "
              f"focal={focal_px:.0f}px  |  ~{METRES_PER_FRAME:.2f} m/frame")
    cv2.putText(img, status, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return img, {
        'poles'           : len(targets),
        'trees'           : len(all_trees),
        'hazards'         : len(hazards_log),
        'hazards_log'     : hazards_log,
        'pole_distances'  : pole_distances,
        'focal_px'        : focal_px,
        'metres_per_frame': METRES_PER_FRAME,
    }


# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────

model = load_model()

with st.sidebar:
    st.markdown(
        '<h2 style="font-size:1.15rem;margin-bottom:1.2rem;">⚙️ Configuration</h2>',
        unsafe_allow_html=True,
    )

    vfov = st.slider(
        "Camera Vertical FOV (°)",
        min_value=40, max_value=100, value=int(CAMERA_VFOV_DEG), step=1,
        help="Typical phone: 60–75°. Calibrate against a known distance for accuracy."
    )
    pole_prox = st.slider(
        "Pole Proximity Threshold (px)",
        min_value=50, max_value=600, value=POLE_PROXIMITY_THRESHOLD_PX, step=25,
        help="Tree base within this pixel distance of a pole base → NEAR POLE hazard."
    )

    st.divider()

    st.markdown('<h2 style="font-size:1.15rem;margin-bottom:1rem;">📂 Upload Image</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a roadside image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    st.divider()

    # Model info as styled tags
    st.markdown(
        '<div class="meta-tag">🤖 &nbsp;Model: YOLOv9-seg</div>'
        '<div class="meta-tag">📐 &nbsp;Distance: Pinhole camera</div>'
        f'<div class="meta-tag">🚗 &nbsp;{VEHICLE_SPEED_KMH} km/h @ {FRAME_RATE_FPS} FPS</div>'
        f'<div class="meta-tag">📏 &nbsp;≈ {METRES_PER_FRAME:.2f} m per frame</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        '<p style="font-size:0.75rem;color:#7a8299;line-height:1.7;">'
        '👨‍💻 Ankur Kumar Singh<br>'
        '👨‍💻 Digvijay Patel<br>'
        '👨‍💻 Krishna Singh'
        '</p>',
        unsafe_allow_html=True,
    )

# ── MAIN CONTENT ──────────────────────────────────────────────────────────────

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_raw    = cv2.imdecode(file_bytes, 1)

    with st.spinner("Running hazard audit…"):
        annotated, info = run_audit(img_raw.copy(), model, vfov, pole_prox)

    col_img, col_stats = st.columns([3, 1])

    with col_img:
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_container_width=True,
            caption=f"Annotated — {uploaded_file.name}",
        )

    with col_stats:
        # ── Metrics ──
        st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Poles",   info['poles'])
        c2.metric("Trees",   info['trees'])
        c3.metric("Hazards", info['hazards'])

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Hazard detail ──
        st.markdown('<div class="section-label">⚠️ Hazard Detail</div>', unsafe_allow_html=True)
        if info['hazards'] > 0:
            for h_item in info['hazards_log']:
                tag = " + ".join(h_item['reasons'])
                st.markdown(
                    f'<div class="hazard-box">🌳 <b>{h_item["tree"]}</b> &nbsp;—&nbsp; {tag}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="safe-box">✅ No tree hazards detected</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Pole spacing ──
        st.markdown('<div class="section-label">📏 Pole Spacing</div>', unsafe_allow_html=True)
        if info['pole_distances']:
            for label, span, d_near, d_far in info['pole_distances']:
                st.markdown(
                    f'<div class="info-box">'
                    f'<b>{label}</b> &nbsp;→&nbsp; {span:.1f} m span<br>'
                    f'<span style="font-size:0.76rem;opacity:0.7;">'
                    f'near {d_near:.0f} m · far {d_far:.0f} m from cam'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
            st.caption(
                f"~{info['metres_per_frame']:.2f} m/frame · focal {info['focal_px']:.0f} px"
            )
        else:
            st.caption("No consecutive pole pairs detected.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Download ──
        _, buf = cv2.imencode('.jpg', annotated)
        st.download_button(
            label="⬇️ Download Annotated Image",
            data=buf.tobytes(),
            file_name=f"audit_{uploaded_file.name}",
            mime="image/jpeg",
            use_container_width=True,
        )

else:
    # ── Empty state ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 &nbsp;Upload a roadside image in the sidebar to begin the audit.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label" style="max-width:700px;">How it works</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div class="how-card">
            <div class="how-card-icon">🔍</div>
            <div class="how-card-title">Detection</div>
            <div class="how-card-body">YOLOv9-seg model detects poles and trees with instance segmentation masks, resolving class IDs dynamically from model metadata.</div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="how-card">
            <div class="how-card-icon">📐</div>
            <div class="how-card-title">Distance</div>
            <div class="how-card-body">Pinhole camera model: <code style="background:rgba(198,241,53,0.1);color:#c6f135;padding:1px 5px;border-radius:4px;">D = (H_real × f) / H_px</code>. Pole-to-pole span equals far distance minus near distance.</div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div class="how-card">
            <div class="how-card-icon">⚠️</div>
            <div class="how-card-title">Hazard Logic</div>
            <div class="how-card-body">A tree is flagged when it is taller than the nearest pole, its canopy overhangs the road centre, or its base falls within the proximity threshold of a pole.</div>
        </div>
        """, unsafe_allow_html=True)
