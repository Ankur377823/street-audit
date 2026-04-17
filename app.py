"""
Streamlit Roadside Audit App — v5 Enhanced UI
NIT Calicut | Automated Roadside Hazard Audit System

Logic: street_audit_v5.py (unchanged)
UI: Redesigned with dark industrial/tech aesthetic
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
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@400;500;600;700&family=Exo+2:wght@300;400;600;800&display=swap');

/* ── GLOBAL RESET & BASE ── */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
}

.stApp {
    background: #080c14;
    background-image:
        radial-gradient(ellipse at 20% 10%, rgba(0, 180, 255, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(255, 100, 0, 0.05) 0%, transparent 50%),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 39px,
            rgba(0, 200, 255, 0.025) 39px,
            rgba(0, 200, 255, 0.025) 40px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 39px,
            rgba(0, 200, 255, 0.025) 39px,
            rgba(0, 200, 255, 0.025) 40px
        );
    min-height: 100vh;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0b1120 !important;
    border-right: 1px solid rgba(0, 200, 255, 0.15) !important;
    box-shadow: 4px 0 32px rgba(0, 180, 255, 0.08);
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    color: #c8ddf0 !important;
}

/* ── MAIN TITLE ── */
h1 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.6rem !important;
    letter-spacing: 0.06em !important;
    background: linear-gradient(90deg, #00c8ff 0%, #ffffff 50%, #ff6b00 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-transform: uppercase !important;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
    line-height: 1.1 !important;
}

/* ── CAPTION / SUBTITLE ── */
.stApp [data-testid="stCaptionContainer"] p,
.element-container .stMarkdown > p:first-child {
    color: #4a7fa5 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* ── HEADINGS ── */
h2, h3 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    color: #00c8ff !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

h4, h5 {
    font-family: 'Rajdhani', sans-serif !important;
    color: #7ab8d4 !important;
    letter-spacing: 0.05em !important;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(0, 200, 255, 0.18) !important;
    margin: 1.2rem 0 !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1a2e 0%, #0f2035 100%) !important;
    border: 1px solid rgba(0, 200, 255, 0.2) !important;
    border-top: 2px solid #00c8ff !important;
    border-radius: 4px !important;
    padding: 14px 16px !important;
    box-shadow: 0 4px 24px rgba(0, 180, 255, 0.1), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    transition: box-shadow 0.3s ease !important;
}

[data-testid="stMetric"]:hover {
    box-shadow: 0 6px 32px rgba(0, 200, 255, 0.22) !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #4a7fa5 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

/* ── HAZARD / SAFE / INFO BOXES ── */
.hazard-box {
    background: linear-gradient(135deg, #1a0808 0%, #200d0d 100%);
    border-left: 3px solid #ff3d2e;
    border-top: 1px solid rgba(255, 61, 46, 0.25);
    border-right: 1px solid rgba(255, 61, 46, 0.1);
    border-bottom: 1px solid rgba(255, 61, 46, 0.1);
    border-radius: 2px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.74rem;
    color: #ff8070;
    letter-spacing: 0.04em;
    box-shadow: 0 2px 16px rgba(255, 50, 30, 0.15), inset 0 0 20px rgba(255, 50, 30, 0.04);
    position: relative;
    overflow: hidden;
}

.hazard-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, #ff3d2e, transparent);
}

.safe-box {
    background: linear-gradient(135deg, #071510 0%, #091a12 100%);
    border-left: 3px solid #00e676;
    border-top: 1px solid rgba(0, 230, 118, 0.2);
    border-right: 1px solid rgba(0, 230, 118, 0.08);
    border-bottom: 1px solid rgba(0, 230, 118, 0.08);
    border-radius: 2px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.74rem;
    color: #4fffaa;
    letter-spacing: 0.04em;
    box-shadow: 0 2px 16px rgba(0, 230, 100, 0.12), inset 0 0 20px rgba(0, 230, 100, 0.03);
}

.info-box {
    background: linear-gradient(135deg, #071525 0%, #091d30 100%);
    border-left: 3px solid #00c8ff;
    border-top: 1px solid rgba(0, 200, 255, 0.2);
    border-right: 1px solid rgba(0, 200, 255, 0.08);
    border-bottom: 1px solid rgba(0, 200, 255, 0.08);
    border-radius: 2px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.74rem;
    color: #5eb8d8;
    letter-spacing: 0.04em;
    box-shadow: 0 2px 16px rgba(0, 180, 255, 0.1), inset 0 0 20px rgba(0, 180, 255, 0.03);
}

/* ── SPINNER ── */
[data-testid="stSpinner"] {
    color: #00c8ff !important;
}

/* ── IMAGE CONTAINER ── */
[data-testid="stImage"] {
    border: 1px solid rgba(0, 200, 255, 0.15);
    border-radius: 4px;
    box-shadow: 0 8px 48px rgba(0, 140, 255, 0.12);
    overflow: hidden;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, #0a1525 0%, #0d1e35 100%) !important;
    border: 1px dashed rgba(0, 200, 255, 0.3) !important;
    border-radius: 6px !important;
    transition: border-color 0.3s !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 200, 255, 0.6) !important;
}

[data-testid="stFileUploader"] * {
    color: #7ab8d4 !important;
}

/* ── SLIDERS ── */
[data-testid="stSlider"] .rc-slider-track {
    background: linear-gradient(90deg, #0080cc, #00c8ff) !important;
}

[data-testid="stSlider"] .rc-slider-handle {
    border-color: #00c8ff !important;
    background: #00c8ff !important;
    box-shadow: 0 0 8px rgba(0, 200, 255, 0.6) !important;
}

.stSlider label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #4a7fa5 !important;
}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #003355 0%, #004470 100%) !important;
    color: #00c8ff !important;
    border: 1px solid rgba(0, 200, 255, 0.4) !important;
    border-radius: 3px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 12px rgba(0, 180, 255, 0.15) !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #00446e 0%, #005d96 100%) !important;
    border-color: #00c8ff !important;
    box-shadow: 0 4px 20px rgba(0, 200, 255, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── INFO MESSAGE ── */
[data-testid="stAlert"] {
    background: linear-gradient(135deg, #07101e 0%, #0a1828 100%) !important;
    border: 1px solid rgba(0, 200, 255, 0.2) !important;
    border-left: 3px solid #00c8ff !important;
    border-radius: 3px !important;
    color: #7ab8d4 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── MARKDOWN IN MAIN ── */
.stMarkdown p {
    color: #8bafc8 !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
}

.stMarkdown strong {
    color: #00c8ff !important;
    font-weight: 600 !important;
}

/* ── CAPTION (small text) ── */
.stCaption {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.66rem !important;
    color: #3d607a !important;
    letter-spacing: 0.06em !important;
}

/* ── SIDEBAR CAPTION ── */
[data-testid="stSidebar"] .stCaption {
    color: #2d4a60 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.08em !important;
    white-space: pre-line;
}

/* ── HOW IT WORKS CARDS ── */
.how-card {
    background: linear-gradient(145deg, #0c1828 0%, #0f2035 100%);
    border: 1px solid rgba(0, 200, 255, 0.12);
    border-radius: 6px;
    padding: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    height: 100%;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080c14; }
::-webkit-scrollbar-thumb { background: #1a3a54; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00c8ff; }

/* ── BADGE DECORATIONS ── */
.badge-accent {
    display: inline-block;
    background: rgba(0, 200, 255, 0.12);
    border: 1px solid rgba(0, 200, 255, 0.3);
    color: #00c8ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 2px;
    margin-right: 6px;
}

/* ── SIDEBAR LABEL TAGS ── */
.sidebar-tag {
    display: inline-block;
    background: rgba(0, 200, 255, 0.08);
    border: 1px solid rgba(0, 200, 255, 0.18);
    color: #5eb8d8;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 2px;
    margin-bottom: 4px;
    display: block;
    width: fit-content;
}

/* ── SECTION HEADER LINE ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}

.section-header-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,200,255,0.4), transparent);
}

</style>
""", unsafe_allow_html=True)

# ── HERO HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 0.2rem;">
    <span style="font-family:'Space Mono',monospace; font-size:0.68rem; letter-spacing:0.22em;
                 color:#00c8ff; text-transform:uppercase; opacity:0.7;">
        ◈ NIT Calicut · Final Year Project
    </span>
</div>
""", unsafe_allow_html=True)

st.title("🛣️ Roadside Hazard Audit System")

st.markdown("""
<div style="font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:0.18em;
            color:#2d6e8a; text-transform:uppercase; margin-top:-6px; margin-bottom:20px;
            border-bottom: 1px solid rgba(0,200,255,0.1); padding-bottom:16px;">
    YOLOv9-seg · Pinhole Depth Model · Instance Segmentation · Real-time Hazard Classification
</div>
""", unsafe_allow_html=True)

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
    # Logo / brand block
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 20px;">
        <div style="font-family:'Rajdhani',sans-serif; font-size:1.5rem; font-weight:700;
                    color:#00c8ff; letter-spacing:0.12em; text-transform:uppercase;
                    line-height:1.1;">
            NITC
        </div>
        <div style="font-family:'Space Mono',monospace; font-size:0.58rem;
                    color:#2d5a72; letter-spacing:0.18em; text-transform:uppercase;
                    margin-top:2px;">
            Hazard Audit System
        </div>
        <div style="height:1px; background:linear-gradient(90deg, transparent, rgba(0,200,255,0.4), transparent);
                    margin: 14px 0 0;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sidebar-tag">⚙ Parameters</span>', unsafe_allow_html=True)

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

    st.markdown('<div style="height:1px; background:linear-gradient(90deg, transparent, rgba(0,200,255,0.2), transparent); margin:18px 0;"></div>', unsafe_allow_html=True)

    st.markdown('<span class="sidebar-tag">📂 Input</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload roadside image",
        type=["jpg", "jpeg", "png"],
    )

    st.markdown('<div style="height:1px; background:linear-gradient(90deg, transparent, rgba(0,200,255,0.2), transparent); margin:18px 0;"></div>', unsafe_allow_html=True)

    # System info panel
    st.markdown("""
    <div style="background:rgba(0,200,255,0.04); border:1px solid rgba(0,200,255,0.12);
                border-radius:4px; padding:12px; font-family:'Space Mono',monospace;
                font-size:0.65rem; color:#3d7a96; line-height:2;">
        <div><span style="color:#00c8ff;">MODEL</span> &nbsp; YOLOv9-seg</div>
        <div><span style="color:#00c8ff;">DEPTH </span> &nbsp; Pinhole camera</div>
        <div><span style="color:#00c8ff;">SPEED </span> &nbsp; {speed} km/h @ {fps} FPS</div>
        <div><span style="color:#00c8ff;">STEP  </span> &nbsp; ~{mpf:.2f} m / frame</div>
    </div>
    """.format(speed=VEHICLE_SPEED_KMH, fps=FRAME_RATE_FPS, mpf=METRES_PER_FRAME),
    unsafe_allow_html=True)

    st.markdown('<div style="height:1px; background:linear-gradient(90deg, transparent, rgba(0,200,255,0.2), transparent); margin:18px 0;"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:#1e4058;
                letter-spacing:0.08em; text-align:center; line-height:2;">
        Ankur Kumar Singh<br>Digvijay Patel<br>Krishna Singh<br>
        <span style="color:#153040;">NIT Calicut · 2025</span>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN CONTENT ──────────────────────────────────────────────────────────────

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_raw    = cv2.imdecode(file_bytes, 1)

    with st.spinner("Running segmentation & hazard audit..."):
        annotated, info = run_audit(img_raw.copy(), model, vfov, pole_prox)

    col_img, col_stats = st.columns([3, 1])

    with col_img:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace; font-size:0.64rem;
                    color:#2d6e8a; letter-spacing:0.14em; text-transform:uppercase;
                    margin-bottom:6px;">
            ◈ Annotated Output
        </div>
        """, unsafe_allow_html=True)
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_container_width=True,
            caption=f"▸ {uploaded_file.name}",
        )

    with col_stats:
        st.markdown("""
        <div style="font-family:'Rajdhani',sans-serif; font-size:1.1rem; font-weight:600;
                    color:#00c8ff; letter-spacing:0.12em; text-transform:uppercase;
                    border-bottom:1px solid rgba(0,200,255,0.15); padding-bottom:8px;
                    margin-bottom:14px;">
            ◈ Audit Results
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Poles",   info['poles'])
        c2.metric("Trees",   info['trees'])
        c3.metric("Hazards", info['hazards'])

        st.markdown('<div style="height:1px; background:linear-gradient(90deg, transparent, rgba(0,200,255,0.2), transparent); margin:14px 0;"></div>', unsafe_allow_html=True)

        if info['hazards'] > 0:
            st.markdown("""
            <div style="font-family:'Space Mono',monospace; font-size:0.64rem;
                        color:#ff3d2e; letter-spacing:0.14em; text-transform:uppercase;
                        margin-bottom:10px;">
                ⚠ Hazard Detail
            </div>
            """, unsafe_allow_html=True)
            for h_item in info['hazards_log']:
                tag = " + ".join(h_item['reasons'])
                st.markdown(
                    f'<div class="hazard-box">🌳 <b>{h_item["tree"]}</b> &nbsp;—&nbsp; {tag}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="safe-box">✅ &nbsp;No tree hazards detected</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:1px; background:linear-gradient(90deg, transparent, rgba(0,200,255,0.2), transparent); margin:14px 0;"></div>', unsafe_allow_html=True)

        if info['pole_distances']:
            st.markdown("""
            <div style="font-family:'Space Mono',monospace; font-size:0.64rem;
                        color:#00c8ff; letter-spacing:0.14em; text-transform:uppercase;
                        margin-bottom:10px;">
                ◈ Pole Spacing
            </div>
            """, unsafe_allow_html=True)
            for label, span, d_near, d_far in info['pole_distances']:
                st.markdown(
                    f'<div class="info-box">'
                    f'<b>{label}</b> → {span:.1f} m span<br>'
                    f'<span style="font-size:0.68rem; color:#2d6e8a;">'
                    f'near {d_near:.0f} m · far {d_far:.0f} m from cam'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div style="font-family:\'Space Mono\',monospace; font-size:0.6rem; '
                f'color:#1e4a5e; margin-top:6px;">'
                f'~{info["metres_per_frame"]:.2f} m/frame · focal {info["focal_px"]:.0f} px'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-family:\'Space Mono\',monospace; font-size:0.65rem; '
                'color:#1e4a5e;">No consecutive pole pairs detected.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:1px; background:linear-gradient(90deg, transparent, rgba(0,200,255,0.2), transparent); margin:14px 0;"></div>', unsafe_allow_html=True)

        _, buf = cv2.imencode('.jpg', annotated)
        st.download_button(
            label="⬇ Download Annotated Image",
            data=buf.tobytes(),
            file_name=f"audit_{uploaded_file.name}",
            mime="image/jpeg",
            use_container_width=True,
        )

else:
    # ── LANDING / EMPTY STATE ─────────────────────────────────────────────────

    st.markdown("""
    <div style="background:linear-gradient(135deg, #0a1525 0%, #0d1e35 100%);
                border:1px solid rgba(0,200,255,0.15); border-radius:6px;
                padding:28px 32px; margin-bottom:32px;
                box-shadow:0 4px 40px rgba(0,140,255,0.08);">
        <div style="font-family:'Space Mono',monospace; font-size:0.7rem;
                    color:#00c8ff; letter-spacing:0.18em; text-transform:uppercase;
                    margin-bottom:10px;">◈ System Ready</div>
        <div style="font-family:'Exo 2',sans-serif; font-size:1.05rem; color:#7ab8d4;
                    line-height:1.6;">
            Upload a roadside photograph using the sidebar panel to begin the audit.
            The system will detect <span style="color:#00c8ff;">poles</span> and
            <span style="color:#4fffaa;">trees</span>, compute distances via the
            pinhole camera model, and flag any
            <span style="color:#ff6060;">hazardous conditions</span>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif; font-size:1rem; font-weight:600;
                color:#00c8ff; letter-spacing:0.15em; text-transform:uppercase;
                border-bottom:1px solid rgba(0,200,255,0.12); padding-bottom:10px;
                margin-bottom:20px;">
        ◈ How It Works
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    cards = [
        ("🔍", "Detection",
         "YOLOv9-seg model segments poles and trees with instance-level precision masks, "
         "enabling sub-pixel boundary extraction."),
        ("📐", "Distance",
         "Pinhole camera model: <code style='color:#00c8ff;'>D = (H_real × f) / H_px</code>. "
         "Pole span = far distance − near distance along the road axis."),
        ("⚠️", "Hazard Logic",
         "Tree is flagged <b style='color:#ff6060;'>HAZARD</b> if it's taller than nearest pole, "
         "has a canopy over road centre, or is within the proximity threshold of a pole."),
    ]

    for col, (icon, title, body) in zip([col_a, col_b, col_c], cards):
        with col:
            st.markdown(f"""
            <div class="how-card">
                <div style="font-size:1.8rem; margin-bottom:10px;">{icon}</div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:1rem; font-weight:600;
                            color:#00c8ff; letter-spacing:0.1em; text-transform:uppercase;
                            margin-bottom:8px;">{title}</div>
                <div style="font-family:'Exo 2',sans-serif; font-size:0.82rem; color:#5e8fa8;
                            line-height:1.65;">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:28px; display:flex; gap:12px; flex-wrap:wrap;">
        <span class="badge-accent">Segmentation</span>
        <span class="badge-accent">Pinhole Model</span>
        <span class="badge-accent">Hazard Classification</span>
        <span class="badge-accent">Depth Estimation</span>
        <span class="badge-accent">Canopy Detection</span>
        <span class="badge-accent">Pole Spacing</span>
    </div>
    """, unsafe_allow_html=True)
