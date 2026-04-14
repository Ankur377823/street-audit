"""
Streamlit Roadside Audit App — v5
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
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { font-size: 0.82rem; color: #555; }
.hazard-box {
    background: #fff2f2;
    border-left: 4px solid #e53935;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    color: #b71c1c;
}
.safe-box {
    background: #f2fff4;
    border-left: 4px solid #43a047;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    color: #1b5e20;
}
.info-box {
    background: #f0f4ff;
    border-left: 4px solid #3949ab;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    color: #1a237e;
}
</style>
""", unsafe_allow_html=True)

st.title("🛣️ Automated Roadside Hazard Audit System")
st.caption("Final Year Project | National Institute of Technology, Calicut")

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

MODEL_PATH                  = 'best.pt'
REAL_POLE_HEIGHT_M          = 8.0
VEHICLE_SPEED_KMH           = 35.0
FRAME_RATE_FPS              = 4.0
CAMERA_VFOV_DEG             = 69.0   # Calibrate for your phone
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
    """Resolve pole/tree class indices from model.names dynamically."""
    names    = {v.lower(): k for k, v in model.names.items()}
    pole_cls = names.get('pole', 0)
    tree_cls = names.get('tree', 1)
    return pole_cls, tree_cls


def focal_length_px(image_h, vfov_deg):
    """Vertical focal length from image height and vertical FOV."""
    vfov_rad = np.radians(vfov_deg)
    return (image_h / 2.0) / np.tan(vfov_rad / 2.0)


def robust_mask_extremes(pts):
    """
    Percentile-based top and base — immune to single outlier polygon points.
    Returns top (x,y), base (x,y), and pixel height.
    """
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
    """
    Pinhole camera similar-triangles model:
        D = (H_real × focal_px) / H_px
    """
    if px_h < 5:
        return None
    return (REAL_POLE_HEIGHT_M * focal_px) / px_h


def pole_to_pole_span_m(p_near, p_far, focal_px):
    """
    Along-road span between consecutive poles.
    p_near: larger px_h (closer), p_far: smaller px_h (farther)
    span = distance_to_far - distance_to_near
    """
    d_near = distance_to_pole_m(p_near['px_h'], focal_px)
    d_far  = distance_to_pole_m(p_far['px_h'],  focal_px)
    if d_near is None or d_far is None:
        return None, None, None
    return abs(d_far - d_near), d_near, d_far


# ─── CORE AUDIT FUNCTION ──────────────────────────────────────────────────────

def run_audit(img, model, vfov_deg, pole_prox_px):
    """
    Runs the full v5 audit pipeline on a BGR image.
    Returns annotated BGR image + structured results dict.
    """
    h, w, _  = img.shape
    results  = model(img)[0]
    POLE_CLS, TREE_CLS = get_class_indices(model)

    focal_px  = focal_length_px(h, vfov_deg)
    horizon_y = h * 0.47

    all_poles, all_trees = [], []
    targets              = []   # always defined
    hazards_log          = []   # list of dicts for sidebar display

    # ── DATA EXTRACTION ──────────────────────────────────────────────────────

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

    # ── DOMINANT SIDE FILTER ─────────────────────────────────────────────────

    if all_poles:
        closest_p = max(all_poles, key=lambda x: x['base'][1])
        is_left   = closest_p['base'][0] < w / 2

        if is_left:
            targets = [p for p in all_poles if p['base'][0] < (w / 2 + SIDE_BUFFER_PX)]
        else:
            targets = [p for p in all_poles if p['base'][0] > (w / 2 - SIDE_BUFFER_PX)]

        # Sort nearest-first: largest px_h = closest
        targets.sort(key=lambda x: x['px_h'], reverse=True)

    # ── ANNOTATE POLES & SPAN DISTANCES ──────────────────────────────────────

    pole_distances = []   # list of (label, span_m, d_near, d_far)

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

    # ── TREE HAZARD LOGIC ────────────────────────────────────────────────────

    for i, tree in enumerate(all_trees):
        is_hazard = False
        reasons   = []

        if targets:
            cp = min(targets,
                     key=lambda p: np.linalg.norm(
                         np.array(p['base']) - np.array(tree['base'])
                     ))

            # Height check
            if tree['top'][1] < cp['top'][1]:
                is_hazard = True
                reasons.append("HEIGHT")

            # Near-pole check
            dist_to_pole = np.linalg.norm(
                np.array(cp['base']) - np.array(tree['base'])
            )
            if dist_to_pole < pole_prox_px:
                is_hazard = True
                reasons.append("NEAR POLE")

        # Canopy overhang — depth-gated
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

    # Horizon line
    cv2.line(img, (0, int(horizon_y)), (w, int(horizon_y)),
             (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, "horizon", (5, int(horizon_y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Dashboard overlay on image
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (820, 88), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    status = (f"NIT CALICUT AUDIT  |  POLES: {len(targets)}  |  "
              f"TREES: {len(all_trees)}  |  HAZARDS: {len(hazards_log)}  |  "
              f"focal={focal_px:.0f}px  |  ~{METRES_PER_FRAME:.2f} m/frame")
    cv2.putText(img, status, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return img, {
        'poles'          : len(targets),
        'trees'          : len(all_trees),
        'hazards'        : len(hazards_log),
        'hazards_log'    : hazards_log,
        'pole_distances' : pole_distances,
        'focal_px'       : focal_px,
        'metres_per_frame': METRES_PER_FRAME,
    }


# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────

model = load_model()

with st.sidebar:
    st.header("⚙️ Configuration")

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
    st.header("📂 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a roadside image",
        type=["jpg", "jpeg", "png"],
    )

    st.divider()
    st.markdown("**Model:** YOLOv9-seg")
    st.markdown("**Distance:** Pinhole camera model")
    st.markdown(f"**Speed:** {VEHICLE_SPEED_KMH} km/h @ {FRAME_RATE_FPS} FPS")
    st.markdown(f"**≈ {METRES_PER_FRAME:.2f} m per frame**")
    st.divider()
    st.caption("Developer: Ankur Kumar Singh\nDigvijay Patel\nKrishna Singh")

# ── MAIN CONTENT ──────────────────────────────────────────────────────────────

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_raw    = cv2.imdecode(file_bytes, 1)

    with st.spinner("Running audit..."):
        annotated, info = run_audit(img_raw.copy(), model, vfov, pole_prox)

    # Layout
    col_img, col_stats = st.columns([3, 1])

    with col_img:
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_container_width=True,
            caption=f"Annotated — {uploaded_file.name}",
        )

    with col_stats:
        st.subheader("📊 Audit Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Poles",   info['poles'])
        c2.metric("Trees",   info['trees'])
        c3.metric("Hazards", info['hazards'])

        st.divider()

        # Hazard summary
        if info['hazards'] > 0:
            st.markdown("#### ⚠️ Hazard Detail")
            for h_item in info['hazards_log']:
                tag = " + ".join(h_item['reasons'])
                st.markdown(
                    f'<div class="hazard-box">🌳 {h_item["tree"]} — <b>{tag}</b></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="safe-box">✅ No tree hazards detected</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Pole distances
        if info['pole_distances']:
            st.markdown("#### 📏 Pole Spacing")
            for label, span, d_near, d_far in info['pole_distances']:
                st.markdown(
                    f'<div class="info-box">'
                    f'<b>{label}</b> → {span:.1f} m span<br>'
                    f'<span style="font-size:0.78rem;color:#3949ab">'
                    f'near {d_near:.0f} m · far {d_far:.0f} m from cam'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
            st.caption(
                f"Frame-speed ref: ~{info['metres_per_frame']:.2f} m/frame "
                f"| focal: {info['focal_px']:.0f} px"
            )
        else:
            st.caption("No consecutive pole pairs detected.")

        st.divider()

        # Download annotated image
        _, buf = cv2.imencode('.jpg', annotated)
        st.download_button(
            label="⬇️ Download Annotated Image",
            data=buf.tobytes(),
            file_name=f"audit_{uploaded_file.name}",
            mime="image/jpeg",
            use_container_width=True,
        )

else:
    st.info("👈 Upload a roadside image in the sidebar to begin the audit.")

    st.markdown("---")
    st.markdown("### How it works")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**🔍 Detection**\n\nYOLOv9-seg model detects poles and trees with instance segmentation masks.")
    with col_b:
        st.markdown("**📐 Distance**\n\nPinhole camera model: `D = (H_real × f) / H_px`. Span = far distance − near distance.")
    with col_c:
        st.markdown("**⚠️ Hazard logic**\n\nTree flagged if: taller than nearest pole, canopy over road centre, or within proximity threshold of a pole.")