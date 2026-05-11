"""
app.py - Migraine Risk Prediction (CHARLS Lite Model)
Streamlit Web Application — Revised 2026

KEY FIX: Continuous and ordinal features (age, cesd10, edu, memeory, srh)
must be z-score standardized before being passed to the model, because the
model was trained on StandardScaler-transformed data. Scaler parameters are
reconstructed from feat_meta z-range + known CHARLS original ranges.

edu NOTE: Although edu is a categorical variable (3 levels: 0=illiterate,
1=primary/middle, 2=high school+), the model pipeline treated it as an
ordinal integer and applied StandardScaler to it. The feat_meta confirms this:
nuniq=5 (z-scored artifacts), z_min≈-0.993 and z_max≈1.917 map exactly to
raw values 0 and 2 under mean≈0.683, std≈0.687. So the UI presents edu as
a 3-option select_slider, but the value is z-scored before prediction.
"""

import streamlit as st
import numpy as np
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Migraine Risk Predictor | CHARLS 2026",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* -------- top banner -------- */
.banner {
    background: linear-gradient(135deg, #0f2944 0%, #1a4a7a 60%, #1e5799 100%);
    color: #ffffff;
    padding: 28px 36px 22px 36px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(30, 87, 153, 0.25);
}
.banner h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.0rem;
    font-weight: 400;
    margin: 0 0 6px 0;
    letter-spacing: 0.3px;
    color: #ffffff;
}
.banner p {
    font-size: 0.92rem;
    margin: 0;
    color: #b8d4f0;
    line-height: 1.5;
}

/* -------- section header -------- */
.section-header {
    background: #f0f4fa;
    border-left: 5px solid #1e5799;
    padding: 10px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f2944;
    margin: 20px 0 14px 0;
    letter-spacing: 0.2px;
}

/* -------- risk boxes -------- */
.risk-high {
    background: #fde8e8;
    border-left: 6px solid #c0392b;
    padding: 18px 22px;
    border-radius: 8px;
    margin: 14px 0;
    font-size: 1.02rem;
    line-height: 1.6;
}
.risk-mid {
    background: #fef9e7;
    border-left: 6px solid #e67e22;
    padding: 18px 22px;
    border-radius: 8px;
    margin: 14px 0;
    font-size: 1.02rem;
    line-height: 1.6;
}
.risk-low {
    background: #eafaf1;
    border-left: 6px solid #27ae60;
    padding: 18px 22px;
    border-radius: 8px;
    margin: 14px 0;
    font-size: 1.02rem;
    line-height: 1.6;
}

/* -------- feature summary tags -------- */
.feat-tag {
    display: inline-block;
    background: #eef2fb;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.87rem;
    margin: 3px;
    color: #1a3a5c;
    border: 1px solid #d0ddf2;
}

/* -------- global font -------- */
label, .stRadio label, .stSelectbox label,
.stNumberInput label, .stSlider label {
    font-size: 1.02rem !important;
    font-weight: 600 !important;
    color: #0f2944 !important;
}
.stRadio > div > label {
    font-size: 1.0rem !important;
    padding: 8px 18px !important;
}
input[type="number"] {
    font-size: 1.1rem !important;
    padding: 8px 10px !important;
    height: 44px !important;
}
.stSelectbox > div > div {
    font-size: 1.05rem !important;
    min-height: 44px !important;
}
.stButton > button[kind="primary"] {
    font-size: 1.15rem !important;
    padding: 14px 0 !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #1a4a7a, #1e5799) !important;
    border-color: #1e5799 !important;
    border-radius: 8px !important;
    letter-spacing: 0.3px !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0f2944, #1a4a7a) !important;
    border-color: #0f2944 !important;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = "lite_model_migraine.pkl"

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

payload = load_model(MODEL_PATH)

# ── Scaler reconstruction ─────────────────────────────────────────────────────
#
# CRITICAL: The model was trained on z-scored continuous features.
# The feat_meta stores min/max in z-score space (mean≈0, std≈1).
# We reconstruct original-scale (mean_orig, std_orig) using:
#   raw_std  = (raw_max - raw_min) / (z_max - z_min)
#   raw_mean = raw_min - z_min * raw_std
#
# Original CHARLS ranges (used during training):
ORIG_RANGES = {
    "age":     {"min": 45,  "max": 90},   # CHARLS eligibility
    "cesd10":  {"min": 0,   "max": 30},   # CESD-10 total
    "edu":     {"min": 0,   "max": 2},    # categorical but ordinal-encoded (0/1/2) then z-scored by pipeline
    "memeory": {"min": 0,   "max": 10},   # Cognitive test score (z-range confirms 0-10 bounds)
    "srh":     {"min": 1,   "max": 5},    # Self-rated health 1-5
}


def build_scaler_params(feat_meta):
    """Return dict of {feat: (mean_orig, std_orig)} for continuous features."""
    scalers = {}
    for feat, meta in feat_meta.items():
        if meta["type"] == "binary":
            continue
        if feat not in ORIG_RANGES:
            continue
        z_min = float(meta["min"])
        z_max = float(meta["max"])
        raw_min = ORIG_RANGES[feat]["min"]
        raw_max = ORIG_RANGES[feat]["max"]
        raw_std  = (raw_max - raw_min) / (z_max - z_min)
        raw_mean = raw_min - z_min * raw_std
        scalers[feat] = (raw_mean, raw_std)
    return scalers


# ── Publishable feature metadata ──────────────────────────────────────────────
PUBLISHED_META = {
    "age": {
        "label": "Age (years)",
        "type": "continuous", "min": 45, "max": 90, "mean": 60, "nuniq": 46,
    },
    "gender": {
        "label": "Gender",
        "type": "binary_sex", "min": 0, "max": 1, "mean": 0, "nuniq": 2,
    },
    "edu": {
        "label": "Educational Attainment",
        "type": "edu_cat", "min": 0, "max": 2, "mean": 1, "nuniq": 3,
    },
    "cesd10": {
        "label": "Depressive Symptoms (CESD-10, 0–30)",
        "type": "continuous", "min": 0, "max": 30, "mean": 8, "nuniq": 31,
    },
    "srh": {
        "label": "Self-Rated Health",
        "type": "srh_cat", "min": 1, "max": 5, "mean": 3, "nuniq": 5,
    },
    "arthre": {
        "label": "Arthritis (physician-diagnosed)",
        "type": "binary", "min": 0, "max": 1, "mean": 0, "nuniq": 2,
    },
    "chronic": {
        "label": "Chronic Disease (any)",
        "type": "binary", "min": 0, "max": 1, "mean": 0, "nuniq": 2,
    },
    "digeste": {
        "label": "Digestive Disease",
        "type": "binary", "min": 0, "max": 1, "mean": 0, "nuniq": 2,
    },
    "memeory": {
        "label": "Memory / Cognitive Score",
        "type": "continuous", "min": 0, "max": 10, "mean": 5, "nuniq": 11,
    },
}

DEMO_FEATURES = ["age", "cesd10", "srh", "arthre", "chronic", "digeste", "edu", "memeory", "gender"]

if payload is None:
    st.warning(
        "Model file `lite_model_migraine.pkl` not found. "
        "Running in **demo mode** — predictions are illustrative only."
    )
    features_list  = DEMO_FEATURES
    feat_meta_app  = {k: PUBLISHED_META[k] for k in features_list}
    threshold      = 0.25
    use_fallback   = True
    scaler_params  = {}
else:
    raw_meta      = payload.get("feat_meta", {})
    features_list = payload["features"]
    feat_meta_app = {}
    for f in features_list:
        base = dict(raw_meta.get(f, {"type": "binary", "min": 0, "max": 1, "mean": 0, "nuniq": 2}))
        if f in PUBLISHED_META:
            base["label"] = PUBLISHED_META[f]["label"]
            base["type"]  = PUBLISHED_META[f]["type"]
            base["min"]   = PUBLISHED_META[f]["min"]
            base["max"]   = PUBLISHED_META[f]["max"]
            base["nuniq"] = PUBLISHED_META[f]["nuniq"]
            base["mean"]  = PUBLISHED_META[f]["mean"]
        feat_meta_app[f] = base
    threshold    = payload["threshold"]
    use_fallback = False
    scaler_params = build_scaler_params(raw_meta)

# ── Sidebar — model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Information")
    if payload is not None:
        m = payload["metrics"]
        st.metric("AUC", "{:.3f}".format(m["AUC"]),
                  help="95% CI: {:.3f}–{:.3f}".format(m["CI_lo"], m["CI_hi"]))
        st.metric("Sensitivity", "{:.3f}".format(m["Sensitivity"]))
        st.metric("Specificity", "{:.3f}".format(m["Specificity"]))
        st.metric("Youden J",    "{:.3f}".format(m["Youden_J"]))
        st.metric("Optimal Threshold", "{:.3f}".format(m["Threshold"]))
        st.caption("Features: {} / {} total".format(
            len(features_list), payload.get("n_features_full", "?")))
        st.caption("Algorithm: LightGBM + isotonic calibration")
    else:
        st.info("Upload `lite_model_migraine.pkl` to activate the full model.")

    st.markdown("---")
    st.markdown(
        "**Data:** CHARLS (China Health and Retirement Longitudinal Study)\n\n"
        "**Outcome:** New-onset migraine at 4-year follow-up\n\n"
        "**Features:** SHAP-consensus across LR, RF, XGBoost, LightGBM, DNN"
    )

# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <h1>🧠 Migraine Risk Prediction</h1>
  <p>
    Based on the <strong>China Health and Retirement Longitudinal Study (CHARLS)</strong> —
    A machine-learning model using SHAP-consensus features across 5 algorithms.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Instructions ──────────────────────────────────────────────────────────────
st.markdown("""<div class="section-header">Instructions</div>""", unsafe_allow_html=True)
st.markdown(
    "Please complete all fields below to calculate your estimated migraine risk. "
    "This tool uses **{}** key predictors identified by SHAP-consensus analysis.".format(
        len(features_list))
)

# ── Widget builder ────────────────────────────────────────────────────────────
def make_widget(feat_name, meta, col):
    label  = meta.get("label", feat_name.capitalize())
    ftype  = meta.get("type",  "binary")
    f_min  = meta.get("min",  0)
    f_max  = meta.get("max",  1)
    f_mean = meta.get("mean", 0)

    with col:
        # ── Sex / Gender ──────────────────────────────────────────────────────
        if ftype == "binary_sex":
            val = st.radio(label, options=[0, 1],
                           format_func=lambda x: "Male" if x == 0 else "Female",
                           horizontal=True, key=feat_name)
            return val

        # ── Rural/Urban ───────────────────────────────────────────────────────
        if ftype == "binary_rural":
            val = st.radio(label, options=[0, 1],
                           format_func=lambda x: "Urban" if x == 0 else "Rural",
                           horizontal=True, key=feat_name)
            return val

        # ── Education (3-level categorical) ──────────────────────────────────
        if ftype == "edu_cat":
            edu_map = {
                0: "No formal education / Illiterate",
                1: "Primary or middle school",
                2: "High school or above",
            }
            val = st.select_slider(
                label,
                options=[0, 1, 2],
                value=int(round(float(f_mean))),
                format_func=lambda x: edu_map[x],
                key=feat_name,
            )
            return val

        # ── Self-rated health (5-level ordinal) ───────────────────────────────
        if ftype == "srh_cat" or "srh" in feat_name.lower():
            srh_map = {
                1: "1 – Excellent",
                2: "2 – Very Good",
                3: "3 – Good",
                4: "4 – Fair",
                5: "5 – Poor",
            }
            default = max(1, min(5, int(round(float(f_mean)))))
            val = st.select_slider(
                label,
                options=[1, 2, 3, 4, 5],
                value=default,
                format_func=lambda x: srh_map[x],
                key=feat_name,
            )
            return val

        # ── Standard binary (Yes / No) ────────────────────────────────────────
        if ftype == "binary":
            val = st.radio(
                label, options=[0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True, key=feat_name,
            )
            return val

        # ── CESD-10: always integer 0–30 ──────────────────────────────────────
        if "cesd" in feat_name.lower():
            val = st.number_input(
                label,
                min_value=0, max_value=30,
                value=int(float(f_mean)), step=1,
                help=(
                    "Sum of 10 CESD items (each 0–3: "
                    "0=Rarely, 1=Some, 2=Occasionally, 3=Most/All). "
                    "Range: 0–30. Higher = more depressive symptoms."
                ),
                key=feat_name,
            )
            return val

        # ── Age ───────────────────────────────────────────────────────────────
        if "age" in feat_name.lower():
            val = st.number_input(
                label,
                min_value=45, max_value=90,
                value=int(float(f_mean)), step=1,
                help="CHARLS eligibility: participants aged 45 and above.",
                key=feat_name,
            )
            return val

        # ── Memory / cognitive ────────────────────────────────────────────────
        if "mem" in feat_name.lower() or "cogn" in feat_name.lower():
            val = st.number_input(
                label,
                min_value=int(f_min), max_value=int(f_max),
                value=int(float(f_mean)), step=1,
                help="Cognitive function test score. Higher score = better memory.",
                key=feat_name,
            )
            return val

        # ── Generic continuous ────────────────────────────────────────────────
        safe_min  = float(f_min)
        safe_max  = float(f_max)
        safe_mean = max(safe_min, min(safe_max, float(f_mean)))
        val = st.number_input(
            label,
            min_value=safe_min, max_value=safe_max,
            value=round(safe_mean, 1), step=0.1, format="%.1f",
            key=feat_name,
        )
        return val


# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("""<div class="section-header">Please fill in the information below</div>""",
            unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
user_inputs = {}

for i, feat in enumerate(features_list):
    meta = feat_meta_app.get(feat, PUBLISHED_META.get(feat, {
        "type": "binary", "label": feat.capitalize(),
        "min": 0, "max": 1, "mean": 0, "nuniq": 2,
    }))
    col = col1 if i % 2 == 0 else col2
    user_inputs[feat] = make_widget(feat, meta, col)

# ── Prediction ────────────────────────────────────────────────────────────────
st.divider()
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run_pred = st.button(
        "Calculate My Migraine Risk",
        use_container_width=True,
        type="primary",
    )

if run_pred:
    # ── Build feature array with proper z-score scaling ───────────────────────
    # Binary features (0/1) are passed as-is.
    # All other features — including edu which is CATEGORICAL but was
    # ordinal-encoded (0/1/2) and StandardScaler'd during training — must be
    # z-scored before being passed to the model.
    feat_values = []
    for f in features_list:
        raw_val = float(user_inputs[f])
        meta    = feat_meta_app.get(f, {})
        ftype   = meta.get("type", "binary")

        if ftype in ("binary", "binary_sex", "binary_rural"):
            # Binary: pass raw 0/1 directly — no scaling
            feat_values.append(raw_val)
        elif f in scaler_params:
            # Continuous OR ordinal-encoded categorical (e.g. edu):
            # z-score using reconstructed scaler params
            mean_orig, std_orig = scaler_params[f]
            z = (raw_val - mean_orig) / std_orig
            feat_values.append(z)
        else:
            feat_values.append(raw_val)

    feat_arr = np.array([feat_values], dtype=np.float32)

    if not use_fallback and payload is not None:
        prob = float(payload["model"].predict_proba(feat_arr)[0, 1])
        model_note = (
            "LightGBM Lite ({} features, AUC = {:.3f}, 95% CI: {:.3f}–{:.3f})".format(
                len(features_list),
                payload["metrics"]["AUC"],
                payload["metrics"]["CI_lo"],
                payload["metrics"]["CI_hi"],
            )
        )
    else:
        mean_input = feat_arr[0].mean()
        prob = float(1 / (1 + np.exp(-(-2.5 + mean_input * 0.12))))
        prob = max(0.04, min(0.95, prob))
        model_note = "Demo mode — illustrative estimate only"

    pct       = prob * 100
    high_risk = prob >= threshold
    bar_color = "#c0392b" if high_risk else ("#e67e22" if prob >= threshold * 0.7 else "#27ae60")

    st.markdown("""<div class="section-header">Your Migraine Risk Estimate</div>""",
                unsafe_allow_html=True)

    # Progress bar + percentage
    bar_pct = min(pct, 100)
    st.markdown("""
    <div style="background:#dde4ee; border-radius:12px; height:30px; margin:16px 0 6px 0;
                overflow:hidden; box-shadow:inset 0 1px 3px rgba(0,0,0,0.1);">
      <div style="width:{pct:.1f}%; background:{color}; height:100%;
                  border-radius:12px; transition: width 0.8s ease;
                  box-shadow:0 2px 6px rgba(0,0,0,0.2);"></div>
    </div>
    <p style="text-align:center; font-size:3.2rem; font-weight:800;
              color:{color}; margin:10px 0 2px 0; font-family:'DM Serif Display',serif;">{pct:.1f}%</p>
    <p style="text-align:center; color:#555; font-size:1.0rem; margin-bottom:12px;">
      Estimated probability of new-onset migraine
    </p>
    """.format(pct=bar_pct, color=bar_color), unsafe_allow_html=True)

    # Risk interpretation
    if high_risk:
        st.markdown("""<div class="risk-high">
        <b>⚠ High Risk</b> — Estimated probability {:.1f}% exceeds the model threshold ({:.1f}%).
        We recommend consulting a neurologist, particularly if you experience recurrent headaches,
        visual disturbances, photophobia, or phonophobia.
        </div>""".format(pct, threshold * 100), unsafe_allow_html=True)
    elif prob >= threshold * 0.7:
        st.markdown("""<div class="risk-mid">
        <b>⚡ Moderate Risk</b> — Estimated probability {:.1f}%.
        Consider monitoring headache frequency, identifying personal triggers,
        and discussing preventive strategies with your physician.
        </div>""".format(pct), unsafe_allow_html=True)
    else:
        st.markdown("""<div class="risk-low">
        <b>✓ Low Risk</b> — Estimated probability {:.1f}% is below the threshold.
        Maintain a healthy lifestyle and continue regular check-ups.
        </div>""".format(pct), unsafe_allow_html=True)

    # Input summary
    st.markdown("#### Summary of Your Inputs")
    summary_cols = st.columns(3)
    for i, feat in enumerate(features_list):
        meta  = feat_meta_app.get(feat, {})
        label = str(meta.get("label", feat.capitalize()))
        short_label = label.split("(")[0].strip()
        val   = user_inputs[feat]
        ftype = meta.get("type", "binary")
        if ftype == "binary":
            val_str = "Yes" if val else "No"
        elif ftype == "binary_sex":
            val_str = "Male" if val == 0 else "Female"
        elif ftype == "binary_rural":
            val_str = "Urban" if val == 0 else "Rural"
        elif ftype == "edu_cat":
            val_str = ["Illiterate/None", "Primary/Middle", "High school+"][val]
        elif ftype in ("srh_cat",) or "srh" in feat.lower():
            srh_lbl = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
            val_str = "{} – {}".format(val, srh_lbl.get(val, val))
        else:
            val_str = str(val)
        summary_cols[i % 3].markdown(
            '<span class="feat-tag"><b>{}</b>: {}</span>'.format(short_label, val_str),
            unsafe_allow_html=True,
        )

    st.caption("Model: {}".format(model_note))

    # Model performance metrics
    if payload is not None:
        m = payload["metrics"]
        st.divider()
        st.markdown("#### Model Performance (Held-Out Test Set)")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("AUC (95% CI)", "{:.3f}".format(m["AUC"]),
                      help="{:.3f}–{:.3f}".format(m["CI_lo"], m["CI_hi"]))
        with c2:
            st.metric("Sensitivity", "{:.3f}".format(m["Sensitivity"]))
        with c3:
            st.metric("Specificity", "{:.3f}".format(m["Specificity"]))
        with c4:
            st.metric("Youden J",    "{:.3f}".format(m["Youden_J"]))
        with c5:
            st.metric("Threshold",   "{:.3f}".format(m["Threshold"]))

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<small style="color:#888; line-height:1.6;">
<b>Disclaimer:</b> This tool is intended for <b>research and educational purposes only</b>.
It does not constitute medical advice, diagnosis, or treatment.
Always consult a qualified healthcare professional for clinical decisions.<br><br>
<b>Reference:</b> Migraine Onset Prediction — CHARLS — 2026 —
LightGBM Lite Model — SHAP-consensus features — Isotonic calibration
</small>
""", unsafe_allow_html=True)
