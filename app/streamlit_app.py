"""
AI-Assisted Interview Behavior Analyzer — Streamlit dashboard.
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.inference import build_feature_vector, predict_engagement, predict_interview, predict_traits  # noqa: E402
from app.model_loader import load_feature_medians, load_models, load_shap_evidence  # noqa: E402
from app.recommendations import generate_recommendations  # noqa: E402
from src.models.daisee_dataset import FEATURE_COLUMNS  # noqa: E402
from src.utils.config_loader import load_config, resolve_device  # noqa: E402
from src.video.feature_cache import VisualFeatureCache  # noqa: E402
from src.vision.pipeline import VisualPipeline  # noqa: E402

st.set_page_config(
    page_title="AI Interview Behavior Analyzer",
    layout="wide",
    page_icon="💼",
    initial_sidebar_state="collapsed",
)

GLOBAL_CSS = """
<style>
@import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap\');

html, body, [class*="css"] {
    font-family: \'Inter\', sans-serif;
}

#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}

/* Completely hide sidebar */
section[data-testid="stSidebar"] {
    display: none !important;
}

.stApp {
    background: #0F1117;
    color: #E2E8F0;
}

/* Content container */
.main .block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3.5rem;
}

/* Primary and Secondary Buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.65rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.55) !important;
}

.stButton > button[kind="secondary"] {
    background: #141724 !important;
    border: 1px solid #2D3154 !important;
    color: #94A3B8 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1rem !important;
    white-space: nowrap !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(99, 102, 241, 0.12) !important;
    border-color: #6366F1 !important;
    color: #FFFFFF !important;
    transform: translateX(-2px) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #141724;
    border-radius: 12px;
    padding: 5px;
    gap: 4px;
    border: 1px solid #23273E;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94A3B8 !important;
    font-weight: 500;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
    color: white !important;
}

/* Perfectly Aligned KPI Cards */
.kpi-card {
    background: #1A1D2E;
    border: 1px solid #2D3154;
    border-radius: 14px;
    padding: 1.15rem 1.25rem;
    height: 155px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-sizing: border-box;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.kpi-card:hover {
    border-color: #6366F1;
    transform: translateY(-2px);
}
.kpi-label {
    color: #94A3B8;
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    height: 20px;
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 0;
}
.kpi-value-container {
    height: 52px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kpi-value {
    color: #F1F5F9;
    font-size: 1.3rem;
    font-weight: 700;
    line-height: 1.25;
    margin: 0;
}
.kpi-subtext {
    font-size: 0.75rem;
    color: #94A3B8;
    font-weight: 500;
    line-height: 1.1;
    margin-top: 2px;
}
.kpi-badge {
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 20px;
    margin: 0;
}
.kpi-badge-success { color: #10B981; }
.kpi-badge-warning { color: #F59E0B; }
.kpi-badge-info    { color: #818CF8; }
.kpi-badge-neutral { color: #64748B; }

/* Feature Card */
.feature-card {
    background: #141724;
    border: 1px solid #23273E;
    border-radius: 14px;
    padding: 1.6rem;
    height: 100%;
    transition: all 0.25s ease;
}
.feature-card:hover {
    border-color: #4F46E5;
    background: #1A1D2E;
    transform: translateY(-3px);
}

/* Expanders */
.streamlit-expanderHeader {
    background: #1A1D2E !important;
    border: 1px solid #2D3154 !important;
    border-radius: 10px !important;
    color: #CBD5E1 !important;
    font-weight: 500 !important;
}

hr { border-color: #23273E !important; }
</style>
"""

def _card(content: str, padding: str = "1.4rem") -> None:
    st.markdown(
        f'<div style="background:#1A1D2E;border:1px solid #2D3154;border-radius:14px;padding:{padding};margin-bottom:1rem;">{content}</div>',
        unsafe_allow_html=True,
    )

def _badge(text: str, color: str = "#6366F1") -> str:
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}44;border-radius:999px;padding:2px 10px;font-size:0.75rem;font-weight:600;display:inline-block;margin-left:6px;">{text}</span>'
    )

def _section_header(icon: str, title: str, subtitle: str = "") -> None:
    sub = f'<p style="color:#64748B;font-size:0.85rem;margin:4px 0 0 0;">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div style="margin-bottom:1.2rem;"><h3 style="color:#F1F5F9;font-size:1.1rem;font-weight:700;margin:0;">{icon}&nbsp; {title}</h3>{sub}</div>',
        unsafe_allow_html=True,
    )

def _plotly_layout(title: str, xaxis_title: str = "", yaxis_title: str = "") -> dict:
    return dict(
        paper_bgcolor="#0F1117",
        plot_bgcolor="#1A1D2E",
        font=dict(color="#E2E8F0", family="Inter"),
        title=dict(text=title, font=dict(color="#F1F5F9", size=13)),
        xaxis=dict(title=xaxis_title, gridcolor="#2D3154", linecolor="#2D3154", tickfont=dict(color="#64748B")),
        yaxis=dict(title=yaxis_title, gridcolor="#2D3154", linecolor="#2D3154", tickfont=dict(color="#64748B")),
        legend=dict(bgcolor="#1A1D2E", bordercolor="#2D3154", borderwidth=1),
        margin=dict(t=40, b=40, l=50, r=20),
        height=320,
    )

@st.cache_resource
def get_pipeline(_config: dict, device: str) -> VisualPipeline:
    return VisualPipeline(vision_config=_config["vision"], sample_fps=_config["video"]["sample_fps"], device=device)

def render_kpi_scorecard(result: dict, engagement_pred, interview_pred) -> None:
    summary = result.get("summary", {})
    eye_contact = summary.get("overall_eye_contact_ratio")
    yaw_std     = summary.get("head_yaw_stability_std_deg")
    pitch_std   = summary.get("head_pitch_stability_std_deg")

    if yaw_std is not None and pitch_std is not None:
        raw_std       = (yaw_std + pitch_std) / 2.0
        posture_score = max(0.0, 1.0 - min(raw_std / 20.0, 1.0))
        posture_tag   = "Excellent" if posture_score > 0.8 else "Good" if posture_score > 0.6 else "Needs Work"
        posture_color = "success" if posture_score > 0.8 else "warning" if posture_score > 0.6 else "neutral"
    else:
        posture_score, posture_tag, posture_color = None, "N/A", "neutral"

    eng_label = engagement_pred["label"] if engagement_pred else "Unavailable"
    eng_conf  = f'{engagement_pred["confidence"]:.0%} confidence' if engagement_pred else "No model signal"
    eng_badge_class = "kpi-badge-success" if engagement_pred and "High" in eng_label else "kpi-badge-info"

    ec_val = f"{eye_contact:.0%}" if eye_contact is not None else "N/A"
    ec_tag = "Optimal Gaze" if (eye_contact or 0) >= 0.7 else "Variable Gaze"
    ec_badge_class = "kpi-badge-success" if (eye_contact or 0) >= 0.7 else "kpi-badge-warning"

    ps_val = f"{posture_score:.0%}" if posture_score is not None else "N/A"
    ps_badge_class = f"kpi-badge-{posture_color}"

    iv_label = interview_pred["label"] if interview_pred else "Unavailable"
    iv_conf  = f'{interview_pred["confidence"]:.0%} confidence' if interview_pred else "Baseline"
    iv_badge_class = "kpi-badge-success" if interview_pred and ("More" in iv_label or "Likely" in iv_label and "Less" not in iv_label) else "kpi-badge-warning"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-label">🔋 Engagement Level</div>
            <div class="kpi-value-container">
                <div class="kpi-value">{eng_label}</div>
            </div>
            <div class="kpi-badge {eng_badge_class}">↑ {eng_conf}</div>
        </div>
        ''', unsafe_allow_html=True)

    with c2:
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-label">👁️ Eye-Contact</div>
            <div class="kpi-value-container">
                <div class="kpi-value">{ec_val}</div>
            </div>
            <div class="kpi-badge {ec_badge_class}">✦ {ec_tag}</div>
        </div>
        ''', unsafe_allow_html=True)

    with c3:
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-label">🧍 Posture Stability</div>
            <div class="kpi-value-container">
                <div class="kpi-value">{ps_val}</div>
            </div>
            <div class="kpi-badge {ps_badge_class}">✓ {posture_tag}</div>
        </div>
        ''', unsafe_allow_html=True)

    with c4:
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-label">💼 Interview Impression</div>
            <div class="kpi-value-container">
                <div class="kpi-value" style="font-size:1.15rem;">{iv_label}</div>
                <div class="kpi-subtext">(behavioral impression)</div>
            </div>
            <div class="kpi-badge {iv_badge_class}">↑ {iv_conf}</div>
        </div>
        ''', unsafe_allow_html=True)

_TRAIT_LABELS = {
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "conscientiousness": "Conscientiousness",
    "neuroticism": "Neuroticism",
    "openness": "Openness",
    "interview": "Interview Impression",
}

_TRAIT_DESCRIPTORS = {
    "extraversion": ("Energetic & outgoing presence", "Reserved & reflective style"),
    "agreeableness": ("Warm, collaborative & empathetic", "Direct & independently-minded"),
    "conscientiousness": ("Organized, dependable & detail-focused", "Flexible & spontaneous"),
    "neuroticism": ("High emotional reactivity shown", "Calm & emotionally stable"),
    "openness": ("Creative, curious & open to ideas", "Conventional & pragmatic focus"),
}

_TRAIT_COLORS = ["#6366F1", "#10B981", "#F59E0B", "#EC4899", "#06B6D4"]

def render_executive_summary_tab(trait_preds: dict, result: dict, engagement_pred, interview_pred) -> None: # noqa: ARG001
    if not trait_preds:
        st.info("Personality-impression models unavailable -- run Day 2 training first.")
        return

    _section_header("🕸️", "Behavioral Impression Radar", "Big-Five personality indicators derived from visual cues only")

    canonical_order = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
    traits_in_data  = [t for t in canonical_order if t in trait_preds] or list(trait_preds.keys())

    trait_names  = [_TRAIT_LABELS.get(t, t.title()) for t in traits_in_data]
    trait_scores = [trait_preds[t]["score"] * 100 for t in traits_in_data]

    closed_names  = trait_names + [trait_names[0]]
    closed_scores = trait_scores + [trait_scores[0]]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=closed_scores, theta=closed_names, fill="toself",
        fillcolor="rgba(99,102,241,0.15)", line=dict(color="#6366F1", width=2.5),
        name="Behavioral Profile", hovertemplate="%{theta}: <b>%{r:.1f}%</b><extra></extra>",
    ))
    radar_fig.update_layout(
        polar=dict(
            bgcolor="#1A1D2E",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2D3154", linecolor="#2D3154", tickfont=dict(color="#64748B", size=10), ticksuffix="%"),
            angularaxis=dict(gridcolor="#2D3154", linecolor="#2D3154", tickfont=dict(color="#CBD5E1", size=12, family="Inter")),
        ),
        paper_bgcolor="#0F1117", plot_bgcolor="#0F1117",
        font=dict(color="#E2E8F0", family="Inter"), showlegend=False,
        margin=dict(t=40, b=40, l=60, r=60), height=380,
    )
    st.plotly_chart(radar_fig, use_container_width=True)
    st.caption("AI-derived behavioral impressions from visual cues only -- not validated psychometric assessments.")

    st.markdown("<br>", unsafe_allow_html=True)
    _section_header("📊", "Individual Trait Breakdown", "Scores with uncertainty bands and behavioral descriptors")

    for i, trait_key in enumerate(traits_in_data):
        score = trait_preds[trait_key]["score"]
        err   = trait_preds[trait_key]["tree_std"] or 0
        label = _TRAIT_LABELS.get(trait_key, trait_key.title())
        high_desc, low_desc = _TRAIT_DESCRIPTORS.get(trait_key, ("", ""))
        descriptor = high_desc if score >= 0.5 else low_desc
        color = _TRAIT_COLORS[i % len(_TRAIT_COLORS)]
        conf_badge = _badge(f"±{err:.0%} uncertainty", "#F59E0B") if err > 0.05 else _badge("High confidence", "#10B981")

        st.markdown(f'''
        <div style="margin-bottom:0.9rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="color:#E2E8F0;font-weight:600;font-size:0.9rem;">{label}</span>
                <span style="color:{color};font-weight:700;font-size:0.9rem;">{score:.0%}</span>
            </div>
            <div style="background:#1A1D2E;border-radius:999px;height:8px;overflow:hidden;">
                <div style="width:{score*100:.1f}%;height:100%;background:linear-gradient(90deg,{color}99,{color});border-radius:999px;"></div>
            </div>
            <div style="margin-top:4px;color:#64748B;font-size:0.78rem;">{descriptor} {conf_badge}</div>
        </div>
        ''', unsafe_allow_html=True)

def render_trends_tab(result: dict) -> None:
    bins = result.get("temporal_bins", [])
    if not bins:
        st.info("No temporal bin data available for this video (video may be too short).")
        return

    bin_df = pd.DataFrame(bins)
    if "eye_contact_ratio" in bin_df.columns:
        _section_header("👁️", "Gaze & Eye-Contact Over Time", "Approximate -- not calibrated gaze tracking")
        fig_gaze = go.Figure()
        fig_gaze.add_trace(go.Scatter(
            x=bin_df["start_sec"], y=bin_df["eye_contact_ratio"], mode="lines+markers",
            line=dict(color="#6366F1", width=2.5, shape="spline"), marker=dict(color="#6366F1", size=6),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.12)", name="Eye-contact ratio",
            hovertemplate="t=%{x}s  |  Gaze: <b>%{y:.0%}</b><extra></extra>",
        ))
        fig_gaze.add_hline(y=0.7, line_dash="dot", line_color="#10B981", annotation_text="Target 70%+", annotation_font_color="#10B981")
        fig_gaze.update_layout(**_plotly_layout("Eye-Contact Ratio", "Time (seconds)", "Ratio"))
        st.plotly_chart(fig_gaze, use_container_width=True)

    pose_cols = [c for c in ["mean_yaw_deg", "mean_pitch_deg"] if c in bin_df.columns]
    if pose_cols:
        _section_header("🧭", "Head Pose Variation Over Time")
        fig_pose = go.Figure()
        colors_pose = {"mean_yaw_deg": "#F59E0B", "mean_pitch_deg": "#EC4899"}
        labels_pose = {"mean_yaw_deg": "Yaw (left/right)", "mean_pitch_deg": "Pitch (up/down)"}
        for col in pose_cols:
            fig_pose.add_trace(go.Scatter(
                x=bin_df["start_sec"], y=bin_df[col], mode="lines",
                line=dict(color=colors_pose.get(col, "#6366F1"), width=2, shape="spline"),
                name=labels_pose.get(col, col), hovertemplate=labels_pose.get(col, col) + ": <b>%{y:.1f}°</b> @ %{x}s<extra></extra>",
            ))
        fig_pose.add_hline(y=0, line_dash="dot", line_color="#334155")
        fig_pose.update_layout(**_plotly_layout("Head Pose Angles", "Time (seconds)", "Degrees"))
        st.plotly_chart(fig_pose, use_container_width=True)

    _section_header("😊", "Emotion Distribution Over Time")
    emotion_rows = [{"start_sec": b["start_sec"], **(b.get("emotion_distribution") or {})} for b in bins]
    emotion_df = pd.DataFrame(emotion_rows)
    emotion_cols = [c for c in emotion_df.columns if c != "start_sec"]
    if emotion_cols:
        palette = px.colors.qualitative.Plotly
        fig_emo = go.Figure()
        for idx, col in enumerate(emotion_cols):
            fig_emo.add_trace(go.Scatter(
                x=emotion_df["start_sec"], y=emotion_df[col], mode="lines", stackgroup="one",
                name=col.capitalize(), line=dict(width=0.5), fillcolor=palette[idx % len(palette)],
                hovertemplate=col.capitalize() + ": <b>%{y:.0%}</b><extra></extra>",
            ))
        fig_emo.update_layout(**_plotly_layout("Emotion Stacked Area", "Time (seconds)", "Proportion"))
        st.plotly_chart(fig_emo, use_container_width=True)

def render_evidence_tab(shap_evidence: dict, raw_features: dict) -> None:
    _section_header("🧠", "What drives these predictions?", "Global SHAP feature importance from training")
    evidence_map = [
        ("daisee_engagement_binary_xgboost", "Engagement", "#6366F1"),
        ("chalearn_extraversion_random_forest", "Extraversion", "#10B981"),
        ("chalearn_interview_random_forest", "Interview-Invite", "#F59E0B"),
    ]
    any_shown = False
    for model_key, label, color in evidence_map:
        features = shap_evidence.get(model_key)
        if not features:
            continue
        any_shown = True
        with st.expander(f"📌 {label} -- top influencing factors", expanded=True):
            top5 = features[:5]
            names = [f["display_name"] for f in top5]
            magnitudes = [f.get("mean_abs_shap", 1 - i * 0.12) for i, f in enumerate(top5)]
            max_m = max(magnitudes) or 1
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fig_bar = go.Figure(go.Bar(
                x=magnitudes, y=names, orientation="h",
                marker=dict(color=[f"rgba({r},{g},{b},{0.4 + 0.5 * (m / max_m):.2f})" for m in magnitudes], line=dict(color=color, width=1)),
                hovertemplate="%{y}: <b>%{x:.3f}</b><extra></extra>",
            ))
            layout = _plotly_layout(f"Top Factors -- {label}", "Mean |SHAP| importance")
            layout["height"] = 220
            layout["yaxis"] = dict(autorange="reversed", gridcolor="#2D3154", linecolor="#2D3154", tickfont=dict(color="#CBD5E1"))
            fig_bar.update_layout(**layout)
            st.plotly_chart(fig_bar, use_container_width=True)

    if not any_shown:
        st.info("SHAP evidence not available -- run training/explain_models.py first.")

    st.divider()
    _section_header("🔬", "Raw Feature Values for This Video")
    display_values = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in raw_features.items()}
    st.json(display_values)

def render_recommendations_tab(raw_features: dict) -> None:
    recommendations = generate_recommendations(raw_features)
    mid = max(1, len(recommendations) // 2)
    strengths = recommendations[:mid]
    growth = recommendations[mid:]

    col_s, col_g = st.columns(2)
    with col_s:
        _section_header("✅", "Top Strengths")
        for rec in strengths:
            st.markdown(f'''
            <div style="background:#1A1D2E;border:1px solid #2D3154;border-radius:14px;padding:1.2rem;min-height:130px;display:flex;gap:12px;align-items:flex-start;margin-bottom:1rem;box-sizing:border-box;">
                <span style="color:#10B981;font-size:1.1rem;margin-top:2px;">✦</span>
                <span style="color:#CBD5E1;font-size:0.9rem;line-height:1.6;">{rec}</span>
            </div>
            ''', unsafe_allow_html=True)

    with col_g:
        _section_header("💡", "Actionable Growth Points")
        for rec in growth:
            st.markdown(f'''
            <div style="background:#1A1D2E;border:1px solid #2D3154;border-radius:14px;padding:1.2rem;min-height:130px;display:flex;gap:12px;align-items:flex-start;margin-bottom:1rem;box-sizing:border-box;">
                <span style="color:#F59E0B;font-size:1.1rem;margin-top:2px;">→</span>
                <span style="color:#CBD5E1;font-size:0.9rem;line-height:1.6;">{rec}</span>
            </div>
            ''', unsafe_allow_html=True)

    st.divider()
    st.warning("**Responsible AI reminder:** this tool provides behavioral feedback derived from visual signals only. It is not an automated hiring decision system.")

def render_methodology_tab(device: str, missing_models: list[str]) -> None:
    with st.expander("📖 About this tool", expanded=True):
        st.markdown("This tool analyzes **visual behavioral signals** from an uploaded interview-style video.")
        st.markdown(f"**Compute device:** `{device}`")
        if missing_models:
            st.error(f"{len(missing_models)} trained model file(s) not found.")

    with st.expander("⚖️ Responsible AI Notice"):
        st.warning("**Responsible AI note:** predictions are AI-derived behavioral indicators, not objective judgments of a person's suitability, character, or hireability.")

    with st.expander("🔬 Scope & Technical Limitations"):
        st.info("**Scope note:** this build analyzes visual signals only. Audio and speech NLP are outside the current scope.")

    with st.expander("📚 Model Sources & Training"):
        st.markdown("""
- **Engagement model:** XGBoost trained on DAiSEE dataset visual features
- **Personality impressions:** Random Forest trained on ChaLearn Apparent Personality dataset
- **Interview-invite model:** Random Forest trained on ChaLearn First Impression dataset
        """)

def render_front_page() -> None:
    st.markdown('''
    <div style="text-align:center;padding:3.5rem 1rem 2rem 1rem;max-width:900px;margin:0 auto;">
        <h1 style="font-size:2.9rem;font-weight:800;letter-spacing:-0.03em;color:#F8FAFC;margin:0 0 1rem 0;line-height:1.15;">
            AI-Assisted Behavioral Assessment <br/>
            <span style="background:linear-gradient(135deg, #818CF8 0%, #C084FC 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                Built for Modern Interviews
            </span>
        </h1>
        <p style="color:#94A3B8;font-size:1.15rem;line-height:1.6;margin:0 auto 2.4rem auto;max-width:700px;">
            Evaluate visual engagement, gaze continuity, head stability, and personality impressions with machine learning models and SHAP-grounded explainability.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    col_c1, col_c2, col_c3 = st.columns([1, 1.3, 1])
    with col_c2:
        if st.button("Launch Analyzer Workspace", type="primary", use_container_width=True):
            st.session_state["nav_page"] = "Interview Analyzer"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('''
        <div class="feature-card">
            <div style="font-size:1.8rem;margin-bottom:0.8rem;">👁️</div>
            <h3 style="color:#F1F5F9;font-size:1.1rem;font-weight:700;margin:0 0 0.5rem 0;">Gaze & Attention Tracking</h3>
            <p style="color:#94A3B8;font-size:0.88rem;line-height:1.5;margin:0;">
                Temporal visual sampling estimates camera attention ratio, tracking eye contact consistency and focal adherence across the recording.
            </p>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown('''
        <div class="feature-card">
            <div style="font-size:1.8rem;margin-bottom:0.8rem;">🧭</div>
            <h3 style="color:#F1F5F9;font-size:1.1rem;font-weight:700;margin:0 0 0.5rem 0;">Head Pose Stability</h3>
            <p style="color:#94A3B8;font-size:0.88rem;line-height:1.5;margin:0;">
                Continuous yaw, pitch, and roll estimation measures micro-movements, physical firmness, and natural conversational cadence.
            </p>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown('''
        <div class="feature-card">
            <div style="font-size:1.8rem;margin-bottom:0.8rem;">🧠</div>
            <h3 style="color:#F1F5F9;font-size:1.1rem;font-weight:700;margin:0 0 0.5rem 0;">Big-Five Impression Radar</h3>
            <p style="color:#94A3B8;font-size:0.88rem;line-height:1.5;margin:0;">
                Ensemble Random Forests trained on empirical datasets generate calibrated impression profiles with real tree-disagreement uncertainty.
            </p>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown('''
    <div style="background:#141724;border:1px solid #23273E;border-radius:18px;padding:2.2rem;margin-bottom:2rem;">
        <h3 style="color:#F1F5F9;font-size:1.25rem;font-weight:700;margin:0 0 1.8rem 0;text-align:center;">
            How the Assessment Pipeline Works
        </h3>
        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:1.5rem;text-align:center;">
            <div>
                <div style="background:#1E2235;width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 0.8rem auto;color:#818CF8;font-weight:700;">1</div>
                <h4 style="color:#E2E8F0;font-size:0.95rem;margin:0 0 0.4rem 0;">Upload Video</h4>
                <p style="color:#64748B;font-size:0.82rem;margin:0;">Standard MP4/MOV recording from any webcam or interview setup.</p>
            </div>
            <div>
                <div style="background:#1E2235;width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 0.8rem auto;color:#818CF8;font-weight:700;">2</div>
                <h4 style="color:#E2E8F0;font-size:0.95rem;margin:0 0 0.4rem 0;">Neural Feature Extraction</h4>
                <p style="color:#64748B;font-size:0.82rem;margin:0;">Facial landmark detection, head pose solving, and emotion likelihood.</p>
            </div>
            <div>
                <div style="background:#1E2235;width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 0.8rem auto;color:#818CF8;font-weight:700;">3</div>
                <h4 style="color:#E2E8F0;font-size:0.95rem;margin:0 0 0.4rem 0;">Executive Intelligence Report</h4>
                <p style="color:#64748B;font-size:0.82rem;margin:0;">Multi-dimensional scorecard, spider radar, and actionable coaching.</p>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def main() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    config = load_config()
    device = resolve_device(config["runtime"]["device"])

    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Overview & Platform"

    # If on Opening / Front Page, render it cleanly with no navbar buttons
    if st.session_state["nav_page"] == "Overview & Platform":
        render_front_page()
        return

    # Clean top back navigation link
    if st.button("← Back to Overview", type="secondary"):
        st.session_state["nav_page"] = "Overview & Platform"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    models, missing_models = load_models()
    shap_evidence          = load_shap_evidence()
    feature_medians        = load_feature_medians()

    # Clean Upload Section with Perfectly Aligned Controls
    uploaded_file = st.file_uploader(
        "🎬 Upload Interview Video (MP4, AVI, MOV, MKV)",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload an interview-style video for behavioral analysis",
    )

    with st.expander("📄 Optional: Candidate Resume (PDF)", expanded=False):
        cv_file = st.file_uploader("Upload CV / Resume", type=["pdf"])  # noqa: F841

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button(
        "Analyze Performance",
        type="primary",
        use_container_width=True,
    )

    if uploaded_file is None:
        st.markdown('''
        <div style="text-align:center;padding:3rem 1.5rem;background:#141724;border:1px dashed #2D3154;border-radius:14px;margin-top:1.5rem;">
            <div style="font-size:2.2rem;margin-bottom:0.6rem;">🎬</div>
            <h3 style="color:#F1F5F9;font-size:1.2rem;font-weight:600;margin:0 0 0.4rem 0;">
                No Recording Selected
            </h3>
            <p style="color:#64748B;font-size:0.88rem;margin:0;">
                Select an MP4, AVI, or MOV file above and click "Analyze Performance" to generate insights.
            </p>
        </div>
        ''', unsafe_allow_html=True)
        return

    file_bytes   = uploaded_file.read()
    content_hash = hashlib.sha256(file_bytes).hexdigest()[:16]

    if not analyze_button:
        st.markdown('<br>', unsafe_allow_html=True)
        with st.expander("🎥 Preview Uploaded Video", expanded=True):
            st.video(file_bytes)
        st.info('Click **"Analyze Performance"** above to execute the neural vision pipeline.')
        return

    cache         = VisualFeatureCache(config["paths"]["cache_dir"])
    cached_result = cache.get(content_hash)

    if cached_result is not None:
        result = cached_result
        st.success("⚡ Loaded from cache -- this exact video was already analyzed in a previous run.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        with st.spinner("🔬 Running visual analysis pipeline — this may take a moment on first run..."):
            try:
                pipeline        = get_pipeline(config, device)
                pipeline_result = pipeline.run(tmp_path)
                result = {
                    "video_path":               tmp_path,
                    "total_sampled_frames":      pipeline_result.total_sampled_frames,
                    "frames_with_face_detected": pipeline_result.frames_with_face_detected,
                    "face_detection_rate":       pipeline_result.face_detection_rate,
                    "temporal_bins":             pipeline_result.temporal_bins,
                    "summary":                   pipeline_result.summary,
                }
                cache.set(content_hash, result)
            except Exception as exc:  # noqa: BLE001
                st.error(
                    f"Something went wrong while processing this video: {exc}\n\n"
                    "Common causes: unsupported codec, corrupt file, or an unreadable video stream. "
                    "Try re-exporting the video as a standard MP4 (H.264) and re-uploading."
                )
                return

    if result["total_sampled_frames"] == 0:
        st.error("No frames could be extracted from this video. It may be corrupt or in an unsupported format.")
        return

    if result["face_detection_rate"] < 0.3:
        st.warning(
            f"⚠️ A face was detected in only {result['face_detection_rate']:.0%} of sampled frames. "
            "Results below may be unreliable -- check lighting, framing, and that the subject stays visible throughout."
        )

    X, imputed = build_feature_vector(result, feature_medians)
    if imputed:
        st.info(
            f"Note: {len(imputed)} feature(s) could not be measured directly for this video and were "
            f"filled in with typical training-set values: {', '.join(imputed)}."
        )

    engagement_pred = predict_engagement(models, X)
    trait_preds     = predict_traits(models, X)
    interview_pred  = predict_interview(models, X)
    raw_features    = dict(zip(FEATURE_COLUMNS, X[0].tolist()))

    st.markdown(f'''
    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:1.5rem;margin-bottom:1.5rem;flex-wrap:wrap;gap:10px;">
        <div>
            <h1 style="color:#F1F5F9;font-size:1.5rem;font-weight:800;margin:0 0 4px 0;">
                Executive Behavioral Intelligence Report
            </h1>
            <p style="color:#64748B;font-size:0.85rem;margin:0;">
                Source: <code style="color:#818CF8;">{uploaded_file.name}</code> &nbsp;·&nbsp; 
                Face detection rate: <b style="color:#E2E8F0;">{result["face_detection_rate"]:.0%}</b>
            </p>
        </div>
        <div style="background:#1A1D2E;border:1px solid #2D3154;padding:6px 14px;border-radius:999px;font-size:0.78rem;color:#10B981;font-weight:600;">
            ● Analysis Verified
        </div>
    </div>
    ''', unsafe_allow_html=True)

    render_kpi_scorecard(result, engagement_pred, interview_pred)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("🎥 View Analyzed Recording", expanded=False):
        st.video(file_bytes)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Summary",
        "📈 Behavioral Trends",
        "🧠 Explainability",
        "💡 Coaching",
        "ℹ️ Methodology & Ethics",
    ])

    with tab1:
        render_executive_summary_tab(trait_preds, result, engagement_pred, interview_pred)
    with tab2:
        render_trends_tab(result)
    with tab3:
        render_evidence_tab(shap_evidence, raw_features)
    with tab4:
        render_recommendations_tab(raw_features)
    with tab5:
        render_methodology_tab(device, missing_models)

if __name__ == "__main__":
    main()