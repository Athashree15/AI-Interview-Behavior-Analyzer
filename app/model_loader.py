"""
Cached resource loading for the Streamlit app.

Everything here uses st.cache_resource / st.cache_data so model files,
SHAP JSON, and median fallbacks are loaded ONCE per app session, not
re-read from disk on every Streamlit rerun (which happens on every
widget interaction).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.models.daisee_dataset import FEATURE_COLUMNS

MODEL_FILES = {
    "daisee_engagement_binary": "models/trained/daisee_engagement_binary_xgboost.joblib",
    "chalearn_extraversion": "models/trained/chalearn_extraversion_random_forest.joblib",
    "chalearn_agreeableness": "models/trained/chalearn_agreeableness_random_forest.joblib",
    "chalearn_conscientiousness": "models/trained/chalearn_conscientiousness_random_forest.joblib",
    "chalearn_neuroticism": "models/trained/chalearn_neuroticism_random_forest.joblib",
    "chalearn_openness": "models/trained/chalearn_openness_random_forest.joblib",
    "chalearn_interview": "models/trained/chalearn_interview_random_forest.joblib",
}


@st.cache_resource
def load_models() -> tuple[dict[str, dict], list[str]]:
    """
    Load every trained model bundle ({"model": ..., "scaler": ...}).
    Returns (models_dict, missing_paths) — missing models are reported,
    not silently skipped, so the UI can warn the user which predictions
    are unavailable rather than mysteriously not showing up.
    """
    models = {}
    missing = []
    for key, rel_path in MODEL_FILES.items():
        path = Path(rel_path)
        if path.exists():
            models[key] = joblib.load(path)
        else:
            missing.append(rel_path)
    return models, missing


@st.cache_data
def load_shap_evidence(outputs_dir: str = "outputs") -> dict[str, list[dict]]:
    """Load the Day 3 SHAP top-features JSON, keyed by model_key."""
    path = Path(outputs_dir) / "shap_top_features.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry["model_key"]: entry["top_features"] for entry in data}


@st.cache_data
def load_feature_medians(outputs_dir: str = "outputs") -> dict[str, float]:
    """
    Fallback values for any feature that comes back as None for a given
    uploaded video (e.g. gaze never estimated because iris landmarks
    were unavailable in every sampled frame). Computed from the DAiSEE
    TRAIN split specifically — matching the no-leakage discipline used
    throughout training, not an arbitrary convenience choice.

    Falls back to 0.0 for every feature if the training summary CSV
    isn't available (e.g. app is run on a machine that only has the
    trained model files, not the raw feature CSVs) — a degraded but
    non-crashing fallback, with the UI expected to disclose when
    imputation happened (see app/inference.py).
    """
    daisee_csv = Path(outputs_dir) / "daisee_visual_features_summary.csv"
    if not daisee_csv.exists():
        return {col: 0.0 for col in FEATURE_COLUMNS}

    df = pd.read_csv(daisee_csv)
    train_df = df[df["split"] == "train"]
    medians = train_df[FEATURE_COLUMNS].median()
    return medians.to_dict()
