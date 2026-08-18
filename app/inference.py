"""
Feature-vector construction and multi-model inference for the
Streamlit app. Kept separate from the UI file per Module 24
(training / inference / UI / config must stay separated).
"""

from __future__ import annotations

import numpy as np

from src.models.daisee_dataset import FEATURE_COLUMNS

TRAIT_NAMES = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]


def build_feature_vector(result: dict, feature_medians: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    """
    Build the 12-feature vector (same order/definition as training) from
    a single video's VisualPipeline result.

    Args:
        result: dict with "face_detection_rate" (top-level) and "summary"
            (nested dict with the rest) — matches the structure written
            by scripts/run_visual_pipeline_demo.py and the batch scripts.
        feature_medians: fallback values for any feature that's None.

    Returns:
        (X, imputed_feature_names) — X is shape (1, 12), ready for
        model.predict(). imputed_feature_names lists which features (if
        any) had to be filled in with training-set medians, so the UI
        can disclose this rather than silently substituting values.
    """
    summary = result.get("summary", {})
    emotion_dist = summary.get("overall_emotion_distribution") or {}

    raw = {
        "face_detection_rate": result.get("face_detection_rate"),
        "overall_eye_contact_ratio": summary.get("overall_eye_contact_ratio"),
        "head_yaw_stability_std_deg": summary.get("head_yaw_stability_std_deg"),
        "head_pitch_stability_std_deg": summary.get("head_pitch_stability_std_deg"),
        "multiple_faces_detected_frame_count": summary.get("multiple_faces_detected_frame_count"),
        "emotion_neutral": emotion_dist.get("neutral"),
        "emotion_happy": emotion_dist.get("happy"),
        "emotion_sad": emotion_dist.get("sad"),
        "emotion_angry": emotion_dist.get("angry"),
        "emotion_fear": emotion_dist.get("fear"),
        "emotion_surprise": emotion_dist.get("surprise"),
        "emotion_disgust": emotion_dist.get("disgust"),
    }

    imputed = []
    values = []
    for col in FEATURE_COLUMNS:
        val = raw.get(col)
        if val is None:
            val = feature_medians.get(col, 0.0)
            imputed.append(col)
        values.append(val)

    X = np.array(values, dtype=np.float64).reshape(1, -1)
    return X, imputed


def _predict_with_bundle(bundle: dict, X: np.ndarray):
    """Apply a bundle's scaler (if any) and return the fitted model's prediction inputs."""
    model = bundle["model"]
    scaler = bundle.get("scaler")
    X_in = scaler.transform(X) if scaler is not None else X
    return model, X_in


def predict_engagement(models: dict, X: np.ndarray) -> dict | None:
    """DAiSEE engagement_binary (XGBoost, the Day 1 best F1-macro model)."""
    bundle = models.get("daisee_engagement_binary")
    if bundle is None:
        return None
    model, X_in = _predict_with_bundle(bundle, X)
    pred = int(model.predict(X_in)[0])
    proba = model.predict_proba(X_in)[0]
    return {
        "label": "High Engagement" if pred == 1 else "Low Engagement",
        "confidence": float(max(proba)),
        "raw_class": pred,
    }


def predict_traits(models: dict, X: np.ndarray) -> dict[str, dict]:
    """
    ChaLearn Big-Five traits (Random Forest, the Day 2 best per-trait models).

    Uncertainty proxy: standard deviation across individual trees'
    predictions within the Random Forest. This is a REAL, model-derived
    uncertainty signal (not fabricated) — a forest where all trees agree
    indicates a stable prediction; high disagreement indicates the
    model itself is unsure for this input.
    """
    traits = {}
    for trait in TRAIT_NAMES:
        bundle = models.get(f"chalearn_{trait}")
        if bundle is None:
            continue
        model, X_in = _predict_with_bundle(bundle, X)
        pred = float(model.predict(X_in)[0])

        tree_std = None
        if hasattr(model, "estimators_"):
            tree_preds = np.array([tree.predict(X_in)[0] for tree in model.estimators_])
            tree_std = float(tree_preds.std())

        traits[trait] = {"score": pred, "tree_std": tree_std}
    return traits


def predict_interview(models: dict, X: np.ndarray) -> dict | None:
    """ChaLearn interview-invite (Random Forest, the Day 2 best F1-macro model)."""
    bundle = models.get("chalearn_interview")
    if bundle is None:
        return None
    model, X_in = _predict_with_bundle(bundle, X)
    pred = int(model.predict(X_in)[0])
    proba = model.predict_proba(X_in)[0]
    return {
        "label": "Likely Invited (behavioral impression)" if pred == 1 else "Less Likely Invited (behavioral impression)",
        "confidence": float(max(proba)),
        "raw_class": pred,
    }
