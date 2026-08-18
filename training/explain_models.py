"""
Phase 5 (Day 3, 5-day scope) — Explainability via SHAP (Module 14).

Loads the already-trained models from Day 1/Day 2 and computes SHAP
feature-importance explanations, so the system can answer "why did it
predict this?" with actual evidence rather than an opaque score.

Uses TreeExplainer (exact, fast) since all our models are tree-based
(Random Forest / XGBoost) or linear (Logistic/Linear Regression, for
which SHAP's LinearExplainer is used instead) — no expensive
approximate/kernel SHAP needed anywhere in this project.

"Hero" models explained (chosen because they were the best performer
in their respective task per Day 1/Day 2 results):
  - DAiSEE engagement_binary: XGBoost (best F1-macro: 0.552)
  - ChaLearn extraversion: Random Forest (best R2: 0.242)
  - ChaLearn interview-invite: Random Forest (best F1-macro: 0.638)

Usage:
    python training/explain_models.py

Output:
    - outputs/plots/shap_summary_<model_key>.png   (global feature importance)
    - outputs/shap_top_features.json                (top features per model, machine-readable — used later by the Streamlit "evidence" panel)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.chalearn_dataset import (  # noqa: E402
    FEATURE_COLUMNS as CHALEARN_FEATURE_COLUMNS,
    load_chalearn_baseline_data,
    prepare_chalearn_splits,
)
from src.models.daisee_dataset import (  # noqa: E402
    FEATURE_COLUMNS as DAISEE_FEATURE_COLUMNS,
    load_daisee_baseline_data,
    prepare_daisee_splits,
)
from src.utils.config_loader import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

# Human-readable labels for the plots/JSON — the raw column names are
# fine for code but not for a dashboard evidence panel.
FEATURE_DISPLAY_NAMES = {
    "face_detection_rate": "Face detection rate",
    "overall_eye_contact_ratio": "Eye-contact ratio (approx.)",
    "head_yaw_stability_std_deg": "Head yaw stability",
    "head_pitch_stability_std_deg": "Head pitch stability",
    "multiple_faces_detected_frame_count": "Multiple-faces-detected frames",
    "emotion_neutral": "Neutral expression share",
    "emotion_happy": "Happy expression share",
    "emotion_sad": "Sad expression share",
    "emotion_angry": "Angry expression share",
    "emotion_fear": "Fear expression share",
    "emotion_surprise": "Surprise expression share",
    "emotion_disgust": "Disgust expression share",
}


def explain_model(
    model_key: str,
    model_path: Path,
    X_test,
    feature_columns: list[str],
    logger,
    outputs_dir: Path,
    top_n: int = 12,
) -> dict:
    """Load a trained model, compute SHAP values on the test set, save a summary plot, and return the top features."""
    if not model_path.exists():
        logger.warning(f"Model file not found, skipping: {model_path}")
        return {}

    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle.get("scaler")

    X_for_shap = scaler.transform(X_test) if scaler is not None else X_test
    display_names = [FEATURE_DISPLAY_NAMES.get(c, c) for c in feature_columns]

    logger.info(f"Computing SHAP values for {model_key} ({type(model).__name__})...")

    # TreeExplainer for tree models, LinearExplainer for linear models —
    # both are exact and fast, no approximation needed for this model size.
    model_class_name = type(model).__name__
    if "Forest" in model_class_name or "XGB" in model_class_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_for_shap)
        # SHAP's TreeExplainer output shape varies by model type and
        # library version:
        #   - Older shap versions / some models: list of arrays, one per class.
        #   - Newer shap versions, multi-class-capable models (even when
        #     used for binary classification, e.g. sklearn's
        #     RandomForestClassifier): a single 3D ndarray shaped
        #     (samples, features, n_classes).
        #   - Single-output models (XGBClassifier binary, any regressor):
        #     plain 2D ndarray (samples, features) — used as-is.
        # Confirmed empirically during Phase 5 Day 3: this exact 3D-ndarray
        # case broke plotting for the ChaLearn interview RandomForestClassifier
        # while the other two (XGBoost, RandomForestRegressor) never hit it,
        # since both of those produce plain 2D output.
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # (samples, features, classes) -> take the positive/high class
            shap_values = shap_values[:, :, 1] if shap_values.shape[-1] > 1 else shap_values[:, :, 0]
    else:
        explainer = shap.LinearExplainer(model, X_for_shap)
        shap_values = explainer.shap_values(X_for_shap)

    fig = plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, X_for_shap, feature_names=display_names, show=False, plot_size=None)
    plt.title(f"SHAP feature importance — {model_key}")
    plt.tight_layout()
    plot_path = outputs_dir / "plots" / f"shap_summary_{model_key}.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {plot_path}")

    # Mean absolute SHAP value per feature = global importance ranking.
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_columns, display_names, mean_abs_shap), key=lambda x: -x[2])

    return {
        "model_key": model_key,
        "model_type": model_class_name,
        "top_features": [
            {"feature": col, "display_name": name, "mean_abs_shap": float(val)}
            for col, name, val in ranked[:top_n]
        ],
    }


def main() -> None:
    config = load_config()
    logger = get_logger(__name__, log_dir=config["paths"]["logs_dir"], log_filename="explain_models.log")

    outputs_dir = Path(config["paths"]["outputs_dir"])
    (outputs_dir / "plots").mkdir(parents=True, exist_ok=True)
    models_dir = Path("models/trained")

    all_explanations = []

    # --- DAiSEE: engagement_binary, XGBoost (best F1-macro per Day 1 results) ---
    daisee_csv = outputs_dir / "daisee_visual_features_summary.csv"
    if daisee_csv.exists():
        df = load_daisee_baseline_data(daisee_csv)
        splits = prepare_daisee_splits(df, target_column="Engagement")
        explanation = explain_model(
            model_key="daisee_engagement_binary_xgboost",
            model_path=models_dir / "daisee_engagement_binary_xgboost.joblib",
            X_test=splits["test"]["X"],
            feature_columns=DAISEE_FEATURE_COLUMNS,
            logger=logger,
            outputs_dir=outputs_dir,
        )
        if explanation:
            all_explanations.append(explanation)
    else:
        logger.warning(f"{daisee_csv} not found — skipping DAiSEE explainability.")

    # --- ChaLearn: extraversion, Random Forest (best R2 per Day 2 results) ---
    chalearn_csv = outputs_dir / "chalearn_visual_features_summary.csv"
    if chalearn_csv.exists():
        df = load_chalearn_baseline_data(chalearn_csv)
        splits = prepare_chalearn_splits(df)

        explanation = explain_model(
            model_key="chalearn_extraversion_random_forest",
            model_path=models_dir / "chalearn_extraversion_random_forest.joblib",
            X_test=splits["test"]["X"],
            feature_columns=CHALEARN_FEATURE_COLUMNS,
            logger=logger,
            outputs_dir=outputs_dir,
        )
        if explanation:
            all_explanations.append(explanation)

        # --- ChaLearn: interview-invite, Random Forest (best F1-macro per Day 2 results) ---
        explanation = explain_model(
            model_key="chalearn_interview_random_forest",
            model_path=models_dir / "chalearn_interview_random_forest.joblib",
            X_test=splits["test"]["X"],
            feature_columns=CHALEARN_FEATURE_COLUMNS,
            logger=logger,
            outputs_dir=outputs_dir,
        )
        if explanation:
            all_explanations.append(explanation)
    else:
        logger.warning(f"{chalearn_csv} not found — skipping ChaLearn explainability.")

    output_json_path = outputs_dir / "shap_top_features.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_explanations, f, indent=2)
    logger.info(f"Top-feature summary (all models) written to: {output_json_path}")

    for explanation in all_explanations:
        logger.info(f"\nTop features for {explanation['model_key']} ({explanation['model_type']}):")
        for feat in explanation["top_features"][:5]:
            logger.info(f"  {feat['display_name']}: {feat['mean_abs_shap']:.4f}")


if __name__ == "__main__":
    main()
