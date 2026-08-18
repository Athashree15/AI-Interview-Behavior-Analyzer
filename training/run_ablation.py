"""
Phase 5 (Day 3, 5-day scope) — Feature-group ablation study.

IMPORTANT SCOPING NOTE: Module 12/28's original ablation design compares
visual/audio/NLP modality contributions. Under the 5-day submission
deadline, audio/NLP/fusion were cut from scope (see project limitations
section) — this script performs the analogous ablation WITHIN the visual
pipeline instead: gaze/head-pose features vs. emotion features vs. both
combined. This is the honest, available substitute for the originally
planned cross-modality ablation, not a claim that it's the same study.

Feature groups:
  - GAZE_HEADPOSE: face_detection_rate, eye-contact ratio, head yaw/pitch
    stability, multiple-faces-detected count.
  - EMOTION: the 7 emotion-distribution features.

Answers the research-relevant question: "does combining gaze/head-pose
behavior with emotion expression actually improve prediction over
either alone?" — for both DAiSEE engagement and ChaLearn's two tasks.

Usage:
    python training/run_ablation.py

Output:
    - outputs/ablation_results.csv (one row per dataset x task x feature-group)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.class_weight import compute_sample_weight

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.chalearn_dataset import load_chalearn_baseline_data, prepare_chalearn_splits  # noqa: E402
from src.models.daisee_dataset import load_daisee_baseline_data, prepare_daisee_splits  # noqa: E402
from src.models.evaluation import evaluate_classifier, evaluate_regressor  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

GAZE_HEADPOSE_FEATURES = [
    "face_detection_rate",
    "overall_eye_contact_ratio",
    "head_yaw_stability_std_deg",
    "head_pitch_stability_std_deg",
    "multiple_faces_detected_frame_count",
]
EMOTION_FEATURES = [
    "emotion_neutral",
    "emotion_happy",
    "emotion_sad",
    "emotion_angry",
    "emotion_fear",
    "emotion_surprise",
    "emotion_disgust",
]
ALL_FEATURES = GAZE_HEADPOSE_FEATURES + EMOTION_FEATURES

FEATURE_GROUPS = {
    "gaze_headpose_only": GAZE_HEADPOSE_FEATURES,
    "emotion_only": EMOTION_FEATURES,
    "all_features": ALL_FEATURES,
}


def run_daisee_ablation(config, logger) -> list[dict]:
    """Ablation for DAiSEE engagement_binary using XGBoost (the Day 1 best-F1-macro model)."""
    outputs_dir = Path(config["paths"]["outputs_dir"])
    csv_path = outputs_dir / "daisee_visual_features_summary.csv"
    if not csv_path.exists():
        logger.warning(f"{csv_path} not found — skipping DAiSEE ablation.")
        return []
    if not XGBOOST_AVAILABLE:
        logger.warning("xgboost not installed — skipping DAiSEE ablation (uses XGBoost as the hero model).")
        return []

    df = load_daisee_baseline_data(csv_path)
    seed = config["runtime"]["seed"]

    results = []
    for group_name, feature_cols in FEATURE_GROUPS.items():
        splits = prepare_daisee_splits(df, target_column="Engagement", feature_columns=feature_cols)
        X_train, y_train = splits["train"]["X"], splits["train"]["y_binary"]
        X_test, y_test = splits["test"]["X"], splits["test"]["y_binary"]

        model = XGBClassifier(n_estimators=300, random_state=seed, eval_metric="logloss", n_jobs=-1)
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = evaluate_classifier(y_test, y_pred, y_proba, ["Low", "High"])

        logger.info(
            f"[DAiSEE engagement_binary][{group_name}] "
            f"n_features={len(feature_cols)} acc={metrics['accuracy']:.3f} f1_macro={metrics['f1_macro']:.3f}"
        )
        results.append(
            {
                "dataset": "daisee",
                "task": "engagement_binary",
                "feature_group": group_name,
                "n_features": len(feature_cols),
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "roc_auc": metrics.get("roc_auc"),
            }
        )
    return results


def run_chalearn_ablation(config, logger) -> list[dict]:
    """Ablation for ChaLearn extraversion (regression) and interview-invite (classification), both Random Forest."""
    outputs_dir = Path(config["paths"]["outputs_dir"])
    csv_path = outputs_dir / "chalearn_visual_features_summary.csv"
    if not csv_path.exists():
        logger.warning(f"{csv_path} not found — skipping ChaLearn ablation.")
        return []

    df = load_chalearn_baseline_data(csv_path)
    seed = config["runtime"]["seed"]

    results = []
    for group_name, feature_cols in FEATURE_GROUPS.items():
        splits = prepare_chalearn_splits(df, feature_columns=feature_cols)

        # --- Extraversion regression (best trait per Day 2 results) ---
        X_train, y_train = splits["train"]["X"], splits["train"]["extraversion"]
        X_test, y_test = splits["test"]["X"], splits["test"]["extraversion"]

        reg_model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        reg_model.fit(X_train, y_train)
        y_pred = reg_model.predict(X_test)
        reg_metrics = evaluate_regressor(y_test, y_pred)

        logger.info(
            f"[ChaLearn extraversion][{group_name}] "
            f"n_features={len(feature_cols)} R2={reg_metrics['r2']:.3f} MAE={reg_metrics['mae']:.4f}"
        )
        results.append(
            {
                "dataset": "chalearn",
                "task": "extraversion_regression",
                "feature_group": group_name,
                "n_features": len(feature_cols),
                "r2": reg_metrics["r2"],
                "mae": reg_metrics["mae"],
                "pearson_r": reg_metrics["pearson_r"],
            }
        )

        # --- Interview-invite classification ---
        X_train, y_train = splits["train"]["X"], splits["train"]["y_interview"]
        X_test, y_test = splits["test"]["X"], splits["test"]["y_interview"]

        clf_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
        clf_model.fit(X_train, y_train)
        y_pred = clf_model.predict(X_test)
        y_proba = clf_model.predict_proba(X_test)
        clf_metrics = evaluate_classifier(y_test, y_pred, y_proba, ["NotInvited", "Invited"])

        logger.info(
            f"[ChaLearn interview][{group_name}] "
            f"n_features={len(feature_cols)} acc={clf_metrics['accuracy']:.3f} f1_macro={clf_metrics['f1_macro']:.3f}"
        )
        results.append(
            {
                "dataset": "chalearn",
                "task": "interview_classification",
                "feature_group": group_name,
                "n_features": len(feature_cols),
                "accuracy": clf_metrics["accuracy"],
                "f1_macro": clf_metrics["f1_macro"],
                "roc_auc": clf_metrics.get("roc_auc"),
            }
        )

    return results


def main() -> None:
    config = load_config()
    logger = get_logger(__name__, log_dir=config["paths"]["logs_dir"], log_filename="run_ablation.log")

    all_results = []
    all_results += run_daisee_ablation(config, logger)
    all_results += run_chalearn_ablation(config, logger)

    if not all_results:
        logger.error("No ablation results produced — check that both summary CSVs and trained models exist.")
        sys.exit(1)

    results_df = pd.DataFrame(all_results)
    outputs_dir = Path(config["paths"]["outputs_dir"])
    results_path = outputs_dir / "ablation_results.csv"
    results_df.to_csv(results_path, index=False)

    logger.info("=" * 70)
    logger.info(f"Ablation results written to: {results_path}")
    logger.info("=" * 70)
    logger.info("\n" + results_df.to_string(index=False))


if __name__ == "__main__":
    main()
