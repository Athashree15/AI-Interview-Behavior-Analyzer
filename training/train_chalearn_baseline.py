"""
Phase 5 (Day 2, 5-day scope) — ChaLearn First Impressions V2 baseline
model training.

Two task types from the same visual features:
  1. REGRESSION — five Big-Five personality traits (continuous, ~[0,1]).
     Baseline progression (Module 27): Mean baseline -> Linear Regression
     -> Random Forest Regressor -> XGBoost Regressor.
  2. CLASSIFICATION — binary interview-invite (thresholded at 0.5).
     Same progression as DAiSEE: Majority Class -> Logistic Regression
     -> Random Forest -> XGBoost.

Uses the internal_split column (our own locked 80/10/10 split) — NOT
the official ChaLearn folders, since val/test annotations are
encrypted (documented limitation, see src/data/chalearn_manifest.py).

Usage:
    python training/train_chalearn_baseline.py

Output:
    - outputs/chalearn_baseline_regression_results.csv
    - outputs/chalearn_baseline_interview_results.csv
    - outputs/plots/chalearn_confusion_interview_*.png
    - models/trained/chalearn_*.joblib
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.chalearn_dataset import (  # noqa: E402
    TRAIT_COLUMNS,
    load_chalearn_baseline_data,
    prepare_chalearn_splits,
)
from src.models.evaluation import evaluate_classifier, evaluate_regressor, save_confusion_matrix_plot  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

INTERVIEW_LABELS = ["NotInvited", "Invited"]


def run_regression_task(
    trait: str, splits: dict, seed: int, logger, outputs_dir: Path, models_dir: Path
) -> list[dict]:
    """Train + evaluate the regression baseline progression for one Big-Five trait."""
    X_train, y_train = splits["train"]["X"], splits["train"][trait]
    X_val, y_val = splits["validation"]["X"], splits["validation"][trait]
    X_test, y_test = splits["test"]["X"], splits["test"][trait]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    models: dict[str, object] = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBRegressor(n_estimators=300, random_state=seed, n_jobs=-1)

    results = []
    for model_name, model in models.items():
        use_scaled = model_name == "linear_regression"
        X_tr = X_train_scaled if use_scaled else X_train
        X_v = X_val_scaled if use_scaled else X_val
        X_te = X_test_scaled if use_scaled else X_test

        model.fit(X_tr, y_train)

        model_path = models_dir / f"chalearn_{trait}_{model_name}.joblib"
        joblib.dump({"model": model, "scaler": scaler if use_scaled else None}, model_path)

        for split_name, X_split, y_split in [("validation", X_v, y_val), ("test", X_te, y_test)]:
            y_pred = model.predict(X_split)
            metrics = evaluate_regressor(y_split, y_pred)
            logger.info(
                f"  [{trait}][{split_name}] {model_name}: MAE={metrics['mae']:.4f} "
                f"R2={metrics['r2']:.4f} Pearson_r={metrics['pearson_r']:.4f}"
            )
            results.append({"trait": trait, "model": model_name, "split": split_name, **metrics})

    return results


def run_interview_classification(splits: dict, seed: int, logger, outputs_dir: Path, models_dir: Path) -> list[dict]:
    """Train + evaluate the classification baseline progression for the binary interview-invite target."""
    X_train, y_train = splits["train"]["X"], splits["train"]["y_interview"]
    X_val, y_val = splits["validation"]["X"], splits["validation"]["y_interview"]
    X_test, y_test = splits["test"]["X"], splits["test"]["y_interview"]

    logger.info(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    models: dict[str, object] = {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(n_estimators=300, random_state=seed, eval_metric="logloss", n_jobs=-1)

    results = []
    for model_name, model in models.items():
        use_scaled = model_name == "logistic_regression"
        X_tr = X_train_scaled if use_scaled else X_train
        X_v = X_val_scaled if use_scaled else X_val
        X_te = X_test_scaled if use_scaled else X_test

        if model_name == "xgboost":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(X_tr, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_tr, y_train)

        model_path = models_dir / f"chalearn_interview_{model_name}.joblib"
        joblib.dump({"model": model, "scaler": scaler if use_scaled else None}, model_path)

        for split_name, X_split, y_split in [("validation", X_v, y_val), ("test", X_te, y_test)]:
            y_pred = model.predict(X_split)
            y_proba = model.predict_proba(X_split) if hasattr(model, "predict_proba") else None
            metrics = evaluate_classifier(y_split, y_pred, y_proba, INTERVIEW_LABELS)

            plot_path = outputs_dir / "plots" / f"chalearn_confusion_interview_{model_name}_{split_name}.png"
            save_confusion_matrix_plot(y_split, y_pred, INTERVIEW_LABELS, f"{model_name} (interview, {split_name})", plot_path)

            logger.info(
                f"  [interview][{split_name}] {model_name}: acc={metrics['accuracy']:.3f} "
                f"f1_macro={metrics['f1_macro']:.3f}"
            )
            results.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "accuracy": metrics["accuracy"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "f1_macro": metrics["f1_macro"],
                    "f1_weighted": metrics["f1_weighted"],
                    "roc_auc": metrics.get("roc_auc") or metrics.get("roc_auc_ovr_macro"),
                }
            )

    return results


def main() -> None:
    config = load_config()
    logger = get_logger(__name__, log_dir=config["paths"]["logs_dir"], log_filename="train_chalearn_baseline.log")

    if not XGBOOST_AVAILABLE:
        logger.warning("xgboost is not installed — skipping XGBoost variants.")

    outputs_dir = Path(config["paths"]["outputs_dir"])
    (outputs_dir / "plots").mkdir(parents=True, exist_ok=True)
    models_dir = Path("models/trained")
    models_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = outputs_dir / "chalearn_visual_features_summary.csv"
    if not summary_csv_path.exists():
        logger.error(f"{summary_csv_path} not found — run scripts/batch_process_chalearn.py (full run) first.")
        sys.exit(1)

    logger.info(f"Loading {summary_csv_path}...")
    df = load_chalearn_baseline_data(summary_csv_path)
    logger.info(f"Loaded {len(df)} rows.")

    splits = prepare_chalearn_splits(df)
    seed = config["runtime"]["seed"]

    logger.info("=== Regression: Big-Five traits ===")
    regression_results = []
    for trait in TRAIT_COLUMNS:
        logger.info(f"--- Trait: {trait} ---")
        regression_results += run_regression_task(trait, splits, seed, logger, outputs_dir, models_dir)

    regression_df = pd.DataFrame(regression_results)
    regression_csv_path = outputs_dir / "chalearn_baseline_regression_results.csv"
    regression_df.to_csv(regression_csv_path, index=False)

    logger.info("=== Classification: interview-invite ===")
    interview_results = run_interview_classification(splits, seed, logger, outputs_dir, models_dir)
    interview_df = pd.DataFrame(interview_results)
    interview_csv_path = outputs_dir / "chalearn_baseline_interview_results.csv"
    interview_df.to_csv(interview_csv_path, index=False)

    logger.info("=" * 70)
    logger.info(f"Regression results written to: {regression_csv_path}")
    logger.info(f"Interview classification results written to: {interview_csv_path}")
    logger.info("=" * 70)

    logger.info("\nRegression test-set summary (R2 / Pearson r):")
    reg_test = regression_df[regression_df["split"] == "test"][["trait", "model", "r2", "pearson_r", "mae"]]
    logger.info("\n" + reg_test.to_string(index=False))

    logger.info("\nInterview classification test-set summary:")
    interview_test = interview_df[interview_df["split"] == "test"][["model", "accuracy", "f1_macro"]]
    logger.info("\n" + interview_test.to_string(index=False))


if __name__ == "__main__":
    main()
