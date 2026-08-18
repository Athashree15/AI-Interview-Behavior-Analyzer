"""
Phase 5 (Day 1, 5-day scope) — DAiSEE baseline model training.

Trains and evaluates the baseline progression required by Module 27:
    Majority Class -> Logistic Regression -> Random Forest -> XGBoost

...on the Engagement target, BOTH as the full 4-class task and as a
binary collapse (low=0,1 vs high=2,3), per the class-imbalance design
decision (see src/models/daisee_dataset.py docstring).

All splits (train/validation/test) come from DAiSEE's own
subject-disjoint folder structure, confirmed leak-free in Phase 1 —
NOT re-split here.

Class imbalance handling: `class_weight="balanced"` for Logistic
Regression and Random Forest; XGBoost uses computed sample weights
(XGBoost's multiclass mode has no direct class_weight parameter).

Usage:
    python training/train_daisee_baseline.py

Output:
    - outputs/daisee_baseline_results.csv          (all models x both tasks x both splits, every metric)
    - outputs/plots/daisee_confusion_*.png          (confusion matrices)
    - models/trained/daisee_*.joblib                (trained model artifacts, for the Streamlit app later)
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.daisee_dataset import load_daisee_baseline_data, prepare_daisee_splits  # noqa: E402
from src.models.evaluation import evaluate_classifier, save_confusion_matrix_plot  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


ENGAGEMENT_4CLASS_LABELS = ["0-VeryLow", "1-Low", "2-High", "3-VeryHigh"]
ENGAGEMENT_BINARY_LABELS = ["Low", "High"]


def build_models(task: str, seed: int) -> dict[str, object]:
    """Instantiate the baseline model progression. `task` is 'multiclass' or 'binary'."""
    models: dict[str, object] = {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed, multi_class="auto"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
    }
    if XGBOOST_AVAILABLE:
        if task == "binary":
            models["xgboost"] = XGBClassifier(
                n_estimators=300, random_state=seed, eval_metric="logloss", n_jobs=-1
            )
        else:
            models["xgboost"] = XGBClassifier(
                n_estimators=300, random_state=seed, eval_metric="mlogloss", n_jobs=-1
            )
    return models


def run_task(
    task_name: str,
    y_key: str,
    class_labels: list[str],
    splits: dict[str, dict[str, np.ndarray]],
    seed: int,
    logger,
    outputs_dir: Path,
    models_dir: Path,
) -> list[dict]:
    """Train + evaluate every baseline model for one task (multiclass or binary). Returns a list of result rows."""
    logger.info(f"--- Task: {task_name} ---")

    X_train, y_train = splits["train"]["X"], splits["train"][y_key]
    X_val, y_val = splits["validation"]["X"], splits["validation"][y_key]
    X_test, y_test = splits["test"]["X"], splits["test"][y_key]

    logger.info(f"Train/Val/Test sizes: {len(y_train)}/{len(y_val)}/{len(y_test)}")
    logger.info(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Scale features for Logistic Regression (tree models don't need this,
    # but scaling doesn't hurt them either — fit scaler on TRAIN only).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    task_type = "binary" if y_key == "y_binary" else "multiclass"
    models = build_models(task_type, seed)

    results = []

    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")

        # Logistic Regression uses scaled features; tree-based models use raw features.
        use_scaled = model_name == "logistic_regression"
        X_tr = X_train_scaled if use_scaled else X_train
        X_v = X_val_scaled if use_scaled else X_val
        X_te = X_test_scaled if use_scaled else X_test

        if model_name == "xgboost":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(X_tr, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_tr, y_train)

        model_path = models_dir / f"daisee_{task_name}_{model_name}.joblib"
        joblib.dump({"model": model, "scaler": scaler if use_scaled else None}, model_path)

        for split_name, X_split, y_split in [("validation", X_v, y_val), ("test", X_te, y_test)]:
            y_pred = model.predict(X_split)
            y_proba = model.predict_proba(X_split) if hasattr(model, "predict_proba") else None

            metrics = evaluate_classifier(y_split, y_pred, y_proba, class_labels)

            plot_path = outputs_dir / "plots" / f"daisee_confusion_{task_name}_{model_name}_{split_name}.png"
            save_confusion_matrix_plot(
                y_split, y_pred, class_labels, f"{model_name} ({task_name}, {split_name})", plot_path
            )

            logger.info(
                f"  [{split_name}] {model_name}: acc={metrics['accuracy']:.3f} "
                f"f1_macro={metrics['f1_macro']:.3f} f1_weighted={metrics['f1_weighted']:.3f}"
            )

            results.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "split": split_name,
                    "accuracy": metrics["accuracy"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "f1_macro": metrics["f1_macro"],
                    "f1_weighted": metrics["f1_weighted"],
                    "roc_auc": metrics.get("roc_auc") or metrics.get("roc_auc_ovr_macro"),
                    "per_class_f1": metrics["per_class_f1"],
                    "confusion_matrix": metrics["confusion_matrix"],
                }
            )

    return results


def main() -> None:
    config = load_config()
    logger = get_logger(__name__, log_dir=config["paths"]["logs_dir"], log_filename="train_daisee_baseline.log")

    if not XGBOOST_AVAILABLE:
        logger.warning("xgboost is not installed — skipping XGBoost, will only train majority/LR/RF. `pip install xgboost` to include it.")

    outputs_dir = Path(config["paths"]["outputs_dir"])
    (outputs_dir / "plots").mkdir(parents=True, exist_ok=True)
    models_dir = Path("models/trained")
    models_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = outputs_dir / "daisee_visual_features_summary.csv"
    if not summary_csv_path.exists():
        logger.error(f"{summary_csv_path} not found — run scripts/batch_process_daisee.py (full run) first.")
        sys.exit(1)

    logger.info(f"Loading {summary_csv_path}...")
    df = load_daisee_baseline_data(summary_csv_path)
    logger.info(f"Loaded {len(df)} rows.")

    splits = prepare_daisee_splits(df, target_column="Engagement")

    seed = config["runtime"]["seed"]

    all_results = []
    all_results += run_task(
        "engagement_4class", "y", ENGAGEMENT_4CLASS_LABELS, splits, seed, logger, outputs_dir, models_dir
    )
    all_results += run_task(
        "engagement_binary", "y_binary", ENGAGEMENT_BINARY_LABELS, splits, seed, logger, outputs_dir, models_dir
    )

    results_df = pd.DataFrame(all_results)
    results_csv_path = outputs_dir / "daisee_baseline_results.csv"
    results_df.to_csv(results_csv_path, index=False)

    logger.info("=" * 70)
    logger.info(f"All results written to: {results_csv_path}")
    logger.info(f"Confusion matrix plots written to: {outputs_dir / 'plots'}")
    logger.info(f"Trained models written to: {models_dir}")
    logger.info("=" * 70)
    logger.info("\nTest-set summary (accuracy / f1_macro):")
    test_only = results_df[results_df["split"] == "test"][["task", "model", "accuracy", "f1_macro", "f1_weighted"]]
    logger.info("\n" + test_only.to_string(index=False))


if __name__ == "__main__":
    main()
