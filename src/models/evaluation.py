"""Shared evaluation utilities for classification baseline models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — no display available, just save to file
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None, class_labels: list[str]
) -> dict[str, Any]:
    """
    Compute the standard metric set (Module 26): accuracy, precision,
    recall, F1 (macro + weighted), confusion matrix, and ROC-AUC where
    applicable (binary case, or multiclass one-vs-rest if y_proba given).

    All metrics use `zero_division=0` explicitly rather than letting
    sklearn silently warn/guess — a minority class with zero predicted
    samples correctly reports 0, not a warning we might miss in logs.
    """
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # Per-class F1, so minority-class performance is never hidden behind
    # a macro average alone.
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=sorted(set(y_true) | set(y_pred)))
    metrics["per_class_f1"] = {
        str(label): float(score)
        for label, score in zip(sorted(set(y_true) | set(y_pred)), per_class_f1)
    }

    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except ValueError as exc:
            # Happens if a class present in y_proba's columns never appears
            # in y_true for this split — report the failure, don't crash.
            metrics["roc_auc_error"] = str(exc)

    return metrics


def save_confusion_matrix_plot(
    y_true: np.ndarray, y_pred: np.ndarray, class_labels: list[str], title: str, output_path: str | Path
) -> None:
    """Save a confusion matrix plot to disk (Module 17 visualization requirement)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """
    Compute the standard regression metric set (Module 26): MAE, MSE,
    RMSE, R^2, and Pearson correlation. Used for ChaLearn's continuous
    Big-Five trait targets.
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # Pearson correlation is undefined if predictions have zero variance
    # (e.g. a degenerate model predicting a constant) — handle explicitly
    # rather than letting scipy emit a warning we might not notice in logs.
    if np.std(y_pred) < 1e-8:
        pearson_r, pearson_p = float("nan"), float("nan")
    else:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        pearson_r, pearson_p = float(pearson_r), float(pearson_p)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "pearson_r": pearson_r, "pearson_p": pearson_p}
