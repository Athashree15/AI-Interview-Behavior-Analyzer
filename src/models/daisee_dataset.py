"""
DAiSEE baseline dataset preparation.

Loads outputs/daisee_visual_features_summary.csv (produced by
scripts/batch_process_daisee.py) and prepares feature matrices for
baseline model training.

Design decisions (and why):
  - Uses the `split` column already in the summary CSV, which comes
    straight from the subject-disjoint DAiSEE train/validation/test
    folders confirmed leak-free during Phase 1. We do NOT re-split
    here — reusing the official, verified split is what makes the
    resulting metrics trustworthy.
  - Provides BOTH the full 4-class Engagement target and a binary
    collapse (low=0,1 vs high=2,3), per the design decision made
    after seeing the real class distribution (94.2% of labels are
    classes 2-3; classes 0-1 total 1,017 samples across the entire
    dataset). The 4-class result is still reported, but flagged as
    unreliable for the minority classes rather than hidden.
  - Missing values (NaN) can occur in emotion columns for a frame
    where the emotion classifier failed on every sampled frame of a
    clip — imputed with the TRAINING set's median only, to avoid
    leaking validation/test statistics into imputation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "face_detection_rate",
    "overall_eye_contact_ratio",
    "head_yaw_stability_std_deg",
    "head_pitch_stability_std_deg",
    "multiple_faces_detected_frame_count",
    "emotion_neutral",
    "emotion_happy",
    "emotion_sad",
    "emotion_angry",
    "emotion_fear",
    "emotion_surprise",
    "emotion_disgust",
]


def load_daisee_baseline_data(summary_csv_path: str | Path) -> pd.DataFrame:
    """Load the raw summary CSV, unmodified aside from basic dtype handling."""
    df = pd.read_csv(summary_csv_path)
    return df


def prepare_daisee_splits(
    df: pd.DataFrame, target_column: str = "Engagement", feature_columns: list[str] | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """
    Prepare train/validation/test feature matrices and targets, using
    the dataset's existing subject-disjoint `split` column.

    Args:
        df: raw dataframe from load_daisee_baseline_data().
        target_column: one of "Boredom", "Engagement", "Confusion", "Frustration".
        feature_columns: optional override of which feature columns to
            use (defaults to the full FEATURE_COLUMNS set). Used by the
            ablation study to train on feature subsets without touching
            this function's default behavior for normal training.

    Returns:
        Dict keyed by split name ("train"/"validation"/"test"), each
        containing "X" (feature matrix), "y" (4-class target), and
        "y_binary" (collapsed low/high target — only meaningful for
        Engagement; for the other three, low=0 vs high>=1 is used as
        a reasonable generic collapse, but Engagement's specific
        0,1-vs-2,3 collapse is what's used in this project per the
        design decision above).
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    missing_cols = set(feature_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Summary CSV is missing expected feature columns: {missing_cols}")

    # Impute NaNs using TRAIN-split medians only (no leakage from val/test).
    train_df = df[df["split"] == "train"]
    medians = train_df[feature_columns].median()

    df = df.copy()
    df[feature_columns] = df[feature_columns].fillna(medians)

    if target_column == "Engagement":
        df["y_binary"] = df[target_column].apply(lambda v: 0 if v in (0, 1) else 1)
    else:
        df["y_binary"] = df[target_column].apply(lambda v: 0 if v == 0 else 1)

    splits: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ("train", "validation", "test"):
        split_df = df[df["split"] == split_name]
        splits[split_name] = {
            "X": split_df[feature_columns].to_numpy(dtype=np.float64),
            "y": split_df[target_column].to_numpy(dtype=np.int64),
            "y_binary": split_df["y_binary"].to_numpy(dtype=np.int64),
            "clip_id": split_df["clip_id"].to_numpy(),
        }

    return splits
