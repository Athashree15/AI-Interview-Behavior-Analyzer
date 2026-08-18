"""
ChaLearn First Impressions V2 baseline dataset preparation.

Loads outputs/chalearn_visual_features_summary.csv (produced by
scripts/batch_process_chalearn.py) and prepares feature matrices for
baseline model training against the five continuous Big-Five traits
(regression) and the binary `interview` variable (classification).

Uses the `internal_split` column already in the summary CSV — our own
locked 80/10/10 split (seed 42) of the 6,000 officially-labeled
training clips, since val/test annotations are encrypted and
unavailable (see src/data/chalearn_manifest.py for the full rationale).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Identical feature set to DAiSEE (src/models/daisee_dataset.py) —
# kept as a separate constant rather than a shared import to avoid
# coupling the two dataset modules together for what is currently a
# one-line list; if this drifts out of sync between the two files,
# that's a signal to refactor into src/models/feature_columns.py.
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

TRAIT_COLUMNS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]


def load_chalearn_baseline_data(summary_csv_path: str | Path) -> pd.DataFrame:
    """Load the raw summary CSV, unmodified aside from basic dtype handling."""
    return pd.read_csv(summary_csv_path)


def prepare_chalearn_splits(df: pd.DataFrame, feature_columns: list[str] | None = None) -> dict[str, dict[str, np.ndarray]]:
    """
    Prepare train/validation/test feature matrices and targets using
    the dataset's `internal_split` column (our own locked 80/10/10
    split — see module docstring).

    Args:
        df: raw dataframe from load_chalearn_baseline_data().
        feature_columns: optional override of which feature columns to
            use (defaults to the full FEATURE_COLUMNS set) — used by
            the ablation study.

    Returns:
        Dict keyed by split name, each containing "X" (feature matrix),
        one array per Big-Five trait (keyed by trait name), "y_interview"
        (binary interview-invite target), and "video_id".
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    missing_cols = set(feature_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Summary CSV is missing expected feature columns: {missing_cols}")

    train_df = df[df["internal_split"] == "train"]
    medians = train_df[feature_columns].median()

    df = df.copy()
    df[feature_columns] = df[feature_columns].fillna(medians)

    # The official 'interview' variable is continuous (a probability-like
    # score in [0,1] in the original release) — binarize at 0.5 for a
    # clean classification task, consistent with how it's commonly used
    # in the personality-computing literature.
    df["y_interview"] = (df["interview"] >= 0.5).astype(int)

    splits: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ("train", "validation", "test"):
        split_df = df[df["internal_split"] == split_name]
        split_data = {
            "X": split_df[feature_columns].to_numpy(dtype=np.float64),
            "y_interview": split_df["y_interview"].to_numpy(dtype=np.int64),
            "video_id": split_df["video_id"].to_numpy(),
        }
        for trait in TRAIT_COLUMNS:
            split_data[trait] = split_df[trait].to_numpy(dtype=np.float64)
        splits[split_name] = split_data

    return splits
