"""
Phase 1 — DAiSEE dataset inspection.

Run this after filling in `datasets.daisee.root` in configs/config.yaml.

What this script does (and why):
  1. Recursively finds all video files under the DAiSEE root.
  2. Locates and loads the label CSV(s) (Train/Validation/Test).
  3. Cross-checks: every labeled clip has a matching video file, and
     vice versa — flags mismatches instead of silently ignoring them.
  4. Extracts participant/subject IDs from filenames (DAiSEE encodes
     subject ID as a prefix) and reports how many unique subjects
     exist per split — this is what lets us verify subject-disjoint
     splits later (Module 19: no subject in both train and test).
  5. Reports label distributions for Boredom / Engagement / Confusion /
     Frustration (0-3 ordinal scale) — flags class imbalance now,
     before it becomes a training-time surprise.
  6. Probes a random sample of videos (see `inspection.sample_videos_per_split`
     in config) for fps / resolution / duration / corruption, since
     decoding every one of ~9,000 clips just for a feasibility check
     is unnecessary and slow.

Usage:
    python scripts/inspect_daisee.py

Output:
    - Console summary (the important part — read this first)
    - outputs/daisee_inventory.csv (full per-file inventory, if enabled in config)
    - logs/daisee_inspection.log
"""

from __future__ import annotations

import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Allow running as `python scripts/inspect_daisee.py` from repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config_loader import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.video_utils import probe_video  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
LABEL_COLUMNS_EXPECTED = {"Boredom", "Engagement", "Confusion", "Frustration"}


def find_video_files(root: Path) -> list[Path]:
    """Recursively find all video files under `root`."""
    return [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]


def find_label_files(root: Path) -> list[Path]:
    """Find likely DAiSEE label CSV files (names vary slightly across releases)."""
    candidates = list(root.rglob("*abel*.csv")) + list(root.rglob("*Labels*.csv"))
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c.resolve() not in seen:
            seen.add(c.resolve())
            unique.append(c)
    return unique


def extract_subject_id(filename: str) -> str | None:
    """
    DAiSEE clip filenames typically encode subject ID as a numeric
    prefix, e.g. '1100011002.avi' -> subject '110001'.
    This regex is intentionally permissive; the script reports how
    many filenames it FAILED to parse so we can inspect edge cases
    manually rather than silently mis-grouping subjects.
    """
    match = re.match(r"^(\d{6})", filename)
    return match.group(1) if match else None


def infer_split_from_path(path: Path) -> str:
    """Infer Train/Validation/Test split from the folder path."""
    parts_lower = [p.lower() for p in path.parts]
    for split in ("train", "validation", "test"):
        if split in parts_lower:
            return split
    return "unknown"


def main() -> None:
    config = load_config()
    logger = get_logger(__name__, log_dir=config["paths"]["logs_dir"], log_filename="daisee_inspection.log")

    daisee_root = Path(config["datasets"]["daisee"]["root"])
    if not daisee_root.exists() or "PATH/TO" in str(daisee_root):
        logger.error(
            f"DAiSEE root path is not set or does not exist: {daisee_root}\n"
            "Edit configs/config.yaml -> datasets.daisee.root before running this script."
        )
        sys.exit(1)

    logger.info(f"Inspecting DAiSEE at: {daisee_root}")

    # ---------------- 1. Find videos ----------------
    videos = find_video_files(daisee_root)
    logger.info(f"Found {len(videos)} video files.")
    if len(videos) == 0:
        logger.error("No video files found. Check the folder path and structure, then re-run.")
        sys.exit(1)

    # ---------------- 2. Find and load label files ----------------
    label_files = find_label_files(daisee_root)
    logger.info(f"Found {len(label_files)} label CSV file(s): {[str(f) for f in label_files]}")

    if len(label_files) == 0:
        logger.error("No label CSVs found. Verify the DAiSEE 'Labels' folder is present and re-run.")
        sys.exit(1)

    all_labels = []
    for lf in label_files:
        try:
            df = pd.read_csv(lf)
            df.columns = [c.strip() for c in df.columns]
            df["__source_file"] = str(lf)
            df["__split"] = infer_split_from_path(lf)
            all_labels.append(df)
            logger.info(f"Loaded {lf.name}: {len(df)} rows, columns={list(df.columns)}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to read {lf}: {exc}")

    if not all_labels:
        logger.error("Could not load any label file successfully.")
        sys.exit(1)

    labels_df = pd.concat(all_labels, ignore_index=True)

    missing_label_cols = LABEL_COLUMNS_EXPECTED - set(labels_df.columns)
    if missing_label_cols:
        logger.warning(
            f"Expected label columns not found: {missing_label_cols}. "
            f"Actual columns present: {list(labels_df.columns)}. "
            "DAiSEE releases have varied column naming — verify manually against the CSV."
        )

    # ---------------- 3. Cross-check videos vs labels ----------------
    video_filenames = {v.name for v in videos}
    clip_id_col = next((c for c in labels_df.columns if "clip" in c.lower() or "name" in c.lower()), None)

    if clip_id_col is None:
        logger.warning(
            "Could not confidently identify the clip-identifier column in the label file. "
            f"Columns available: {list(labels_df.columns)}. Skipping video<->label cross-check."
        )
    else:
        labeled_filenames = set(labels_df[clip_id_col].astype(str))
        videos_without_labels = video_filenames - labeled_filenames
        labels_without_videos = labeled_filenames - video_filenames
        logger.info(f"Clip identifier column used for cross-check: '{clip_id_col}'")
        logger.info(f"Videos with NO matching label row: {len(videos_without_labels)}")
        logger.info(f"Label rows with NO matching video file: {len(labels_without_videos)}")
        if videos_without_labels:
            logger.warning(f"Example unmatched videos (up to 5): {list(videos_without_labels)[:5]}")
        if labels_without_videos:
            logger.warning(f"Example unmatched label rows (up to 5): {list(labels_without_videos)[:5]}")

    # ---------------- 4. Subject ID extraction & split overlap check ----------------
    subject_by_split: dict[str, set[str]] = defaultdict(set)
    unparsed_filenames = []

    for v in videos:
        split = infer_split_from_path(v)
        subj = extract_subject_id(v.name)
        if subj:
            subject_by_split[split].add(subj)
        else:
            unparsed_filenames.append(v.name)

    logger.info("Unique subjects per split (from filename parsing):")
    for split, subjects in subject_by_split.items():
        logger.info(f"  {split}: {len(subjects)} unique subjects")

    # THE critical leakage check
    splits = list(subject_by_split.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = subject_by_split[splits[i]] & subject_by_split[splits[j]]
            if overlap:
                logger.warning(
                    f"SUBJECT OVERLAP between '{splits[i]}' and '{splits[j]}': "
                    f"{len(overlap)} shared subject IDs. Example: {list(overlap)[:5]}. "
                    "If this is the official DAiSEE split, this is expected to be EMPTY — "
                    "investigate before trusting any train/test separation."
                )
            else:
                logger.info(f"No subject overlap between '{splits[i]}' and '{splits[j]}' — good.")

    if unparsed_filenames:
        logger.warning(
            f"{len(unparsed_filenames)} filenames did not match the expected subject-ID pattern. "
            f"Examples: {unparsed_filenames[:5]}. Inspect the DAiSEE filename convention manually."
        )

    # ---------------- 5. Label distribution ----------------
    logger.info("Label distributions:")
    for col in LABEL_COLUMNS_EXPECTED & set(labels_df.columns):
        counts = Counter(labels_df[col].dropna().astype(int))
        total = sum(counts.values())
        dist_str = ", ".join(f"{k}: {v} ({v / total:.1%})" for k, v in sorted(counts.items()))
        logger.info(f"  {col}: {dist_str}")

    # ---------------- 6. Sample video health probe ----------------
    sample_n = config["inspection"]["sample_videos_per_split"]
    logger.info(f"Probing a random sample of up to {sample_n} videos per split for fps/resolution/duration...")

    videos_by_split: dict[str, list[Path]] = defaultdict(list)
    for v in videos:
        videos_by_split[infer_split_from_path(v)].append(v)

    probe_rows = []
    for split, split_videos in videos_by_split.items():
        sample = random.sample(split_videos, min(sample_n, len(split_videos)))
        corrupt_count = 0
        for v in sample:
            result = probe_video(v)
            probe_rows.append(
                {
                    "split": split,
                    "path": result.path,
                    "readable": result.readable,
                    "fps": result.fps,
                    "duration_sec": result.duration_sec,
                    "width": result.width,
                    "height": result.height,
                    "error": result.error,
                }
            )
            if not result.readable:
                corrupt_count += 1
        logger.info(f"  Split '{split}': probed {len(sample)} videos, {corrupt_count} unreadable/corrupt.")

    # ---------------- 7. Write full inventory CSV ----------------
    if config["inspection"]["write_inventory_csv"]:
        outputs_dir = Path(config["paths"]["outputs_dir"])
        outputs_dir.mkdir(parents=True, exist_ok=True)
        inventory_path = outputs_dir / "daisee_inventory.csv"

        inventory_rows = [
            {"video_path": str(v), "split": infer_split_from_path(v), "subject_id": extract_subject_id(v.name)}
            for v in videos
        ]
        pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
        logger.info(f"Full video inventory written to: {inventory_path}")

        probe_path = outputs_dir / "daisee_video_health_sample.csv"
        pd.DataFrame(probe_rows).to_csv(probe_path, index=False)
        logger.info(f"Video health probe sample written to: {probe_path}")

    logger.info("DAiSEE inspection complete. Review warnings above before proceeding to Phase 2.")


if __name__ == "__main__":
    main()
