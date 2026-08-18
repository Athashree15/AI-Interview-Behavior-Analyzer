"""
Phase 1 — ChaLearn First Impressions V2 dataset inspection.

Run this after filling in `datasets.chalearn_fi_v2.root` in configs/config.yaml.

What this script does (and why):
  1. Recursively finds all video files and groups them by split
     (train / validation / test), inferred from folder names.
  2. Locates annotation files. The official release ships these as
     Python-pickled dicts (often pickled under Python 2, hence the
     'latin1' encoding fallback below) — but Kaggle re-uploads vary,
     so this script also tries .csv/.json as a fallback and reports
     exactly which format it found.
  3. Validates the annotation content: expects five continuous
     Big-Five trait scores per clip (Extraversion, Agreeableness,
     Conscientiousness, Neuroticism, Openness) roughly in [0, 1],
     plus an 'interview' variable if present. Reports actual value
     ranges rather than assuming — if your re-upload doesn't match
     this, this script will tell you exactly how it differs instead
     of silently proceeding.
  4. Cross-checks videos against annotation keys (missing/orphaned entries).
  5. Checks for a transcription folder/file and reports whether it's usable.
  6. Probes a random sample of videos for fps/resolution/duration/corruption.

Usage:
    python scripts/inspect_chalearn.py

Output:
    - Console summary (read this first)
    - outputs/chalearn_inventory.csv
    - logs/chalearn_inspection.log
"""

from __future__ import annotations

import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config_loader import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.video_utils import probe_video  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
EXPECTED_TRAITS = {"extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"}


def find_video_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]


def infer_split_from_path(path: Path) -> str:
    parts_lower = [p.lower() for p in path.parts]
    for split in ("train", "training", "val", "validation", "test"):
        for part in parts_lower:
            if split in part:
                return "validation" if split in ("val", "validation") else ("train" if split == "training" else split)
    return "unknown"


def load_annotation_file(path: Path) -> dict | None:
    """
    Attempt to load an annotation file trying, in order: pickle
    (latin1-encoded, matching the official CVPR'17 release), plain
    pickle, JSON, then CSV. Returns None if nothing works, with the
    caller responsible for logging the failure.
    """
    # Try pickle with latin1 encoding (handles Python2-pickled files)
    try:
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        return {"format": "pickle-latin1", "data": data}
    except Exception:  # noqa: BLE001
        pass

    # Try plain pickle
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return {"format": "pickle", "data": data}
    except Exception:  # noqa: BLE001
        pass

    # Try JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"format": "json", "data": data}
    except Exception:  # noqa: BLE001
        pass

    # Try CSV
    try:
        data = pd.read_csv(path)
        return {"format": "csv", "data": data}
    except Exception:  # noqa: BLE001
        pass

    return None


def summarize_pickle_annotation(data: dict, logger) -> None:
    """
    The official FI-V2 annotation pickle is typically a dict keyed by
    trait name -> {video_filename: score}, e.g.
        {'extraversion': {'abc.mp4': 0.62, ...}, 'interview': {...}, ...}
    This function detects that shape and reports it; if the shape is
    different (common in re-uploads), it reports the actual structure
    instead of guessing.
    """
    if not isinstance(data, dict):
        logger.warning(f"Annotation pickle root is not a dict (type={type(data)}). Inspect manually.")
        return

    top_keys = set(k.lower() for k in data.keys())
    logger.info(f"Annotation top-level keys: {sorted(data.keys())}")

    found_traits = EXPECTED_TRAITS & top_keys
    missing_traits = EXPECTED_TRAITS - top_keys
    logger.info(f"Big-Five traits found: {sorted(found_traits) if found_traits else 'NONE'}")
    if missing_traits:
        logger.warning(f"Big-Five traits NOT found under expected names: {sorted(missing_traits)}")

    if "interview" in top_keys:
        logger.info("'interview' (invite-to-interview) variable IS present — usable for your interview-invite objective.")
    else:
        logger.warning("'interview' variable NOT found in this annotation file — check if it's a separate file.")

    # If we found at least one trait dict, report value range + sample size
    for trait_key in data:
        if trait_key.lower() in EXPECTED_TRAITS and isinstance(data[trait_key], dict):
            values = list(data[trait_key].values())
            if values:
                logger.info(
                    f"  '{trait_key}': {len(values)} labeled clips, "
                    f"min={min(values):.4f}, max={max(values):.4f}, mean={sum(values)/len(values):.4f}"
                )


def main() -> None:
    config = load_config()
    logger = get_logger(__name__, log_dir=config["paths"]["logs_dir"], log_filename="chalearn_inspection.log")

    root = Path(config["datasets"]["chalearn_fi_v2"]["root"])
    if not root.exists() or "PATH/TO" in str(root):
        logger.error(
            f"ChaLearn FI-V2 root path is not set or does not exist: {root}\n"
            "Edit configs/config.yaml -> datasets.chalearn_fi_v2.root before running this script."
        )
        sys.exit(1)

    logger.info(f"Inspecting ChaLearn First Impressions V2 at: {root}")

    # ---------------- 1. Find videos ----------------
    videos = find_video_files(root)
    logger.info(f"Found {len(videos)} video files.")
    if len(videos) == 0:
        logger.error("No video files found. Check the folder path/extraction, then re-run.")
        sys.exit(1)

    videos_by_split: dict[str, list[Path]] = defaultdict(list)
    for v in videos:
        videos_by_split[infer_split_from_path(v)].append(v)

    logger.info("Videos per split:")
    for split, vids in videos_by_split.items():
        logger.info(f"  {split}: {len(vids)} videos")

    # ---------------- 2. Find annotation files ----------------
    annotation_candidates = (
        list(root.rglob("*annotation*"))
        + list(root.rglob("*Annotation*"))
        + list(root.rglob("*.pkl"))
    )
    annotation_candidates = [p for p in annotation_candidates if p.is_file()]
    # de-dupe
    annotation_candidates = list({p.resolve(): p for p in annotation_candidates}.values())

    logger.info(f"Found {len(annotation_candidates)} candidate annotation file(s).")

    all_labeled_filenames: set[str] = set()
    any_annotation_loaded = False

    for ann_path in annotation_candidates:
        logger.info(f"--- Inspecting annotation file: {ann_path.name} ---")
        loaded = load_annotation_file(ann_path)
        if loaded is None:
            logger.warning(f"Could not parse {ann_path.name} as pickle/json/csv. Skipping.")
            continue

        any_annotation_loaded = True
        logger.info(f"Successfully parsed as: {loaded['format']}")

        if loaded["format"] in ("pickle-latin1", "pickle"):
            summarize_pickle_annotation(loaded["data"], logger)
            if isinstance(loaded["data"], dict):
                for trait_key, trait_dict in loaded["data"].items():
                    if isinstance(trait_dict, dict):
                        all_labeled_filenames.update(trait_dict.keys())
        elif loaded["format"] == "csv":
            df = loaded["data"]
            df.columns = [c.strip().lower() for c in df.columns]
            logger.info(f"CSV columns: {list(df.columns)}")
            found = EXPECTED_TRAITS & set(df.columns)
            logger.info(f"Big-Five traits found as CSV columns: {sorted(found) if found else 'NONE'}")
            id_col = next((c for c in df.columns if "video" in c or "file" in c or "name" in c), None)
            if id_col:
                all_labeled_filenames.update(df[id_col].astype(str))
        elif loaded["format"] == "json":
            data = loaded["data"]
            if isinstance(data, dict):
                logger.info(f"JSON top-level keys (sample up to 10): {list(data.keys())[:10]}")

    if not any_annotation_loaded:
        logger.error(
            "No annotation file could be parsed successfully. "
            "This is a BLOCKER — without labels, no supervised training is possible on this dataset. "
            "Manually open the annotation file to determine its actual format before proceeding."
        )

    # ---------------- 3. Cross-check videos vs annotation keys ----------------
    if all_labeled_filenames:
        video_filenames = {v.name for v in videos}
        videos_without_labels = video_filenames - all_labeled_filenames
        labels_without_videos = all_labeled_filenames - video_filenames
        logger.info(f"Videos with NO matching annotation entry: {len(videos_without_labels)}")
        logger.info(f"Annotation entries with NO matching video file: {len(labels_without_videos)}")
        if videos_without_labels:
            logger.warning(f"Example unmatched videos (up to 5): {list(videos_without_labels)[:5]}")

    # ---------------- 4. Transcription check ----------------
    transcription_candidates = list(root.rglob("*transcription*")) + list(root.rglob("*Transcription*"))
    transcription_files = [p for p in transcription_candidates if p.is_file()]
    logger.info(f"Found {len(transcription_files)} transcription-related file(s).")
    if transcription_files:
        logger.info(f"Example: {transcription_files[0]}")
    else:
        logger.warning(
            "No transcription files found. This is NOT a blocker — Whisper will be used to "
            "generate transcripts during Phase 4 — but note it in your report as 'transcripts "
            "generated via ASR rather than provided ground truth' if relevant."
        )

    # ---------------- 5. Sample video health probe ----------------
    sample_n = config["inspection"]["sample_videos_per_split"]
    logger.info(f"Probing a random sample of up to {sample_n} videos per split...")

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

    # ---------------- 6. Write inventory ----------------
    if config["inspection"]["write_inventory_csv"]:
        outputs_dir = Path(config["paths"]["outputs_dir"])
        outputs_dir.mkdir(parents=True, exist_ok=True)

        inventory_path = outputs_dir / "chalearn_inventory.csv"
        inventory_rows = [{"video_path": str(v), "split": infer_split_from_path(v)} for v in videos]
        pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
        logger.info(f"Full video inventory written to: {inventory_path}")

        probe_path = outputs_dir / "chalearn_video_health_sample.csv"
        pd.DataFrame(probe_rows).to_csv(probe_path, index=False)
        logger.info(f"Video health probe sample written to: {probe_path}")

    logger.info("ChaLearn FI-V2 inspection complete. Review warnings above before proceeding to Phase 2.")


if __name__ == "__main__":
    main()
