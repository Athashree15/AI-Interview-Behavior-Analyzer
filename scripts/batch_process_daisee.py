"""
Phase 2 — Batch visual-feature extraction across the full DAiSEE dataset.

Design goals (why it's built this way):
  1. RESUMABLE: writes one output row per video IMMEDIATELY after
     processing it (not buffered in memory until the end), so a crash,
     Ctrl+C, or laptop sleep at video 6,000/8,986 doesn't lose the
     first 6,000. On restart, already-completed videos (identified by
     clip_id already present in the output CSV) are skipped entirely —
     no reprocessing, no wasted GPU time.
  2. FAULT-TOLERANT: a single corrupt/unreadable video must not kill
     an overnight run. Failures are caught per-video, logged to a
     separate failures CSV with the actual error, and the run continues.
  3. HONEST PROGRESS ESTIMATION: prints running average seconds/video
     and an ETA, recalculated periodically — not a fixed guess.
  4. Reuses the per-video on-disk cache from Phase 2 (VisualFeatureCache) —
     if you already ran the demo script on a specific clip, this script
     will not reprocess it.

Usage:
    # Full run (all splits):
    python scripts/batch_process_daisee.py

    # Quick test run on a small subset first (STRONGLY recommended
    # before committing to an overnight run):
    python scripts/batch_process_daisee.py --limit 20

    # Process only one split:
    python scripts/batch_process_daisee.py --split train

Output:
    - outputs/daisee_visual_features_summary.csv   (one row per video: summary stats + labels)
    - outputs/daisee_visual_features_failures.csv  (one row per failed video: clip_id + error)
    - cache/<video_hash>/visual_features.json      (full per-frame + temporal-bin detail, for Phase 5)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.daisee_manifest import build_daisee_manifest  # noqa: E402
from src.utils.config_loader import load_config, resolve_device  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.video.feature_cache import VisualFeatureCache  # noqa: E402
from src.video.frame_sampler import compute_video_hash  # noqa: E402
from src.vision.pipeline import VisualPipeline  # noqa: E402

SUMMARY_CSV_FIELDS = [
    "clip_id",
    "split",
    "video_path",
    "video_hash",
    "Boredom",
    "Engagement",
    "Confusion",
    "Frustration",
    "total_sampled_frames",
    "face_detection_rate",
    "overall_eye_contact_ratio",
    "head_yaw_stability_std_deg",
    "head_pitch_stability_std_deg",
    "multiple_faces_detected_frame_count",
    "top_emotion_overall",
    "emotion_neutral",
    "emotion_happy",
    "emotion_sad",
    "emotion_angry",
    "emotion_fear",
    "emotion_surprise",
    "emotion_disgust",
]

FAILURES_CSV_FIELDS = ["clip_id", "video_path", "split", "error"]


def load_already_processed_clip_ids(summary_csv_path: Path) -> set[str]:
    """Read clip_ids already present in the output CSV, for resumability."""
    if not summary_csv_path.exists():
        return set()
    import pandas as pd

    existing = pd.read_csv(summary_csv_path, usecols=["clip_id"])
    return set(existing["clip_id"].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-process DAiSEE videos through the visual pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N videos (for a quick test run).")
    parser.add_argument(
        "--split", type=str, default=None, choices=["train", "validation", "test"], help="Restrict to one split."
    )
    args = parser.parse_args()

    config = load_config()
    logger = get_logger(__name__, log_dir=config["paths"]["logs_dir"], log_filename="batch_process_daisee.log")

    daisee_root = config["datasets"]["daisee"]["root"]
    drop_unlabeled = config["daisee_split"]["drop_unlabeled_videos"]

    logger.info("Building DAiSEE manifest...")
    manifest = build_daisee_manifest(daisee_root, drop_unlabeled=drop_unlabeled)
    logger.info(f"Manifest built: {len(manifest)} labeled videos.")

    if args.split:
        manifest = manifest[manifest["split"] == args.split].reset_index(drop=True)
        logger.info(f"Restricted to split='{args.split}': {len(manifest)} videos.")

    if args.limit:
        manifest = manifest.head(args.limit)
        logger.info(f"Limited to first {args.limit} videos (test run).")

    outputs_dir = Path(config["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = outputs_dir / "daisee_visual_features_summary.csv"
    failures_csv_path = outputs_dir / "daisee_visual_features_failures.csv"

    already_done = load_already_processed_clip_ids(summary_csv_path)
    if already_done:
        logger.info(f"Found {len(already_done)} already-processed videos in existing output — will skip these.")

    remaining = manifest[~manifest["clip_id"].astype(str).isin(already_done)]
    logger.info(f"{len(remaining)} videos remaining to process.")

    if len(remaining) == 0:
        logger.info("Nothing to do — all videos in this selection are already processed.")
        return

    device = resolve_device(config["runtime"]["device"])
    logger.info(f"Resolved device: {device}. Loading models (one-time)...")

    pipeline = VisualPipeline(
        vision_config=config["vision"],
        sample_fps=config["video"]["sample_fps"],
        device=device,
    )
    cache = VisualFeatureCache(config["paths"]["cache_dir"])

    # Open output files in append mode; write header only if file is new.
    summary_file_is_new = not summary_csv_path.exists()
    failures_file_is_new = not failures_csv_path.exists()

    summary_file = open(summary_csv_path, "a", newline="", encoding="utf-8")
    failures_file = open(failures_csv_path, "a", newline="", encoding="utf-8")
    summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_CSV_FIELDS)
    failures_writer = csv.DictWriter(failures_file, fieldnames=FAILURES_CSV_FIELDS)

    if summary_file_is_new:
        summary_writer.writeheader()
    if failures_file_is_new:
        failures_writer.writeheader()

    processed_count = 0
    failed_count = 0
    start_time = time.time()

    try:
        for idx, row in enumerate(remaining.itertuples(index=False), start=1):
            clip_id = row.clip_id
            video_path = row.video_path
            split = row.split

            try:
                video_hash = compute_video_hash(video_path)
                cached = cache.get(video_hash)

                if cached is not None:
                    result_dict = cached
                else:
                    result = pipeline.run(video_path)
                    result_dict = {
                        "video_path": result.video_path,
                        "total_sampled_frames": result.total_sampled_frames,
                        "frames_with_face_detected": result.frames_with_face_detected,
                        "face_detection_rate": result.face_detection_rate,
                        "temporal_bins": result.temporal_bins,
                        "summary": result.summary,
                        "frame_records": result.frame_records,
                    }
                    cache.set(video_hash, result_dict)

                summary = result_dict["summary"]
                emotion_dist = summary.get("overall_emotion_distribution") or {}

                summary_writer.writerow(
                    {
                        "clip_id": clip_id,
                        "split": split,
                        "video_path": video_path,
                        "video_hash": video_hash,
                        "Boredom": row.Boredom,
                        "Engagement": row.Engagement,
                        "Confusion": row.Confusion,
                        "Frustration": row.Frustration,
                        "total_sampled_frames": result_dict["total_sampled_frames"],
                        "face_detection_rate": result_dict["face_detection_rate"],
                        "overall_eye_contact_ratio": summary.get("overall_eye_contact_ratio"),
                        "head_yaw_stability_std_deg": summary.get("head_yaw_stability_std_deg"),
                        "head_pitch_stability_std_deg": summary.get("head_pitch_stability_std_deg"),
                        "multiple_faces_detected_frame_count": summary.get("multiple_faces_detected_frame_count"),
                        "top_emotion_overall": max(emotion_dist, key=emotion_dist.get) if emotion_dist else None,
                        "emotion_neutral": emotion_dist.get("neutral"),
                        "emotion_happy": emotion_dist.get("happy"),
                        "emotion_sad": emotion_dist.get("sad"),
                        "emotion_angry": emotion_dist.get("angry"),
                        "emotion_fear": emotion_dist.get("fear"),
                        "emotion_surprise": emotion_dist.get("surprise"),
                        "emotion_disgust": emotion_dist.get("disgust"),
                    }
                )
                summary_file.flush()  # ensure this row survives a crash immediately after
                processed_count += 1

            except Exception as exc:  # noqa: BLE001 — must not let one bad video kill an overnight run
                failures_writer.writerow(
                    {"clip_id": clip_id, "video_path": video_path, "split": split, "error": str(exc)}
                )
                failures_file.flush()
                failed_count += 1
                logger.warning(f"FAILED [{clip_id}]: {exc}")

            if idx % 25 == 0 or idx == len(remaining):
                elapsed = time.time() - start_time
                avg_sec_per_video = elapsed / idx
                remaining_count = len(remaining) - idx
                eta_min = (remaining_count * avg_sec_per_video) / 60
                logger.info(
                    f"[{idx}/{len(remaining)}] processed={processed_count} failed={failed_count} "
                    f"| avg={avg_sec_per_video:.2f}s/video | ETA={eta_min:.1f} min"
                )

    finally:
        pipeline.close()
        summary_file.close()
        failures_file.close()

    logger.info("=" * 60)
    logger.info(f"Batch run complete. Processed: {processed_count}, Failed: {failed_count}")
    logger.info(f"Summary written to: {summary_csv_path}")
    if failed_count > 0:
        logger.warning(f"Failures logged to: {failures_csv_path} — review before treating the dataset as complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
