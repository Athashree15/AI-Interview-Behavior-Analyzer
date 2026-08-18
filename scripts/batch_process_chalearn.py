"""
Phase 2 — Batch visual-feature extraction across ChaLearn First
Impressions V2's 6,000 officially-labeled training clips.

Identical design principles to scripts/batch_process_daisee.py:
resumable (flushes each row immediately, skips already-done
video_ids on restart), fault-tolerant (one bad video doesn't kill
the run), honest progress/ETA reporting. See that script's docstring
for the full rationale — not repeated here.

Usage:
    python scripts/batch_process_chalearn.py --limit 20   # test first
    python scripts/batch_process_chalearn.py               # full run

Output:
    - outputs/chalearn_visual_features_summary.csv
    - outputs/chalearn_visual_features_failures.csv
    - cache/<video_hash>/visual_features.json (shared cache with DAiSEE run)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.chalearn_manifest import build_chalearn_manifest  # noqa: E402
from src.utils.config_loader import load_config, resolve_device  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.video.feature_cache import VisualFeatureCache  # noqa: E402
from src.video.frame_sampler import compute_video_hash  # noqa: E402
from src.vision.pipeline import VisualPipeline  # noqa: E402

SUMMARY_CSV_FIELDS = [
    "video_id",
    "internal_split",
    "official_video_folder",
    "video_path",
    "video_hash",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
    "interview",
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

FAILURES_CSV_FIELDS = ["video_id", "video_path", "internal_split", "error"]


def load_already_processed_video_ids(summary_csv_path: Path) -> set[str]:
    if not summary_csv_path.exists():
        return set()
    import pandas as pd

    existing = pd.read_csv(summary_csv_path, usecols=["video_id"])
    return set(existing["video_id"].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-process ChaLearn FI-V2 videos through the visual pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N videos (for a quick test run).")
    parser.add_argument(
        "--split", type=str, default=None, choices=["train", "validation", "test"],
        help="Restrict to one internal split (our own 80/10/10 re-split, not the official ChaLearn folders).",
    )
    args = parser.parse_args()

    config = load_config()
    logger = get_logger(__name__, log_dir=config["paths"]["logs_dir"], log_filename="batch_process_chalearn.log")

    chalearn_root = config["datasets"]["chalearn_fi_v2"]["root"]
    split_ratios = tuple(config["chalearn_split"]["custom_split_ratios"])
    seed = config["chalearn_split"]["split_seed"]

    logger.info("Building ChaLearn manifest (training-annotation-only, custom 80/10/10 split)...")
    manifest = build_chalearn_manifest(chalearn_root, split_ratios=split_ratios, seed=seed)
    logger.info(f"Manifest built: {len(manifest)} labeled videos.")
    logger.info(
        f"Internal split sizes: "
        f"train={sum(manifest['internal_split'] == 'train')}, "
        f"validation={sum(manifest['internal_split'] == 'validation')}, "
        f"test={sum(manifest['internal_split'] == 'test')}"
    )

    if args.split:
        manifest = manifest[manifest["internal_split"] == args.split].reset_index(drop=True)
        logger.info(f"Restricted to internal_split='{args.split}': {len(manifest)} videos.")

    if args.limit:
        manifest = manifest.head(args.limit)
        logger.info(f"Limited to first {args.limit} videos (test run).")

    outputs_dir = Path(config["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = outputs_dir / "chalearn_visual_features_summary.csv"
    failures_csv_path = outputs_dir / "chalearn_visual_features_failures.csv"

    already_done = load_already_processed_video_ids(summary_csv_path)
    if already_done:
        logger.info(f"Found {len(already_done)} already-processed videos in existing output — will skip these.")

    remaining = manifest[~manifest["video_id"].astype(str).isin(already_done)]
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
            video_id = row.video_id
            video_path = row.video_path
            internal_split = row.internal_split

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
                        "video_id": video_id,
                        "internal_split": internal_split,
                        "official_video_folder": row.official_video_folder,
                        "video_path": video_path,
                        "video_hash": video_hash,
                        "extraversion": row.extraversion,
                        "agreeableness": row.agreeableness,
                        "conscientiousness": row.conscientiousness,
                        "neuroticism": row.neuroticism,
                        "openness": row.openness,
                        "interview": row.interview,
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
                summary_file.flush()
                processed_count += 1

            except Exception as exc:  # noqa: BLE001
                failures_writer.writerow(
                    {"video_id": video_id, "video_path": video_path, "internal_split": internal_split, "error": str(exc)}
                )
                failures_file.flush()
                failed_count += 1
                logger.warning(f"FAILED [{video_id}]: {exc}")

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
