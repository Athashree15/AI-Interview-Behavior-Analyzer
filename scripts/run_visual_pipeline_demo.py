"""
Phase 2 — Visual pipeline demo.

Runs the complete visual analysis pipeline (face detection, head pose,
approximate gaze, emotion classification, temporal aggregation) on a
single video and prints/saves the result. Use this to validate the
pipeline works on your machine before we wire it into batch dataset
processing for model training.

Usage:
    # Process a specific video:
    python scripts/run_visual_pipeline_demo.py --video "C:/path/to/clip.avi"

    # Or, with no --video argument, automatically picks the first video
    # from outputs/daisee_inventory.csv (generated in Phase 1):
    python scripts/run_visual_pipeline_demo.py

Output:
    - Console summary
    - outputs/visual_pipeline_demo/<video_stem>/result.json (full result)
    - Cached under cache/<video_hash>/visual_features.json for reuse
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config_loader import load_config, resolve_device  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.video.feature_cache import VisualFeatureCache  # noqa: E402
from src.video.frame_sampler import compute_video_hash  # noqa: E402
from src.vision.pipeline import VisualPipeline  # noqa: E402


def pick_default_video(config: dict) -> str:
    """Fall back to the first video in the Phase-1-generated DAiSEE inventory."""
    inventory_path = Path(config["paths"]["outputs_dir"]) / "daisee_inventory.csv"
    if not inventory_path.exists():
        raise FileNotFoundError(
            f"No --video given and {inventory_path} doesn't exist. "
            "Either pass --video explicitly, or run scripts/inspect_daisee.py first."
        )
    df = pd.read_csv(inventory_path)
    if df.empty:
        raise ValueError(f"{inventory_path} is empty.")
    return str(df.iloc[0]["video_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the visual pipeline on a single video.")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file. Defaults to the first DAiSEE clip found.")
    parser.add_argument("--no-cache", action="store_true", help="Force reprocessing even if a cached result exists.")
    args = parser.parse_args()

    config = load_config()
    logger = get_logger(__name__, log_dir=config["paths"]["logs_dir"], log_filename="visual_pipeline_demo.log")

    video_path = args.video or pick_default_video(config)
    video_path = str(Path(video_path))

    if not Path(video_path).exists():
        logger.error(f"Video file does not exist: {video_path}")
        sys.exit(1)

    logger.info(f"Processing video: {video_path}")

    device = resolve_device(config["runtime"]["device"])
    logger.info(f"Resolved device: {device}")

    video_hash = compute_video_hash(video_path)
    cache = VisualFeatureCache(config["paths"]["cache_dir"])

    if not args.no_cache:
        cached = cache.get(video_hash)
        if cached is not None:
            logger.info(f"Found cached result for this video (hash={video_hash}). Use --no-cache to force reprocessing.")
            _print_summary(cached, logger)
            _write_output(config, video_path, cached, logger)
            return

    logger.info("No cached result found (or --no-cache set). Loading models — this may take a minute the first time (downloading pretrained weights)...")

    start_time = time.time()
    pipeline = VisualPipeline(
        vision_config=config["vision"],
        sample_fps=config["video"]["sample_fps"],
        device=device,
    )

    try:
        result = pipeline.run(video_path)
    finally:
        pipeline.close()

    elapsed = time.time() - start_time
    logger.info(f"Pipeline finished in {elapsed:.1f}s")

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
    logger.info(f"Result cached (hash={video_hash}).")

    _print_summary(result_dict, logger)
    _write_output(config, video_path, result_dict, logger)


def _print_summary(result: dict, logger) -> None:
    logger.info("=" * 60)
    logger.info(f"Total sampled frames: {result['total_sampled_frames']}")
    logger.info(f"Frames with face detected: {result['frames_with_face_detected']} ({result['face_detection_rate']:.1%})")
    logger.info(f"Overall eye-contact ratio (approximate): {result['summary'].get('overall_eye_contact_ratio')}")
    logger.info(f"Head yaw stability (std dev, deg): {result['summary'].get('head_yaw_stability_std_deg')}")
    logger.info(f"Head pitch stability (std dev, deg): {result['summary'].get('head_pitch_stability_std_deg')}")
    logger.info(f"Multiple-faces-detected frame count: {result['summary'].get('multiple_faces_detected_frame_count')}")
    emotion_dist = result["summary"].get("overall_emotion_distribution")
    if emotion_dist:
        logger.info("Overall emotion distribution:")
        for label, prob in sorted(emotion_dist.items(), key=lambda x: -x[1]):
            logger.info(f"  {label}: {prob:.1%}")
    logger.info(f"Number of temporal bins ({result['temporal_bins'][0]['end_sec'] - result['temporal_bins'][0]['start_sec'] if result['temporal_bins'] else '?'}s each): {len(result['temporal_bins'])}")
    logger.info("=" * 60)


def _write_output(config: dict, video_path: str, result: dict, logger) -> None:
    video_stem = Path(video_path).stem
    out_dir = Path(config["paths"]["outputs_dir"]) / "visual_pipeline_demo" / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Full result written to: {out_path}")


if __name__ == "__main__":
    main()
