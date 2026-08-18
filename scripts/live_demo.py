"""
Live webcam demonstration (demo-only, not a deployable component).

Reuses the exact same pipeline building blocks as the batch/dashboard
system (face landmarks, head pose, gaze approximation, emotion
classification, and the trained DAiSEE engagement model) — this is
NOT a separate reimplementation, which is why it's trustworthy as a
demo: what you see live is produced by the same code that produced
your Day 1-4 results, not a mocked-up visualization.

Design: the webcam feed is displayed at full framerate, but the heavy
per-frame analysis (face landmarks, emotion classification) is
throttled to a target processing rate (`--process-fps`, default 5) to
stay smooth on a laptop GPU/CPU. Between processed frames, the last
known values are kept on screen rather than flickering to blank.

Every `--update-interval` seconds, a rolling window of the last
`--window-seconds` seconds of processed frames is aggregated into the
SAME feature schema used for training (face detection rate, eye
contact ratio, head stability, emotion distribution) and fed through
the trained DAiSEE engagement_binary model — the same
build_feature_vector/predict_engagement functions used by the
Streamlit dashboard, imported directly (no duplicated logic).

Usage:
    python scripts/live_demo.py
    python scripts/live_demo.py --camera-index 1 --process-fps 8

Controls:
    Press 'q' in the video window to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.inference import build_feature_vector, predict_engagement  # noqa: E402
from src.utils.config_loader import load_config, resolve_device  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.vision.emotion import EmotionClassifier  # noqa: E402
from src.vision.face_crop import crop_face_from_landmarks  # noqa: E402
from src.vision.face_landmarks import FaceLandmarkDetector  # noqa: E402
from src.vision.gaze import estimate_eye_contact  # noqa: E402
from src.vision.head_pose import estimate_head_pose  # noqa: E402
from src.models.daisee_dataset import FEATURE_COLUMNS  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_engagement_bundle() -> dict | None:
    """Load the trained DAiSEE engagement model directly (no Streamlit dependency here)."""
    model_path = Path("models/trained/daisee_engagement_binary_xgboost.joblib")
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def build_window_result(window_records: list[dict]) -> dict | None:
    """
    Aggregate a rolling window of per-frame records into the same
    result shape (`face_detection_rate` + `summary`) that
    build_feature_vector() expects — mirroring exactly what the batch
    pipeline produces per-video, just computed live over a time window
    instead of a whole clip.
    """
    if not window_records:
        return None

    face_frames = [r for r in window_records if r["face_detected"]]
    face_detection_rate = len(face_frames) / len(window_records)

    yaws = [r["yaw"] for r in face_frames if r["yaw"] is not None]
    pitches = [r["pitch"] for r in face_frames if r["pitch"] is not None]
    gaze_frames = [r for r in face_frames if r["looking_at_camera"] is not None]
    eye_contact_ratio = (
        sum(1 for r in gaze_frames if r["looking_at_camera"]) / len(gaze_frames) if gaze_frames else None
    )
    yaw_std = float(np.std(yaws)) if len(yaws) > 1 else None
    pitch_std = float(np.std(pitches)) if len(pitches) > 1 else None
    multi_face_count = sum(1 for r in window_records if r.get("multiple_faces"))

    emotion_valid = [r["emotion_probs"] for r in face_frames if r.get("emotion_probs")]
    emotion_dist = None
    if emotion_valid:
        keys = emotion_valid[0].keys()
        emotion_dist = {k: float(np.mean([e.get(k, 0.0) for e in emotion_valid])) for k in keys}

    return {
        "face_detection_rate": face_detection_rate,
        "summary": {
            "overall_eye_contact_ratio": eye_contact_ratio,
            "head_yaw_stability_std_deg": yaw_std,
            "head_pitch_stability_std_deg": pitch_std,
            "multiple_faces_detected_frame_count": multi_face_count,
            "overall_emotion_distribution": emotion_dist,
        },
    }


def draw_overlay(
    frame: np.ndarray,
    fps: float,
    face_detected: bool,
    yaw: float | None,
    pitch: float | None,
    looking_at_camera: bool | None,
    top_emotion: str | None,
    emotion_conf: float | None,
    engagement_label: str | None,
    engagement_conf: float | None,
    landmarks_px: np.ndarray | None,
) -> np.ndarray:
    """Draw all live overlay text/graphics onto the frame."""
    h, w = frame.shape[:2]
    y = 30

    def put(text: str, color=(0, 255, 0)):
        nonlocal y
        cv2.putText(frame, text, (10, y), FONT, 0.6, color, 2)
        y += 28

    put(f"FPS (display): {fps:.1f}", (200, 200, 200))
    put(f"Face detected: {'YES' if face_detected else 'NO'}", (0, 255, 0) if face_detected else (0, 0, 255))

    if face_detected:
        if landmarks_px is not None:
            xs = landmarks_px[:, 0]
            ys = landmarks_px[:, 1]
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        if yaw is not None and pitch is not None:
            put(f"Head yaw: {yaw:+.1f} deg | pitch: {pitch:+.1f} deg", (255, 200, 0))

        if looking_at_camera is not None:
            status = "Looking at camera (approx.)" if looking_at_camera else "Looking away (approx.)"
            put(status, (0, 255, 0) if looking_at_camera else (0, 165, 255))

        if top_emotion is not None:
            put(f"Top emotion: {top_emotion} ({emotion_conf:.0%})", (255, 255, 0))

    cv2.line(frame, (0, y), (w, y), (80, 80, 80), 1)
    y += 30

    if engagement_label is not None:
        color = (0, 255, 0) if "High" in engagement_label else (0, 165, 255)
        put(f"Live engagement estimate: {engagement_label} ({engagement_conf:.0%} confidence)", color)
    else:
        put("Live engagement estimate: gathering data...", (150, 150, 150))

    cv2.putText(frame, "Press 'q' to quit", (10, h - 15), FONT, 0.5, (150, 150, 150), 1)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam demo (demonstration only, not deployable).")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--process-fps", type=float, default=5.0, help="Target rate for heavy analysis (landmarks/emotion).")
    parser.add_argument("--window-seconds", type=float, default=8.0, help="Rolling window size for the engagement prediction.")
    parser.add_argument("--update-interval", type=float, default=3.0, help="How often (seconds) to refresh the engagement prediction.")
    args = parser.parse_args()

    config = load_config()
    logger = get_logger(__name__, log_dir=config["paths"]["logs_dir"], log_filename="live_demo.log")

    device = resolve_device(config["runtime"]["device"])
    logger.info(f"Resolved device: {device}")

    engagement_bundle = load_engagement_bundle()
    if engagement_bundle is None:
        logger.warning(
            "Trained engagement model not found at models/trained/daisee_engagement_binary_xgboost.joblib "
            "— live engagement estimate will be disabled. Run training/train_daisee_baseline.py first."
        )

    logger.info("Loading face detector and emotion model (one-time)...")
    face_detector = FaceLandmarkDetector(
        detection_confidence=config["vision"]["face_detection_confidence"],
        tracking_confidence=config["vision"]["face_tracking_confidence"],
        refine_landmarks=config["vision"]["refine_landmarks"],
    )
    emotion_classifier = EmotionClassifier(
        model_name=config["vision"]["emotion_model_name"],
        device=device,
        cache_dir=config["vision"]["model_cache_dir"],
    )
    yaw_threshold = config["vision"]["facing_camera_yaw_threshold_deg"]
    pitch_threshold = config["vision"]["facing_camera_pitch_threshold_deg"]

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        logger.error(f"Could not open webcam at index {args.camera_index}.")
        sys.exit(1)

    window_records: deque = deque()
    last_process_time = 0.0
    last_update_time = 0.0
    process_interval = 1.0 / args.process_fps

    # Last-known values, kept on screen between processed frames.
    last_face_detected = False
    last_yaw = last_pitch = None
    last_looking_at_camera = None
    last_top_emotion = None
    last_emotion_conf = None
    last_landmarks = None
    engagement_label = None
    engagement_conf = None

    fps_counter_time = time.time()
    fps_counter_frames = 0
    display_fps = 0.0

    logger.info("Starting live demo. Press 'q' in the video window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read from webcam.")
                break

            now = time.time()
            fps_counter_frames += 1
            if now - fps_counter_time >= 1.0:
                display_fps = fps_counter_frames / (now - fps_counter_time)
                fps_counter_frames = 0
                fps_counter_time = now

            if now - last_process_time >= process_interval:
                last_process_time = now
                landmark_result = face_detector.detect(frame)

                record = {
                    "face_detected": landmark_result.detected,
                    "yaw": None,
                    "pitch": None,
                    "looking_at_camera": None,
                    "emotion_probs": None,
                    "multiple_faces": landmark_result.num_faces_detected > 1,
                    "timestamp": now,
                }

                last_face_detected = landmark_result.detected
                last_landmarks = landmark_result.landmarks_px if landmark_result.detected else None

                if landmark_result.detected:
                    pose_result = estimate_head_pose(
                        landmark_result.landmarks_px, landmark_result.frame_width, landmark_result.frame_height
                    )
                    if pose_result.success:
                        record["yaw"] = pose_result.yaw_deg
                        record["pitch"] = pose_result.pitch_deg
                        last_yaw, last_pitch = pose_result.yaw_deg, pose_result.pitch_deg

                    gaze_result = estimate_eye_contact(
                        landmark_result.landmarks_px, record["yaw"], record["pitch"], yaw_threshold, pitch_threshold
                    )
                    if gaze_result.estimated:
                        record["looking_at_camera"] = gaze_result.looking_at_camera
                        last_looking_at_camera = gaze_result.looking_at_camera

                    face_crop = crop_face_from_landmarks(frame, landmark_result.landmarks_px)
                    emotion_result = emotion_classifier.predict(face_crop)
                    if emotion_result.success:
                        record["emotion_probs"] = emotion_result.probabilities
                        last_top_emotion = emotion_result.top_emotion
                        last_emotion_conf = emotion_result.probabilities[emotion_result.top_emotion]

                window_records.append(record)
                cutoff = now - args.window_seconds
                while window_records and window_records[0]["timestamp"] < cutoff:
                    window_records.popleft()

            if engagement_bundle is not None and now - last_update_time >= args.update_interval:
                last_update_time = now
                window_result = build_window_result(list(window_records))
                if window_result is not None and window_result["face_detection_rate"] > 0.1:
                    X, _imputed = build_feature_vector(window_result, {col: 0.0 for col in FEATURE_COLUMNS})
                    pred = predict_engagement({"daisee_engagement_binary": engagement_bundle}, X)
                    if pred:
                        engagement_label = pred["label"]
                        engagement_conf = pred["confidence"]

            frame = draw_overlay(
                frame,
                display_fps,
                last_face_detected,
                last_yaw,
                last_pitch,
                last_looking_at_camera,
                last_top_emotion,
                last_emotion_conf,
                engagement_label,
                engagement_conf,
                last_landmarks,
            )

            cv2.imshow("AI Interview Analyzer — Live Demo (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_detector.close()
        logger.info("Live demo stopped.")


if __name__ == "__main__":
    main()
