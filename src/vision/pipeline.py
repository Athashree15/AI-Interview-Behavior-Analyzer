"""
Visual pipeline orchestrator.

Ties together frame sampling, face landmark detection, head pose,
gaze approximation, and emotion classification into a single
per-video run, producing:
  1. A per-frame record list (raw, for debugging/inspection).
  2. Temporal trajectories binned into fixed-width windows
     (Module 2's "temporal emotion trajectory" spec, e.g. 0-10s,
     10-20s, ...) rather than one single frame-averaged number.
  3. Summary statistics (eye-contact ratio, head stability, emotion
     distribution) used later by the scoring module.

Explicitly does NOT silently drop failed frames — every frame's
success/failure is recorded, and the final summary reports what
fraction of frames had a usable face detection, so downstream
consumers (and your report) know exactly how much of the video
contributed to the result (Module 22: graceful error handling,
not silent data loss).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.vision.emotion import EmotionClassifier
from src.vision.face_crop import crop_face_from_landmarks
from src.vision.face_landmarks import FaceLandmarkDetector
from src.vision.gaze import estimate_eye_contact
from src.vision.head_pose import estimate_head_pose
from src.video.frame_sampler import sample_frames


@dataclass
class FrameRecord:
    """Everything extracted from a single sampled frame."""

    timestamp_sec: float
    face_detected: bool
    num_faces_detected: int = 0
    multiple_faces_warning: bool = False
    pose_success: bool = False
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    roll_deg: float | None = None
    gaze_estimated: bool = False
    looking_at_camera: bool | None = None
    emotion_success: bool = False
    emotion_probabilities: dict[str, float] | None = None
    top_emotion: str | None = None


@dataclass
class VisualPipelineResult:
    """Full result of running the visual pipeline on one video."""

    video_path: str
    total_sampled_frames: int
    frames_with_face_detected: int
    face_detection_rate: float
    frame_records: list[dict[str, Any]] = field(default_factory=list)
    temporal_bins: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class VisualPipeline:
    """
    Orchestrates the full visual analysis pipeline for one video.
    Instantiate once (loads the emotion model + face detector), then
    call `.run(video_path)` for as many videos as needed — avoids
    reloading model weights per video.
    """

    def __init__(self, vision_config: dict[str, Any], sample_fps: float, device: str):
        self.sample_fps = sample_fps
        self.yaw_threshold = vision_config["facing_camera_yaw_threshold_deg"]
        self.pitch_threshold = vision_config["facing_camera_pitch_threshold_deg"]
        self.temporal_bin_seconds = vision_config["temporal_bin_seconds"]

        self._face_detector = FaceLandmarkDetector(
            detection_confidence=vision_config["face_detection_confidence"],
            tracking_confidence=vision_config["face_tracking_confidence"],
            refine_landmarks=vision_config["refine_landmarks"],
        )
        self._emotion_classifier = EmotionClassifier(
            model_name=vision_config["emotion_model_name"],
            device=device,
            cache_dir=vision_config["model_cache_dir"],
        )

    def run(self, video_path: str) -> VisualPipelineResult:
        """Run the full visual pipeline on a single video file."""
        frame_records: list[FrameRecord] = []

        for timestamp_sec, frame_bgr in sample_frames(video_path, self.sample_fps):
            record = self._process_frame(timestamp_sec, frame_bgr)
            frame_records.append(record)

        total = len(frame_records)
        with_face = sum(1 for r in frame_records if r.face_detected)
        face_detection_rate = (with_face / total) if total > 0 else 0.0

        temporal_bins = self._aggregate_temporal_bins(frame_records)
        summary = self._compute_summary(frame_records)

        return VisualPipelineResult(
            video_path=video_path,
            total_sampled_frames=total,
            frames_with_face_detected=with_face,
            face_detection_rate=round(face_detection_rate, 4),
            frame_records=[asdict(r) for r in frame_records],
            temporal_bins=temporal_bins,
            summary=summary,
        )

    def _process_frame(self, timestamp_sec: float, frame_bgr: np.ndarray) -> FrameRecord:
        landmark_result = self._face_detector.detect(frame_bgr)

        if not landmark_result.detected:
            return FrameRecord(timestamp_sec=timestamp_sec, face_detected=False)

        record = FrameRecord(
            timestamp_sec=timestamp_sec,
            face_detected=True,
            num_faces_detected=landmark_result.num_faces_detected,
            multiple_faces_warning=landmark_result.num_faces_detected > 1,
        )

        # Head pose
        pose_result = estimate_head_pose(
            landmark_result.landmarks_px, landmark_result.frame_width, landmark_result.frame_height
        )
        if pose_result.success:
            record.pose_success = True
            record.pitch_deg = pose_result.pitch_deg
            record.yaw_deg = pose_result.yaw_deg
            record.roll_deg = pose_result.roll_deg

        # Gaze approximation
        gaze_result = estimate_eye_contact(
            landmark_result.landmarks_px,
            record.yaw_deg,
            record.pitch_deg,
            self.yaw_threshold,
            self.pitch_threshold,
        )
        if gaze_result.estimated:
            record.gaze_estimated = True
            record.looking_at_camera = gaze_result.looking_at_camera

        # Emotion classification
        face_crop = crop_face_from_landmarks(frame_bgr, landmark_result.landmarks_px)
        emotion_result = self._emotion_classifier.predict(face_crop)
        if emotion_result.success:
            record.emotion_success = True
            record.emotion_probabilities = emotion_result.probabilities
            record.top_emotion = emotion_result.top_emotion

        return record

    def _aggregate_temporal_bins(self, frame_records: list[FrameRecord]) -> list[dict[str, Any]]:
        """
        Group frames into fixed-width time bins and compute per-bin
        emotion distribution + eye-contact ratio + head stability —
        the "temporal trajectory" required by Module 2, rather than
        one single video-level average.
        """
        if not frame_records:
            return []

        bin_width = self.temporal_bin_seconds
        max_time = max(r.timestamp_sec for r in frame_records)
        num_bins = int(max_time // bin_width) + 1

        bins: list[dict[str, Any]] = []
        for bin_idx in range(num_bins):
            bin_start = bin_idx * bin_width
            bin_end = bin_start + bin_width
            bin_records = [r for r in frame_records if bin_start <= r.timestamp_sec < bin_end]

            if not bin_records:
                continue

            bins.append(
                {
                    "start_sec": bin_start,
                    "end_sec": bin_end,
                    "num_frames": len(bin_records),
                    "face_detection_rate": round(
                        sum(1 for r in bin_records if r.face_detected) / len(bin_records), 4
                    ),
                    "emotion_distribution": self._emotion_distribution(bin_records),
                    "eye_contact_ratio": self._eye_contact_ratio(bin_records),
                    "mean_yaw_deg": self._mean_or_none([r.yaw_deg for r in bin_records if r.yaw_deg is not None]),
                    "mean_pitch_deg": self._mean_or_none(
                        [r.pitch_deg for r in bin_records if r.pitch_deg is not None]
                    ),
                }
            )

        return bins

    def _compute_summary(self, frame_records: list[FrameRecord]) -> dict[str, Any]:
        """Video-level summary statistics, computed only over frames with valid data (not the full frame count)."""
        face_frames = [r for r in frame_records if r.face_detected]

        return {
            "overall_emotion_distribution": self._emotion_distribution(face_frames),
            "overall_eye_contact_ratio": self._eye_contact_ratio(face_frames),
            "head_pose_success_rate": round(
                sum(1 for r in face_frames if r.pose_success) / len(face_frames), 4
            )
            if face_frames
            else 0.0,
            "multiple_faces_detected_frame_count": sum(1 for r in frame_records if r.multiple_faces_warning),
            "head_yaw_stability_std_deg": self._std_or_none(
                [r.yaw_deg for r in face_frames if r.yaw_deg is not None]
            ),
            "head_pitch_stability_std_deg": self._std_or_none(
                [r.pitch_deg for r in face_frames if r.pitch_deg is not None]
            ),
        }

    @staticmethod
    def _emotion_distribution(records: list[FrameRecord]) -> dict[str, float] | None:
        valid = [r for r in records if r.emotion_success and r.emotion_probabilities]
        if not valid:
            return None

        totals: dict[str, float] = defaultdict(float)
        for r in valid:
            for label, prob in r.emotion_probabilities.items():
                totals[label] += prob

        n = len(valid)
        return {label: round(total / n, 4) for label, total in totals.items()}

    @staticmethod
    def _eye_contact_ratio(records: list[FrameRecord]) -> float | None:
        valid = [r for r in records if r.gaze_estimated]
        if not valid:
            return None
        return round(sum(1 for r in valid if r.looking_at_camera) / len(valid), 4)

    @staticmethod
    def _mean_or_none(values: list[float]) -> float | None:
        return round(float(np.mean(values)), 3) if values else None

    @staticmethod
    def _std_or_none(values: list[float]) -> float | None:
        return round(float(np.std(values)), 3) if len(values) > 1 else None

    def close(self) -> None:
        """Release resources (face detector). Call when fully done processing."""
        self._face_detector.close()
