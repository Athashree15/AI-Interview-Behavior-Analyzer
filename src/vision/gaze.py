"""
Approximate eye-contact / gaze estimation (Module 3).

IMPORTANT — READ BEFORE USING THIS MODULE'S OUTPUT:
This is NOT a calibrated gaze-estimation model (e.g. ETH-XGaze-trained
regression to a 3D gaze vector). It is a geometric APPROXIMATION built
from two signals:
  1. Head orientation (yaw/pitch from head_pose.py) — is the head
     turned toward the camera.
  2. Iris position relative to the eye-corner landmarks — is the iris
     centered within the eye socket, vs. looking toward the socket edge.

Combining these gives a reasonable proxy for "appears to be looking
toward the camera" but is explicitly NOT the same as true gaze
direction (which requires eyeball rotation modeling, ideally with a
calibrated camera and per-user calibration). This distinction is
documented here, in the Responsible AI section, and in every UI
label that surfaces this metric — it must always be labeled
"approximate eye-contact indicator", never "gaze accuracy" or
"attention score".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Iris landmark indices, only present when FaceLandmarkDetector was
# constructed with refine_landmarks=True.
_RIGHT_IRIS_CENTER = 468
_LEFT_IRIS_CENTER = 473
_RIGHT_EYE_LEFT_CORNER = 33
_RIGHT_EYE_RIGHT_CORNER = 133
_LEFT_EYE_LEFT_CORNER = 362
_LEFT_EYE_RIGHT_CORNER = 263


@dataclass
class GazeEstimate:
    """Approximate eye-contact estimate for a single frame."""

    estimated: bool
    looking_at_camera: bool | None = None
    horizontal_offset_ratio: float | None = None  # -1 (looking left) to +1 (looking right), 0 = centered
    error: str | None = None


def estimate_eye_contact(
    landmarks_px: np.ndarray,
    yaw_deg: float | None,
    pitch_deg: float | None,
    yaw_threshold_deg: float,
    pitch_threshold_deg: float,
) -> GazeEstimate:
    """
    Approximate whether the subject appears to be looking toward the
    camera, combining head orientation with iris-centering.

    Args:
        landmarks_px: (N, 3) landmark array, must include iris points
            (i.e. detector was created with refine_landmarks=True).
        yaw_deg, pitch_deg: head pose from head_pose.estimate_head_pose().
            If None (head pose estimation failed for this frame), this
            function falls back to iris-only estimation.
        yaw_threshold_deg, pitch_threshold_deg: from config.yaml
            (`vision.facing_camera_yaw_threshold_deg` / `_pitch_...`).

    Returns:
        GazeEstimate. `estimated=False` if iris landmarks are unavailable
        (e.g. refine_landmarks was off, or landmarks array is too short).
    """
    if landmarks_px.shape[0] <= _LEFT_IRIS_CENTER:
        return GazeEstimate(estimated=False, error="Iris landmarks not present (refine_landmarks was off)")

    try:
        # Head-orientation gate: if the head itself is turned away
        # beyond threshold, we call it "not looking at camera"
        # regardless of iris position — a turned head with eyes at
        # the socket edge is not meaningfully "eye contact".
        head_facing_camera = True
        if yaw_deg is not None and pitch_deg is not None:
            head_facing_camera = abs(yaw_deg) <= yaw_threshold_deg and abs(pitch_deg) <= pitch_threshold_deg

        # Iris-centering signal (horizontal only — vertical iris
        # offset is noisier with this landmark set and not used here).
        right_offset = _horizontal_iris_offset(
            landmarks_px, _RIGHT_IRIS_CENTER, _RIGHT_EYE_LEFT_CORNER, _RIGHT_EYE_RIGHT_CORNER
        )
        left_offset = _horizontal_iris_offset(
            landmarks_px, _LEFT_IRIS_CENTER, _LEFT_EYE_LEFT_CORNER, _LEFT_EYE_RIGHT_CORNER
        )
        avg_offset = float(np.mean([right_offset, left_offset]))

        # Iris considered "centered enough" if within 30% of the eye
        # width from center — a conservative, documented threshold,
        # not empirically tuned (no ground-truth gaze dataset used
        # for calibration, per the limitation stated in the module docstring).
        iris_centered = abs(avg_offset) <= 0.3

        looking_at_camera = head_facing_camera and iris_centered

        return GazeEstimate(estimated=True, looking_at_camera=looking_at_camera, horizontal_offset_ratio=avg_offset)

    except Exception as exc:  # noqa: BLE001
        return GazeEstimate(estimated=False, error=str(exc))


def _horizontal_iris_offset(landmarks_px: np.ndarray, iris_idx: int, left_corner_idx: int, right_corner_idx: int) -> float:
    """Return iris horizontal position as a ratio in [-1, 1] relative to eye width, 0 = centered."""
    iris_x = landmarks_px[iris_idx][0]
    left_x = landmarks_px[left_corner_idx][0]
    right_x = landmarks_px[right_corner_idx][0]

    eye_width = right_x - left_x
    if abs(eye_width) < 1e-6:
        return 0.0

    eye_center_x = (left_x + right_x) / 2
    offset = (iris_x - eye_center_x) / (eye_width / 2)
    return float(np.clip(offset, -1.0, 1.0))
