"""
Head pose estimation (Module 4) using OpenCV's solvePnP against a
generic 3D face model, driven by six stable MediaPipe Face Mesh
landmark indices.

This is a standard, widely-used technique (canonical 3D face point
correspondence + solvePnP) — not a novel contribution, and not
presented as one. What IS specific to this project is how the
resulting yaw/pitch/roll feed into the head-stability and
facing-camera-ratio features used downstream (Module 4/5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# MediaPipe Face Mesh landmark indices for 6 stable, well-separated
# facial points, paired with a generic 3D face model (in an arbitrary
# but consistent unit scale — solvePnP only needs consistent ratios,
# not real-world millimeters, since we don't calibrate the camera).
_LANDMARK_INDEX_NOSE_TIP = 1
_LANDMARK_INDEX_CHIN = 152
_LANDMARK_INDEX_LEFT_EYE_LEFT_CORNER = 33
_LANDMARK_INDEX_RIGHT_EYE_RIGHT_CORNER = 263
_LANDMARK_INDEX_LEFT_MOUTH_CORNER = 61
_LANDMARK_INDEX_RIGHT_MOUTH_CORNER = 291

_MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)


@dataclass
class HeadPoseResult:
    """Pitch/yaw/roll in degrees, plus whether estimation succeeded."""

    success: bool
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    roll_deg: float | None = None
    error: str | None = None


def estimate_head_pose(landmarks_px: np.ndarray, frame_width: int, frame_height: int) -> HeadPoseResult:
    """
    Estimate head pose (pitch, yaw, roll) from MediaPipe landmarks.

    Args:
        landmarks_px: (N, 3) array of pixel-space landmarks from
            FaceLandmarkDetector.detect(...).landmarks_px
        frame_width, frame_height: frame dimensions in pixels.

    Returns:
        HeadPoseResult with pitch/yaw/roll in degrees, or success=False
        with an error message if solvePnP fails to converge (this can
        happen on low-quality or heavily occluded frames — callers
        must handle this rather than assuming pose is always available).
    """
    try:
        image_points = np.array(
            [
                landmarks_px[_LANDMARK_INDEX_NOSE_TIP][:2],
                landmarks_px[_LANDMARK_INDEX_CHIN][:2],
                landmarks_px[_LANDMARK_INDEX_LEFT_EYE_LEFT_CORNER][:2],
                landmarks_px[_LANDMARK_INDEX_RIGHT_EYE_RIGHT_CORNER][:2],
                landmarks_px[_LANDMARK_INDEX_LEFT_MOUTH_CORNER][:2],
                landmarks_px[_LANDMARK_INDEX_RIGHT_MOUTH_CORNER][:2],
            ],
            dtype=np.float64,
        )

        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64
        )
        dist_coeffs = np.zeros((4, 1))  # assume no lens distortion (standard simplifying assumption)

        success, rotation_vector, _translation_vector = cv2.solvePnP(
            _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return HeadPoseResult(success=False, error="solvePnP did not converge")

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = _rotation_matrix_to_euler_angles(rotation_matrix)

        return HeadPoseResult(success=True, pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll)

    except Exception as exc:  # noqa: BLE001
        return HeadPoseResult(success=False, error=str(exc))


def _rotation_matrix_to_euler_angles(R: np.ndarray) -> tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to pitch/yaw/roll in degrees."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0

    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    roll_deg = math.degrees(roll)

    # KNOWN QUIRK of this 6-point model + solvePnP convention: for a
    # roughly forward-facing head, the raw pitch angle lands near
    # +-180 degrees instead of near 0. Left uncorrected, small natural
    # head movements flip the sign near the +-180 boundary (e.g. +178
    # <-> -178), which looks like huge instability even when the head
    # barely moved, and also breaks any "is pitch within threshold"
    # check downstream (confirmed empirically: this caused a 102-degree
    # std dev and a 0% eye-contact ratio on a visibly still subject
    # during Phase 2 testing). We normalize it back into a sane
    # around-zero range here, once, so every downstream consumer
    # (gaze estimation, stability metrics) gets a physically sensible value.
    if pitch_deg > 90:
        pitch_deg -= 180
    elif pitch_deg < -90:
        pitch_deg += 180

    return pitch_deg, yaw_deg, roll_deg
