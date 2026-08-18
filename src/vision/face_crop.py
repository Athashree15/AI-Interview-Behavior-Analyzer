"""Face cropping utility, used to prepare input crops for the emotion classifier."""

from __future__ import annotations

import numpy as np


def crop_face_from_landmarks(frame_bgr: np.ndarray, landmarks_px: np.ndarray, margin_ratio: float = 0.25) -> np.ndarray:
    """
    Crop the face region from a frame using the landmark bounding box,
    with a margin so the emotion model sees full facial context
    (eyebrows, jaw) rather than a tight crop.

    Args:
        frame_bgr: full HxWx3 BGR frame.
        landmarks_px: (N, 3) landmark array in pixel coordinates.
        margin_ratio: fraction of bounding-box size to pad on each side.

    Returns:
        Cropped BGR face image. If the computed box would be degenerate
        (e.g. landmarks all collapse to a point), returns the full frame
        as a safe fallback rather than raising.
    """
    height, width = frame_bgr.shape[:2]

    xs = landmarks_px[:, 0]
    ys = landmarks_px[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    box_w = x_max - x_min
    box_h = y_max - y_min

    if box_w <= 1 or box_h <= 1:
        return frame_bgr

    x_min -= box_w * margin_ratio
    x_max += box_w * margin_ratio
    y_min -= box_h * margin_ratio
    y_max += box_h * margin_ratio

    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(width, int(x_max))
    y_max = min(height, int(y_max))

    return frame_bgr[y_min:y_max, x_min:x_max]
