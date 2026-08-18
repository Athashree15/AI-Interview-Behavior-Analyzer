"""
Frame sampling and video-cache-key utilities.

Design decision: we do NOT decode every frame of a video. We sample at
a fixed rate (`video.sample_fps` in config.yaml, default 2 fps) since
decoding a 10-second DAiSEE clip at its native ~30fps for every one of
9,000 clips is wasteful and unnecessary for behavioral-trend features
that don't change frame-to-frame (Module 1, Module 20 — training
efficiency constraint on a laptop GPU).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def compute_video_hash(video_path: str | Path) -> str:
    """
    Compute a stable hash for a video file used as its cache-directory
    name. Uses file path + size + mtime rather than hashing full file
    bytes, since hashing every DAiSEE clip's contents would itself be
    a slow, unnecessary preprocessing step.
    """
    video_path = Path(video_path)
    stat = video_path.stat()
    key = f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def sample_frames(video_path: str | Path, sample_fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """
    Generator yielding (timestamp_seconds, frame_bgr) pairs sampled at
    `sample_fps` frames per second, regardless of the source video's
    native fps.

    Args:
        video_path: path to a video file.
        sample_fps: desired sampling rate in frames per second.

    Yields:
        (timestamp_seconds, frame) tuples. `frame` is a BGR numpy array
        (OpenCV's native format).

    Raises:
        RuntimeError: if the video cannot be opened.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 0:
            native_fps = 30.0  # safe fallback; most webcam/interview footage is 24-30fps

        # Sample every Nth frame such that the sampled rate approximates sample_fps
        frame_interval = max(1, round(native_fps / sample_fps))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_sec = frame_idx / native_fps
                yield timestamp_sec, frame

            frame_idx += 1
    finally:
        cap.release()
