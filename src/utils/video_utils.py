"""
Lightweight video-probing utilities built on OpenCV.

Used during dataset inspection (Phase 1) and later during the
real preprocessing pipeline (Phase 2), so probing logic lives
in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoProbeResult:
    """Container for basic metadata extracted from a video file."""

    path: str
    readable: bool
    fps: float | None = None
    frame_count: int | None = None
    duration_sec: float | None = None
    width: int | None = None
    height: int | None = None
    error: str | None = None


def probe_video(video_path: str | Path) -> VideoProbeResult:
    """
    Open a video file with OpenCV and extract basic metadata without
    decoding every frame. Never raises — always returns a result object,
    with `readable=False` and an `error` message if the file can't be opened.

    Args:
        video_path: path to a video file.

    Returns:
        VideoProbeResult with fps/resolution/duration, or an error message.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        return VideoProbeResult(path=str(video_path), readable=False, error="File does not exist")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return VideoProbeResult(path=str(video_path), readable=False, error="OpenCV could not open file")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration_sec = frame_count / fps if fps and fps > 0 else None

        # Sanity check: some corrupt files report fps=0 or frame_count=0
        # without OpenCV throwing an error.
        if not fps or fps <= 0 or not frame_count or frame_count <= 0:
            return VideoProbeResult(
                path=str(video_path),
                readable=False,
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
                error="Invalid fps or frame_count (likely corrupt or unsupported codec)",
            )

        return VideoProbeResult(
            path=str(video_path),
            readable=True,
            fps=round(fps, 3),
            frame_count=frame_count,
            duration_sec=round(duration_sec, 3) if duration_sec else None,
            width=width,
            height=height,
        )
    except Exception as exc:  # noqa: BLE001 — we want to catch and report anything OpenCV throws
        return VideoProbeResult(path=str(video_path), readable=False, error=str(exc))
    finally:
        cap.release()
