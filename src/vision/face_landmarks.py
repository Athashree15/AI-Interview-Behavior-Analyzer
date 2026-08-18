"""
Face landmark detection using MediaPipe Face Mesh.

Why MediaPipe over RetinaFace/MTCNN/YOLO-face/InsightFace (Module 1
model-selection decision):
  - Pure pip install, no compiled CUDA extensions to fight with on
    Windows (RetinaFace/InsightFace commonly hit build issues on
    Windows without Visual C++ build tools already configured).
  - Runs comfortably on CPU in real time, so it works even if the
    GPU is busy running the emotion/Whisper models.
  - `refine_landmarks=True` gives iris landmarks for free, which we
    need for the gaze-approximation module — avoids needing a
    separate gaze-specific model.
  - 468 (or 478 with iris) landmarks give us everything head-pose
    estimation needs (Module 4) without a second detector.

Trade-off, documented honestly: MediaPipe's face mesh is a geometric
landmark model, not a state-of-the-art face *detector* for extreme
angles/occlusion. For interview-style footage (person facing a
webcam, reasonably front-on) this is a reasonable, well-justified
choice; it would NOT be the right choice for, e.g., surveillance
footage with extreme angles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FaceLandmarkResult:
    """Container for a single frame's face landmark detection result."""

    detected: bool
    landmarks_px: np.ndarray | None = None   # shape (N, 3): x_px, y_px, z (relative depth)
    frame_width: int | None = None
    frame_height: int | None = None
    num_faces_detected: int = 0


class FaceLandmarkDetector:
    """
    Thin wrapper around MediaPipe FaceMesh. Instantiate once and reuse
    across all frames of a video — creating a new FaceMesh per frame
    is expensive and unnecessary.
    """

    def __init__(
        self,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
        max_num_faces: int = 1,
    ):
        # Imported here (not at module top) so that importing this file
        # doesn't hard-require mediapipe unless a detector is actually
        # instantiated — keeps unit-testable modules lighter elsewhere.
        import mediapipe as mp

        self._mp_face_mesh = mp.solutions.face_mesh
        self._detector = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> FaceLandmarkResult:
        """
        Run face landmark detection on a single BGR frame.

        Args:
            frame_bgr: HxWx3 BGR frame (OpenCV's native format).

        Returns:
            FaceLandmarkResult. If no face is detected, `detected=False`
            and `landmarks_px=None` — callers MUST handle this case
            (Module 1 requirement: "handle frames where no face is
            detected" rather than crashing).
        """
        import cv2

        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(frame_rgb)

        if not results.multi_face_landmarks:
            return FaceLandmarkResult(
                detected=False, frame_width=width, frame_height=height, num_faces_detected=0
            )

        num_faces = len(results.multi_face_landmarks)
        # Module 22 requirement: handle multiple faces gracefully.
        # We take the largest face (by bounding-box area) as the candidate,
        # since interview footage should have exactly one subject — if
        # more than one face is present, this is logged upstream by the
        # pipeline as a data-quality warning, not silently ignored.
        best_landmarks = self._select_largest_face(results.multi_face_landmarks, width, height)

        landmarks_px = np.array(
            [[lm.x * width, lm.y * height, lm.z * width] for lm in best_landmarks.landmark],
            dtype=np.float32,
        )

        return FaceLandmarkResult(
            detected=True,
            landmarks_px=landmarks_px,
            frame_width=width,
            frame_height=height,
            num_faces_detected=num_faces,
        )

    @staticmethod
    def _select_largest_face(multi_face_landmarks, width: int, height: int):
        """Pick the face with the largest landmark bounding-box area."""
        best = None
        best_area = -1.0
        for face_landmarks in multi_face_landmarks:
            xs = [lm.x * width for lm in face_landmarks.landmark]
            ys = [lm.y * height for lm in face_landmarks.landmark]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if area > best_area:
                best_area = area
                best = face_landmarks
        return best

    def close(self) -> None:
        """Release MediaPipe resources. Call when done processing a video."""
        self._detector.close()

    def __enter__(self) -> "FaceLandmarkDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
