"""
Facial expression/emotion classification using a pretrained model
(Module 2). No training performed here — we use an existing,
published pretrained model rather than training an emotion classifier
from scratch, consistent with Module 20 (training-efficiency
constraints) and standard practice when the goal is feature
extraction, not novel emotion-recognition research.

Why a HuggingFace/PyTorch model instead of DeepFace:
DeepFace defaults to a TensorFlow backend, and this project's other
components (Whisper, sentence-transformers, the temporal LSTM/GRU
models in Phase 5) are all PyTorch-based. Standardizing on a single
framework avoids a dual TF+PyTorch install (a common source of
Windows-specific CUDA/cuDNN version conflicts, especially on a 6GB
laptop GPU where memory headroom is already tight), and keeps
`requirements.txt` simpler. `trpakov/vit-face-expression` is a
ViT-based model fine-tuned on FER2013 + additional data, distributed
through the standard `transformers` pipeline API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmotionResult:
    """Emotion classification result for a single face crop."""

    success: bool
    probabilities: dict[str, float] | None = None   # e.g. {"happy": 0.62, "neutral": 0.30, ...}
    top_emotion: str | None = None
    error: str | None = None


class EmotionClassifier:
    """
    Thin wrapper around a HuggingFace image-classification pipeline.
    Instantiate ONCE per script/session — loading model weights per
    frame would be extremely slow.
    """

    def __init__(self, model_name: str, device: str, cache_dir: str):
        # Imported lazily so this module can be imported without
        # requiring transformers/torch to already be installed
        # (useful during early Phase 1-style dependency-light checks).
        from transformers import pipeline

        pipeline_device = 0 if device == "cuda" else -1
        self._classifier = pipeline(
            task="image-classification",
            model=model_name,
            device=pipeline_device,
            model_kwargs={"cache_dir": cache_dir},
            # Explicitly force softmax and return every class. Left to
            # its default auto-detection, this pipeline produced
            # independent per-class (sigmoid-style) scores that summed
            # to ~236% instead of 100% (confirmed empirically during
            # Phase 2 testing) — which silently breaks every downstream
            # consumer of "emotion distribution" as a proper probability
            # distribution (temporal aggregation, engagement scoring).
            # Forcing softmax here guarantees scores sum to 1 per frame.
            top_k=None,
            function_to_apply="softmax",
        )

    def predict(self, face_crop_bgr: np.ndarray) -> EmotionResult:
        """
        Classify emotion from a cropped face image.

        Args:
            face_crop_bgr: cropped face region, BGR (from crop_face_from_landmarks).

        Returns:
            EmotionResult with a probability distribution over emotion
            classes. `success=False` with an error message on failure
            (e.g. degenerate/empty crop) — callers must handle this,
            not assume every frame yields a valid prediction.
        """
        try:
            if face_crop_bgr.size == 0:
                return EmotionResult(success=False, error="Empty face crop")

            import cv2
            from PIL import Image

            rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            predictions = self._classifier(pil_image)
            # `predictions` is a list of {"label": str, "score": float}, all classes
            probabilities = {p["label"].lower(): float(p["score"]) for p in predictions}
            top_emotion = max(probabilities, key=probabilities.get)

            # Sanity check: a proper softmax distribution over mutually
            # exclusive emotion classes must sum to ~1. If this ever
            # fails again (e.g. a different pretrained model with a
            # different default config), fail loudly here rather than
            # silently feeding an invalid distribution into temporal
            # aggregation and scoring.
            prob_sum = sum(probabilities.values())
            if not (0.95 <= prob_sum <= 1.05):
                return EmotionResult(
                    success=False,
                    error=f"Emotion probabilities do not sum to ~1 (sum={prob_sum:.3f}) — check pipeline activation function.",
                )

            return EmotionResult(success=True, probabilities=probabilities, top_emotion=top_emotion)

        except Exception as exc:  # noqa: BLE001
            return EmotionResult(success=False, error=str(exc))
