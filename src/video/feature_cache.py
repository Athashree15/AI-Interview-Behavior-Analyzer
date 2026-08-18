"""
On-disk caching for per-video features (Module 21).

Prevents re-running expensive face detection / emotion inference /
head-pose estimation on a video that's already been processed.
Cache layout:

    cache_dir/
        <video_hash>/
            visual_features.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class VisualFeatureCache:
    """Simple JSON-based cache for per-video visual pipeline outputs."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, video_hash: str) -> Path:
        video_cache_dir = self.cache_dir / video_hash
        video_cache_dir.mkdir(parents=True, exist_ok=True)
        return video_cache_dir / "visual_features.json"

    def get(self, video_hash: str) -> dict[str, Any] | None:
        """Return cached results for a video hash, or None if not cached."""
        path = self._path_for(video_hash)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt cache entry — treat as a cache miss rather than crashing.
            return None

    def set(self, video_hash: str, data: dict[str, Any]) -> None:
        """Write results for a video hash to the cache."""
        path = self._path_for(video_hash)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
