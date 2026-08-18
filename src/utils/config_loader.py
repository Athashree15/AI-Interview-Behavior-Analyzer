"""
Config loading utility.

Every script imports `load_config()` from here instead of parsing
YAML itself, and instead of hardcoding any path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load the project YAML config.

    Args:
        config_path: path to config.yaml. Defaults to the repo's configs/config.yaml.

    Returns:
        Parsed config as a nested dict.

    Raises:
        FileNotFoundError: if the config file doesn't exist.
        yaml.YAMLError: if the file is not valid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Copy/edit configs/config.yaml and make sure dataset paths are filled in."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def resolve_device(device_setting: str) -> str:
    """
    Resolve 'auto' device setting to an actual torch device string.
    Falls back gracefully if torch isn't installed yet (e.g. during
    a Phase-1 inspection-only environment).
    """
    if device_setting != "auto":
        return device_setting

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
