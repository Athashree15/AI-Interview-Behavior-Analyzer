"""
Shared logging configuration.

Every script in the project calls `get_logger(__name__)` instead of
`print()`, so output is consistent, timestamped, and (optionally)
written to a log file under `paths.logs_dir` from config.yaml.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str | None = None, log_filename: str = "run.log") -> logging.Logger:
    """
    Create (or fetch) a configured logger.

    Args:
        name: usually `__name__` of the calling module.
        log_dir: if provided, logs are also written to `log_dir/log_filename`.
        log_filename: filename used inside `log_dir`.

    Returns:
        A configured `logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid attaching duplicate handlers if get_logger() is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
