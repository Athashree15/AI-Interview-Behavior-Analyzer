"""
One-time utility: extract val/test annotation zip files for ChaLearn
First Impressions V2. The training annotation loaded fine as a plain
pickle, but validation/test shipped as .zip archives (containing the
pickle inside) — this unpacks them in place next to the original zip.

Usage:
    python scripts/extract_chalearn_annotations.py

After running this, re-run scripts/inspect_chalearn.py — it should
report all three (train/val/test) annotation sets loading successfully.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config_loader import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402


def main() -> None:
    config = load_config()
    logger = get_logger(__name__, log_dir=config["paths"]["logs_dir"], log_filename="extract_annotations.log")

    root = Path(config["datasets"]["chalearn_fi_v2"]["root"])
    zip_files = list(root.rglob("*annotation*.zip"))

    if not zip_files:
        logger.info("No annotation zip files found — nothing to extract (or already extracted).")
        return

    for zip_path in zip_files:
        extract_dir = zip_path.parent
        logger.info(f"Extracting {zip_path.name} -> {extract_dir}")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                logger.info(f"  Contents: {names}")
                zf.extractall(extract_dir)
            logger.info(f"  Extracted successfully.")
        except zipfile.BadZipFile:
            logger.error(
                f"  {zip_path.name} is not a valid zip file, or requires a password "
                "(the original ChaLearn competition test-set annotations were sometimes "
                "distributed encrypted during the active competition). If this fails, "
                "open the file manually with 7-Zip and check for a password prompt."
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  Failed to extract {zip_path.name}: {exc}")

    logger.info("Done. Re-run scripts/inspect_chalearn.py to confirm the extracted files load correctly.")


if __name__ == "__main__":
    main()
