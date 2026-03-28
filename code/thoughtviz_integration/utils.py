"""
ThoughtViz integration utilities: path resolution, validation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from . import get_thoughtviz_root

log = logging.getLogger("thoughtviz_integration")


def resolve_thoughtviz_paths(
    data_dir: Optional[str] = None,
    image_dir: Optional[str] = None,
    eeg_model_path: Optional[str] = None,
    gan_model_path: Optional[str] = None,
) -> dict:
    """
    Resolve paths relative to ThoughtViz root.
    Returns dict with keys: root, data_dir, image_dir, eeg_model_path, gan_model_path.
    """
    root = get_thoughtviz_root()
    out = {
        "root": root,
        "data_dir": None,
        "image_dir": None,
        "eeg_model_path": None,
        "gan_model_path": None,
    }
    if root is None:
        log.warning("ThoughtViz root not found (code/ThoughtViz or codes/ThoughtViz).")
        return out
    root = Path(root)
    out["data_dir"] = Path(data_dir) if data_dir else root / "data"
    out["image_dir"] = Path(image_dir) if image_dir else root / "training" / "images"
    out["eeg_model_path"] = Path(eeg_model_path) if eeg_model_path else root / "models" / "eeg_models" / "image" / "run_final.h5"
    out["gan_model_path"] = Path(gan_model_path) if gan_model_path else root / "models" / "gan_models" / "final" / "image" / "generator.model"
    return out


def check_thoughtviz_available() -> bool:
    """Return True if ThoughtViz root exists and has expected structure."""
    root = get_thoughtviz_root()
    if root is None:
        return False
    root = Path(root)
    if not (root / "training").is_dir():
        return False
    if not (root / "testing").is_dir():
        return False
    return True
