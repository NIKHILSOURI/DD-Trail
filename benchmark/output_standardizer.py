"""
Standardized output layout: results/benchmark_outputs/<dataset>/sample_<id>/ground_truth.png, thoughtviz.png, dreamdiffusion.png, sarhm.png, metadata.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from .benchmark_config import BenchmarkConfig
from .utils import ensure_dir, setup_logger

log = setup_logger(__name__)

EVAL_SIZE = 256  # Standard size for metric evaluation


def _to_uint8(img: Union[np.ndarray, "Image.Image"]) -> np.ndarray:
    """Convert to (H, W, 3) uint8."""
    if hasattr(img, "numpy"):
        img = img.cpu().numpy()
    if hasattr(img, "numpy") and callable(img.numpy):
        img = img.numpy()
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = np.asarray(img)
    # Accept accidental leading singleton batch dims from model wrappers.
    while img.ndim > 3 and img.shape[0] == 1:
        img = img[0]
    # Normalize common grayscale layouts before PIL conversion.
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def save_image_standardized(
    img: Union[np.ndarray, "Image.Image"],
    path: Path,
    eval_size: Optional[int] = None,
) -> None:
    """Save image; optionally resize to eval_size for consistent metrics."""
    arr = _to_uint8(img)
    if eval_size and (arr.shape[0] != eval_size or arr.shape[1] != eval_size):
        from PIL import Image as PImage
        pil = PImage.fromarray(arr)
        pil = pil.resize((eval_size, eval_size), PImage.BILINEAR)
        arr = np.array(pil)
    ensure_dir(path.parent)
    Image.fromarray(arr).save(path)


def write_sample_outputs(
    sample_id: str,
    dataset_name: str,
    output_dir: str | Path,
    ground_truth: Optional[Union[np.ndarray, "Image.Image"]] = None,
    thoughtviz: Optional[Union[np.ndarray, "Image.Image"]] = None,
    dreamdiffusion: Optional[Union[np.ndarray, "Image.Image"]] = None,
    sarhm: Optional[Union[np.ndarray, "Image.Image"]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    eval_size: int = EVAL_SIZE,
    merge_with_existing: bool = True,
) -> Path:
    """
    Write one sample's outputs to output_dir/dataset_name/sample_<id>/.
    If merge_with_existing, load existing metadata.json and merge; only write image keys that are provided.
    Returns path to sample dir.
    """
    base = Path(output_dir) / dataset_name / ("sample_%s" % sample_id.replace("/", "_"))
    ensure_dir(base)
    if ground_truth is not None:
        save_image_standardized(ground_truth, base / "ground_truth.png", eval_size=eval_size)
    if thoughtviz is not None:
        save_image_standardized(thoughtviz, base / "thoughtviz.png", eval_size=eval_size)
    if dreamdiffusion is not None:
        save_image_standardized(dreamdiffusion, base / "dreamdiffusion.png", eval_size=eval_size)
    if sarhm is not None:
        save_image_standardized(sarhm, base / "sarhm.png", eval_size=eval_size)
    meta = dict(metadata or {}, sample_id=sample_id, dataset=dataset_name)
    if merge_with_existing and (base / "metadata.json").exists():
        try:
            with open(base / "metadata.json", "r", encoding="utf-8") as f:
                existing = json.load(f)
            meta = {**existing, **meta}
        except Exception:
            pass
    with open(base / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return base
