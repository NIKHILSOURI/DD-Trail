"""
Build qualitative comparison panels: Ground Truth | ThoughtViz | DreamDiffusion | SAR-HM.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

from .utils import ensure_dir, setup_logger

log = setup_logger(__name__)


def build_comparison_panel(
    sample_dir: Path,
    output_path: Path,
    models: Optional[List[str]] = None,
) -> None:
    """Create one horizontal panel: GT | model1 | model2 | model3 for one sample."""
    models = models or ["thoughtviz", "dreamdiffusion", "sarhm"]
    imgs = []
    gt = sample_dir / "ground_truth.png"
    if gt.exists():
        imgs.append(np.array(Image.open(gt).convert("RGB")))
    for m in models:
        p = sample_dir / ("%s.png" % m)
        if p.exists():
            imgs.append(np.array(Image.open(p).convert("RGB")))
    if not imgs:
        return
    # Resize to same height (e.g. 256)
    h = 256
    out = []
    for arr in imgs:
        pil = Image.fromarray(arr)
        w = int(arr.shape[1] * h / arr.shape[0])
        pil = pil.resize((w, h), Image.BILINEAR)
        out.append(np.array(pil))
    concat = np.concatenate(out, axis=1)
    ensure_dir(output_path.parent)
    Image.fromarray(concat).save(output_path)
    log.info("Saved panel %s", output_path)


def run_visualization(output_dir: Path, dataset_name: str, max_panels: int = 10) -> None:
    """Build comparison panels for up to max_panels samples; save to output_dir/dataset_name/panels/."""
    base = output_dir / dataset_name
    if not base.is_dir():
        return
    panels_dir = base / "panels"
    ensure_dir(panels_dir)
    count = 0
    for d in sorted(base.iterdir()):
        if count >= max_panels:
            break
        if not d.is_dir() or not d.name.startswith("sample_"):
            continue
        build_comparison_panel(d, panels_dir / ("%s.png" % d.name))
        # Summary panel
        _build_summary_panel(d, panels_dir / ("%s_summary.png" % d.name))
        # Segmentation overlay panel
        _build_segmentation_overlay_panel(d, panels_dir / ("%s_segmentation.png" % d.name))
        count += 1


def _caption_for(sample_dir: Path, key: str) -> str:
    import json
    p = sample_dir / "summaries" / ("summary_%s.json" % key)
    if key == "gt":
        p = sample_dir / "summaries" / "summary_gt.json"
    if not p.exists():
        return "no summary"
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return (d.get("short_caption") or d.get("detailed_caption") or "no caption")[:80]
    except Exception:
        return "summary read error"


def _build_summary_panel(sample_dir: Path, output_path: Path) -> None:
    labels = [("ground_truth.png", "gt"), ("thoughtviz.png", "thoughtviz"), ("dreamdiffusion.png", "dreamdiffusion"), ("sarhm.png", "sarhm")]
    blocks = []
    for fname, key in labels:
        p = sample_dir / fname
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR)
        canvas = Image.new("RGB", (256, 320), color=(20, 20, 20))
        canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((6, 262), key, fill=(255, 255, 0))
        d.text((6, 282), _caption_for(sample_dir, key), fill=(220, 220, 220))
        blocks.append(np.array(canvas))
    if not blocks:
        return
    out = np.concatenate(blocks, axis=1)
    ensure_dir(output_path.parent)
    Image.fromarray(out).save(output_path)


def _build_segmentation_overlay_panel(sample_dir: Path, output_path: Path) -> None:
    paths = [
        sample_dir / "segmentation" / "gt" / "overlays" / "overlay.png",
        sample_dir / "segmentation" / "thoughtviz" / "overlays" / "overlay.png",
        sample_dir / "segmentation" / "dreamdiffusion" / "overlays" / "overlay.png",
        sample_dir / "segmentation" / "sarhm" / "overlays" / "overlay.png",
    ]
    imgs = []
    for p in paths:
        if p.exists():
            imgs.append(np.array(Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR)))
    if not imgs:
        return
    out = np.concatenate(imgs, axis=1)
    ensure_dir(output_path.parent)
    Image.fromarray(out).save(output_path)
