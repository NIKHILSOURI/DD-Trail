"""
Run core image metrics (MSE, SSIM, PCC, CLIP, FID, Top-1/Top-5) on benchmark outputs.
Reads results/benchmark_outputs/<dataset>/sample_*/ and writes CSV/JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .benchmark_config import BenchmarkConfig
from .caption_eval import run_caption_eval
from .segmentation_eval import run_segmentation_eval
from .utils import ensure_dir, setup_logger

log = setup_logger(__name__)


def collect_sample_paths(output_dir: Path, dataset_name: str) -> List[Dict[str, str]]:
    """Collect paths to ground_truth.png and each model's PNG per sample."""
    base = output_dir / dataset_name
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not d.name.startswith("sample_"):
            continue
        gt = d / "ground_truth.png"
        entry = {"sample_id": d.name, "ground_truth": str(gt) if gt.exists() else None}
        for m in ("thoughtviz", "dreamdiffusion", "sarhm"):
            p = d / ("%s.png" % m)
            if p.exists():
                entry[m] = str(p)
        out.append(entry)
    return out


def run_core_metrics(
    output_dir: str | Path,
    dataset_name: str,
    config: Optional[BenchmarkConfig] = None,
) -> Dict[str, Any]:
    """Compute SSIM, PCC, CLIP (and optionally FID, Top-K) per model; return dict and write CSV."""
    import sys
    code_dir = Path(__file__).resolve().parent.parent / "code"
    if code_dir.is_dir() and str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    try:
        from utils_eval import compute_metrics
    except ImportError:
        log.warning("utils_eval not found; skipping core metrics")
        return {}
    output_dir = Path(output_dir)
    samples = collect_sample_paths(output_dir, dataset_name)
    if not samples:
        log.warning("No samples in %s/%s", output_dir, dataset_name)
        return {}
    results = {}
    for model in ("thoughtviz", "dreamdiffusion", "sarhm"):
        gen_list = []
        gt_list = []
        for s in samples:
            gt_path = s.get("ground_truth")
            gen_path = s.get(model)
            if not gt_path or not gen_path:
                continue
            from PIL import Image
            gt_list.append(np.array(Image.open(gt_path).convert("RGB")))
            gen_list.append(np.array(Image.open(gen_path).convert("RGB")))
        if not gen_list or len(gen_list) != len(gt_list):
            continue
        m = compute_metrics(np.array(gen_list), np.array(gt_list))
        results[model] = m
    out_path = output_dir / dataset_name / "metrics_summary.json"
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Wrote %s", out_path)
    return results


def run_all_metrics(output_dir: str | Path, dataset_name: str, config: Optional[BenchmarkConfig] = None) -> Dict[str, Any]:
    """Run core metrics + mandatory summary + mandatory segmentation metrics."""
    cfg = config or BenchmarkConfig()
    core = run_core_metrics(output_dir, dataset_name, cfg)
    summary = run_caption_eval(Path(output_dir), dataset_name, config=cfg) if cfg.summary_enabled else {}
    seg = run_segmentation_eval(Path(output_dir), dataset_name, config=cfg) if cfg.segmentation_enabled else {}
    return {"core": core, "summary": summary, "segmentation": seg}
