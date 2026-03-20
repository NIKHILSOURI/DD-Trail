from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from .benchmark_config import BenchmarkConfig
from .segmentation_metrics import compare_instances
from .segmentation_model import SegmentationModel
from .utils import ensure_dir, save_json, setup_logger

log = setup_logger(__name__)


def _read_img(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def _sample_dirs(output_dir: Path, dataset_name: str) -> List[Path]:
    base = output_dir / dataset_name
    if not base.exists():
        return []
    return [p for p in sorted(base.iterdir()) if p.is_dir() and p.name.startswith("sample_")]


def run_segmentation_eval(output_dir: Path, dataset_name: str, config: BenchmarkConfig) -> Dict[str, Any]:
    seg = SegmentationModel(config)
    rows: List[Dict[str, Any]] = []
    for sdir in _sample_dirs(output_dir, dataset_name):
        root = sdir / "segmentation"
        ensure_dir(root)
        p = {
            "gt": sdir / "ground_truth.png",
            "thoughtviz": sdir / "thoughtviz.png",
            "dreamdiffusion": sdir / "dreamdiffusion.png",
            "sarhm": sdir / "sarhm.png",
        }
        if not p["gt"].exists():
            continue
        gt = seg.detect_and_segment(_read_img(p["gt"]), root / "gt", "gt")
        save_json(gt, root / "gt" / "segmentation.json")
        compare = {}
        for m in ("thoughtviz", "dreamdiffusion", "sarhm"):
            if not p[m].exists():
                compare[m] = {"status": "missing_image"}
                continue
            pred = seg.detect_and_segment(_read_img(p[m]), root / m, m)
            save_json(pred, root / m / "segmentation.json")
            met = compare_instances(gt, pred)
            met.update({"sample_id": sdir.name, "model": m, "dataset": dataset_name, "status": "ok"})
            rows.append(met)
            compare[m] = met
        save_json(compare, root / "segmentation_comparison.json")
    out_root = output_dir.parent / "segmentation_metrics" / dataset_name
    ensure_dir(out_root)
    if rows:
        keys = sorted(rows[0].keys())
        with open(out_root / "segmentation_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    agg: Dict[str, Any] = {"dataset": dataset_name, "n_rows": len(rows), "by_model": {}}
    for m in ("thoughtviz", "dreamdiffusion", "sarhm"):
        mr = [r for r in rows if r["model"] == m]
        if not mr:
            continue
        def _mean(k: str) -> float:
            return float(np.mean([x[k] for x in mr]))
        agg["by_model"][m] = {
            "label_precision_mean": _mean("label_precision"),
            "label_recall_mean": _mean("label_recall"),
            "label_f1_mean": _mean("label_f1"),
            "label_set_iou_mean": _mean("label_set_iou"),
            "matched_mask_iou_mean": _mean("matched_mask_iou_mean"),
            "matched_dice_mean": _mean("matched_dice_mean"),
            "hallucination_rate_mean": _mean("hallucination_rate"),
            "omission_rate_mean": _mean("omission_rate"),
            "n": len(mr),
        }
    save_json(agg, out_root / "segmentation_metrics.json")
    return agg
