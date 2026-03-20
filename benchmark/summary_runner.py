from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from .benchmark_config import BenchmarkConfig
from .summary_metrics import compare_summary_dicts
from .summary_model import SummaryModel
from .utils import ensure_dir, load_json, save_json, setup_logger

log = setup_logger(__name__)


def _read_img(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def _sample_dirs(output_dir: Path, dataset_name: str) -> List[Path]:
    base = output_dir / dataset_name
    if not base.exists():
        return []
    return [p for p in sorted(base.iterdir()) if p.is_dir() and p.name.startswith("sample_")]


def run_summary_eval(output_dir: Path, dataset_name: str, config: BenchmarkConfig) -> Dict[str, Any]:
    model = SummaryModel(config)
    rows: List[Dict[str, Any]] = []
    samples = _sample_dirs(output_dir, dataset_name)
    for sdir in samples:
        ssum = sdir / "summaries"
        ensure_dir(ssum)
        paths = {
            "gt": sdir / "ground_truth.png",
            "thoughtviz": sdir / "thoughtviz.png",
            "dreamdiffusion": sdir / "dreamdiffusion.png",
            "sarhm": sdir / "sarhm.png",
        }
        if not paths["gt"].exists():
            continue
        gt_img = _read_img(paths["gt"])
        gt_sum = model.summarize(gt_img, "gt").to_dict()
        save_json(gt_sum, ssum / "summary_gt.json")
        per_cmp = {}
        for m in ("thoughtviz", "dreamdiffusion", "sarhm"):
            if not paths[m].exists():
                per_cmp[m] = {"status": "missing_image"}
                continue
            img = _read_img(paths[m])
            pred_sum = model.summarize(img, m).to_dict()
            save_json(pred_sum, ssum / ("summary_%s.json" % m))
            sem = model.sentence_cosine(gt_sum["detailed_caption"], pred_sum["detailed_caption"])
            clip_self = model.clip_text_image_score(img, pred_sum["detailed_caption"])
            cmp_row = compare_summary_dicts(gt_sum, pred_sum, sem, clip_self)
            cmp_row.update({"sample_id": sdir.name, "model": m, "dataset": dataset_name, "status": "ok"})
            rows.append(cmp_row)
            per_cmp[m] = cmp_row
        save_json(per_cmp, ssum / "summary_comparison.json")
    # aggregate
    out_root = output_dir.parent / "summary_metrics" / dataset_name
    ensure_dir(out_root)
    if rows:
        keys = sorted(rows[0].keys())
        with open(out_root / "summary_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    agg: Dict[str, Any] = {"dataset": dataset_name, "n_rows": len(rows), "by_model": {}}
    for m in ("thoughtviz", "dreamdiffusion", "sarhm"):
        mr = [r for r in rows if r["model"] == m]
        if not mr:
            continue
        agg["by_model"][m] = {
            "summary_semantic_similarity_mean": float(np.mean([r["summary_semantic_similarity"] for r in mr])),
            "clip_text_image_score_mean": float(np.mean([r["clip_text_image_score"] for r in mr])),
            "object_mention_f1_mean": float(np.mean([r["object_mention_f1"] for r in mr])),
            "attribute_overlap_mean": float(np.mean([r["attribute_overlap"] for r in mr])),
            "n": len(mr),
        }
    save_json(agg, out_root / "summary_metrics.json")
    return agg
