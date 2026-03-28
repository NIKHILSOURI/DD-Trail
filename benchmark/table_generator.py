"""
Generate benchmark tables (TABLE 1-6) from metrics and timing outputs.
MSC: not defined in codebase; column included as NA unless added later.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import ensure_dir, setup_logger

log = setup_logger(__name__)


def load_metrics_summary(output_dir: Path, dataset_name: str) -> Dict[str, Any]:
    """Load metrics_summary.json for a dataset."""
    p = output_dir / dataset_name / "metrics_summary.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_table_1_2(metrics: Dict[str, Any], dataset_name: str) -> List[Dict[str, Any]]:
    """TABLE 1 (ImageNet-EEG) or TABLE 2 (ThoughtViz): Model, MSE, SSIM, PCC, CLIP, FID, Top-1, Top-5, MSC."""
    rows = []
    for model, m in metrics.items():
        rows.append({
            "Model": model,
            "Dataset": dataset_name,
            "MSE": m.get("mse_mean", "NA"),
            "SSIM": m.get("ssim_mean", "NA"),
            "PCC": m.get("pcc_mean", "NA"),
            "CLIP_similarity": m.get("clip_sim_mean", "NA"),
            "FID": m.get("fid", "NA"),
            "Top-1": m.get("top1_acc", "NA"),
            "Top-5": m.get("top5_acc", "NA"),
            "MSC": "NA",  # Not defined in project; document in thesis
        })
    return rows


def write_tables_csv(rows: List[Dict], path: Path) -> None:
    """Write rows to CSV."""
    if not rows:
        return
    import csv
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def generate_all_tables(output_dir: str | Path, out_tables_dir: Optional[str | Path] = None) -> None:
    """Load metrics from output_dir and write TABLE 1, 2, 3 (cross-dataset), 4 (timing if present)."""
    output_dir = Path(output_dir)
    out_tables_dir = out_tables_dir or output_dir / "tables"
    ensure_dir(out_tables_dir)
    for ds in ("imagenet_eeg", "thoughtviz"):
        metrics = load_metrics_summary(output_dir, ds)
        if not metrics:
            continue
        rows = build_table_1_2(metrics, ds)
        write_tables_csv(rows, Path(out_tables_dir) / ("table_%s.csv" % ds))
    # Summary comparison table
    summary_rows = []
    seg_rows = []
    for ds in ("imagenet_eeg", "thoughtviz"):
        sp = output_dir.parent / "summary_metrics" / ds / "summary_metrics.json"
        if sp.exists():
            with open(sp, "r", encoding="utf-8") as f:
                s = json.load(f)
            for m, v in s.get("by_model", {}).items():
                summary_rows.append({
                    "model": m,
                    "dataset": ds,
                    "summary_semantic_similarity_mean": v.get("summary_semantic_similarity_mean", "NA"),
                    "clip_text_image_score_mean": v.get("clip_text_image_score_mean", "NA"),
                    "object_mention_f1_mean": v.get("object_mention_f1_mean", "NA"),
                    "attribute_overlap_mean": v.get("attribute_overlap_mean", "NA"),
                })
        gp = output_dir.parent / "segmentation_metrics" / ds / "segmentation_metrics.json"
        if gp.exists():
            with open(gp, "r", encoding="utf-8") as f:
                g = json.load(f)
            for m, v in g.get("by_model", {}).items():
                seg_rows.append({
                    "model": m,
                    "dataset": ds,
                    "label_precision_mean": v.get("label_precision_mean", "NA"),
                    "label_recall_mean": v.get("label_recall_mean", "NA"),
                    "label_f1_mean": v.get("label_f1_mean", "NA"),
                    "label_set_iou_mean": v.get("label_set_iou_mean", "NA"),
                    "matched_mask_iou_mean": v.get("matched_mask_iou_mean", "NA"),
                    "matched_dice_mean": v.get("matched_dice_mean", "NA"),
                    "hallucination_rate_mean": v.get("hallucination_rate_mean", "NA"),
                    "omission_rate_mean": v.get("omission_rate_mean", "NA"),
                })
    write_tables_csv(summary_rows, Path(out_tables_dir) / "table_summary_comparison.csv")
    write_tables_csv(seg_rows, Path(out_tables_dir) / "table_segmentation_comparison.csv")
    log.info("Tables written to %s", out_tables_dir)
