from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np


def _set_metrics(pred: Set[str], gt: Set[str]) -> Dict[str, float]:
    inter = len(pred.intersection(gt))
    p = inter / max(len(pred), 1)
    r = inter / max(len(gt), 1)
    f1 = 2 * p * r / max((p + r), 1e-8)
    j = inter / max(len(pred.union(gt)), 1)
    return {"label_precision": p, "label_recall": r, "label_f1": f1, "label_set_iou": j}


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(aa + ba - inter, 1e-8)


def compare_instances(gt: dict, pred: dict) -> Dict[str, float]:
    gt_labels = set(gt.get("labels_norm", []))
    pd_labels = set(pred.get("labels_norm", []))
    out = _set_metrics(pd_labels, gt_labels)
    gti = gt.get("instances", [])
    pdi = pred.get("instances", [])
    used = set()
    ious = []
    for gi in gti:
        best_j = -1
        best_score = 0.0
        for j, pj in enumerate(pdi):
            if j in used:
                continue
            if gi.get("label_norm") != pj.get("label_norm"):
                continue
            s = _bbox_iou(gi.get("bbox_xyxy", [0, 0, 0, 0]), pj.get("bbox_xyxy", [0, 0, 0, 0]))
            if s > best_score:
                best_score = s
                best_j = j
        if best_j >= 0:
            used.add(best_j)
            ious.append(best_score)
    out["matched_bbox_iou_mean"] = float(np.mean(ious)) if ious else 0.0
    out["matched_mask_iou_mean"] = out["matched_bbox_iou_mean"]  # fallback
    out["matched_dice_mean"] = (2 * out["matched_bbox_iou_mean"]) / max(1 + out["matched_bbox_iou_mean"], 1e-8)
    out["hallucination_rate"] = max(len(pdi) - len(used), 0) / max(len(pdi), 1)
    out["omission_rate"] = max(len(gti) - len(used), 0) / max(len(gti), 1)
    out["count_diff_total"] = abs(len(gti) - len(pdi))
    return out
