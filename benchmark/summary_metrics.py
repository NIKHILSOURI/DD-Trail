from __future__ import annotations

from typing import Dict, List, Set


def overlap_prf(pred: Set[str], gt: Set[str]) -> Dict[str, float]:
    inter = len(pred.intersection(gt))
    p = inter / max(len(pred), 1)
    r = inter / max(len(gt), 1)
    f1 = 2 * p * r / max((p + r), 1e-8)
    return {"precision": p, "recall": r, "f1": f1}


def jaccard(a: Set[str], b: Set[str]) -> float:
    return len(a.intersection(b)) / max(len(a.union(b)), 1)


def compare_summary_dicts(gt: dict, pred: dict, semantic_sim: float, clip_self: float) -> dict:
    gt_obj = set(gt.get("objects_mentioned", []))
    pd_obj = set(pred.get("objects_mentioned", []))
    gt_attr = set(gt.get("attributes", []))
    pd_attr = set(pred.get("attributes", []))
    obj_prf = overlap_prf(pd_obj, gt_obj)
    attr_j = jaccard(pd_attr, gt_attr)
    return {
        "summary_semantic_similarity": semantic_sim,
        "clip_text_image_score": clip_self,
        "object_mention_precision": obj_prf["precision"],
        "object_mention_recall": obj_prf["recall"],
        "object_mention_f1": obj_prf["f1"],
        "attribute_overlap": attr_j,
    }
