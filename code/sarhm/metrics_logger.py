"""
SAR-HM metrics: retrieval accuracy, attention entropy, confidence statistics.
Helper to compute from cond_stage_model._sarhm_extra and labels.
Writes CSV/JSON to output_path for thesis reproducibility.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def retrieval_accuracy(attn: torch.Tensor, labels: torch.Tensor) -> float:
    """argmax(attn) == labels, mean. attn [B,K], labels [B] long."""
    if attn is None or labels is None:
        return float("nan")
    pred = attn.argmax(dim=-1)
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)
    labels = labels.to(attn.device).long()
    return (pred == labels).float().mean().item()


def attention_entropy_mean(attn: Optional[torch.Tensor]) -> float:
    """Mean entropy over batch. attn [B,K]."""
    if attn is None:
        return float("nan")
    eps = 1e-12
    ent = -(attn * (attn + eps).log()).sum(dim=-1)
    return ent.mean().item()


def confidence_stats(confidence: Optional[torch.Tensor]) -> Dict[str, float]:
    """Mean, std, min, max of confidence."""
    if confidence is None:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    c = confidence.detach().float()
    return {"mean": c.mean().item(), "std": c.std().item(), "min": c.min().item(), "max": c.max().item()}


def log_hopfield_stats_once(
    extra: Dict[str, Any],
    labels: Optional[torch.Tensor] = None,
    step: int = 0,
    log_every: int = 50,
    is_smoke_test: bool = False,
) -> None:
    """Print Hopfield stats once per run (smoke test) or every log_every steps (training). Includes alpha mean/min/max and sample0 (conf, entropy)."""
    if not extra:
        return
    attn = extra.get("attn")
    conf = extra.get("confidence")
    alpha = extra.get("alpha")
    entropy = extra.get("entropy")
    # For smoke test: always log first batch; for training: every log_every steps or step 0
    if not (is_smoke_test or step % log_every == 0):
        return
    # Alpha stats once per run (mean/min/max)
    if alpha is not None:
        a = alpha.detach().float()
        print("alpha stats: mean=%.3f min=%.3f max=%.3f" % (a.mean().item(), a.min().item(), a.max().item()))
    # For 1 sample (sample0): conf, entropy
    if conf is not None and conf.numel() > 0:
        c0 = conf[0].item()
        e0 = entropy[0].item() if entropy is not None and entropy.numel() > 0 else float("nan")
        print("sample0: conf=%.3f entropy=%.3f" % (c0, e0))
    if attn is None:
        return
    eps = 1e-12
    top1 = attn.argmax(dim=-1)
    conf_vals = attn.max(dim=-1).values if conf is None else conf
    if entropy is None:
        entropy = -(attn * (attn + eps).log()).sum(dim=-1)
    top1_str = ",".join(str(int(i)) for i in top1.cpu().tolist()[:8])
    if top1.numel() > 8:
        top1_str += "..."
    conf_mean = conf_vals.mean().item() if conf_vals is not None else float("nan")
    ent_mean = entropy.mean().item() if entropy is not None else float("nan")
    print(f"Hopfield: top1={top1_str} conf={conf_mean:.2f} entropy={ent_mean:.2f}")
    if labels is not None:
        try:
            lbl = labels.to(attn.device).long()
            if lbl.dim() == 0:
                lbl = lbl.unsqueeze(0)
            acc = (top1 == lbl).float().mean().item()
            print(f"Hopfield retrieval acc (batch)={acc:.2f}")
        except Exception:
            print("Hopfield retrieval acc (batch)=NA")
    else:
        print("Hopfield retrieval acc (batch)=NA")


def sarhm_metrics_from_extra(
    extra: Dict[str, Any],
    labels: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Build dict of SAR-HM metrics from cond_stage_model._sarhm_extra and optional labels."""
    out = {}
    attn = extra.get("attn")
    conf = extra.get("confidence")
    out["sarhm_attention_entropy"] = attention_entropy_mean(attn)
    out.update({f"sarhm_confidence_{k}": v for k, v in confidence_stats(conf).items()})
    if attn is not None and labels is not None:
        out["sarhm_retrieval_acc"] = retrieval_accuracy(attn, labels)
    else:
        out["sarhm_retrieval_acc"] = float("nan")
    return out


def append_metrics_json(metrics: Dict[str, float], path: str, step: Optional[int] = None) -> None:
    """Append a row to a JSONL-like file or update a single JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"step": step, **metrics} if step is not None else metrics
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def write_metrics_csv(rows: list, path: str, fieldnames: Optional[list] = None) -> None:
    """Write list of metric dicts to CSV."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = fieldnames or sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_sarhm_metrics(
    output_path: str,
    metrics_dict: Dict[str, float],
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    mode: str = "train",
) -> None:
    """Save SAR-HM metrics to output_path/sarhm_metrics.json and sarhm_metrics.csv."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    row = dict(metrics_dict)
    if step is not None:
        row["step"] = step
    if epoch is not None:
        row["epoch"] = epoch
    row["mode"] = mode
    json_path = out / "sarhm_metrics.jsonl"
    append_metrics_json(row, str(json_path), step=None)
    csv_path = out / "sarhm_metrics.csv"
    if not csv_path.exists():
        write_metrics_csv([row], str(csv_path))
    else:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            import csv
            w = csv.DictWriter(f, fieldnames=sorted(row.keys()), extrasaction="ignore")
            w.writerow(row)


# Ablation results table (professor-friendly artifact)
ABLATION_CSV_COLUMNS = [
    "mode",
    "clip_similarity",
    "fid",
    "ssim",
    "pcc",
    "retrieval_acc",
    "mean_confidence",
    "mean_entropy",
    "timestamp",
]


def append_ablation_results_row(
    run_dir: str,
    mode: str,
    clip_similarity: Optional[float] = None,
    fid: Optional[float] = None,
    ssim: Optional[float] = None,
    pcc: Optional[float] = None,
    retrieval_acc: Optional[float] = None,
    mean_confidence: Optional[float] = None,
    mean_entropy: Optional[float] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Append one row to outputs/<run_dir>/ablation_results.csv.
    Unavailable metrics can be omitted; they are written as NA.
    Returns path to the CSV file.
    """
    import csv
    from datetime import datetime

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    csv_path = run_path / "ablation_results.csv"
    row = {
        "mode": mode,
        "clip_similarity": clip_similarity if clip_similarity is not None else "NA",
        "fid": fid if fid is not None else "NA",
        "ssim": ssim if ssim is not None else "NA",
        "pcc": pcc if pcc is not None else "NA",
        "retrieval_acc": retrieval_acc if retrieval_acc is not None else "NA",
        "mean_confidence": mean_confidence if mean_confidence is not None else "NA",
        "mean_entropy": mean_entropy if mean_entropy is not None else "NA",
        "timestamp": timestamp or datetime.now().isoformat(),
    }
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ABLATION_CSV_COLUMNS, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    return str(csv_path)
