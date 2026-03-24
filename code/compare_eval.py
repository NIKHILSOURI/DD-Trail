"""
Dual evaluation: BASELINE vs SAR-HM (or SAR-HM++) on the same dataset/seed/samples.
Generates exactly N test images per mode, saves comparison grids and metrics.

Ready-to-run (from repo root, with code on PYTHONPATH):
  python code/compare_eval.py --dataset EEG \\
    --splits_path datasets/block_splits_by_image_single.pth \\
    --eeg_signals_path datasets/eeg_5_95_std.pth --config_patch pretrains/models/config15.yaml \\
    --baseline_ckpt exps/results/generation/<baseline_run>/checkpoint.pth \\
    --sarhm_ckpt exps/results/generation/<sarhm_run>/checkpoint.pth \\
    --sarhm_proto exps/results/generation/<sarhm_run>/prototypes.pt \\
    --n_samples 20 --ddim_steps 250 --seed 2022 --out_dir results/compare_eval

SAR-HM++: You can pass a SAR-HM++ checkpoint as --sarhm_ckpt. semantic_prototypes.pt is then
  loaded from the checkpoint directory automatically (see utils_eval.load_model). --sarhm_proto
  is ignored for SAR-HM++ (class prototypes); use the same run dir that contains semantic_prototypes.pt.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import Subset

# Add code dir for imports
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from dataset import create_EEG_dataset
from utils_eval import (
    set_seed,
    load_model,
    save_grid,
    compute_metrics,
)


def _log(tag: str, msg: str) -> None:
    print("[COMPARE] [%s] %s" % (tag, msg))


# Stubs for unpickling checkpoints saved from eeg_ldm/gen_eval_eeg (they reference __main__.normalize, channel_last)
def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, "h w c -> c h w")
    img = torch.tensor(img)
    return img * 2.0 - 1.0


def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, "c h w -> h w c")


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dual evaluation: BASELINE vs SAR-HM")
    p.add_argument("--dataset", type=str, default="EEG")
    p.add_argument("--splits_path", type=str, required=True)
    p.add_argument("--eeg_signals_path", type=str, required=True)
    p.add_argument("--config_patch", type=str, default="pretrains/models/config15.yaml")
    p.add_argument("--baseline_ckpt", type=str, required=True)
    p.add_argument("--sarhm_ckpt", type=str, required=True)
    p.add_argument("--sarhm_proto", type=str, default=None, help="Class prototypes for SAR-HM (ignored for SAR-HM++).")
    p.add_argument("--sarhmpp_proto", type=str, default=None, help="semantic_prototypes.pt for SAR-HM++ (overrides auto-detect from ckpt dir).")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--ddim_steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--force_fp32", action="store_true")
    p.add_argument("--fail_if_proto_missing", action="store_true", help="If set, error when SAR-HM needs proto but --sarhm_proto not given")
    p.add_argument("--use_train_split", action="store_true", help="Use train split instead of test")
    p.add_argument("--pretrain_root", type=str, default="pretrains")
    p.add_argument("--imagenet_path", type=str, default=None, help="ImageNet root for EEG GT images (required when dataset=EEG). Set or use env IMAGENET_PATH.")
    return p


def _identity(x):
    return x


def main() -> None:
    args = get_parser().parse_args()
    if getattr(args, 'imagenet_path', None) is None and os.environ.get('IMAGENET_PATH'):
        args.imagenet_path = os.environ.get('IMAGENET_PATH')
    if args.dataset == 'EEG' and (not getattr(args, 'imagenet_path', None) or not str(getattr(args, 'imagenet_path', '')).strip()):
        print("ERROR: dataset=EEG requires imagenet_path for real GT images.")
        print("  Pass --imagenet_path /path/to/ILSVRC2012 or set IMAGENET_PATH.")
        sys.exit(1)
    set_seed(args.seed)

    out_dir = args.out_dir
    if not out_dir:
        out_dir = os.path.join("results", "compare_eval", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    _log("MAIN", "out_dir=%s" % out_dir)

    # Dataset (same transform as gen_eval_eeg: normalize -> Resize -> channel_last)
    import torchvision.transforms as T
    def _normalize(img):
        if img.shape[-1] == 3:
            img = rearrange(img, "h w c -> c h w")
        img = torch.tensor(img)
        return img * 2.0 - 1.0
    def _channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, "c h w -> h w c")
    img_transform = T.Compose([_normalize, T.Resize((512, 512)), _channel_last])
    dataset_train, dataset_test = create_EEG_dataset(
        eeg_signals_path=args.eeg_signals_path,
        splits_path=args.splits_path,
        imagenet_path=args.imagenet_path,
        image_transform=[img_transform, img_transform],
        subject=4,
    )
    dataset_eval = dataset_train if getattr(args, "use_train_split", False) else dataset_test
    if len(dataset_eval) == 0:
        raise RuntimeError("[COMPARE] Test split is empty. Check splits_path and eeg_signals_path.")
    n = min(args.n_samples, len(dataset_eval))
    if n < args.n_samples:
        _log("MAIN", "n_samples clamped to %d (dataset size %d)" % (n, len(dataset_eval)))
    indices = list(range(n))
    subset = Subset(dataset_eval, indices)
    num_voxels = dataset_eval.dataset.data_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_root = getattr(args, "pretrain_root", "pretrains")

    # ---------- Baseline ----------
    _log("BASELINE", "loading %s" % args.baseline_ckpt)
    model_baseline, config_b = load_model(
        args.baseline_ckpt,
        args.config_patch,
        device,
        pretrain_root=pretrain_root,
        force_fp32=args.force_fp32,
        num_voxels=num_voxels,
    )
    model_baseline.model.ddim_steps = args.ddim_steps
    sf = getattr(model_baseline.model, "scale_factor", None)
    sf_val = float(sf.item()) if sf is not None and hasattr(sf, "item") else (float(sf) if sf is not None else None)
    print("[SCALE_FACTOR] value=%s match=%s" % (sf_val, (abs(sf_val - 0.18215) < 1e-5) if sf_val is not None else False))
    _log("BASELINE", "scale_factor=%s" % sf_val)

    _log("BASELINE", "generating n=%d" % n)
    grid_b, samples_b = model_baseline.generate(
        subset, num_samples=1, ddim_steps=args.ddim_steps, HW=None, limit=n,
        state=None, output_path=None, cfg_scale=1.0, cfg_uncond="zeros",
    )
    # samples_b: (N, 1+1, C, H, W) -> (N, 2, C, H, W); first is GT, second is gen
    gen_baseline = samples_b[:, 1]  # (N, C, H, W)
    gen_baseline_np = rearrange(gen_baseline, "n c h w -> n h w c")
    gt_from_baseline = samples_b[:, 0]
    gt_from_baseline_np = rearrange(gt_from_baseline, "n c h w -> n h w c")

    # ---------- SAR-HM ----------
    _log("SARHM", "loading %s" % args.sarhm_ckpt)
    set_seed(args.seed)
    model_sarhm, config_s = load_model(
        args.sarhm_ckpt,
        args.config_patch,
        device,
        pretrain_root=pretrain_root,
        force_fp32=args.force_fp32,
        num_voxels=num_voxels,
    )
    model_sarhm.model.ddim_steps = args.ddim_steps
    sf2 = getattr(model_sarhm.model, "scale_factor", None)
    sf2_val = float(sf2.item()) if sf2 is not None and hasattr(sf2, "item") else (float(sf2) if sf2 is not None else None)
    print("[SCALE_FACTOR] value=%s match=%s" % (sf2_val, (abs(sf2_val - 0.18215) < 1e-5) if sf2_val is not None else False))
    _log("SARHM", "scale_factor=%s" % sf2_val)

    csm = getattr(model_sarhm.model, "cond_stage_model", None)
    # SAR-HM++: optionally override semantic prototypes path (else use path set in load_model from ckpt dir)
    if csm is not None and getattr(csm, "use_sarhmpp", False) and getattr(args, "sarhmpp_proto", None) and os.path.isfile(args.sarhmpp_proto):
        bank = getattr(csm, "semantic_memory_bank", None)
        if bank is not None and hasattr(bank, "load_from_path") and bank.load_from_path(args.sarhmpp_proto):
            _log("SARHMPP", "loaded semantic_prototypes from %s" % args.sarhmpp_proto)
    if csm is not None and getattr(csm, "use_sarhm", False):
        if args.sarhm_proto and os.path.isfile(args.sarhm_proto):
            if getattr(csm, "sarhm_prototypes", None) is not None:
                ok = csm.sarhm_prototypes.load_from_path(args.sarhm_proto)
                if ok:
                    csm._proto_source = "loaded"
                    P = csm.sarhm_prototypes.P
                    if P is not None:
                        _log("PROTO", "loaded path=%s shape=%s dtype=%s finite=%s" % (
                            args.sarhm_proto, tuple(P.shape), str(P.dtype), torch.isfinite(P).all().item()))
                else:
                    _log("PROTO", "load_from_path returned False for %s" % args.sarhm_proto)
            else:
                _log("PROTO", "cond_stage_model has no sarhm_prototypes")
        else:
            if args.fail_if_proto_missing:
                raise RuntimeError("[COMPARE] SAR-HM requires prototypes but --sarhm_proto not provided or file missing. Set --sarhm_proto or do not use --fail_if_proto_missing.")
            _log("SARHM", "no proto path -> alpha=0 fallback (baseline-only conditioning)")
            if csm is not None:
                csm._baseline_only = True

    _log("SARHM", "generating n=%d" % n)
    set_seed(args.seed)
    grid_s, samples_s = model_sarhm.generate(
        subset, num_samples=1, ddim_steps=args.ddim_steps, HW=None, limit=n,
        state=None, output_path=None, cfg_scale=1.0, cfg_uncond="zeros",
    )
    gen_sarhm = samples_s[:, 1]
    gen_sarhm_np = rearrange(gen_sarhm, "n c h w -> n h w c")

    # ---------- Save samples ----------
    base_dir = os.path.join(out_dir, "baseline", "samples")
    sarhm_dir = os.path.join(out_dir, "sarhm", "samples")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(sarhm_dir, exist_ok=True)
    for i in range(n):
        Image.fromarray(gen_baseline_np[i].astype(np.uint8)).save(os.path.join(base_dir, "%04d.png" % i))
        Image.fromarray(gen_sarhm_np[i].astype(np.uint8)).save(os.path.join(sarhm_dir, "%04d.png" % i))

    # ---------- Grids ----------
    grids_dir = os.path.join(out_dir, "grids")
    os.makedirs(grids_dir, exist_ok=True)
    save_grid(gen_baseline_np, os.path.join(grids_dir, "baseline_grid.png"), nrow=5)
    save_grid(gen_sarhm_np, os.path.join(grids_dir, "sarhm_grid.png"), nrow=5)
    # Side-by-side: each row = baseline left, sarhm right (same sample id)
    side_by_side = []
    for i in range(n):
        side_by_side.append(gen_baseline_np[i])
        side_by_side.append(gen_sarhm_np[i])
    save_grid(side_by_side, os.path.join(grids_dir, "side_by_side.png"), nrow=2 * 5)

    # ---------- GT for metrics (use same GT from baseline run; 0-255) ----------
    gt_for_metrics = gt_from_baseline_np
    if args.imagenet_path is None:
        gt_for_metrics = None  # treat as no real GT -> NA for CLIP/SSIM/PCC

    # ---------- Metrics ----------
    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    m_baseline = compute_metrics(gen_baseline_np, gt_for_metrics, device=device)
    m_sarhm = compute_metrics(gen_sarhm_np, gt_for_metrics, device=device)

    def _num(x):
        if x == "NA":
            return float("nan")
        try:
            return float(x)
        except Exception:
            return float("nan")

    rows = [
        {"mode": "baseline", **{k: m_baseline.get(k, "NA") for k in ["ssim_mean", "pcc_mean", "clip_sim_mean", "mean_variance", "n_samples"]}},
        {"mode": "sarhm", **{k: m_sarhm.get(k, "NA") for k in ["ssim_mean", "pcc_mean", "clip_sim_mean", "mean_variance", "n_samples"]}},
    ]
    delta_row = {"mode": "delta"}
    for k in ["ssim_mean", "pcc_mean", "clip_sim_mean"]:
        b, s = _num(m_baseline.get(k)), _num(m_sarhm.get(k))
        if np.isfinite(b) and np.isfinite(s):
            delta_row[k] = s - b
        else:
            delta_row[k] = "NA"
    delta_row["mean_variance"] = _num(m_sarhm.get("mean_variance")) - _num(m_baseline.get("mean_variance")) if m_sarhm.get("mean_variance") != "NA" else "NA"
    delta_row["n_samples"] = n
    rows.append(delta_row)

    csv_path = os.path.join(metrics_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "ssim_mean", "pcc_mean", "clip_sim_mean", "mean_variance", "n_samples"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    _log("MAIN", "metrics written to %s" % csv_path)

    # ---------- Report ----------
    gt_note = "Ground-truth images from dataset (imagenet_path provided)." if args.imagenet_path else "No ImageNet path provided; CLIP/SSIM/PCC use dataset image (may be placeholder). Set --imagenet_path for real GT."
    report = [
        "# Compare Eval Report",
        "",
        "## Setup",
        "- baseline_ckpt: %s" % args.baseline_ckpt,
        "- sarhm_ckpt: %s" % args.sarhm_ckpt,
        "- sarhm_proto: %s" % (args.sarhm_proto or "not set"),
        "- n_samples: %d" % n,
        "- ddim_steps: %d" % args.ddim_steps,
        "- seed: %d" % args.seed,
        "- %s" % gt_note,
        "",
        "## Metrics",
        "| mode | ssim_mean | pcc_mean | clip_sim_mean | mean_variance | n_samples |",
        "|------|-----------|----------|----------------|---------------|-----------|",
    ]
    for r in rows:
        report.append("| %s | %s | %s | %s | %s | %s |" % (
            r.get("mode", ""),
            r.get("ssim_mean", "NA"),
            r.get("pcc_mean", "NA"),
            r.get("clip_sim_mean", "NA"),
            r.get("mean_variance", "NA"),
            r.get("n_samples", "NA"),
        ))
    report.append("")
    report.append("## Outputs")
    report.append("- Grids: %s" % grids_dir)
    report.append("- Baseline samples: %s" % base_dir)
    report.append("- SAR-HM samples: %s" % sarhm_dir)
    report_path = os.path.join(metrics_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    _log("MAIN", "report written to %s" % report_path)
    _log("MAIN", "done.")


if __name__ == "__main__":
    main()
