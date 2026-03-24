from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def latest(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def main() -> int:
    repo_root = Path(os.environ.get("REPO_ROOT", "/workspace/project/DREAMDIFFUSION_RUNPOD")).resolve()
    cfg_path = repo_root / "configs" / "benchmark_unified.yaml"
    out_json = repo_root / "results" / "benchmark_unified" / "combined" / "logs" / "asset_discovery.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    ckpts = list((repo_root / "exps" / "results" / "generation").glob("*/checkpoint_best.pth"))
    baseline_ckpt = Path(os.environ["BASELINE_RUN"]).joinpath("checkpoint_best.pth") if os.environ.get("BASELINE_RUN") else None
    sarhm_ckpt = Path(os.environ["SARHM_RUN"]).joinpath("checkpoint_best.pth") if os.environ.get("SARHM_RUN") else None
    if baseline_ckpt is None and ckpts:
        baseline_ckpt = sorted(ckpts, key=lambda p: p.stat().st_mtime)[0]
    if sarhm_ckpt is None and ckpts:
        sarhm_ckpt = latest(ckpts)
    sarhm_proto = sarhm_ckpt.parent / "prototypes.pt" if sarhm_ckpt and (sarhm_ckpt.parent / "prototypes.pt").exists() else None

    tv_eeg_img = repo_root / "code" / "ThoughtViz" / "models" / "eeg_models" / "image" / "run_final.h5"
    tv_eeg_char = repo_root / "code" / "ThoughtViz" / "models" / "eeg_models" / "char" / "run_final.h5"
    tv_gan_img = repo_root / "code" / "ThoughtViz" / "models" / "gan_models" / "final" / "image" / "generator.model"
    tv_saved_root = repo_root / "code" / "ThoughtViz" / "saved_models" / "thoughtviz_with_eeg"
    tv_gan_fallback = None
    tv_gan_img_fallback = None
    tv_gan_char_fallback = None
    if tv_saved_root.exists():
        # Keep fallback bounded to avoid expensive deep scans.
        for sub in ("Image", "image"):
            cand_dir = tv_saved_root / sub
            if cand_dir.exists():
                cands = sorted(cand_dir.glob("generator_*"))
                if not cands:
                    cands = sorted(cand_dir.glob("run_*/generator_*"))
                if cands:
                    tv_gan_img_fallback = cands[-1]
                    break
        for sub in ("Char", "char"):
            cand_dir = tv_saved_root / sub
            if cand_dir.exists():
                cands = sorted(cand_dir.glob("generator_*"))
                if not cands:
                    cands = sorted(cand_dir.glob("run_*/generator_*"))
                if cands:
                    tv_gan_char_fallback = cands[-1]
                    break
        tv_gan_fallback = tv_gan_img_fallback or tv_gan_char_fallback

    discovered: Dict[str, Optional[str]] = {
        "repo_root": str(repo_root),
        "imagenet_path": str(repo_root / "datasets" / "imageNet_images"),
        "eeg_signals_path": str(repo_root / "datasets" / "eeg_5_95_std.pth"),
        "splits_path": str(repo_root / "datasets" / "block_splits_by_image_single.pth"),
        "thoughtviz_data_dir": str(repo_root / "code" / "ThoughtViz" / "data"),
        "thoughtviz_image_dir": str(repo_root / "code" / "ThoughtViz" / "training" / "images"),
        "baseline_ckpt": str(baseline_ckpt) if baseline_ckpt and baseline_ckpt.exists() else None,
        "sarhm_ckpt": str(sarhm_ckpt) if sarhm_ckpt and sarhm_ckpt.exists() else None,
        "sarhm_proto": str(sarhm_proto) if sarhm_proto else None,
        "thoughtviz_eeg_model_path": str(tv_eeg_img if tv_eeg_img.exists() else tv_eeg_char),
        "thoughtviz_gan_model_path": str(tv_gan_img if tv_gan_img.exists() else tv_gan_fallback) if (tv_gan_img.exists() or tv_gan_fallback) else None,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(discovered, f, indent=2)

    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    cfg.setdefault("paths", {}).update({k: v for k, v in discovered.items() if k in cfg.get("paths", {}) or k != "repo_root"})
    cfg.setdefault("paths", {})["repo_root"] = discovered["repo_root"]
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(json.dumps(discovered, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
