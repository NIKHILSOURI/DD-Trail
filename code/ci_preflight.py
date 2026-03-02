"""
Preflight script: verify required files, compile code, import modules, run minimal smoke generation.
Exits with 0 only when all checks pass. Use --smoke_only to skip compile/import and only run smoke.
"""
from __future__ import annotations

import argparse
import compileall
import json
import os
import sys
from pathlib import Path

# Resolve paths (match config.py)
_CODE_DIR = Path(__file__).parent.absolute()
_REPO_ROOT = _CODE_DIR.parent.absolute()
_DATA_ROOT = Path(os.environ.get("DREAMDIFFUSION_DATA_ROOT", str(_REPO_ROOT / "datasets")))
_PRETRAIN_ROOT = Path(os.environ.get("DREAMDIFFUSION_PRETRAIN_ROOT", str(_REPO_ROOT / "pretrains")))

REQUIRED_FILES = [
    _DATA_ROOT / "eeg_5_95_std.pth",
    _PRETRAIN_ROOT / "models" / "config15.yaml",
    _PRETRAIN_ROOT / "models" / "v1-5-pruned.ckpt",
]
# Splits: at least one of these must exist
SPLITS_OPTIONS = [
    _DATA_ROOT / "block_splits_by_image_single.pth",
    _DATA_ROOT / "block_splits_by_image_all.pth",
]


def main():
    parser = argparse.ArgumentParser(description="CI preflight and smoke test")
    parser.add_argument("--smoke_only", action="store_true", help="Only run smoke generation, skip compile/import")
    parser.add_argument("--require_gpu", action="store_true", help="Exit with error if CUDA is not available (ensures preflight runs on GPU)")
    args = parser.parse_args()

    if sys.path[0] != str(_CODE_DIR):
        sys.path.insert(0, str(_CODE_DIR))

    failures = []
    # Always check required files first (no imports needed)
    missing = []
    for p in REQUIRED_FILES:
        if not p.is_file():
            missing.append(str(p))
    splits_ok = any(p.is_file() for p in SPLITS_OPTIONS)
    if not splits_ok:
        missing.append("one of: " + ", ".join(str(p) for p in SPLITS_OPTIONS))
    if missing:
        print("PRE-FLIGHT FAIL (missing required files):")
        for m in missing:
            print("  ", m)
        sys.exit(1)
    print("Resolved paths:")
    print("  DATA_ROOT:", _DATA_ROOT)
    print("  PRETRAIN_ROOT:", _PRETRAIN_ROOT)
    print("  eeg_5_95_std.pth:", REQUIRED_FILES[0], "(exists)")
    print("  splits:", "ok (one of single/all exists)")
    print("  config15.yaml:", REQUIRED_FILES[1], "(exists)")
    print("  v1-5-pruned.ckpt:", REQUIRED_FILES[2], "(exists)")

    if not args.smoke_only:
        # 1) Compile all code
        try:
            compileall.compile_dir(str(_CODE_DIR), quiet=1, force=False)
        except Exception as e:
            failures.append("compileall: %s" % e)

        # 2) Import key modules (requires torch, numpy, etc. – run from project env)
        for mod in ["config", "dataset", "dc_ldm.ldm_for_eeg", "sarhm.sarhm_modules", "sarhm.prototypes"]:
            try:
                __import__(mod)
            except Exception as e:
                failures.append("import %s: %s" % (mod, e))

        if failures:
            print("PRE-FLIGHT FAIL (compile/import):")
            for f in failures:
                print("  ", f)
            print("(Run from environment with project deps, e.g. conda activate dreamdiffusion)")
            sys.exit(1)

    # 4) Smoke generation: batch=1, limit=1, ddim_steps=5, num_samples=1
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _REPO_ROOT / "outputs" / "smoke_test" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics = {"smoke_baseline": "fail", "smoke_sarhm": "fail", "output_dir": str(out_dir)}

    try:
        import torch
        from config import Config_Generative_Model
        from dataset import create_EEG_dataset
        from dc_ldm.ldm_for_eeg import eLDM

        cuda_available = torch.cuda.is_available()
        if getattr(args, "require_gpu", False) and not cuda_available:
            print("PRE-FLIGHT FAIL: --require_gpu set but CUDA is not available.")
            print("Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            sys.exit(1)
        device = torch.device("cuda" if cuda_available else "cpu")
        if cuda_available:
            print("Using GPU: %s" % torch.cuda.get_device_name(0))
        else:
            print("Using CPU (no CUDA). For GPU: install PyTorch with CUDA and run without --require_gpu to allow fallback.")
        splits_path = None
        for p in SPLITS_OPTIONS:
            if p.is_file():
                splits_path = str(p)
                break
        if not splits_path:
            print("PRE-FLIGHT FAIL: no splits file found")
            sys.exit(1)

        config = Config_Generative_Model()
        config.eeg_signals_path = str(_DATA_ROOT / "eeg_5_95_std.pth")
        config.splits_path = splits_path
        config.imagenet_path = os.environ.get("IMAGENET_PATH") or str(_DATA_ROOT / "imageNet_images")
        if not os.path.isdir(config.imagenet_path):
            print("PRE-FLIGHT FAIL: imagenet_path missing or not a directory: %s" % config.imagenet_path)
            print("  Set IMAGENET_PATH or place ImageNet at datasets/imageNet_images.")
            sys.exit(1)
        config.pretrain_gm_path = str(_PRETRAIN_ROOT)
        config.pretrain_mbm_path = str(_PRETRAIN_ROOT / "eeg_pretain" / "checkpoint.pth")
        config.num_samples = 1
        config.ddim_steps = 5
        config.val_gen_limit = 1

        # Transforms (minimal)
        def normalize(img):
            import torch
            from einops import rearrange
            if img.shape[-1] == 3:
                img = rearrange(img, "h w c -> c h w")
            img = torch.tensor(img).float()
            return img * 2.0 - 1.0

        def channel_last(img):
            from einops import rearrange
            if img.shape[-1] == 3:
                return img
            return rearrange(img, "c h w -> h w c")

        from torchvision import transforms
        img_t = transforms.Compose([normalize, transforms.Resize((512, 512)), channel_last])
        train_ds, test_ds = create_EEG_dataset(
            eeg_signals_path=config.eeg_signals_path,
            splits_path=config.splits_path,
            imagenet_path=config.imagenet_path,
            image_transform=[img_t, img_t],
            subject=4,
        )
        num_voxels = train_ds.dataset.data_len
        pretrain_mbm = torch.load(config.pretrain_mbm_path, map_location="cpu", weights_only=False)

        # Baseline
        config_baseline = Config_Generative_Model()
        config_baseline.eeg_signals_path = config.eeg_signals_path
        config_baseline.splits_path = config.splits_path
        config_baseline.pretrain_gm_path = config.pretrain_gm_path
        config_baseline.pretrain_mbm_path = config.pretrain_mbm_path
        config_baseline.num_samples = 1
        config_baseline.ddim_steps = 5
        config_baseline.use_sarhm = False
        config_baseline.ablation_mode = "baseline"

        eldm_baseline = eLDM(
            pretrain_mbm, num_voxels, device=device, pretrain_root=config.pretrain_gm_path,
            ddim_steps=5, global_pool=config.global_pool, use_time_cond=config.use_time_cond,
            clip_tune=config.clip_tune, cls_tune=getattr(config, "cls_tune", False), main_config=config_baseline,
        )
        out_baseline = out_dir / "baseline"
        out_baseline.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            eldm_baseline.model.eval()
            grid_b, samples_b = eldm_baseline.generate(
                test_ds, num_samples=1, ddim_steps=5, HW=None, limit=1, output_path=str(out_baseline)
            )
        baseline_ok = (out_baseline / "test0-1.png").is_file() and (out_baseline / "test0-1.png").stat().st_size > 0
        metrics["smoke_baseline"] = "pass" if baseline_ok else "fail"
        if not baseline_ok:
            failures.append("baseline: no or empty output image")

        # SAR-HM (full_sarhm)
        config_sarhm = Config_Generative_Model()
        config_sarhm.eeg_signals_path = config.eeg_signals_path
        config_sarhm.splits_path = config.splits_path
        config_sarhm.pretrain_gm_path = config.pretrain_gm_path
        config_sarhm.pretrain_mbm_path = config.pretrain_mbm_path
        config_sarhm.num_samples = 1
        config_sarhm.ddim_steps = 5
        config_sarhm.use_sarhm = True
        config_sarhm.ablation_mode = "full_sarhm"
        config_sarhm.proto_path = None
        config_sarhm.alpha_mode = "entropy"
        config_sarhm.alpha_max = 0.2
        config_sarhm.conf_threshold = 0.2

        eldm_sarhm = eLDM(
            pretrain_mbm, num_voxels, device=device, pretrain_root=config.pretrain_gm_path,
            ddim_steps=5, global_pool=config.global_pool, use_time_cond=config.use_time_cond,
            clip_tune=config.clip_tune, cls_tune=getattr(config, "cls_tune", False), main_config=config_sarhm,
        )
        out_sarhm = out_dir / "full_sarhm"
        out_sarhm.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            eldm_sarhm.model.eval()
            grid_s, samples_s = eldm_sarhm.generate(
                test_ds, num_samples=1, ddim_steps=5, HW=None, limit=1, output_path=str(out_sarhm)
            )
        sarhm_ok = (out_sarhm / "test0-1.png").is_file() and (out_sarhm / "test0-1.png").stat().st_size > 0
        metrics["smoke_sarhm"] = "pass" if sarhm_ok else "fail"
        if not sarhm_ok:
            failures.append("full_sarhm: no or empty output image")

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        failures.append("smoke run: %s" % e)
        metrics["error"] = str(e)
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    if failures:
        print("PRE-FLIGHT FAIL:")
        for f in failures:
            print("  ", f)
        print("metrics written to", metrics_path)
        sys.exit(1)
    print("PRE-FLIGHT PASS")
    print("Outputs:", out_dir)
    print("metrics.json:", metrics_path)
    sys.exit(0)


if __name__ == "__main__":
    main()
