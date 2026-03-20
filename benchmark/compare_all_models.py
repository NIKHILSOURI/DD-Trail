"""
CLI: Run unified benchmark (ThoughtViz, DreamDiffusion baseline, SAR-HM) on ImageNet-EEG and/or ThoughtViz.
No SAR-HM++. Usage: python -m benchmark.compare_all_models --dataset imagenet_eeg --max_samples 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root and code on path
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if CODE_DIR.is_dir() and str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from benchmark.benchmark_config import BenchmarkConfig
from benchmark.benchmark_runner import run_all_models
from benchmark.caption_eval import run_caption_eval
from benchmark.segmentation_eval import run_segmentation_eval
from benchmark.utils import setup_logger

log = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified benchmark: ThoughtViz, DreamDiffusion, SAR-HM")
    p.add_argument("--dataset", type=str, default="imagenet_eeg", choices=["imagenet_eeg", "thoughtviz", "both"],
                   help="Dataset to run on")
    p.add_argument("--max_samples", type=int, default=None, help="Limit samples (e.g. 10, 20 for smoke)")
    p.add_argument("--models", type=str, nargs="+", default=["thoughtviz", "dreamdiffusion", "sarhm"],
                   help="Models to run")
    p.add_argument("--run_name", type=str, default=None, help="Experiment run name (e.g. smoke_test)")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--imagenet_path", type=str, default=None, help="ImageNet root for EEG GT")
    p.add_argument("--eeg_signals_path", type=str, default=None)
    p.add_argument("--splits_path", type=str, default=None)
    p.add_argument("--baseline_ckpt", type=str, default=None, help="DreamDiffusion baseline checkpoint")
    p.add_argument("--sarhm_ckpt", type=str, default=None)
    p.add_argument("--sarhm_proto", type=str, default=None)
    p.add_argument("--thoughtviz_data_dir", type=str, default=None)
    p.add_argument("--thoughtviz_image_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--ddim_steps", type=int, default=250)
    p.add_argument("--summary_enabled", type=str, default="true", choices=["true", "false"])
    p.add_argument("--segmentation_enabled", type=str, default="true", choices=["true", "false"])
    p.add_argument("--strict_eval", type=str, default="false", choices=["true", "false"])
    p.add_argument("--florence2_model_id", type=str, default=None)
    p.add_argument("--summary_sentence_model_id", type=str, default=None)
    p.add_argument("--grounding_dino_model_id", type=str, default=None)
    p.add_argument("--sam2_model_id", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = BenchmarkConfig()
    config.resolve_paths()
    config.max_samples = args.max_samples
    config.models = args.models
    config.seed = args.seed
    config.ddim_steps = args.ddim_steps
    config.summary_enabled = args.summary_enabled == "true"
    config.segmentation_enabled = args.segmentation_enabled == "true"
    config.strict_eval = args.strict_eval == "true"
    if args.florence2_model_id:
        config.florence2_model_id = args.florence2_model_id
    if args.summary_sentence_model_id:
        config.summary_sentence_model_id = args.summary_sentence_model_id
    if args.grounding_dino_model_id:
        config.grounding_dino_model_id = args.grounding_dino_model_id
    if args.sam2_model_id:
        config.sam2_model_id = args.sam2_model_id
    if args.run_name:
        config.run_name = args.run_name
        config.output_dir = str(REPO_ROOT / "results" / "experiments" / args.run_name / "benchmark_outputs")
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.imagenet_path:
        config.imagenet_path = args.imagenet_path
    if args.eeg_signals_path:
        config.imagenet_eeg_eeg_path = args.eeg_signals_path
    if args.splits_path:
        config.imagenet_eeg_splits_path = args.splits_path
    if args.baseline_ckpt:
        config.dreamdiffusion_baseline_ckpt = args.baseline_ckpt
    if args.sarhm_ckpt:
        config.sarhm_ckpt = args.sarhm_ckpt
    if args.sarhm_proto:
        config.sarhm_proto_path = args.sarhm_proto
    if args.thoughtviz_data_dir:
        config.thoughtviz_data_dir = args.thoughtviz_data_dir
    if args.thoughtviz_image_dir:
        config.thoughtviz_image_dir = args.thoughtviz_image_dir

    import os
    if os.environ.get("IMAGENET_PATH") and not config.imagenet_path:
        config.imagenet_path = os.environ.get("IMAGENET_PATH")

    datasets = ["imagenet_eeg", "thoughtviz"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        log.info("Benchmark dataset: %s (max_samples=%s)", ds, config.max_samples)
        run_all_models(ds, config, max_samples=config.max_samples, models=config.models)
        try:
            if config.summary_enabled:
                run_caption_eval(Path(config.output_dir), ds, config=config)
            if config.segmentation_enabled:
                run_segmentation_eval(Path(config.output_dir), ds, config=config)
        except Exception as e:
            if config.strict_eval:
                raise
            log.warning("Post-generation eval failed for %s: %s", ds, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
