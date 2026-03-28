from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import yaml

from benchmark.benchmark_config import BenchmarkConfig
from benchmark.benchmark_runner import run_all_models
from benchmark.caption_eval import run_caption_eval
from benchmark.progress_util import tqdm
from benchmark.segmentation_eval import run_segmentation_eval
from benchmark.utils import ensure_dir, setup_logger

log = setup_logger(__name__)


def load_cfg(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_config(cfg: Dict) -> BenchmarkConfig:
    c = BenchmarkConfig()
    p = cfg.get("paths", {})
    out = cfg.get("output", {})
    ev = cfg.get("evaluation", {})
    c.seed = int(cfg.get("seed", c.seed))
    c.max_samples = cfg.get("max_samples", c.max_samples)
    c.ddim_steps = int(cfg.get("ddim_steps", c.ddim_steps))
    c.num_samples_per_item = int(cfg.get("num_samples_per_item", c.num_samples_per_item))
    c.show_progress = bool(cfg.get("show_progress", c.show_progress))
    c.models = list(cfg.get("models", c.models))

    c.imagenet_path = p.get("imagenet_path")
    c.imagenet_eeg_eeg_path = p.get("eeg_signals_path")
    c.imagenet_eeg_splits_path = p.get("splits_path")
    c.thoughtviz_data_dir = p.get("thoughtviz_data_dir")
    c.thoughtviz_image_dir = p.get("thoughtviz_image_dir")
    c.thoughtviz_eeg_model_path = p.get("thoughtviz_eeg_model_path")
    c.thoughtviz_gan_model_path = p.get("thoughtviz_gan_model_path")
    c.dreamdiffusion_baseline_ckpt = p.get("baseline_ckpt")
    c.sarhm_ckpt = p.get("sarhm_ckpt")
    c.sarhm_proto_path = p.get("sarhm_proto")

    run_name = out.get("run_name", "thesis_unified")
    root = Path(out.get("root", "results/benchmark_unified")).resolve()
    c.output_dir = str(root / run_name / "benchmark_outputs")

    c.summary_enabled = bool(ev.get("summary_enabled", True))
    c.segmentation_enabled = bool(ev.get("segmentation_enabled", True))
    c.strict_eval = bool(ev.get("strict_eval", False))
    c.thoughtviz_strict_checkpoint_match = bool(ev.get("thoughtviz_strict_checkpoint_match", False))
    c.florence2_model_id = str(ev.get("florence2_model_id", c.florence2_model_id))
    c.summary_fallback_model_id = str(ev.get("summary_fallback_model_id", c.summary_fallback_model_id))
    c.summary_sentence_model_id = str(ev.get("summary_sentence_model_id", c.summary_sentence_model_id))
    c.grounding_dino_model_id = str(ev.get("grounding_dino_model_id", c.grounding_dino_model_id))
    c.grounding_dino_fallback_model_id = str(
        ev.get("grounding_dino_fallback_model_id", c.grounding_dino_fallback_model_id)
    )
    c.sam2_model_id = str(ev.get("sam2_model_id", c.sam2_model_id))
    c.resolve_paths()
    return c


def write_timing_row(path: Path, row: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Unified inference for all models and datasets")
    ap.add_argument("--config", type=str, default="configs/benchmark_unified.yaml")
    ap.add_argument("--dataset", type=str, choices=["imagenet_eeg", "thoughtviz", "both"], default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    cfg_raw = load_cfg(Path(args.config))
    cfg = make_config(cfg_raw)
    if args.models:
        cfg.models = args.models
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    ds_mode = args.dataset or cfg_raw.get("dataset", "both")
    datasets: List[str] = ["imagenet_eeg", "thoughtviz"] if ds_mode == "both" else [ds_mode]

    timing_csv = Path(cfg.output_dir).parent / "combined" / "logs" / "timing_rows.csv"
    iter_ds = tqdm(datasets, desc="Unified datasets", unit="dataset") if cfg.show_progress else datasets
    for ds in iter_ds:
        t0 = time.perf_counter()
        log.info("Running inference: dataset=%s models=%s", ds, cfg.models)
        run_all_models(ds, cfg, max_samples=cfg.max_samples, models=cfg.models)
        infer_sec = time.perf_counter() - t0
        write_timing_row(
            timing_csv,
            {
                "dataset": ds,
                "stage": "inference",
                "models": ",".join(cfg.models),
                "n_samples": cfg.max_samples or -1,
                "elapsed_sec": round(infer_sec, 4),
            },
        )
        if cfg.summary_enabled:
            s0 = time.perf_counter()
            run_caption_eval(Path(cfg.output_dir), ds, config=cfg)
            write_timing_row(
                timing_csv,
                {"dataset": ds, "stage": "summary", "models": "all", "n_samples": cfg.max_samples or -1, "elapsed_sec": round(time.perf_counter() - s0, 4)},
            )
        if cfg.segmentation_enabled:
            g0 = time.perf_counter()
            run_segmentation_eval(Path(cfg.output_dir), ds, config=cfg)
            write_timing_row(
                timing_csv,
                {"dataset": ds, "stage": "segmentation", "models": "all", "n_samples": cfg.max_samples or -1, "elapsed_sec": round(time.perf_counter() - g0, 4)},
            )

    report = {
        "config": str(Path(args.config).resolve()),
        "output_dir": cfg.output_dir,
        "datasets": datasets,
        "models": cfg.models,
        "max_samples": cfg.max_samples,
    }
    out_report = Path(cfg.output_dir).parent / "combined" / "logs" / "inference_report.json"
    ensure_dir(out_report.parent)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Unified inference complete. Report: %s", out_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
