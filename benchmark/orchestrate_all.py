from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml

from .benchmark_config import BenchmarkConfig
from .benchmark_runner import run_all_models
from .build_manifest import build_manifest
from .caption_eval import run_caption_eval
from .metrics_runner import run_core_metrics
from .segmentation_eval import run_segmentation_eval
from .table_generator import generate_all_tables
from .utils import default_thoughtviz_python, ensure_dir, setup_logger
from .visualization_runner import run_visualization

log = setup_logger(__name__)


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg_to_benchmark(c: dict) -> BenchmarkConfig:
    cfg = BenchmarkConfig()
    p = c.get("paths", {})
    out = c.get("output", {})
    ev = c.get("evaluation", {})
    cfg.seed = int(c.get("seed", cfg.seed))
    cfg.max_samples = c.get("max_samples", cfg.max_samples)
    cfg.ddim_steps = int(c.get("ddim_steps", cfg.ddim_steps))
    cfg.num_samples_per_item = int(c.get("num_samples_per_item", cfg.num_samples_per_item))
    cfg.show_progress = bool(c.get("show_progress", cfg.show_progress))
    cfg.models = list(c.get("models", cfg.models))
    cfg.imagenet_path = p.get("imagenet_path")
    cfg.imagenet_eeg_eeg_path = p.get("eeg_signals_path")
    cfg.imagenet_eeg_splits_path = p.get("splits_path")
    cfg.thoughtviz_data_dir = p.get("thoughtviz_data_dir")
    cfg.thoughtviz_image_dir = p.get("thoughtviz_image_dir")
    cfg.thoughtviz_eeg_model_path = p.get("thoughtviz_eeg_model_path")
    cfg.thoughtviz_gan_model_path = p.get("thoughtviz_gan_model_path")
    cfg.dreamdiffusion_baseline_ckpt = p.get("baseline_ckpt")
    cfg.sarhm_ckpt = p.get("sarhm_ckpt")
    cfg.sarhm_proto_path = p.get("sarhm_proto")
    root = Path(out.get("root", "results/benchmark_unified")).resolve()
    run_name = out.get("run_name", "thesis_unified")
    cfg.output_dir = str(root / run_name / "benchmark_outputs")
    cfg.summary_enabled = bool(ev.get("summary_enabled", True))
    cfg.segmentation_enabled = bool(ev.get("segmentation_enabled", True))
    cfg.strict_eval = bool(ev.get("strict_eval", False))
    cfg.resolve_paths()
    return cfg


def _run_subprocess(cmd: List[str], env: dict) -> None:
    log.info("Subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _resolve_thoughtviz_python_exe(raw: Optional[str], repo: Path) -> str:
    """Strip quotes, expand cmd-style %CD%, resolve path; exit if missing."""
    if raw is None or not str(raw).strip():
        return default_thoughtviz_python(repo)
    s = str(raw).strip().strip('"').strip("'")
    if "%CD%" in s or "%cd%" in s:
        s = s.replace("%CD%", os.getcwd()).replace("%cd%", os.getcwd())
    s = os.path.expandvars(s)
    p = Path(s)
    p = (Path.cwd() / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_file():
        log.error(
            "thoughtviz_python is not an existing file: %s. Use an absolute path, e.g. "
            "$REPO_ROOT/venv_thoughtviz_gpu/bin/python (Linux) or "
            "$PWD\\venv_thoughtviz_gpu\\Scripts\\python.exe (Windows; %%CD%% is cmd.exe only and is not expanded here).",
            p,
        )
        raise SystemExit(1)
    return str(p)


def _preflight_orchestrator_python() -> None:
    """Fail fast if this interpreter cannot load PyTorch (split-env mistake)."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        if "TypeIs" in str(e) or "typing_extensions" in str(e):
            log.error(
                "This Python cannot import torch (typing_extensions too old for your torch wheel). "
                "TensorFlow 2.13 ThoughtViz envs pin typing_extensions<4.6; the orchestrator needs the main "
                "DreamDiffusion venv instead. Activate `venv`, rerun the same command, and pass ThoughtViz only via "
                "--thoughtviz_python $REPO_ROOT/venv_thoughtviz_gpu/bin/python (Linux) or "
                "path\\to\\venv_thoughtviz_gpu\\Scripts\\python.exe (Windows)"
            )
            raise SystemExit(1) from e
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description="Robust benchmark orchestration with isolated ThoughtViz runtime")
    ap.add_argument("--config", type=str, default="configs/benchmark_unified.yaml")
    ap.add_argument("--dataset", type=str, choices=["imagenet_eeg", "thoughtviz", "both"], default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--thoughtviz_python", type=str, default=None, help="Python executable inside venv_thoughtviz")
    ap.add_argument("--skip_panels", action="store_true")
    ap.add_argument("--skip_metrics", action="store_true")
    ap.add_argument("--skip_tables", action="store_true")
    args = ap.parse_args()
    _preflight_orchestrator_python()

    cfg_raw = _load_cfg(Path(args.config))
    cfg = _cfg_to_benchmark(cfg_raw)
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    ds_mode = args.dataset or cfg_raw.get("dataset", "both")
    datasets = ["imagenet_eeg", "thoughtviz"] if ds_mode == "both" else [ds_mode]
    log.info("Orchestrator datasets=%s models=%s", datasets, cfg.models)

    repo_root = Path(__file__).resolve().parent.parent
    tv_python = _resolve_thoughtviz_python_exe(args.thoughtviz_python, repo_root)
    tv_enabled = "thoughtviz" in cfg.models
    dd_models = [m for m in cfg.models if m in ("dreamdiffusion", "sarhm")]

    for ds in datasets:
        manifest = build_manifest(ds, cfg, max_samples=cfg.max_samples)
        if dd_models:
            run_all_models(ds, cfg, max_samples=cfg.max_samples, models=dd_models)
        if tv_enabled:
            env = os.environ.copy()
            repo = str(Path(__file__).resolve().parent.parent)
            env["PYTHONPATH"] = os.pathsep.join([repo, str(Path(repo) / "code"), str(Path(repo) / "benchmark")])
            cmd = [
                tv_python,
                "-m",
                "benchmark.run_thoughtviz_from_manifest",
                "--manifest",
                str(manifest),
                "--output_dir",
                cfg.output_dir,
                "--dataset",
                ds,
                "--seed",
                str(cfg.seed),
                "--eval_size",
                str(cfg.eval_size),
            ]
            if cfg.thoughtviz_data_dir:
                cmd += ["--thoughtviz_data_dir", cfg.thoughtviz_data_dir]
            if cfg.thoughtviz_image_dir:
                cmd += ["--thoughtviz_image_dir", cfg.thoughtviz_image_dir]
            if cfg.thoughtviz_eeg_model_path:
                cmd += ["--thoughtviz_eeg_model_path", cfg.thoughtviz_eeg_model_path]
            if cfg.thoughtviz_gan_model_path:
                cmd += ["--thoughtviz_gan_model_path", cfg.thoughtviz_gan_model_path]
            _run_subprocess(cmd, env=env)
        if cfg.summary_enabled:
            run_caption_eval(Path(cfg.output_dir), ds, config=cfg)
        if cfg.segmentation_enabled:
            run_segmentation_eval(Path(cfg.output_dir), ds, config=cfg)
        if not args.skip_metrics:
            run_core_metrics(Path(cfg.output_dir), ds, config=cfg)
        if not args.skip_panels:
            run_visualization(Path(cfg.output_dir), ds, max_panels=50)

    if not args.skip_tables:
        tables_dir = Path(cfg.output_dir).parent / "tables"
        ensure_dir(tables_dir)
        generate_all_tables(Path(cfg.output_dir), out_tables_dir=tables_dir)

    log.info("Benchmark orchestration completed. Outputs: %s", cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
