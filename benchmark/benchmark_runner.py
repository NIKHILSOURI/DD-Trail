"""
Run one or all benchmark models on a dataset subset; save standardized outputs.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .benchmark_config import BenchmarkConfig
from .dataset_registry import get_dataset
from .model_registry import (
    generate_dreamdiffusion,
    generate_thoughtviz,
    get_model,
)
from .output_standardizer import write_sample_outputs
from .utils import setup_logger

log = setup_logger(__name__)


def run_one_model(
    model_name: str,
    dataset_name: str,
    samples: List[Dict[str, Any]],
    config: BenchmarkConfig,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a single model on samples; save outputs to output_dir/dataset_name/sample_*/.
    Returns dict with keys: model_name, dataset_name, n_samples, times, errors.
    """
    output_dir = output_dir or config.output_dir
    times = []
    errors = []
    model_obj = get_model(model_name, config)
    if model_obj is None:
        log.warning("Model %s not available; skipping.", model_name)
        return {"model_name": model_name, "dataset_name": dataset_name, "n_samples": 0, "times": [], "errors": ["model not loaded"]}
    try:
        if model_name == "thoughtviz":
            imgs = generate_thoughtviz(model_obj, samples)
        else:
            imgs = generate_dreamdiffusion(
                model_obj,
                samples,
                num_samples_per_item=config.num_samples_per_item,
                ddim_steps=config.ddim_steps,
            )
    except Exception as e:
        log.exception("Generate failed for %s: %s", model_name, e)
        return {"model_name": model_name, "dataset_name": dataset_name, "n_samples": 0, "times": [], "errors": [str(e)]}
    for i, s in enumerate(samples):
        sid = s.get("sample_id", "sample_%04d" % i)
        gt = s.get("gt_image")
        if i < len(imgs):
            gen = imgs[i]
        else:
            gen = None
        out_map = {"ground_truth": gt, model_name: gen}
        write_sample_outputs(
            sample_id=sid,
            dataset_name=dataset_name,
            output_dir=output_dir,
            ground_truth=gt,
            **{model_name: gen},
            metadata=s.get("metadata"),
            eval_size=config.eval_size,
        )
    return {"model_name": model_name, "dataset_name": dataset_name, "n_samples": len(samples), "times": times, "errors": []}


def run_all_models(
    dataset_name: str,
    config: BenchmarkConfig,
    max_samples: Optional[int] = None,
    models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load dataset, run all requested models, write standardized outputs."""
    models = models or config.models
    max_samples = max_samples or config.max_samples
    samples = get_dataset(dataset_name, config, split="test", max_samples=max_samples)
    if not samples:
        log.warning("No samples for dataset %s.", dataset_name)
        return {}
    log.info("Running %s on %s with %d samples.", models, dataset_name, len(samples))
    results = {}
    for name in models:
        if name not in ("thoughtviz", "dreamdiffusion", "sarhm"):
            continue
        res = run_one_model(name, dataset_name, samples, config)
        results[name] = res
    return results
