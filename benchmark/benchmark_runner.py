"""
Run one or all benchmark models on a dataset subset; save standardized outputs.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from .benchmark_config import BenchmarkConfig
from .progress_util import tqdm
from .dataset_registry import get_dataset
from .model_registry import (
    generate_dreamdiffusion,
    generate_thoughtviz,
    get_model,
)
from .output_standardizer import write_sample_outputs
from .status_utils import update_model_status, validate_image_array
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
    log.info("Model step: loading %s …", model_name)
    model_obj = get_model(model_name, config, dataset_name=dataset_name)
    if model_obj is None:
        log.warning("Model %s not available; skipping.", model_name)
        for i, s in enumerate(samples):
            sid = s.get("sample_id", "sample_%04d" % i)
            update_model_status(output_dir, dataset_name, sid, model_name, "failed", reason="model not loaded")
        return {"model_name": model_name, "dataset_name": dataset_name, "n_samples": 0, "times": [], "errors": ["model not loaded"]}
    pbar_desc = "%s · %s" % (model_name, dataset_name)
    try:
        log.info("Model step: generating %d samples with %s …", len(samples), model_name)
        if model_name == "thoughtviz":
            imgs = generate_thoughtviz(model_obj, samples)
        else:
            imgs = generate_dreamdiffusion(
                model_obj,
                samples,
                num_samples_per_item=config.num_samples_per_item,
                ddim_steps=config.ddim_steps,
                pbar_desc=pbar_desc,
            )
    except Exception as e:
        log.exception("Generate failed for %s: %s", model_name, e)
        for i, s in enumerate(samples):
            sid = s.get("sample_id", "sample_%04d" % i)
            update_model_status(output_dir, dataset_name, sid, model_name, "failed", reason="generate failed: %s" % e)
        return {"model_name": model_name, "dataset_name": dataset_name, "n_samples": 0, "times": [], "errors": [str(e)]}
    save_iter = enumerate(samples)
    if getattr(config, "show_progress", True) and len(samples) > 0:
        save_iter = tqdm(
            save_iter,
            total=len(samples),
            desc="Write outputs (%s)" % model_name,
            unit="sample",
        )
    for i, s in save_iter:
        sid = s.get("sample_id", "sample_%04d" % i)
        gt = s.get("gt_image")
        if gt is None and s.get("gt_image_path"):
            try:
                gt = np.array(Image.open(s["gt_image_path"]).convert("RGB"))
            except Exception as e:
                log.warning("Could not load gt_image_path for %s: %s", sid, e)
                gt = None
        if i < len(imgs):
            gen = imgs[i]
        else:
            gen = None
        if gen is not None:
            ok, reason = validate_image_array(gen)
            if not ok:
                update_model_status(output_dir, dataset_name, sid, model_name, "failed", reason=reason)
                gen = None
            else:
                update_model_status(output_dir, dataset_name, sid, model_name, "success")
        else:
            update_model_status(output_dir, dataset_name, sid, model_name, "failed", reason="missing generated output")
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
    log.info("Loading dataset %s …", dataset_name)
    samples = get_dataset(dataset_name, config, split="test", max_samples=max_samples)
    if not samples:
        log.warning("No samples for dataset %s.", dataset_name)
        return {}
    n_out = len(samples) * max(1, config.num_samples_per_item)
    log.info(
        "Dataset %s: %d EEG samples × %d gen/sample → %d images per model.",
        dataset_name,
        len(samples),
        max(1, config.num_samples_per_item),
        n_out,
    )
    log.info("Running %s on %s with %d samples.", models, dataset_name, len(samples))
    results = {}
    to_run = [n for n in models if n in ("thoughtviz", "dreamdiffusion", "sarhm")]
    model_iter = to_run
    if getattr(config, "show_progress", True) and len(to_run) > 1:
        model_iter = tqdm(to_run, desc="Models on %s" % dataset_name, unit="model")
    for name in model_iter:
        res = run_one_model(name, dataset_name, samples, config)
        results[name] = res
    return results
