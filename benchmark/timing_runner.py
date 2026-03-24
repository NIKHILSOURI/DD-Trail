"""
Measure inference time per sample for each model; save machine-readable table.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .benchmark_config import BenchmarkConfig
from .dataset_registry import get_dataset
from .model_registry import generate_dreamdiffusion, generate_thoughtviz, get_model
from .utils import ensure_dir, setup_logger

log = setup_logger(__name__)


def run_timing(
    dataset_name: str,
    config: BenchmarkConfig,
    max_samples: int = 5,
) -> Dict[str, Any]:
    """Run each model on up to max_samples and record mean time per sample. Returns dict for TABLE 4."""
    samples = get_dataset(dataset_name, config, split="test", max_samples=max_samples)
    if not samples:
        return {}
    results = {"dataset": dataset_name, "n_samples": len(samples), "models": {}}
    for name in ("thoughtviz", "dreamdiffusion", "sarhm"):
        m = get_model(name, config)
        if m is None:
            continue
        start = time.perf_counter()
        try:
            if name == "thoughtviz":
                generate_thoughtviz(m, samples)
            else:
                generate_dreamdiffusion(m, samples, num_samples_per_item=1, ddim_steps=config.ddim_steps)
        except Exception as e:
            log.warning("%s timing failed: %s", name, e)
            results["models"][name] = {"mean_sec": None, "total_sec": None, "error": str(e)}
            continue
        total = time.perf_counter() - start
        results["models"][name] = {"mean_sec": total / len(samples), "total_sec": total}
    return results


def save_timing_table(results: Dict[str, Any], output_path: Path) -> None:
    """Write timing results to JSON and CSV."""
    ensure_dir(output_path.parent)
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    rows = []
    for model, v in results.get("models", {}).items():
        rows.append({"model": model, "dataset": results.get("dataset", ""), "mean_sec": v.get("mean_sec"), "total_sec": v.get("total_sec")})
    if rows:
        import csv
        with open(output_path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["model", "dataset", "mean_sec", "total_sec"])
            w.writeheader()
            w.writerows(rows)
    log.info("Wrote timing to %s", output_path)
