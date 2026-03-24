from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .benchmark_config import BenchmarkConfig
from .dataset_registry import get_dataset
from .output_standardizer import write_sample_outputs
from .status_utils import update_model_status
from .utils import ensure_dir, setup_logger

log = setup_logger(__name__)


def _to_numpy_eeg(eeg: Any) -> np.ndarray:
    if hasattr(eeg, "detach"):
        return eeg.detach().float().cpu().numpy().astype(np.float32)
    if hasattr(eeg, "numpy"):
        return np.asarray(eeg.numpy(), dtype=np.float32)
    return np.asarray(eeg, dtype=np.float32)


def build_manifest(
    dataset_name: str,
    config: BenchmarkConfig,
    max_samples: Optional[int] = None,
) -> Path:
    output_dir = Path(config.output_dir)
    ds_dir = output_dir / dataset_name
    prep_dir = ds_dir / "_prepared"
    ensure_dir(prep_dir)

    samples = get_dataset(dataset_name, config, split="test", max_samples=max_samples, show_progress=config.show_progress)
    if not samples:
        raise RuntimeError("No samples available for dataset=%s" % dataset_name)

    manifest: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        sid = str(s.get("sample_id", "%s_%04d" % (dataset_name, i)))
        eeg = _to_numpy_eeg(s.get("eeg"))
        eeg_path = prep_dir / ("%s_eeg.npy" % sid.replace("/", "_"))
        np.save(eeg_path, eeg.astype(np.float32))
        gt_image_path = s.get("gt_image_path")
        # Always write ground truth once during manifest build when present as array.
        if s.get("gt_image") is not None:
            write_sample_outputs(
                sample_id=sid,
                dataset_name=dataset_name,
                output_dir=output_dir,
                ground_truth=s.get("gt_image"),
                metadata=s.get("metadata"),
                eval_size=config.eval_size,
            )
        elif gt_image_path:
            write_sample_outputs(
                sample_id=sid,
                dataset_name=dataset_name,
                output_dir=output_dir,
                metadata=s.get("metadata"),
                eval_size=config.eval_size,
            )
        item = {
            "dataset": dataset_name,
            "sample_id": sid,
            "split": s.get("split", "test"),
            "eeg_path": str(eeg_path),
            "gt_image_path": gt_image_path,
            "label": int(s["label"]) if s.get("label") is not None and hasattr(s.get("label"), "__int__") else s.get("label"),
            "metadata": s.get("metadata", {}),
        }
        manifest.append(item)
        for model_name in ("thoughtviz", "dreamdiffusion", "sarhm"):
            update_model_status(config.output_dir, dataset_name, sid, model_name, "pending")

    manifest_path = ds_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"dataset": dataset_name, "count": len(manifest), "samples": manifest}, indent=2), encoding="utf-8")
    log.info("Wrote canonical manifest: %s (%d samples)", manifest_path, len(manifest))
    return manifest_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build canonical benchmark manifest for one dataset")
    ap.add_argument("--dataset", type=str, required=True, choices=["imagenet_eeg", "thoughtviz"])
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    cfg = BenchmarkConfig()
    cfg.resolve_paths()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    build_manifest(args.dataset, cfg, max_samples=args.max_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
