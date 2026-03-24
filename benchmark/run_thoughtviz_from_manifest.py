from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from .benchmark_config import BenchmarkConfig
from .output_standardizer import save_image_standardized, write_sample_outputs
from .status_utils import update_model_status, validate_image_array
from .utils import setup_logger

log = setup_logger(__name__)


def _prepare_eeg_for_thoughtviz(eeg: np.ndarray) -> np.ndarray:
    from scipy.ndimage import zoom

    x = np.asarray(eeg, dtype=np.float32)
    while x.ndim >= 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim == 3 and x.shape[-1] == 1:
        plane = x[..., 0]
    elif x.ndim == 2:
        plane = x
    elif x.ndim == 3:
        plane = x[..., 0]
    else:
        raise ValueError("invalid eeg shape %s" % (x.shape,))
    if plane.shape != (14, 32):
        plane = zoom(plane, (14 / plane.shape[0], 32 / plane.shape[1]), order=1).astype(np.float32)
    return plane[..., np.newaxis]


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("samples", [])


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ThoughtViz inference in isolated environment from manifest")
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True, choices=["imagenet_eeg", "thoughtviz"])
    ap.add_argument("--thoughtviz_data_dir", type=str, default=None)
    ap.add_argument("--thoughtviz_image_dir", type=str, default=None)
    ap.add_argument("--thoughtviz_eeg_model_path", type=str, default=None)
    ap.add_argument("--thoughtviz_gan_model_path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--eval_size", type=int, default=256)
    args = ap.parse_args()

    np.random.seed(args.seed)
    manifest_path = Path(args.manifest).resolve()
    samples = _load_manifest(manifest_path)
    if not samples:
        raise RuntimeError("empty manifest: %s" % manifest_path)

    cfg = BenchmarkConfig()
    cfg.thoughtviz_data_dir = args.thoughtviz_data_dir
    cfg.thoughtviz_image_dir = args.thoughtviz_image_dir
    cfg.thoughtviz_eeg_model_path = args.thoughtviz_eeg_model_path
    cfg.thoughtviz_gan_model_path = args.thoughtviz_gan_model_path

    from .model_registry import get_thoughtviz_wrapper

    wrapper = get_thoughtviz_wrapper(cfg, dataset_name=args.dataset)
    if wrapper is None:
        raise RuntimeError("ThoughtViz wrapper is unavailable in current env")

    output_dir = Path(args.output_dir)
    for i, s in enumerate(samples):
        sid = str(s["sample_id"])
        eeg = np.load(s["eeg_path"]).astype(np.float32)
        try:
            eeg_tv = _prepare_eeg_for_thoughtviz(eeg)
            img = wrapper.generate_from_eeg([eeg_tv], num_samples=1)[0]
            ok, reason = validate_image_array(img)
            if not ok:
                update_model_status(output_dir, args.dataset, sid, "thoughtviz", "failed", reason=reason)
                continue
            sample_dir = write_sample_outputs(
                sample_id=sid,
                dataset_name=args.dataset,
                output_dir=output_dir,
                thoughtviz=img,
                metadata={"manifest_index": i},
                eval_size=args.eval_size,
            )
            # If GT exists only as path and was not materialized, materialize it now for consistent metrics.
            gt_path = s.get("gt_image_path")
            gt_png = sample_dir / "ground_truth.png"
            if gt_path and not gt_png.exists():
                try:
                    gt = Image.open(gt_path).convert("RGB")
                    save_image_standardized(gt, gt_png, eval_size=args.eval_size)
                except Exception as e:
                    update_model_status(output_dir, args.dataset, sid, "thoughtviz", "failed", reason="gt load failed: %s" % e)
                    continue
            update_model_status(output_dir, args.dataset, sid, "thoughtviz", "success")
        except Exception as e:
            update_model_status(output_dir, args.dataset, sid, "thoughtviz", "failed", reason=str(e))
            log.exception("ThoughtViz generation failed for %s: %s", sid, e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
