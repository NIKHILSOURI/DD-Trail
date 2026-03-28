"""
Dataset-agnostic evaluation runner: generate samples, save grids/samples, compute metrics, log to MetricLogger.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from einops import rearrange
from PIL import Image
from tqdm import tqdm

# Optional: import metrics from same package
try:
    from eval.metrics import all_eval_metrics, NA
except ImportError:
    import sys
    _code = Path(__file__).resolve().parent.parent
    if str(_code) not in sys.path:
        sys.path.insert(0, str(_code))
    from eval.metrics import all_eval_metrics, NA


def _samples_to_arrays(
    all_samples: List[Any],
    num_samples: int,
    take_first_pred: bool = True,
) -> tuple:
    """
    all_samples: list of (1+num_samples, C, H, W) tensors (gt + generated).
    Returns pred_imgs (N, H, W, 3), gt_imgs (N, H, W, 3) as numpy 0-255.
    """
    pred_list = []
    gt_list = []
    # all_samples may be list of tensors (0-1) or numpy (N, 1+num_samples, C, H, W) in 0-255 from eLDM
    if hasattr(all_samples, "shape") and all_samples.ndim == 5:
        # Single array (N, 1+num_samples, C, H, W); iterate over first dim
        batches = [all_samples[i] for i in range(len(all_samples))]
    else:
        batches = all_samples
    for batch in batches:
        if batch is None:
            continue
        arr = batch.cpu().numpy() if hasattr(batch, "cpu") else np.asarray(batch).copy()
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        # Normalize to [0,1] if values are 0-255 (eLDM returns uint8 0-255)
        if arr.dtype == np.uint8 or (arr.max() > 1.0 if arr.size > 0 else False):
            arr = arr.astype(np.float64) / 255.0
        # arr: (1+num_samples, C, H, W)
        gt = arr[0]
        if gt.shape[0] == 3:
            gt = np.transpose(gt, (1, 2, 0))
        gt_list.append((255.0 * np.clip(gt, 0, 1)).astype(np.uint8))
        if take_first_pred:
            pred = arr[1]
        else:
            pred = arr[1:].mean(axis=0)
        if pred.shape[0] == 3:
            pred = np.transpose(pred, (1, 2, 0))
        pred_list.append((255.0 * np.clip(pred, 0, 1)).astype(np.uint8))
    pred_imgs = np.stack(pred_list, axis=0) if pred_list else np.zeros((0, 64, 64, 3), dtype=np.uint8)
    gt_imgs = np.stack(gt_list, axis=0) if gt_list else np.zeros((0, 64, 64, 3), dtype=np.uint8)
    return pred_imgs, gt_imgs


def evaluate(
    generative_model: Any,
    dataset: Any,
    dataset_name: str,
    logger: Any,
    epoch: int,
    run_dir: Union[str, Path],
    config: Any,
    max_samples: Optional[int] = None,
    real_imgs_dir: Optional[Union[str, Path]] = None,
    text_prompts: Optional[List[str]] = None,
    paired_images_available: bool = True,
    save_grid: bool = True,
    grid_size: int = 16,
    save_samples: bool = False,
    num_samples_per_item: Optional[int] = None,
    ddim_steps: Optional[int] = None,
    device: Any = None,
) -> Dict[str, Any]:
    """
    Generate images for up to max_samples items, save grid + optional samples,
    compute FID/IS/SSIM/CLIP, log to logger.log_eval(epoch, dataset_name, metrics).

    generative_model: eLDM-like with .generate(dataset, num_samples, ddim_steps, HW, limit=...).
    dataset: iterable of dicts with 'eeg' (and optionally 'image' for gt).
    logger: MetricLogger with log_eval(epoch, dataset_name, metrics_dict).
    run_dir: run directory; artifacts go to run_dir/artifacts/<dataset_name>/.
    config: object with .num_samples, .ddim_steps, etc.
    max_samples: cap number of items to generate (for speed).
    real_imgs_dir: path to real images for FID (optional).
    text_prompts: for image-text CLIP when no paired images (e.g. MOABB).
    paired_images_available: if False, SSIM and image-image CLIP are NA.
    """
    run_path = Path(run_dir)
    artifacts_dir = run_path / "artifacts" / dataset_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    num_samp = num_samples_per_item or getattr(config, "num_samples", 5)
    steps = ddim_steps or getattr(config, "ddim_steps", 250)
    limit = max_samples

    # Generate: use model's generate(dataset, num_samples, ddim_steps, HW, limit)
    try:
        grid, all_samples = generative_model.generate(
            dataset, num_samp, steps, getattr(config, "HW", None), limit=limit
        )
    except Exception as e:
        print(f"Evaluate generate failed ({dataset_name}): {e}")
        metrics_out = {
            f"eval/{dataset_name}/n_samples": 0,
            f"eval/{dataset_name}/fid": NA,
            f"eval/{dataset_name}/is_mean": NA,
            f"eval/{dataset_name}/is_std": NA,
            f"eval/{dataset_name}/ssim_mean": NA,
            f"eval/{dataset_name}/ssim_std": NA,
            f"eval/{dataset_name}/clip_sim_mean": NA,
            f"eval/{dataset_name}/clip_sim_std": NA,
        }
        logger.log_eval(epoch, dataset_name, metrics_out)
        return metrics_out

    if all_samples is None or (hasattr(all_samples, "__len__") and len(all_samples) == 0):
        metrics_out = {
            f"eval/{dataset_name}/n_samples": 0,
            f"eval/{dataset_name}/fid": NA,
            f"eval/{dataset_name}/is_mean": NA,
            f"eval/{dataset_name}/is_std": NA,
            f"eval/{dataset_name}/ssim_mean": NA,
            f"eval/{dataset_name}/ssim_std": NA,
            f"eval/{dataset_name}/clip_sim_mean": NA,
            f"eval/{dataset_name}/clip_sim_std": NA,
        }
        logger.log_eval(epoch, dataset_name, metrics_out)
        return metrics_out

    pred_imgs, gt_imgs = _samples_to_arrays(all_samples, num_samp)
    n = len(pred_imgs)

    # Save grid
    if save_grid and grid is not None and grid.size > 0:
        grid_path = artifacts_dir / f"grid_epoch{epoch:02d}.png"
        if grid.dtype != np.uint8:
            grid = (255.0 * np.clip(grid.astype(np.float64), 0, 1)).astype(np.uint8)
        Image.fromarray(grid).save(grid_path)

    # Optional: save a few individual samples
    if save_samples and n > 0:
        samples_dir = artifacts_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        for i in range(min(16, n)):
            arr = pred_imgs[i]
            if arr.shape[-1] != 3:
                arr = np.transpose(arr, (1, 2, 0))
            Image.fromarray(arr).save(samples_dir / f"epoch{epoch:02d}_sample{i:02d}.png")

    # Metrics (prefix with eval/<dataset_name>/ for logger)
    temp_dir = artifacts_dir / "tmp_fid"
    temp_dir.mkdir(exist_ok=True)
    metrics = all_eval_metrics(
        pred_imgs,
        gt_imgs=gt_imgs if paired_images_available else None,
        real_imgs_dir=real_imgs_dir,
        text_prompts=text_prompts,
        paired_images_available=paired_images_available,
        temp_dir=str(temp_dir),
        device=device,
    )
    # Rename keys to eval/<dataset_name>/...
    metrics_out = {}
    for k, v in metrics.items():
        new_k = k.replace("eval/", f"eval/{dataset_name}/", 1) if k.startswith("eval/") else f"eval/{dataset_name}/{k}"
        metrics_out[new_k] = v
    if "eval/n_samples" in metrics:
        metrics_out[f"eval/{dataset_name}/n_samples"] = metrics["eval/n_samples"]
    logger.log_eval(epoch, dataset_name, metrics_out)
    return metrics_out
