"""
Evaluation metrics: FID, IS, SSIM, CLIP similarity.
Each function returns a dict of metric names -> values (or "NA" for inapplicable).
Modular: call from evaluate.py with generated/real data; no side effects except optional caching.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

NA = "NA"


def _to_uint8(imgs: np.ndarray) -> np.ndarray:
    """Ensure shape (N,H,W,3) and dtype uint8, range 0-255."""
    if imgs.ndim == 3:
        imgs = imgs[np.newaxis, ...]
    if imgs.shape[-1] != 3:
        imgs = np.transpose(imgs, (0, 2, 3, 1))
    if imgs.dtype != np.uint8:
        imgs = np.clip(imgs.astype(np.float64), 0, 255).astype(np.uint8)
    return imgs


def compute_ssim(
    pred_imgs: np.ndarray,
    gt_imgs: Optional[np.ndarray],
) -> Dict[str, Union[float, str]]:
    """
    SSIM (pair-wise) when paired real images exist.
    pred_imgs, gt_imgs: (N,H,W,3) or (N,3,H,W), 0-255.
    Returns dict with eval/ssim_mean, eval/ssim_std; or NA if no gt.
    """
    out = {"eval/ssim_mean": NA, "eval/ssim_std": NA}
    if gt_imgs is None or len(gt_imgs) == 0 or len(pred_imgs) != len(gt_imgs):
        return out
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return out
    pred_imgs = _to_uint8(pred_imgs)
    gt_imgs = _to_uint8(gt_imgs)
    vals = []
    for i in range(len(pred_imgs)):
        p, g = pred_imgs[i], gt_imgs[i]
        if p.shape[-1] != 3:
            p = np.transpose(p, (1, 2, 0))
        if g.shape[-1] != 3:
            g = np.transpose(g, (1, 2, 0))
        v = ssim(p, g, data_range=255, channel_axis=-1)
        vals.append(float(v))
    if vals:
        out["eval/ssim_mean"] = float(np.mean(vals))
        out["eval/ssim_std"] = float(np.std(vals))
    return out


def compute_clip_sim_image_image(
    pred_imgs: np.ndarray,
    gt_imgs: Optional[np.ndarray],
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, str]]:
    """
    CLIP cosine similarity (image-image) when paired real images exist.
    Returns eval/clip_sim_mean, eval/clip_sim_std; or NA if no gt.
    """
    out = {"eval/clip_sim_mean": NA, "eval/clip_sim_std": NA}
    if gt_imgs is None or len(gt_imgs) == 0 or len(pred_imgs) != len(gt_imgs):
        return out
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        return out
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_imgs = _to_uint8(pred_imgs)
    gt_imgs = _to_uint8(gt_imgs)
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    except Exception:
        return out
    from PIL import Image
    sims = []
    with torch.no_grad():
        for i in range(len(pred_imgs)):
            pil_p = Image.fromarray(pred_imgs[i])
            pil_g = Image.fromarray(gt_imgs[i])
            inp = processor(images=[pil_p, pil_g], return_tensors="pt").to(device)
            feats = model.get_image_features(**inp)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            s = (feats[0] @ feats[1]).item()
            sims.append(s)
    if sims:
        out["eval/clip_sim_mean"] = float(np.mean(sims))
        out["eval/clip_sim_std"] = float(np.std(sims))
    return out


def compute_clip_sim_image_text(
    pred_imgs: np.ndarray,
    text_prompts: List[str],
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, str]]:
    """
    CLIP similarity (image-text) when labels or text prompts are provided (e.g. MOABB).
    text_prompts: one per image, or one for all.
    Returns eval/clip_sim_mean, eval/clip_sim_std.
    """
    out = {"eval/clip_sim_mean": NA, "eval/clip_sim_std": NA}
    if not text_prompts or len(pred_imgs) == 0:
        return out
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        return out
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_imgs = _to_uint8(pred_imgs)
    if len(text_prompts) == 1:
        text_prompts = text_prompts * len(pred_imgs)
    if len(text_prompts) != len(pred_imgs):
        text_prompts = (text_prompts * (len(pred_imgs) // len(text_prompts) + 1))[:len(pred_imgs)]
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    except Exception:
        return out
    from PIL import Image
    sims = []
    with torch.no_grad():
        text_inp = processor(text=text_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        text_feats = model.get_text_features(**text_inp)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        for i in range(len(pred_imgs)):
            pil = Image.fromarray(pred_imgs[i])
            img_inp = processor(images=pil, return_tensors="pt").to(device)
            img_feat = model.get_image_features(**img_inp)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            s = (img_feat @ text_feats[i : i + 1].T).item()
            sims.append(s)
    if sims:
        out["eval/clip_sim_mean"] = float(np.mean(sims))
        out["eval/clip_sim_std"] = float(np.std(sims))
    return out


def _save_images_to_dir(imgs: np.ndarray, d: Path, prefix: str = "img") -> None:
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    imgs = _to_uint8(imgs)
    for i in range(len(imgs)):
        arr = imgs[i]
        if arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        Image.fromarray(arr).save(d / f"{prefix}_{i:04d}.png")


def compute_fid_is(
    pred_imgs: np.ndarray,
    real_imgs_dir: Optional[Union[str, Path]],
    temp_dir: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, str]]:
    """
    FID and Inception Score using torch_fidelity (or fallback).
    pred_imgs: (N,H,W,3) or (N,3,H,W), 0-255.
    real_imgs_dir: path to folder of real images (for FID). If None, FID is NA; IS can still be computed.
    temp_dir: where to save generated images; if None, use a temp directory.
    Returns eval/fid, eval/is_mean, eval/is_std; inapplicable values as NA.
    """
    out = {"eval/fid": NA, "eval/is_mean": NA, "eval/is_std": NA}
    pred_imgs = _to_uint8(pred_imgs)
    if len(pred_imgs) == 0:
        return out
    try:
        import torch_fidelity
    except ImportError:
        # Fallback: use torchmetrics FID only (no IS)
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from einops import rearrange
            if real_imgs_dir is None or not Path(real_imgs_dir).exists():
                return out
            fid_metric = FrechetInceptionDistance(feature=64).to(device or torch.device("cuda"))
            real_path = Path(real_imgs_dir)
            real_list = list(real_path.glob("*.png")) + list(real_path.glob("*.jpg"))[:len(pred_imgs)]
            from PIL import Image
            for p in real_list[: len(pred_imgs)]:
                img = np.array(Image.open(p).convert("RGB"))
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                t = torch.from_numpy(rearrange(img, "h w c -> 1 c h w")).float().to(fid_metric.device)
                fid_metric.update(t, real=True)
            pred_t = torch.from_numpy(rearrange(pred_imgs, "n h w c -> n c h w")).float().to(fid_metric.device)
            fid_metric.update(pred_t, real=False)
            out["eval/fid"] = float(fid_metric.compute().item())
            return out
        except Exception:
            return out

    tmp = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="eval_fid_"))
    gen_dir = tmp / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)
    _save_images_to_dir(pred_imgs, gen_dir, "gen")

    try:
        if real_imgs_dir and Path(real_imgs_dir).exists():
            metrics = torch_fidelity.calculate_metrics(
                str(gen_dir),
                str(real_imgs_dir),
                fid=True,
                isc=True,
                verbose=False,
            )
            out["eval/fid"] = float(metrics.get("frechet_inception_distance", np.nan))
            out["eval/is_mean"] = float(metrics.get("inception_score_mean", np.nan))
            out["eval/is_std"] = float(metrics.get("inception_score_std", np.nan))
        else:
            metrics = torch_fidelity.calculate_metrics(str(gen_dir), isc=True, verbose=False)
            out["eval/is_mean"] = float(metrics.get("inception_score_mean", np.nan))
            out["eval/is_std"] = float(metrics.get("inception_score_std", np.nan))
    except Exception:
        pass
    if temp_dir is None and gen_dir.exists():
        import shutil
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass
    return out


def all_eval_metrics(
    pred_imgs: np.ndarray,
    gt_imgs: Optional[np.ndarray] = None,
    real_imgs_dir: Optional[Union[str, Path]] = None,
    text_prompts: Optional[List[str]] = None,
    paired_images_available: bool = True,
    temp_dir: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, str]]:
    """
    Compute all applicable metrics; return one dict with eval/<dataset>/ prefixed keys
    when called from evaluate() (caller can add dataset prefix).
    paired_images_available: if False (e.g. MOABB), SSIM and image-image CLIP are NA.
    """
    n = len(pred_imgs)
    out = {
        "eval/n_samples": n,
        "eval/fid": NA,
        "eval/is_mean": NA,
        "eval/is_std": NA,
        "eval/ssim_mean": NA,
        "eval/ssim_std": NA,
        "eval/clip_sim_mean": NA,
        "eval/clip_sim_std": NA,
    }
    # FID / IS
    fid_is = compute_fid_is(pred_imgs, real_imgs_dir, temp_dir=temp_dir, device=device)
    out.update(fid_is)
    if paired_images_available and gt_imgs is not None:
        ssim_d = compute_ssim(pred_imgs, gt_imgs)
        out.update(ssim_d)
        clip_d = compute_clip_sim_image_image(pred_imgs, gt_imgs, device=device)
        out.update(clip_d)
    elif text_prompts:
        clip_d = compute_clip_sim_image_text(pred_imgs, text_prompts, device=device)
        out.update(clip_d)
    return out
