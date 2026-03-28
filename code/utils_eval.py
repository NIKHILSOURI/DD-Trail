"""
Shared evaluation helpers for compare_eval and other eval scripts.
No training logic; only loading, filtering, seeding, grids, and metrics.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torchvision.utils import make_grid
from einops import rearrange

try:
    from utils.state_dict_utils import filter_state_dict_for_model, log_filter_info
except ImportError:
    import sys
    from pathlib import Path
    _CODE_DIR = Path(__file__).resolve().parent
    if str(_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(_CODE_DIR))
    from utils.state_dict_utils import filter_state_dict_for_model, log_filter_info


def set_seed(seed: int) -> None:
    """Set RNG seed for reproducibility (torch, numpy, random)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def filter_state_dict(
    state_dict: dict,
    model_state_keys: Optional[set] = None,
    prune_unexpected_keys: bool = False,
) -> Tuple[dict, dict]:
    """Filter checkpoint state_dict for loading into model. Returns (filtered_sd, info_dict)."""
    return filter_state_dict_for_model(
        state_dict,
        model_state_keys=model_state_keys,
        drop_exact_keys=None,
        drop_prefixes=None,
        prune_unexpected_keys=prune_unexpected_keys,
    )


def load_model(
    checkpoint_path: str,
    config_patch: str,
    device: torch.device,
    pretrain_root: str = "pretrains",
    force_fp32: bool = False,
    num_voxels: Optional[int] = None,
) -> Tuple[Any, Any]:
    """
    Load eLDM_eval from checkpoint and config_patch.
    Uses safe state_dict filtering. Returns (generative_model, config).
    """
    from omegaconf import OmegaConf
    from dc_ldm.util import instantiate_from_config
    from dc_ldm.ldm_for_eeg import eLDM_eval
    from config import Config_Generative_Model

    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = sd.get("config")
    if config is None:
        config = Config_Generative_Model()
    # else use checkpoint config as-is (may be namespace or have __dict__)

    # SAR-HM++: set semantic_prototypes_path from checkpoint dir if not set (for compare_eval / eval with SAR-HM++ ckpt)
    if getattr(config, "use_sarhmpp", False) and not getattr(config, "semantic_prototypes_path", None):
        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        for name in ("semantic_prototypes.pt", "prototypes.pt"):
            candidate = os.path.join(ckpt_dir, name)
            if os.path.isfile(candidate):
                setattr(config, "semantic_prototypes_path", candidate)
                break

    num_voxels = num_voxels if num_voxels is not None else 512

    ldm = eLDM_eval(
        config_patch,
        num_voxels,
        device=device,
        pretrain_root=pretrain_root,
        logger=None,
        ddim_steps=getattr(config, "ddim_steps", 250),
        global_pool=getattr(config, "global_pool", False),
        use_time_cond=getattr(config, "use_time_cond", False),
        clip_tune=getattr(config, "clip_tune", True),
        cls_tune=getattr(config, "cls_tune", False),
        main_config=config,
    )

    model_keys = set(ldm.model.state_dict().keys())
    ckpt_raw = sd.get("model_state_dict") or sd.get("state_dict") or sd.get("model")
    if ckpt_raw is None:
        raise KeyError("Checkpoint has no model state (model_state_dict/state_dict/model).")
    ckpt_sd, filter_info = filter_state_dict(ckpt_raw, model_state_keys=model_keys, prune_unexpected_keys=False)
    log_filter_info(filter_info, tag="[CKPT_FILTER]")
    missing, unexpected = ldm.model.load_state_dict(ckpt_sd, strict=False)
    print("[CKPT_LOAD] missing=%d unexpected=%d" % (len(missing), len(unexpected)))

    if force_fp32:
        ldm.model = ldm.model.float()

    return ldm, config


def save_grid(images: Union[np.ndarray, List[np.ndarray]], path: str, nrow: Optional[int] = None) -> None:
    """Save a grid of images to path. images: (N, H, W, 3) uint8 or list of (H,W,3)."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    if isinstance(images, list):
        images = np.stack([np.asarray(x) for x in images], axis=0)
    if images.ndim == 3:
        images = images[np.newaxis, ...]
    if images.shape[-1] != 3:
        images = rearrange(images, "n c h w -> n h w c")
    if images.dtype != np.uint8:
        images = np.clip(images.astype(np.float64), 0, 255).astype(np.uint8)
    t = torch.from_numpy(images).float() / 255.0
    t = rearrange(t, "n h w c -> n c h w")
    grid = make_grid(t, nrow=nrow or min(5, len(t)))
    grid_np = (255.0 * rearrange(grid, "c h w -> h w c").numpy()).astype(np.uint8)
    from PIL import Image
    Image.fromarray(grid_np).save(path)


def _to_uint8_nhwc(imgs: np.ndarray) -> np.ndarray:
    if imgs.ndim == 3:
        imgs = imgs[np.newaxis, ...]
    if imgs.shape[-1] != 3:
        imgs = np.transpose(imgs, (0, 2, 3, 1))
    if imgs.dtype != np.uint8:
        imgs = np.clip(imgs.astype(np.float64), 0, 255).astype(np.uint8)
    return imgs


def compute_metrics(
    gen_images: np.ndarray,
    gt_images: Optional[np.ndarray],
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, str]]:
    """
    Compute SSIM, PCC, CLIP similarity (pair-wise gen vs gt).
    If gt_images is None or length 0, returns NA for those and adds mean_variance.
    """
    out = {
        "ssim_mean": "NA",
        "pcc_mean": "NA",
        "clip_sim_mean": "NA",
        "mean_variance": "NA",
        "n_samples": len(gen_images) if gen_images is not None else 0,
    }
    gen_images = np.asarray(gen_images)
    if len(gen_images) == 0:
        return out

    gen_images = _to_uint8_nhwc(gen_images)
    out["mean_variance"] = float(np.var(gen_images))

    if gt_images is None or len(gt_images) == 0 or len(gt_images) != len(gen_images):
        return out

    gt_images = _to_uint8_nhwc(np.asarray(gt_images))

    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_vals = []
        for i in range(len(gen_images)):
            p, g = gen_images[i], gt_images[i]
            if p.shape[-1] != 3:
                p = np.transpose(p, (1, 2, 0))
            if g.shape[-1] != 3:
                g = np.transpose(g, (1, 2, 0))
            ssim_vals.append(float(ssim(p, g, data_range=255, channel_axis=-1)))
        out["ssim_mean"] = float(np.mean(ssim_vals))
    except Exception:
        pass

    try:
        pcc_vals = []
        for i in range(len(gen_images)):
            p, g = gen_images[i].reshape(-1), gt_images[i].reshape(-1)
            if p.size != g.size:
                continue
            r = np.corrcoef(p.astype(np.float64), g.astype(np.float64))[0, 1]
            if np.isfinite(r):
                pcc_vals.append(float(r))
        if pcc_vals:
            out["pcc_mean"] = float(np.mean(pcc_vals))
    except Exception:
        pass

    try:
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
        sims = []
        with torch.no_grad():
            for i in range(len(gen_images)):
                pil_p = Image.fromarray(gen_images[i])
                pil_g = Image.fromarray(gt_images[i])
                inp = processor(images=[pil_p, pil_g], return_tensors="pt").to(dev)
                feats = model.get_image_features(**inp)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                sims.append((feats[0] @ feats[1]).item())
        if sims:
            out["clip_sim_mean"] = float(np.mean(sims))
    except Exception:
        pass

    return out
