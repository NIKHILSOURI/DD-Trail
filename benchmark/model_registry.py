"""
Unified model registry: thoughtviz, dreamdiffusion, sarhm.
Each model wrapper: load(), generate_from_eeg(samples) -> list of images, save_outputs().
No SAR-HM++ in this benchmark.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .benchmark_config import BenchmarkConfig
from .output_standardizer import _to_uint8
from .utils import setup_logger

log = setup_logger(__name__)
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = BENCHMARK_ROOT / "code"


def _ensure_code_on_path() -> None:
    if CODE_DIR.is_dir() and str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))


# eLDM MAE PatchEmbed1D: in_chans=128, time_len=512 → tensor layout (B, 128, 512).
# eLDM_eval.generate uses einops repeat(latent, 'h w -> c h w'), so each item must be 2D (128, 512).
_LDM_EEG_H = 128
_LDM_EEG_W = 512

# ThoughtViz Keras EEG encoder: fixed spatial grid (see pretrained classifier input).
_THOUGHTVIZ_H = 14
_THOUGHTVIZ_W = 32


def _prepare_eeg_for_thoughtviz(eeg: Any) -> np.ndarray:
    """
    Map any benchmark EEG tensor/array to float32 (14, 32, 1) for ThoughtViz.
    ImageNet-EEG is (128, 512); we resize with scipy (order=1) to match the Keras model.
    """
    th, tw = _THOUGHTVIZ_H, _THOUGHTVIZ_W
    if isinstance(eeg, np.ndarray):
        x = np.asarray(eeg, dtype=np.float32)
    elif hasattr(eeg, "detach"):
        x = eeg.detach().float().cpu().numpy()
    elif hasattr(eeg, "numpy"):
        x = np.asarray(eeg.numpy(), dtype=np.float32)
    else:
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
        raise ValueError(
            "ThoughtViz EEG must be 2D or H×W×C; got shape %s" % (x.shape,)
        )
    if plane.shape == (th, tw):
        return plane[..., np.newaxis].astype(np.float32)
    from scipy.ndimage import zoom

    zh = th / plane.shape[0]
    zw = tw / plane.shape[1]
    plane = zoom(plane, (zh, zw), order=1).astype(np.float32)
    return plane[..., np.newaxis]


def _prepare_eeg_for_ldm(eeg: Any) -> Any:
    """
    Convert dataset EEG to float32 [128, 512] for DreamDiffusion/SAR-HM generate().
    ImageNet-EEG is already (128, 512); ThoughtViz and others are resized with bilinear
    so the checkpoint encoder receives the expected grid size (not identical semantics to training).
    """
    import torch
    import torch.nn.functional as F

    if isinstance(eeg, torch.Tensor):
        t = eeg.detach().float().cpu()
    else:
        t = torch.as_tensor(np.asarray(eeg), dtype=torch.float32)
    # Accidental batch dimension from older code paths: (1, H, W)
    if t.dim() == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    while t.dim() > 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    if t.dim() != 2:
        raise ValueError(
            "EEG must be 2D [H,W] (or 1×H×W) for LDM conditioning; got shape %s" % (tuple(t.shape),)
        )
    h, w = _LDM_EEG_H, _LDM_EEG_W
    if t.shape == (w, h):
        t = t.t()
    elif t.shape != (h, w):
        x = t.unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        t = x.squeeze(0).squeeze(0)
    return t.contiguous()


class _ListDataset:
    """Minimal dataset wrapper for LDM generate(): list of dicts with 'eeg' and 'image'."""

    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.items[i]


def get_thoughtviz_wrapper(config: BenchmarkConfig, dataset_name: Optional[str] = None):
    """Return ThoughtViz wrapper instance (lazy load on first generate)."""
    _ensure_code_on_path()
    try:
        from thoughtviz_integration.model_wrapper import ThoughtVizWrapper
        from thoughtviz_integration.config import ThoughtVizConfig
    except ImportError as e:
        log.warning("ThoughtViz wrapper not available: %s", e)
        return None
    eeg_model_path = config.thoughtviz_eeg_model_path
    gan_model_path = config.thoughtviz_gan_model_path
    # Guard against accidental Char checkpoints on ImageNet-EEG runs.
    if dataset_name == "imagenet_eeg":
        eeg_s = str(eeg_model_path or "").lower()
        gan_s = str(gan_model_path or "").lower()
        if any(k in eeg_s for k in ("/char/", "\\char\\", "/digit/", "\\digit\\")) or any(
            k in gan_s for k in ("/char/", "\\char\\", "/digit/", "\\digit\\")
        ):
            msg = (
                "ThoughtViz checkpoints look non-image for imagenet_eeg dataset "
                "(eeg=%s, gan=%s). Image checkpoints are recommended for meaningful outputs."
            )
            if getattr(config, "thoughtviz_strict_checkpoint_match", False):
                log.error(msg, eeg_model_path, gan_model_path)
                return None
            log.warning(msg, eeg_model_path, gan_model_path)
    tv_config = ThoughtVizConfig(
        data_dir=config.thoughtviz_data_dir,
        image_dir=config.thoughtviz_image_dir,
        eeg_model_path=eeg_model_path,
        gan_model_path=gan_model_path,
    )
    return ThoughtVizWrapper(config=tv_config)


def get_dreamdiffusion_wrapper(config: BenchmarkConfig, use_sarhm: bool = False):
    """Load DreamDiffusion (baseline or SAR-HM) via utils_eval; return (generative_model, device)."""
    import torch

    _ensure_code_on_path()
    from utils_eval import load_model
    ckpt = config.sarhm_ckpt if use_sarhm else config.dreamdiffusion_baseline_ckpt
    if not ckpt or not Path(ckpt).exists():
        log.warning("Checkpoint not found: %s", ckpt)
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading checkpoint (this can take a few minutes): %s", ckpt)
    try:
        model, cfg = load_model(
            ckpt,
            config.config_patch,
            device,
            pretrain_root=config.pretrain_root,
            num_voxels=512,
        )
        if use_sarhm and config.sarhm_proto_path and Path(config.sarhm_proto_path).exists():
            # Ensure prototypes are loaded (handled inside load_model via config from ckpt; proto_path can be set on config)
            cfg.proto_path = config.sarhm_proto_path
            # Reload if needed; for now load_model uses ckpt config - proto_path may be in ckpt dir
            pass
        log.info("Checkpoint ready: %s", ckpt)
        return (model, device, cfg)
    except Exception as e:
        log.exception("DreamDiffusion load failed: %s", e)
        return None


def generate_dreamdiffusion(
    model_device_cfg: Tuple[Any, Any, Any],
    samples: List[Dict[str, Any]],
    num_samples_per_item: int = 1,
    ddim_steps: int = 250,
    pbar_desc: Optional[str] = None,
) -> List[np.ndarray]:
    """Run DreamDiffusion/SAR-HM generate on list of unified samples; return list of (H,W,3) uint8."""
    import torch

    model, device, cfg = model_device_cfg
    # Build minimal dataset with 'eeg' and 'image' (GT for concatenation in generate)
    items = []
    for s in samples:
        eeg = s.get("eeg")
        gt = s.get("gt_image")
        if gt is None and s.get("gt_image_path"):
            from PIL import Image
            gt = np.array(Image.open(s["gt_image_path"]).convert("RGB")) / 255.0
        if gt is None:
            gt = np.zeros((64, 64, 3), dtype=np.float32)  # placeholder
        eeg = _prepare_eeg_for_ldm(eeg)
        if isinstance(gt, np.ndarray):
            if gt.ndim == 2:
                gt = np.stack([gt] * 3, axis=-1)
            if gt.shape[-1] != 3:
                gt = np.transpose(gt, (1, 2, 0))
        gt_t = torch.from_numpy(np.asarray(gt, dtype=np.float32))
        if gt_t.dim() == 2:
            gt_t = gt_t.unsqueeze(-1).expand(-1, -1, 3)
        items.append({"eeg": eeg, "image": gt_t})
    ds = _ListDataset(items)
    grid, all_samples = model.generate(
        ds,
        num_samples=num_samples_per_item,
        ddim_steps=ddim_steps,
        HW=None,
        limit=len(items),
        state=None,
        output_path=None,
        cfg_scale=1.0,
        cfg_uncond="zeros",
        pbar_desc=pbar_desc,
    )
    # all_samples: numpy (N, 1+num_samples, C, H, W) uint8 from eLDM_eval.generate, or legacy torch list
    out = []
    for batch in all_samples:
        # batch: (1+num_samples, C, H, W)
        gen = batch[1]  # first generated (skip GT at index 0)
        if isinstance(gen, torch.Tensor):
            arr = gen.detach().float().cpu().numpy()
        else:
            arr = np.asarray(gen)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        out.append((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))
    return out


def generate_thoughtviz(
    wrapper: Any,
    samples: List[Dict[str, Any]],
) -> List[np.ndarray]:
    """Run ThoughtViz on list of samples; return list of (H,W,3) uint8."""
    eeg_list = [_prepare_eeg_for_thoughtviz(s["eeg"]) for s in samples]
    return wrapper.generate_from_eeg(eeg_list, num_samples=1)


def get_model(
    name: str,
    config: BenchmarkConfig,
    dataset_name: Optional[str] = None,
) -> Optional[Any]:
    """
    Get model by name. Returns wrapper or (model, device, cfg) for DreamDiffusion/SAR-HM.
    name: 'thoughtviz' | 'dreamdiffusion' | 'sarhm'
    """
    if name == "thoughtviz":
        return get_thoughtviz_wrapper(config, dataset_name=dataset_name)
    if name == "dreamdiffusion":
        return get_dreamdiffusion_wrapper(config, use_sarhm=False)
    if name == "sarhm":
        return get_dreamdiffusion_wrapper(config, use_sarhm=True)
    log.warning("Unknown model: %s", name)
    return None
