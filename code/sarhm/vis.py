"""
Professor-friendly SAR-HM visualization: attention bar chart and baseline-vs-SAR-HM comparison.
Uses matplotlib only. No heavy dependencies.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


def save_hopfield_attention_bar(
    attn_vec: Union[np.ndarray, "torch.Tensor"],
    class_names_or_ids: Optional[List[str]] = None,
    out_path: Union[str, Path] = "hopfield_attention_sample0.png",
    top_k: int = 5,
) -> str:
    """
    Save a bar chart of top-k Hopfield attention weights.
    attn_vec: shape (K,) or (1, K); will take first row if 2D.
    class_names_or_ids: optional labels for x-axis (length K or at least top_k indices).
    out_path: output PNG path.
    top_k: number of top classes to show (default 5).
    Returns: absolute path of saved file.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""
    if hasattr(attn_vec, "numpy"):
        attn_vec = attn_vec.detach().cpu().numpy()
    attn_vec = np.asarray(attn_vec, dtype=np.float64)
    if attn_vec.ndim == 2:
        attn_vec = attn_vec[0]
    K = attn_vec.size
    top_k = min(top_k, K)
    idx = np.argsort(-attn_vec)[:top_k]
    weights = attn_vec[idx]
    if class_names_or_ids is not None and len(class_names_or_ids) >= K:
        labels = [str(class_names_or_ids[i]) for i in idx]
    else:
        labels = [str(int(i)) for i in idx]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(labels)), weights, color="steelblue", edgecolor="navy")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Attention weight")
    ax.set_title("Hopfield retrieval (top-%d classes)" % top_k)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_path.resolve())


def save_baseline_vs_sarhm_grid(
    img_baseline: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    img_sarhm: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    out_path: Union[str, Path] = "compare_baseline_vs_sarhm_sample0.png",
    sample_idx: int = 0,
) -> str:
    """
    Save a side-by-side comparison: baseline | SAR-HM.
    If only one image is provided, save that one only (no comparison).
    Images: HWC uint8 [0,255] or float [0,1]; will be normalized to uint8.
    Returns: absolute path of saved file.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_np_uint8(img):
        if img is None:
            return None
        if hasattr(img, "numpy"):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8 and np.issubdtype(img.dtype, np.floating):
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return img

    base = _to_np_uint8(img_baseline)
    sarhm = _to_np_uint8(img_sarhm)
    if base is None and sarhm is None:
        return ""
    if base is not None and sarhm is not None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(base)
        axes[0].set_title("Baseline")
        axes[0].axis("off")
        axes[1].imshow(sarhm)
        axes[1].set_title("SAR-HM")
        axes[1].axis("off")
        fig.suptitle("Baseline vs SAR-HM (sample %d)" % sample_idx)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(base if base is not None else sarhm)
        ax.set_title("Baseline" if base is not None else "SAR-HM")
        ax.axis("off")
        fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out_path.resolve())


# Optional: torch type hint for vis without importing torch
try:
    import torch as _torch
except ImportError:
    _torch = None
