"""
SAR-HM++: Semantic target extraction from GT images.
Per-sample package: CLIP image embed, scene/object/summary/region embeddings.
Offline preprocessing; used only during training as supervision.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .semantic_config import SEMANTIC_TARGETS_VERSION


def _get_clip_encoder(device: torch.device, use_fast: bool = True):
    """Load CLIP model and processor. Returns (model, processor)."""
    try:
        from transformers import AutoProcessor, CLIPModel
    except ImportError:
        raise ImportError("semantic_targets requires transformers and CLIP.")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=use_fast)
    return model.to(device).eval(), processor


def extract_clip_image_embed(
    images: torch.Tensor,
    model,
    processor,
    device: torch.device,
    allow_grad: bool = False,
) -> torch.Tensor:
    """
    Extract CLIP image features.
    images: [B, C, H, W] float in [0,1] or list of PIL; processor normalizes.
    Returns: [B, 768] L2-normalized.
    allow_grad: if True, do not use no_grad so gradients can flow (e.g. for training L_clip on generated images).
    """
    from PIL import Image
    import numpy as np

    if isinstance(images, torch.Tensor):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        B = images.shape[0]
        if allow_grad:
            images_for_proc = images
        else:
            imgs = []
            for i in range(B):
                t = images[i]
                if t.shape[0] == 3:
                    t = t.permute(1, 2, 0)
                arr = (t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                imgs.append(Image.fromarray(arr))
            images_for_proc = imgs
    else:
        images_for_proc = images
    inputs = processor(images=images_for_proc, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if allow_grad:
        out = model.get_image_features(**inputs)
    else:
        with torch.no_grad():
            out = model.get_image_features(**inputs)
    return F.normalize(out, dim=-1, eps=1e-12)


def extract_clip_text_embed(
    texts: List[str],
    model,
    processor,
    device: torch.device,
) -> torch.Tensor:
    """
    texts: list of str.
    Returns: [B, 768] L2-normalized.
    """
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    return F.normalize(out, dim=-1, eps=1e-12)


def build_semantic_target_package(
    clip_img_embed: torch.Tensor,
    scene_text: Optional[str] = None,
    scene_embed: Optional[torch.Tensor] = None,
    object_tags: Optional[List[str]] = None,
    object_text: Optional[str] = None,
    object_embed: Optional[torch.Tensor] = None,
    summary_text: Optional[str] = None,
    summary_embed: Optional[torch.Tensor] = None,
    region_embed: Optional[torch.Tensor] = None,
    sample_id: Optional[Any] = None,
    class_id: Optional[int] = None,
    image_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one semantic target item for saving (all tensors moved to CPU)."""
    item = {
        "clip_img_embed": clip_img_embed.cpu().float(),
        "scene_text": scene_text or "",
        "scene_embed": scene_embed.cpu().float() if scene_embed is not None else None,
        "object_tags": object_tags or [],
        "object_text": object_text or "",
        "object_embed": object_embed.cpu().float() if object_embed is not None else None,
        "summary_text": summary_text or "",
        "summary_embed": summary_embed.cpu().float() if summary_embed is not None else None,
        "region_embed": region_embed.cpu().float() if region_embed is not None else None,
        "has_region": region_embed is not None,
        "sample_id": sample_id,
        "class_id": class_id,
        "image_path": image_path,
        "metadata": metadata or {},
    }
    return item


def fuse_semantic_target(
    clip_img_embed: torch.Tensor,
    scene_embed: Optional[torch.Tensor] = None,
    object_embed: Optional[torch.Tensor] = None,
    summary_embed: Optional[torch.Tensor] = None,
    region_embed: Optional[torch.Tensor] = None,
    dim: int = 768,
    mode: str = "weighted_avg",
) -> torch.Tensor:
    """
    Fuse available embeddings into z_sem_gt for training supervision.
    mode: "weighted_avg" -> mean of available embeddings then normalize (deterministic);
          "concat_project" -> concat and project with a new Linear (not saved; use for single-batch only).
    Inputs: each [D] or [1, D]; clip_img_embed required.
    Returns: [dim] or [1, dim] L2-normalized (squeezed to 1d if single sample).
    """
    parts = [clip_img_embed]
    if scene_embed is not None:
        parts.append(scene_embed)
    if object_embed is not None:
        parts.append(object_embed)
    if summary_embed is not None:
        parts.append(summary_embed)
    if region_embed is not None:
        parts.append(region_embed)
    if mode == "weighted_avg":
        stacked = torch.stack([p.flatten() for p in parts], dim=0)
        out = stacked.mean(dim=0)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        if out.shape[-1] != dim and out.shape[-1] == 768:
            out = out[..., :dim]
        return F.normalize(out, dim=-1, eps=1e-12).squeeze(0)
    # concat_project
    concat = torch.cat([p.flatten().unsqueeze(0) for p in parts], dim=-1)
    if concat.dim() == 1:
        concat = concat.unsqueeze(0)
    in_dim = concat.shape[-1]
    if in_dim != dim:
        proj = torch.nn.Linear(in_dim, dim, device=concat.device, dtype=concat.dtype)
        torch.nn.init.xavier_uniform_(proj.weight)
        torch.nn.init.zeros_(proj.bias)
        out = proj(concat)
    else:
        out = concat
    return F.normalize(out, dim=-1, eps=1e-12).squeeze(0)


def fuse_semantic_target_from_item(
    item: Dict[str, Any],
    device: Optional[torch.device] = None,
    dim: int = 768,
    mode: str = "weighted_avg",
) -> torch.Tensor:
    """
    Build z_sem_gt from a saved semantic target item (e.g. for dataloader).
    item: dict from load_semantic_targets with keys clip_img_embed, scene_embed, etc.
    Returns: [dim] tensor on device, L2-normalized.
    """
    clip_img = item["clip_img_embed"]
    if device is not None:
        clip_img = clip_img.to(device)
    scene_embed = item.get("scene_embed")
    object_embed = item.get("object_embed")
    summary_embed = item.get("summary_embed")
    region_embed = item.get("region_embed")
    if device is not None:
        if scene_embed is not None:
            scene_embed = scene_embed.to(device)
        if object_embed is not None:
            object_embed = object_embed.to(device)
        if summary_embed is not None:
            summary_embed = summary_embed.to(device)
        if region_embed is not None:
            region_embed = region_embed.to(device)
    return fuse_semantic_target(
        clip_img,
        scene_embed=scene_embed,
        object_embed=object_embed,
        summary_embed=summary_embed,
        region_embed=region_embed,
        dim=dim,
        mode=mode,
    )


def load_semantic_targets(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load semantic_targets.pt.
    Returns: (list of per-sample packages, global_metadata).
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    items = state.get("items", state.get("targets", []))
    meta = state.get("global_metadata", state.get("metadata", {}))
    return items, meta


def save_semantic_targets(
    items: List[Dict[str, Any]],
    path: str,
    embedding_dim: int = 768,
    encoder_name: str = "openai/clip-vit-large-patch14",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Save semantic targets with global metadata."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    global_metadata = {
        "embedding_dim": embedding_dim,
        "encoder_name": encoder_name,
        "config": config or {},
        "version": SEMANTIC_TARGETS_VERSION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "num_items": len(items),
    }
    torch.save({"items": items, "global_metadata": global_metadata}, path)
    return path
