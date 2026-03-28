"""
SAR-HM++: Semantic alignment and retrieval consistency losses.
Training-only: supervise q_sem and m_sem with z_sem_gt; optional CLIP and object losses.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def cosine_similarity_loss(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """1 - mean(cosine_sim). a, b same shape [..., D]. Returns scalar."""
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    sim = F.cosine_similarity(a, b, dim=dim)
    return (1.0 - sim).mean()


def semantic_alignment_loss(q_sem: torch.Tensor, z_sem_gt: torch.Tensor) -> torch.Tensor:
    """L_sem_align = 1 - cosine(q_sem, z_sem_gt). Both [B, D]."""
    return cosine_similarity_loss(q_sem, z_sem_gt)


def retrieval_consistency_loss(m_sem: torch.Tensor, z_sem_gt: torch.Tensor) -> torch.Tensor:
    """L_retr = 1 - cosine(m_sem, z_sem_gt). Both [B, D]."""
    return cosine_similarity_loss(m_sem, z_sem_gt)


def clip_image_loss(gen_embed: torch.Tensor, gt_embed: torch.Tensor) -> torch.Tensor:
    """L_clip_img = 1 - cosine(CLIP(I_gen), CLIP(I_gt)). Both [B, 768]."""
    return cosine_similarity_loss(gen_embed, gt_embed)


def clip_text_loss(
    image_embed: torch.Tensor,
    text_embed: torch.Tensor,
) -> torch.Tensor:
    """L_clip_text = 1 - cosine(CLIP_img(I_gen), CLIP_text(T_gt)). Both [B, 768]."""
    return cosine_similarity_loss(image_embed, text_embed)


def object_consistency_loss(m_obj: torch.Tensor, z_obj_gt: torch.Tensor) -> torch.Tensor:
    """L_obj = 1 - cosine(m_obj, z_obj_gt). Optional. Both [B, D]."""
    return cosine_similarity_loss(m_obj, z_obj_gt)


def compute_semantic_losses(
    q_sem: Optional[torch.Tensor] = None,
    m_sem: Optional[torch.Tensor] = None,
    z_sem_gt: Optional[torch.Tensor] = None,
    lambda_sem: float = 0.1,
    lambda_retr: float = 0.1,
    lambda_clip_img: float = 0.0,
    lambda_clip_text: float = 0.0,
    clip_img_gen: Optional[torch.Tensor] = None,
    clip_img_gt: Optional[torch.Tensor] = None,
    clip_text_gt: Optional[torch.Tensor] = None,
    lambda_obj: float = 0.0,
    m_obj: Optional[torch.Tensor] = None,
    z_obj_gt: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute requested semantic losses; only includes terms where inputs are provided.
    Returns: (total weighted loss, dict of individual loss tensors for logging).
    """
    device = None
    for t in (q_sem, m_sem, z_sem_gt, clip_img_gen, clip_img_gt, clip_text_gt, m_obj, z_obj_gt):
        if t is not None:
            device = t.device
            break
    total = torch.tensor(0.0, device=device or torch.device("cpu"), dtype=torch.float32)
    loss_dict: Dict[str, torch.Tensor] = {}

    if q_sem is not None and z_sem_gt is not None and lambda_sem > 0:
        L_sem = semantic_alignment_loss(q_sem, z_sem_gt)
        total = total + lambda_sem * L_sem
        loss_dict["loss_sem_align"] = L_sem

    if m_sem is not None and z_sem_gt is not None and lambda_retr > 0:
        L_retr = retrieval_consistency_loss(m_sem, z_sem_gt)
        total = total + lambda_retr * L_retr
        loss_dict["loss_retr"] = L_retr

    if clip_img_gen is not None and clip_img_gt is not None and lambda_clip_img > 0:
        L_clip_img = clip_image_loss(clip_img_gen, clip_img_gt)
        total = total + lambda_clip_img * L_clip_img
        loss_dict["loss_clip_img"] = L_clip_img

    if clip_img_gen is not None and clip_text_gt is not None and lambda_clip_text > 0:
        L_clip_text = clip_text_loss(clip_img_gen, clip_text_gt)
        total = total + lambda_clip_text * L_clip_text
        loss_dict["loss_clip_text"] = L_clip_text

    if m_obj is not None and z_obj_gt is not None and lambda_obj > 0:
        L_obj = object_consistency_loss(m_obj, z_obj_gt)
        total = total + lambda_obj * L_obj
        loss_dict["loss_obj"] = L_obj

    return total, loss_dict
