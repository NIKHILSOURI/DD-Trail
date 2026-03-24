"""
SAR-HM++: Multi-level semantic prototype memory.
Schema: keys (fused) [N, 768], optional component tensors, metadata.
Supports per-sample and clustered modes; load/save semantic_prototypes.pt.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_semantic_embeddings(
    clip_img: torch.Tensor,
    scene: Optional[torch.Tensor] = None,
    object_emb: Optional[torch.Tensor] = None,
    summary: Optional[torch.Tensor] = None,
    region: Optional[torch.Tensor] = None,
    mode: str = "concat_project",
    dim: int = 768,
    fusion_mlp: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Fuse multiple semantic embeddings into one vector.
    clip_img: [D] or [1, D]; others optional same shape.
    mode: "concat_project" (concat then linear to dim) | "weighted_avg".
    fusion_mlp: if given, applied to concat (input dim = sum of component dims).
    Returns: [dim] or [1, dim] L2-normalized (squeezed to 1d if single sample).
    """
    parts = [clip_img.flatten().unsqueeze(0) if clip_img.dim() == 1 else clip_img]
    if scene is not None:
        p = scene.flatten().unsqueeze(0) if scene.dim() == 1 else scene
        parts.append(p)
    if object_emb is not None:
        p = object_emb.flatten().unsqueeze(0) if object_emb.dim() == 1 else object_emb
        parts.append(p)
    if summary is not None:
        p = summary.flatten().unsqueeze(0) if summary.dim() == 1 else summary
        parts.append(p)
    if region is not None:
        p = region.flatten().unsqueeze(0) if region.dim() == 1 else region
        parts.append(p)
    if mode == "weighted_avg":
        stacked = torch.stack([p.squeeze(0) for p in parts], dim=0)
        out = stacked.mean(dim=0, keepdim=True)
        out = F.normalize(out, dim=-1, eps=1e-12)
        return out.squeeze(0)
    concat = torch.cat(parts, dim=-1)
    if fusion_mlp is not None:
        out = fusion_mlp(concat)
    else:
        in_dim = concat.shape[-1]
        if in_dim != dim:
            proj = nn.Linear(in_dim, dim, device=concat.device, dtype=concat.dtype)
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
            out = proj(concat)
        else:
            out = concat
    out = F.normalize(out, dim=-1, eps=1e-12)
    return out.squeeze(0) if out.dim() == 2 and out.shape[0] == 1 else out


class SemanticMemoryBank(nn.Module):
    """
    Holds semantic prototype keys [N, 768] for retrieval.
    Load/save compatible with semantic_prototypes.pt (version sarhmpp_v1).
    """

    def __init__(
        self,
        keys: Optional[torch.Tensor] = None,
        num_prototypes: int = 0,
        dim: int = 768,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        if keys is not None:
            keys = keys.float()
            if device is not None:
                keys = keys.to(device)
            self.register_buffer("keys", keys)
            self.num_prototypes = keys.shape[0]
        else:
            self.register_buffer(
                "keys",
                torch.zeros(max(1, num_prototypes), dim, device=device),
            )
            self.num_prototypes = num_prototypes

        self._class_ids: Optional[torch.Tensor] = None
        self._metadata: Optional[List[Dict[str, Any]]] = None

    @property
    def K(self) -> torch.Tensor:
        """Alias for keys [N, dim]."""
        return self.keys

    def load_from_path(self, path: Optional[str] = None) -> bool:
        """Load keys (and optional class_ids, metadata) from file. Returns True on success."""
        if path is None or not os.path.isfile(path):
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "keys" in state:
            keys = state["keys"]
        elif "prototypes" in state:
            keys = state["prototypes"]
        else:
            return False
        keys = torch.as_tensor(keys, dtype=torch.float32)
        if keys.dim() != 2:
            return False
        N, D = keys.shape[0], keys.shape[1]
        if self.num_prototypes == 0:
            self.num_prototypes = N
            self.dim = D
            self.register_buffer("keys", keys.to(self.keys.device))
        else:
            if keys.shape[0] != self.num_prototypes or keys.shape[1] != self.dim:
                return False
            self.keys.data.copy_(keys.to(self.keys.device))
        self._class_ids = state.get("class_ids")
        self._metadata = state.get("metadata")
        return True

    def save_to_path(
        self,
        path: str,
        class_ids: Optional[Union[torch.Tensor, List[int]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save keys and optional class_ids, metadata to file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "keys": self.keys.detach().cpu().float(),
            "version": "sarhmpp_v1",
            "config": config or {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        }
        if class_ids is not None:
            payload["class_ids"] = (
                class_ids.cpu() if isinstance(class_ids, torch.Tensor) else torch.tensor(class_ids)
            )
        if metadata is not None:
            payload["metadata"] = metadata
        torch.save(payload, path)
        return path


def build_fused_keys(
    clip_img_embeds: torch.Tensor,
    scene_embeds: Optional[torch.Tensor] = None,
    object_embeds: Optional[torch.Tensor] = None,
    summary_embeds: Optional[torch.Tensor] = None,
    region_embeds: Optional[torch.Tensor] = None,
    dim: int = 768,
    mode: str = "concat_project",
    fusion_mlp: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Build [N, dim] fused keys from component tensors.
    Each component: [N, 768] (or broadcastable). Missing components omitted.
    mode: "concat_project" | "weighted_avg". fusion_mlp: optional nn.Module(concat_dim -> dim).
    Returns: [N, dim] L2-normalized.
    """
    parts = [clip_img_embeds]
    if scene_embeds is not None:
        parts.append(scene_embeds)
    if object_embeds is not None:
        parts.append(object_embeds)
    if summary_embeds is not None:
        parts.append(summary_embeds)
    if region_embeds is not None:
        parts.append(region_embeds)
    if mode == "weighted_avg":
        stacked = torch.stack(parts, dim=0)
        out = stacked.mean(dim=0)
    elif fusion_mlp is not None:
        concat = torch.cat(parts, dim=-1)
        out = fusion_mlp(concat)
    else:
        concat = torch.cat(parts, dim=-1)
        in_dim = concat.shape[-1]
        if in_dim != dim:
            proj = nn.Linear(in_dim, dim, device=concat.device, dtype=concat.dtype)
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
            out = proj(concat)
        else:
            out = concat
    return F.normalize(out, dim=-1, eps=1e-12)
