"""
Class-level prototypes for SAR-HM Hopfield memory.
Prototypes live in semantic (CLIP-aligned) space.
Built only from training split (no leakage).
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class ClassPrototypes(nn.Module):
    """
    One prototype per class (K = num_classes) in semantic (CLIP-aligned) space.
    Built from training data; supports load/save for reproducibility.
    """

    def __init__(
        self,
        num_classes: int,
        dim: int,
        proto_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.proto_path = proto_path
        self._device = device
        # Learnable or fixed prototypes; default learnable so they can be built/updated
        self.prototypes = nn.Parameter(torch.zeros(num_classes, dim))
        nn.init.normal_(self.prototypes, std=0.02)
        self._counts: Optional[torch.Tensor] = None  # optional running counts per class

    def to(self, device, *args, **kwargs):
        self._device = device
        return super().to(device, *args, **kwargs)

    @property
    def P(self) -> torch.Tensor:
        """Memory matrix [K, dim]."""
        return self.prototypes

    def load_from_path(self, path: Optional[str] = None) -> bool:
        path = path or self.proto_path
        if path is None or not os.path.isfile(path):
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, torch.Tensor):
            P = state
        elif "prototypes" in state:
            P = state["prototypes"]
        elif "state_dict" in state and "prototypes" in state["state_dict"]:
            P = state["state_dict"]["prototypes"]
        else:
            P = state.get("P", state.get("class_prototypes"))
        if P is None:
            return False
        P = torch.as_tensor(P, dtype=self.prototypes.dtype)
        if P.shape != self.prototypes.shape:
            return False
        self.prototypes.data.copy_(P)
        return True

    def save_to_path(self, path: Optional[str] = None) -> str:
        path = path or self.proto_path
        if path is None:
            path = "prototypes.pt"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"prototypes": self.prototypes.detach().cpu()}, path)
        return path

    def save_to_path_with_metadata(
        self,
        path: Optional[str] = None,
        proto_source: str = "train",
        normalization_type: str = "layernorm",
        **extra_meta,
    ) -> str:
        path = path or self.proto_path
        if path is None:
            path = "prototypes.pt"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        import time
        K, D = self.prototypes.shape
        meta = {
            "K": K,
            "dim": D,
            "proto_source": proto_source,
            "normalization_type": normalization_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            **extra_meta,
        }
        try:
            import subprocess
            r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2)
            if r.returncode == 0 and r.stdout:
                meta["git_hash"] = r.stdout.strip()[:12]
        except Exception:
            meta["git_hash"] = None
        P = self.prototypes.detach().cpu().float()
        if P.shape[1] != 768:
            pass  # allow other dims; shape is [K, dim]
        P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        torch.save({"prototypes": P, "metadata": meta}, path)
        return path

    def update_from_batch(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        momentum: float = 0.99,
    ) -> None:
        """
        Update prototypes with batch (e.g. during training).
        embeddings: [B, dim], labels: [B] long.
        Exponential moving average per class.
        """
        B, D = embeddings.shape
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        with torch.no_grad():
            for k in range(self.num_classes):
                mask = labels == k
                if mask.any():
                    mean_k = embeddings[mask].mean(dim=0)
                    self.prototypes.data[k] = (
                        momentum * self.prototypes.data[k] + (1 - momentum) * mean_k
                    )


def build_prototypes_from_loader(
    loader,
    proj_fn,
    num_classes: int,
    dim: int,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> ClassPrototypes:
    """
    Build class prototypes from a DataLoader (training split only).
    proj_fn(eeg_tensor) -> [B, dim] in CLIP space.
    """
    prototypes = ClassPrototypes(num_classes=num_classes, dim=dim, device=device)
    prototypes.to(device)
    sums = torch.zeros(num_classes, dim, device=device)
    counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            eeg = batch["eeg"]
            if isinstance(eeg, torch.Tensor):
                eeg = eeg.to(device)
            else:
                eeg = torch.from_numpy(np.asarray(eeg)).float().to(device)
            if eeg.dim() == 2:
                eeg = eeg.unsqueeze(0)
            labels = batch["label"]
            if isinstance(labels, (int, np.integer)):
                labels = torch.tensor([labels], device=device, dtype=torch.long)
            else:
                labels = torch.as_tensor(labels, device=device, dtype=torch.long)

            z = proj_fn(eeg)
            if z.dim() == 3:
                z = z.mean(dim=1)
            for k in range(num_classes):
                mask = labels == k
                if mask.any():
                    sums[k] += z[mask].sum(dim=0)
                    counts[k] += mask.sum().item()

    for k in range(num_classes):
        if counts[k] > 0:
            prototypes.prototypes.data[k] = (sums[k] / counts[k]).cpu()
    return prototypes


def build_baseline_centroids(
    loader,
    cond_stage_model,
    num_classes: int,
    dim: int,
    device: torch.device,
    save_path: Optional[str] = None,
    max_batches: Optional[int] = None,
):
    """
    Build class prototypes from training EEG via baseline path: mae -> pool -> sarhm_projection.
    Stable memory (no random init). Saves to save_path if provided.
    """
    from .sarhm_modules import pool_eeg_tokens
    cond_stage_model.eval()
    cond_stage_model.to(device)

    def proj_fn(eeg):
        with torch.no_grad():
            lat = cond_stage_model.mae(eeg)
            pooled = pool_eeg_tokens(lat, cond_stage_model.global_pool)
            return cond_stage_model.sarhm_projection(pooled)

    prototypes = build_prototypes_from_loader(
        loader, proj_fn, num_classes=num_classes, dim=dim, device=device, max_batches=max_batches
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({"prototypes": prototypes.prototypes.detach().cpu()}, save_path)
    return prototypes


def build_prototypes_clip_text(
    num_classes: int,
    dim: int,
    device: torch.device,
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> ClassPrototypes:
    """
    Build prototypes from CLIP text encoder (e.g. "class_0", ... or custom class names).
    Uses ViT-L/14 for 768-dim to match SAR-HM clip_dim.
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        raise ImportError("clip_text proto_source requires transformers and CLIP.")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
    model = model.to(device).eval()
    texts = class_names or [f"class {k}" for k in range(num_classes)]
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model.get_text_features(**inputs)  # [K, text_dim]
    if out.shape[-1] != dim:
        proj = torch.nn.Linear(out.shape[-1], dim, device=device)
        torch.nn.init.xavier_uniform_(proj.weight)
        with torch.no_grad():
            out = proj(out)
    prototypes = ClassPrototypes(num_classes=num_classes, dim=dim, device=device)
    prototypes.prototypes.data = out.cpu().clone()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({"prototypes": prototypes.prototypes.detach().cpu()}, save_path)
    return prototypes
