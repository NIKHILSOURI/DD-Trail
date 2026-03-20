"""
SAR-HM++: Wrapper dataset that injects semantic target fields into training batches.
Uses precomputed semantic_targets.pt; matches by split index (same order as build_semantic_targets).
Training-only; do not use for inference.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from .semantic_targets import (
    load_semantic_targets,
    fuse_semantic_target_from_item,
)


class SemanticTargetWrapper(torch.utils.data.Dataset):
    """
    Wraps a dataset (e.g. Splitter over EEGDataset) and adds semantic target fields to each sample.
    Items in semantic_targets.pt are assumed to be in the same order as the wrapped dataset
    (e.g. built from the same train split). When an item is missing or index is out of range,
    semantic keys are still added but with zero tensors and has_semantic_gt=False so that
    collation works and the training loop can mask losses.
    """

    SEMANTIC_DIM = 768

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        semantic_targets_path: Optional[str] = None,
        use_semantic_targets: bool = True,
        fusion_mode: str = "weighted_avg",
        dim: int = 768,
        use_summary: bool = True,
        use_scene: bool = True,
        use_object: bool = True,
        use_region: bool = False,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.use_semantic_targets = use_semantic_targets and (semantic_targets_path or "").strip() != ""
        self.fusion_mode = fusion_mode
        self.dim = dim
        self.use_summary = use_summary
        self.use_scene = use_scene
        self.use_object = use_object
        self.use_region = use_region
        self._items: List[Dict[str, Any]] = []
        self._loaded = False
        self._semantic_targets_path = semantic_targets_path
        if self.use_semantic_targets and semantic_targets_path:
            try:
                self._items, _ = load_semantic_targets(semantic_targets_path)
                self._loaded = True
            except Exception as e:
                self._loaded = False
                import warnings
                warnings.warn(
                    "[SemanticTargetWrapper] Failed to load %s: %s. Semantic fields will be zeros and has_semantic_gt=False."
                    % (semantic_targets_path, e)
                )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _get_semantic_tensors(
        self,
        item: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Build z_sem_gt and component embeds from a loaded item; respect use_* flags."""
        out: Dict[str, Any] = {}
        clip_img = item.get("clip_img_embed")
        if clip_img is None:
            return out
        if not self.use_summary:
            item = {**item, "summary_embed": None}
        if not self.use_scene:
            item = {**item, "scene_embed": None}
        if not self.use_object:
            item = {**item, "object_embed": None}
        if not self.use_region:
            item = {**item, "region_embed": None}
        try:
            z_sem_gt = fuse_semantic_target_from_item(item, device=device, dim=self.dim, mode=self.fusion_mode)
        except Exception:
            return out
        out["z_sem_gt"] = z_sem_gt
        out["clip_img_embed_gt"] = clip_img.to(device) if device else clip_img
        out["summary_embed_gt"] = item.get("summary_embed")
        if out["summary_embed_gt"] is not None and device:
            out["summary_embed_gt"] = out["summary_embed_gt"].to(device)
        out["object_embed_gt"] = item.get("object_embed")
        if out["object_embed_gt"] is not None and device:
            out["object_embed_gt"] = out["object_embed_gt"].to(device)
        out["scene_embed_gt"] = item.get("scene_embed")
        if out["scene_embed_gt"] is not None and device:
            out["scene_embed_gt"] = out["scene_embed_gt"].to(device)
        out["has_region_semantics"] = bool(item.get("has_region") and item.get("region_embed") is not None)
        out["sample_id"] = item.get("sample_id")
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.base_dataset[idx]
        if not isinstance(base, dict):
            sample = {"eeg": base[0], "label": base[1], "image": base[2], "image_raw": base[3] if len(base) > 3 else None}
        else:
            sample = dict(base)
        if not self.use_semantic_targets or not self._loaded or idx >= len(self._items):
            # Add placeholder keys so collation works; training will skip semantic loss when has_semantic_gt is False
            sample["z_sem_gt"] = torch.zeros(self.dim, dtype=torch.float32)
            sample["clip_img_embed_gt"] = torch.zeros(self.dim, dtype=torch.float32)
            sample["summary_embed_gt"] = torch.zeros(self.dim, dtype=torch.float32)
            sample["object_embed_gt"] = torch.zeros(self.dim, dtype=torch.float32)
            sample["scene_embed_gt"] = torch.zeros(self.dim, dtype=torch.float32)
            sample["has_semantic_gt"] = False
            sample["has_region_semantics"] = False
            sample["sample_id"] = idx
            return sample
        item = self._items[idx]
        sem = self._get_semantic_tensors(item, device=None)
        if not sem:
            sample["z_sem_gt"] = torch.zeros(self.dim, dtype=torch.float32)
            sample["clip_img_embed_gt"] = torch.zeros(self.dim, dtype=torch.float32)
            sample["summary_embed_gt"] = torch.zeros(self.dim, dtype=torch.float32)
            sample["object_embed_gt"] = torch.zeros(self.dim, dtype=torch.float32)
            sample["scene_embed_gt"] = torch.zeros(self.dim, dtype=torch.float32)
            sample["has_semantic_gt"] = False
            sample["has_region_semantics"] = False
            sample["sample_id"] = idx
            return sample
        sample["z_sem_gt"] = sem["z_sem_gt"]
        sample["clip_img_embed_gt"] = sem["clip_img_embed_gt"]
        sample["summary_embed_gt"] = sem.get("summary_embed_gt") or torch.zeros(self.dim, dtype=torch.float32)
        sample["object_embed_gt"] = sem.get("object_embed_gt") or torch.zeros(self.dim, dtype=torch.float32)
        sample["scene_embed_gt"] = sem.get("scene_embed_gt") or torch.zeros(self.dim, dtype=torch.float32)
        sample["has_semantic_gt"] = True
        sample["has_region_semantics"] = sem.get("has_region_semantics", False)
        sample["sample_id"] = sem.get("sample_id", idx)
        return sample
