"""
Shared state_dict filtering for Stage B and Stage C checkpoint loading.
Safe filtering: only drop known incompatible keys (e.g. HF position_ids), not whole prefixes,
unless explicitly requested. Optional prune_unexpected_keys keeps only keys in model (default off).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Any

# Always drop this exact key (HuggingFace/CLIP image_embedder; not used by our model)
DEFAULT_DROP_EXACT: Set[str] = {
    "image_embedder.transformer.vision_model.embeddings.position_ids",
}


def filter_state_dict_for_model(
    state_dict: dict,
    model_state_keys: Optional[Set[str]] = None,
    *,
    drop_exact_keys: Optional[List[str]] = None,
    drop_prefixes: Optional[List[str]] = None,
    prune_unexpected_keys: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Filter a state_dict before load_state_dict(strict=False).

    - Always drops exact key: image_embedder.transformer.vision_model.embeddings.position_ids
    - Optionally drops other exact keys via drop_exact_keys
    - Does NOT drop whole embeddings prefix by default (only drop_prefixes if provided)
    - If prune_unexpected_keys is True and model_state_keys is provided, keeps only keys in model
      (default False to avoid hiding bugs)

    Returns:
        (filtered_sd, info_dict)
        info_dict: dropped_exact, dropped_prefix, dropped_not_in_model (if enabled), kept
    """
    drop_exact = set(DEFAULT_DROP_EXACT)
    if drop_exact_keys:
        drop_exact.update(drop_exact_keys)
    drop_prefixes = drop_prefixes or []

    dropped_exact: List[str] = []
    dropped_prefix: List[str] = []
    dropped_not_in_model: List[str] = []
    out: Dict[str, Any] = {}

    for k, v in state_dict.items():
        if k in drop_exact:
            dropped_exact.append(k)
            continue
        if any(k.startswith(p) for p in drop_prefixes):
            dropped_prefix.append(k)
            continue
        if prune_unexpected_keys and model_state_keys is not None and k not in model_state_keys:
            dropped_not_in_model.append(k)
            continue
        out[k] = v

    info = {
        "dropped_exact": dropped_exact,
        "dropped_prefix": dropped_prefix,
        "dropped_not_in_model": dropped_not_in_model,
        "kept": len(out),
    }
    return out, info


def log_filter_info(info: Dict[str, Any], tag: str = "[CKPT_FILTER]") -> None:
    """Print one-line summary and first 10 dropped keys if any."""
    exact = len(info["dropped_exact"])
    prefix = len(info["dropped_prefix"])
    not_in = len(info["dropped_not_in_model"])
    kept = info["kept"]
    print("%s dropped_exact=%d dropped_prefix=%d dropped_not_in_model=%d kept=%d" % (tag, exact, prefix, not_in, kept))
    all_dropped = (info["dropped_exact"] + info["dropped_prefix"] + info["dropped_not_in_model"])[:10]
    if all_dropped:
        print("%s first 10 dropped keys: %s" % (tag, all_dropped))


def is_mae_pretrain_ckpt(sd: dict) -> bool:
    """
    Heuristic: True if checkpoint looks like MAE/MBM pretrain (not Stage-B LatentDiffusion).
    Conservative: requires model_state_dict to have MAE-ish keys and to lack LDM keys.
    """
    state_dict = sd.get("model_state_dict") or sd.get("state_dict") or sd.get("model")
    if state_dict is None or not isinstance(state_dict, dict):
        return False
    keys = set(state_dict.keys())
    # MAE pretrain typically has cls_token, pos_embed, or patch_embed (possibly with prefix)
    mae_ish = any(
        "cls_token" in k or "pos_embed" in k or "patch_embed" in k
        for k in keys
    )
    # Stage-B LDM has these prefixes
    has_ldm = any(
        "model.diffusion_model" in k or "first_stage_model" in k
        for k in keys
    )
    has_config = "config" in sd
    has_state = "state" in sd
    return bool(has_config and has_state and mae_ish and not has_ldm)
