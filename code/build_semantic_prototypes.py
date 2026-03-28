"""
Offline script: build semantic prototype memory from semantic_targets.pt.
Output: semantic_prototypes.pt (keys [N, 768], class_ids, metadata).
Usage:
  python code/build_semantic_prototypes.py --semantic_targets_path datasets/semantic_targets.pt ^
    --out_path datasets/semantic_prototypes.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from sarhm.semantic_targets import load_semantic_targets
from sarhm.semantic_memory import build_fused_keys, SemanticMemoryBank


def get_parser():
    p = argparse.ArgumentParser(description="Build semantic prototypes from semantic targets")
    p.add_argument("--semantic_targets_path", type=str, required=True)
    p.add_argument("--out_path", type=str, default="datasets/semantic_prototypes.pt")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--fusion_mode", type=str, default="concat_project")
    p.add_argument("--no_scene", action="store_true", help="Exclude scene from fused keys (ablation).")
    p.add_argument("--no_object", action="store_true", help="Exclude object from fused keys (ablation).")
    p.add_argument("--no_summary", action="store_true", help="Exclude summary from fused keys (ablation).")
    p.add_argument("--no_region", action="store_true", help="Exclude region from fused keys (ablation).")
    return p


def main():
    args = get_parser().parse_args()
    items, global_meta = load_semantic_targets(args.semantic_targets_path)
    if not items:
        raise RuntimeError("No items in semantic_targets.pt")
    N = len(items)
    clip_list = []
    scene_list = []
    obj_list = []
    summary_list = []
    region_list = []
    class_ids = []
    ref = items[0]["clip_img_embed"]
    for it in items:
        clip_list.append(it["clip_img_embed"])
        scene_list.append(it.get("scene_embed"))
        obj_list.append(it.get("object_embed"))
        summary_list.append(it.get("summary_embed"))
        region_list.append(it.get("region_embed"))
        class_ids.append(it.get("class_id", 0))
    clip_img = torch.stack(clip_list, dim=0)
    zero = torch.zeros_like(ref)
    scene_emb = None if getattr(args, "no_scene", False) else torch.stack([s if s is not None else zero for s in scene_list], dim=0)
    object_emb = None if getattr(args, "no_object", False) else torch.stack([o if o is not None else zero for o in obj_list], dim=0)
    summary_emb = None if getattr(args, "no_summary", False) else torch.stack([s if s is not None else zero for s in summary_list], dim=0)
    region_emb = None
    if not getattr(args, "no_region", False) and any(r is not None for r in region_list):
        region_emb = torch.stack([r if r is not None else zero for r in region_list], dim=0)
    keys = build_fused_keys(
        clip_img_embeds=clip_img,
        scene_embeds=scene_emb,
        object_embeds=object_emb,
        summary_embeds=summary_emb,
        region_embeds=region_emb,
        dim=args.dim,
        mode=args.fusion_mode,
    )
    bank = SemanticMemoryBank(keys=keys, dim=args.dim)
    bank.save_to_path(
        args.out_path,
        class_ids=torch.tensor(class_ids),
        metadata=items,
        config={"fusion_mode": args.fusion_mode, "num_prototypes": N},
    )
    print("Saved semantic prototypes N=%d to %s" % (N, args.out_path))


if __name__ == "__main__":
    main()
