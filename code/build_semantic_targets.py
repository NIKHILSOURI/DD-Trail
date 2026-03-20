"""
Offline script: extract semantic targets from training GT images.
Output: semantic_targets.pt (per-sample packages + global metadata).
Usage:
  python code/build_semantic_targets.py --eeg_signals_path datasets/eeg_5_95_std.pth ^
    --splits_path datasets/block_splits_by_image_single.pth --imagenet_path /path/to/ILSVRC ^
    --out_path datasets/semantic_targets.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from dataset import create_EEG_dataset
from sarhm.semantic_targets import (
    _get_clip_encoder,
    extract_clip_image_embed,
    extract_clip_text_embed,
    build_semantic_target_package,
    save_semantic_targets,
)


def get_parser():
    p = argparse.ArgumentParser(description="Build semantic targets from training images")
    p.add_argument("--eeg_signals_path", type=str, default="datasets/eeg_5_95_std.pth")
    p.add_argument("--splits_path", type=str, default="datasets/block_splits_by_image_single.pth")
    p.add_argument("--imagenet_path", type=str, required=True)
    p.add_argument("--out_path", type=str, default="datasets/semantic_targets.pt")
    p.add_argument("--subject", type=int, default=4)
    p.add_argument("--max_items", type=int, default=None, help="Cap number of samples (default: all)")
    p.add_argument("--no_scene", action="store_true", help="Omit scene semantics (ablation).")
    p.add_argument("--no_object", action="store_true", help="Omit object semantics (ablation).")
    p.add_argument("--no_summary", action="store_true", help="Omit summary semantics (ablation).")
    p.add_argument("--no_region", action="store_true", help="Omit region semantics (ablation).")
    return p


def identity(x):
    return x


def main():
    args = get_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_train, _ = create_EEG_dataset(
        eeg_signals_path=args.eeg_signals_path,
        splits_path=args.splits_path,
        imagenet_path=args.imagenet_path,
        image_transform=identity,
        subject=args.subject,
    )
    model, processor = _get_clip_encoder(device)

    items = []
    n = len(dataset_train)
    if args.max_items is not None:
        n = min(n, args.max_items)

    for idx in tqdm(range(n), desc="Semantic targets"):
        sample = dataset_train[idx]
        image_raw = sample["image"]
        label = sample["label"]
        if hasattr(image_raw, "numpy"):
            image_raw = image_raw.numpy()
        if hasattr(image_raw, "cpu"):
            image_raw = image_raw.cpu().numpy()
        if image_raw.shape[-1] == 3:
            img_uint8 = (np.clip(np.asarray(image_raw), 0, 1) * 255).astype(np.uint8)
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img_uint8)
            image_t = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        else:
            image_t = torch.as_tensor(image_raw).float()
            if image_t.dim() == 3:
                image_t = image_t.unsqueeze(0)
        image_t = image_t.to(device)
        clip_embed = extract_clip_image_embed(image_t, model, processor, device)
        clip_embed = clip_embed.squeeze(0)

        scene_text = "" if getattr(args, "no_scene", False) else ""
        summary_text = "" if getattr(args, "no_summary", False) else f"class {int(label)}"
        object_text = "" if getattr(args, "no_object", False) else f"class {int(label)}"
        scene_embed = None if getattr(args, "no_scene", False) else extract_clip_text_embed([scene_text or "scene"], model, processor, device).squeeze(0)
        summary_embed = None if getattr(args, "no_summary", False) else extract_clip_text_embed([summary_text or "class"], model, processor, device).squeeze(0)
        object_embed = None if getattr(args, "no_object", False) else extract_clip_text_embed([object_text or "object"], model, processor, device).squeeze(0)
        region_embed = None  # region not computed in this script unless extended

        item = build_semantic_target_package(
            clip_img_embed=clip_embed,
            scene_text=scene_text,
            scene_embed=scene_embed,
            object_tags=[object_text] if object_text else [],
            object_text=object_text,
            object_embed=object_embed,
            summary_text=summary_text,
            summary_embed=summary_embed,
            region_embed=region_embed if not getattr(args, "no_region", True) else None,
            sample_id=idx,
            class_id=int(label),
            image_path=None,
        )
        items.append(item)

    save_semantic_targets(
        items,
        args.out_path,
        embedding_dim=768,
        encoder_name="openai/clip-vit-large-patch14",
        config={"max_items": args.max_items, "subject": args.subject},
    )
    print("Saved %d semantic targets to %s" % (len(items), args.out_path))


if __name__ == "__main__":
    main()
