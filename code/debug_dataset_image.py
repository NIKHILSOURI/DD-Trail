"""
Debug script: load EEG dataset and save 5 test images (real GT from ImageNet).
Requires imagenet_path so the dataset returns real images (no random noise).

Run from repo root:
  IMAGENET_PATH=/path/to/ILSVRC2012 PYTHONPATH=code python code/debug_dataset_image.py
  # Or with IMG_DEBUG=1 for per-item path/size logs:
  IMG_DEBUG=1 IMAGENET_PATH=/path/to/ILSVRC2012 PYTHONPATH=code python code/debug_dataset_image.py
"""
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from einops import rearrange

# Add code dir so "dataset" resolves when run from repo root
_CODE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_CODE)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from dataset import create_EEG_dataset, identity


def main():
    parser = argparse.ArgumentParser(description="Save 5 dataset GT images for verification.")
    parser.add_argument("--imagenet_path", type=str, default=None, help="ImageNet root (or set IMAGENET_PATH)")
    parser.add_argument("--eeg_signals_path", type=str, default=None, help="EEG pth (default: datasets/eeg_5_95_std.pth)")
    parser.add_argument("--splits_path", type=str, default=None, help="Splits pth (default: datasets/block_splits_by_image_single.pth)")
    parser.add_argument("--subject", type=int, default=4, help="Subject index")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to save (default 5)")
    args = parser.parse_args()

    imagenet_path = args.imagenet_path or os.environ.get("IMAGENET_PATH")
    if not imagenet_path or not str(imagenet_path).strip():
        print("ERROR: imagenet_path is required for real GT images.")
        print("  Set IMAGENET_PATH or pass --imagenet_path /path/to/ILSVRC2012")
        sys.exit(1)
    imagenet_path = os.path.normpath(os.path.abspath(imagenet_path))
    if not os.path.isdir(imagenet_path):
        print("ERROR: imagenet_path is not a directory: %s" % imagenet_path)
        sys.exit(1)

    eeg_path = args.eeg_signals_path or os.path.join(_REPO, "datasets", "eeg_5_95_std.pth")
    splits_path = args.splits_path or os.path.join(_REPO, "datasets", "block_splits_by_image_single.pth")
    if not os.path.isfile(eeg_path):
        print("ERROR: EEG file not found: %s" % eeg_path)
        sys.exit(1)
    if not os.path.isfile(splits_path):
        print("ERROR: Splits file not found: %s" % splits_path)
        sys.exit(1)

    _, dataset_test = create_EEG_dataset(
        eeg_signals_path=eeg_path,
        splits_path=splits_path,
        imagenet_path=imagenet_path,
        image_transform=identity,
        subject=args.subject,
    )
    n_test = len(dataset_test)
    print("Test set size:", n_test)
    if n_test == 0:
        print("Empty test set.")
        sys.exit(1)

    n_save = min(args.num_images, n_test)
    out_dir = os.path.join(_REPO, "debug_dataset_images")
    os.makedirs(out_dir, exist_ok=True)

    for i in range(n_save):
        item = dataset_test[i]
        img = item["image"]
        label = item["label"]
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy()
        else:
            img_np = np.asarray(img, dtype=np.float32)
        if img_np.ndim == 3 and img_np.shape[0] == 3:
            img_np = rearrange(img_np, "c h w -> h w c")
        if img_np.max() <= 1.0 and img_np.min() >= 0:
            img_np = (255.0 * img_np).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        out_path = os.path.join(out_dir, "debug_dataset_image%d.png" % i)
        Image.fromarray(img_np).save(out_path)
        print("Saved %s | idx=%d label=%s shape=%s min=%s max=%s mean=%.2f std=%.2f" % (
            out_path, i, label.item(), img_np.shape, img_np.min(), img_np.max(), img_np.mean(), img_np.std()))
    print("Done. Saved %d images to %s (must be natural images, not noise)." % (n_save, out_dir))


if __name__ == "__main__":
    main()
