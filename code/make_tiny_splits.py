"""
Create a tiny splits file for fast comparison runs (10 epochs, small train/test).
Usage: from repo root with venv activated:
  python code/make_tiny_splits.py
  python code/make_tiny_splits.py --train 200 --test 50
Output: datasets/block_splits_tiny.pth (same format as block_splits_by_image_single.pth).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

_CODE_DIR = Path(__file__).parent.absolute()
_REPO_ROOT = _CODE_DIR.parent.absolute()
_DATA_ROOT = Path(os.environ.get("DREAMDIFFUSION_DATA_ROOT", str(_REPO_ROOT / "datasets")))


def main():
    parser = argparse.ArgumentParser(description="Create tiny train/test splits for comparison runs")
    parser.add_argument("--source", type=str, default=None,
                        help="Source splits file (default: datasets/block_splits_by_image_single.pth)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output tiny splits file (default: datasets/block_splits_tiny.pth)")
    parser.add_argument("--train", type=int, default=150, help="Max number of train indices to keep")
    parser.add_argument("--test", type=int, default=40, help="Max number of test indices to keep")
    args = parser.parse_args()

    source = Path(args.source) if args.source else _DATA_ROOT / "block_splits_by_image_single.pth"
    output = Path(args.output) if args.output else _DATA_ROOT / "block_splits_tiny.pth"

    if not source.is_file():
        print(f"ERROR: Source not found: {source}")
        return 1

    data = torch.load(source, map_location="cpu", weights_only=False)
    if "splits" not in data:
        print(f"ERROR: Source has no 'splits' key. Keys: {list(data.keys())}")
        return 1

    splits = data["splits"]
    if not splits or "train" not in splits[0] or "test" not in splits[0]:
        print("ERROR: Expected splits[0] to have 'train' and 'test' keys")
        return 1

    train_idx = list(splits[0]["train"])[: args.train]
    test_idx = list(splits[0]["test"])[: args.test]
    tiny_splits = [{"train": train_idx, "test": test_idx}]
    out_data = {"splits": tiny_splits}

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_data, output)
    print(f"Created {output} with {len(train_idx)} train and {len(test_idx)} test indices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
