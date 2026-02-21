"""
Create a tiny split file for minimal proof-of-pipeline demo (RTX 3060 6GB).
Preserves exact structure expected by Splitter (dataset.py).

Usage (from repo root):
  python scripts/make_tiny_splits.py
  python scripts/make_tiny_splits.py --input datasets/block_splits_by_image_all.pth --N_train 50 --N_val 10 --N_test 10
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=str(REPO_ROOT / "datasets" / "block_splits_by_image_single.pth"),
                        help="Input split file path")
    parser.add_argument("--output", type=str,
                        default=str(REPO_ROOT / "datasets" / "block_splits_tiny.pth"),
                        help="Output tiny split file path")
    parser.add_argument("--N_train", type=int, default=50)
    parser.add_argument("--N_val", type=int, default=10,
                        help="Used only if 'val' exists in split; otherwise ignored")
    parser.add_argument("--N_test", type=int, default=10)
    args = parser.parse_args()

    import torch
    if not os.path.exists(args.input):
        print(f"Error: input not found: {args.input}")
        sys.exit(1)

    loaded = torch.load(args.input, map_location='cpu', weights_only=False)
    # Expected: {"splits": [ {"train": [...], "test": [...], optionally "val": [...]}, ... ]}
    if "splits" not in loaded:
        print(f"Error: expected key 'splits' in {args.input}, got keys {list(loaded.keys())}")
        sys.exit(1)

    new_splits = []
    for split_idx, split_dict in enumerate(loaded["splits"]):
        new_dict = {}
        for split_name, indices in split_dict.items():
            lst = list(indices)  # handle list or tensor
            if split_name == "train":
                n = min(args.N_train, len(lst))
            elif split_name == "val":
                n = min(args.N_val, len(lst))
            elif split_name == "test":
                n = min(args.N_test, len(lst))
            else:
                n = min(10, len(lst))  # arbitrary cap for unknown keys
            new_dict[split_name] = lst[:n]
        new_splits.append(new_dict)

    out_obj = {"splits": new_splits}
    # Preserve any other keys from original (e.g. metadata)
    for k, v in loaded.items():
        if k != "splits":
            out_obj[k] = v

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(out_obj, args.output)

    print(f"Created {args.output}")
    for i, d in enumerate(new_splits):
        print(f"  split[{i}]: " + ", ".join(f"{k}={len(v)}" for k, v in d.items()))

if __name__ == "__main__":
    main()
