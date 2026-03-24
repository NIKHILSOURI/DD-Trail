"""
Inspect block split file structure for DreamDiffusion.
Output is copy-paste friendly for debugging.
Run from repo root: python scripts/inspect_splits.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

def inspect_split(pth_path):
    import torch
    if not os.path.exists(pth_path):
        print(f"FILE NOT FOUND: {pth_path}")
        return
    loaded = torch.load(pth_path, map_location='cpu', weights_only=False)
    print("=" * 60)
    print(f"File: {pth_path}")
    print("=" * 60)
    print(f"type(loaded) = {type(loaded).__name__}")
    print(f"Top-level keys = {list(loaded.keys())}")
    print()
    for k, v in loaded.items():
        print(f"  [{k}]")
        print(f"    type = {type(v).__name__}")
        if isinstance(v, (list, tuple)):
            print(f"    len = {len(v)}")
            if len(v) > 0:
                print(f"    first element type = {type(v[0]).__name__}")
                if isinstance(v[0], dict):
                    print(f"    first element keys = {list(v[0].keys())}")
                elif hasattr(v[0], 'tolist'):
                    print(f"    first 5 values = {v[:5]}")
                else:
                    print(f"    first 5 = {v[:5]}")
        elif isinstance(v, dict):
            print(f"    keys = {list(v.keys())}")
            for kk, vv in v.items():
                print(f"      [{kk}] type={type(vv).__name__}, len={len(vv) if hasattr(vv,'__len__') else 'N/A'}")
                if isinstance(vv, (list, tuple)) and len(vv) > 0:
                    sample = vv[0]
                    if hasattr(sample, 'tolist'):
                        print(f"        first 5 = {[x for x in vv[:5]]}")
                    else:
                        print(f"        first 5 = {vv[:5]}")
        else:
            print(f"    value = {v}")
        print()
    # Check expected structure for Splitter (dataset.py line 356)
    if "splits" in loaded:
        splits = loaded["splits"]
        print("SPLITTER CONSUMES: loaded['splits'][split_num][split_name]")
        print(f"  split_num=0, split_name='train' -> len(splits[0]['train']) = {len(splits[0]['train']) if splits and 'train' in splits[0] else 'N/A'}")
        print(f"  split_num=0, split_name='test'  -> len(splits[0]['test'])  = {len(splits[0]['test']) if splits and 'test' in splits[0] else 'N/A'}")
        if splits and 'train' in splits[0]:
            train_idx = splits[0]['train']
            print(f"  train indices: first 5 = {train_idx[:5]}, last 5 = {train_idx[-5:]}")
        if splits and 'test' in splits[0]:
            test_idx = splits[0]['test']
            print(f"  test indices:  first 5 = {test_idx[:5]}, last 5 = {test_idx[-5:]}")
    print()

if __name__ == "__main__":
    datasets_dir = REPO_ROOT / "datasets"
    paths = [
        datasets_dir / "block_splits_by_image_single.pth",
        datasets_dir / "block_splits_by_image_all.pth",
    ]
    for p in paths:
        inspect_split(str(p))
    if not any(os.path.exists(str(p)) for p in paths):
        print("No split files found. Place block_splits_by_image_single.pth or block_splits_by_image_all.pth in datasets/")
