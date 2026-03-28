"""
Unified dataset registry: imagenet_eeg, thoughtviz.
Exposes a common sample interface: sample_id, eeg, gt_image (path or array), label, metadata, split.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .benchmark_config import BenchmarkConfig
from .imagenet_eeg_internal import create_EEG_dataset_benchmark
from .progress_util import tqdm
from .utils import setup_logger

log = setup_logger(__name__)
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = BENCHMARK_ROOT / "code"


def _ensure_code_on_path() -> None:
    if CODE_DIR.is_dir() and str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))


def get_imagenet_eeg_samples(
    config: BenchmarkConfig,
    split: str = "test",
    max_samples: Optional[int] = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Load ImageNet-EEG dataset and return list of unified sample dicts."""
    _ensure_code_on_path()

    eeg_path = config.imagenet_eeg_eeg_path or str(BENCHMARK_ROOT / "datasets" / "eeg_5_95_std.pth")
    splits_path = config.imagenet_eeg_splits_path or str(BENCHMARK_ROOT / "datasets" / "block_splits_by_image_single.pth")
    imagenet_path = config.imagenet_path
    if not imagenet_path:
        log.warning("imagenet_path not set; ImageNet-EEG dataset may fail.")
    identity = lambda x: x
    try:
        train_split, test_split = create_EEG_dataset_benchmark(
            eeg_signals_path=eeg_path,
            splits_path=splits_path,
            imagenet_path=imagenet_path,
            image_transform=identity,
            subject=4,
        )
    except ImportError as e:
        if "torch" in str(e).lower() or e.name == "torch":
            log.error(
                "ImageNet-EEG needs PyTorch to load .pth files (CPU wheel is enough): "
                "pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )
        else:
            log.error("create_EEG_dataset_benchmark failed: %s", e)
        return []
    except Exception as e:
        log.error("create_EEG_dataset_benchmark failed: %s", e)
        return []
    dataset = test_split if split == "test" else train_split
    n = len(dataset)
    if max_samples is not None and max_samples > 0:
        n = min(n, max_samples)
    samples = []
    idx_range = range(n)
    if show_progress and n > 0:
        idx_range = tqdm(
            idx_range,
            desc="Load ImageNet-EEG samples",
            total=n,
            unit="sample",
        )
    for i in idx_range:
        item = dataset[i]
        if isinstance(item, dict):
            eeg = item.get("eeg")
            label = item.get("label")
            image = item.get("image")
            image_raw = item.get("image_raw")
        else:
            eeg, label, image = item[0], item[1], item[2] if len(item) > 2 else None
        # GT image: we have tensor or path from dataset; for benchmark we need path or array for metrics
        gt_image_path = None
        gt_image = image  # may be tensor
        if isinstance(item, dict) and "image" in item:
            gt_image = item["image"]
        samples.append({
            "sample_id": "imagenet_eeg_%s_%04d" % (split, i),
            "eeg": eeg,
            "label": label,
            "gt_image": gt_image,
            "gt_image_path": gt_image_path,
            "metadata": {"split": split, "index": i},
            "split": split,
        })
    return samples


def get_thoughtviz_samples(
    config: BenchmarkConfig,
    split: str = "test",
    max_samples: Optional[int] = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Load ThoughtViz dataset and return list of unified sample dicts."""
    _ensure_code_on_path()
    try:
        from thoughtviz_integration.dataset_adapter import ThoughtVizDatasetAdapter
    except ImportError as e:
        log.error("ThoughtViz adapter not importable: %s", e)
        return []
    data_dir = config.thoughtviz_data_dir
    image_dir = config.thoughtviz_image_dir
    adapter = ThoughtVizDatasetAdapter(
        data_dir=data_dir,
        image_dir=image_dir,
        split=split,
        max_samples=max_samples or config.max_samples,
    )
    n_ad = len(adapter)
    idx_range = range(n_ad)
    if show_progress and n_ad > 0:
        idx_range = tqdm(idx_range, desc="Load ThoughtViz samples", total=n_ad, unit="sample")
    return [adapter[i] for i in idx_range]


def get_dataset(
    name: str,
    config: BenchmarkConfig,
    split: str = "test",
    max_samples: Optional[int] = None,
    show_progress: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Get dataset by name. Returns list of unified sample dicts.
    name: 'imagenet_eeg' | 'thoughtviz'
    """
    sp = config.show_progress if show_progress is None else show_progress
    if name == "imagenet_eeg":
        return get_imagenet_eeg_samples(config, split=split, max_samples=max_samples, show_progress=sp)
    if name == "thoughtviz":
        return get_thoughtviz_samples(config, split=split, max_samples=max_samples, show_progress=sp)
    log.warning("Unknown dataset: %s", name)
    return []
