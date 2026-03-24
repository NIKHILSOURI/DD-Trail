"""
ThoughtViz dataset adapter: unified interface (sample_id, eeg, gt_image, label, metadata).
Uses ThoughtViz data.pkl and image folders.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from . import get_thoughtviz_root
from .utils import resolve_thoughtviz_paths

# ThoughtViz 10 image classes (from data_input_util.py)
IMAGE_CLASSES = {"Apple": 0, "Car": 1, "Dog": 2, "Gold": 3, "Mobile": 4, "Rose": 5, "Scooter": 6, "Tiger": 7, "Wallet": 8, "Watch": 9}
CLASS_NAMES = list(IMAGE_CLASSES.keys())


def load_thoughtviz_pkl(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data.pkl; return x_train, y_train, x_test, y_test (numpy). Handles bytes keys."""
    pkl_path = data_dir / "data.pkl"
    if not pkl_path.is_file():
        # Try eeg/image subdir
        pkl_path = data_dir / "eeg" / "image" / "data.pkl"
    if not pkl_path.is_file():
        raise FileNotFoundError("ThoughtViz data.pkl not found under %s" % data_dir)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    x_train = data.get(b"x_train", data.get("x_train"))
    y_train = data.get(b"y_train", data.get("y_train"))
    x_test = data.get(b"x_test", data.get("x_test"))
    y_test = data.get(b"y_test", data.get("y_test"))
    if x_test is None:
        raise KeyError("data.pkl must contain x_test (or b'x_test')")
    if y_test is None:
        y_test = np.zeros(len(x_test), dtype=np.int64)
    return x_train, y_train, x_test, y_test


def get_thoughtviz_sample(
    index: int,
    x_data: np.ndarray,
    y_data: np.ndarray,
    image_dir: Path,
    split: str = "test",
) -> Dict[str, Any]:
    """
    Build one sample in unified format.
    image_dir: root of class-named folders (Apple, Car, ...); we need a mapping from index to image path.
    ThoughtViz pkl does not give image paths; we have one EEG per test index. For GT image we either
    need a separate list of paths or we load by class. Here we return eeg and label; gt_image_path
    can be None if we don't have a direct index->path mapping (caller may resolve by class).
    """
    eeg = x_data[index]
    if hasattr(y_data[index], "argmax"):
        label = int(np.argmax(y_data[index]))
    else:
        label = int(y_data[index])
    class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else "Unknown"
    # ThoughtViz EEG pickle does not include a strict EEG->image identity mapping.
    # We expose a deterministic same-class reference image as "ground truth" proxy.
    gt_image_path = None
    class_dir = image_dir / class_name
    if class_dir.is_dir():
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.PNG"):
            files.extend(list(class_dir.glob(ext)))
        if files:
            files = sorted(files)
            gt_image_path = str(files[index % len(files)])
    return {
        "sample_id": "thoughtviz_%s_%04d" % (split, index),
        "eeg": eeg,
        "label": label,
        "class_name": class_name,
        "gt_image_path": gt_image_path,
        "metadata": {
            "split": split,
            "index": index,
            "has_true_gt": False,
            "gt_type": "class_reference",
        },
        "split": split,
    }


class ThoughtVizDatasetAdapter:
    """Unified dataset adapter for ThoughtViz: yields samples with sample_id, eeg, gt_image_path, label, metadata."""

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        image_dir: Optional[str | Path] = None,
        split: str = "test",
        max_samples: Optional[int] = None,
    ):
        paths = resolve_thoughtviz_paths(data_dir=str(data_dir) if data_dir else None, image_dir=str(image_dir) if image_dir else None)
        self.data_dir = paths["data_dir"]
        self.image_dir = paths["image_dir"]
        if self.data_dir is None or self.image_dir is None:
            raise RuntimeError("ThoughtViz paths not resolved (root not found or missing data/image dirs).")
        self.data_dir = Path(self.data_dir)
        self.image_dir = Path(self.image_dir)
        # Support passing training/images root; image classes are under ImageNet-Filtered.
        if (self.image_dir / "ImageNet-Filtered").is_dir():
            self.image_dir = self.image_dir / "ImageNet-Filtered"
        self.split = split
        self.max_samples = max_samples
        self._x_train, self._y_train, self._x_test, self._y_test = load_thoughtviz_pkl(self.data_dir)
        if split == "test":
            self._x, self._y = self._x_test, self._y_test
        else:
            self._x, self._y = self._x_train, self._y_train
        self._len = len(self._x)
        if self.max_samples is not None and self.max_samples > 0:
            self._len = min(self._len, self.max_samples)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index >= self._len:
            raise IndexError(index)
        return get_thoughtviz_sample(
            index,
            self._x,
            self._y,
            self.image_dir,
            split=self.split,
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]
