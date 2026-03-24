"""
ImageNet-EEG loading for the unified benchmark without importing code/dataset.py.

code/dataset.py pulls torch (and transformers for CLIP) at import time, which breaks
minimal venvs (e.g. venv_thoughtviz with TensorFlow only). This module mirrors
EEGDataset + Splitter + create_EEG_dataset but:
  - no torch.* at module import time (import torch inside __init__ / __getitem__);
  - skips CLIP AutoProcessor / image_raw (benchmark only needs eeg, label, image).

You still need ``pip install torch`` (CPU wheel is enough) to torch.load the .pth files.
"""
from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError


def identity(x: Any) -> Any:
    return x


class EEGDatasetBenchmark:
    """Same logic as code.dataset.EEGDataset, without Dataset base class or CLIP processor."""

    def __init__(
        self,
        eeg_signals_path: str,
        imagenet_path: Optional[str],
        image_transform: Callable[[Any], Any] = identity,
        subject: int = 4,
    ):
        import torch

        if not imagenet_path or not str(imagenet_path).strip():
            raise RuntimeError(
                "EEGDatasetBenchmark requires imagenet_path for real GT images. "
                "Set --imagenet_path to the ImageNet root (e.g. ILSVRC2012)."
            )
        loaded = torch.load(eeg_signals_path, map_location="cpu", weights_only=False)
        if subject != 0:
            self.data = [loaded["dataset"][i] for i in range(len(loaded["dataset"])) if loaded["dataset"][i]["subject"] == subject]
        else:
            self.data = loaded["dataset"]
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.imagenet = imagenet_path
        self.image_transform = image_transform
        self.num_voxels = 440
        self.data_len = 512
        self.valid_indices = self._build_valid_indices()
        self.size = len(self.data)

    def _image_readable(self, idx: int) -> bool:
        image_name = self.images[self.data[idx]["image"]]
        image_path = os.path.join(self.imagenet, image_name.split("_")[0], image_name + ".JPEG")
        image_path = os.path.normpath(os.path.abspath(image_path))
        if not os.path.isfile(image_path):
            return False
        try:
            with Image.open(image_path) as im:
                im.load()
        except (UnidentifiedImageError, OSError):
            return False
        return True

    def _build_valid_indices(self) -> set:
        n = len(self.data)
        valid: List[int] = []
        for i in range(n):
            if self._image_readable(i):
                valid.append(i)
        n_bad = n - len(valid)
        if n_bad > 0:
            print("[dataset] Skipping %d indices with missing/corrupt images (valid=%d)." % (n_bad, len(valid)))
        return set(valid)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int):
        import torch
        from scipy.interpolate import interp1d

        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[20:460, :]
        eeg = np.array(eeg.transpose(0, 1))
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, self.data_len)
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()
        label = torch.tensor(self.data[i]["label"]).long()
        image_name = self.images[self.data[i]["image"]]
        image_path = os.path.join(self.imagenet, image_name.split("_")[0], image_name + ".JPEG")
        image_path = os.path.normpath(os.path.abspath(image_path))
        if i not in self.valid_indices:
            raise RuntimeError(
                "Index %d excluded: image missing or corrupt at %s (valid_indices built at init)."
                % (i, image_path)
            )
        if not os.path.isfile(image_path):
            raise RuntimeError("Image file missing: %s (imagenet_path=%s, image_name=%s)" % (image_path, self.imagenet, image_name))
        try:
            image_raw = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(
                "Image file is corrupt or unreadable: %s (remove or replace the file and re-run). Original: %s"
                % (image_path, e)
            ) from e
        image = np.array(image_raw, dtype=np.float32) / 255.0
        return {"eeg": eeg, "label": label, "image": self.image_transform(image), "image_raw": None}


class SplitterBenchmark:
    def __init__(self, dataset: EEGDatasetBenchmark, split_path: str, split_num: int = 0, split_name: str = "train", subject: int = 4):
        import torch

        self.dataset = dataset
        loaded = torch.load(split_path, map_location="cpu", weights_only=False)
        if "splits" not in loaded:
            raise KeyError("splits_path must contain 'splits' key. Keys found: %s" % list(loaded.keys()))
        splits = loaded["splits"]
        if split_num >= len(splits):
            raise IndexError("split_num=%d but splits has only %d split(s)." % (split_num, len(splits)))
        if split_name not in splits[split_num]:
            raise KeyError("split_name '%s' not in splits[%d]. Keys: %s" % (split_name, split_num, list(splits[split_num].keys())))
        self.split_idx = list(splits[split_num][split_name])
        max_idx = len(self.dataset.data) - 1
        filtered: List[int] = []
        for i in self.split_idx:
            if i < 0 or i > max_idx:
                continue
            try:
                eeg = self.dataset.data[i].get("eeg")
                if eeg is None:
                    continue
                L = eeg.size(1) if hasattr(eeg, "size") else eeg.shape[1]
                if 450 <= L <= 600:
                    filtered.append(i)
            except (KeyError, IndexError, TypeError):
                continue
        self.split_idx = filtered
        if hasattr(self.dataset, "valid_indices"):
            self.split_idx = [idx for idx in self.split_idx if idx in self.dataset.valid_indices]
        self.size = len(self.split_idx)
        self.num_voxels = 440
        self.data_len = 512

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int):
        return self.dataset[self.split_idx[i]]


def create_EEG_dataset_benchmark(
    eeg_signals_path: str = "../dreamdiffusion/datasets/eeg_5_95_std.pth",
    splits_path: str = "../dreamdiffusion/datasets/block_splits_by_image_single.pth",
    imagenet_path: Optional[str] = None,
    image_transform: Any = identity,
    subject: int = 0,
):
    if isinstance(image_transform, list):
        dataset_train = EEGDatasetBenchmark(eeg_signals_path, imagenet_path, image_transform[0], subject)
        dataset_test = EEGDatasetBenchmark(eeg_signals_path, imagenet_path, image_transform[1], subject)
    else:
        dataset_train = EEGDatasetBenchmark(eeg_signals_path, imagenet_path, image_transform, subject)
        dataset_test = EEGDatasetBenchmark(eeg_signals_path, imagenet_path, image_transform, subject)
    split_train = SplitterBenchmark(dataset_train, split_path=splits_path, split_num=0, split_name="train", subject=subject)
    split_test = SplitterBenchmark(dataset_test, split_path=splits_path, split_num=0, split_name="test", subject=subject)
    return (split_train, split_test)
