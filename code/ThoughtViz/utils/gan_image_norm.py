"""
GAN image range helpers. Generator uses tanh → outputs in [-1, 1].
Real images from JPEG/PNG are mapped to the same range for discriminator training.
"""
from __future__ import annotations

import numpy as np


def rgb_unit_to_tanh(images: np.ndarray) -> np.ndarray:
    """Map [0, 1] float images to [-1, 1] (matches tanh generator output)."""
    return images.astype(np.float32) * 2.0 - 1.0


def tanh_to_rgb_unit(images: np.ndarray) -> np.ndarray:
    """Map [-1, 1] to [0, 1]."""
    return (np.asarray(images, dtype=np.float32) + 1.0) * 0.5


def tensor_to_image_uint8(
    x: np.ndarray,
    *,
    from_tanh: bool = True,
) -> np.ndarray:
    """
    Convert model output to uint8 RGB or grayscale image array.

    If from_tanh is True, values are assumed in [-1, 1] and mapped to [0, 255].
    If False, values are assumed already in [0, 1].

    Result is clipped to valid uint8 range; shape (H, W, C) with C=1 or 3.
    """
    x = np.asarray(x, dtype=np.float32)
    if from_tanh:
        x = tanh_to_rgb_unit(x)
    x = np.clip(x, 0.0, 1.0)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def batch_to_preview_uint8(
    batch: np.ndarray,
    *,
    from_tanh: bool = True,
) -> np.ndarray:
    """Vectorized (N, H, W, C) → uint8 batch for preview grids."""
    x = np.asarray(batch, dtype=np.float32)
    if from_tanh:
        x = tanh_to_rgb_unit(x)
    x = np.clip(x, 0.0, 1.0)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
