"""
Discriminator train/freeze helpers and small utilities for ThoughtViz GAN training.

The discriminator has two heads: real/fake (Dense) and auxiliary class (frozen Sequential classifier).
Only the convolutional trunk + fake head should receive gradients on D steps; the aux branch stays frozen.
For generator steps, the entire D must be frozen so gradients flow only into G.
"""
from __future__ import annotations

import json
import os
from typing import Any

import keras
import numpy as np


def _is_frozen_aux_classifier_layer(layer: keras.layers.Layer) -> bool:
    """Identify the pretrained image-classifier branch inside discriminator_model_rgb."""
    name = (getattr(layer, "name", "") or "").lower()
    if name.startswith("sequential"):
        return True
    if isinstance(layer, keras.Sequential):
        return True
    return False


def freeze_discriminator_for_generator_training(d: keras.Model) -> None:
    """No weights in D may update during d_on_g / generator step."""
    for layer in d.layers:
        layer.trainable = False


def unfreeze_discriminator_for_d_step(d: keras.Model) -> None:
    """Train conv trunk + fake head; keep auxiliary classifier frozen."""
    for layer in d.layers:
        layer.trainable = not _is_frozen_aux_classifier_layer(layer)


def count_trainable_params(model: keras.Model) -> int:
    return int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))


def onehot_labels_to_feature_vectors(
    labels_int: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    """
    Map integer class labels (N,) to (N, feature_dim) using a fixed matrix multiply:
    one-hot (N, K) @ projection (K, D) -> (N, D). Separates GAN mechanics from EEG conditioning.
    """
    n = labels_int.shape[0]
    k = projection.shape[0]
    oh = np.zeros((n, k), dtype=np.float32)
    oh[np.arange(n), labels_int] = 1.0
    return oh @ projection


def build_onehot_projection(num_classes: int, feature_dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    w = rng.randn(num_classes, feature_dim).astype(np.float32)
    # Normalize rows so scale is comparable to typical EEG feature vectors
    norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
    return (w / norms).astype(np.float32)


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def print_trainable_param_report(g: keras.Model, d: keras.Model, d_on_g: keras.Model) -> None:
    print("\n[trainable params] (not from summary(); counts actual trainable weights)", flush=True)
    print(f"  generator:                         {count_trainable_params(g):,}", flush=True)
    print(f"  discriminator (standalone D step): {count_trainable_params(d):,}", flush=True)
    print(f"  combined d_on_g (expect ~G only):    {count_trainable_params(d_on_g):,}", flush=True)
    print("", flush=True)
