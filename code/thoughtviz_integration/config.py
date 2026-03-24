"""
ThoughtViz integration config: paths and inference options.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .utils import resolve_thoughtviz_paths


@dataclass
class ThoughtVizConfig:
    """Config for ThoughtViz wrapper (paths and inference)."""

    data_dir: Optional[str] = None
    image_dir: Optional[str] = None
    eeg_model_path: Optional[str] = None
    gan_model_path: Optional[str] = None
    feature_layer_index: int = 9  # Keras layer index for EEG encoding (from test.py)
    input_noise_dim: int = 100
    num_classes: int = 10
    image_size: tuple = (64, 64)  # RGB generator output

    def get_paths(self) -> dict:
        """Resolve all paths; return dict from resolve_thoughtviz_paths."""
        return resolve_thoughtviz_paths(
            data_dir=self.data_dir,
            image_dir=self.image_dir,
            eeg_model_path=self.eeg_model_path,
            gan_model_path=self.gan_model_path,
        )
