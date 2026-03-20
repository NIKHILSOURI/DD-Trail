"""
Thin inference entry point for ThoughtViz (used by benchmark runner).
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from .config import ThoughtVizConfig
from .model_wrapper import ThoughtVizWrapper


def run_thoughtviz_inference(
    eeg_batch: np.ndarray | List[np.ndarray],
    config: Optional[ThoughtVizConfig] = None,
    num_samples: int = 1,
    **kwargs: Any,
) -> List[np.ndarray]:
    """Run ThoughtViz inference; return list of (H, W, 3) uint8 images."""
    wrapper = ThoughtVizWrapper(config=config)
    return wrapper.generate_from_eeg(eeg_batch, num_samples=num_samples, **kwargs)
