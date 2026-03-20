"""
ThoughtViz model wrapper: load pretrained generator + EEG classifier, run inference from EEG.
Outputs numpy arrays (H, W, C) in [0, 255] for compatibility with benchmark metrics.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .config import ThoughtVizConfig
from .utils import check_thoughtviz_available, resolve_thoughtviz_paths

log = logging.getLogger("thoughtviz_wrapper")


class ThoughtVizWrapper:
    """
    Wrapper for ThoughtViz inference: EEG -> generator -> image.
    Requires Keras and the ThoughtViz repo (code/ThoughtViz). Optional dependency for benchmark.
    """

    def __init__(self, config: Optional[ThoughtVizConfig] = None):
        self.config = config or ThoughtVizConfig()
        self._paths = self.config.get_paths()
        self._classifier = None
        self._generator = None
        self._get_encoding_fn = None
        self._loaded = False

    def load_pretrained(self) -> bool:
        """Load EEG classifier and GAN generator. Returns True on success."""
        if not check_thoughtviz_available():
            log.warning("ThoughtViz root not available; cannot load models.")
            return False
        root = self._paths.get("root")
        eeg_path = self._paths.get("eeg_model_path")
        gan_path = self._paths.get("gan_model_path")
        if not eeg_path or not eeg_path.exists():
            log.warning("ThoughtViz EEG model not found: %s", eeg_path)
            return False
        if not gan_path or not gan_path.exists():
            log.warning("ThoughtViz GAN model not found: %s", gan_path)
            return False
        try:
            import keras.backend as K
            from keras.models import load_model
        except ImportError as e:
            log.warning("Keras not available; ThoughtViz wrapper cannot load models: %s", e)
            return False
        try:
            # Add ThoughtViz to path for custom layer
            if root:
                import sys
                sys.path.insert(0, str(root))
            from layers.mog_layer import MoGLayer
            custom = {"MoGLayer": MoGLayer}
            self._classifier = load_model(str(eeg_path), custom_objects=custom)
            self._generator = load_model(str(gan_path), custom_objects=custom)
            layer_index = getattr(self.config, "feature_layer_index", 9)
            self._get_encoding_fn = K.function(
                [self._classifier.layers[0].input],
                [self._classifier.layers[layer_index].output],
            )
            K.set_learning_phase(0)
            self._loaded = True
            log.info("ThoughtViz loaded: classifier %s, generator %s", eeg_path, gan_path)
            return True
        except Exception as e:
            log.exception("ThoughtViz load failed: %s", e)
            return False

    def generate_from_eeg(
        self,
        eeg_batch: Union[np.ndarray, List[np.ndarray]],
        num_samples: int = 1,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """
        Generate images from EEG. eeg_batch: (B, ...) or list of (D,). Returns list of (H, W, 3) uint8 [0,255].
        """
        if not self._loaded:
            if not self.load_pretrained():
                raise RuntimeError("ThoughtViz not loaded; cannot generate.")
        if isinstance(eeg_batch, list):
            eeg_batch = np.array(eeg_batch)
        if eeg_batch.ndim == 2:
            eeg_batch = eeg_batch[np.newaxis, ...]
        B = eeg_batch.shape[0]
        noise_dim = getattr(self.config, "input_noise_dim", 100)
        noise = np.random.uniform(-1, 1, (B, noise_dim)).astype(np.float32)
        enc = self._get_encoding_fn([eeg_batch])[0]
        if enc.shape[0] != B:
            enc = np.repeat(enc[:1], B, axis=0)
        out = self._generator.predict([noise, enc], verbose=0)
        # out: (B, H, W, 3) in [-1, 1]
        out = (out * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        return [out[i] for i in range(B)]

    def save_outputs(
        self,
        outputs: List[np.ndarray],
        out_dir: str | Path,
        sample_ids: Optional[List[str]] = None,
        prefix: str = "thoughtviz",
    ) -> None:
        """Save generated images to out_dir; filenames: {prefix}_{sample_id}.png or {prefix}_{i}.png."""
        from PIL import Image
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(outputs):
            if img.shape[-1] != 3:
                img = np.transpose(img, (1, 2, 0)) if img.ndim == 3 else img
            sid = sample_ids[i] if sample_ids and i < len(sample_ids) else ("%04d" % i)
            path = out_dir / ("%s_%s.png" % (prefix, sid))
            Image.fromarray(img.astype(np.uint8)).save(path)
        log.info("Saved %d ThoughtViz outputs to %s", len(outputs), out_dir)
