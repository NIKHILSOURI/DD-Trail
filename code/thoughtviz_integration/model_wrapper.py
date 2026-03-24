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


def _is_nchw_cpu_conv_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "conv2d" in msg and "cpu" in msg and "nchw" in msg and "nhwc" in msg


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
        self._cpu_only = True

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
            from tensorflow import config as tf_config
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
            try:
                self._cpu_only = len(tf_config.list_physical_devices("GPU")) == 0
            except Exception:
                self._cpu_only = True
            layer_index = getattr(self.config, "feature_layer_index", 9)
            self._get_encoding_fn = K.function(
                [self._classifier.layers[0].input],
                [self._classifier.layers[layer_index].output],
            )
            # Keras 3 removed set_learning_phase; keep legacy behavior when available.
            if hasattr(K, "set_learning_phase"):
                K.set_learning_phase(0)
            self._loaded = True
            log.info("ThoughtViz loaded: classifier %s, generator %s", eeg_path, gan_path)
            return True
        except Exception as e:
            log.exception("ThoughtViz load failed: %s", e)
            return False

    def _fallback_conditioning(self, eeg_batch: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Build deterministic conditioning vectors directly from EEG when classifier
        encoding fails on CPU due legacy NCHW Conv2D constraints.
        """
        enc_dim = int(self._generator.input_shape[1][-1])
        flat = eeg_batch.reshape(batch_size, -1).astype(np.float32)
        if flat.shape[1] == enc_dim:
            out = flat
        elif flat.shape[1] > enc_dim:
            idx = np.linspace(0, flat.shape[1] - 1, enc_dim).astype(np.int64)
            out = flat[:, idx]
        else:
            rep = int(np.ceil(float(enc_dim) / float(flat.shape[1])))
            out = np.tile(flat, (1, rep))[:, :enc_dim]
        mean = out.mean(axis=1, keepdims=True)
        std = out.std(axis=1, keepdims=True) + 1e-6
        return ((out - mean) / std).astype(np.float32)

    def _encode_eeg(self, eeg_batch: np.ndarray) -> np.ndarray:
        batch_size = eeg_batch.shape[0]
        try:
            enc = self._get_encoding_fn([eeg_batch])[0]
            if enc.shape[0] != batch_size:
                enc = np.repeat(enc[:1], batch_size, axis=0)
            return enc.astype(np.float32)
        except Exception as e:
            if self._cpu_only and _is_nchw_cpu_conv_error(e):
                log.warning(
                    "ThoughtViz classifier encoding hit CPU NCHW Conv2D limitation. "
                    "Using EEG-projection conditioning fallback to keep inference running."
                )
                return self._fallback_conditioning(eeg_batch, batch_size)
            raise

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
        enc = self._encode_eeg(eeg_batch)
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
