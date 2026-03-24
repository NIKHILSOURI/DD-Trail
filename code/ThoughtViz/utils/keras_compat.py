"""Keras 2 / 3 compatibility for legacy ThoughtViz (optimizers + learning phase)."""
from __future__ import annotations

import inspect


def adam_opt(lr: float, beta_1: float = 0.9, **kwargs):
    """Adam: Keras 3 uses `learning_rate`; Keras 2 often uses `lr`."""
    from keras.optimizers import Adam

    params = inspect.signature(Adam.__init__).parameters
    if "learning_rate" in params:
        return Adam(learning_rate=lr, beta_1=beta_1, **kwargs)
    return Adam(lr=lr, beta_1=beta_1, **kwargs)


def set_learning_phase_inference() -> None:
    """K.set_learning_phase(0) was removed in Keras 3; no-op there."""
    try:
        import warnings

        import keras.backend as K

        if hasattr(K, "set_learning_phase"):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*set_learning_phase.*",
                    category=UserWarning,
                )
                K.set_learning_phase(0)
    except Exception:
        pass
