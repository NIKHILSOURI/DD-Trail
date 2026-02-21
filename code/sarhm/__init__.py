# SAR-HM: Semantic Associative Retrieval with Hopfield Memory for EEG-conditioned image generation.
# Integration point: cond_stage_model.forward in dc_ldm/ldm_for_eeg.py

from .sarhm_modules import (
    SemanticProjection,
    HopfieldRetrieval,
    ConfidenceGatedFusion,
    ConditioningAdapter,
    pool_eeg_tokens,
)
from .prototypes import ClassPrototypes, build_prototypes_from_loader

__all__ = [
    "SemanticProjection",
    "HopfieldRetrieval",
    "ConfidenceGatedFusion",
    "ConditioningAdapter",
    "pool_eeg_tokens",
    "ClassPrototypes",
    "build_prototypes_from_loader",
]
