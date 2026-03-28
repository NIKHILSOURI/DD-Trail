# SAR-HM: Semantic Associative Retrieval with Hopfield Memory for EEG-conditioned image generation.
# SAR-HM++: Multi-Level Semantic Prototype Retrieval (semantic_query, semantic_memory, semantic_retrieval, semantic_adapter).
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

# Optional SAR-HM++ modules (may fail if timm not available for SemanticAdapter)
try:
    from .semantic_query import SemanticQueryHead, pool_eeg_for_query, AttentionPool
    from .semantic_retrieval import SemanticRetrieval, confidence_from_attention
    from .semantic_adapter import SemanticAdapter as SemanticAdapterPP
    from .semantic_memory import SemanticMemoryBank, build_fused_keys, fuse_semantic_embeddings
    from .semantic_losses import compute_semantic_losses, semantic_alignment_loss, retrieval_consistency_loss
    __all__ += [
        "SemanticQueryHead", "pool_eeg_for_query", "AttentionPool",
        "SemanticRetrieval", "confidence_from_attention",
        "SemanticAdapterPP", "SemanticMemoryBank", "build_fused_keys", "fuse_semantic_embeddings",
        "compute_semantic_losses", "semantic_alignment_loss", "retrieval_consistency_loss",
    ]
except Exception:
    pass
