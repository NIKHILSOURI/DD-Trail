"""
SAR-HM++ default configuration constants.
Used by semantic_* modules; main config lives in config.Config_Generative_Model.
"""
from __future__ import annotations

# Semantic memory
SEMANTIC_DIM = 768
SEMANTIC_TOP_K = 5
SEMANTIC_TEMPERATURE = 1.0
SEMANTIC_MEMORY_MODE = "per_sample"  # "per_sample" | "clustered"
SEMANTIC_FUSION_MODE = "concat_project"  # "concat_project" | "weighted_avg" | "learned_mlp"

# Query
SEMANTIC_QUERY_POOLING = "mean"  # "mean" | "cls" | "attention"
SEMANTIC_QUERY_HIDDEN_DIM = 512

# Adapter
SEMANTIC_ADAPTER_MODE = "mlp_tokens_plus_transformer"  # repeat | linear_project_only | mlp_tokens | mlp_tokens_plus_transformer
SEMANTIC_TRANSFORMER_LAYERS = 1
SEMANTIC_SEQ_LEN = 77

# Confidence
CONF_W1 = 0.5  # max_attn
CONF_W2 = 0.3  # 1 - normalized_entropy
CONF_W3 = 0.2  # gap (top1 - top2 sim)

# Loss weights (defaults; overridden by config)
LAMBDA_DIFF = 1.0
LAMBDA_CLIP_IMG = 0.2
LAMBDA_CLIP_TEXT = 0.1
LAMBDA_SEM = 0.1
LAMBDA_RETR = 0.1
LAMBDA_SSIM = 0.05
LAMBDA_OBJ = 0.0
CLIP_LOSS_EVERY_N_STEPS = 1

# Flags
USE_REGION_SEMANTICS = False
USE_SUMMARY_SEMANTICS = True
USE_SCENE_SEMANTICS = True
USE_OBJECT_SEMANTICS = True

# Version for saved artifacts
SEMANTIC_TARGETS_VERSION = "sarhmpp_v1"
SEMANTIC_PROTOTYPES_VERSION = "sarhmpp_v1"
