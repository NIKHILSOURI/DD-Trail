"""
SAR-HM++: Semantic query branch.
Maps pooled EEG features to a semantic query vector q_sem in CLIP-compatible space (768).
Distinct from baseline dim_mapper path.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Learnable attention pooling over sequence dimension. Input [B, L, D] -> output [B, D]."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.q)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent [B, L, D]
        scores = torch.matmul(latent, self.q.transpose(-2, -1)) / math.sqrt(latent.shape[-1])
        weights = F.softmax(scores, dim=1)
        return (latent * weights).sum(dim=1)


def pool_eeg_for_query(
    latent: torch.Tensor,
    pooling: str = "mean",
    cls_index: int = 0,
    attention_pool: Optional[AttentionPool] = None,
) -> torch.Tensor:
    """
    Pool MAE token sequence to a single vector for semantic query.
    latent: [B, seq_len, embed_dim]
    pooling: "mean" | "cls" | "attention"
    attention_pool: used when pooling == "attention"
    Returns: [B, embed_dim]
    """
    if latent.dim() != 3:
        raise ValueError(f"pool_eeg_for_query expects [B, seq, dim], got {latent.shape}")
    if pooling == "mean":
        return latent.mean(dim=1)
    if pooling == "cls":
        return latent[:, cls_index, :]
    if pooling == "attention" and attention_pool is not None:
        return attention_pool(latent)
    return latent.mean(dim=1)


class SemanticQueryHead(nn.Module):
    """
    Maps pooled EEG features to semantic query q_sem in CLIP space.
    LayerNorm -> Linear -> GELU -> Dropout -> Linear -> L2 normalize.
    Input: [B, input_dim], Output: [B, output_dim] (default 768).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 768,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or min(input_dim * 2, 512)
        self.ln = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        pooled: [B, input_dim]
        Returns: [B, output_dim] L2-normalized.
        """
        x = self.ln(pooled)
        q = self.mlp(x)
        return F.normalize(q, dim=-1, eps=1e-12)
