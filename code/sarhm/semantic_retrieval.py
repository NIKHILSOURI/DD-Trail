"""
SAR-HM++: Top-k semantic retrieval over prototype memory.
Cosine similarity, top-k softmax, confidence from max + entropy + gap.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


def topk_retrieval(
    query: torch.Tensor,
    keys: torch.Tensor,
    top_k: int = 5,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    query: [B, D], keys: [N, D]
    Returns: (retrieved [B, D], attention_weights [B, N], top_k_indices [B, top_k])
    """
    B, D = query.shape
    N = keys.shape[0]
    query = F.normalize(query, dim=-1, eps=1e-12)
    keys = F.normalize(keys, dim=-1, eps=1e-12)
    logits = torch.matmul(query, keys.t()) / max(temperature, 1e-8)
    topk_scores, topk_idx = torch.topk(logits, min(top_k, N), dim=-1)
    # Softmax over full N (or over top_k only for efficiency)
    attn = F.softmax(logits, dim=-1)
    retrieved = torch.matmul(attn, keys)
    return retrieved, attn, topk_idx


def confidence_from_attention(
    attn: torch.Tensor,
    w1: float = 0.5,
    w2: float = 0.3,
    w3: float = 0.2,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    conf = w1*max_attn + w2*(1 - normalized_entropy) + w3*gap
    attn: [B, K]
    Returns: [B]
    """
    B, K = attn.shape
    max_attn = attn.max(dim=-1).values
    entropy = -(attn * (attn + eps).log()).sum(dim=-1)
    max_entropy = math.log(K + eps)
    norm_entropy = (entropy / (max_entropy + eps)).clamp(0, 1)
    inv_entropy = 1.0 - norm_entropy
    sorted_attn, _ = torch.sort(attn, dim=-1, descending=True)
    gap = (sorted_attn[:, 0] - sorted_attn[:, 1]) if K >= 2 else sorted_attn[:, 0]
    conf = w1 * max_attn + w2 * inv_entropy + w3 * gap
    return conf.clamp(0, 1)


class SemanticRetrieval(nn.Module):
    """
    Top-k retrieval over semantic memory bank.
    Returns retrieved m_sem, attention weights, confidence, and optional diagnostics.
    """

    def __init__(
        self,
        top_k: int = 5,
        temperature: float = 1.0,
        conf_w1: float = 0.5,
        conf_w2: float = 0.3,
        conf_w3: float = 0.2,
    ):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.conf_w1 = conf_w1
        self.conf_w2 = conf_w2
        self.conf_w3 = conf_w3

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        query: [B, D], keys: [N, D]
        Returns: (m_sem [B,D], attn [B,N], confidence [B], top_k_indices [B, top_k])
        """
        if query.dim() == 3:
            query = query.mean(dim=1)
        m_sem, attn, topk_idx = topk_retrieval(
            query, keys, top_k=self.top_k, temperature=self.temperature
        )
        conf = confidence_from_attention(
            attn, w1=self.conf_w1, w2=self.conf_w2, w3=self.conf_w3
        )
        return m_sem, attn, conf, topk_idx
