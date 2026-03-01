"""
SAR-HM modules: projection, Hopfield retrieval, confidence-gated fusion, conditioning adapter.
All logic is applied after MAE output; no modification to MAE or Stable Diffusion internals.
Safe fusion: c_final = c_base + alpha * (c_sar - c_base) with alpha from confidence.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from .prototypes import ClassPrototypes


def compute_alpha_from_attention(
    attn: torch.Tensor,
    alpha_mode: str = "entropy",
    alpha_max: float = 0.2,
    conf_threshold: float = 0.2,
    alpha_constant: float = 0.1,
    K: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute fusion alpha from Hopfield attention.
    alpha_mode: 'entropy' -> conf = 1 - H(attn)/log(K); 'max' -> conf = max(attn); 'constant' -> alpha_constant.
    Returns: (alpha [B], confidence [B], entropy [B]).
    """
    B = attn.shape[0]
    device = attn.device
    if alpha_mode == "constant":
        a = torch.full((B,), min(alpha_constant, alpha_max), device=device, dtype=attn.dtype if attn is not None else torch.float32)
        conf = a.clone()
        eps = 1e-12
        ent = -(attn * (attn + eps).log()).sum(dim=-1) if attn is not None else torch.zeros(B, device=device)
        return a, conf, ent
    if attn is None:
        a = torch.full((B,), alpha_constant, device=device, dtype=torch.float32)
        return a, a.clone(), torch.zeros(B, device=device)

    eps = 1e-12
    K_dim = attn.shape[-1] if K is None else K
    max_entropy = math.log(K_dim + eps)

    if alpha_mode == "max":
        conf = attn.max(dim=-1).values
    else:
        # entropy: conf = 1 - H(attn)/log(K)
        entropy = -(attn * (attn + eps).log()).sum(dim=-1)
        conf = 1.0 - (entropy / (max_entropy + eps)).clamp(0, 1)
    # Hard fallback: if conf < conf_threshold -> alpha = 0 (pure baseline)
    conf = conf.clamp(0, 1)
    alpha = (alpha_max * conf).clamp(0, alpha_max)
    alpha = alpha.clone()
    alpha[conf < conf_threshold] = 0.0
    entropy = -(attn * (attn + eps).log()).sum(dim=-1)
    return alpha, conf, entropy


def pool_eeg_tokens(latent: torch.Tensor, global_pool: bool = True) -> torch.Tensor:
    """
    Pool MAE token sequence to a single vector.
    latent: [B, seq_len, embed_dim]
    Returns: [B, embed_dim]
    """
    if latent.dim() != 3:
        raise ValueError(f"pool_eeg_tokens expects [B,seq,dim], got {latent.shape}")
    if global_pool:
        return latent.mean(dim=1)
    return latent[:, 0]


class SemanticProjection(nn.Module):
    """
    Maps pooled EEG embedding to CLIP-space (e.g. 768-dim to match FrozenImageEmbedder).
    Input: [B, fmri_latent_dim], Output: [B, clip_dim].
    """

    def __init__(self, fmri_latent_dim: int, clip_dim: int = 768):
        super().__init__()
        self.fmri_latent_dim = fmri_latent_dim
        self.clip_dim = clip_dim
        self.proj = nn.Sequential(
            nn.Linear(fmri_latent_dim, fmri_latent_dim),
            nn.GELU(),
            nn.Linear(fmri_latent_dim, clip_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() >= 2, f"SemanticProjection expects (B,...,D), got {x.shape}"
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.proj(x)


class HopfieldRetrieval(nn.Module):
    """
    Hopfield-style retrieval: query q, memory P (class prototypes).
    similarity = q @ P^T; attention = softmax(similarity / tau); r = attention @ P.
    Returns: (retrieved [B,dim], attention_weights [B,K], logits [B,K]).
    """

    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        query: torch.Tensor,
        memory: Union[ClassPrototypes, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        query: [B, dim], memory: ClassPrototypes or [K, dim]
        """
        if isinstance(memory, ClassPrototypes):
            P = memory.P
        else:
            P = memory
        if query.dim() == 3:
            query = query.mean(dim=1)
        logits = torch.matmul(query, P.t()) / max(self.tau, 1e-8)
        attn = F.softmax(logits, dim=-1)
        retrieved = torch.matmul(attn, P)
        return retrieved, attn, logits


class ConfidenceGatedFusion(nn.Module):
    """
    z_final = confidence * z_retrieved + (1 - confidence) * z_original
    confidence from max(attention) or entropy-based (normalized).
    gate_mode: 'max' | 'entropy'
    """

    def __init__(self, gate_mode: str = "max"):
        super().__init__()
        self.gate_mode = gate_mode

    def forward(
        self,
        z_original: torch.Tensor,
        z_retrieved: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_original, z_retrieved: [B, dim]
        attention_weights: [B, K]
        Returns: (z_fused [B,dim], confidence [B]).
        """
        if self.gate_mode == "max":
            confidence = attention_weights.max(dim=-1).values
        else:
            entropy = -(attention_weights * (attention_weights + 1e-12).log()).sum(dim=-1)
            max_entropy = math.log(attention_weights.shape[-1] + 1e-12)
            confidence = 1.0 - (entropy / (max_entropy + 1e-12)).clamp(0, 1)
        confidence = confidence.clamp(0, 1)
        if confidence.dim() == 1:
            confidence = confidence.unsqueeze(-1)
        z_fused = confidence * z_retrieved + (1 - confidence) * z_original
        return z_fused, confidence.squeeze(-1)


class ConditioningAdapter(nn.Module):
    """
    Maps semantic vector to Stable Diffusion conditioning shape [B, 77, 768].
    MLP: semantic [B, clip_dim] -> [B, 768], then repeat to [B, 77, 768].
    """

    def __init__(self, clip_dim: int = 768, cond_dim: int = 768, seq_len: int = 77):
        super().__init__()
        self.clip_dim = clip_dim
        self.cond_dim = cond_dim
        self.seq_len = seq_len
        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, cond_dim * 2),
            nn.GELU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_near_zero_delta(self, scale: float = 0.01) -> None:
        """Initialize so output is near zero (small residual). c_sar ≈ 0 => c_final ≈ c_base at epoch 0."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, clip_dim] -> out: [B, 77, 768]
        """
        if z.dim() == 3:
            z = z.mean(dim=1)
        out = self.mlp(z)
        out = out.unsqueeze(1).repeat(1, self.seq_len, 1)
        assert out.shape == (z.shape[0], self.seq_len, self.cond_dim), (
            f"ConditioningAdapter output shape {out.shape} != (B,{self.seq_len},{self.cond_dim})"
        )
        return out
