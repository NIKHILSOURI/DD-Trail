"""
SAR-HM++: Convert retrieved semantic vector(s) into SD conditioning tokens [B, 77, 768].
MLP -> reshape; optional lightweight transformer refinement.
Uses only PyTorch (no timm) for transformer blocks.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAdapter(nn.Module):
    """
    Map retrieved m_sem [B, 768] to conditioning tokens [B, seq_len, cond_dim] for SD cross-attention.
    Modes: repeat | linear_project_only | mlp_tokens | mlp_tokens_plus_transformer.
    """

    def __init__(
        self,
        input_dim: int = 768,
        cond_dim: int = 768,
        seq_len: int = 77,
        mode: str = "mlp_tokens_plus_transformer",
        num_transformer_layers: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.seq_len = seq_len
        self.mode = mode
        self.num_transformer_layers = num_transformer_layers

        if mode == "repeat":
            self.proj = nn.Linear(input_dim, cond_dim)
        elif mode == "linear_project_only":
            self.proj = nn.Linear(input_dim, seq_len * cond_dim)
        else:
            hidden = (seq_len * cond_dim) * 2
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, seq_len * cond_dim),
            )

        self.ln = nn.LayerNorm(cond_dim)
        self.refine_blocks: Optional[nn.ModuleList] = None
        if mode == "mlp_tokens_plus_transformer" and num_transformer_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=cond_dim,
                nhead=8,
                dim_feedforward=cond_dim * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.refine_blocks = nn.ModuleList([layer for _ in range(num_transformer_layers)])
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_near_zero_delta(self, scale: float = 0.01) -> None:
        """Initialize so output is near zero (small residual)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, m_sem: torch.Tensor) -> torch.Tensor:
        """
        m_sem: [B, input_dim] (or [B, 1, input_dim]; will be mean over dim=1).
        Returns: [B, seq_len, cond_dim].
        """
        if m_sem.dim() == 3:
            m_sem = m_sem.mean(dim=1)
        B = m_sem.shape[0]
        if self.mode == "repeat":
            out = self.proj(m_sem)
            out = out.unsqueeze(1).repeat(1, self.seq_len, 1)
        else:
            out = self.proj(m_sem)
            out = out.view(B, self.seq_len, self.cond_dim)
        out = self.ln(out)
        if self.refine_blocks is not None:
            for blk in self.refine_blocks:
                out = blk(out) + out
            out = self.ln(out)
        return out
