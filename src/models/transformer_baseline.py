"""
Transformer baseline with time2vec positional encoding.
Handles irregular time by encoding actual timestamps (not position indices).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.challenge_utils import N_FEATURES


class Time2Vec(nn.Module):
    """Time2Vec: Learning a Vector Representation of Time (Kazemi et al., 2019)."""

    def __init__(self, d_out: int):
        super().__init__()
        self.d_out = d_out
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(d_out - 1))
        self.b = nn.Parameter(torch.zeros(d_out - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (...) → (..., d_out)"""
        t = t.unsqueeze(-1)
        linear = self.w0 * t + self.b0               # (..., 1)
        periodic = torch.sin(self.w * t + self.b)    # (..., d_out-1)
        return torch.cat([linear, periodic], dim=-1)


class TransformerBaseline(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        d_time: int = 64,
    ):
        super().__init__()
        self.time_emb = Time2Vec(d_time)
        self.input_proj = nn.Linear(n_features * 2 + d_time, d_model)  # x + mask + t

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        delta_t: torch.Tensor,
        s: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        t_emb = self.time_emb(delta_t)                       # (B, T, d_time)
        inp = torch.cat([x, m, t_emb], dim=-1)               # (B, T, F*2+d_time)
        h = self.input_proj(inp)                              # (B, T, d_model)

        # src_key_padding_mask: True for positions to IGNORE
        key_pad = ~attn_mask
        h = self.encoder(h, src_key_padding_mask=key_pad)    # (B, T, d_model)

        logit = self.head(h)
        logit = logit * attn_mask.unsqueeze(-1).float()
        return {"logit_sepsis": logit}


def build_model(cfg: dict, **kwargs) -> TransformerBaseline:
    m = cfg.get("model", {})
    return TransformerBaseline(
        n_features=m.get("n_features", N_FEATURES),
        d_model=m.get("d_model", 256),
        nhead=m.get("nhead", 8),
        num_encoder_layers=m.get("num_encoder_layers", 4),
        dim_feedforward=m.get("dim_feedforward", 512),
        dropout=m.get("dropout", 0.1),
    )
