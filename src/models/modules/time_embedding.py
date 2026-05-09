"""
Time embedding module: converts scalar elapsed-time values into dense vectors.

Two components:
  1. Sinusoidal basis at clinically meaningful frequencies (1h, 4h, 12h, 24h, 48h)
  2. Learnable linear projection on top

Reference: Time2Vec (Kazemi et al., 2019) — but simplified for scalar input.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """
    Embed a scalar elapsed time (in hours) into a dense vector.

    Args:
        d_out: output embedding dimension
        n_freqs: number of sinusoidal frequencies
    """

    def __init__(self, d_out: int = 64, n_freqs: int = 16):
        super().__init__()
        self.d_out = d_out
        self.n_freqs = n_freqs

        # Fixed frequencies: log-spaced from 1/48h to 1/0.25h
        # (captures patterns from 15 min to 2 days)
        freqs = torch.exp(
            torch.linspace(math.log(1 / 48.0), math.log(4.0), n_freqs)
        )
        self.register_buffer("freqs", freqs)

        # Learnable projection: 2*n_freqs (sin + cos) + 1 (linear) → d_out
        self.proj = nn.Linear(2 * n_freqs + 1, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (...) float tensor of elapsed times in hours

        Returns:
            (..., d_out) time embeddings
        """
        # t shape: (...), freqs shape: (n_freqs,)
        t_unsq = t.unsqueeze(-1)           # (..., 1)
        angles = 2 * math.pi * self.freqs * t_unsq   # (..., n_freqs)

        sin_emb = torch.sin(angles)        # (..., n_freqs)
        cos_emb = torch.cos(angles)        # (..., n_freqs)

        # Linear component
        linear = t_unsq                    # (..., 1)

        # Concatenate all components
        features = torch.cat([linear, sin_emb, cos_emb], dim=-1)  # (..., 2*n_freqs+1)

        out = self.proj(features)          # (..., d_out)
        return self.norm(out)
