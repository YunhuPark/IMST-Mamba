"""
RETAIN: Reverse Time Attention Model for Interpretable Prediction of Clinical Events.
Choi et al. (2016), NeurIPS.

Two-level attention:
  - Alpha: visit-level attention (importance of each time step)
  - Beta:  variable-level attention (importance of each feature at each step)

Adapted from visit-level to observation-event-level for irregular time series.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.challenge_utils import N_FEATURES


class RETAIN(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features

        # Input embedding
        self.embed = nn.Linear(n_features, hidden_size)

        # GRU-alpha: generates visit attention weights
        self.gru_alpha = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.attn_alpha = nn.Linear(hidden_size, 1)

        # GRU-beta: generates variable attention weights
        self.gru_beta = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.attn_beta = nn.Linear(hidden_size, n_features)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_features // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,           # (B, T, F)
        m: torch.Tensor,           # (B, T, F)
        delta_t: torch.Tensor,     # ignored
        s: torch.Tensor,           # ignored
        attn_mask: torch.Tensor,   # (B, T)
    ) -> dict[str, torch.Tensor]:
        B, T, F = x.shape

        # Embed input (with mask)
        e = self.embed(x * m)              # (B, T, hidden)
        e = self.dropout(e)

        # Reverse sequences for reverse-time attention
        # Only reverse up to valid length
        e_rev = torch.flip(e, dims=[1])

        # Alpha attention (visit importance)
        g, _ = self.gru_alpha(e_rev)       # (B, T, hidden)
        g = torch.flip(g, dims=[1])
        alpha = self.attn_alpha(g)         # (B, T, 1)
        alpha = alpha.masked_fill(~attn_mask.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(alpha, dim=1)  # (B, T, 1)

        # Beta attention (variable importance)
        h_rev, _ = self.gru_beta(e_rev)   # (B, T, hidden)
        h_rev = torch.flip(h_rev, dims=[1])
        beta = torch.tanh(self.attn_beta(h_rev))  # (B, T, F)

        # Context vector per step (cumulative)
        # For step-wise prediction, compute context up to each step
        logits = []
        for t in range(T):
            # Context: attention-weighted sum of past steps
            # alpha: (B, T, 1), x: (B, T, F), beta: (B, T, F)
            alpha_t = alpha[:, :t+1]       # (B, t+1, 1)
            beta_t = beta[:, :t+1]         # (B, t+1, F)
            x_t = x[:, :t+1]              # (B, t+1, F)
            ctx = (alpha_t * beta_t * x_t).sum(dim=1)   # (B, F)
            logits.append(self.head(ctx))   # (B, 1)

        logit_sepsis = torch.stack(logits, dim=1)   # (B, T, 1)
        logit_sepsis = logit_sepsis * attn_mask.unsqueeze(-1).float()
        return {"logit_sepsis": logit_sepsis}


def build_model(cfg: dict, **kwargs) -> RETAIN:
    m = cfg.get("model", {})
    return RETAIN(
        n_features=m.get("n_features", N_FEATURES),
        hidden_size=m.get("hidden_size", 256),
        dropout=m.get("dropout", 0.1),
    )
