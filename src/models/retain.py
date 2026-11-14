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

        # Reverse attention must be recomputed for each observed prefix.
        # Reversing the complete stay first leaks future visits into alpha/beta.
        logits = []
        for t in range(T):
            e_rev = e[:, :t+1].flip(1)
            g, _ = self.gru_alpha(e_rev)
            h, _ = self.gru_beta(e_rev)
            valid = attn_mask[:, :t+1].unsqueeze(-1)
            scores = self.attn_alpha(g.flip(1)).masked_fill(~valid, -1e9)
            alpha_t = torch.softmax(scores, dim=1) * valid
            alpha_t = alpha_t / alpha_t.sum(dim=1, keepdim=True).clamp_min(1e-12)
            beta_t = torch.tanh(self.attn_beta(h.flip(1)))
            ctx = (alpha_t * beta_t * x[:, :t+1]).sum(dim=1)
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
