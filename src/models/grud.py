"""
GRU-D: Recurrent Neural Networks for Multivariate Time Series with Missing Values.
Che et al. (2018), Scientific Reports.

Key mechanisms:
  1. Exponential decay imputation: x̃_f(t) = exp(-γ_f · Δ_f) · x_f(last) + (1-...) · μ_f
  2. Binary observation mask appended to input
  3. Decay factor fed into GRU cell as part of the hidden state reset

This is the primary baseline to beat with IMST-Mamba.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.challenge_utils import N_FEATURES, get_recency_thresholds_array


class GRUDCell(nn.Module):
    """One step of GRU-D."""

    def __init__(self, input_size: int, hidden_size: int, n_features: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_features = n_features

        # GRU-D: gate equations also take decay factors as input
        # Input to gates: [x_imputed (input_size), m (n_features), decay (n_features), h_prev (hidden_size)]
        gate_input = input_size + 2 * n_features + hidden_size

        self.W_r = nn.Linear(gate_input, hidden_size, bias=True)
        self.W_z = nn.Linear(gate_input, hidden_size, bias=True)
        self.W_h = nn.Linear(gate_input, hidden_size, bias=True)

    def forward(
        self,
        x_imputed: torch.Tensor,   # (B, F)
        m: torch.Tensor,           # (B, F)
        decay: torch.Tensor,       # (B, F)
        h: torch.Tensor,           # (B, hidden)
    ) -> torch.Tensor:
        inp = torch.cat([x_imputed, m, decay, h], dim=-1)
        r = torch.sigmoid(self.W_r(inp))
        z = torch.sigmoid(self.W_z(inp))
        h_candidate = torch.tanh(self.W_h(torch.cat([x_imputed, m, decay, r * h], dim=-1)))
        h_new = (1 - z) * h + z * h_candidate
        return h_new


class GRUD(nn.Module):
    """
    GRU-D model for sepsis prediction.

    Processes irregular time series: one step per observation event.
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_size: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Per-feature decay rates (log parameterized)
        tau_hours = torch.tensor(get_recency_thresholds_array() / 3600.0)
        log_gamma_init = torch.log(torch.tensor(0.693147) / tau_hours.clamp(min=0.1))
        self.log_gamma = nn.Parameter(log_gamma_init)

        # Population mean (updated from stats)
        self.register_buffer("x_mean", torch.zeros(n_features))

        # GRU-D cells (stacked)
        self.cells = nn.ModuleList()
        for i in range(n_layers):
            in_size = n_features if i == 0 else hidden_size
            self.cells.append(GRUDCell(in_size, hidden_size, n_features))

        self.dropout = nn.Dropout(dropout)

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,           # (B, T, F)
        m: torch.Tensor,           # (B, T, F)
        delta_t: torch.Tensor,     # (B, T)    hours
        s: torch.Tensor,           # (B, T, F) log1p hours since last obs
        attn_mask: torch.Tensor,   # (B, T)
    ) -> dict[str, torch.Tensor]:
        B, T, n_feat = x.shape
        device = x.device

        gamma = F.softplus(self.log_gamma)   # (n_feat,)
        x_mean = self.x_mean.to(device)

        # Initialize hidden states
        hs = [torch.zeros(B, self.hidden_size, device=device) for _ in self.cells]

        # Running last value
        x_last = torch.zeros(B, n_feat, device=device)

        logits = []
        for t in range(T):
            s_hours_t = torch.expm1(s[:, t])       # (B, F)

            # Decay imputation
            s_clamped = s_hours_t.clamp(0, 1e4)
            decay_t = torch.exp(-gamma * s_clamped)  # (B, F)
            x_imputed = decay_t * x_last + (1 - decay_t) * x_mean
            # Where observed, use real value
            x_t = torch.where(m[:, t] > 0.5, x[:, t], x_imputed)

            # Update x_last with currently observed values
            x_last = torch.where(m[:, t] > 0.5, x[:, t], x_last)

            # Stacked GRU-D cells
            h_in = x_t
            for i, cell in enumerate(self.cells):
                h = cell(h_in, m[:, t], decay_t, hs[i])
                if i < len(self.cells) - 1:
                    h_in = self.dropout(h)
                else:
                    h_in = h
                hs[i] = h

            logits.append(self.head(hs[-1]))   # (B, 1)

        logit_sepsis = torch.stack(logits, dim=1)   # (B, T, 1)

        # Mask padding
        logit_sepsis = logit_sepsis * attn_mask.unsqueeze(-1).float()

        return {"logit_sepsis": logit_sepsis}


def build_model(cfg: dict, **kwargs) -> GRUD:
    m = cfg.get("model", {})
    return GRUD(
        n_features=m.get("n_features", N_FEATURES),
        hidden_size=m.get("hidden_size", 256),
        n_layers=m.get("n_layers", 2),
        dropout=m.get("dropout", 0.1),
    )
