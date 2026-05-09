"""
LSTM baseline with hourly binning + forward-fill imputation.
Optional: append binary missingness mask to input (Lipton et al., 2016).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.challenge_utils import N_FEATURES


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_size: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_mask: bool = True,
    ):
        super().__init__()
        self.use_mask = use_mask
        input_size = n_features * 2 if use_mask else n_features

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
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
        delta_t: torch.Tensor,     # ignored for standard LSTM
        s: torch.Tensor,           # ignored
        attn_mask: torch.Tensor,   # (B, T)
    ) -> dict[str, torch.Tensor]:
        if self.use_mask:
            inp = torch.cat([x, m], dim=-1)
        else:
            inp = x

        out, _ = self.lstm(inp)          # (B, T, hidden)
        out = self.dropout(out)
        logit = self.head(out)           # (B, T, 1)
        logit = logit * attn_mask.unsqueeze(-1).float()
        return {"logit_sepsis": logit}


def build_model(cfg: dict, **kwargs) -> LSTMBaseline:
    m = cfg.get("model", {})
    return LSTMBaseline(
        n_features=m.get("n_features", N_FEATURES),
        hidden_size=m.get("hidden_size", 256),
        n_layers=m.get("n_layers", 2),
        dropout=m.get("dropout", 0.1),
        use_mask=m.get("use_mask", True),
    )
