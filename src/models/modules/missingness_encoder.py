"""Informative missingness encoder with an explicit per-feature contract."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.challenge_utils import (
    FEATURE_NAMES,
    N_FEATURES,
    RECENCY_THRESHOLD_INIT,
)


class MissingnessEncoder(nn.Module):
    """Encode never/recent/stale missingness states for each input feature."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        d_miss: int = 32,
        init_tau_secs: torch.Tensor | None = None,
        temperature: float = 5.0,
        feature_names: list[str] | None = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_miss = d_miss
        self.temperature = temperature

        if feature_names is None:
            feature_names = FEATURE_NAMES[:n_features]
        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must equal n_features ({n_features})"
            )
        unknown = [name for name in feature_names if name not in RECENCY_THRESHOLD_INIT]
        if unknown:
            raise ValueError(f"unknown feature names: {unknown}")
        self.feature_names = list(feature_names)

        self.embeddings = nn.Embedding(n_features * 3, d_miss)
        nn.init.normal_(self.embeddings.weight, std=0.02)

        if init_tau_secs is None:
            init_tau_secs = torch.tensor(
                [RECENCY_THRESHOLD_INIT[name] for name in self.feature_names],
                dtype=torch.float32,
            )
        else:
            init_tau_secs = torch.as_tensor(init_tau_secs, dtype=torch.float32)
        if init_tau_secs.numel() != n_features:
            raise ValueError(
                f"init_tau_secs length ({init_tau_secs.numel()}) must equal n_features ({n_features})"
            )

        init_tau_hours = init_tau_secs / 3600.0
        self.log_tau = nn.Parameter(torch.log(init_tau_hours + 1.0))

    def get_tau_hours(self) -> torch.Tensor:
        return torch.exp(self.log_tau) - 1.0 + 1e-3

    def compute_soft_states(
        self,
        s_hours: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        if s_hours.shape[-1] != self.n_features or m.shape[-1] != self.n_features:
            raise ValueError("missingness inputs do not match configured feature count")

        tau = self.get_tau_hours()
        never_threshold = 1e4
        p_never = torch.sigmoid(
            self.temperature * (s_hours - never_threshold * 0.9)
        )
        p_recent_given_seen = torch.sigmoid(
            self.temperature * (tau - s_hours) / (tau + 1e-6)
        )
        p_seen = 1.0 - p_never
        p_recent = p_seen * p_recent_given_seen
        p_stale = p_seen * (1.0 - p_recent_given_seen)
        return torch.stack([p_never, p_recent, p_stale], dim=-1)

    def forward(self, s_hours: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        soft_probs = self.compute_soft_states(s_hours, m)
        indices = torch.arange(self.n_features * 3, device=s_hours.device)
        all_embs = self.embeddings(indices).view(self.n_features, 3, self.d_miss)
        miss_emb = torch.einsum("...fk,fkd->...fd", soft_probs, all_embs)
        return miss_emb.flatten(start_dim=-2)

    def get_hard_states(self, s_hours: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.compute_soft_states(s_hours, m).argmax(dim=-1)
