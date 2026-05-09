"""
Informative Missingness Encoder — the primary novelty of IMST-Mamba.

3-state classification per feature per time step:
  State 0 — Never Measured: feature not observed at all in this ICU stay yet.
             Clinical signal: "not yet indicated / not in the differential"
  State 1 — Recently Measured: observed within learned recency threshold τ_f.
             Clinical signal: "actively monitored"
  State 2 — Stale: previously observed but not re-ordered beyond τ_f.
             Clinical signal: "clinician doesn't see urgency"

Key innovations vs GRU-D:
  1. Binary mask (observed/not) → 3-state semantic encoding
  2. Per-feature learned recency threshold τ_f
     (vitals default ~2h, labs ~12h — but jointly learned)
  3. Soft state assignment via temperature sigmoid (differentiable)
  4. Separate embedding table per feature × state
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.challenge_utils import N_FEATURES, get_recency_thresholds_array


class MissingnessEncoder(nn.Module):
    """
    Encode the informative missingness state of all features at each time step.

    Args:
        n_features:    number of features (F = 25)
        d_miss:        embedding dimension per feature (d_miss)
        init_tau_secs: initial recency thresholds in seconds (F,)
                       will be converted to hours internally
        temperature:   softness of state boundary (higher = harder)
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        d_miss: int = 32,
        init_tau_secs: torch.Tensor | None = None,
        temperature: float = 5.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_miss = d_miss
        self.temperature = temperature

        # Embedding table: shape (F, 3, d_miss)
        # E[f, state, :] = embedding for feature f in state {0, 1, 2}
        self.embeddings = nn.Embedding(n_features * 3, d_miss)
        nn.init.normal_(self.embeddings.weight, std=0.02)

        # Per-feature learnable recency threshold τ_f (in hours)
        # Initialized from clinical priors
        if init_tau_secs is None:
            init_tau_secs = torch.tensor(
                get_recency_thresholds_array(), dtype=torch.float32
            )
        # Convert to hours and take log for unconstrained parameterization
        init_tau_hours = init_tau_secs / 3600.0
        self.log_tau = nn.Parameter(torch.log(init_tau_hours + 1.0))

    def get_tau_hours(self) -> torch.Tensor:
        """Return current recency thresholds in hours (F,). Always positive."""
        return torch.exp(self.log_tau) - 1.0 + 1e-3  # always > 1e-3 hours

    def compute_soft_states(
        self,
        s_hours: torch.Tensor,   # (..., F) time since last obs in hours
        m: torch.Tensor,         # (..., F) current obs mask
    ) -> torch.Tensor:
        """
        Compute soft state probabilities p(state | s_f, m_f).

        State logic:
          - If s_f == NEVER_SEEN (very large): state 0
          - Elif s_f <= τ_f: state 1 (recently seen)
          - Else: state 2 (stale)

        Returns soft_probs: (..., F, 3) probabilities over 3 states.
        """
        tau = self.get_tau_hours()   # (F,)

        NEVER_THRESHOLD = 1e4   # hours (corresponds to NEVER_SEEN)

        # P(never seen): sigmoid of (s - NEVER_THRESHOLD), temperature-scaled
        p_never = torch.sigmoid(
            self.temperature * (s_hours - NEVER_THRESHOLD * 0.9)
        )   # (..., F)

        # P(recently seen | not never): sigmoid of (τ - s) / τ
        # → 1 when s << τ, 0 when s >> τ
        p_recent_given_seen = torch.sigmoid(
            self.temperature * (tau - s_hours) / (tau + 1e-6)
        )   # (..., F)

        # Combine
        p_seen = 1.0 - p_never
        p_recent = p_seen * p_recent_given_seen
        p_stale = p_seen * (1.0 - p_recent_given_seen)

        # Stack: (..., F, 3)
        soft_probs = torch.stack([p_never, p_recent, p_stale], dim=-1)

        return soft_probs

    def forward(
        self,
        s_hours: torch.Tensor,   # (B, T, F) or (T, F)
        m: torch.Tensor,         # (B, T, F) or (T, F)
    ) -> torch.Tensor:
        """
        Encode missingness states for all features.

        Args:
            s_hours: time since last observation per feature (hours)
            m:       observation mask

        Returns:
            miss_emb: (..., F * d_miss) missingness embeddings
                      (flattened per-feature embeddings, concatenated)
        """
        shape = s_hours.shape   # (..., F)
        F = self.n_features

        # Soft state probabilities: (..., F, 3)
        soft_probs = self.compute_soft_states(s_hours, m)

        # Look up embeddings for all feature-state combinations
        # embedding indices: feature f state k → f*3 + k
        indices = torch.arange(F * 3, device=s_hours.device)
        all_embs = self.embeddings(indices)   # (F*3, d_miss)
        all_embs = all_embs.view(F, 3, self.d_miss)   # (F, 3, d_miss)

        # Weighted sum over states using soft probabilities
        # soft_probs: (..., F, 3)  all_embs: (F, 3, d_miss)
        # → (..., F, d_miss)
        miss_emb = torch.einsum("...fk,fkd->...fd", soft_probs, all_embs)

        # Flatten last two dims: (..., F * d_miss)
        flat = miss_emb.flatten(start_dim=-2)

        return flat

    def get_hard_states(
        self,
        s_hours: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return hard (argmax) state assignments for interpretability.
        Returns: (..., F) LongTensor of {0, 1, 2}
        """
        soft_probs = self.compute_soft_states(s_hours, m)
        return soft_probs.argmax(dim=-1)
