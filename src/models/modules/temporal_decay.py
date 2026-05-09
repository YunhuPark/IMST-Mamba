"""
Per-feature learnable temporal decay for missing value imputation.

GRU-D style decay, but with per-feature learned decay rates γ_f.

Decay factor:  d_f(t) = exp(-softplus(γ_f) · Δt_f)
Imputed value: x̃_f(t) = d_f · x_f(last) + (1 - d_f) · μ_f

where:
  γ_f      = per-feature learnable log-decay (initialized from observation frequency)
  Δt_f     = hours since last observation of feature f
  x_f(last) = last observed value of feature f
  μ_f      = population mean of feature f (from training stats, frozen)

Reference: Che et al. (2018) Recurrent Neural Networks for Multivariate Time Series
           with Missing Values. Scientific Reports.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDecayLayer(nn.Module):
    """
    Per-feature exponential decay for missing value imputation.

    Args:
        n_features: number of input features (F)
        init_decay_hours: initial expected decay time in hours per feature.
                          γ_f is initialized so that d_f → 0.5 at init_decay_hours.
    """

    def __init__(
        self,
        n_features: int,
        init_decay_hours: float | list[float] = 6.0,
    ):
        super().__init__()
        self.n_features = n_features

        if isinstance(init_decay_hours, (int, float)):
            init_hours = [float(init_decay_hours)] * n_features
        else:
            assert len(init_decay_hours) == n_features
            init_hours = [float(h) for h in init_decay_hours]

        # γ_f initialized so that softplus(γ_f) ≈ log(2) / init_decay_hours_f
        # → d_f(init_hours) ≈ 0.5
        init_gamma = [
            torch.log(torch.expm1(torch.tensor(0.693147 / max(h, 0.1))))
            for h in init_hours
        ]
        self.log_gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))

    def get_decay_rates(self) -> torch.Tensor:
        """Return positive decay rates (F,)."""
        return F.softplus(self.log_gamma)

    def forward(
        self,
        s_hours: torch.Tensor,   # (..., F) time since last obs in hours
        x_last: torch.Tensor,    # (..., F) last observed value
        x_mean: torch.Tensor,    # (F,)     population mean
        m: torch.Tensor,         # (..., F) observation mask (1=observed)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute decay-imputed feature values.

        Args:
            s_hours: time since last observation per feature (hours)
            x_last:  last seen value (running forward-fill)
            x_mean:  population mean (training set)
            m:       current observation mask

        Returns:
            x_imputed: (..., F) imputed values
            decay_factor: (..., F) for analysis / loss weighting
        """
        gamma = self.get_decay_rates()   # (F,)

        # Clamp s to avoid inf/nan
        s_clamped = torch.clamp(s_hours, min=0.0, max=1e4)

        # d_f = exp(-γ_f · s_f)
        decay = torch.exp(-gamma * s_clamped)    # (..., F)

        # Imputed value: interpolate between last value and population mean
        x_imputed = decay * x_last + (1.0 - decay) * x_mean

        # Where feature is currently observed, use real value
        x_final = torch.where(m > 0.5, x_last, x_imputed)

        return x_final, decay
