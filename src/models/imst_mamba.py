"""
IMST-Mamba: Informative-Missingness State-Space model for sepsis prediction.

Architecture (per observation event):
  1. Feature Encoding:   observed values + decay imputation -> LayerNorm
  2. Missingness Emb:    3-state soft encoder -> flatten
  3. Time Embedding:     sinusoidal + learnable (delta_t -> d_time)
  4. Fusion:             concat all -> Linear -> d_model
  5. IMST-Mamba Blocks x n_layers: time-conditioned SSM + FFN
  6. Temporal Aggregator: multi-head attention pooling
  7. Classification Heads (multi-task):
       Primary:   P(sepsis warning state) per time step
       Auxiliary: P(in-hospital mortality) [optional]
       Auxiliary: SOFA score prediction   [optional]
"""
from __future__ import annotations

import json
from typing import Optional

import torch
import torch.nn as nn

from src.models.modules.time_embedding import TimeEmbedding
from src.models.modules.missingness_encoder import MissingnessEncoder
from src.models.modules.temporal_decay import TemporalDecayLayer
from src.models.modules.selective_ssm import IMSTMambaBlock
from src.utils.challenge_utils import FEATURE_NAMES, N_FEATURES, RECENCY_THRESHOLD_INIT


def running_last_observed(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Return the most recent observed value at every step, using no future rows.

    `x` must contain the current normalized observation where `m==1` and any
    placeholder (normally zero) where `m==0`. At an observed step the returned
    value is the current observation; at a missing step it is the last value
    observed earlier in the same sequence. Before the first observation it is 0.
    """
    if x.shape != m.shape:
        raise ValueError("x and m must have identical shapes")
    if x.ndim != 3:
        raise ValueError("x and m must have shape (B, T, F)")

    last = torch.zeros_like(x[:, 0])
    history: list[torch.Tensor] = []
    for t in range(x.shape[1]):
        last = torch.where(m[:, t] > 0.5, x[:, t], last)
        history.append(last)
    return torch.stack(history, dim=1)


class AttentionPooling(nn.Module):
    """Multi-head attention pooling from a sequence to one patient vector."""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        key_padding_mask = ~attn_mask
        out, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))


class IMSTMamba(nn.Module):
    """Full IMST-Mamba model with an explicit ordered feature contract."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 4,
        d_miss: int = 32,
        d_time: int = 64,
        dropout: float = 0.1,
        use_auxiliary: bool = True,
        stats_path: Optional[str] = None,
        feature_names: Optional[list[str]] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.use_auxiliary = use_auxiliary

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

        # Per-feature decay initialization must follow the exact feature order.
        tau_init_hours = [RECENCY_THRESHOLD_INIT[name] / 3600.0 for name in self.feature_names]
        self.temporal_decay = TemporalDecayLayer(
            n_features=n_features,
            init_decay_hours=tau_init_hours,
        )

        # Inputs used by the official benchmark are standardized in model space,
        # so the population mean is zero unless a compatible stats file is loaded.
        self.register_buffer("x_mean", torch.zeros(n_features))

        self.miss_encoder = MissingnessEncoder(
            n_features=n_features,
            d_miss=d_miss,
        )
        self.time_emb = TimeEmbedding(d_out=d_time)

        fusion_in = n_features + n_features * d_miss + d_time
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mamba_blocks = nn.ModuleList([
            IMSTMambaBlock(d_model=d_model, d_state=d_state, d_time=d_time, dropout=dropout)
            for _ in range(n_layers)
        ])

        # The benchmark disables auxiliary heads, but keep them available for the
        # original research pipeline. n_heads is bounded so small benchmark models
        # remain valid while the default d_model=256 still uses 8 heads.
        n_heads = 8 if d_model % 8 == 0 else 4 if d_model % 4 == 0 else 1
        self.aggregator = AttentionPooling(d_model=d_model, n_heads=n_heads)

        self.head_sepsis = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        if use_auxiliary:
            self.head_mortality = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
            )
            self.head_sofa = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
            )

        if stats_path is not None:
            self.load_stats(stats_path)

    def load_stats(self, stats_path: str) -> None:
        """Load a population mean only when its feature contract is compatible."""
        with open(stats_path) as handle:
            stats = json.load(handle)
        mean = stats.get("mean", [])
        if len(mean) != self.n_features:
            raise ValueError(
                f"stats mean length ({len(mean)}) does not match n_features ({self.n_features})"
            )
        self.x_mean.copy_(torch.tensor(mean, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        delta_t: torch.Tensor,
        s: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.shape[-1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} features, received {x.shape[-1]}"
            )

        # s is log1p(hours since last observation). Convert it back to hours.
        s_hours = torch.expm1(s.clamp(max=10.0))

        # The previous implementation shifted x by one row. That made a current
        # observation use the previous row and made t=0 observations become zero.
        # This running state uses the current observation when present and only
        # past observations when the current value is missing.
        x_last = running_last_observed(x, m)
        x_mean = self.x_mean.to(x.device)
        x_imputed, _ = self.temporal_decay(
            s_hours=s_hours,
            x_last=x_last,
            x_mean=x_mean,
            m=m,
        )

        miss_emb = self.miss_encoder(s_hours, m)
        t_emb = self.time_emb(delta_t)
        h = torch.cat([x_imputed, miss_emb, t_emb], dim=-1)
        h = self.fusion(h)
        h = h * attn_mask.unsqueeze(-1).float()

        for block in self.mamba_blocks:
            h = block(h, t_emb, attn_mask)

        logit_sepsis = self.head_sepsis(h)
        outputs = {"logit_sepsis": logit_sepsis}

        if self.use_auxiliary:
            h_patient = self.aggregator(h, attn_mask)
            outputs["logit_mortality"] = self.head_mortality(h_patient)
            outputs["pred_sofa"] = self.head_sofa(h)

        return outputs

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        delta_t: torch.Tensor,
        s: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        out = self.forward(x, m, delta_t, s, attn_mask)
        return torch.sigmoid(out["logit_sepsis"]).squeeze(-1)


def build_model(cfg: dict, stats_path: Optional[str] = None) -> IMSTMamba:
    """Instantiate IMST-Mamba from config while preserving feature ordering."""
    m_cfg = cfg.get("model", {})
    n_features = int(m_cfg.get("n_features", N_FEATURES))
    data_cfg = cfg.get("data", {})
    configured_features = list(data_cfg.get("vital_features", [])) + list(data_cfg.get("lab_features", []))
    if len(configured_features) == n_features:
        feature_names = configured_features
    else:
        feature_names = FEATURE_NAMES[:n_features]

    return IMSTMamba(
        n_features=n_features,
        d_model=m_cfg.get("d_model", 256),
        d_state=m_cfg.get("d_state", 64),
        n_layers=m_cfg.get("n_layers", 4),
        d_miss=m_cfg.get("d_miss", 32),
        d_time=m_cfg.get("d_time", 64),
        dropout=m_cfg.get("dropout", 0.1),
        use_auxiliary=m_cfg.get("use_auxiliary_tasks", True),
        stats_path=stats_path,
        feature_names=feature_names,
    )
