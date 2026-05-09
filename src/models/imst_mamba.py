"""
IMST-Mamba: Informative-Missingness State-Space model for sepsis prediction.

Architecture (per observation event):
  1. Feature Encoding:   observed values + decay imputation → LayerNorm
  2. Missingness Emb:    3-state soft encoder → flatten
  3. Time Embedding:     sinusoidal + learnable (delta_t → d_time)
  4. Fusion:             concat all → Linear → d_model
  5. IMST-Mamba Blocks × n_layers: time-conditioned SSM + FFN
  6. Temporal Aggregator: multi-head attention pooling
  7. Classification Heads (multi-task):
       Primary:   P(sepsis in H hours)   [focal loss]
       Auxiliary: P(in-hospital mort.)   [BCE]
       Auxiliary: SOFA score prediction  [MSE]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.time_embedding import TimeEmbedding
from src.models.modules.missingness_encoder import MissingnessEncoder
from src.models.modules.temporal_decay import TemporalDecayLayer
from src.models.modules.selective_ssm import IMSTMambaBlock
from src.utils.challenge_utils import N_FEATURES, get_recency_thresholds_array


class AttentionPooling(nn.Module):
    """
    Multi-head attention pooling: aggregate a sequence into a single vector.
    Uses a learned query vector and attention over all time steps.
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,             # (B, T, d_model)
        attn_mask: torch.Tensor,      # (B, T) bool — True for valid
    ) -> torch.Tensor:                # (B, d_model)
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)   # (B, 1, d_model)
        # key_padding_mask: True for positions to IGNORE
        key_padding_mask = ~attn_mask
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))   # (B, d_model)


class IMSTMamba(nn.Module):
    """
    Full IMST-Mamba model.

    Args:
        n_features:    number of input features (F = 25)
        d_model:       model dimension
        d_state:       SSM state dimension
        n_layers:      number of IMST-Mamba blocks
        d_miss:        missingness embedding dim per feature
        d_time:        time embedding dimension
        dropout:       dropout rate
        use_auxiliary: whether to include auxiliary prediction heads
        stats_path:    path to normalization stats (for x_mean initialization)
    """

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
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.use_auxiliary = use_auxiliary

        # ── Module 1: Temporal Decay (for imputation) ─────────────────────────
        tau_init_hours = get_recency_thresholds_array() / 3600.0
        self.temporal_decay = TemporalDecayLayer(
            n_features=n_features,
            init_decay_hours=tau_init_hours.tolist(),
        )

        # Population mean for decay imputation (updated from stats.json)
        self.register_buffer("x_mean", torch.zeros(n_features))

        # ── Module 2: Missingness Encoder ────────────────────────────────────
        self.miss_encoder = MissingnessEncoder(
            n_features=n_features,
            d_miss=d_miss,
        )

        # ── Module 3: Time Embedding ──────────────────────────────────────────
        self.time_emb = TimeEmbedding(d_out=d_time)

        # ── Module 4: Input Fusion ────────────────────────────────────────────
        # Input: [x_imputed (F)] + [miss_emb (F*d_miss)] + [time_emb (d_time)]
        fusion_in = n_features + n_features * d_miss + d_time
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Module 5: IMST-Mamba Blocks ───────────────────────────────────────
        self.mamba_blocks = nn.ModuleList([
            IMSTMambaBlock(d_model=d_model, d_state=d_state, d_time=d_time, dropout=dropout)
            for _ in range(n_layers)
        ])

        # ── Module 6: Temporal Aggregator ─────────────────────────────────────
        self.aggregator = AttentionPooling(d_model=d_model, n_heads=8)

        # ── Module 7: Classification Heads ────────────────────────────────────
        # Primary: step-wise prediction (per time step)
        self.head_sepsis = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        if use_auxiliary:
            # Auxiliary 1: patient-level mortality
            self.head_mortality = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
            )
            # Auxiliary 2: step-wise SOFA regression
            self.head_sofa = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
            )

        # Load normalization stats if available
        if stats_path is not None:
            self.load_stats(stats_path)

    def load_stats(self, stats_path: str) -> None:
        """Load x_mean from normalization stats file."""
        with open(stats_path) as f:
            stats = json.load(f)
        self.x_mean = torch.tensor(stats["mean"], dtype=torch.float32)

    def forward(
        self,
        x: torch.Tensor,          # (B, T, F)  normalized feature values
        m: torch.Tensor,          # (B, T, F)  observation mask
        delta_t: torch.Tensor,    # (B, T)     elapsed time in hours
        s: torch.Tensor,          # (B, T, F)  log1p(hours since last obs)
        attn_mask: torch.Tensor,  # (B, T)     True for valid positions
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns dict with:
            logit_sepsis:   (B, T, 1)  step-wise sepsis logit
            logit_mortality:(B, 1)     patient-level mortality logit  [if use_auxiliary]
            pred_sofa:      (B, T, 1)  step-wise SOFA prediction     [if use_auxiliary]
        """
        B, T, F = x.shape
        device = x.device

        # ── 1. Compute decay-imputed values ───────────────────────────────────
        # s is log1p-transformed hours from dataset.py; keep in log space
        # and clamp to avoid numerical explosion in decay/missingness
        s_hours = torch.expm1(s.clamp(max=10.0))   # (B, T, F) — hours since last obs, clamped

        x_mean = self.x_mean.to(device)

        # Running forward-fill for x_last: shift x by 1 step
        # (use previous time step's observed value as "last value")
        x_last = torch.zeros_like(x)
        x_last[:, 1:] = x[:, :-1]

        x_imputed, decay_factor = self.temporal_decay(
            s_hours=s_hours,
            x_last=x_last,
            x_mean=x_mean,
            m=m,
        )   # (B, T, F)

        # ── 2. Missingness embeddings ──────────────────────────────────────────
        miss_emb = self.miss_encoder(s_hours, m)   # (B, T, F * d_miss)

        # ── 3. Time embeddings ─────────────────────────────────────────────────
        t_emb = self.time_emb(delta_t)            # (B, T, d_time)

        # ── 4. Fusion ──────────────────────────────────────────────────────────
        h = torch.cat([x_imputed, miss_emb, t_emb], dim=-1)   # (B, T, fusion_in)
        h = self.fusion(h)                                       # (B, T, d_model)

        # Mask padding
        h = h * attn_mask.unsqueeze(-1).float()

        # ── 5. IMST-Mamba blocks ───────────────────────────────────────────────
        for block in self.mamba_blocks:
            h = block(h, t_emb, attn_mask)   # (B, T, d_model)

        # ── 6. Primary head: step-wise sepsis prediction ───────────────────────
        logit_sepsis = self.head_sepsis(h)   # (B, T, 1)

        outputs = {"logit_sepsis": logit_sepsis}

        # ── 7. Auxiliary heads ─────────────────────────────────────────────────
        if self.use_auxiliary:
            # Patient-level representation via attention pooling
            h_patient = self.aggregator(h, attn_mask)   # (B, d_model)
            logit_mortality = self.head_mortality(h_patient)   # (B, 1)
            pred_sofa = self.head_sofa(h)                      # (B, T, 1)
            outputs["logit_mortality"] = logit_mortality
            outputs["pred_sofa"] = pred_sofa

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
        """Return P(sepsis) per time step. Shape: (B, T)."""
        self.eval()
        out = self.forward(x, m, delta_t, s, attn_mask)
        return torch.sigmoid(out["logit_sepsis"]).squeeze(-1)


def build_model(cfg: dict, stats_path: Optional[str] = None) -> IMSTMamba:
    """Instantiate IMST-Mamba from config dict."""
    m_cfg = cfg.get("model", {})
    return IMSTMamba(
        n_features=m_cfg.get("n_features", N_FEATURES),
        d_model=m_cfg.get("d_model", 256),
        d_state=m_cfg.get("d_state", 64),
        n_layers=m_cfg.get("n_layers", 4),
        d_miss=m_cfg.get("d_miss", 32),
        d_time=m_cfg.get("d_time", 64),
        dropout=m_cfg.get("dropout", 0.1),
        use_auxiliary=m_cfg.get("use_auxiliary_tasks", True),
        stats_path=stats_path,
    )
