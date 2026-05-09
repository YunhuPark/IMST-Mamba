"""
Time-Conditioned Selective State Space Model (SSM) block.

Core idea: Standard Mamba uses a learned scalar Δ (discretization step).
We replace it with a function of actual elapsed time delta_t, so that
the state transition rate adapts to real temporal gaps between observations.

Architecture:
  Standard Mamba SSM:
    Δ = softplus(Linear(x))          ← learned, ignores time
  IMST-Mamba SSM:
    Δ = softplus(Linear([x; t_emb])) ← conditioned on real elapsed time

This pure-PyTorch implementation does not require CUDA extensions,
ensuring reproducibility across platforms (Windows, Linux, macOS).

For production speedup, the SSM scan can be replaced with
mamba-ssm's optimized CUDA kernels after correctness is verified.

Reference:
  Gu & Dao (2023) Mamba: Linear-Time Sequence Modeling with Selective State Spaces
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMLayer(nn.Module):
    """
    One selective SSM block with time-conditioned discretization.

    Args:
        d_model: input/output dimension
        d_state: SSM state dimension (N in Mamba notation)
        d_conv:  local convolution width (= 4 in original Mamba)
        expand:  expansion factor for inner dimension (= 2 in Mamba)
        dt_rank: rank of Δ projection (= ceil(d_model/16) in Mamba)
        d_time:  dimension of time embedding for Δ conditioning
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        d_time: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.d_time = d_time

        if dt_rank is None:
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        # Input projection: x → (z, x') [Mamba style]
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Local depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        # Δ projection — conditioned on time embedding
        # Input: dt_rank + d_time → d_inner (one Δ per inner channel)
        self.dt_proj = nn.Linear(self.dt_rank + d_time, self.d_inner, bias=True)

        # A: state transition matrix (log, to keep negative)
        # Shape (d_inner, d_state), initialized as log of 1..d_state
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def _ssm_scan(
        self,
        x: torch.Tensor,    # (B, T, d_inner)
        delta: torch.Tensor, # (B, T, d_inner)
        A: torch.Tensor,     # (d_inner, d_state) — negative
        B: torch.Tensor,     # (B, T, d_state)
        C: torch.Tensor,     # (B, T, d_state)
    ) -> torch.Tensor:
        """
        Discretize and run the SSM scan (sequential, works on any device).

        Discretization (ZOH):
          Δ_bar = softplus(delta)
          A_bar = exp(Δ_bar · A)    shape (B, T, d_inner, d_state)
          B_bar = Δ_bar · B         shape (B, T, d_inner, d_state)

        Scan:
          h_t = A_bar_t · h_{t-1} + B_bar_t · x_t
          y_t = C_t · h_t
        """
        B_batch, T, d_inner = x.shape
        d_state = A.shape[1]
        device = x.device

        # Discretize delta once — clamp to avoid exp overflow
        delta = F.softplus(delta).clamp(max=10.0)         # (B, T, d_inner)

        # Sequential scan — compute A_bar/B_bar per step to avoid
        # allocating the full (B, T, d_inner, d_state) tensor at once.
        h = torch.zeros(B_batch, d_inner, d_state, device=device, dtype=x.dtype)
        ys = []
        for t in range(T):
            dt = delta[:, t]                              # (B, d_inner)
            # A_bar_t: (B, d_inner, d_state) — clamp exponent for stability
            A_bar_t = torch.exp((dt.unsqueeze(-1) * A.unsqueeze(0)).clamp(min=-20.0, max=0.0))
            # B_bar_t: (B, d_inner, d_state)
            B_bar_t = dt.unsqueeze(-1) * B[:, t].unsqueeze(1)
            # h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            h = A_bar_t * h + B_bar_t * x[:, t].unsqueeze(-1)
            # y_t = sum_n C_t[n] * h[..., n]
            y_t = (C[:, t].unsqueeze(1) * h).sum(-1)   # (B, d_inner)
            ys.append(y_t)

        return torch.stack(ys, dim=1)   # (B, T, d_inner)

    def forward(
        self,
        x: torch.Tensor,         # (B, T, d_model)
        t_emb: torch.Tensor,     # (B, T, d_time) time embeddings
        attn_mask: torch.Tensor | None = None,  # (B, T) bool
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, T, d_model) input sequence
            t_emb:     (B, T, d_time) time embeddings for Δ conditioning
            attn_mask: (B, T) True for valid positions

        Returns:
            (B, T, d_model) output sequence
        """
        residual = x
        B_batch, T, _ = x.shape

        # 1. Input projection → split into gated branches
        xz = self.in_proj(x)                              # (B, T, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                    # each (B, T, d_inner)

        # 2. Local convolution (causal)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)                           # (B, T, d_inner)

        # 3. Compute B, C, and dt_rank from x_conv
        bcd = self.x_proj(x_conv)                         # (B, T, dt_rank + 2*d_state)
        dt_raw, B_ssm, C_ssm = torch.split(
            bcd, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # 4. Time-conditioned Δ
        dt_input = torch.cat([dt_raw, t_emb], dim=-1)    # (B, T, dt_rank + d_time)
        delta = self.dt_proj(dt_input)                    # (B, T, d_inner)

        # 5. Run SSM scan
        A = -torch.exp(self.A_log)                        # (d_inner, d_state) — negative
        y = self._ssm_scan(x_conv, delta, A, B_ssm, C_ssm)

        # 6. Gating (SiLU) + D skip
        y = y * F.silu(z) + self.D * x_conv

        # 7. Output projection
        out = self.out_proj(y)                            # (B, T, d_model)

        # 8. Residual + LayerNorm
        out = self.norm(out + residual)

        # Mask out padding
        if attn_mask is not None:
            out = out * attn_mask.unsqueeze(-1).float()

        return out


class IMSTMambaBlock(nn.Module):
    """
    Full IMST-Mamba block with SSM + feedforward sublayer.
    Dropout between blocks.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 64,
        d_time: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ssm = SelectiveSSMLayer(
            d_model=d_model, d_state=d_state, d_time=d_time
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SSM sublayer
        h = self.ssm(x, t_emb, attn_mask)
        h = self.dropout(h)
        # FFN sublayer (with residual inside ffn)
        h = h + self.ffn(h)
        return h
