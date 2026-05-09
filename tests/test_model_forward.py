"""
Unit tests: verify model forward passes run without errors.
Run with: pytest tests/test_model_forward.py -v
"""
import pytest
import torch

from src.utils.challenge_utils import N_FEATURES


def make_batch(B=2, T=48, F=N_FEATURES):
    """Create a minimal dummy batch for testing."""
    x = torch.randn(B, T, F)
    m = (torch.rand(B, T, F) > 0.5).float()
    delta_t = torch.rand(B, T) * 2     # hours
    s = torch.log1p(torch.rand(B, T, F) * 12)
    attn_mask = torch.ones(B, T, dtype=torch.bool)
    return x, m, delta_t, s, attn_mask


def test_imst_mamba_forward():
    from src.models.imst_mamba import IMSTMamba
    model = IMSTMamba(d_model=64, d_state=16, n_layers=2, d_miss=8, d_time=32)
    model.eval()
    x, m, dt, s, mask = make_batch()
    with torch.no_grad():
        out = model(x, m, dt, s, mask)
    assert "logit_sepsis" in out
    assert out["logit_sepsis"].shape == (2, 48, 1)
    assert "logit_mortality" in out
    assert "pred_sofa" in out


def test_grud_forward():
    from src.models.grud import GRUD
    model = GRUD(hidden_size=64, n_layers=1)
    model.eval()
    x, m, dt, s, mask = make_batch()
    with torch.no_grad():
        out = model(x, m, dt, s, mask)
    assert out["logit_sepsis"].shape == (2, 48, 1)


def test_lstm_forward():
    from src.models.lstm_baseline import LSTMBaseline
    model = LSTMBaseline(hidden_size=64, n_layers=1)
    model.eval()
    x, m, dt, s, mask = make_batch()
    with torch.no_grad():
        out = model(x, m, dt, s, mask)
    assert out["logit_sepsis"].shape == (2, 48, 1)


def test_transformer_forward():
    from src.models.transformer_baseline import TransformerBaseline
    model = TransformerBaseline(d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128)
    model.eval()
    x, m, dt, s, mask = make_batch()
    with torch.no_grad():
        out = model(x, m, dt, s, mask)
    assert out["logit_sepsis"].shape == (2, 48, 1)


def test_retain_forward():
    from src.models.retain import RETAIN
    model = RETAIN(hidden_size=64)
    model.eval()
    x, m, dt, s, mask = make_batch(B=2, T=10)   # shorter for speed
    with torch.no_grad():
        out = model(x, m, dt, s, mask)
    assert out["logit_sepsis"].shape == (2, 10, 1)


def test_missingness_encoder():
    from src.models.modules.missingness_encoder import MissingnessEncoder
    enc = MissingnessEncoder(n_features=N_FEATURES, d_miss=16)
    s = torch.rand(2, 10, N_FEATURES) * 24    # hours
    m = (torch.rand(2, 10, N_FEATURES) > 0.5).float()
    out = enc(s, m)
    assert out.shape == (2, 10, N_FEATURES * 16)


def test_focal_loss():
    from src.training.losses import FocalLoss
    loss_fn = FocalLoss()
    logit = torch.randn(4, 10)
    target = (torch.rand(4, 10) > 0.8).float()
    loss = loss_fn(logit, target)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_delong_test():
    import numpy as np
    from src.evaluation.significance_tests import delong_roc_test
    rng = np.random.default_rng(42)
    n = 500
    y = (rng.random(n) > 0.8).astype(int)
    s1 = rng.random(n) + 0.2 * y   # slightly better for positives
    s2 = rng.random(n) + 0.1 * y
    a1, a2, p = delong_roc_test(y, s1, s2)
    assert 0 <= a1 <= 1
    assert 0 <= a2 <= 1
    assert 0 <= p <= 1
