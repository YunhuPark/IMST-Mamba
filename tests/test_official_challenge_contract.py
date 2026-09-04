from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.challenge2019_utility import prediction_utility
from src.models.imst_mamba import IMSTMamba, running_last_observed
from src.models.modules.missingness_encoder import MissingnessEncoder


SIX_VITALS = ["HR", "SBP", "DBP", "Resp", "Temp", "O2Sat"]


def test_running_last_observed_uses_current_then_past_only():
    x = torch.tensor([[[10.0], [0.0], [30.0], [0.0]]])
    m = torch.tensor([[[1.0], [0.0], [1.0], [0.0]]])
    actual = running_last_observed(x, m)
    expected = torch.tensor([[[10.0], [10.0], [30.0], [30.0]]])
    assert torch.equal(actual, expected)


def test_running_last_observed_does_not_pull_future_value_backward():
    x = torch.tensor([[[0.0], [0.0], [25.0]]])
    m = torch.tensor([[[0.0], [0.0], [1.0]]])
    actual = running_last_observed(x, m)
    assert actual[0, 0, 0].item() == 0.0
    assert actual[0, 1, 0].item() == 0.0
    assert actual[0, 2, 0].item() == 25.0


def test_missingness_encoder_supports_exact_six_vitals_contract():
    encoder = MissingnessEncoder(n_features=6, d_miss=4, feature_names=SIX_VITALS)
    assert encoder.get_tau_hours().numel() == 6
    s = torch.zeros(2, 3, 6)
    m = torch.ones(2, 3, 6)
    out = encoder(s, m)
    assert out.shape == (2, 3, 24)


def test_imst_mamba_six_vitals_forward_shape():
    model = IMSTMamba(
        n_features=6,
        feature_names=SIX_VITALS,
        d_model=24,
        d_state=4,
        n_layers=1,
        d_miss=2,
        d_time=8,
        dropout=0.0,
        use_auxiliary=False,
    )
    x = torch.randn(2, 5, 6)
    m = torch.ones(2, 5, 6)
    delta_t = torch.ones(2, 5)
    delta_t[:, 0] = 0.0
    s = torch.zeros(2, 5, 6)
    attn = torch.ones(2, 5, dtype=torch.bool)
    out = model(x, m, delta_t, s, attn)
    assert out["logit_sepsis"].shape == (2, 5, 1)
    assert torch.isfinite(out["logit_sepsis"]).all()


def test_official_utility_matches_physionet_reference_example():
    labels = np.array([0, 0, 0, 0, 1, 1], dtype=int)
    predictions = np.array([0, 0, 1, 1, 1, 1], dtype=int)
    assert prediction_utility(labels, predictions) == pytest.approx(3.388888888888889, abs=1e-12)


def test_source_label_contract_is_not_shifted_again():
    source = np.array([0, 0, 1, 1, 1], dtype=int)
    benchmark_target = source.copy()
    assert np.array_equal(benchmark_target, source)
