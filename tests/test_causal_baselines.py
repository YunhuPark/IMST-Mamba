"""Online predictions must not depend on observations after the prediction time."""
import pytest
import torch

from src.models.transformer_baseline import TransformerBaseline
from src.models.retain import RETAIN


@pytest.fixture(params=["transformer", "retain"])
def model(request):
    torch.manual_seed(7)
    if request.param == "transformer":
        result = TransformerBaseline(n_features=4, d_model=16, nhead=2,
                                     num_encoder_layers=1, dim_feedforward=24,
                                     d_time=4, dropout=0)
    else:
        result = RETAIN(n_features=4, hidden_size=8, dropout=0)
    return result.eval()


def batch(length):
    torch.manual_seed(19)
    x = torch.randn(1, length, 4)
    return [x, torch.ones_like(x), torch.ones(1, length),
            torch.zeros_like(x), torch.ones(1, length, dtype=torch.bool)]


def test_prefix_predictions_equal_full_sequence(model):
    args = batch(6)
    with torch.no_grad():
        full = model(*args)["logit_sepsis"]
        for length in (1, 3, 5):
            prefix = model(*(x[:, :length] for x in args))["logit_sepsis"]
            torch.testing.assert_close(full[:, :length], prefix, rtol=1e-5, atol=1e-6)


def test_future_values_and_right_padding_do_not_change_past(model):
    args = batch(6)
    altered = [x.clone() for x in args]
    altered[0][:, 3:] = 100
    altered[4][:, 3:] = False
    with torch.no_grad():
        actual = model(*args)["logit_sepsis"][:, :3]
        other = model(*altered)["logit_sepsis"]
    torch.testing.assert_close(actual, other[:, :3], rtol=1e-5, atol=1e-6)
    assert torch.isfinite(other).all()
    assert (other[:, 3:] == 0).all()
