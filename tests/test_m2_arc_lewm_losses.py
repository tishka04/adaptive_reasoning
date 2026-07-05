import pytest
import torch

from theory.m2.arc_lewm_losses import (
    collapse_diagnostics,
    prediction_mse,
    sigreg_loss,
)


def test_prediction_mse_non_negative():
    z_hat = torch.zeros(4, 8)
    z_t1 = torch.ones(4, 8)

    loss = prediction_mse(z_hat, z_t1)

    assert loss.item() >= 0


def test_prediction_mse_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="latent shape mismatch"):
        prediction_mse(torch.zeros(2, 4), torch.zeros(2, 5))


def test_sigreg_loss_non_negative_and_robust():
    z = torch.randn(8, 16)

    loss = sigreg_loss(z)

    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_constant_batch_triggers_collapse_alert():
    diagnostics = collapse_diagnostics(torch.ones(8, 16))

    assert diagnostics["collapse_detected"] is True
    assert diagnostics["latent_variance_above_min"] is False
    assert diagnostics["nan_or_inf_detected"] is False


def test_diverse_batch_avoids_collapse_alert():
    diagnostics = collapse_diagnostics(torch.eye(8, 8))

    assert diagnostics["collapse_detected"] is False
    assert diagnostics["latent_variance_above_min"] is True
    assert diagnostics["nan_or_inf_detected"] is False


def test_nan_or_inf_detected():
    z = torch.randn(4, 4)
    z[0, 0] = float("nan")

    diagnostics = collapse_diagnostics(z)

    assert diagnostics["nan_or_inf_detected"] is True
    assert diagnostics["collapse_detected"] is True
