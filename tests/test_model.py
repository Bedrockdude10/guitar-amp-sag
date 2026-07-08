"""Tests for the full end-to-end gray-box model."""

import torch

from power_sag import load_config
from power_sag.nn import PowerSagModel


def build_model():
    cfg = load_config()
    # Keep it small/fast for tests but keep V_oc learnable to exercise its grad.
    cfg["hidden_size"] = 8
    cfg["coupling_hidden"] = 8
    cfg["learn_V_oc"] = True
    return PowerSagModel.from_config(cfg)


def test_end_to_end_forward_shape():
    model = build_model()
    x = torch.randn(2, 64, 1) * 0.5
    y = model(x)
    assert y.shape == (2, 64, 1)


def test_backward_runs():
    model = build_model()
    x = torch.randn(1, 64, 1) * 0.5
    target = torch.randn(1, 64, 1) * 0.5
    y = model(x)
    loss = ((y - target) ** 2).mean()
    loss.backward()  # must not raise


def test_all_component_gradients_nonzero():
    model = build_model()
    x = torch.randn(2, 80, 1) * 0.5
    target = torch.randn(2, 80, 1) * 0.5
    y = model(x)
    loss = ((y - target) ** 2).mean()
    loss.backward()

    # Physics parameters.
    assert model.physics._R_eff.grad is not None and model.physics._R_eff.grad.abs() > 0
    assert model.physics.C1.grad is not None and model.physics.C1.grad.abs() > 0
    assert model.physics.V_oc.grad is not None and model.physics.V_oc.grad.abs() > 0

    # Coupling network parameters.
    coupling_grad = sum(
        p.grad.abs().sum() for p in model.coupling.parameters() if p.grad is not None
    )
    assert coupling_grad > 0

    # Audio path parameters.
    audio_grad = sum(
        p.grad.abs().sum() for p in model.audio.parameters() if p.grad is not None
    )
    assert audio_grad > 0
