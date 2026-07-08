"""Tests for the supply-voltage-conditioned LSTM audio path."""

import torch

from power_sag.audio_model import PowerSagLSTM


def test_forward_shape():
    model = PowerSagLSTM(hidden_size=32, num_layers=1)
    x = torch.randn(3, 40, 1)
    V = torch.full((3, 40, 1), 410.0)
    y, state = model(x, V)
    assert y.shape == (3, 40, 1)


def test_stateful_continuation():
    model = PowerSagLSTM(hidden_size=16, num_layers=1)
    model.eval()
    x = torch.randn(1, 60, 1)
    V = torch.full((1, 60, 1), 405.0)

    # Whole sequence at once.
    with torch.no_grad():
        y_full, _ = model(x, V)
        # Split into two halves, carrying LSTM state across the boundary.
        y1, state = model(x[:, :30], V[:, :30])
        y2, _ = model(x[:, 30:], V[:, 30:], state)
    y_split = torch.cat([y1, y2], dim=1)
    assert torch.allclose(y_full, y_split, atol=1e-5)


def test_gradients_flow_to_supply_voltage():
    model = PowerSagLSTM(hidden_size=16, num_layers=1)
    x = torch.randn(2, 30, 1)
    V = torch.full((2, 30, 1), 410.0, requires_grad=True)
    y, _ = model(x, V)
    y.sum().backward()
    assert V.grad is not None and V.grad.abs().sum() > 0
