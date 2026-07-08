"""Tests for the load-current coupling network."""

import torch

from power_sag.nn import CouplingNetwork


def test_output_shape():
    net = CouplingNetwork(hidden=16)
    x = torch.randn(3, 50, 1)
    V = torch.full((3, 50, 1), 410.0)
    out = net(x, V)
    assert out.shape == (3, 50, 1)


def test_output_non_negative():
    net = CouplingNetwork(hidden=16)
    x = torch.randn(4, 200, 1) * 5.0  # large swings, both signs
    V = torch.full((4, 200, 1), 400.0)
    out = net(x, V)
    assert (out >= 0.0).all()


def test_gradients_flow():
    net = CouplingNetwork(hidden=16)
    x = torch.randn(2, 20, 1)
    V = torch.full((2, 20, 1), 410.0)
    out = net(x, V)
    out.sum().backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
