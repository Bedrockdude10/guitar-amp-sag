"""Tests for the FiLM conditioning layer."""

import torch

from power_sag.nn import FiLMLayer


def test_identity_when_gamma_one_beta_zero():
    layer = FiLMLayer(feature_dim=8, cond_dim=1)
    # Force gamma == 1, beta == 0 regardless of conditioning.
    with torch.no_grad():
        layer.to_gamma.weight.zero_()
        layer.to_gamma.bias.fill_(1.0)
        layer.to_beta.weight.zero_()
        layer.to_beta.bias.zero_()
    h = torch.randn(2, 10, 8)
    v = torch.randn(2, 10, 1)
    out = layer(h, v)
    assert torch.allclose(out, h, atol=1e-6)


def test_output_shape_matches_input():
    layer = FiLMLayer(feature_dim=16, cond_dim=1)
    h = torch.randn(4, 25, 16)
    v = torch.randn(4, 25, 1)
    out = layer(h, v)
    assert out.shape == h.shape


def test_gradients_flow():
    layer = FiLMLayer(feature_dim=8, cond_dim=1)
    h = torch.randn(2, 10, 8)
    v = torch.randn(2, 10, 1, requires_grad=True)
    out = layer(h, v)
    out.sum().backward()
    assert v.grad is not None and v.grad.abs().sum() > 0
    for p in layer.parameters():
        assert p.grad is not None
