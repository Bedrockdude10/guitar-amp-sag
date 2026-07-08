"""Tests for the fixed cabinet IR convolution stage."""

import torch

from power_sag.cabinet import CabinetIR


def test_output_length_equals_input_length():
    ir = torch.randn(128)
    cab = CabinetIR(ir)
    x = torch.randn(2, 500, 1)
    y = cab(x)
    assert y.shape == x.shape


def test_identity_ir_passes_signal_unchanged():
    delta = torch.tensor([1.0, 0.0, 0.0, 0.0])
    cab = CabinetIR(delta)
    x = torch.randn(1, 200, 1)
    y = cab(x)
    assert torch.allclose(y, x, atol=1e-5)


def test_gradients_do_not_flow_through_ir():
    ir = torch.randn(64)
    cab = CabinetIR(ir)
    # The IR is a buffer, not a trainable parameter.
    assert "ir" not in dict(cab.named_parameters())
    assert not cab.ir.requires_grad

    x = torch.randn(1, 100, 1, requires_grad=True)
    y = cab(x)
    y.sum().backward()
    # No gradient accumulates on the IR buffer.
    assert cab.ir.grad is None
    # But the input still receives gradients (the stage is differentiable in x).
    assert x.grad is not None
