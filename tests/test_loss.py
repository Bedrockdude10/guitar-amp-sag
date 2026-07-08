"""Tests for the ESR and pre-emphasis losses."""

import torch

from power_sag.losses import ESRLoss, PreEmphasisLoss


def test_esr_zero_when_equal():
    loss = ESRLoss()
    x = torch.randn(2, 100, 1)
    assert loss(x, x).item() < 1e-8


def test_esr_positive_when_different():
    loss = ESRLoss()
    target = torch.randn(2, 100, 1)
    pred = target + 0.1 * torch.randn_like(target)
    assert loss(pred, target).item() > 0.0


def test_esr_one_for_zero_prediction():
    loss = ESRLoss()
    target = torch.randn(2, 100, 1)
    pred = torch.zeros_like(target)
    assert abs(loss(pred, target).item() - 1.0) < 1e-6


def test_preemphasis_output_shape_matches_input():
    loss = PreEmphasisLoss(coeff=0.85)
    x = torch.randn(3, 50, 1)
    filtered = loss.pre_emphasis(x)
    assert filtered.shape == x.shape


def test_preemphasis_loss_is_scalar():
    loss = PreEmphasisLoss(coeff=0.85)
    pred = torch.randn(2, 40, 1)
    target = torch.randn(2, 40, 1)
    value = loss(pred, target)
    assert value.dim() == 0
