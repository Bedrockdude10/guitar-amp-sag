"""Tests for the comparison baseline models."""

import torch

from power_sag import load_config
from power_sag.nn import ConditionedLSTMNoPhysics, UnconditionedLSTM


def test_unconditioned_lstm_shape_and_grad():
    model = UnconditionedLSTM(hidden_size=8)
    x = torch.randn(2, 40, 1)
    y = model(x)
    assert y.shape == (2, 40, 1)
    y.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_conditioned_no_physics_shape_and_grad():
    model = ConditionedLSTMNoPhysics(hidden_size=8, state_hidden=4)
    x = torch.randn(2, 40, 1)
    y = model(x)
    assert y.shape == (2, 40, 1)
    y.sum().backward()
    # Gradient reaches both the audio path and the learned slow-state GRU.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.lstm.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.state_gru.parameters())


def test_baselines_from_config():
    cfg = load_config()
    cfg.update(hidden_size=8, num_layers=1, coupling_hidden=4)
    x = torch.randn(1, 32, 1)
    for model in (UnconditionedLSTM.from_config(cfg), ConditionedLSTMNoPhysics.from_config(cfg)):
        assert model(x).shape == (1, 32, 1)


def test_baselines_are_drop_in_for_evaluator():
    """Baselines share PowerSagModel's ``model(x) -> y`` call convention."""
    from power_sag.evaluation import SagEvaluator

    ev = SagEvaluator(fs=48000.0)
    model = UnconditionedLSTM(hidden_size=8)
    result = ev.attack_bloom(model, duration=0.02)
    assert set(result) == {"attack_rms", "settle_rms", "settle_over_attack"}
