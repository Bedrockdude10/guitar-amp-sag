"""Tests for reproducibility scaffolding: seeding, model factory, config merge."""

import pytest
import torch

from power_sag import load_config, seed_everything
from power_sag.nn import (
    MODEL_TYPES,
    ConditionedLSTMNoPhysics,
    PowerSagModel,
    UnconditionedLSTM,
    build_model,
)


def test_seed_everything_makes_init_reproducible():
    cfg = load_config()
    cfg.update(hidden_size=8, coupling_hidden=8)

    seed_everything(123)
    a = build_model(cfg)
    seed_everything(123)
    b = build_model(cfg)

    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_seed_everything_returns_seed():
    assert seed_everything(7) == 7


def test_build_model_all_types_and_forward():
    cfg = load_config()
    cfg.update(hidden_size=8, coupling_hidden=8)
    x = torch.randn(1, 32, 1)
    expected = {
        "physics": PowerSagModel,
        "no_physics": ConditionedLSTMNoPhysics,
        "black_box": UnconditionedLSTM,
    }
    for kind in MODEL_TYPES:
        cfg["model"] = kind
        model = build_model(cfg)
        assert isinstance(model, expected[kind])
        assert model(x).shape == (1, 32, 1)  # model(x) -> y for all


def test_build_model_default_is_physics():
    cfg = load_config()
    cfg.pop("model", None)
    assert isinstance(build_model(cfg), PowerSagModel)


def test_build_model_unknown_raises():
    cfg = load_config()
    cfg["model"] = "nope"
    with pytest.raises(ValueError):
        build_model(cfg)


def test_config_defaults_merge():
    cfg = load_config("configs/experiment_no_physics.yaml")
    assert cfg["model"] == "no_physics"     # override applied
    assert cfg["fs"] == 48000               # inherited from default.yaml
    assert "hidden_size" in cfg and "C1" in cfg


def test_uniform_forward_signature_across_models():
    """Every model accepts (x, V0, state, return_state) and returns a triple."""
    cfg = load_config()
    cfg.update(hidden_size=8, coupling_hidden=8)
    x = torch.randn(1, 20, 1)
    for kind in MODEL_TYPES:
        cfg["model"] = kind
        model = build_model(cfg)
        y, V_final, state = model(x, V0=None, state=None, return_state=True)
        assert y.shape == (1, 20, 1)
        # Only the physics model carries a supply state.
        assert (V_final is None) == (kind != "physics")
