"""Neural network components: coupling, FiLM conditioning, and audio path."""

from typing import Any, Dict

from .audio_model import PowerSagLSTM
from .baselines import ConditionedLSTMNoPhysics, UnconditionedLSTM
from .coupling import CouplingNetwork
from .film import FiLMLayer
from .model import PowerSagModel
from .recurrence import SupplyRecurrence

__all__ = [
    "CouplingNetwork",
    "FiLMLayer",
    "PowerSagLSTM",
    "PowerSagModel",
    "SupplyRecurrence",
    "UnconditionedLSTM",
    "ConditionedLSTMNoPhysics",
    "build_model",
    "MODEL_TYPES",
]

# Declarative model selection for reproducible experiments (config-driven).
MODEL_TYPES = ("physics", "no_physics", "black_box")


def build_model(config: Dict[str, Any], cabinet=None):
    """Build a model from ``config['model']`` (defaults to the physics model).

    * ``"physics"``    -> PowerSagModel (the proposed gray-box model)
    * ``"no_physics"`` -> ConditionedLSTMNoPhysics (learned slow state ablation)
    * ``"black_box"``  -> UnconditionedLSTM (no supply awareness)

    All share the ``(x, V0, state, return_state)`` forward signature, so the
    training loop is identical across them.
    """
    kind = config.get("model", "physics")
    if kind == "physics":
        return PowerSagModel.from_config(config, cabinet=cabinet)
    if kind == "no_physics":
        return ConditionedLSTMNoPhysics.from_config(config)
    if kind == "black_box":
        return UnconditionedLSTM.from_config(config)
    raise ValueError(f"unknown model type {kind!r}; expected one of {MODEL_TYPES}")
