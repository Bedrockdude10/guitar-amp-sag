"""Physics-informed gray-box neural model of guitar amplifier power supply sag.

The package decomposes a tube guitar amplifier into two coupled subsystems:

* a *slow*, physics-governed power supply (:class:`power_sag.physics.PowerSupplyODE`), and
* a *fast*, neural audio path (:class:`power_sag.nn.PowerSagLSTM`)

coupled through a learned load-current estimator
(:class:`power_sag.nn.CouplingNetwork`) and conditioned via feature-wise linear
modulation (:class:`power_sag.nn.FiLMLayer`).  The full end-to-end model lives
in :class:`power_sag.nn.PowerSagModel`.

Public classes are re-exported here so callers can ``from power_sag import X``
regardless of which subpackage ``X`` physically lives in.
"""

from importlib import import_module
from typing import Any

from .config import load_config

__all__ = [
    "load_config",
    "PowerSupplyODE",
    "CouplingNetwork",
    "FiLMLayer",
    "PowerSagLSTM",
    "PowerSagModel",
    "UnconditionedLSTM",
    "ConditionedLSTMNoPhysics",
    "ESRLoss",
    "PreEmphasisLoss",
    "AudioDataset",
    "SequenceDataset",
    "SequenceBatchSampler",
    "CabinetIR",
    "SagEvaluator",
    "SyntheticSagAmp",
    "resolve_device",
    "configure_backends",
]

# Maps public name -> submodule providing it (kept lazy so ``import power_sag``
# does not eagerly pull in torch until a component is actually requested).
_EXPORTS = {
    "PowerSupplyODE": "physics",
    "CouplingNetwork": "nn",
    "FiLMLayer": "nn",
    "PowerSagLSTM": "nn",
    "PowerSagModel": "nn",
    "UnconditionedLSTM": "nn",
    "ConditionedLSTMNoPhysics": "nn",
    "ESRLoss": "losses",
    "PreEmphasisLoss": "losses",
    "AudioDataset": "data",
    "SequenceDataset": "data",
    "SequenceBatchSampler": "data",
    "CabinetIR": "dsp",
    "SagEvaluator": "evaluation",
    "SyntheticSagAmp": "synthetic",
    "resolve_device": "utils",
    "configure_backends": "utils",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial dispatch
    if name in _EXPORTS:
        module = import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
