"""Physics-informed gray-box neural model of guitar amplifier power supply sag.

The package decomposes a tube guitar amplifier into two coupled subsystems:

* a *slow*, physics-governed power supply (``physics.PowerSupplyODE``), and
* a *fast*, neural audio path (``audio_model.PowerSagLSTM``)

coupled through a learned load-current estimator (``coupling.CouplingNetwork``)
and conditioned via feature-wise linear modulation (``film.FiLMLayer``).  The
full end-to-end model lives in ``model.PowerSagModel``.
"""

from importlib import import_module
from pathlib import Path
from typing import Any, Dict

import yaml

__all__ = [
    "load_config",
    "PowerSupplyODE",
    "CouplingNetwork",
    "FiLMLayer",
    "PowerSagLSTM",
    "PowerSagModel",
    "ESRLoss",
    "PreEmphasisLoss",
    "AudioDataset",
    "SequenceDataset",
    "CabinetIR",
    "SagEvaluator",
]

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load a YAML configuration file (defaults to ``configs/default.yaml``)."""
    path = Path(path) if path is not None else _DEFAULT_CONFIG
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


# Lazy re-exports keep ``import power_sag`` cheap while still exposing the API.
def __getattr__(name: str) -> Any:  # pragma: no cover - trivial dispatch
    _mapping = {
        "PowerSupplyODE": "physics",
        "CouplingNetwork": "coupling",
        "FiLMLayer": "film",
        "PowerSagLSTM": "audio_model",
        "PowerSagModel": "model",
        "ESRLoss": "loss",
        "PreEmphasisLoss": "loss",
        "AudioDataset": "data",
        "SequenceDataset": "data",
        "CabinetIR": "cabinet",
        "SagEvaluator": "evaluation",
    }
    if name in _mapping:
        module = import_module(f".{_mapping[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
