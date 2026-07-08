"""Structured metric reports (JSON-serialisable, schema-checked).

A report is a nested dict of floats with a fixed schema, so it is trivially
testable: tests assert the keys are present and every value lands in its known
range (and, on synthetic ground truth, close to the known answer).  Reports feed
both the paper's tables and the plotting layer.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

import torch

from ..evaluation import SagEvaluator
from ..synthetic import SyntheticSagAmp
from . import metrics

ModelFn = Callable[[torch.Tensor], torch.Tensor]


def regression_report(model: ModelFn, x: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Standard audio regression metrics of ``model(x)`` against ``target``."""
    with torch.no_grad():
        y = model(x)
    return metrics.regression_metrics(y, target)


def trajectory_report(model: Any, x: torch.Tensor, v_true: torch.Tensor) -> Dict[str, float]:
    """Supply-trajectory metrics; requires a model exposing ``supply_trajectory``."""
    with torch.no_grad():
        v_pred, _ = model.supply_trajectory(x)
    return metrics.trajectory_metrics(v_pred, v_true)


def protocol_report(model: ModelFn, fs: float) -> Dict[str, Dict[str, float]]:
    """The four-part sag protocol run on any ``model(x) -> y``."""
    return SagEvaluator(fs=fs).run_protocol(model)


def synthetic_report(
    model: Any,
    amp: Optional[SyntheticSagAmp] = None,
    duration: float = 1.0,
    seed: int = 0,
    fs: float = 48000.0,
) -> Dict[str, Any]:
    """Full report on synthetic ground truth: regression + trajectory + sag.

    The trajectory block is only included when the model exposes
    ``supply_trajectory`` (i.e. the physics model, not the black-box baselines).
    """
    amp = amp or SyntheticSagAmp(fs=fs)
    x = amp.generate_excitation(duration, seed=seed)
    _, y_true, v_true = amp.generate(x)

    report: Dict[str, Any] = {"regression": regression_report(model, x, y_true)}
    if hasattr(model, "supply_trajectory"):
        report["trajectory"] = trajectory_report(model, x, v_true)
    report["sag_protocol"] = protocol_report(model, fs=amp.fs)
    return report


def to_json(report: Dict[str, Any], path: str) -> str:
    """Write a report to ``path`` as pretty JSON; returns the path."""
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return path
