"""Pure metric functions for reporting (the test-infected core).

Every function here is a pure ``tensor -> float`` (or ``-> dict`` of floats) with
a **known, checkable range**, which is what makes the reporting layer
test-infected: we do not eyeball whether a report is well-formed, we assert it.

Range contracts (all verified in ``tests/test_metrics.py``):

* ``esr``            in [0, inf); 0 iff prediction == target; ~1 for a zero
                     prediction against a non-zero target.
* ``mae``, ``rmse``  in [0, inf); 0 iff equal; ``rmse >= mae`` always.
* ``esr_db``         10*log10(esr); -inf for a perfect fit.
* ``segmental_snr_db`` higher is better; +inf for a perfect fit.
* ``correlation``    in [-1, 1]; 1 for identical, -1 for negated signals.
* ``sag_depth``      in [0, V[0]]; the peak droop below the starting voltage.
* ``relative_error`` in [0, inf); 0 when equal.

On synthetic ground truth the values are further pinned to physically-known
answers (recovered tau matches the true RC constant, trajectory correlation is
high, sag depth is positive under load).
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch

Tensorish = Union[torch.Tensor, np.ndarray]


def _flat(x: Tensorish) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.detach().reshape(-1).to(torch.float64)


# --------------------------------------------------------------- regression
def esr(prediction: Tensorish, target: Tensorish, eps: float = 1e-12) -> float:
    """Error-to-signal ratio ``sum((p-t)^2)/sum(t^2)`` in ``[0, inf)``."""
    p, t = _flat(prediction), _flat(target)
    return float(torch.sum((p - t) ** 2) / (torch.sum(t ** 2) + eps))


def mae(prediction: Tensorish, target: Tensorish) -> float:
    """Mean absolute error in ``[0, inf)``."""
    p, t = _flat(prediction), _flat(target)
    return float(torch.mean(torch.abs(p - t)))


def rmse(prediction: Tensorish, target: Tensorish) -> float:
    """Root mean squared error in ``[0, inf)`` (always ``>= mae``)."""
    p, t = _flat(prediction), _flat(target)
    return float(torch.sqrt(torch.mean((p - t) ** 2)))


def esr_db(prediction: Tensorish, target: Tensorish) -> float:
    """ESR in decibels, ``10*log10(esr)`` (``-inf`` for a perfect fit)."""
    value = esr(prediction, target)
    return float(10.0 * np.log10(value)) if value > 0 else float("-inf")


def segmental_snr_db(prediction: Tensorish, target: Tensorish, eps: float = 1e-12) -> float:
    """Signal-to-noise ratio in dB (``+inf`` for a perfect fit)."""
    p, t = _flat(prediction), _flat(target)
    noise = torch.sum((p - t) ** 2)
    signal = torch.sum(t ** 2)
    if noise <= eps:
        return float("inf")
    return float(10.0 * torch.log10(signal / (noise + eps)))


def correlation(a: Tensorish, b: Tensorish, eps: float = 1e-12) -> float:
    """Pearson correlation in ``[-1, 1]``."""
    x, y = _flat(a), _flat(b)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm() + eps
    return float(torch.clamp((x @ y) / denom, -1.0, 1.0))


def relative_error(estimate: float, truth: float, eps: float = 1e-12) -> float:
    """Relative error ``|estimate - truth| / |truth|`` in ``[0, inf)``."""
    return float(abs(estimate - truth) / (abs(truth) + eps))


def regression_metrics(prediction: Tensorish, target: Tensorish) -> Dict[str, float]:
    """Standard audio regression metrics as a dict."""
    return {
        "esr": esr(prediction, target),
        "esr_db": esr_db(prediction, target),
        "mae": mae(prediction, target),
        "rmse": rmse(prediction, target),
        "snr_db": segmental_snr_db(prediction, target),
    }


# ------------------------------------------------------------------- sag
def sag_depth(voltage: Tensorish) -> float:
    """Peak droop below the starting voltage, ``V[0] - min(V)`` in ``[0, V[0]]``."""
    v = _flat(voltage)
    return float(v[0] - v.min())


def trajectory_metrics(
    v_pred: Tensorish, v_true: Tensorish
) -> Dict[str, float]:
    """How well a predicted supply trajectory matches a reference one."""
    return {
        "rmse_volts": rmse(v_pred, v_true),
        "correlation": correlation(v_pred, v_true),
        "sag_depth_pred": sag_depth(v_pred),
        "sag_depth_true": sag_depth(v_true),
    }
