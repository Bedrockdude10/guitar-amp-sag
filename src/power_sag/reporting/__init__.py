"""Reporting and visualisation for the paper.

Split into a heavily-tested pure-metric core (:mod:`metrics`, :mod:`report`) and
a thin rendering layer (:mod:`plots`, :mod:`style`).  Plotting imports matplotlib
lazily, so ``import power_sag.reporting`` works without it installed; install the
extra with ``pip install -e ".[viz]"``.
"""

from . import metrics, report
from .report import (
    protocol_report,
    regression_report,
    synthetic_report,
    to_json,
    trajectory_report,
)

__all__ = [
    "metrics",
    "report",
    "regression_report",
    "trajectory_report",
    "protocol_report",
    "synthetic_report",
    "to_json",
]
