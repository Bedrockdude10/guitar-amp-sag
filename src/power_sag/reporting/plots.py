"""Publication figures for the paper.

Rendering is kept thin and separate from metric computation: the numbers these
figures show come from :mod:`power_sag.reporting.metrics` (heavily tested), and
these functions only lay them out.  Each returns a matplotlib ``Figure`` so the
caller decides whether to save or show; tests render them on the ``Agg`` backend
and assert structure (axis count, line count, saved-file size).

Design rules (see the project data-viz method): a fixed-order colourblind-safe
palette, thin marks, recessive grid, legends for >= 2 series, and **no dual-axis
charts** -- quantities with different units (audio vs. volts, R_eff vs. C1) go in
stacked subplots that share the x-axis, never on two y-scales.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .style import (
    FIGSIZE_SMALL,
    FIGSIZE_SMALL_STACKED,
    FIGSIZE_WIDE,
    FIGSIZE_WIDE_STACKED,
    MAX_PLOT_POINTS,
    ROLE_COLORS,
    apply_style,
    color_for_index,
)


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def _stride(n: int) -> int:
    """Decimation stride that keeps a length-``n`` signal under MAX_PLOT_POINTS."""
    return max(int(np.ceil(n / MAX_PLOT_POINTS)), 1)


def _time_axis(n: int, fs: float, step: int) -> np.ndarray:
    return np.arange(0, n, step) / fs


def plot_signal_comparison(prediction, target, fs: float, title: str = "Output"):
    """Overlay predicted vs. target audio on a single (shared-unit) axis."""
    import matplotlib.pyplot as plt

    apply_style()
    p, t = _np(prediction), _np(target)
    step = _stride(max(len(p), len(t)))
    tp = np.arange(0, len(p), step) / fs
    tt = np.arange(0, len(t), step) / fs
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(tt, t[::step], color=ROLE_COLORS["target"], label="target", alpha=0.9)
    ax.plot(tp, p[::step], color=ROLE_COLORS["prediction"], label="prediction", alpha=0.9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return fig


def plot_output_and_supply(output, voltage, fs: float, v_reference=None):
    """Output audio and the B+ trajectory as stacked subplots sharing time.

    Deliberately two subplots, not a dual-axis plot: amplitude and volts have
    different units and must not share a y-scale.
    """
    import matplotlib.pyplot as plt

    apply_style()
    y, v = _np(output), _np(voltage)
    step_y = _stride(len(y))
    step_v = _stride(len(v))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=FIGSIZE_WIDE_STACKED, sharex=True)
    ax0.plot(_time_axis(len(y), fs, step_y), y[::step_y],
             color=ROLE_COLORS["prediction"], label="output")
    ax0.set_ylabel("amplitude")
    ax0.legend(loc="upper right")

    ax1.plot(_time_axis(len(v), fs, step_v), v[::step_v],
             color=ROLE_COLORS["prediction"], label="V_B+ (model)")
    if v_reference is not None:
        vr = _np(v_reference)
        step_r = _stride(len(vr))
        ax1.plot(_time_axis(len(vr), fs, step_r), vr[::step_r],
                 color=ROLE_COLORS["reference"], label="V_B+ (reference)")
        ax1.legend(loc="upper right")
    ax1.set_ylabel("B+ (V)")
    ax1.set_xlabel("time (s)")
    ax1.set_title("Supply voltage", loc="left")
    return fig


def plot_recovery_fit(envelope, fs: float):
    """Recovery envelope with its fitted ``A*exp(-t/tau)+C`` overlay + tau label."""
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    apply_style()
    y = _np(envelope).astype(np.float64)
    t = np.arange(len(y)) / fs

    def model(tt, A, tau, C):
        return A * np.exp(-tt / tau) + C

    p0 = (y[0] - y[-1] or 1e-6, max(t[-1] / 3, 1e-6), y[-1])
    params, _ = curve_fit(model, t, y, p0=p0, maxfev=20000)
    tau = abs(params[1])

    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    ax.plot(t, y, color=ROLE_COLORS["target"], label="envelope", alpha=0.8)
    ax.plot(t, model(t, *params), color=ROLE_COLORS["fit"], linestyle="--",
            label=f"fit (τ = {tau * 1e3:.1f} ms)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("RMS")
    ax.set_title("Recovery curve")
    ax.legend(loc="upper right")
    return fig


def plot_training_curves(epochs: Sequence[int], train: Sequence[float], val: Sequence[float]):
    """Train/val loss vs. epoch (one shared loss axis, log-scaled)."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    ax.plot(epochs, train, color=ROLE_COLORS["prediction"], label="train")
    ax.plot(epochs, val, color=ROLE_COLORS["reference"], label="val")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training")
    ax.legend(loc="upper right")
    return fig


def plot_metric_comparison(results: Dict[str, Dict[str, float]], metric: str):
    """Grouped bar chart of one metric across models (fixed colour per model)."""
    import matplotlib.pyplot as plt

    apply_style()
    models = list(results)
    values = [results[m][metric] for m in models]
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2, 3.2))
    for i, (name, value) in enumerate(zip(models, values)):
        ax.bar(i, value, color=color_for_index(i), width=0.6, label=name)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by model")
    return fig


def plot_parameter_recovery(
    steps: Sequence[int],
    r_eff: Sequence[float],
    c1_uf: Sequence[float],
    true_r_eff: float,
    true_c1_uf: float,
):
    """R_eff and C1 vs. training step in stacked subplots (never a dual axis)."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=FIGSIZE_SMALL_STACKED, sharex=True)
    ax0.plot(steps, r_eff, color=ROLE_COLORS["prediction"], label="estimate")
    ax0.axhline(true_r_eff, color=ROLE_COLORS["target"], linestyle="--", label="truth")
    ax0.set_ylabel("R_eff (Ω)")
    ax0.legend(loc="upper right")
    ax1.plot(steps, c1_uf, color=ROLE_COLORS["prediction"], label="estimate")
    ax1.axhline(true_c1_uf, color=ROLE_COLORS["target"], linestyle="--", label="truth")
    ax1.set_ylabel("C1 (µF)")
    ax1.set_xlabel("training step")
    return fig


def save_figure(fig, path: str) -> str:
    """Save a figure and close it; returns the path."""
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
