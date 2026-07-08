"""Smoke tests for the plotting layer.

Rendering is exercised on the headless ``Agg`` backend; we assert figure
structure (axis / line counts) and that files save to non-empty PNGs. The
*numbers* the plots show are validated in test_metrics / test_report.
"""

import matplotlib

matplotlib.use("Agg")  # headless, before pyplot is imported anywhere

import numpy as np  # noqa: E402
import torch  # noqa: E402

from power_sag.reporting import plots  # noqa: E402


def test_signal_comparison_has_two_lines():
    fig = plots.plot_signal_comparison(torch.randn(2000), torch.randn(2000), fs=48000.0)
    ax = fig.axes[0]
    assert len(ax.lines) == 2  # target + prediction
    assert ax.get_legend() is not None


def test_output_and_supply_uses_two_stacked_axes_not_dual():
    y = torch.randn(3000)
    v = torch.linspace(415.0, 395.0, 3000)
    fig = plots.plot_output_and_supply(y, v, fs=48000.0)
    # Two separate subplots (no dual-axis / twinx): distinct y-units per axis.
    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylabel() == "amplitude"
    assert fig.axes[1].get_ylabel() == "B+ (V)"


def test_output_and_supply_reference_overlay():
    y = torch.randn(1500)
    v = torch.linspace(415.0, 400.0, 1500)
    vr = torch.linspace(415.0, 398.0, 1500)
    fig = plots.plot_output_and_supply(y, v, fs=48000.0, v_reference=vr)
    assert len(fig.axes[1].lines) == 2  # model + reference V_B+


def test_recovery_fit_annotates_tau():
    fs = 2000.0
    t = np.arange(1000) / fs
    env = 3.0 * np.exp(-t / 0.02) + 0.5  # known tau = 20 ms
    fig = plots.plot_recovery_fit(env, fs=fs)
    labels = [ln.get_label() for ln in fig.axes[0].lines]
    assert any("τ" in lbl for lbl in labels)


def test_training_curves_two_series():
    fig = plots.plot_training_curves([1, 2, 3], [1.0, 0.5, 0.3], [1.1, 0.6, 0.4])
    assert len(fig.axes[0].lines) == 2


def test_metric_comparison_one_bar_per_model():
    results = {
        "physics": {"esr": 0.02},
        "no-physics": {"esr": 0.05},
        "black-box": {"esr": 0.08},
    }
    fig = plots.plot_metric_comparison(results, metric="esr")
    bars = [p for p in fig.axes[0].patches]
    assert len(bars) == 3


def test_parameter_recovery_two_stacked_axes():
    steps = list(range(10))
    fig = plots.plot_parameter_recovery(
        steps, r_eff=[150 + i for i in steps], c1_uf=[47 - i for i in steps],
        true_r_eff=300.0, true_c1_uf=22.0,
    )
    assert len(fig.axes) == 2  # R_eff and C1 never share a y-axis


def test_save_figure_writes_nonempty_png(tmp_path):
    fig = plots.plot_signal_comparison(torch.randn(500), torch.randn(500), fs=48000.0)
    path = plots.save_figure(fig, str(tmp_path / "fig.png"))
    assert (tmp_path / "fig.png").stat().st_size > 0
    assert path.endswith("fig.png")
