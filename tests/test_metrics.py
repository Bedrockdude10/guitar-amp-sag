"""Tests for reporting metrics: known-range invariants + synthetic ground truth.

This is the test-infected core of the reporting layer: every metric has a range
we assert, and on synthetic data we assert the physically-known answers.
"""

import math

import numpy as np
import torch

from power_sag.reporting import metrics


# ---------------------------------------------------------------- ranges
def test_esr_range_and_anchors():
    t = torch.randn(500)
    assert metrics.esr(t, t) < 1e-9                 # perfect fit -> 0
    assert abs(metrics.esr(torch.zeros_like(t), t) - 1.0) < 1e-6  # zero pred -> 1
    assert metrics.esr(t + 0.1, t) > 0.0            # non-negative, positive when wrong


def test_mae_rmse_nonnegative_and_ordering():
    p, t = torch.randn(400), torch.randn(400)
    assert metrics.mae(p, t) >= 0.0
    assert metrics.rmse(p, t) >= 0.0
    # RMSE >= MAE always (power mean inequality).
    assert metrics.rmse(p, t) + 1e-9 >= metrics.mae(p, t)
    assert metrics.mae(t, t) < 1e-9 and metrics.rmse(t, t) < 1e-9


def test_correlation_bounds_and_anchors():
    x = torch.randn(300)
    assert abs(metrics.correlation(x, x) - 1.0) < 1e-6
    assert abs(metrics.correlation(x, -x) + 1.0) < 1e-6
    c = metrics.correlation(torch.randn(300), torch.randn(300))
    assert -1.0 <= c <= 1.0


def test_snr_and_esr_db_perfect_fit():
    t = torch.randn(200)
    assert math.isinf(metrics.segmental_snr_db(t, t))   # +inf
    assert metrics.esr_db(t, t) == float("-inf")        # -inf


def test_snr_decreases_with_noise():
    t = torch.randn(1000)
    little = t + 0.01 * torch.randn(1000)
    lots = t + 0.5 * torch.randn(1000)
    assert metrics.segmental_snr_db(little, t) > metrics.segmental_snr_db(lots, t)


def test_relative_error():
    assert metrics.relative_error(300.0, 300.0) < 1e-9
    assert abs(metrics.relative_error(150.0, 300.0) - 0.5) < 1e-9


def test_regression_metrics_schema():
    p, t = torch.randn(100), torch.randn(100)
    r = metrics.regression_metrics(p, t)
    assert set(r) == {"esr", "esr_db", "mae", "rmse", "snr_db"}


# --------------------------------------------------- synthetic ground truth
def test_sag_depth_positive_under_load():
    from power_sag.synthetic import SyntheticSagAmp

    amp = SyntheticSagAmp(fs=48000.0)
    loud = 0.9 * torch.sin(2 * torch.pi * 100 * torch.arange(4800) / 48000.0)
    _, _, V = amp.generate(loud.view(1, -1, 1))
    depth = metrics.sag_depth(V)
    assert depth > 1.0                        # meaningful droop (volts)
    assert depth <= float(V.reshape(-1)[0])   # can't droop below zero from start


def test_trajectory_metrics_on_identical_is_perfect():
    v = torch.linspace(415.0, 400.0, 500)
    m = metrics.trajectory_metrics(v, v)
    assert m["rmse_volts"] < 1e-6
    assert abs(m["correlation"] - 1.0) < 1e-6
    assert abs(m["sag_depth_pred"] - m["sag_depth_true"]) < 1e-6
