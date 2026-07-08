"""Tests for the sag evaluation protocol."""

import numpy as np
import torch

from power_sag.evaluation import SagEvaluator


def test_sustained_chord_duration():
    fs = 48000.0
    ev = SagEvaluator(fs=fs)
    duration = 0.25
    sig = ev.sustained_chord_signal(duration)
    expected = int(round(duration * fs))
    assert sig.shape == (1, expected, 1)


def test_fit_recovery_curve_returns_positive_tau():
    ev = SagEvaluator(fs=1000.0)
    t = np.arange(500) / 1000.0
    y = np.exp(-t / 0.05)
    tau = ev.fit_recovery_curve(torch.tensor(y), fs=1000.0)
    assert tau > 0.0


def test_fit_recovery_curve_recovers_known_tau():
    fs = 2000.0
    ev = SagEvaluator(fs=fs)
    true_tau = 0.02
    t = np.arange(1000) / fs
    y = 3.0 * np.exp(-t / true_tau) + 0.5
    tau = ev.fit_recovery_curve(torch.tensor(y), fs=fs)
    rel_err = abs(tau - true_tau) / true_tau
    assert rel_err < 0.05
