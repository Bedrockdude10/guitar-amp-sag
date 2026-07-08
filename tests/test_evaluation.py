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


def test_rms_envelope_tracks_amplitude():
    ev = SagEvaluator(fs=48000.0)
    t = torch.arange(4800) / 48000.0
    quiet = 0.1 * torch.sin(2 * torch.pi * 200 * t)
    loud = 0.9 * torch.sin(2 * torch.pi * 200 * t)
    sig = torch.cat([quiet, loud]).view(1, -1, 1)
    env = ev.rms_envelope(sig, win_ms=5.0)
    assert env.numel() == sig.numel()
    assert env[-1] > env[:4800].max()  # louder half has bigger envelope


def _identity_model(x):
    # A trivial model-under-test with the model(x) -> y convention.
    return x


def test_run_protocol_returns_all_four_tests():
    ev = SagEvaluator(fs=8000.0)  # low fs keeps the synthetic signals short
    result = ev.run_protocol(_identity_model)
    assert set(result) == {"attack_bloom", "recovery_tau", "presag_vs_cold", "quiet_to_loud"}
    assert result["recovery_tau"]["tau_s"] > 0.0
    assert "presag_over_cold" in result["presag_vs_cold"]


def test_protocol_on_real_model_runs():
    from power_sag import load_config
    from power_sag.nn import PowerSagModel

    cfg = load_config()
    cfg.update(hidden_size=8, coupling_hidden=8)
    model = PowerSagModel.from_config(cfg)
    ev = SagEvaluator(fs=8000.0)
    bloom = ev.attack_bloom(model, duration=0.05)
    assert bloom["attack_rms"] >= 0.0 and bloom["settle_rms"] >= 0.0
