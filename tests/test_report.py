"""Tests for structured reports: schema presence + value ranges."""

import json

import torch

from power_sag import load_config
from power_sag.nn import PowerSagModel, UnconditionedLSTM
from power_sag.reporting import report as rp
from power_sag.synthetic import SyntheticSagAmp


def _small_model():
    cfg = load_config()
    cfg.update(hidden_size=8, coupling_hidden=8)
    return PowerSagModel.from_config(cfg)


def _assert_regression_ranges(reg):
    assert reg["esr"] >= 0.0
    assert reg["mae"] >= 0.0 and reg["rmse"] >= 0.0
    assert reg["rmse"] + 1e-9 >= reg["mae"]
    assert reg["esr_db"] <= 0.0 or reg["esr"] >= 1.0  # dB sign tracks esr vs 1


def test_regression_report_schema_and_ranges():
    model = _small_model()
    x, t = 0.5 * torch.randn(1, 400, 1), 0.5 * torch.randn(1, 400, 1)
    reg = rp.regression_report(model, x, t)
    assert set(reg) == {"esr", "esr_db", "mae", "rmse", "snr_db"}
    _assert_regression_ranges(reg)


def test_synthetic_report_has_all_blocks_and_valid_ranges():
    model = _small_model()
    amp = SyntheticSagAmp(fs=8000.0)  # low fs keeps signals short
    rep = rp.synthetic_report(model, amp=amp, duration=0.1)

    assert set(rep) >= {"regression", "trajectory", "sag_protocol"}
    _assert_regression_ranges(rep["regression"])

    traj = rep["trajectory"]
    assert -1.0 <= traj["correlation"] <= 1.0
    assert traj["rmse_volts"] >= 0.0
    assert traj["sag_depth_true"] >= 0.0

    tau = rep["sag_protocol"]["recovery_tau"]["tau_s"]
    assert tau > 0.0


def test_baseline_report_omits_trajectory_block():
    """A black-box baseline has no supply_trajectory, so no trajectory block."""
    model = UnconditionedLSTM(hidden_size=8)
    amp = SyntheticSagAmp(fs=8000.0)
    rep = rp.synthetic_report(model, amp=amp, duration=0.1)
    assert "trajectory" not in rep
    assert "regression" in rep and "sag_protocol" in rep


def test_report_is_json_serialisable(tmp_path):
    model = _small_model()
    amp = SyntheticSagAmp(fs=8000.0)
    rep = rp.synthetic_report(model, amp=amp, duration=0.1)
    path = rp.to_json(rep, str(tmp_path / "report.json"))
    loaded = json.load(open(path))
    assert loaded["regression"]["esr"] == rep["regression"]["esr"]
