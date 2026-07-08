"""Tests for capture-readiness tooling: alignment, splitting, supervised V_B+."""

import numpy as np
import torch

from power_sag.data import SequenceDataset, chronological_split
from power_sag.utils import align_signals, estimate_delay


def test_estimate_delay_recovers_known_lag():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000).astype(np.float32)
    delay = 37
    t = np.concatenate([np.zeros(delay, dtype=np.float32), x])[: len(x)]  # x delayed
    assert estimate_delay(x, t, max_lag=200) == delay


def test_align_signals_compensates_latency():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4000).astype(np.float32)
    delay = 50
    t = np.concatenate([np.zeros(delay, dtype=np.float32), x])[: len(x)]
    ax, at, d = align_signals(x, t, max_lag=200)
    assert d == delay
    assert len(ax) == len(at)
    # After alignment the two are sample-aligned (target == input again).
    n = min(len(ax), len(at))
    corr = np.corrcoef(ax[:n].numpy(), at[:n].numpy())[0, 1]
    assert corr > 0.99


def test_chronological_split_fractions_and_order():
    x = torch.arange(1000, dtype=torch.float32)
    t = torch.arange(1000, dtype=torch.float32)
    (xtr, ttr), (xv, tv), (xte, tte) = chronological_split(x, t, val_frac=0.2, test_frac=0.1)
    assert len(xtr) == 700 and len(xv) == 200 and len(xte) == 100
    # Chronological: train precedes val precedes test (no shuffling).
    assert xtr[-1] < xv[0] < xte[0]
    # Reassembly reproduces the original signal in order.
    assert torch.equal(torch.cat([xtr, xv, xte]), x)


def test_supervised_vb_segments_available():
    n = 500
    x = torch.randn(n)
    t = torch.randn(n)
    vb = torch.linspace(415.0, 395.0, n)  # a measured B+ droop
    ds = SequenceDataset(
        x, t, segment_len=100, normalize=True, measured_vb=vb
    )
    assert ds.has_measured_vb
    seg = ds.measured_vb_segment(0)
    assert seg is not None and seg.shape == (100, 1)
    # V_B+ is not normalised (kept in volts) and matches the source slice.
    assert torch.allclose(seg.squeeze(-1), vb[:100])


def test_no_measured_vb_by_default():
    ds = SequenceDataset(torch.randn(300), torch.randn(300), segment_len=100)
    assert not ds.has_measured_vb
    assert ds.measured_vb_segment(0) is None
