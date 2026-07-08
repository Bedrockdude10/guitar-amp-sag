"""Tests for the audio and stateful sequence datasets."""

import torch

from power_sag.data import AudioDataset, SequenceDataset


def test_segments_returned_in_order():
    n = 1000
    x = torch.arange(n, dtype=torch.float32)
    t = torch.arange(n, dtype=torch.float32)
    ds = SequenceDataset(x, t, segment_len=100, normalize=False, drop_last=True)
    reconstructed = torch.cat([ds[i][0].squeeze(-1) for i in range(len(ds))])
    assert torch.equal(reconstructed, x[: len(reconstructed)])


def test_segment_length_matches_config():
    x = torch.randn(2500)
    t = torch.randn(2500)
    seg = 500
    ds = SequenceDataset(x, t, segment_len=seg, normalize=False)
    for i in range(len(ds)):
        inp, tgt, _ = ds[i]
        assert inp.shape[0] == seg
        assert tgt.shape[0] == seg


def test_terminal_state_carries_to_next_segment():
    x = torch.randn(1000)
    t = torch.randn(1000)
    ds = SequenceDataset(x, t, segment_len=100, V_idle=415.0, normalize=False)
    # First segment starts at idle.
    assert torch.allclose(ds[0][2], torch.tensor(415.0))
    # After processing segment 0, its terminal state seeds segment 1.
    terminal = torch.tensor(402.5)
    ds.update_state(0, terminal)
    assert torch.allclose(ds[1][2], terminal)


def test_audio_normalized_to_unit_range():
    x = torch.randn(2000) * 12.0
    t = torch.randn(2000) * 7.0
    ds = SequenceDataset(x, t, segment_len=500, normalize=True)
    for i in range(len(ds)):
        inp, tgt, _ = ds[i]
        assert inp.abs().max() <= 1.0 + 1e-6
        assert tgt.abs().max() <= 1.0 + 1e-6


def test_audio_dataset_normalization():
    x = torch.randn(500) * 9.0
    t = torch.randn(500) * 4.0
    ds = AudioDataset(x, t, normalize=True)
    inp, tgt = ds[0]
    assert abs(inp.abs().max().item() - 1.0) < 1e-6
    assert abs(tgt.abs().max().item() - 1.0) < 1e-6
    assert inp.shape[-1] == 1
