"""Tests for the audio and stateful sequence datasets."""

import torch

from power_sag.data import AudioDataset, SequenceBatchSampler, SequenceDataset


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


# --------------------------------------------------------------------------- #
# Multi-recording state boundaries                                            #
# --------------------------------------------------------------------------- #
def _multi_recording_ds(seg=100, lens=(300, 200)):
    recs = [
        (torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32))
        for n in lens
    ]
    return SequenceDataset(
        segment_len=seg, V_idle=415.0, normalize=False, recordings=recs
    )


def test_multiple_recordings_tracked_separately():
    ds = _multi_recording_ds(seg=100, lens=(300, 200))
    # 3 segments from recording 0, 2 from recording 1.
    assert ds.sequences == [[0, 1, 2], [3, 4]]
    assert ds.recording_of == [0, 0, 0, 1, 1]


def test_update_state_does_not_cross_recording_boundary():
    ds = _multi_recording_ds(seg=100, lens=(300, 200))
    # Last segment of recording 0 is index 2; index 3 is recording 1's first.
    ds.update_state(2, torch.tensor(390.0))
    assert torch.allclose(ds[3][2], torch.tensor(415.0))  # unchanged: new recording
    # Within recording 0, the carry does apply.
    ds.update_state(0, torch.tensor(388.0))
    assert torch.allclose(ds[1][2], torch.tensor(388.0))


def test_carried_state_is_detached():
    """V_B+ carried across a boundary must be detached (truncated BPTT)."""
    ds = _multi_recording_ds(seg=100, lens=(300,))
    # A tensor that still carries a grad graph, as a model would produce.
    p = torch.tensor(400.0, requires_grad=True)
    V_terminal = (p * 1.01).reshape(1, 1)  # has grad_fn
    assert V_terminal.requires_grad and V_terminal.grad_fn is not None
    ds.update_state(0, V_terminal)
    stored = ds[1][2]
    assert not stored.requires_grad
    assert stored.grad_fn is None


# --------------------------------------------------------------------------- #
# Sequence-level batch sampler                                                #
# --------------------------------------------------------------------------- #
def test_sampler_covers_every_segment_once():
    ds = _multi_recording_ds(seg=50, lens=(500, 300, 200))
    sampler = SequenceBatchSampler(ds, batch_size=2, shuffle=True, seed=0)
    emitted = [idx for batch in sampler for idx in batch]
    assert sorted(emitted) == list(range(len(ds)))


def test_sampler_preserves_recording_order_and_keeps_recordings_intact():
    ds = _multi_recording_ds(seg=50, lens=(500, 300, 200, 150))
    for shuffle in (False, True):
        sampler = SequenceBatchSampler(ds, batch_size=3, shuffle=shuffle, seed=1)
        lanes = sampler._build_lanes()
        # Every recording's segments stay contiguous, in order, within one lane.
        for seq in ds.sequences:
            in_lane = [lane for lane in lanes if seq[0] in lane]
            assert len(in_lane) == 1
            lane = in_lane[0]
            start = lane.index(seq[0])
            assert lane[start : start + len(seq)] == seq
        # Lanes partition all segments.
        assert sorted(i for lane in lanes for i in lane) == list(range(len(ds)))


def test_sampler_batch_size_one_is_sequential():
    ds = _multi_recording_ds(seg=100, lens=(300, 200))
    sampler = SequenceBatchSampler(ds, batch_size=1, shuffle=False)
    emitted = [idx for batch in sampler for idx in batch]
    assert emitted == list(range(len(ds)))  # all in order in a single lane
