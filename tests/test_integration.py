"""Integration tests exercising several components together.

Where the ``test_<module>.py`` files check each component in isolation, these
tests wire the pieces into the flows they actually run in:

* the stateful data loader driving the real model and carrying ``V_B+`` across
  segment boundaries (the contract ``SequenceDataset`` exists to support),
* a full train step (dataset -> model -> loss -> backward -> optimiser),
* audio loaded from an on-disk WAV through the dataset,
* the evaluator's test signals pushed through the end-to-end model,
* the cabinet IR attached as the model's final stage.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from power_sag import load_config
from power_sag.data import AudioDataset, SequenceDataset
from power_sag.dsp import CabinetIR
from power_sag.evaluation import SagEvaluator
from power_sag.losses import ESRLoss, PreEmphasisLoss
from power_sag.nn import PowerSagModel


def small_config(seg_len=48):
    """A tiny but complete config so integration tests stay fast."""
    cfg = load_config()
    cfg["hidden_size"] = 8
    cfg["coupling_hidden"] = 8
    cfg["segment_len"] = seg_len
    return cfg


def build_model(cfg):
    torch.manual_seed(0)
    return PowerSagModel.from_config(cfg)


# --------------------------------------------------------------------------- #
# Data loader as a whole                                                      #
# --------------------------------------------------------------------------- #
def test_dataloader_batches_and_preserves_order():
    seg = 50
    n = seg * 6
    x = torch.arange(n, dtype=torch.float32)
    t = torch.arange(n, dtype=torch.float32)
    ds = SequenceDataset(x, t, segment_len=seg, normalize=False)

    # batch_size=1, no shuffle: the ordered, stateful usage pattern.
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    collected = []
    for inp, tgt, V0 in loader:
        assert inp.shape == (1, seg, 1)
        assert tgt.shape == (1, seg, 1)
        assert V0.shape == (1,)  # default collate stacks the scalar states
        collected.append(inp.squeeze())
    reconstructed = torch.cat(collected)
    assert torch.equal(reconstructed, x[: len(reconstructed)])


def test_stateful_loop_matches_monolithic_supply_trajectory():
    """Carrying V_B+ segment-by-segment must equal one unbroken ODE run.

    This is the reason ``SequenceDataset`` tracks per-segment initial states:
    the terminal V_B+ the model produces for segment N has to be exactly the
    initial V_B+ served for segment N+1, and the concatenation of the
    per-segment trajectories must reproduce the trajectory over the whole
    recording.
    """
    cfg = small_config(seg_len=40)
    model = build_model(cfg)
    V_idle = float(cfg["V_idle"])

    rng = np.random.default_rng(0)
    audio = rng.standard_normal(40 * 5).astype(np.float32)
    ds = SequenceDataset(
        audio, audio, segment_len=cfg["segment_len"], V_idle=V_idle, normalize=True
    )

    # Monolithic run over the (normalised) full recording.
    full = torch.cat([ds[i][0].squeeze(-1) for i in range(len(ds))]).view(1, -1, 1)
    with torch.no_grad():
        V_full, _ = model.supply_trajectory(full, V0=torch.tensor([[V_idle]]))

    # Segment-by-segment run, carrying state through the dataset.
    per_segment = []
    with torch.no_grad():
        for i in range(len(ds)):
            inp, _, V0 = ds[i]
            if i == 0:
                assert torch.allclose(V0, torch.tensor(V_idle))
            V_seq, V_final = model.supply_trajectory(
                inp.unsqueeze(0), V0=V0.view(1, 1)
            )
            per_segment.append(V_seq)
            ds.update_state(i, V_final)
            # The carry contract: next segment's served V0 == this terminal.
            if i + 1 < len(ds):
                assert torch.allclose(ds[i + 1][2], V_final.reshape(-1)[0], atol=1e-5)

    stitched = torch.cat(per_segment, dim=1)
    assert stitched.shape == V_full.shape
    assert torch.allclose(stitched, V_full, atol=1e-3)


def test_audio_dataset_wav_roundtrip(tmp_path):
    import soundfile as sf

    fs = 48000
    x = (0.5 * np.sin(2 * np.pi * 220 * np.arange(fs // 10) / fs)).astype(np.float32)
    t = (7.0 * x).astype(np.float32)  # louder "amp output"
    xp, tp = tmp_path / "di.wav", tmp_path / "amp.wav"
    sf.write(xp, x, fs)
    sf.write(tp, t, fs)

    ds = AudioDataset(str(xp), str(tp), normalize=True)
    inp, tgt = ds[0]
    assert inp.shape == tgt.shape
    assert inp.shape[-1] == 1
    assert abs(inp.abs().max().item() - 1.0) < 1e-4
    assert abs(tgt.abs().max().item() - 1.0) < 1e-4


# --------------------------------------------------------------------------- #
# Full training / evaluation flows                                            #
# --------------------------------------------------------------------------- #
def test_training_step_reduces_loss():
    cfg = small_config(seg_len=48)
    model = build_model(cfg)

    torch.manual_seed(1)
    x = 0.5 * torch.randn(1, cfg["segment_len"], 1)
    target = 0.5 * torch.randn(1, cfg["segment_len"], 1)

    esr = ESRLoss()
    preemph = PreEmphasisLoss(coeff=cfg["preemph_coeff"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def step():
        y = model(x)
        return esr(y, target) + cfg["preemph_weight"] * preemph(y, target)

    initial = step().item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = step()
        loss.backward()
        optimizer.step()
    final = step().item()

    assert final < initial  # the whole pipeline can actually learn


def test_full_model_on_evaluator_signal():
    cfg = small_config()
    model = build_model(cfg)
    ev = SagEvaluator(fs=cfg["fs"])

    signal = ev.sustained_chord_signal(0.01)  # ~480 samples at 48 kHz
    with torch.no_grad():
        y = model(signal)
        V_seq, _ = model.supply_trajectory(signal)

    assert y.shape == signal.shape
    assert torch.isfinite(y).all()
    # The physical state stays inside its bounds throughout.
    assert (V_seq >= 0.0).all()
    assert (V_seq <= cfg["V_oc"] + 1e-3).all()


def test_model_with_cabinet_stage():
    cfg = small_config(seg_len=32)
    torch.manual_seed(2)
    ir = torch.randn(16)
    cabinet = CabinetIR(ir)
    model = PowerSagModel.from_config(cfg, cabinet=cabinet)

    x = 0.5 * torch.randn(1, cfg["segment_len"], 1)
    target = 0.5 * torch.randn(1, cfg["segment_len"], 1)
    y = model(x)
    assert y.shape == x.shape  # cabinet convolution preserves length

    loss = ((y - target) ** 2).mean()
    loss.backward()
    # Trainable subsystems receive gradients; the fixed cabinet IR does not.
    assert model.cabinet.ir.grad is None
    trainable_grad = sum(
        p.grad.abs().sum()
        for p in model.parameters()
        if p.grad is not None
    )
    assert trainable_grad > 0
