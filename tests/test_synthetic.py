"""Tests for the synthetic ground-truth amplifier and identifiability signal."""

import torch

from power_sag import load_config
from power_sag.nn import PowerSagModel
from power_sag.synthetic import SyntheticSagAmp


def test_generate_shapes_and_range():
    amp = SyntheticSagAmp(fs=48000.0)
    x = 0.6 * torch.randn(1, 500, 1)
    xo, y, V = amp.generate(x)
    assert xo.shape == y.shape == V.shape == (1, 500, 1)
    assert torch.isfinite(y).all()
    # V_B+ stays within physical bounds.
    assert (V <= amp.V_oc + 1e-3).all()
    assert (V >= 0.0).all()


def test_supply_sags_under_load():
    """A loud sustained input must droop V_B+ below idle."""
    amp = SyntheticSagAmp(fs=48000.0)
    loud = 0.9 * torch.sin(2 * torch.pi * 100 * torch.arange(4800) / 48000.0)
    _, _, V = amp.generate(loud.view(1, -1, 1))
    assert V[0, 0, 0].item() > V[0, -1, 0].item()  # drooped over time
    assert V.min().item() < amp.V_idle - 1.0  # meaningful sag


def test_sag_is_load_dependent():
    """Louder input must sag more than quiet input (the effect is real)."""
    amp = SyntheticSagAmp(fs=48000.0)
    t = torch.arange(4800) / 48000.0
    tone = torch.sin(2 * torch.pi * 100 * t).view(1, -1, 1)
    _, _, V_quiet = amp.generate(0.1 * tone)
    _, _, V_loud = amp.generate(0.9 * tone)
    assert V_loud.min().item() < V_quiet.min().item()


def test_excitation_signal_shape():
    amp = SyntheticSagAmp(fs=48000.0)
    sig = amp.generate_excitation(0.05, seed=0)
    assert sig.shape == (1, int(0.05 * 48000), 1)
    assert sig.abs().max() <= 1.0 + 1e-6


def test_model_fits_synthetic_data():
    """A short fit on synthetic data reduces loss and finds *a* sag state.

    Full parameter recovery is a research question explored by
    scripts/validate_synthetic.py; here we assert the pipeline can learn on
    ground-truth sag data at all, and that the learned latent V_B+ responds to
    input energy (drooping under load), i.e. the model discovers the hidden
    state rather than ignoring it.
    """
    torch.manual_seed(0)
    amp = SyntheticSagAmp(fs=48000.0)
    # A clear quiet->loud input so the true V_B+ has strong, learnable dynamics.
    t = torch.arange(1200) / 48000.0
    tone = torch.sin(2 * torch.pi * 100 * t)
    amp_env = torch.cat([torch.full((600,), 0.2), torch.full((600,), 0.9)])
    x = (tone * amp_env).view(1, -1, 1)
    _, y, V_true = amp.generate(x)

    cfg = load_config()
    cfg.update(hidden_size=8, coupling_hidden=8)
    model = PowerSagModel.from_config(cfg)
    model.enable_script()  # ~4x faster inner loop keeps the test snappy
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    def loss_fn():
        pred = model(x)
        return ((pred - y) ** 2).mean()

    initial = loss_fn().item()
    for _ in range(30):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    assert loss_fn().item() < initial

    # The learned latent state should correlate with the true one (it tracks
    # the same load-driven droop), even if absolute scale differs.
    with torch.no_grad():
        V_pred, _ = model.supply_trajectory(x)
    a = (V_pred - V_pred.mean()).reshape(-1)
    b = (V_true - V_true.mean()).reshape(-1)
    corr = (a @ b) / (a.norm() * b.norm() + 1e-9)
    assert corr.item() > 0.3
