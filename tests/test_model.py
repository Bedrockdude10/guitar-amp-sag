"""Tests for the full end-to-end gray-box model."""

import torch

from power_sag import load_config
from power_sag.nn import PowerSagModel


def build_model():
    cfg = load_config()
    # Keep it small/fast for tests but keep V_oc learnable to exercise its grad.
    cfg["hidden_size"] = 8
    cfg["coupling_hidden"] = 8
    cfg["learn_V_oc"] = True
    return PowerSagModel.from_config(cfg)


def test_end_to_end_forward_shape():
    model = build_model()
    x = torch.randn(2, 64, 1) * 0.5
    y = model(x)
    assert y.shape == (2, 64, 1)


def test_backward_runs():
    model = build_model()
    x = torch.randn(1, 64, 1) * 0.5
    target = torch.randn(1, 64, 1) * 0.5
    y = model(x)
    loss = ((y - target) ** 2).mean()
    loss.backward()  # must not raise


def test_all_component_gradients_nonzero():
    model = build_model()
    x = torch.randn(2, 80, 1) * 0.5
    target = torch.randn(2, 80, 1) * 0.5
    y = model(x)
    loss = ((y - target) ** 2).mean()
    loss.backward()

    # Physics parameters (positive ones are learned in log space).
    assert model.physics._log_R_eff.grad is not None and model.physics._log_R_eff.grad.abs() > 0
    assert model.physics._log_C1.grad is not None and model.physics._log_C1.grad.abs() > 0
    assert model.physics.V_oc.grad is not None and model.physics.V_oc.grad.abs() > 0

    # Coupling network parameters.
    coupling_grad = sum(
        p.grad.abs().sum() for p in model.coupling.parameters() if p.grad is not None
    )
    assert coupling_grad > 0

    # Audio path parameters.
    audio_grad = sum(
        p.grad.abs().sum() for p in model.audio.parameters() if p.grad is not None
    )
    assert audio_grad > 0


def test_scripted_recurrence_matches_eager():
    model = build_model()
    x = 0.5 * torch.randn(1, 200, 1)
    V0 = torch.full((1, 1), 415.0)

    n_params_before = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        V_eager, F_eager = model.supply_trajectory(x, V0)
    model.enable_script()
    with torch.no_grad():
        V_script, F_script = model.supply_trajectory(x, V0)

    assert torch.allclose(V_eager, V_script, atol=1e-4)
    assert torch.allclose(F_eager, F_script, atol=1e-4)
    # Enabling the scripted path must not add (double-counted) parameters.
    assert sum(p.numel() for p in model.parameters()) == n_params_before


def test_scripted_path_still_trains():
    model = build_model()
    model.enable_script()
    x = 0.5 * torch.randn(1, 64, 1)
    target = 0.5 * torch.randn(1, 64, 1)
    y = model(x)
    ((y - target) ** 2).mean().backward()
    # Gradients still reach the physics parameters through the scripted loop.
    assert model.physics._log_R_eff.grad is not None and model.physics._log_R_eff.grad.abs() > 0
    assert sum(
        p.grad.abs().sum() for p in model.coupling.parameters() if p.grad is not None
    ) > 0


def test_lstm_state_threading_matches_monolithic():
    """Carrying (V_B+, LSTM state) across chunks reproduces one unbroken pass.

    This underpins truncated BPTT: chunking changes only the gradient graph,
    never the forward values.
    """
    model = build_model()
    model.eval()
    x = 0.5 * torch.randn(1, 90, 1)

    with torch.no_grad():
        y_full = model(x)
        # Process in three chunks, threading both slow and fast state.
        ys, V, state = [], None, None
        for start in range(0, 90, 30):
            chunk = x[:, start : start + 30]
            y_c, V, state = model(chunk, V0=V, lstm_state=state, return_state=True)
            ys.append(y_c)
    y_chunked = torch.cat(ys, dim=1)
    assert torch.allclose(y_full, y_chunked, atol=1e-5)
