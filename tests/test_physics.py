"""Tests for the differentiable power-supply ODE."""

import math

import torch

from power_sag.physics import PowerSupplyODE


def make_ode(**overrides):
    defaults = dict(
        fs=48000.0,
        C1=22e-6,
        R_eff=300.0,
        V_oc=420.0,
        V_idle=415.0,
    )
    defaults.update(overrides)
    return PowerSupplyODE(**defaults)


def test_zero_load_converges_monotonically_to_voc():
    ode = make_ode(V_idle=380.0)
    V = ode.init_state(1)
    I = torch.zeros(1, 1)
    prev = V.clone()
    for _ in range(5000):
        V = ode.step(V, I)
        # Monotonically non-decreasing and never overshooting V_oc.
        assert (V >= prev - 1e-5).all()
        assert (V <= 420.0 + 1e-4).all()
        prev = V.clone()
    assert torch.allclose(V, torch.tensor(420.0), atol=1e-1)


def test_constant_load_steady_state_ohms_law():
    ode = make_ode()
    I_val = 0.05  # 50 mA
    I = torch.full((1, 1), I_val)
    V = ode.init_state(1)
    for _ in range(5000):  # >> tau (~316 samples), fully converged
        V = ode.step(V, I)
    expected = 420.0 - I_val * 300.0  # V_oc - I * R_eff
    assert torch.allclose(V, torch.tensor(expected), atol=1e-1)


def test_step_size_stability_ratio():
    ode = make_ode()
    assert ode.step_ratio < 0.01


def test_state_is_bounded():
    ode = make_ode(V_idle=415.0)
    V = ode.init_state(4)
    # Alternating huge and zero load; state must stay within [0, V_oc].
    for n in range(20000):
        I = torch.full((4, 1), 10.0 if n % 2 == 0 else 0.0)
        V = ode.step(V, I)
        assert (V <= 420.0 + 1e-4).all()
        assert (V >= 0.0 - 1e-4).all()


def test_backward_produces_nonzero_gradients():
    ode = make_ode()
    I = torch.full((2, 100, 1), 0.02)
    V_seq, V_final = ode(I)
    loss = ((V_seq - 400.0) ** 2).mean()
    loss.backward()
    assert ode._R_eff.grad is not None and ode._R_eff.grad.abs() > 0
    assert ode.C1.grad is not None and ode.C1.grad.abs() > 0


def test_time_constant_matches_rc():
    ode = make_ode(V_idle=380.0)
    tau = 300.0 * 22e-6  # R_eff * C1
    n_steps = int(round(tau * ode.fs))
    V = ode.init_state(1)
    I = torch.zeros(1, 1)
    for _ in range(n_steps):
        V = ode.step(V, I)
    target = 380.0 + (1.0 - 1.0 / math.e) * (420.0 - 380.0)
    rel_err = abs(V.detach().item() - target) / target
    assert rel_err < 0.05


def test_nonlinear_reff_increases_with_current():
    ode = make_ode(reff_mode="nonlinear", R0=250.0, R1=100.0, k=1.0)
    r_low = ode.r_eff(torch.tensor([[0.0]]))
    r_high = ode.r_eff(torch.tensor([[1.0]]))
    assert (r_high > r_low).all()
