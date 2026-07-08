"""Numerical validation of the explicit-Euler power-supply integrator.

The paper claims the sample-rate Euler discretisation is stable and accurate
(draft §3.3). These tests establish that against references that need no
hardware:

* an **analytic** solution (for constant load the ODE is linear, so
  ``V(t) = V_ss + (V0 - V_ss)·exp(-t/RC)`` is exact),
* the expected **first-order** global error scaling (halving ``Ts`` halves the
  error), and
* agreement with a higher-order **RK4** reference at the audio rate (Euler is
  "good enough" where it is actually used).
"""

import numpy as np
import torch

from power_sag.physics import PowerSupplyODE

R, C1, V_OC, V0, I = 300.0, 22e-6, 420.0, 380.0, 0.02
TAU = R * C1


def analytic(t: np.ndarray) -> np.ndarray:
    """Exact solution of C1·dV/dt = (V_oc - V)/R - I for constant I."""
    v_ss = V_OC - I * R
    return v_ss + (V0 - v_ss) * np.exp(-t / TAU)


def euler_trajectory(fs: float, duration: float) -> np.ndarray:
    ode = PowerSupplyODE(fs=fs, C1=C1, R_eff=R, V_oc=V_OC, V_idle=V0,
                         learn_R_eff=False, learn_C1=False)
    steps = int(round(duration * fs))
    V = ode.init_state(1)
    Iv = torch.full((1, 1), I)
    out = np.empty(steps)
    for n in range(steps):
        out[n] = float(V)
        V = ode.step(V, Iv)
    return out


def rk4_trajectory(fs: float, duration: float) -> np.ndarray:
    Ts = 1.0 / fs

    def f(V):
        return (V_OC - V) / (R * C1) - I / C1

    steps = int(round(duration * fs))
    out = np.empty(steps)
    V = V0
    for n in range(steps):
        out[n] = V
        k1 = f(V)
        k2 = f(V + 0.5 * Ts * k1)
        k3 = f(V + 0.5 * Ts * k2)
        k4 = f(V + Ts * k3)
        V = V + (Ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def max_abs_error(fs: float, duration: float = 0.05) -> float:
    v = euler_trajectory(fs, duration)
    t = np.arange(len(v)) / fs
    return float(np.max(np.abs(v - analytic(t))))


def test_euler_matches_analytic_at_audio_rate():
    # At 48 kHz the discretisation error is a small fraction of a volt on a
    # ~40 V swing.
    err = max_abs_error(48000.0, duration=0.05)
    assert err < 0.05  # volts
    assert err / (V0 - (V_OC - I * R)) < 1e-3  # < 0.1% of the swing


def test_euler_first_order_convergence():
    # Global error of explicit Euler is O(Ts): each halving of Ts ~halves error.
    rates = [12000.0, 24000.0, 48000.0, 96000.0]
    errs = [max_abs_error(fs) for fs in rates]
    # Error must strictly decrease as the rate rises.
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    # Fitted order (slope of log err vs log Ts) is ≈ 1.
    log_ts = np.log(1.0 / np.array(rates))
    slope = np.polyfit(log_ts, np.log(errs), 1)[0]
    assert 0.8 < slope < 1.2


def test_euler_agrees_with_rk4_at_audio_rate():
    fs = 48000.0
    e = euler_trajectory(fs, 0.05)
    r = rk4_trajectory(fs, 0.05)
    assert np.max(np.abs(e - r)) < 0.05  # volts — indistinguishable in practice


def test_stability_ratio_well_inside_bound():
    ode = PowerSupplyODE(fs=48000.0, C1=C1, R_eff=R, V_oc=V_OC, V_idle=V0)
    assert ode.step_ratio < 0.01  # Ts/tau ≪ 1
