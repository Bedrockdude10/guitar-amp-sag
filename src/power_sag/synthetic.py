"""Synthetic ground-truth amplifier for identifiability validation.

The whole project rests on a bet: that backpropagation through the unrolled
Euler ODE can *identify* a latent supply state (``V_B+``) and its physics
(``R_eff``, ``C1``) from the output audio alone.  Latent-variable ODE fitting
is prone to non-identifiability and bad minima, so we test the bet on synthetic
data -- generated from a *known* physics + coupling + nonlinearity -- before
recording a real amplifier.

``SyntheticSagAmp`` produces ``(input, output, V_B+_true)`` triples:

* load current from a physically-motivated envelope coupling
  ``I_load = I0 + alpha * env(x)^2`` (option 1 of the research notes),
* the supply voltage from the same explicit-Euler ODE the model uses, and
* a sag-dependent nonlinearity whose clipping headroom scales with ``V_B+``
  (supply droop compresses/softens the output -- the audible sag effect).

Because the ground-truth ``V_B+`` trajectory is returned, we can measure how
well a trained model recovers both the hidden state and its parameters.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .physics.ode import PowerSupplyODE


class SyntheticSagAmp:
    """A known-physics amplifier used to generate labelled sag data.

    Parameters
    ----------
    fs:
        Sample rate (Hz).
    V_oc, V_idle, C1, R_eff:
        Ground-truth power-supply parameters.
    I0:
        Quiescent load current (A).
    alpha:
        Envelope-to-current scale (A) -- how hard loud signal loads the supply.
    gain:
        Pre-clip drive of the nonlinearity.
    env_tau:
        Time constant (s) of the one-pole envelope follower on ``|x|``.
    """

    def __init__(
        self,
        fs: float = 48000.0,
        V_oc: float = 420.0,
        V_idle: float = 415.0,
        C1: float = 22e-6,
        R_eff: float = 300.0,
        I0: float = 0.02,
        alpha: float = 0.35,
        gain: float = 4.0,
        env_tau: float = 5e-3,
    ) -> None:
        self.fs = float(fs)
        self.V_idle = float(V_idle)
        self.V_oc = float(V_oc)
        self.I0 = float(I0)
        self.alpha = float(alpha)
        self.gain = float(gain)
        self.env_beta = float(np.exp(-1.0 / (env_tau * fs)))
        # Reuse the model's ODE as the ground-truth integrator (fixed params).
        self.ode = PowerSupplyODE(
            fs=fs,
            C1=C1,
            R_eff=R_eff,
            V_oc=V_oc,
            V_idle=V_idle,
            reff_mode="scalar",
            learn_R_eff=False,
            learn_C1=False,
            learn_V_oc=False,
        )

    # --------------------------------------------------------------- helpers
    def _envelope(self, x: torch.Tensor) -> torch.Tensor:
        """One-pole envelope follower on ``|x|`` (shape preserved).

        A one-pole IIR ``env[n] = beta*env[n-1] + (1-beta)*|x[n]|`` is a linear
        recursive filter, so it runs in vectorised C via ``scipy.signal.lfilter``
        along the time axis instead of a Python sample loop.
        """
        from scipy.signal import lfilter

        beta = self.env_beta
        rect = x.abs().detach().cpu().numpy()
        env = lfilter([1.0 - beta], [1.0, -beta], rect, axis=1)
        return torch.from_numpy(env.astype(np.float32)).to(x.device)

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the known amp on ``x``; return ``(x, y, V_true)``.

        ``x`` may be ``(T,)``, ``(B, T)`` or ``(B, T, 1)``; outputs are
        ``(B, T, 1)``.
        """
        if x.dim() == 1:
            x = x.view(1, -1, 1)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        x = x.to(torch.float32)

        env = self._envelope(x)
        V = self.ode.init_state(x.shape[0], x.device, x.dtype)
        V_states = torch.zeros_like(x)
        y = torch.zeros_like(x)
        for n in range(x.shape[1]):
            V_states[:, n, :] = V
            headroom = V / self.V_idle  # sag -> less headroom -> more compression
            y[:, n, :] = headroom * torch.tanh(self.gain * x[:, n, :])
            I_n = self.I0 + self.alpha * env[:, n, :] ** 2
            V = self.ode.step(V, I_n)
        return x, y, V_states

    def generate_excitation(
        self, duration: float, seed: Optional[int] = None
    ) -> torch.Tensor:
        """A sag-exercising test input: bursts of tone at varying amplitude.

        Alternates loud and quiet segments so the supply repeatedly droops and
        recovers -- the trajectory identifiability needs.
        """
        rng = np.random.default_rng(seed)
        n = int(round(duration * self.fs))
        t = np.arange(n) / self.fs
        f0 = 100.0
        tone = np.sin(2 * np.pi * f0 * t).astype(np.float32)
        # Amplitude envelope: random piecewise levels, ~100 ms blocks.
        block = max(int(0.1 * self.fs), 1)
        amps = rng.uniform(0.1, 1.0, size=(n // block + 1,)).astype(np.float32)
        amp_env = np.repeat(amps, block)[:n]
        signal = tone * amp_env
        return torch.from_numpy(signal).view(1, n, 1)
