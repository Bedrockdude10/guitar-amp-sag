"""Differentiable power-supply ODE (the *slow* physics subsystem).

The B+ supply voltage evolves according to a first-order RC circuit driven by
the load current drawn by the power tubes::

    C1 * dV_B+/dt = (V_oc - V_B+) / R_eff(I_load) - I_load

Discretised at sample rate with explicit Euler (Ts = 1/fs)::

    V_B+[n+1] = V_B+[n] + (Ts / C1) * ((V_oc - V_B+[n]) / R_eff - I_load[n])

Every operation is a plain PyTorch tensor op, so autograd differentiates
through the unrolled Euler steps (BPTT) without a custom ``autograd.Function``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def euler_step(
    V: torch.Tensor,
    I_load: torch.Tensor,
    Ts: float,
    C1: torch.Tensor,
    V_oc: torch.Tensor,
    R_eff: torch.Tensor,
) -> torch.Tensor:
    """One explicit-Euler update of the power-supply ODE, clamped to ``[0, V_oc]``.

    This is the single source of truth for the discretised update: both the
    eager :meth:`PowerSupplyODE.step` and the optional TorchScript recurrence
    (:mod:`power_sag.nn.recurrence`) call it, so the two paths cannot drift
    apart.  It is a plain function -- TorchScript compiles it automatically
    when it is reached from a scripted module.

    Note: the clamp zeroes the local gradient whenever ``V_B+`` saturates at a
    bound.  In normal operation neither bound is reached -- with ``I_load >= 0``
    (guaranteed by the coupling's softplus) ``V_B+`` approaches ``V_oc`` from
    below and only nears ``0`` under pathological current draw -- so physics
    gradients are unaffected in practice.
    """
    dV = (Ts / C1) * ((V_oc - V) / R_eff - I_load)
    V_next = torch.minimum(V + dV, V_oc * torch.ones_like(V))
    return torch.clamp(V_next, min=0.0)


class PowerSupplyODE(nn.Module):
    """Explicit-Euler power-supply model with a GZ34-style ``R_eff`` option.

    Parameters
    ----------
    fs:
        Sample rate in Hz (sets the Euler step ``Ts = 1/fs``).
    C1:
        First filter capacitance in farads.
    R_eff:
        Effective source resistance in ohms (scalar mode) / ``R0`` seed.
    V_oc:
        Open-circuit supply voltage in volts.
    V_idle:
        Quiescent supply voltage used to initialise the state.
    reff_mode:
        ``"scalar"`` for a single learnable resistance, or ``"nonlinear"`` for
        the current-dependent GZ34 model ``R_eff(I) = R0 + R1 * softplus(I*k)``.
    learn_R_eff, learn_C1, learn_V_oc:
        Whether the corresponding physical parameter is a learnable
        ``nn.Parameter`` (``True``) or a fixed buffer (``False``).

    Notes
    -----
    The strictly-positive parameters (``C1``, ``R_eff``/``R0``/``R1``) are stored
    and learned in **log space**.  Their physical values span many orders of
    magnitude (``C1 ~ 2e-5`` F, ``R_eff ~ 3e2`` ohm), and an optimiser like Adam
    takes steps of roughly ``lr`` in raw parameter units regardless of scale --
    which blows up the tiny capacitance while barely moving the large
    resistance.  Learning ``log C1`` etc. puts every parameter on an ``O(1)``,
    multiplicative footing (and guarantees positivity for free).  The physical
    values remain available through the ``C1``, ``R_eff``, ``R0``, ``R1``
    properties, so callers and the ODE math are unchanged.
    """

    def __init__(
        self,
        fs: float = 48000.0,
        C1: float = 22e-6,
        R_eff: float = 300.0,
        V_oc: float = 420.0,
        V_idle: float = 415.0,
        reff_mode: str = "scalar",
        R0: float = 250.0,
        R1: float = 100.0,
        k: float = 1.0,
        learn_R_eff: bool = True,
        learn_C1: bool = True,
        learn_V_oc: bool = False,
    ) -> None:
        super().__init__()
        if reff_mode not in ("scalar", "nonlinear"):
            raise ValueError(f"unknown reff_mode: {reff_mode!r}")

        self.fs = float(fs)
        self.Ts = 1.0 / float(fs)
        self.reff_mode = reff_mode
        self.V_idle = float(V_idle)

        # Positive parameters learned in log space (see class docstring).
        self._register_log("_log_C1", C1, learn_C1)
        self._register_raw("V_oc", V_oc, learn_V_oc)

        if reff_mode == "scalar":
            self._register_log("_log_R_eff", R_eff, learn_R_eff)
        else:
            self._register_log("_log_R0", R0, learn_R_eff)
            self._register_log("_log_R1", R1, learn_R_eff)
            self._register_raw("k", k, learn_R_eff)

    # ------------------------------------------------------------------ utils
    def _register_raw(self, name: str, value: float, learnable: bool) -> None:
        tensor = torch.tensor(float(value), dtype=torch.float32)
        if learnable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

    def _register_log(self, name: str, value: float, learnable: bool) -> None:
        tensor = torch.log(torch.tensor(float(value), dtype=torch.float32))
        if learnable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

    # Physical values recovered from their log-space parameters.
    @property
    def C1(self) -> torch.Tensor:
        return torch.exp(self._log_C1)

    @property
    def _R_eff(self) -> torch.Tensor:  # scalar mode
        return torch.exp(self._log_R_eff)

    @property
    def R0(self) -> torch.Tensor:
        return torch.exp(self._log_R0)

    @property
    def R1(self) -> torch.Tensor:
        return torch.exp(self._log_R1)

    @property
    def tau(self) -> torch.Tensor:
        """Nominal recovery time constant ``tau = R_eff * C1`` (seconds)."""
        r = self._R_eff if self.reff_mode == "scalar" else self.R0
        return (r * self.C1).detach()

    @property
    def step_ratio(self) -> float:
        """Stability ratio ``Ts / tau`` (must be << 1 for explicit Euler)."""
        return float(self.Ts / self.tau)

    # ------------------------------------------------------------- dynamics
    def r_eff(self, I_load: torch.Tensor) -> torch.Tensor:
        """Effective source resistance for the given load current."""
        if self.reff_mode == "scalar":
            return self._R_eff
        # Current-dependent GZ34 model: resistance rises with demanded current.
        return self.R0 + self.R1 * F.softplus(I_load * self.k)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Initial B+ state (``V_idle``) with shape ``(batch_size, 1)``."""
        dtype = dtype or self.C1.dtype
        device = device or self.C1.device
        return torch.full((batch_size, 1), self.V_idle, device=device, dtype=dtype)

    def step(self, V: torch.Tensor, I_load: torch.Tensor) -> torch.Tensor:
        """Single explicit-Euler update ``V_B+[n] -> V_B+[n+1]``.

        The next voltage is clamped to ``[0, V_oc]`` so the state stays
        physically bounded (it can never charge above the open-circuit voltage
        nor fall below ground).
        """
        R = self.r_eff(I_load)
        return euler_step(V, I_load, self.Ts, self.C1, self.V_oc, R)

    def forward(
        self,
        I_load: torch.Tensor,
        V0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unroll the ODE over a load-current sequence.

        Parameters
        ----------
        I_load:
            Load current, shape ``(batch, seq_len, 1)``.
        V0:
            Optional initial state ``(batch, 1)``; defaults to ``V_idle``.

        Returns
        -------
        (V_seq, V_final):
            ``V_seq`` has shape ``(batch, seq_len, 1)`` where ``V_seq[:, n]`` is
            the state *before* applying ``I_load[:, n]``.  ``V_final`` is the
            terminal state ``V_B+[seq_len]`` for stateful continuation.
        """
        if I_load.dim() != 3 or I_load.size(-1) != 1:
            raise ValueError("I_load must have shape (batch, seq_len, 1)")
        batch, seq_len, _ = I_load.shape
        V = self.init_state(batch, I_load.device, I_load.dtype) if V0 is None else V0

        states = []
        for n in range(seq_len):
            states.append(V)
            V = self.step(V, I_load[:, n, :])
        V_seq = torch.stack(states, dim=1)
        return V_seq, V
