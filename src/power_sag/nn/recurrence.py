"""TorchScript-able coupling+physics recurrence for faster unrolling.

Unrolling the coupled ``coupling -> Euler step`` recurrence sample-by-sample in
a Python ``for`` loop is correct but dominated by interpreter dispatch overhead
(~24k iterations per 0.5 s segment).  Wrapping the loop in a
:func:`torch.jit.script` module removes that overhead -- on CPU it roughly
halves the wall time, and the win is larger on GPU where per-op launch latency
dominates.

The recurrence reuses :meth:`CouplingNetwork.forward` and
:func:`power_sag.physics.ode.euler_step` verbatim, so it cannot diverge from the
eager path (an equivalence test guards this).  A scripted module shares its
parameter tensors with the eager modules, so in-place optimiser updates are
reflected and gradients flow back -- it is safe for training as well as
inference.  Only the scalar ``R_eff`` mode is supported; the nonlinear GZ34
model falls back to the eager loop.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from ..physics.ode import euler_step
from .coupling import CouplingNetwork


class SupplyRecurrence(nn.Module):
    """Scriptable ``V_B+`` trajectory driven by the learned coupling network."""

    def __init__(self, coupling: CouplingNetwork, Ts: float) -> None:
        super().__init__()
        self.coupling = coupling
        self.Ts = Ts

    def forward(
        self,
        x: torch.Tensor,
        C1: torch.Tensor,
        V_oc: torch.Tensor,
        R_eff: torch.Tensor,
        V0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(V_seq, V_final)`` for input ``x`` of shape ``(B, T, 1)``.

        ``C1``, ``V_oc`` and ``R_eff`` are passed as tensors (rather than read
        from attributes) so the scripted module always uses the current
        parameter values without needing to be rebuilt after each step.
        """
        T = x.shape[1]
        V = V0
        states: List[torch.Tensor] = []
        for n in range(T):
            states.append(V)
            x_n = x[:, n : n + 1, :]
            I_n = self.coupling(x_n, V.unsqueeze(1)).squeeze(1)
            V = euler_step(V, I_n, self.Ts, C1, V_oc, R_eff)
        return torch.stack(states, dim=1), V
