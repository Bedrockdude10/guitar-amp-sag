"""Coupling network (θ): estimates load current from signal and supply state.

Implements option 2 of the research notes -- a small MLP mapping the pair
``(x[n], V_B+[n])`` to the instantaneous load current ``I_load[n]``.  The output
passes through a softplus so the estimated current is always non-negative
(the power tubes draw current from the supply; they never source it).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CouplingNetwork(nn.Module):
    """MLP mapping ``(x, V_B+) -> I_load`` (non-negative).

    Parameters
    ----------
    hidden:
        Width of the hidden layers.
    V_idle, delta_V:
        Used to normalise the supply voltage to roughly ``[-1, 1]`` before
        feeding it to the network, matching the FiLM conditioning convention.
    """

    def __init__(self, hidden: int = 16, V_idle: float = 415.0, delta_V: float = 40.0) -> None:
        super().__init__()
        self.V_idle = float(V_idle)
        self.delta_V = float(delta_V)
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Return ``I_load`` of shape ``(batch, seq_len, 1)``, non-negative.

        Parameters
        ----------
        x:
            Input signal, shape ``(batch, seq_len, 1)``.
        V:
            Supply voltage, shape ``(batch, seq_len, 1)``.
        """
        v_norm = (V - self.V_idle) / self.delta_V
        features = torch.cat([x, v_norm], dim=-1)
        raw = self.net(features)
        return F.softplus(raw)
