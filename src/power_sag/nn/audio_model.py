"""Neural audio path (φ): an LSTM conditioned on the supply voltage via FiLM.

The audio path follows the black-box LSTM design of Wright et al., modified to
accept the B+ supply voltage as a conditioning signal.  FiLM is applied to the
LSTM hidden sequence before the output projection, letting the sag state
modulate the amplifier's response sample by sample.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .. import constants as const
from ..utils import normalize_supply
from .film import FiLMLayer

LSTMState = Tuple[torch.Tensor, torch.Tensor]


class PowerSagLSTM(nn.Module):
    """Supply-voltage-conditioned LSTM amp model.

    Parameters
    ----------
    hidden_size:
        LSTM hidden width.
    num_layers:
        Number of stacked LSTM layers.
    input_size:
        Number of input channels (default ``1``, mono guitar).
    V_idle, delta_V:
        Normalisation constants for the supply voltage: the FiLM conditioning
        signal is ``(V_B+ - V_idle) / delta_V``.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        input_size: int = 1,
        V_idle: float = const.V_IDLE,
        delta_V: float = const.DELTA_V,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.V_idle = float(V_idle)
        self.delta_V = float(delta_V)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.film = FiLMLayer(hidden_size, cond_dim=1)
        self.output = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        V: torch.Tensor,
        state: Optional[LSTMState] = None,
    ) -> Tuple[torch.Tensor, LSTMState]:
        """Run the conditioned audio path.

        Parameters
        ----------
        x:
            Input signal, shape ``(batch, seq_len, 1)``.
        V:
            Supply voltage, shape ``(batch, seq_len, 1)``.
        state:
            Optional ``(h, c)`` LSTM state for stateful continuation.

        Returns
        -------
        (y, state):
            Output signal ``(batch, seq_len, 1)`` and the final LSTM state.
        """
        h_seq, state = self.lstm(x, state)
        v_norm = normalize_supply(V, self.V_idle, self.delta_V)
        h_mod = self.film(h_seq, v_norm)
        y = self.output(h_mod)
        return y, state
