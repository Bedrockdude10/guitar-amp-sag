"""Baseline models for the paper's comparison.

Both are black-box ``x -> y`` models with the same call signature as
:class:`PowerSagModel` (``model(x)`` returns ``(batch, seq_len, 1)``), so the
evaluation harness can treat every model uniformly.

* :class:`UnconditionedLSTM` -- a standard Wright-style black-box LSTM with no
  supply conditioning at all.  Ablates *any* power-supply awareness.
* :class:`ConditionedLSTMNoPhysics` -- the same conditioned audio path as our
  model, but the slow conditioning state is produced by a learned GRU rather
  than the physics ODE.  Ablates the *physics* while keeping the conditioning
  mechanism, isolating the benefit of the physical structure.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .audio_model import LSTMState
from .film import FiLMLayer


class UnconditionedLSTM(nn.Module):
    """Standard black-box LSTM amplifier model (no supply conditioning).

    Shares the model-agnostic forward signature ``(x, V0, state, return_state)``;
    ``V0`` is accepted and ignored (there is no supply state).
    """

    def __init__(
        self, hidden_size: int = 32, num_layers: int = 1, input_size: int = 1
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        V0: Optional[torch.Tensor] = None,
        state: Optional[LSTMState] = None,
        return_state: bool = False,
    ):
        h, state = self.lstm(x, state)
        y = self.output(h)
        if return_state:
            return y, None, state  # (output, V_final=None, recurrent state)
        return y

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "UnconditionedLSTM":
        return cls(hidden_size=config["hidden_size"], num_layers=config["num_layers"])


class ConditionedLSTMNoPhysics(nn.Module):
    """Ablation: conditioned audio path with a *learned* slow state (no physics).

    A small GRU maps the input to a scalar latent "supply-like" state, which
    FiLM-modulates the main LSTM path -- structurally identical to the proposed
    model except the physics ODE is replaced by a free recurrent network.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        state_hidden: int = 8,
        input_size: int = 1,
    ) -> None:
        super().__init__()
        self.state_gru = nn.GRU(input_size, state_hidden, batch_first=True)
        self.state_proj = nn.Linear(state_hidden, 1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.film = FiLMLayer(hidden_size, cond_dim=1)
        self.output = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        V0: Optional[torch.Tensor] = None,
        state: Optional[tuple] = None,
        return_state: bool = False,
    ):
        gru_state, lstm_state = (state if state is not None else (None, None))
        slow, gru_state = self.state_gru(x, gru_state)
        latent = self.state_proj(slow)  # (batch, seq_len, 1) learned slow state
        h, lstm_state = self.lstm(x, lstm_state)
        h = self.film(h, latent)
        y = self.output(h)
        if return_state:
            return y, None, (gru_state, lstm_state)
        return y

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConditionedLSTMNoPhysics":
        return cls(
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            state_hidden=config.get("coupling_hidden", 8),
        )
