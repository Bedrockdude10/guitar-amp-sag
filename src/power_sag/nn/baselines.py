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
    """Standard black-box LSTM amplifier model (no supply conditioning)."""

    def __init__(
        self, hidden_size: int = 32, num_layers: int = 1, input_size: int = 1
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, state: Optional[LSTMState] = None
    ) -> torch.Tensor:
        h, _ = self.lstm(x, state)
        return self.output(h)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slow, _ = self.state_gru(x)
        latent = self.state_proj(slow)  # (batch, seq_len, 1) learned slow state
        h, _ = self.lstm(x)
        h = self.film(h, latent)
        return self.output(h)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConditionedLSTMNoPhysics":
        return cls(
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            state_hidden=config.get("coupling_hidden", 8),
        )
