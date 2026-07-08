"""End-to-end gray-box model wiring together all subsystems.

The forward pass unrolls the coupled fast/slow system sample by sample::

    I_load[n] = coupling(x[n], V_B+[n])          # learned load current (θ)
    V_B+[n+1] = physics.step(V_B+[n], I_load[n])  # power-supply ODE  (η)
    y[n]      = audio(x[n], V_B+[n])              # conditioned audio path (φ)

An optional fixed cabinet IR can be applied to the output.  Because coupling
and physics form a recurrence (each ``V_B+[n]`` depends on the previous load
current), the supply trajectory is computed with an explicit time loop; the
whole graph remains differentiable via BPTT.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..physics import PowerSupplyODE
from .audio_model import LSTMState, PowerSagLSTM
from .coupling import CouplingNetwork
from .recurrence import SupplyRecurrence


class PowerSagModel(nn.Module):
    """Full physics-informed amp model (coupling + physics ODE + audio path)."""

    def __init__(
        self,
        physics: Optional[PowerSupplyODE] = None,
        coupling: Optional[CouplingNetwork] = None,
        audio: Optional[PowerSagLSTM] = None,
        cabinet: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.physics = physics or PowerSupplyODE()
        self.coupling = coupling or CouplingNetwork()
        self.audio = audio or PowerSagLSTM()
        self.cabinet = cabinet  # fixed processing stage, not trained
        self.use_script = False  # opt-in TorchScript fast path for the recurrence

    # ------------------------------------------------------------ scripting
    def enable_script(self) -> None:
        """Compile the coupling+physics recurrence with TorchScript.

        Roughly halves CPU wall time (more on GPU) for the sample-by-sample
        supply trajectory.  Only the scalar ``R_eff`` mode is accelerated; the
        nonlinear GZ34 model always uses the eager loop.  The scripted module
        shares its parameters with the eager modules, so training still works,
        but call this **after** moving the model to its device.
        """
        recurrence = torch.jit.script(
            SupplyRecurrence(self.coupling, self.physics.Ts)
        )
        # Bypass nn.Module.__setattr__ so the scripted module (whose parameters
        # are shared with self.coupling) is not registered as a submodule and
        # therefore not double-counted in ``parameters()``.
        object.__setattr__(self, "_scripted_recurrence", recurrence)
        self.use_script = True

    def disable_script(self) -> None:
        """Revert to the eager Python recurrence."""
        self.use_script = False

    # ---------------------------------------------------------------- factory
    @classmethod
    def from_config(cls, config: Dict[str, Any], cabinet: Optional[nn.Module] = None) -> "PowerSagModel":
        """Build a model from a config dict (see ``configs/default.yaml``)."""
        physics = PowerSupplyODE(
            fs=config["fs"],
            C1=float(config["C1"]),
            R_eff=float(config["R_eff"]),
            V_oc=float(config["V_oc"]),
            V_idle=float(config["V_idle"]),
            reff_mode=config.get("reff_mode", "scalar"),
            R0=float(config.get("R0", 250.0)),
            R1=float(config.get("R1", 100.0)),
            k=float(config.get("k", 1.0)),
            learn_R_eff=config.get("learn_R_eff", True),
            learn_C1=config.get("learn_C1", True),
            learn_V_oc=config.get("learn_V_oc", False),
        )
        coupling = CouplingNetwork(
            hidden=config["coupling_hidden"],
            V_idle=float(config["V_idle"]),
            delta_V=float(config["delta_V"]),
        )
        audio = PowerSagLSTM(
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            V_idle=float(config["V_idle"]),
            delta_V=float(config["delta_V"]),
        )
        return cls(physics=physics, coupling=coupling, audio=audio, cabinet=cabinet)

    # --------------------------------------------------------------- dynamics
    def supply_trajectory(
        self,
        x: torch.Tensor,
        V0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the B+ trajectory driven by the learned coupling.

        Returns ``(V_seq, V_final)`` with ``V_seq`` shaped ``(batch, seq_len, 1)``
        (state before applying each sample's load current) and ``V_final`` the
        terminal state for stateful continuation.
        """
        batch = x.shape[0]
        V = (
            self.physics.init_state(batch, x.device, x.dtype)
            if V0 is None
            else V0
        )

        # Fast path: scripted recurrence (scalar R_eff only).
        if self.use_script and self.physics.reff_mode == "scalar":
            return self._scripted_recurrence(
                x, self.physics.C1, self.physics.V_oc, self.physics._R_eff, V
            )

        states = []
        for n in range(x.shape[1]):
            states.append(V)
            x_n = x[:, n : n + 1, :]  # (batch, 1, 1)
            I_n = self.coupling(x_n, V.unsqueeze(1)).squeeze(1)  # (batch, 1)
            V = self.physics.step(V, I_n)
        V_seq = torch.stack(states, dim=1)
        return V_seq, V

    def forward(
        self,
        x: torch.Tensor,
        V0: Optional[torch.Tensor] = None,
        state: Optional[LSTMState] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, LSTMState]]:
        """End-to-end forward pass.

        Parameters
        ----------
        x:
            Input signal, shape ``(batch, seq_len, 1)``.
        V0:
            Optional initial supply state ``(batch, 1)``.
        state:
            Optional initial LSTM state ``(h, c)`` for stateful continuation
            across chunks/segments (used by truncated BPTT).  Named ``state``
            (not ``lstm_state``) so every model shares one forward signature
            ``(x, V0, state, return_state)`` and the trainer stays model-agnostic.
        return_state:
            If ``True`` also return the final ``V_B+`` and recurrent state, so
            both slow (physics) and fast (LSTM) states can be carried forward.

        Returns
        -------
        ``y`` of shape ``(batch, seq_len, 1)`` (or ``(y, V_final, state)``).
        """
        if x.dim() != 3 or x.size(-1) != 1:
            raise ValueError("x must have shape (batch, seq_len, 1)")

        V_seq, V_final = self.supply_trajectory(x, V0)
        y, state = self.audio(x, V_seq, state)
        if self.cabinet is not None:
            y = self.cabinet(y)
        if return_state:
            return y, V_final, state
        return y
