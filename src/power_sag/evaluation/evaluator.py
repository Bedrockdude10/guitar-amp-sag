"""Sag-targeted evaluation protocol.

Standard ESR/MAE metrics average over playing contexts and do not reveal
whether sag behaviour is correct.  ``SagEvaluator`` generates the targeted test
signals from the research notes (sustained chords, dynamic transitions) and
provides curve-fitting utilities to measure recovery time constants.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit

# A model under test: maps input (1, T, 1) to output (1, T, 1).
ModelFn = Callable[[torch.Tensor], torch.Tensor]


def _decay_model(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """Exponential recovery/decay model ``A * exp(-t / tau) + C``."""
    return A * np.exp(-t / tau) + C


class SagEvaluator:
    """Generate sag test signals and measure recovery time constants.

    Parameters
    ----------
    fs:
        Sample rate in Hz used for all generated signals and time axes.
    """

    def __init__(self, fs: float = 48000.0) -> None:
        self.fs = float(fs)

    # -------------------------------------------------------------- signals
    def sustained_chord_signal(
        self,
        duration: float,
        f0: float = 82.41,  # low E
        amplitude: float = 0.8,
        intervals: Tuple[float, ...] = (1.0, 1.5, 2.0),  # root, fifth, octave
    ) -> torch.Tensor:
        """A sustained power chord of the given ``duration`` (seconds).

        Returns a tensor of shape ``(1, n_samples, 1)`` where
        ``n_samples = round(duration * fs)``.
        """
        n = int(round(duration * self.fs))
        t = np.arange(n, dtype=np.float32) / self.fs
        wave = np.zeros(n, dtype=np.float32)
        for ratio in intervals:
            wave += np.sin(2.0 * np.pi * f0 * ratio * t).astype(np.float32)
        wave /= max(len(intervals), 1)
        wave *= amplitude
        return torch.from_numpy(wave).view(1, n, 1)

    def dynamic_signal(
        self,
        loud_duration: float,
        quiet_duration: float,
        f0: float = 82.41,
        loud_amp: float = 0.9,
        quiet_amp: float = 0.1,
    ) -> torch.Tensor:
        """A loud passage followed by a quiet one (exercises recovery)."""
        loud = self.sustained_chord_signal(loud_duration, f0=f0, amplitude=loud_amp)
        quiet = self.sustained_chord_signal(quiet_duration, f0=f0, amplitude=quiet_amp)
        return torch.cat([loud, quiet], dim=1)

    # ------------------------------------------------------------ analysis
    def fit_recovery_curve(
        self,
        signal: torch.Tensor,
        fs: Optional[float] = None,
    ) -> float:
        """Fit ``A * exp(-t/tau) + C`` to a decaying signal, returning ``tau``.

        Parameters
        ----------
        signal:
            1-D (or squeezable) tensor/array of a monotonic-ish decay.
        fs:
            Sample rate override (defaults to the evaluator's ``fs``).

        Returns
        -------
        The recovery time constant ``tau`` in seconds (always positive).
        """
        fs = float(fs) if fs is not None else self.fs
        if isinstance(signal, torch.Tensor):
            y = signal.detach().reshape(-1).cpu().numpy().astype(np.float64)
        else:
            y = np.asarray(signal, dtype=np.float64).reshape(-1)
        n = len(y)
        if n < 3:
            raise ValueError("need at least 3 samples to fit a recovery curve")
        t = np.arange(n, dtype=np.float64) / fs

        span = y[0] - y[-1]
        A0 = span if span != 0 else (y.max() - y.min() + 1e-6)
        C0 = y[-1]
        tau0 = max(t[-1] / 3.0, 1e-6)

        params, _ = curve_fit(
            _decay_model,
            t,
            y,
            p0=(A0, tau0, C0),
            maxfev=20000,
        )
        tau = abs(float(params[1]))
        return tau

    def rms_envelope(self, signal: torch.Tensor, win_ms: float = 10.0) -> torch.Tensor:
        """Sliding-window RMS envelope of a signal (shape preserved, 1-D out)."""
        win = max(int(win_ms * 1e-3 * self.fs), 1)
        x = signal.detach().reshape(1, 1, -1)
        power = x ** 2
        kernel = torch.ones(1, 1, win, dtype=power.dtype) / win
        padded = F.pad(power, (win - 1, 0))
        return torch.sqrt(F.conv1d(padded, kernel).reshape(-1) + 1e-12)

    @staticmethod
    def _run(model: ModelFn, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(x)

    # ----------------------------------------------------- sag-targeted tests
    def attack_bloom(
        self, model: ModelFn, duration: float = 1.0, amplitude: float = 0.9
    ) -> Dict[str, float]:
        """Test 1 -- sustained-chord attack/bloom shape.

        Returns the peak attack RMS, the settled (steady-state) RMS, and their
        ratio.  A sagging amp softens the attack and settles lower, so
        ``settle/attack < 1``.
        """
        x = self.sustained_chord_signal(duration, amplitude=amplitude)
        env = self.rms_envelope(self._run(model, x))
        attack = float(env[: env.numel() // 10].max())
        settle = float(env[-env.numel() // 10 :].mean())
        return {"attack_rms": attack, "settle_rms": settle,
                "settle_over_attack": settle / (attack + 1e-9)}

    def recovery_time_constant(
        self, model: ModelFn, loud: float = 0.5, quiet: float = 1.0
    ) -> float:
        """Test 2 -- dynamic recovery time constant.

        Drive loud then quiet and fit the exponential recovery of the output
        RMS envelope in the quiet region.  Returns ``tau`` in seconds.
        """
        x = self.dynamic_signal(loud, quiet, loud_amp=0.9, quiet_amp=0.05)
        env = self.rms_envelope(self._run(model, x))
        start = int(loud * self.fs)
        return self.fit_recovery_curve(env[start:], fs=self.fs)

    def presag_vs_cold_attack(
        self, model: ModelFn, note: float = 0.2, prime: float = 0.8
    ) -> Dict[str, float]:
        """Test 3 -- pre-sagged vs cold-supply attack.

        Compare a note's attack peak when played (a) cold, after silence, and
        (b) pre-sagged, right after a loud sustain.  A sagging amp compresses
        the pre-sagged attack, so ``presag/cold < 1``.
        """
        n_note = int(note * self.fs)
        cold = torch.cat(
            [torch.zeros(1, int(prime * self.fs), 1),
             self.sustained_chord_signal(note, amplitude=0.6)], dim=1)
        primed = torch.cat(
            [self.sustained_chord_signal(prime, amplitude=0.9),
             self.sustained_chord_signal(note, amplitude=0.6)], dim=1)

        cold_env = self.rms_envelope(self._run(model, cold))[-n_note:]
        primed_env = self.rms_envelope(self._run(model, primed))[-n_note:]
        cold_peak = float(cold_env.max())
        presag_peak = float(primed_env.max())
        return {"cold_attack": cold_peak, "presag_attack": presag_peak,
                "presag_over_cold": presag_peak / (cold_peak + 1e-9)}

    def quiet_to_loud_response(
        self, model: ModelFn, quiet: float = 0.5, loud: float = 0.5
    ) -> Dict[str, float]:
        """Test 4 -- quiet-to-loud transition (supply "stiffening")."""
        x = torch.cat(
            [self.sustained_chord_signal(quiet, amplitude=0.1),
             self.sustained_chord_signal(loud, amplitude=0.9)], dim=1)
        env = self.rms_envelope(self._run(model, x))
        n_loud = int(loud * self.fs)
        loud_env = env[-n_loud:]
        peak = float(loud_env.max())
        settle = float(loud_env[-loud_env.numel() // 5 :].mean())
        return {"loud_peak": peak, "loud_settle": settle,
                "settle_over_peak": settle / (peak + 1e-9)}

    def run_protocol(self, model: ModelFn) -> Dict[str, Dict[str, float]]:
        """Run the full 4-part sag protocol; return all measurements."""
        return {
            "attack_bloom": self.attack_bloom(model),
            "recovery_tau": {"tau_s": self.recovery_time_constant(model)},
            "presag_vs_cold": self.presag_vs_cold_attack(model),
            "quiet_to_loud": self.quiet_to_loud_response(model),
        }
