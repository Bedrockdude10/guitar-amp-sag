"""Sag-targeted evaluation protocol.

Standard ESR/MAE metrics average over playing contexts and do not reveal
whether sag behaviour is correct.  ``SagEvaluator`` generates the targeted test
signals from the research notes (sustained chords, dynamic transitions) and
provides curve-fitting utilities to measure recovery time constants.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from scipy.optimize import curve_fit


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
