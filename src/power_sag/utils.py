"""Shared helpers used across the package.

Centralises the audio I/O, mono conversion and normalisation utilities that
were previously duplicated between the data and cabinet modules, plus the
supply-voltage normalisation shared by the coupling and audio-path networks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

AudioSource = Union[str, Path, np.ndarray, torch.Tensor]


def seed_everything(seed: int = 0, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch RNGs for reproducible experiments.

    Returns the seed so callers can log it. With ``deterministic=True`` it also
    requests deterministic cuDNN algorithms (slower, but bit-reproducible on the
    same hardware) -- useful when generating paper numbers.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def detach_state(state):
    """Recursively detach a recurrent state (tensor, tuple/list, or ``None``).

    Lets the trainer truncate BPTT uniformly across models whose state is a
    tensor (none), an LSTM ``(h, c)`` pair, or a nested tuple of both.
    """
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, (tuple, list)):
        return type(state)(detach_state(s) for s in state)
    return state


def resolve_device(preference: Optional[str] = None) -> torch.device:
    """Pick a compute device, preferring CUDA, then Apple MPS, then CPU.

    ``preference`` may be an explicit device string (``"cuda"``, ``"mps"``,
    ``"cpu"``); ``None`` or ``"auto"`` auto-selects.  The model is float32
    throughout, so it runs on MPS (which lacks float64) without changes.
    """
    if preference and preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_backends(device: torch.device) -> None:
    """Enable safe, device-appropriate performance backends.

    On CUDA this turns on cuDNN autotuning (a free speedup for the fixed-shape
    LSTM).  Note: the supply ODE integrates over tens of thousands of Euler
    steps and *must* stay in float32 -- float16/AMP cannot resolve ~0.01 V
    changes on a ~415 V rail, so mixed precision is deliberately not enabled on
    the physics path.
    """
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def to_mono_tensor(array: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Return a 1-D float32 tensor, averaging channels if the input is 2-D."""
    if isinstance(array, torch.Tensor):
        data = array.detach().to(torch.float32)
        if data.dim() == 2:
            data = data.mean(dim=1)
        return data.flatten()
    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return torch.from_numpy(np.ascontiguousarray(data))


def load_audio(source: AudioSource) -> torch.Tensor:
    """Load audio from a file path or in-memory array as a 1-D float tensor.

    File paths are read with ``soundfile`` (imported lazily so array-only usage
    does not require the dependency).
    """
    if isinstance(source, (torch.Tensor, np.ndarray)):
        return to_mono_tensor(source)
    import soundfile as sf  # local import: only needed for file loading

    data, _ = sf.read(str(source), dtype="float32", always_2d=False)
    return to_mono_tensor(data)


def normalize_audio(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Peak-normalise a signal into ``[-1, 1]`` (divide by max abs value)."""
    peak = x.abs().max()
    if peak < eps:
        return x
    return x / peak


def normalize_supply(V: torch.Tensor, V_idle: float, delta_V: float) -> torch.Tensor:
    """Normalise the supply voltage to roughly ``[-1, 1]`` for conditioning.

    ``(V_B+ - V_idle) / delta_V`` -- the convention shared by the coupling
    network and the FiLM conditioning in the audio path.
    """
    return (V - V_idle) / delta_V


def estimate_delay(
    reference: AudioSource, delayed: AudioSource, max_lag: Optional[int] = None
) -> int:
    """Estimate the integer sample delay of ``delayed`` relative to ``reference``.

    Returns ``d`` such that ``delayed[n] ~= reference[n - d]`` (positive ``d``
    means ``delayed`` lags).  Uses FFT cross-correlation of the two (mean-
    removed) signals; ``max_lag`` restricts the search window.
    """
    from scipy.signal import correlate, correlation_lags

    x = load_audio(reference).numpy().astype(np.float64)
    y = load_audio(delayed).numpy().astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    corr = correlate(y, x, mode="full", method="fft")
    lags = correlation_lags(len(y), len(x), mode="full")
    if max_lag is not None:
        keep = np.abs(lags) <= int(max_lag)
        corr, lags = corr[keep], lags[keep]
    return int(lags[int(np.argmax(corr))])


def align_signals(
    input_audio: AudioSource,
    target_audio: AudioSource,
    max_lag: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Time-align a DI/target pair by compensating their latency.

    Returns ``(aligned_input, aligned_target, delay)`` as equal-length 1-D
    tensors.  Misaligned DI/output pairs make the ESR loss meaningless, so this
    should be run on captured data before building datasets.
    """
    x = load_audio(input_audio)
    t = load_audio(target_audio)
    delay = estimate_delay(x, t, max_lag=max_lag)
    if delay > 0:  # target lags: drop its leading samples
        t = t[delay:]
    elif delay < 0:  # input lags: drop its leading samples
        x = x[-delay:]
    n = min(len(x), len(t))
    return x[:n], t[:n], delay
