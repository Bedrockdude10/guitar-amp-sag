"""Shared helpers used across the package.

Centralises the audio I/O, mono conversion and normalisation utilities that
were previously duplicated between the data and cabinet modules, plus the
supply-voltage normalisation shared by the coupling and audio-path networks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

AudioSource = Union[str, Path, np.ndarray, torch.Tensor]


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
