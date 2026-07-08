"""Datasets for training the power-sag model.

``AudioDataset`` loads a paired (input DI, target amp output) recording from
disk and normalises it.  ``SequenceDataset`` slices a recording into ordered,
fixed-length segments and tracks the latent supply state across segment
boundaries so that the terminal ``V_B+`` of one segment seeds the next.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_mono_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a (frames,) or (frames, channels) array to a 1-D float tensor."""
    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return torch.from_numpy(data)


def normalize_audio(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Peak-normalise a signal into ``[-1, 1]`` (divide by max abs value)."""
    peak = x.abs().max()
    if peak < eps:
        return x
    return x / peak


class AudioDataset(Dataset):
    """A single paired (input, target) recording, normalised to ``[-1, 1]``.

    Either provide file paths (loaded with ``soundfile``) or in-memory arrays.
    """

    def __init__(
        self,
        input_source: Union[str, Path, np.ndarray, torch.Tensor],
        target_source: Union[str, Path, np.ndarray, torch.Tensor],
        normalize: bool = True,
    ) -> None:
        self.input = self._load(input_source)
        self.target = self._load(target_source)
        n = min(len(self.input), len(self.target))
        self.input = self.input[:n]
        self.target = self.target[:n]
        if normalize:
            self.input = normalize_audio(self.input)
            self.target = normalize_audio(self.target)

    @staticmethod
    def _load(source: Union[str, Path, np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(source, torch.Tensor):
            return source.detach().to(torch.float32).flatten()
        if isinstance(source, np.ndarray):
            return _to_mono_tensor(source)
        import soundfile as sf  # local import: only needed for file loading

        data, _ = sf.read(str(source), dtype="float32", always_2d=False)
        return _to_mono_tensor(data)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input.unsqueeze(-1), self.target.unsqueeze(-1)


class SequenceDataset(Dataset):
    """Ordered fixed-length segments of one recording with carried supply state.

    Each item is a tuple ``(input_segment, target_segment, initial_V_B+)`` where
    ``initial_V_B+`` is the terminal supply state carried from the previous
    segment (``V_idle`` for the first segment).  Segments are stored in temporal
    order.  After the model processes segment ``i`` and produces a terminal
    ``V_B+``, call :meth:`update_state` to feed it into segment ``i + 1``.

    Parameters
    ----------
    input_audio, target_audio:
        1-D signals (arrays or tensors) of equal length.
    segment_len:
        Number of samples per segment.
    V_idle:
        Initial supply voltage for the first segment.
    normalize:
        Peak-normalise both signals to ``[-1, 1]`` if ``True``.
    drop_last:
        Drop a trailing partial segment (``True``) instead of zero-padding it.
    """

    def __init__(
        self,
        input_audio: Union[np.ndarray, torch.Tensor],
        target_audio: Union[np.ndarray, torch.Tensor],
        segment_len: int,
        V_idle: float = 415.0,
        normalize: bool = True,
        drop_last: bool = True,
    ) -> None:
        self.segment_len = int(segment_len)
        self.V_idle = float(V_idle)

        x = self._as_tensor(input_audio)
        t = self._as_tensor(target_audio)
        n = min(len(x), len(t))
        x, t = x[:n], t[:n]
        if normalize:
            x = normalize_audio(x)
            t = normalize_audio(t)

        self.input_segments: List[torch.Tensor] = []
        self.target_segments: List[torch.Tensor] = []
        for start in range(0, n, self.segment_len):
            end = start + self.segment_len
            xs, ts = x[start:end], t[start:end]
            if len(xs) < self.segment_len:
                if drop_last:
                    break
                pad = self.segment_len - len(xs)
                xs = torch.nn.functional.pad(xs, (0, pad))
                ts = torch.nn.functional.pad(ts, (0, pad))
            self.input_segments.append(xs.unsqueeze(-1))
            self.target_segments.append(ts.unsqueeze(-1))

        # Per-segment initial supply state; segment 0 starts at idle.
        self.initial_states: List[torch.Tensor] = [
            torch.tensor(self.V_idle, dtype=torch.float32)
            for _ in self.input_segments
        ]

    @staticmethod
    def _as_tensor(audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(audio, torch.Tensor):
            return audio.detach().to(torch.float32).flatten()
        return _to_mono_tensor(audio)

    def __len__(self) -> int:
        return len(self.input_segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.input_segments[idx],
            self.target_segments[idx],
            self.initial_states[idx],
        )

    def update_state(self, idx: int, V_terminal: Union[float, torch.Tensor]) -> None:
        """Store segment ``idx``'s terminal ``V_B+`` as segment ``idx+1``'s init."""
        if isinstance(V_terminal, torch.Tensor):
            V_terminal = V_terminal.detach().reshape(-1)[0].clone()
        else:
            V_terminal = torch.tensor(float(V_terminal), dtype=torch.float32)
        if idx + 1 < len(self.initial_states):
            self.initial_states[idx + 1] = V_terminal

    def reset_states(self) -> None:
        """Reset every segment's initial state back to ``V_idle``."""
        for i in range(len(self.initial_states)):
            self.initial_states[i] = torch.tensor(self.V_idle, dtype=torch.float32)
