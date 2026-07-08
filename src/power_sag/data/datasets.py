"""Datasets and samplers for training the power-sag model.

``AudioDataset`` loads a paired (input DI, target amp output) recording from
disk and normalises it.  ``SequenceDataset`` slices one *or more* recordings
into ordered, fixed-length segments and tracks the latent supply state across
segment boundaries so that the terminal ``V_B+`` of one segment seeds the next
-- but never across a *recording* boundary.

Because the carried supply state only makes sense when a recording's segments
are visited in order, a plain ``DataLoader(dataset, shuffle=True)`` would
silently corrupt training: it shuffles individual segments, so most segments
would receive a ``V_B+`` initial condition belonging to an unrelated segment.
Use :class:`SequenceBatchSampler`, which shuffles at the *recording* level and
keeps each recording's segments in order.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..utils import load_audio, normalize_audio, to_mono_tensor

ArrayLike = Union[np.ndarray, torch.Tensor]
Recording = Tuple[ArrayLike, ArrayLike]


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
        self.input = load_audio(input_source)
        self.target = load_audio(target_source)
        n = min(len(self.input), len(self.target))
        self.input = self.input[:n]
        self.target = self.target[:n]
        if normalize:
            self.input = normalize_audio(self.input)
            self.target = normalize_audio(self.target)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input.unsqueeze(-1), self.target.unsqueeze(-1)


class SequenceDataset(Dataset):
    """Ordered fixed-length segments of one or more recordings, with carried state.

    Each item is a tuple ``(input_segment, target_segment, initial_V_B+)`` where
    ``initial_V_B+`` is the terminal supply state carried from the previous
    segment *of the same recording* (``V_idle`` for the first segment of each
    recording).  Segments are stored in temporal order, grouped by recording.
    After the model processes segment ``i`` and produces a terminal ``V_B+``,
    call :meth:`update_state` to feed it into the next segment of that recording.

    Parameters
    ----------
    input_audio, target_audio:
        A single recording's 1-D signals (arrays or tensors).  Ignored when
        ``recordings`` is given.
    segment_len:
        Number of samples per segment.
    V_idle:
        Initial supply voltage for the first segment of each recording.
    normalize:
        Peak-normalise each recording to ``[-1, 1]`` if ``True``.
    drop_last:
        Drop each recording's trailing partial segment (``True``) instead of
        zero-padding it.
    recordings:
        Optional list of ``(input, target)`` pairs for multi-recording training.
        Takes precedence over ``input_audio`` / ``target_audio``.
    """

    def __init__(
        self,
        input_audio: Optional[ArrayLike] = None,
        target_audio: Optional[ArrayLike] = None,
        segment_len: int = 24000,
        V_idle: float = 415.0,
        normalize: bool = True,
        drop_last: bool = True,
        recordings: Optional[Sequence[Recording]] = None,
    ) -> None:
        self.segment_len = int(segment_len)
        self.V_idle = float(V_idle)

        if recordings is None:
            if input_audio is None or target_audio is None:
                raise ValueError(
                    "provide either (input_audio, target_audio) or recordings"
                )
            recordings = [(input_audio, target_audio)]

        self.input_segments: List[torch.Tensor] = []
        self.target_segments: List[torch.Tensor] = []
        # recording_of[i] -> which recording global segment i belongs to;
        # sequences[r]     -> ordered global segment indices of recording r.
        self.recording_of: List[int] = []
        self.sequences: List[List[int]] = []

        for rec_id, (x_raw, t_raw) in enumerate(recordings):
            x = to_mono_tensor(x_raw)
            t = to_mono_tensor(t_raw)
            n = min(len(x), len(t))
            x, t = x[:n], t[:n]
            if normalize:
                x = normalize_audio(x)
                t = normalize_audio(t)

            seq: List[int] = []
            for start in range(0, n, self.segment_len):
                xs, ts = x[start : start + self.segment_len], t[start : start + self.segment_len]
                if len(xs) < self.segment_len:
                    if drop_last:
                        break
                    pad = self.segment_len - len(xs)
                    xs = torch.nn.functional.pad(xs, (0, pad))
                    ts = torch.nn.functional.pad(ts, (0, pad))
                global_idx = len(self.input_segments)
                self.input_segments.append(xs.unsqueeze(-1))
                self.target_segments.append(ts.unsqueeze(-1))
                self.recording_of.append(rec_id)
                seq.append(global_idx)
            self.sequences.append(seq)

        # Per-segment initial supply state; first segment of each recording idles.
        self.initial_states: List[torch.Tensor] = [
            torch.tensor(self.V_idle, dtype=torch.float32)
            for _ in self.input_segments
        ]

    def __len__(self) -> int:
        return len(self.input_segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.input_segments[idx],
            self.target_segments[idx],
            self.initial_states[idx],
        )

    def update_state(self, idx: int, V_terminal: Union[float, torch.Tensor]) -> None:
        """Carry segment ``idx``'s terminal ``V_B+`` to the next segment.

        The state is **detached** (continuity without gradient continuity --
        truncated BPTT at the segment boundary), and it is only carried within
        the same recording: the first segment of the next recording keeps its
        ``V_idle`` initial condition.
        """
        if isinstance(V_terminal, torch.Tensor):
            V_terminal = V_terminal.detach().reshape(-1)[0].clone()
        else:
            V_terminal = torch.tensor(float(V_terminal), dtype=torch.float32)
        nxt = idx + 1
        if nxt < len(self.initial_states) and self.recording_of[nxt] == self.recording_of[idx]:
            self.initial_states[nxt] = V_terminal

    def reset_states(self) -> None:
        """Reset every segment's initial state back to ``V_idle``."""
        for i in range(len(self.initial_states)):
            self.initial_states[i] = torch.tensor(self.V_idle, dtype=torch.float32)


class SequenceBatchSampler(Sampler):
    """Batch sampler that preserves per-recording segment order.

    Recordings are distributed round-robin across ``batch_size`` parallel lanes;
    each lane concatenates its recordings' segment indices in temporal order.
    At every step the sampler yields one segment per active lane, so segments
    that share a batch position across successive steps are consecutive within a
    single recording -- exactly the ordering the carried ``V_B+`` state needs.
    ``shuffle`` only reshuffles the *assignment* of recordings to lanes; it never
    reorders segments within a recording.

    Use it as ``DataLoader(dataset, batch_sampler=SequenceBatchSampler(...))``.
    """

    def __init__(
        self,
        dataset: SequenceDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self._rng = random.Random(seed)

    def _build_lanes(self) -> List[List[int]]:
        sequences = [list(seq) for seq in self.dataset.sequences if seq]
        if self.shuffle:
            self._rng.shuffle(sequences)
        lanes: List[List[int]] = [[] for _ in range(self.batch_size)]
        for i, seq in enumerate(sequences):
            lanes[i % self.batch_size].extend(seq)
        return lanes

    def __iter__(self) -> Iterator[List[int]]:
        lanes = self._build_lanes()
        depth = max((len(lane) for lane in lanes), default=0)
        for step in range(depth):
            batch = [lane[step] for lane in lanes if step < len(lane)]
            if batch:
                yield batch

    def __len__(self) -> int:
        lanes = self._build_lanes()
        return max((len(lane) for lane in lanes), default=0)
