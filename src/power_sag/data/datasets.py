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

from .. import constants as const
from ..utils import load_audio, normalize_audio, to_mono_tensor

ArrayLike = Union[np.ndarray, torch.Tensor]
Recording = Tuple[ArrayLike, ArrayLike]


def chronological_split(
    input_audio: ArrayLike,
    target_audio: ArrayLike,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[Recording, Recording, Recording]:
    """Split a recording into (train, val, test) *by time*, not randomly.

    Audio modelling data must be split chronologically: a random split would
    leak near-identical neighbouring samples across the split boundary and
    inflate the validation score.  Returns three ``(input, target)`` pairs; the
    train portion comes first, then val, then test.
    """
    if not 0.0 <= val_frac < 1.0 or not 0.0 <= test_frac < 1.0:
        raise ValueError("fractions must be in [0, 1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")

    x = to_mono_tensor(input_audio)
    t = to_mono_tensor(target_audio)
    n = min(len(x), len(t))
    x, t = x[:n], t[:n]

    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test
    train = (x[:n_train], t[:n_train])
    val = (x[n_train : n_train + n_val], t[n_train : n_train + n_val])
    test = (x[n_train + n_val :], t[n_train + n_val :])
    return train, val, test


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
    measured_vb:
        Optional directly-measured ``V_B+`` signal (e.g. from a buffered B+
        probe), same length as the single recording.  When present, its
        per-segment slices are available via :meth:`measured_vb_segment` and can
        be used for supervised training of the supply state -- turning the hard
        latent-variable problem into direct regression.  Not normalised.
    """

    def __init__(
        self,
        input_audio: Optional[ArrayLike] = None,
        target_audio: Optional[ArrayLike] = None,
        segment_len: int = const.SEGMENT_LEN,
        V_idle: float = const.V_IDLE,
        normalize: bool = True,
        drop_last: bool = True,
        recordings: Optional[Sequence[Recording]] = None,
        measured_vb: Optional[ArrayLike] = None,
    ) -> None:
        self.segment_len = int(segment_len)
        self.V_idle = float(V_idle)

        if recordings is None:
            if input_audio is None or target_audio is None:
                raise ValueError(
                    "provide either (input_audio, target_audio) or recordings"
                )
            recordings = [(input_audio, target_audio)]
        if measured_vb is not None and len(recordings) != 1:
            raise ValueError("measured_vb is only supported for a single recording")

        vb_full = to_mono_tensor(measured_vb) if measured_vb is not None else None

        self.input_segments: List[torch.Tensor] = []
        self.target_segments: List[torch.Tensor] = []
        self.measured_vb_segments: List[Optional[torch.Tensor]] = []
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
                end = start + self.segment_len
                xs, ts = x[start:end], t[start:end]
                vs = vb_full[start:end] if vb_full is not None else None
                if len(xs) < self.segment_len:
                    if drop_last:
                        break
                    pad = self.segment_len - len(xs)
                    xs = torch.nn.functional.pad(xs, (0, pad))
                    ts = torch.nn.functional.pad(ts, (0, pad))
                    if vs is not None:
                        vs = torch.nn.functional.pad(vs, (0, pad))
                global_idx = len(self.input_segments)
                self.input_segments.append(xs.unsqueeze(-1))
                self.target_segments.append(ts.unsqueeze(-1))
                self.measured_vb_segments.append(
                    vs.unsqueeze(-1) if vs is not None else None
                )
                self.recording_of.append(rec_id)
                seq.append(global_idx)
            self.sequences.append(seq)

        self.has_measured_vb = vb_full is not None
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

    def measured_vb_segment(self, idx: int) -> Optional[torch.Tensor]:
        """Return the measured ``V_B+`` slice for segment ``idx`` (or ``None``).

        Shape ``(segment_len, 1)`` when a B+ probe signal was provided.
        """
        return self.measured_vb_segments[idx]

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
        self._lanes: Optional[List[List[int]]] = None

    def _build_lanes(self) -> List[List[int]]:
        # Build the lane assignment exactly once per sampler and cache it.  A
        # fresh sampler is created each epoch (with a per-epoch seed), so caching
        # gives the intended one-shuffle-per-epoch behaviour while ensuring
        # ``__len__`` and ``__iter__`` see the *same* assignment.  Previously
        # both rebuilt independently and each consumed ``self._rng``, so a
        # ``len(sampler)`` call (e.g. by DataLoader) reshuffled the lanes and
        # desynced the reported length from the batches actually yielded.
        if self._lanes is None:
            sequences = [list(seq) for seq in self.dataset.sequences if seq]
            if self.shuffle:
                self._rng.shuffle(sequences)
            lanes: List[List[int]] = [[] for _ in range(self.batch_size)]
            for i, seq in enumerate(sequences):
                lanes[i % self.batch_size].extend(seq)
            self._lanes = lanes
        return self._lanes

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
