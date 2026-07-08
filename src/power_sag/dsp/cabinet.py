"""Fixed cabinet impulse-response convolution stage.

The speaker/microphone response is approximately linear and time-invariant and
is unaffected by power-supply state, so it is applied as a fixed (non-trainable)
convolution rather than being learned.  The impulse response is stored as a
buffer -- it never receives gradients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import load_audio


class CabinetIR(nn.Module):
    """Causal linear convolution with a fixed impulse response.

    Parameters
    ----------
    ir:
        Impulse response: a file path (loaded with ``soundfile``) or a 1-D
        array/tensor of taps.
    normalize:
        Scale the IR so its samples sum to unity (unity DC gain) if ``True``.
    """

    def __init__(
        self,
        ir: Union[str, Path, np.ndarray, torch.Tensor],
        normalize: bool = False,
    ) -> None:
        super().__init__()
        taps = self._load(ir)
        if normalize:
            total = taps.sum()
            if total.abs() > 1e-12:
                taps = taps / total
        # Register as a buffer so it is fixed and never trained (no gradients).
        self.register_buffer("ir", taps)

    @staticmethod
    def _load(ir: Union[str, Path, np.ndarray, torch.Tensor]) -> torch.Tensor:
        taps = load_audio(ir)
        if taps.numel() == 0:
            raise ValueError("impulse response must have at least one tap")
        return taps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve ``x`` with the IR, preserving length via causal zero-pad.

        Parameters
        ----------
        x:
            Signal of shape ``(batch, seq_len, 1)`` or ``(batch, seq_len)``.

        Returns
        -------
        Same shape as ``x`` with length equal to the input length.
        """
        squeeze_last = x.dim() == 3
        if squeeze_last:
            if x.size(-1) != 1:
                raise ValueError("CabinetIR expects a single channel")
            signal = x.squeeze(-1)  # (batch, seq_len)
        else:
            signal = x

        length = signal.size(-1)
        taps = self.ir
        # True linear convolution = cross-correlation with the flipped kernel.
        kernel = taps.flip(0).view(1, 1, -1).to(signal.dtype)
        padded = F.pad(signal.unsqueeze(1), (taps.numel() - 1, 0))
        conv = F.conv1d(padded, kernel)
        conv = conv[..., :length].squeeze(1)  # (batch, seq_len)

        if squeeze_last:
            return conv.unsqueeze(-1)
        return conv
