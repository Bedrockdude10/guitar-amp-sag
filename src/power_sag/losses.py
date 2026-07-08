"""Training losses: error-to-signal ratio and pre-emphasised ESR.

``ESRLoss`` is the standard error-signal ratio used throughout the neural amp
modelling literature (Wright & Valimaki, 2020).  ``PreEmphasisLoss`` applies a
first-order high-pass pre-emphasis filter before measuring ESR, weighting the
error toward perceptually important high frequencies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESRLoss(nn.Module):
    """Error-to-signal ratio: ``sum((y - t)^2) / sum(t^2)``.

    Returns ``0`` for a perfect prediction and ``1`` when the prediction is
    zero against a non-zero target.
    """

    def __init__(self, eps: float = 1e-10) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = torch.sum((prediction - target) ** 2)
        signal = torch.sum(target ** 2)
        return error / (signal + self.eps)


class PreEmphasisLoss(nn.Module):
    """ESR computed on pre-emphasised signals.

    The pre-emphasis filter is the first-order high-pass ``y[n] - c*y[n-1]``.
    """

    def __init__(self, coeff: float = 0.85, eps: float = 1e-10) -> None:
        super().__init__()
        self.coeff = float(coeff)
        self.esr = ESRLoss(eps=eps)

    def pre_emphasis(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the first-order pre-emphasis filter, preserving shape.

        Expects ``(batch, seq_len, channels)``.  The first sample is left
        unfiltered (equivalent to zero-padding the input history).
        """
        shifted = F.pad(x, (0, 0, 1, 0))[:, :-1, :]  # x[n-1], zero for n=0
        return x - self.coeff * shifted

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.esr(self.pre_emphasis(prediction), self.pre_emphasis(target))
