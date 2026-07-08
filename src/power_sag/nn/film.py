"""Feature-wise Linear Modulation (FiLM).

Given a conditioning signal ``v`` (the normalised supply voltage), FiLM
produces per-feature affine parameters and applies them to a feature map::

    gamma = W_gamma @ v + b_gamma
    beta  = W_beta  @ v + b_beta
    out   = gamma * h + beta

The projections are initialised to the identity transform (``gamma = 1``,
``beta = 0``) via their biases, while the weights keep their standard non-zero
initialisation so that gradients still flow back to ``v``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise linear modulation conditioned on ``v``.

    Parameters
    ----------
    feature_dim:
        Number of features being modulated (produces one gamma/beta per).
    cond_dim:
        Dimensionality of the conditioning signal (default ``1`` for V_B+).
    """

    def __init__(self, feature_dim: int, cond_dim: int = 1) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.cond_dim = cond_dim
        self.to_gamma = nn.Linear(cond_dim, feature_dim)
        self.to_beta = nn.Linear(cond_dim, feature_dim)
        # Start as the identity modulation: gamma == 1, beta == 0.
        nn.init.zeros_(self.to_gamma.bias)  # gamma bias set to 1 below
        with torch.no_grad():
            self.to_gamma.bias.fill_(1.0)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Modulate feature map ``h`` using conditioning ``v``.

        Parameters
        ----------
        h:
            Feature map, shape ``(..., feature_dim)``.
        v:
            Conditioning signal, shape ``(..., cond_dim)`` broadcastable to
            ``h`` across every dimension except the last.
        """
        gamma = self.to_gamma(v)
        beta = self.to_beta(v)
        return gamma * h + beta
