"""Tests for shared utilities: device resolution and the vectorised envelope."""

import numpy as np
import torch

from power_sag.synthetic import SyntheticSagAmp
from power_sag.utils import configure_backends, resolve_device


def test_resolve_device_explicit_preference():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_is_valid():
    dev = resolve_device("auto")
    assert dev.type in {"cuda", "mps", "cpu"}
    # auto never picks an unavailable backend.
    if dev.type == "cuda":
        assert torch.cuda.is_available()


def test_configure_backends_is_safe_on_cpu():
    configure_backends(torch.device("cpu"))  # must not raise


def test_vectorized_envelope_matches_naive_loop():
    """The lfilter envelope must equal the reference one-pole sample loop."""
    amp = SyntheticSagAmp(fs=48000.0, env_tau=5e-3)
    x = 0.7 * torch.randn(2, 600, 1)
    fast = amp._envelope(x)

    beta = amp.env_beta
    ref = torch.zeros_like(x)
    prev = torch.zeros(x.shape[0], 1)
    rect = x.abs()
    for n in range(x.shape[1]):
        prev = beta * prev + (1.0 - beta) * rect[:, n, :]
        ref[:, n, :] = prev

    assert torch.allclose(fast, ref, atol=1e-5)
