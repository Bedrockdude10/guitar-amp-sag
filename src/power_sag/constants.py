"""Central default constants -- the single Python source of truth.

These mirror the values in ``configs/default.yaml``, which remains the *runtime*
source of truth: production entry points load the YAML and pass explicit values.
These module-level constants exist so that the **Python constructor defaults**
(used by tests, synthetic ground-truth generation, and direct instantiation)
share one definition instead of being restated verbatim in every ``__init__`` --
which previously let the audio path, coupling net, physics ODE and synthetic amp
silently drift out of agreement.

The module imports nothing from the package, so it is safe to import anywhere.
"""

from __future__ import annotations

# --- Sampling ---
SAMPLE_RATE: float = 48000.0  # Hz

# --- Power-supply physics (see configs/default.yaml) ---
V_OC: float = 420.0  # volts, open-circuit B+ (transformer)
V_IDLE: float = 415.0  # volts, quiescent B+ at idle
C1: float = 22e-6  # farads, first filter capacitor
R_EFF: float = 300.0  # ohms, effective source resistance (transformer + GZ34)
DELTA_V: float = 40.0  # volts, expected sag range for FiLM normalisation

# Nonlinear R_eff model: R_eff(I) = R0 + R1 * softplus(I * K)
R0: float = 250.0  # ohms, floor
R1: float = 100.0  # ohms, current-dependent scale
K: float = 1.0  # 1/A, softplus input scale

# --- Training / segmentation ---
SEGMENT_LEN: int = 24000  # samples (0.5 s at 48 kHz)

# --- Numerical guards ---
EPS: float = 1e-12  # divide-by-zero guard (normalisation, metrics, cabinet)
