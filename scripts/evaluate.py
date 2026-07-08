#!/usr/bin/env python3
"""Run the sag evaluation protocol on a trained model.

Generates the targeted test signals from the research notes, runs them through
a trained checkpoint, and reports sag-specific measurements (recovery time
constant, attack/bloom shape) that standard ESR metrics do not expose.

Usage
-----
    python scripts/evaluate.py --checkpoint model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from power_sag.evaluation import SagEvaluator  # noqa: E402
from power_sag.nn import PowerSagModel  # noqa: E402
from power_sag.utils import resolve_device  # noqa: E402


def rms_envelope(signal: torch.Tensor, win: int) -> torch.Tensor:
    """Sliding-window RMS envelope of a ``(1, T, 1)`` signal."""
    x = signal.reshape(-1)
    power = x ** 2
    kernel = torch.ones(1, 1, win) / win
    padded = torch.nn.functional.pad(power.view(1, 1, -1), (win - 1, 0))
    return torch.sqrt(torch.nn.functional.conv1d(padded, kernel).reshape(-1) + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sag behaviour.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = PowerSagModel.from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ev = SagEvaluator(fs=cfg["fs"])
    fs = float(cfg["fs"])

    print("== Sag evaluation protocol ==")

    # 1. Sustained chord: inspect the internal supply trajectory + recovery.
    chord = ev.sustained_chord_signal(1.0, amplitude=0.9)
    with torch.no_grad():
        V_seq, _ = model.supply_trajectory(chord)
    V = V_seq.reshape(-1)
    print(f"sustained chord: V_B+ droops from {V[0]:.1f}V to {V.min():.1f}V")

    # 2. Dynamic recovery: loud -> quiet, fit the recovery time constant.
    dyn = ev.dynamic_signal(0.5, 1.0, loud_amp=0.9, quiet_amp=0.05)
    with torch.no_grad():
        V_dyn, _ = model.supply_trajectory(dyn)
    loud_samples = int(0.5 * fs)
    recovery = V_dyn.reshape(-1)[loud_samples:]
    tau = ev.fit_recovery_curve(recovery, fs=fs)
    print(f"recovery time constant: tau = {tau * 1e3:.2f} ms")

    # 3. Attack/bloom shape of the output.
    with torch.no_grad():
        y = model(chord)
    env = rms_envelope(y, win=int(0.01 * fs))
    print(f"output attack peak envelope: {env.max():.4f}")


if __name__ == "__main__":
    main()
