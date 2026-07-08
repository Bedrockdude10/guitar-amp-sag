#!/usr/bin/env python3
"""Generate and document a sag-exercising capture signal.

Standard NAM training signals (chirps, noise sweeps) do not systematically
exercise the sag trajectory.  This script builds a purpose-designed capture
signal -- sustained power chords, high-gain single notes, and deliberate
forte/piano dynamic transitions -- and prints instructions for recording the
amplifier's response.

Usage
-----
    python scripts/capture.py --out capture_di.wav --minutes 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from power_sag import load_config  # noqa: E402
from power_sag.evaluation import SagEvaluator  # noqa: E402

INSTRUCTIONS = """
Capture procedure
=================
1. Connect your guitar/DI reamp box output to the amplifier input.
2. Simultaneously record:
     - the DI signal  (this generated file, or your own playing) -> INPUT
     - the miked/loadbox amp output                              -> TARGET
   Both at {fs} Hz, 24-bit, time-aligned (compensate for any latency).
3. If instrumenting B+ directly: record the buffered resistor-divider probe
   as a third channel for supervised V_B+ training.
4. Play material that exercises sag:
     - sustained power chords held into bloom,
     - hard single-note attacks at high gain,
     - forte -> piano -> forte dynamic swells,
     - palm-muted chugs alternating with open ringing chords.
5. Keep ~10% of the recording aside as a held-out sag test set.
Train with:  python scripts/train.py --input INPUT.wav --target TARGET.wav
""".strip()


def build_capture_signal(fs: float, minutes: float) -> np.ndarray:
    """Concatenate sag-exercising segments into one reference DI signal."""
    ev = SagEvaluator(fs=fs)
    blocks = []
    # Alternate sustained chords and loud->quiet dynamic transitions.
    frequencies = [82.41, 110.0, 146.83, 98.0]  # E2, A2, D3, G2
    while sum(b.shape[1] for b in blocks) < int(minutes * 60 * fs):
        for f0 in frequencies:
            blocks.append(ev.sustained_chord_signal(1.0, f0=f0, amplitude=0.9))
            blocks.append(ev.dynamic_signal(0.75, 0.75, f0=f0))
    signal = np.concatenate([b.squeeze().numpy() for b in blocks])
    total = int(minutes * 60 * fs)
    return signal[:total].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sag capture signal.")
    parser.add_argument("--out", default="capture_di.wav")
    parser.add_argument("--minutes", type=float, default=3.0)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    fs = float(cfg["fs"])
    signal = build_capture_signal(fs, args.minutes)

    try:
        import soundfile as sf

        sf.write(args.out, signal, int(fs))
        print(f"wrote {len(signal) / fs:.1f}s capture signal to {args.out}")
    except Exception as exc:  # pragma: no cover - IO convenience
        print(f"could not write wav ({exc}); signal shape = {signal.shape}")

    print()
    print(INSTRUCTIONS.format(fs=int(fs)))


if __name__ == "__main__":
    main()
