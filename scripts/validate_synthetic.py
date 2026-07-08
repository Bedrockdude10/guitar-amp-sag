#!/usr/bin/env python3
"""Identifiability validation on synthetic ground-truth data.

Runs the core de-risking experiment: generate ``(x, y, V_B+_true)`` from a known
power-supply ODE + coupling + sag nonlinearity, then fit the gray-box model
starting from *wrong* physics parameters and measure how well it recovers

  * the physics parameters (R_eff, C1), and
  * the latent V_B+ trajectory (correlation with ground truth),

from the output audio alone.  If the model cannot recover known sag from
synthetic data, it will not learn it from a real amplifier -- so run this before
data collection.

Usage
-----
    python scripts/validate_synthetic.py --steps 400 --duration 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from power_sag import load_config  # noqa: E402
from power_sag.losses import ESRLoss, PreEmphasisLoss  # noqa: E402
from power_sag.nn import PowerSagModel  # noqa: E402
from power_sag.synthetic import SyntheticSagAmp  # noqa: E402
from power_sag.utils import configure_backends, resolve_device  # noqa: E402


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a = (a - a.mean()).reshape(-1)
    b = (b - b.mean()).reshape(-1)
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic identifiability study.")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--duration", type=float, default=2.0, help="seconds of data")
    p.add_argument("--segment", type=int, default=4800, help="training segment length")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument(
        "--supervise-vb",
        type=float,
        default=0.0,
        metavar="BETA",
        help="Weight of a supervised V_B+ term (simulates a B+ probe). 0 = "
        "latent-only. A positive value pins the physics to the measured state.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    configure_backends(device)

    # --- Ground truth: known physics + coupling + sag nonlinearity ---------
    true_R_eff, true_C1 = 300.0, 22e-6
    amp = SyntheticSagAmp(fs=48000.0, R_eff=true_R_eff, C1=true_C1)
    x = amp.generate_excitation(args.duration, seed=args.seed).to(device)
    _, y, V_true = amp.generate(x)
    y, V_true = y.to(device), V_true.to(device)

    # --- Model: deliberately wrong initial physics, then learn -------------
    cfg = load_config()
    cfg.update(R_eff=150.0, C1=47e-6, hidden_size=32, coupling_hidden=16,
               learn_R_eff=True, learn_C1=True, learn_V_oc=False)
    model = PowerSagModel.from_config(cfg).to(device)
    if cfg.get("use_script"):
        model.enable_script()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    esr, preemph = ESRLoss(), PreEmphasisLoss(coeff=cfg["preemph_coeff"])

    init_R = model.physics._R_eff.item()
    init_C = model.physics.C1.item()
    seg = args.segment
    n_seg = max(x.shape[1] // seg, 1)

    beta = args.supervise_vb
    delta_V = float(cfg["delta_V"])
    V_idle = float(cfg["V_idle"])
    mode = f"supervised V_B+ (beta={beta})" if beta > 0 else "latent (audio only)"
    print("== Synthetic identifiability study ==")
    print(f"mode         : {mode}")
    print(f"ground truth : R_eff={true_R_eff:.1f}  C1={true_C1*1e6:.1f}uF")
    print(f"init (wrong) : R_eff={init_R:.1f}  C1={init_C*1e6:.1f}uF\n")

    for step in range(args.steps):
        # Stateful pass over the recording, carrying detached V_B+.
        V0 = None
        total = 0.0
        for s in range(n_seg):
            xs = x[:, s * seg : (s + 1) * seg]
            ts = y[:, s * seg : (s + 1) * seg]
            opt.zero_grad()
            # Compute the supply trajectory once, then the audio path on it, so
            # a supervised V_B+ term can reuse V_seq without a second recurrence.
            V_seq, V_next = model.supply_trajectory(xs, V0=V0)
            pred, _ = model.audio(xs, V_seq)
            loss = esr(pred, ts) + cfg["preemph_weight"] * preemph(pred, ts)
            if beta > 0:  # B+ probe: supervise the (normalised) supply state
                vt = V_true[:, s * seg : (s + 1) * seg]
                loss = loss + beta * (
                    ((V_seq - V_idle) / delta_V - (vt - V_idle) / delta_V) ** 2
                ).mean()
            loss.backward()
            opt.step()
            V0 = V_next.detach()
            total += loss.item()
        if step % max(args.steps // 10, 1) == 0 or step == args.steps - 1:
            print(
                f"step {step:4d}  esr={total / n_seg:.5f}  "
                f"R_eff={model.physics._R_eff.item():7.1f}  "
                f"C1={model.physics.C1.item()*1e6:5.1f}uF"
            )

    with torch.no_grad():
        V_pred, _ = model.supply_trajectory(x)
    rec_R, rec_C = model.physics._R_eff.item(), model.physics.C1.item()

    print("\n== Recovery ==")
    print(f"R_eff : {init_R:.1f} -> {rec_R:.1f}  (true {true_R_eff:.1f}, "
          f"err {abs(rec_R - true_R_eff) / true_R_eff * 100:.1f}%)")
    print(f"C1    : {init_C*1e6:.1f} -> {rec_C*1e6:.1f}uF  (true {true_C1*1e6:.1f}uF, "
          f"err {abs(rec_C - true_C1) / true_C1 * 100:.1f}%)")
    print(f"latent V_B+ correlation with ground truth: {correlation(V_pred, V_true):.3f}")


if __name__ == "__main__":
    main()
