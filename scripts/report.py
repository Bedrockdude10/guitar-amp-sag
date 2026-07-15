#!/usr/bin/env python3
"""Generate a paper-ready report: metrics JSON + figures.

Runs the reporting layer on a trained checkpoint (and/or a training CSV log) and
writes a metrics ``report.json`` plus PNG figures into an output directory. With
no checkpoint it reports an untrained model on synthetic ground truth, which is
still useful for wiring the pipeline and sanity-checking ranges.

Usage
-----
    python scripts/report.py --checkpoint model.pt --log metrics.csv --outdir report/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we only save files
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from power_sag import load_config  # noqa: E402
from power_sag.nn import PowerSagModel  # noqa: E402
from power_sag.reporting import plots, report  # noqa: E402
from power_sag.synthetic import SyntheticSagAmp  # noqa: E402
from power_sag.utils import resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate metrics + figures.")
    p.add_argument("--checkpoint", default=None, help="Trained checkpoint (.pt).")
    p.add_argument("--log", default=None, help="Training CSV log for loss curves.")
    p.add_argument("--outdir", default="report", help="Output directory.")
    p.add_argument("--duration", type=float, default=1.0, help="Synthetic seconds.")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg = ckpt["config"]
        model = PowerSagModel.from_config(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
    else:
        cfg = load_config()
        cfg.update(hidden_size=8, coupling_hidden=8)
        model = PowerSagModel.from_config(cfg).to(device)
        print("no checkpoint: reporting an untrained model on synthetic data")
    model.eval()

    fs = float(cfg["fs"])
    amp = SyntheticSagAmp(fs=fs)
    x = amp.generate_excitation(args.duration, seed=0)
    _, y_true, v_true = amp.generate(x)

    # Metrics report (JSON).
    rep = report.synthetic_report(model, amp=amp, duration=args.duration)
    report.to_json(rep, str(outdir / "report.json"))
    print(f"metrics -> {outdir / 'report.json'}")
    print(f"  ESR={rep['regression']['esr']:.4f}  "
          f"SNR={rep['regression']['snr_db']:.1f} dB")
    if "trajectory" in rep:
        print(f"  V_B+ trajectory corr={rep['trajectory']['correlation']:.3f}")

    # Figures.
    with torch.no_grad():
        y_pred = model(x)
        v_pred, _ = model.supply_trajectory(x)
    plots.save_figure(
        plots.plot_signal_comparison(y_pred, y_true, fs=fs), str(outdir / "signal.png"))
    plots.save_figure(
        plots.plot_output_and_supply(y_pred, v_pred, fs=fs, v_reference=v_true),
        str(outdir / "supply.png"))

    from power_sag.evaluation import SagEvaluator

    ev = SagEvaluator(fs=fs)
    dyn = ev.dynamic_signal(0.3, 0.6, loud_amp=0.9, quiet_amp=0.05)
    with torch.no_grad():
        ev_env = ev.rms_envelope(model(dyn))[int(0.3 * fs):]
    plots.save_figure(plots.plot_recovery_fit(ev_env, fs=fs), str(outdir / "recovery.png"))

    if args.log and Path(args.log).exists():
        rows = list(csv.DictReader(open(args.log)))
        if rows:
            epochs = [int(r["epoch"]) for r in rows]
            train = [float(r["train_loss"]) for r in rows]
            val = [float(r["val_loss"]) for r in rows]
            plots.save_figure(
                plots.plot_training_curves(epochs, train, val),
                str(outdir / "training.png"))
            print(f"training curve -> {outdir / 'training.png'}")

    print(f"figures -> {outdir}/*.png")


if __name__ == "__main__":
    main()
