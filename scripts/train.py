#!/usr/bin/env python3
"""Main training loop for the power-sag model.

Trains the full gray-box model end-to-end on a paired (DI, amp-output)
recording, carrying the latent supply state across segment boundaries so that
sag time constants longer than a single segment are exercised.

Usage
-----
    python scripts/train.py --input di.wav --target amp.wav \
        --config configs/default.yaml --out model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow running directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from power_sag import load_config  # noqa: E402
from power_sag.cabinet import CabinetIR  # noqa: E402
from power_sag.data import SequenceDataset  # noqa: E402
from power_sag.loss import ESRLoss, PreEmphasisLoss  # noqa: E402
from power_sag.model import PowerSagModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the power-sag amp model.")
    parser.add_argument("--input", required=True, help="Input DI .wav file.")
    parser.add_argument("--target", required=True, help="Target amp output .wav file.")
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument("--out", default="power_sag_model.pt", help="Output checkpoint.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    import soundfile as sf

    x, _ = sf.read(args.input, dtype="float32", always_2d=False)
    t, _ = sf.read(args.target, dtype="float32", always_2d=False)

    dataset = SequenceDataset(
        x, t, segment_len=cfg["segment_len"], V_idle=cfg["V_idle"], normalize=True
    )

    cabinet = None
    if cfg.get("cabinet_ir"):
        cabinet = CabinetIR(cfg["cabinet_ir"]).to(device)

    model = PowerSagModel.from_config(cfg, cabinet=cabinet).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    esr = ESRLoss()
    preemph = PreEmphasisLoss(coeff=cfg["preemph_coeff"])
    lam = cfg["preemph_weight"]

    for epoch in range(cfg["epochs"]):
        model.train()
        dataset.reset_states()
        running = 0.0
        # Process segments strictly in order so the supply state carries over.
        for idx in range(len(dataset)):
            inp, tgt, V0 = dataset[idx]
            inp = inp.unsqueeze(0).to(device)
            tgt = tgt.unsqueeze(0).to(device)
            V0 = V0.view(1, 1).to(device)

            optimizer.zero_grad()
            y, V_final, _ = model(inp, V0=V0, return_state=True)
            loss = esr(y, tgt) + lam * preemph(y, tgt)
            loss.backward()
            optimizer.step()

            dataset.update_state(idx, V_final.detach())
            running += loss.item()

        print(f"epoch {epoch + 1:03d}/{cfg['epochs']}  loss={running / max(len(dataset), 1):.6f}")

    torch.save({"model_state": model.state_dict(), "config": cfg}, args.out)
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
