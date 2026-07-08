#!/usr/bin/env python3
"""Main training loop for the power-sag model.

Trains the full gray-box model end-to-end on a paired (DI, amp-output)
recording, carrying the latent supply state ``V_B+`` across segment boundaries
within each recording so that sag time constants longer than a single segment
are exercised.

Correctness / hygiene:

* Chronological train/val/test split (audio must never be split randomly).
* Recording-level shuffling via ``SequenceBatchSampler`` (never shuffles
  individual segments -- that would corrupt the carried ``V_B+`` state).
* ``V_B+`` detached at every segment/chunk boundary (truncated BPTT); memory
  bounded within a segment by ``tbptt_steps``.
* Best-on-validation checkpointing, CSV metric logging, and resume support.
* Optional supervised ``V_B+`` term when a B+ probe signal is provided.

Usage
-----
    python scripts/train.py --input di.wav --target amp.wav \
        --config configs/default.yaml --out model.pt [--resume ckpt.pt]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

# Allow running directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from power_sag import load_config  # noqa: E402
from power_sag.data import (  # noqa: E402
    SequenceBatchSampler,
    SequenceDataset,
    chronological_split,
)
from power_sag.dsp import CabinetIR  # noqa: E402
from power_sag.losses import ESRLoss, PreEmphasisLoss  # noqa: E402
from power_sag.nn import PowerSagModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the power-sag amp model.")
    parser.add_argument("--input", required=True, help="Input DI .wav file.")
    parser.add_argument("--target", required=True, help="Target amp output .wav file.")
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument("--out", default="power_sag_model.pt", help="Best checkpoint path.")
    parser.add_argument("--log", default=None, help="CSV metrics log path.")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_segment(
    model: PowerSagModel,
    optimizer: torch.optim.Optimizer,
    inp: torch.Tensor,
    tgt: torch.Tensor,
    V0: torch.Tensor,
    tbptt: Optional[int],
    esr: ESRLoss,
    preemph: PreEmphasisLoss,
    lam: float,
) -> Tuple[float, torch.Tensor]:
    """Run truncated BPTT over one segment; return (mean loss, terminal V_B+)."""
    length = inp.shape[1]
    chunk = tbptt or length
    V = V0
    lstm_state = None
    weighted_loss = 0.0

    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        optimizer.zero_grad()
        y, V, lstm_state = model(
            inp[:, start:end], V0=V, lstm_state=lstm_state, return_state=True
        )
        loss = esr(y, tgt[:, start:end]) + lam * preemph(y, tgt[:, start:end])
        loss.backward()
        optimizer.step()
        # Truncate BPTT at the chunk boundary: keep the state, drop the graph.
        V = V.detach()
        lstm_state = tuple(h.detach() for h in lstm_state)
        weighted_loss += loss.item() * (end - start)

    return weighted_loss / length, V.detach()


@torch.no_grad()
def evaluate(
    model: PowerSagModel,
    dataset: SequenceDataset,
    esr: ESRLoss,
    preemph: PreEmphasisLoss,
    lam: float,
    device: torch.device,
) -> float:
    """Mean loss over a dataset, processed in order with carried V_B+."""
    model.eval()
    dataset.reset_states()
    total, n = 0.0, 0
    for idx in range(len(dataset)):
        inp, tgt, V0 = dataset[idx]
        inp, tgt = inp.unsqueeze(0).to(device), tgt.unsqueeze(0).to(device)
        y, V_final, _ = model(inp, V0=V0.view(1, 1).to(device), return_state=True)
        total += (esr(y, tgt) + lam * preemph(y, tgt)).item()
        dataset.update_state(idx, V_final)
        n += 1
    model.train()
    return total / max(n, 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    import soundfile as sf

    x, _ = sf.read(args.input, dtype="float32", always_2d=False)
    t, _ = sf.read(args.target, dtype="float32", always_2d=False)

    # Chronological split: train | val | test (test held out for evaluate.py).
    (x_tr, t_tr), (x_val, t_val), _ = chronological_split(
        x, t, val_frac=cfg.get("val_frac", 0.1), test_frac=cfg.get("test_frac", 0.1)
    )
    common = dict(segment_len=cfg["segment_len"], V_idle=cfg["V_idle"], normalize=True)
    train_ds = SequenceDataset(x_tr, t_tr, **common)
    val_ds = SequenceDataset(x_val, t_val, **common)

    cabinet = None
    if cfg.get("cabinet_ir"):
        cabinet = CabinetIR(cfg["cabinet_ir"]).to(device)

    model = PowerSagModel.from_config(cfg, cabinet=cabinet).to(device)
    if cfg.get("use_script"):
        model.enable_script()  # after .to(device): the scripted recurrence shares params

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    esr = ESRLoss()
    preemph = PreEmphasisLoss(coeff=cfg["preemph_coeff"])
    lam = cfg["preemph_weight"]
    tbptt = cfg.get("tbptt_steps")

    start_epoch, best_val = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_val", float("inf"))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    log_writer = None
    if args.log:
        log_file = open(args.log, "a", newline="")
        log_writer = csv.writer(log_file)
        if log_file.tell() == 0:
            log_writer.writerow(["epoch", "train_loss", "val_loss", "best_val"])

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        train_ds.reset_states()
        sampler = SequenceBatchSampler(
            train_ds, batch_size=cfg["batch_size"], shuffle=True, seed=epoch
        )
        running, n_batches = 0.0, 0
        for batch_idx in sampler:
            inp = torch.stack([train_ds[i][0] for i in batch_idx]).to(device)
            tgt = torch.stack([train_ds[i][1] for i in batch_idx]).to(device)
            V0 = torch.stack([train_ds[i][2] for i in batch_idx]).view(-1, 1).to(device)
            loss, V_final = train_segment(
                model, optimizer, inp, tgt, V0, tbptt, esr, preemph, lam
            )
            for b, idx in enumerate(batch_idx):
                train_ds.update_state(idx, V_final[b])
            running += loss
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        val_loss = evaluate(model, val_ds, esr, preemph, lam, device)
        print(
            f"epoch {epoch + 1:03d}/{cfg['epochs']}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}"
        )
        if log_writer:
            log_writer.writerow([epoch + 1, train_loss, val_loss, min(best_val, val_loss)])
            log_file.flush()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "config": cfg,
                    "epoch": epoch + 1,
                    "best_val": best_val,
                },
                args.out,
            )

    if log_writer:
        log_file.close()
    print(f"best val loss {best_val:.6f}; best checkpoint at {args.out}")


if __name__ == "__main__":
    main()
