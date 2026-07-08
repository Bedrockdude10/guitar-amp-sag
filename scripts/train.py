#!/usr/bin/env python3
"""Main training loop for the power-sag model.

Trains the full gray-box model end-to-end on one or more paired (DI, amp-output)
recordings, carrying the latent supply state ``V_B+`` across segment boundaries
within each recording so that sag time constants longer than a single segment
are exercised.

Key correctness points:

* Segments are drawn with :class:`SequenceBatchSampler`, which shuffles at the
  *recording* level and keeps each recording's segments in order.  A plain
  ``DataLoader(shuffle=True)`` would shuffle individual segments and feed most
  of them a ``V_B+`` initial condition from an unrelated segment.
* The carried ``V_B+`` is detached at every segment/chunk boundary (truncated
  BPTT): state continuity without an ever-growing autograd graph.
* Truncated BPTT within a segment (``tbptt_steps``) bounds memory; the LSTM and
  supply state are carried but detached between chunks.

Usage
-----
    python scripts/train.py --input di.wav --target amp.wav \
        --config configs/default.yaml --out model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Allow running directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from power_sag import load_config  # noqa: E402
from power_sag.data import SequenceBatchSampler, SequenceDataset  # noqa: E402
from power_sag.dsp import CabinetIR  # noqa: E402
from power_sag.losses import ESRLoss, PreEmphasisLoss  # noqa: E402
from power_sag.nn import PowerSagModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the power-sag amp model.")
    parser.add_argument("--input", required=True, help="Input DI .wav file.")
    parser.add_argument("--target", required=True, help="Target amp output .wav file.")
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument("--out", default="power_sag_model.pt", help="Output checkpoint.")
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
    """Run truncated BPTT over one segment; return (mean loss, terminal V_B+).

    The segment is processed in chunks of ``tbptt`` samples.  After each chunk
    the supply and LSTM states are detached, truncating the backprop graph while
    preserving forward-state continuity.
    """
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
    if cfg.get("use_script"):
        model.enable_script()  # after .to(device): the scripted recurrence shares params

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    esr = ESRLoss()
    preemph = PreEmphasisLoss(coeff=cfg["preemph_coeff"])
    lam = cfg["preemph_weight"]
    tbptt = cfg.get("tbptt_steps")

    for epoch in range(cfg["epochs"]):
        model.train()
        dataset.reset_states()
        # Recording-level shuffle, segment order preserved within each recording.
        sampler = SequenceBatchSampler(
            dataset, batch_size=cfg["batch_size"], shuffle=True, seed=epoch
        )

        running, n_batches = 0.0, 0
        for batch_idx in sampler:
            inp = torch.stack([dataset[i][0] for i in batch_idx]).to(device)
            tgt = torch.stack([dataset[i][1] for i in batch_idx]).to(device)
            V0 = torch.stack([dataset[i][2] for i in batch_idx]).view(-1, 1).to(device)

            loss, V_final = train_segment(
                model, optimizer, inp, tgt, V0, tbptt, esr, preemph, lam
            )

            # Carry each lane's terminal state to its next in-recording segment.
            for b, idx in enumerate(batch_idx):
                dataset.update_state(idx, V_final[b])
            running += loss
            n_batches += 1

        print(f"epoch {epoch + 1:03d}/{cfg['epochs']}  loss={running / max(n_batches, 1):.6f}")

    torch.save({"model_state": model.state_dict(), "config": cfg}, args.out)
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
