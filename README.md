# power-sag-model

Physics-informed gray-box neural model of **power supply sag** in a tube guitar
amplifier (Fender Deluxe Reverb, GZ34 rectifier).

The amplifier is decomposed into two coupled subsystems, following the
Universal Differential Equations framework:

* a **slow** power-supply subsystem — a first-order RC ODE governing the B+
  supply voltage, discretised at sample rate with explicit Euler and unrolled
  differentiably through time;
* a **fast** neural audio path — an LSTM conditioned on the supply voltage via
  Feature-wise Linear Modulation (FiLM);

coupled through a small learned **load-current** network. The whole graph is
differentiable, so the physics parameters (`R_eff`, `C1`, optionally `V_oc`),
the coupling network, and the audio path train jointly by backpropagation.

```
x[n] ─────────────────────────────► audio path g_φ(x[n], V_B+[n]) ─► y[n]
  │                                        ▲
  └─► coupling f_θ(x[n], V_B+[n]) = I_load  │
                 │                          │
                 ▼                          │
        Euler ODE step ► V_B+[n+1] ─────────┘
```

## Layout

Modules are grouped by role. Every public class is re-exported at the package
root, so `from power_sag import PowerSagModel` works regardless of layout.

```
src/power_sag/
  config.py            load_config (reads configs/*.yaml)
  utils.py             audio I/O, normalisation, supply scaling, time-alignment
  losses.py            ESRLoss, PreEmphasisLoss
  synthetic.py         SyntheticSagAmp — known-physics ground truth generator
  physics/
    ode.py             PowerSupplyODE — differentiable Euler ODE + GZ34 R_eff
  nn/
    coupling.py        CouplingNetwork — MLP (x, V_B+) -> I_load (non-negative)
    film.py            FiLMLayer — feature-wise linear modulation
    audio_model.py     PowerSagLSTM — LSTM audio path, FiLM-conditioned on V_B+
    recurrence.py      SupplyRecurrence — TorchScript-able coupling+physics loop
    model.py           PowerSagModel — full end-to-end coupled model
    baselines.py       UnconditionedLSTM, ConditionedLSTMNoPhysics (ablations)
  data/
    datasets.py        AudioDataset, SequenceDataset, SequenceBatchSampler,
                       chronological_split
  dsp/
    cabinet.py         CabinetIR — fixed (non-trainable) speaker/mic convolution
  evaluation/
    evaluator.py       SagEvaluator — signals, recovery fitting, 4-part protocol
configs/default.yaml   all hyperparameters (Deluxe Reverb AB763 defaults)
scripts/               train.py, capture.py, evaluate.py, validate_synthetic.py
tests/                 pytest suite covering every mathematical invariant
.github/workflows/     ci.yml — runs pytest on push / PR
```

Shared helpers in `utils.py` (`load_audio`, `to_mono_tensor`, `normalize_audio`,
`normalize_supply`) removed the audio-loading duplication between `data` and
`dsp`, and the `(V_B+ - V_idle)/delta_V` conditioning duplication between the
coupling and audio-path networks.

## Stateful training: correctness and performance

Power sag is a slow hidden state, so training carries `V_B+` across segment
boundaries. A few things this depends on — each locked down by tests:

- **Sequence-level shuffling.** A plain `DataLoader(shuffle=True)` shuffles
  individual segments and would feed most of them a `V_B+` initial condition
  from an unrelated segment — silent corruption. Use `SequenceBatchSampler`,
  which shuffles at the *recording* level and keeps each recording's segments in
  order. `SequenceDataset` supports multiple recordings and never carries state
  across a recording boundary (each recording's first segment starts at
  `V_idle`).
- **Detached state at boundaries.** The carried `V_B+` is `.detach()`ed, so the
  autograd graph does not grow with the number of sequential segments (truncated
  BPTT at the boundary — state continuity without gradient continuity).
- **Truncated BPTT within a segment** (`tbptt_steps`). A 0.5 s segment is 24k
  LSTM steps; `train.py` processes it in chunks, carrying and detaching both the
  supply and LSTM state between chunks to bound memory. Sag time constants
  (~15 ms ≈ 720 samples) fit comfortably inside a chunk.
- **TorchScript fast path** (`use_script`, opt-in via `model.enable_script()`).
  The sample-by-sample coupling+physics recurrence is dominated by Python
  dispatch; scripting it is ~4× faster on CPU (more on GPU) with bit-identical
  results. It reuses `CouplingNetwork.forward` and `euler_step` verbatim (an
  equivalence test guards against drift) and shares parameters with the eager
  modules, so it trains normally.
- **FiLM identity init.** The FiLM biases init to `γ=1, β=0`, so at `V_B+ =
  V_idle` (conditioning input `0`) the layer is exactly the identity — it does
  not zero out the audio path at the start of training.

## Install

```bash
pip install -e .          # torch, numpy, scipy, soundfile, pyyaml
pip install -e ".[dev]"   # + pytest
```

## Test

```bash
pytest tests/ -v
```

`tests/test_<module>.py` cover each component's mathematical invariants in
isolation; `tests/test_integration.py` wires them together — the stateful data
loader carrying `V_B+` across segment boundaries through the real model, a full
train step (dataset → model → loss → backward → optimiser), on-disk WAV loading,
and the evaluator's signals run end-to-end.

## Train / evaluate

```bash
python scripts/capture.py  --out capture_di.wav --minutes 3
python scripts/train.py    --input capture_di.wav --target amp_out.wav \
    --out model.pt --log metrics.csv          # chronological split, best-val ckpt
python scripts/train.py    ... --resume model.pt             # resume training
python scripts/evaluate.py --checkpoint model.pt
```

`train.py` splits the recording chronologically (train/val/test), shuffles only
at the recording level, uses truncated BPTT, saves the best-on-validation
checkpoint, logs metrics to CSV, and resumes. When a B+ probe signal is
available, `SequenceDataset(..., measured_vb=...)` exposes per-segment `V_B+` for
supervised training of the supply state.

## Before data collection: synthetic identifiability

Run the core de-risking experiment with **no hardware** — fit the model to data
from a known ODE + coupling + sag nonlinearity and measure what it recovers:

```bash
python scripts/validate_synthetic.py --steps 400 --duration 2.0    # audio-only
python scripts/validate_synthetic.py ... --supervise-vb 5.0        # + B+ probe
```

What this surfaced (and why it was worth running first):

1. **The physics was untrainable as first parameterised.** Raw `C1 (~2e-5)` and
   `R_eff (~3e2)` differ by orders of magnitude; Adam's ~`lr` steps blew up the
   capacitance and froze the resistance. Fixed by learning **log-parameters**
   (`PowerSupplyODE` now stores `log C1`, `log R_eff`, …). After the fix the fit
   is stable and drives ESR down to ~5e-3, and the recovered latent `V_B+`
   correlates ~0.8 with ground truth.
2. **Individual physics parameters are not identifiable from audio alone.** Even
   at near-zero ESR, `R_eff` and `C1` settle far from truth — the flexible audio
   path compensates, and a free coupling current trades off against `R_eff`/`C1`
   (only the trajectory and the time constant are constrained). Supervising
   `V_B+` (the B+ probe) improves the trajectory but not the individual
   parameters. **Implication for capture:** pin `C1` from the schematic and/or
   constrain the coupling if physical parameter recovery is a goal; otherwise
   treat `V_B+` as a recovered *trajectory*, not a set of identified constants.

## Baselines and the sag protocol

`nn.UnconditionedLSTM` (black-box, no sag) and `nn.ConditionedLSTMNoPhysics`
(learned slow state instead of the ODE — the physics ablation) share the
`model(x) -> y` convention. `SagEvaluator.run_protocol(model)` runs the four
sag-targeted tests (attack/bloom, recovery time constant, pre-sagged vs cold
attack, quiet-to-loud) on any of them for a fair comparison.

## Capture readiness

`utils.align_signals(di, amp, max_lag)` compensates DI↔output latency by
cross-correlation before datasets are built — misaligned pairs make ESR
meaningless. `data.chronological_split` produces the time-ordered train/val/test
partitions.

## Physics

The power-supply ODE (Deluxe Reverb, GZ34 rectifier):

```
C1 · dV_B+/dt = (V_oc - V_B+) / R_eff(I_load) - I_load
```

discretised with explicit Euler at `Ts = 1/fs`:

```
V_B+[n+1] = V_B+[n] + (Ts / C1) · ((V_oc - V_B+[n]) / R_eff - I_load[n])
```

Stability holds since `Ts/τ = Ts/(R_eff·C1) ≈ 0.003 ≪ 1`. `R_eff` is either a
learnable scalar or the current-dependent GZ34 model
`R_eff(I) = R0 + R1·softplus(I·k)`.
