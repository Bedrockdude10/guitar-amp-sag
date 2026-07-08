# power-sag-model

Physics-informed gray-box neural model of **power supply sag** in a tube guitar
amplifier (Fender Deluxe Reverb, GZ34 rectifier).

Neural amp models capture static distortion well but miss *sag* — the droop of
the B+ supply voltage under load that softens attacks and adds bloom. Sag is a
slow (~10–200 ms) hidden state, longer than typical training buffers and never
directly observed. Following the Universal Differential Equations idea, we split
the amp into two coupled subsystems and train them end-to-end:

* a **slow** power supply — a first-order RC ODE for the B+ voltage, discretised
  with explicit Euler and unrolled differentiably through time;
* a **fast** neural audio path — an LSTM conditioned on that voltage via
  Feature-wise Linear Modulation (FiLM);

joined by a small learned **load-current** network. The whole graph is
differentiable, so the physics parameters, the coupling network, and the audio
path train jointly.

```
x[n] ─────────────────────────────► audio path g_φ(x[n], V_B+[n]) ─► y[n]
  │                                        ▲
  └─► coupling f_θ(x[n], V_B+[n]) = I_load  │
                 │                          │
                 ▼                          │
        Euler ODE step ► V_B+[n+1] ─────────┘
```

The power-supply ODE, discretised at `Ts = 1/fs`:

```
C1 · dV_B+/dt = (V_oc - V_B+)/R_eff(I_load) - I_load
V_B+[n+1]     = V_B+[n] + (Ts/C1)·((V_oc - V_B+[n])/R_eff - I_load[n])
```

Explicit Euler is stable here since `Ts/τ = Ts/(R_eff·C1) ≈ 0.003 ≪ 1`. `R_eff`
is a learnable scalar or the current-dependent GZ34 model
`R_eff(I) = R0 + R1·softplus(I·k)`. Full derivation and parameterisation notes
live in `physics/ode.py`.

## Quickstart

```bash
pip install -e ".[dev]"     # core + pytest + matplotlib (runs the full suite)
pip install -e ".[viz]"      # core + matplotlib only (for figures)
pytest tests/ -v
```

```bash
# 1. Validate the approach on synthetic data — no hardware needed (see Findings).
python scripts/validate_synthetic.py --steps 400 --duration 2.0

# 2. Generate a sag-exercising capture signal, then record DI + amp output.
python scripts/capture.py --out capture_di.wav --minutes 3

# 3. Train (chronological split, best-val checkpoint, CSV log; --resume to continue).
python scripts/train.py --input capture_di.wav --target amp_out.wav \
    --out model.pt --log metrics.csv --device auto

# 4. Run the sag-targeted evaluation protocol.
python scripts/evaluate.py --checkpoint model.pt
```

## Repository layout

Modules are grouped by role; every public class is re-exported at the package
root, so `from power_sag import PowerSagModel` works regardless of layout.

```
src/power_sag/
  config.py            load_config (reads configs/*.yaml)
  utils.py             audio I/O, normalisation, supply scaling, time-alignment,
                       device resolution
  losses.py            ESRLoss, PreEmphasisLoss
  synthetic.py         SyntheticSagAmp — known-physics ground-truth generator
  physics/ode.py       PowerSupplyODE — differentiable Euler ODE + GZ34 R_eff
  nn/
    coupling.py        CouplingNetwork — MLP (x, V_B+) -> I_load (non-negative)
    film.py            FiLMLayer — feature-wise linear modulation
    audio_model.py     PowerSagLSTM — LSTM audio path, FiLM-conditioned on V_B+
    recurrence.py      SupplyRecurrence — TorchScript-able coupling+physics loop
    model.py           PowerSagModel — full end-to-end coupled model
    baselines.py       UnconditionedLSTM, ConditionedLSTMNoPhysics (ablations)
  data/datasets.py     AudioDataset, SequenceDataset, SequenceBatchSampler,
                       chronological_split
  dsp/cabinet.py       CabinetIR — fixed (non-trainable) speaker/mic convolution
  evaluation/evaluator.py  SagEvaluator — signals, recovery fitting, 4-part protocol
  reporting/
    metrics.py         pure, range-checked metrics (ESR, SNR, correlation, sag …)
    report.py          schema'd JSON reports (regression + trajectory + protocol)
    plots.py           publication figures (colourblind-safe, no dual-axis)
    style.py           Okabe-Ito palette + recessive matplotlib style
configs/default.yaml   all hyperparameters (Deluxe Reverb AB763 defaults)
scripts/               capture, train, evaluate, validate_synthetic, report
tests/                 pytest suite (per-module invariants + integration)
```

## Design decisions that matter

Each is stated briefly here and documented in full — with rationale — at the
cited source, and pinned by a test. Read the code for the authoritative version.

- **Stateful training without corruption** (`data/datasets.py`, `scripts/train.py`).
  `V_B+` is carried across segments, so segments must be visited in order.
  `SequenceBatchSampler` shuffles only at the *recording* level; a plain
  `DataLoader(shuffle=True)` would silently feed unrelated initial conditions.
  State is **detached** at each boundary (truncated BPTT), and `tbptt_steps`
  bounds within-segment BPTT depth.
- **Log-space physics parameters** (`physics/ode.py`). `C1 (~2e-5)` and
  `R_eff (~3e2)` differ by orders of magnitude, which makes them untrainable
  under a single learning rate; learning `log C1`, `log R_eff`, … fixes this and
  enforces positivity. (This was found by the synthetic study — see Findings.)
- **TorchScript recurrence** (`nn/recurrence.py`). The coupling→ODE loop is
  inherently sequential (`V_B+` feedback), so it can't be vectorised; scripting
  it removes per-step dispatch (~4× on CPU) with bit-identical results and shared
  parameters. Opt in with `model.enable_script()`.
- **FiLM identity init** (`nn/film.py`). Biases init to `γ=1, β=0`, so at
  `V_B+ = V_idle` the layer is exactly the identity and does not zero the audio
  path at the start of training.
- **Fixed cabinet, fixed IR gradients** (`dsp/cabinet.py`). The speaker/mic
  response is a non-trainable convolution; its IR is a buffer and receives no
  gradient.
- **Devices** (`utils.py`). `resolve_device` picks CUDA → MPS → CPU;
  `--device auto` on every script. The model is float32 throughout, so MPS works
  unchanged. Mixed precision is deliberately **not** applied to the ODE: float16
  can't resolve ~0.01 V changes on a ~415 V rail across tens of thousands of
  Euler steps.

## Findings: synthetic identifiability

`scripts/validate_synthetic.py` fits the model to data from a *known* ODE +
coupling + sag nonlinearity (`SyntheticSagAmp`) and measures what it recovers —
the core de-risking step before recording an amp. Two results:

1. **The pipeline learns sag.** After the log-space fix the fit is stable, ESR
   falls to ~5e-3, and the recovered latent `V_B+` correlates ~0.8 with truth.
2. **Individual physics parameters are not identifiable from audio alone.** Even
   at near-zero ESR, `R_eff`/`C1` settle far from truth — the flexible audio path
   and a free coupling current trade off against them (only the trajectory and
   time constant are constrained). Supervising `V_B+` (`--supervise-vb`, i.e. a
   B+ probe) improves the trajectory but not the constants.

**Implication for capture:** if physical parameter recovery is a goal, pin `C1`
from the schematic and/or constrain the coupling; otherwise treat `V_B+` as a
recovered *trajectory*, not a set of identified constants.

## Baselines and evaluation

`nn.UnconditionedLSTM` (black-box) and `nn.ConditionedLSTMNoPhysics` (learned
slow state instead of the ODE — the physics ablation) share the `model(x) -> y`
convention. `SagEvaluator.run_protocol(model)` runs the four sag-targeted tests
(attack/bloom, recovery time constant, pre-sagged vs cold attack, quiet-to-loud)
on any model for a fair comparison. Standard ESR alone does not expose sag.

## Reporting and figures

`python scripts/report.py --checkpoint model.pt --log metrics.csv --outdir report/`
writes a schema'd `report.json` (regression + supply-trajectory + sag-protocol
metrics) and publication PNGs (output overlay, output+B+ stacked, recovery-curve
fit, training curves).

The layer is split so it can be **test-infected**: `reporting/metrics.py` is
pure functions with *known ranges* — ESR ∈ [0, ∞) and 0 iff equal, correlation
∈ [-1, 1], RMSE ≥ MAE, SNR → ∞ for a perfect fit, sag depth ≥ 0 — and on
`SyntheticSagAmp` ground truth the values are pinned to the physically-known
answers. `tests/test_metrics.py` / `test_report.py` assert those ranges and the
report schema; `tests/test_plots.py` renders every figure on the `Agg` backend
and checks structure (e.g. the output/B+ figure has two stacked axes, never a
dual y-scale). So "is this report well-formed?" is a test, not a judgement call.

Figures use the Okabe-Ito colourblind-safe palette in fixed order and never use
a dual y-axis (differing units go in stacked subplots sharing the time axis).

## Where the documentation lives

This project follows code-as-documentation: the sources below are authoritative,
and prose is kept next to the code it describes so it cannot drift.

- **Docstrings** are the API contract — parameters, shapes, and the *why* behind
  non-obvious choices (numpy style).
- **Tests** are the executable specification: `tests/test_<module>.py` pin each
  component's invariants; `tests/test_integration.py` covers the end-to-end
  flows (stateful loader → model → loss → optimiser, WAV round-trip, protocol).
- **This README** is orientation only — what the project is, how to run it, and
  the decisions worth knowing — pointing to the code for detail.
- **`configs/default.yaml`** is the single source for hyperparameters
  (Deluxe Reverb AB763 defaults), each annotated inline.

## Status

Pre-experimental. The software is complete and verified end-to-end on synthetic
data; the open item before hardware is the identifiability decision above (B+
probe vs. pinning `C1`). See `RESEARCH.md` for the full research notes and paper
draft.
