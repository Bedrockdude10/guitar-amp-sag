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

```
src/power_sag/
  physics.py     PowerSupplyODE — differentiable Euler power-supply ODE + GZ34 R_eff
  coupling.py    CouplingNetwork — MLP (x, V_B+) -> I_load (non-negative)
  film.py        FiLMLayer — feature-wise linear modulation
  audio_model.py PowerSagLSTM — LSTM audio path with FiLM conditioning on V_B+
  model.py       PowerSagModel — full end-to-end coupled model
  loss.py        ESRLoss, PreEmphasisLoss
  data.py        AudioDataset, SequenceDataset (stateful ordered segment sampler)
  cabinet.py     CabinetIR — fixed (non-trainable) speaker/mic convolution
  evaluation.py  SagEvaluator — sag test signals + recovery-curve fitting
configs/default.yaml   all hyperparameters (Deluxe Reverb AB763 defaults)
scripts/               train.py, capture.py, evaluate.py
tests/                 pytest suite covering every mathematical invariant
```

## Install

```bash
pip install -e .          # torch, numpy, scipy, soundfile, pyyaml
pip install -e ".[dev]"   # + pytest
```

## Test

```bash
pytest tests/ -v
```

## Train / evaluate

```bash
python scripts/capture.py  --out capture_di.wav --minutes 3
python scripts/train.py    --input capture_di.wav --target amp_out.wav --out model.pt
python scripts/evaluate.py --checkpoint model.pt
```

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
