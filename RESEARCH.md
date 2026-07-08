# Power Supply Sag in Guitar Amplifier Emulation
## Research Notes & Working Paper Draft

*Last updated: June 2026 — pre-experimental stage*

---

## Part I: Research Notes

These notes capture all working thinking, literature context, and architectural decisions.
Update this section throughout the project as decisions solidify or change.

---

### 1. Problem Motivation

Neural amp modeling has achieved substantial accuracy on the static and quasi-static nonlinearities
of guitar amplifiers — preamp tube saturation, tone shaping, even the broad character of the power
section. The state of the art (NAM, Wright et al. LSTM/WaveNet) produces perceptually convincing
results on most playing material.

The gap is **dynamic phenomena tied to slowly-varying hidden state**. Power sag is the clearest
instance: when a player strikes a loud chord or plays at high gain, the power tubes draw significant
current, momentarily drooping the B+ supply voltage. This softens the attack of the note, adds
bloom on the decay, and creates the "touch sensitive" feel that players prize in tube amplifiers.

**Why existing black-box models miss it:**
- LSTMs and WaveNets are trained on short audio segments; the sag time constant (~10–20ms, up to
  ~200ms for full recovery) is long relative to typical training buffer sizes.
- The power supply voltage is a *latent* state — it affects the output but is not directly
  observable in the training data.
- Standard training signals (chirps, random noise, static licks) don't systematically exercise the
  sag trajectory; the model never learns the behavior it needs.
- Even if the network has sufficient temporal depth, the physics-free parameterization forces it
  to learn sag implicitly from the audio signal alone, rather than from the known governing equations.

**The literature gap:** No published work explicitly targets power supply sag as a neural modeling
objective (confirmed by a July 2026 review — see §3.7 for the closest prior work and the
differentiator). The closest *analogous* work is Simionato & Fasciani (JAES 2025), who model optical
compressor dynamics — a structurally similar problem (slow opto-isolator hidden state) — using
Mamba/S6 selective state space models. That result is the proof of concept we build from. The
closest *guitar-amp* work is the DDSP Guitar Amp (Yeh et al., ICASSP 2025), which models a power-amp
stage differentiably but does not treat sag as a dynamic supply-voltage state.

---

### 2. Hardware: Fender Deluxe Reverb

**Target amplifier:** Fender Deluxe Reverb (roommate's amp — likely DRRI/`65 Reissue or equivalent)

**Tube complement:**
- Preamp: 4× 12AX7 (or 7025), 2× 12AT7 (reverb driver + phase inverter)
- Power: 2× 6V6GT
- Rectifier: 1× GZ34 / 5AR4

**Key power supply characteristics (AB763 / DRRI):**
- GZ34 internal resistance: ~100–200Ω under typical load (current-dependent — this is the
  nonlinearity that distinguishes it from a solid-state rectifier)
- Power transformer secondary: ~330-0-330 VAC; effective source resistance with GZ34: ~200–400Ω total
- First filter cap C1: confirm from schematic (varies: ~22–70μF across versions)
- B+ at idle: ~420–430V
- B+ under hard playing: ~380–400V (sag of 20–40V; more than a comparable solid-state rectified amp)
- Recovery time constant: τ = R_source × C1 ≈ 10–20ms (confirm with actual component values)
- Screen supply: sags more dramatically due to downstream 10kΩ–22kΩ series resistor to C2

**TODO before first experiments:**
- [ ] Pull the exact schematic (AB763 for original, or verify DRRI component values)
- [ ] Confirm filter cap values (C1, C2) and the screen supply filter resistor
- [ ] Identify rectifier section component values for ODE parameterization
- [ ] Consider instrumenting B+ rail with a buffered resistor-divider probe for ground truth
      measurement during capture (high-impedance measurement via op-amp buffer; B+ is ~420V,
      bring down to ±1V audio-safe range)

**Why the Deluxe Reverb is a better research target than a Twin Reverb:**
- Uses a tube rectifier (GZ34) rather than silicon diodes → meaningful, audible sag
- 22W at 8Ω; pushed into sag regime at lower volumes → more practically exercisable
- 6V6GT power tubes in fixed-bias configuration → well-characterized operating region
- The amp is famous precisely for its sag/bloom character: the research claim maps directly
  onto what players actually care about

---

### 3. Literature Survey

#### 3.1 Core Neural Amp Modeling

**Wright, Damskägg, Välimäki (2019–2020)** — foundational.
Real-time black-box modeling with RNNs (DAFx-19) and *Real-Time Guitar Amplifier Emulation
with Deep Learning* (Applied Sciences, 2020). Compared WaveNet feedforward models against
LSTM RNNs on amps and pedals. Found ~3 minutes of audio sufficient for training.
Code: `github.com/Alec-Wright/Automated-GuitarAmpModelling` (our primary training framework).

**Damskägg, Juvela, Välimäki (2019)** — *Deep learning for tube amplifier emulation* (ICASSP).

**Juvela et al. (2023)** — *End-to-end amp modeling: from data to controllable guitar amplifier
models* (ICASSP). Extended to parametric/controllable models conditioned on knob positions.

**Vanhatalo et al. (2022)** — *A review of neural network-based emulation of guitar amplifiers*
(Applied Sciences). Good survey for related work section.

#### 3.2 State of the Art (Practical)

**NAM (Neural Amp Modeler) — Steven Atkinson** — de facto community standard.
WaveNet or LSTM architecture, open source, large community of pretrained captures.
`neuralampmodeler.com` — A2 architecture announced January 2026.
**Our baseline**: sdatkinson's Deluxe Reverb capture on Tone3000 (A2 format).

#### 3.3 State Space Models for Audio Effects

**Simionato & Fasciani (2024/2025)** — most closely related work.
*Modeling Time-Variant Responses of Optical Compressors with Selective State Space Models*
(JAES 73(3), 2025). Used Mamba/S6 for optical compressor dynamics; outperformed LSTM-based
methods. Key insight: selective state space models handle long-range temporal dependencies better
than LSTMs for effects with slow hidden states.

**Simionato & Fasciani (2025)** — *Comparative Study of State-based Neural Networks for Virtual
Analog Audio Effects Modeling* (EURASIP JASMP). Compared S4, LRU, LSTM across multiple effects;
SSMs generally outperformed LSTMs for long-range state effects.

**Yin et al. (2024)** — *Modeling analog dynamic range compressors using deep learning and
state-space models* (arXiv 2403.16331). CMU work on compressors with SSMs.

#### 3.4 Gray-Box and Physics-Informed Methods

**Chowdhury & Clarke (2022)** — *Emulating Diode Circuits with Differentiable Wave Digital Filters*
(SMC). Neural networks inside a WDF framework. Related but addresses local nonlinear components
rather than global slow hidden state. Key: demonstrates end-to-end differentiability through
circuit simulation.

**Wright & Välimäki (2022)** — *Grey-box modelling of dynamic range compression* (DAFx). Physics
structure + neural residual for compressors.

**Digital power supply modeling (AmpBooks)** — Thevenin-equivalent model of the 5E3 Tweed Deluxe
power supply, discretized at audio sample rate using explicit Euler. Exact formulation we adapt.
`ampbooks.com/mobile/dsp/power-supply/`

**Wilczek, Wright, Välimäki & Habets (2022)** — *Virtual Analog Modeling of Distortion Circuits
Using Neural Ordinary Differential Equations* (DAFx, arXiv 2205.01897). Learns the ODE of a
first-/second-order diode clipper; physically interpretable, no oversampling, sample-rate can be
raised after training. The key precedent for *neural-ODE VA modeling*, but the ODE governs a
**local, fast** distortion nonlinearity — not a slow latent supply state. We adopt the neural-ODE
idea and apply it to a different subsystem (the power supply).

**Yeh et al. (2025)** — *DDSP Guitar Amp: Interpretable Guitar Amplifier Modeling* (ICASSP, arXiv
2408.11405). The closest competitor: a differentiable, modular amp (preamp / tone stack / power
amp / output transformer). Its power amp is a push-pull *waveshaping* nonlinearity and its output
transformer a GRU(hidden=1) for *hysteresis*; power-supply sag is noted as motivation but is **not
modeled as a dynamic supply-voltage state**. This is the paper to differentiate most carefully.
[PDF pending verification of the power-amp section.]

**Comunità et al. — NablAFx (2025, arXiv 2502.11668)** and *Differentiable Black-box and Gray-box
Modeling of Nonlinear Audio Effects* (arXiv 2502.14405). General differentiable gray-box effect
frameworks with time-varying latent conditioning, but no power-supply-sag model. Potentially a
framework to build *within* rather than a competitor.

#### 3.5 Universal Differential Equations (The Broader Framework)

**Rackauckas et al. (2020)** — *Universal Differential Equations for Scientific Machine Learning*.
The natural theoretical home for our approach: ODE with known physics backbone + learned neural
components, trained end-to-end via differentiable ODE solvers. Used in hydrological modeling,
climate parameterization, biological systems. We apply this to audio physics.

**Key UDE quote** (paraphrased): UDEs blend hard-coded mechanistic terms with learned components,
retaining interpretability and physical constraints from the mechanistic part while capturing
residual or unknown dynamics from data. Training uses differentiable ODE solvers and adjoint
sensitivity analysis.

#### 3.6 Adjacent / Peripheral

- Stiff circuit modeling via Transformer + KAN (Yan et al., arXiv 2510.24727, 2024)
- Differentiable allpass filters (various, 2022–2024)
- Port-Hamiltonian formulations for passivity preservation (Falaize, Hélie, Najnudel, Bernardini)
- Deep WDF with Lipschitz-bounded NNs (Massi, Manino, Bernardini, DAFx-24)

#### 3.7 Novelty verification (literature review, July 2026)

A targeted review (arXiv / JAES / DAFx / ICASSP, English) found **no published work that models
guitar-amplifier power-supply sag as a learned, physics-informed differentiable ODE** — a latent
B+ supply state governed by an RC/rectifier equation, driven by a *neural* load-current coupling,
conditioning a neural audio path, trained end-to-end. Supporting signal: reviews of block-oriented
gray-box (Wiener–Hammerstein) amp models explicitly list **power sagging among the effects those
models cannot capture** — i.e. it is a recognised open gap, not an oversight.

**Closest prior work and how ours differs:**

| Prior work | What it does | How ours differs |
|---|---|---|
| DDSP Guitar Amp (Yeh 2025) | Differentiable modular amp; power amp = waveshaper, transformer = GRU hysteresis | Sag mentioned but not a modeled supply state; we make B+ an explicit differentiable-ODE latent |
| Neural-ODE VA (Wilczek 2022) | Learns ODE of a diode-clipper distortion circuit | Local fast nonlinearity vs. our slow global supply state |
| Optical-compressor SSM (Simionato 2025) | Slow hidden state via learned Mamba/S4 | Same problem *class*, different effect; we use *known physics*, not a learned SSM |
| Gray-box frameworks (NablAFx/Comunità 2025) | Generic differentiable gray-box effects | No sag model; a possible host framework, not a competitor |
| Commercial (Fractal MIMIC, AmpBooks) | Sag via hand-crafted circuit-knowledge DSP + a sag knob | Not learned/trained; we fit the coupling and (optionally) physics from amp audio |

**Defensible novelty claim.** First neural amp model to target power-supply sag *explicitly*, via a
Universal-Differential-Equations framing — a differentiable RC/rectifier ODE for the latent B+
state with a learned load-current coupling, trained end-to-end against amp audio — versus prior work
that omits sag (DDSP amp, WH gray-box), learns an unstructured slow state for a *different* effect
(optical-compressor SSM), or hand-crafts it in DSP (Fractal).

**Caveats.** (1) Rests on abstracts/summaries; full PDFs pending — verify the DDSP-amp power-amp
section, the one possible partial overlap. (2) English/arXiv-biased; re-check near submission for a
DAFx-2026 or thesis in the pipeline.

---

### 4. The Problem Class

**Formal characterization of the problem class:**

A **multi-timescale partially-observable dynamical system** where:
1. A *fast subsystem* (audio signal path, kHz timescales) produces the observable output
2. A *slow subsystem* (power supply, 10–200ms timescales) evolves according to known physics
3. The slow state is **latent** — not measured during training, only affects output indirectly
4. The fast and slow subsystems are **coupled** through a load current function that is itself
   partially unknown
5. The slow subsystem has **known structure** (conservation laws, circuit equations) but
   possibly unknown parameters (component values, operating point)

**Formal notation:**

```
Fast:   y[n]     = g_φ(x[n], z[n])                    // audio output, neural
Slow:   dz/dt    = f_phys(z, u_load(x, z, θ), η)       // power supply ODE, physics
Couple: u_load   = f_θ(x[n], z[n])                     // load current, neural
```

Where x[n] = guitar input, y[n] = amp output, z[n] = B+ voltage (hidden slow state),
φ = audio path parameters, θ = coupling function parameters, η = physical parameters (R, C, V_oc).

**Instances of this problem class in audio physics:**

| System | Fast subsystem | Slow state | Time constant |
|---|---|---|---|
| Power sag (this work) | Audio signal path | B+ supply voltage | 10–200ms |
| Optical compressor | Signal gain | Photoresistor resistance | 10ms–1s |
| Speaker thermal | Signal transduction | Voice coil temperature | 1–60s |
| Output transformer | Signal coupling | Magnetic flux / saturation state | Cycles–10ms |
| Tube bias drift | Gain staging | Cathode temperature | Minutes |
| Tape saturation | Magnetic recording | Residual magnetization | ms–cycles |

**Broader instances (beyond audio):**
- Catalyst deactivation in chemical reactors (activity state, hours timescale)
- Fatigue crack growth under cyclic mechanical loading (crack length state)
- Slow climate modes (ENSO) modulating fast weather dynamics
- Neurovascular coupling in fMRI (blood flow state modulating BOLD signal)

---

### 5. Architecture

#### 5.1 Power Supply ODE (Explicit Euler, Differentiable)

For the Deluxe Reverb with GZ34 rectifier:

```
C₁ · dV_B+/dt = (V_oc - V_B+) / R_eff(I_load)  -  I_load[n]
```

Where:
- `V_oc` ≈ 420–430V (open-circuit B+ from transformer; measurable or learnable)
- `R_eff(I_load)` = R_transformer + R_gz34(I_load) (effective source resistance)
  - R_transformer: fixed, ~50–100Ω (from schematic)
  - R_gz34: current-dependent; GZ34 characteristic from Mullard datasheet
  - For a first model: linearize R_eff as a learned scalar constant
  - For a richer model: parameterize R_gz34(I) = a + b/I^c with learnable {a, b, c}
- `C₁`: filter cap value from schematic; optionally make learnable
- `I_load[n]`: load current drawn by power tubes, see coupling function below

**Discretized at sample rate (explicit Euler, Ts = 1/fs ≈ 20μs at 48kHz):**

```
V_B+[n+1] = V_B+[n]  +  (Ts / C₁) · ((V_oc - V_B+[n]) / R_eff  -  I_load[n])
```

This is stable: Ts/τ ≈ 20μs / 10ms ≈ 0.002 ≪ 1. Fully differentiable with standard autograd.
Gradients flow back through the Euler steps via BPTT.

**Screen supply (optional extension):**

```
V_screen[n+1] = V_screen[n] + (Ts / C₂) · ((V_B+[n] - V_screen[n]) / R_screen  -  I_screen[n])
```

The screen voltage sags more dramatically (higher series impedance) and affects the power tubes'
maximum current. Including this is optional for a first paper; it adds a second latent state.

#### 5.2 Coupling Function (Load Current Estimation)

The load current `I_load[n]` is what the audio circuitry draws from the power supply. It's a
function of both the signal being processed (how hard the tubes are working) and the current
supply voltage (operating point shift).

**Options (from most constrained to most flexible):**

1. **Envelope-based (2 parameters):**
   `I_load[n] = I₀ + α · env(x[n])²`
   Physically motivated (power ∝ signal amplitude²). Simple, interpretable.

2. **MLP coupling (most flexible, our proposed approach):**
   `I_load[n] = MLP_θ(x[n], V_B+[n])`
   Small network (2–3 layers, 16–32 units). Learns the operating-point dependence.
   This is the neural component within the UDE framework.

3. **Physics-based (most constrained):**
   Use a simplified Koren or Leach tube model for 6V6GT to compute I_plate analytically
   from x[n] and V_B+[n]. Parameters (μ, kp, kvb) from datasheet or learned.
   Most principled; best extrapolation; higher implementation effort.

**Recommendation for first paper:** Option 2 (MLP coupling). Clean separation of concerns:
physics in the ODE structure, learning in the coupling. Ablate against options 1 and 3.

#### 5.3 Audio Path Network (Conditioned Amp Model)

Standard WaveNet or LSTM architecture from `Automated-GuitarAmpModelling`, modified to accept
V_B+[n] as a conditioning signal.

**Conditioning method — FiLM (Feature-wise Linear Modulation):**
```
h_conditioned = γ(V_B+[n]) · h  +  β(V_B+[n])
```
Where γ, β are small learned linear projections of the normalized V_B+. This modulates the
hidden activations of the amp network based on the current sag state. No architectural surgery
needed — just insert FiLM after each recurrent/conv layer.

**Alternative: direct input concatenation.** Normalize V_B+[n] to [0,1] and concatenate with
x[n] as an additional input channel. Simpler; FiLM is preferable for architectural cleanliness.

#### 5.4 Full Training Graph (End-to-End)

```
x[n] ──────────────────────────────────────────────── g_φ(x[n], V_B+[n]) ─── y[n]
  │                                                         ↑
  └──→ MLP_θ(x[n], V_B+[n]) = I_load[n]                    │
                  │                                         │
                  ↓                                         │
          Euler ODE step → V_B+[n+1] ───────────────────────┘

Loss: L(y[n], ŷ[n])  // ŷ[n] = measured amp output
Optimize: φ, θ, and optionally {R_eff, C₁, V_oc} jointly via backprop
```

#### 5.5 Training Considerations

- **Segment length:** Must cover multiple sag time constants. Minimum 200ms; 500ms–1s preferred.
  This is longer than typical LSTM training segments.
- **Batch construction:** Carry V_B+[n] state across segment boundaries (stateful), or re-initialize
  to V_idle at each batch. Re-initialization is simpler but introduces boundary artifacts.
- **Training signal design:** Standard NAM training audio (chirps, noise sweeps) is bad for sag.
  Need: sustained power chords, dynamic range variation, heavy picking at high gain. Custom signal.
- **Loss function:** ESR (error-signal ratio) as standard, plus a perceptual pre-emphasis
  (A-weighting or multi-resolution STFT) as used in Wright & Välimäki (2020).
- **Optional ground truth:** Instrument B+ rail with high-impedance probe (resistor divider +
  op-amp buffer) to directly measure V_B+ during capture. Converts latent-variable problem to
  supervised; significantly simplifies training. Practically feasible.

---

### 6. Evaluation Protocol (Proposed)

Standard ESR/MAE metrics will not reveal whether the model captures sag correctly — they average
over all playing contexts and can be high even when sag behavior is wrong. We need targeted tests:

1. **Sustained chord test:** Play a sustained power chord at high gain, measure attack shape and
   bloom compared to ground truth. A correct model should soften the attack and show gradual
   recovery.
2. **Dynamic recovery curve:** Play loud, then quiet. The sag should decay exponentially;
   measure the time constant of the model's recovery vs. the physical amp.
3. **Pre-sagged attack:** Prime the supply with a sustained chord, then play a single note.
   Correct model should have a "compressed" attack relative to a cold-supply note.
4. **Quiet-to-loud transition:** The opposite of above. Model should correctly represent the
   "stiffening" of the attack when the supply is fully charged.

If B+ is directly measured: add a direct V_B+ tracking error metric.

---

### 7. Tools and Ecosystem

| Component | Tool | Notes |
|---|---|---|
| Training framework | `Automated-GuitarAmpModelling` (Wright et al.) | PyTorch, LSTM + WaveNet |
| Baseline model | NAM A2 DR (sdatkinson, Tone3000) | Best existing DR capture |
| Cabinet IR | 1966 DR Jensen C12Q (dahmanmusic, Tone3000) | 7 mic types, 24b/48kHz |
| Physical reference | AB763 schematic + Mullard GZ34 datasheet | For ODE parameterization |
| Real-time deployment | RTNeural (Chowdhury) + JUCE | JSON model export from PyTorch |
| Circuit simulation | LTspice | For ground-truth sag trajectory validation |

---

### 8. Open Questions and TODO

- [ ] What version is the roommate's amp? Pull the exact schematic.
- [ ] Measure or confirm C₁, R_screen, and transformer resistance from the physical circuit.
- [ ] Build the B+ probe circuit if doing direct measurement.
- [ ] Design the training capture signal (sustained chords, dynamic content).
- [ ] Decide: envelope coupling vs. MLP coupling for first experiments.
- [ ] Decide: include screen supply (second latent state) in v1 or save for later.
- [ ] Decide: target venue — DAFx, ICASSP, or JAES?
- [ ] Build the sag evaluation protocol (custom test signal + measurement code).
- [ ] Run baseline NAM on the same test set to establish comparison point.

---
---

## Part II: Working Paper Draft

*Status: Pre-data. Sections 1–4 (method) and 5 (pre-registered protocol) are near-final and match
the implemented, unit-tested code; §6.1 reports preliminary synthetic-validation results. Pending
data collection: §6.2 (amplifier results), §7 discussion specifics, and the abstract's quantitative
claims. Related work (§2 / §3.4 / §3.7) is drafted from a July-2026 literature review and marked
pending against the full PDFs.*
*Target venue: DAFx 2026 or ICASSP 2027 (depending on experimental timeline).*

---

# Physics-Informed Modeling of Power Supply Dynamics for Guitar Amplifier Emulation

**[Authors TBD]**

---

## Abstract

We present a gray-box architecture for guitar amplifier emulation that explicitly models power
supply sag as a differentiable physics layer. Existing neural amp modeling approaches treat
the amplifier as a black-box input-output mapping, implicitly relying on recurrent networks to
learn all temporal dynamics. This fails for power sag — the drooping of the B+ supply voltage
under high load — because sag evolves on timescales (10–200ms) that exceed typical training
segment lengths, and because the supply voltage is a latent variable not directly observable
in the training data. We decompose the amplifier into two coupled subsystems: a physics-informed
ordinary differential equation governing the power supply dynamics, and a neural audio path
conditioned on the resulting supply voltage. The ODE is discretized at sample rate and unrolled
through time in an end-to-end differentiable training graph, following the Universal Differential
Equations framework. We validate on the Fender Deluxe Reverb, a 22-watt tube amplifier with a
GZ34 tube rectifier known for its prominent sag character. The proposed model is compared against
state-of-the-art black-box baselines on a targeted evaluation protocol designed to expose
sag-dependent behaviors. [Results TBD]. The architecture generalizes to a class of problems in
audio effects modeling characterized by slow, physics-governed hidden state coupled to a fast
neural signal processor.

---

## 1. Introduction

Neural approaches to guitar amplifier modeling have matured significantly over the past decade.
Recurrent neural networks, particularly LSTMs, and dilated convolutional architectures inspired
by WaveNet have demonstrated the capacity to reproduce the nonlinear distortion characteristics
of preamp and power amp stages with sufficient fidelity for practical use [WRIGHT2020, NAM].
Open-source tools such as Neural Amp Modeler have made high-quality captures accessible to
a broad community [ATKINSON2023].

Yet a persistent perceptual gap remains in the feel of neural amp models under dynamic playing
conditions. Experienced players frequently describe the gap as a lack of "give" or "sag" —
the characteristic compression and softening of notes played at high gain, followed by a gradual
recovery as the supply voltage restores. This behavior, known as power supply sag, arises from
the dynamic interaction between the power tubes' current demand and the finite impedance of the
power supply. In amplifiers equipped with tube rectifiers such as the GZ34, this interaction is
particularly pronounced: the rectifier's internal resistance rises under load, creating a soft,
current-dependent voltage droop that modulates the operating point of the power amplifier in real
time.

Power sag is not merely a tonal coloration; it is a dynamic state with memory. The supply voltage
at any instant reflects the recent history of the signal's energy content, making it a slowly-
varying hidden state that conditions every sample of the output signal. This structure is
fundamentally different from the memoryless or short-memory nonlinearities that existing black-
box models learn well. Standard LSTM and WaveNet architectures, trained on audio segments short
relative to the sag time constant, cannot reliably encode this behavior. Even when sequence length
is sufficient in principle, the physics-free parameterization offers no inductive bias toward
learning the correct long-range dependency structure.

We propose to address this by decomposing the amp model into two coupled subsystems, following
the Universal Differential Equations (UDE) framework of Rackauckas et al. [RACKAUCKAS2020]. The
power supply is modeled as a first-order ODE with known physical structure — the Thévenin-
equivalent RC circuit of the filter capacitor network, driven by the load current drawn by the
audio circuitry. The load current itself is estimated by a small learned coupling network. The
resulting supply voltage conditions a neural audio path model via feature-wise linear modulation.
The entire system is trained end-to-end by unrolling the Euler-discretized ODE through time.

The approach is validated on the Fender Deluxe Reverb, a 22-watt amplifier universally recognized
for the musical quality of its power supply sag. The evaluation introduces a targeted protocol
for sag-dependent behaviors that standard error metrics do not expose.

Beyond the specific application, we argue that this architecture addresses a general problem
class in audio effects modeling: systems with coupled fast and slow dynamics, where the slow
dynamics follow known physics and the fast dynamics are learned from data. This class includes
optical compressors [SIMIONATO2025], speaker thermal compression, output transformer saturation,
and tube bias drift. The architectural pattern we introduce — physics ODE layer coupled to
conditioned neural audio processor — is applicable across all of these.

---

## 2. Background

### 2.1 Neural Amp Modeling

Early approaches to digital guitar amplifier emulation used circuit simulation methods including
nodal analysis and wave digital filters (WDFs), which directly model the circuit topology as a
system of nonlinear differential equations [PAKARINEN2009, DUNKEL2016]. While physically
principled, these methods scale poorly with circuit complexity: circuits with many coupled
nonlinear elements become computationally intractable for real-time use, and vacuum tube behavior
is particularly difficult to characterize analytically [JUVELA2023].

Data-driven methods emerged as a practical alternative. Covert and Livingston (2013) first applied
recurrent neural networks to tube amp modeling. Zhang et al. (2018) demonstrated LSTM networks
for this task. The work of Wright, Damskägg, and Välimäki [WRIGHT2020] established the current
paradigm: a comparison of WaveNet-based feedforward models against LSTM RNNs showed both
architectures capable of high-fidelity amp emulation from as little as three minutes of training
audio. Subsequent work extended this to parametric models conditioned on control knob positions
[JUVELA2023, COMUNITA2023], multi-rate processing [CARSON2024], and aliasing reduction
[ALIASING2025].

Neural Amp Modeler (NAM) [ATKINSON2023] synthesized these developments into an open-source
tool that has seen widespread community adoption. The current NAM A2 architecture supports both
WaveNet and LSTM backends and achieves state-of-the-art black-box modeling performance on a
broad range of amplifiers and pedals.

### 2.2 Dynamic Range Effects and Slow Hidden State

A parallel literature addresses audio effects with slow time-varying state, most notably dynamic
range compressors. Wright and Välimäki (2022) proposed a gray-box approach to analog compression
that combines a known compressor structure with a neural residual. Simionato and Fasciani (2024,
2025) demonstrated that selective state space models (Mamba/S6) outperform LSTMs for optical
compressors, where a photoresistor's slow time constant creates a hidden state structurally
analogous to power supply sag [SIMIONATO2024, SIMIONATO2025]. Comunità et al. (2023) used
time-varying feature modulation for effects with slowly changing parameters. Yin et al. (2024)
applied state-space models to analog compressor emulation [YIN2024].

These works establish that state space models capture long-range temporal dependencies better
than LSTMs for effects governed by slow physical states. Our work differs in two respects: we
use a *known physics equation* (rather than a learned SSM) for the slow dynamics, and we apply
this to the power amplifier stage rather than standalone effects processors.

### 2.3 Physics-Informed and Gray-Box Audio Modeling

Differentiable wave digital filters [CHOWDHURY2022] embed neural networks inside a WDF circuit
simulation to model specific nonlinear elements end-to-end. This is the most direct precedent
for differentiable physics in audio effects modeling, operating at the level of individual circuit
components. Our work operates at the level of the power supply subsystem — a coarser but more
tractable decomposition that does not require full circuit topology knowledge.

Port-Hamiltonian neural network approaches [FALAIZE, NAJNUDEL] preserve energy balance and
passivity, which are important structural priors for circuit stability. These are relevant for
potential extensions to ensure the simulated supply voltage remains physically bounded.

### 2.4 Universal Differential Equations

Rackauckas et al. (2020) introduced Universal Differential Equations (UDEs) as a framework for
scientific machine learning: differential equations in which some terms are known from physics
and others are replaced by universal function approximators (typically neural networks) [RACKAUCKAS2020].
UDEs generalize neural ODEs (where the entire vector field is a neural network) by allowing
selective hybridization: known mechanistic structure is retained for interpretability and data
efficiency, while learned components capture unmodeled dynamics. Training is performed end-to-end
via differentiable ODE solvers and adjoint sensitivity analysis.

UDEs have been applied to climate model parameterization, hydrological modeling, and battery
dynamics. To our knowledge, this is the first application to audio effects modeling.

---

## 3. The Problem Class and Formal Formulation

### 3.1 Multi-Timescale Partially-Observable Dynamical Systems

We identify the problem addressed in this work as an instance of a general class that we term
**multi-timescale partially-observable dynamical systems with structural physics**. The class is
characterized by:

1. **Two coupled timescales.** A fast subsystem (the audio signal path, operating at sample rate
   kHz) is coupled to a slow subsystem (a physical state, evolving at 10ms–minutes timescales).
2. **Partial observability.** The slow state is not directly measured in training data; it is
   only observable through its effect on the fast subsystem's outputs.
3. **Structural physical knowledge.** The slow state dynamics follow known governing equations
   (conservation laws, circuit equations, thermodynamics), though parameters may be uncertain.
4. **Learned coupling.** The interaction between slow and fast subsystems — specifically, how
   the fast subsystem draws on or affects the slow state — is partially unknown and must be
   learned from data.

Formally, the system is:

```
y[n] = g_φ(x[n], z[n])                    (fast: neural audio path)
dz/dt = f_phys(z, u(x, z, θ), η)          (slow: physics ODE)
u[n] = f_θ(x[n], z[n])                    (coupling: learned)
```

where x[n] is the input signal, y[n] is the output, z[n] is the slow hidden state, u[n] is the
coupling variable (load current in our case), φ are the audio path parameters, θ are the
coupling function parameters, and η are the physical parameters of the slow subsystem.

### 3.2 Power Supply Sag as an Instance

For guitar amplifier power supply sag, the components map as:

- **Fast subsystem g_φ:** The audio signal path from guitar input through preamp and power amp
  to speaker output. Parameterized as an LSTM or WaveNet conditioned on z[n].
- **Slow state z[n]:** The B+ plate supply voltage V_B+(t). This is the voltage across the
  first filter capacitor, which droops under load and recovers when load decreases.
- **Slow ODE f_phys:** The RC circuit equation for the power supply, incorporating the rectifier
  and filter capacitor dynamics. For the Deluxe Reverb (GZ34 rectifier):

  ```
  C₁ · dV_B+/dt = (V_oc - V_B+) / R_eff(I_load)  -  I_load
  ```

  where V_oc is the open-circuit supply voltage, R_eff incorporates transformer secondary
  resistance and GZ34 internal resistance, and C₁ is the first filter capacitance.

- **Coupling u[n] = I_load[n]:** The total plate and screen current drawn by the power tubes,
  as a function of the input signal and current supply voltage. This is partially captured by
  tube models but depends on the complex interaction between signal content and operating point;
  we learn it with a small MLP.
- **Physical parameters η:** {V_oc, R_transformer, C₁}. Measurable from the circuit or jointly
  learnable.

The GZ34 tube rectifier introduces a physically important nonlinearity: its effective internal
resistance is current-dependent, increasing as the demanded current rises. This creates the
characteristically "soft" sag of tube-rectified amplifiers. The full model of R_eff(I_load) uses
the published GZ34 characteristic curve, parameterized with learnable scalars.

### 3.3 Discretization

At audio sample rate (fs = 48kHz, Ts = 1/fs ≈ 20.8μs), the power supply ODE is discretized
using explicit Euler:

```
V_B+[n+1] = V_B+[n]  +  (Ts / C₁) · ((V_oc - V_B+[n]) / R_eff(I_load[n])  -  I_load[n])
```

The step-size to time-constant ratio Ts/τ ≈ 20μs / 15ms ≈ 0.0013, ensuring stability and
accuracy of the explicit integration. The discretized ODE is fully differentiable: gradients
with respect to the learned parameters (φ, θ, and optionally η) backpropagate through the Euler
steps via standard automatic differentiation, without requiring adjoint methods.

**Numerical validation.** We verify the discretisation against references that require no measured
data. For a constant load current the ODE is linear and admits the closed form
V(t) = V_ss + (V₀ − V_ss)·exp(−t/RC) with V_ss = V_oc − I·R; at 48 kHz the explicit-Euler
trajectory matches this analytic solution to within 0.1% of the voltage swing. The global error
scales linearly with Ts (fitted order ≈ 1.0, as expected for explicit Euler), and the trajectory is
indistinguishable (< 0.05 V) from a fourth-order Runge–Kutta reference at the audio rate. These
checks are encoded as unit tests, so the accuracy claim is regression-guarded rather than asserted.

---

## 4. Architecture

### 4.1 Overview

The complete architecture consists of three learned components and one fixed physics layer:

1. **Coupling network** MLP_θ(x[n], V_B+[n]) → I_load[n]: a small feed-forward network
   estimating the instantaneous load current from the input signal and current supply voltage.
2. **Physics ODE layer**: the Euler-discretized power supply equation, producing V_B+[n+1].
3. **Audio path network** g_φ(x[n], V_B+[n]) → y[n]: a WaveNet or LSTM model conditioned on
   the current supply voltage via FiLM.
4. **Cabinet IR** (fixed): a linear convolution representing the speaker and microphone response,
   not jointly trained.

The cabinet IR is excluded from the trainable graph because (a) it is approximately linear and
time-invariant — unaffected by power supply state — and (b) high-quality free IRs of the Deluxe
Reverb are publicly available [DAHMAN2024], making it more practical to use a measured IR than
to learn it.

### 4.2 FiLM Conditioning

The supply voltage V_B+[n] conditions the audio path network via Feature-wise Linear Modulation.
Given a normalized supply voltage v[n] = (V_B+[n] - V_idle) / ΔV_max ∈ [-1, 1]:

```
γ[n] = W_γ · v[n] + b_γ
β[n] = W_β · v[n] + b_β
h_conditioned[n] = γ[n] ⊙ h[n]  +  β[n]
```

FiLM is applied to the hidden sequence of the audio path network before the output projection. The
projection matrices W_γ, W_β are small (hidden_dim × 1) and add negligible parameter count. Their
biases are initialised to b_γ = 1, b_β = 0, so that at the quiescent operating point (V_B+ = V_idle,
hence v = 0) the layer is exactly the identity and does not attenuate the audio path at the start of
training; the weights retain their standard non-zero initialisation so gradients still flow to V_B+
from the first step.

### 4.3 Training Procedure

The full model is trained end-to-end by minimizing:

```
L = ESR(y[n], ŷ[n])  +  λ · L_preemph(y[n], ŷ[n])
```

where ESR is the error-signal ratio [WRIGHT2020], L_preemph is a perceptual pre-emphasis loss
(A-weighting filter, following Wright & Välimäki 2020), and λ is a tuned weight.

Training uses 0.5 s segments (24 000 samples at 48 kHz) to cover several sag time constants. The
V_B+ state is carried across segment boundaries within a recording and re-initialised to V_idle at
the start of each recording. Because this carry makes ordering significant, batches are drawn with a
sequence-level sampler that shuffles at the *recording* level and keeps each recording's segments in
order (a naive per-segment shuffle would feed most segments a V_B+ initial condition from an
unrelated segment); each parallel batch lane carries its own recording's state.

**Training signal:** A purpose-designed capture signal comprising sustained power chords,
high-gain single-note lines, and deliberate dynamic variation (forte to piano transitions and
back) to systematically exercise the sag trajectory. This differs from standard NAM training
signals, which do not exercise long-range dynamics.

### 4.4 Parameterisation and optimisation

Three implementation choices proved necessary for the gray-box model to train, and we report them
because they are non-obvious consequences of coupling a stiff physical parameterisation to a neural
optimiser.

**Log-space physical parameters.** The physical parameters span orders of magnitude
(C₁ ≈ 2×10⁻⁵ F, R_eff ≈ 3×10² Ω). A first-order optimiser such as Adam takes steps of roughly the
learning rate in raw parameter units regardless of scale, which diverges the tiny capacitance while
leaving the large resistance essentially frozen. We therefore learn log C₁, log R_eff (and the GZ34
model's log R₀, log R₁), which places every parameter on an O(1) multiplicative footing and enforces
positivity for free. This was identified by the synthetic study of §6.1, not anticipated.

**Truncated backpropagation through time.** A 0.5 s segment unrolls ~24 000 Euler and LSTM steps.
The carried V_B+ is detached at each segment boundary (state continuity without gradient continuity),
and within a segment we detach the supply and recurrent state every N samples (default 2048),
bounding the backprop graph. Empirically the sag time constants (~15 ms ≈ 720 samples) fit well
within one truncation window, and peak memory stays flat (< 1 GB at batch 4) rather than growing
with segment length.

**Differentiable sample-rate recurrence.** The coupling→ODE recurrence is inherently sequential
(each V_B+[n] feeds the coupling network for the next sample) and so cannot be vectorised across
time; the per-sample Python dispatch dominates cost. We compile the recurrence with TorchScript,
which reuses the eager coupling and Euler-step code verbatim (an equivalence test guards against
drift), roughly halving CPU wall-clock with bit-identical results. The audio LSTM is already
vectorised over time, and batch parallelism scales GPU utilisation.

Optionally, when the B+ rail is instrumented with a buffered probe, a supervised term on the
measured V_B+ can be added to the loss, converting the latent-state problem to direct regression on
the supply trajectory (§6.1 quantifies its effect).

---

## 5. Experimental Setup

*This protocol is fixed in advance of data collection (pre-registration); the software that
implements every model, metric, and test below is complete and unit-tested, so the amplifier
experiments consist of running fixed code on the captured data. Only the bracketed items depend on
the physical capture.*

### 5.1 Target Amplifier

Fender Deluxe Reverb (DRRI, `65 Reissue), vibrato channel. [Confirm exact schematic version.]
Power supply component values from schematic: V_oc = [TBD], C₁ = [TBD], R_eff = [TBD] (defaults
420 V / 22 µF / 300 Ω used until confirmed). Per the identifiability analysis (§6.1), C₁ is read
from the schematic and held fixed rather than learned, since it is not jointly identifiable with
R_eff and the coupling from audio alone.

### 5.2 Data Collection

[TBD: capture procedure, microphone/interface, and whether the B+ rail is instrumented with a
buffered probe for direct V_B+ measurement — see §6.1 for why the probe is recommended.] The DI and
amp-output tracks are latency-aligned by cross-correlation before use (misaligned pairs make the ESR
loss meaningless), and split chronologically into train/validation/test (default 80/10/10) — never a
random split, which would leak neighbouring samples across the boundary.

### 5.3 Baseline Models

All baselines share the proposed model's audio-path capacity and its `(x, V0, state, return_state)`
interface, so they are trained and evaluated by the identical pipeline (selected by a config field):

- **NAM A2 (Deluxe Reverb):** Steven Atkinson's capture (Tone3000). External state-of-the-art
  black-box capture of the same amplifier.
- **Conditioned LSTM (no physics):** the proposed audio path, but the slow conditioning state is a
  learned GRU latent rather than the physics ODE. Ablates the *physics* while retaining the
  conditioning mechanism — isolating the value of the physical structure.
- **Unconditioned LSTM:** standard black-box LSTM (Wright et al. architecture). Ablates any supply
  conditioning.

### 5.4 Evaluation Protocol

**Standard metrics** (implemented, each range-checked in tests): ESR, ESR in dB, MAE, RMSE, and
segmental SNR on a held-out test set of general playing material, plus the pre-emphasised ESR.

**Sag-targeted protocol** (implemented; standard error metrics average over context and do not
expose sag). Each test drives a purpose-built signal through the model and measures the output RMS
envelope; for the proposed model the internal V_B+ trajectory is also reported:
1. Sustained-chord attack/bloom shape (settle-to-attack RMS ratio);
2. Dynamic recovery time constant (exponential fit to the recovery envelope);
3. Pre-sagged vs. cold-supply attack (compression ratio of the primed vs. cold attack);
4. Quiet-to-loud transition ("stiffening") response.

**Reproducibility.** All runs are seeded; hyperparameters live in versioned configs (with a
`defaults:` inheritance so experiment configs restate only what differs); the test suite runs in CI;
and reports are emitted as schema'd JSON plus figures. Each experiment above corresponds to a
committed config file.

---

## 6. Results

### 6.1 Preliminary results: synthetic validation

Before any amplifier capture, we validate the approach on synthetic data with known ground truth. A
reference "amplifier" generates (input, output, V_B+) triples from a known power-supply ODE, an
envelope-based load coupling, and a sag-dependent nonlinearity whose clipping headroom scales with
the supply voltage; the model is then fit to the (input, output) pair alone and asked to recover the
hidden state and its physics. This isolates the central question — *is the latent supply state
identifiable from output audio?* — from the confounds of real capture.

Two findings, both from short demonstration runs (final large-scale runs pending GPU training):

1. **The gray-box model learns sag and is numerically well-behaved.** After the log-space
   reparameterisation (§4.4), fitting is stable and drives the error-to-signal ratio to ≈ 5×10⁻³,
   and the recovered latent V_B+ trajectory correlates ≈ 0.8 with the ground-truth trajectory —
   the model discovers a supply state that tracks the true load-driven droop, from audio alone.

2. **Individual physical parameters are *not* identifiable from audio alone.** Even at near-zero
   ESR, the recovered R_eff and C₁ settle far from their true values: the flexible audio path
   compensates for a mis-set supply, and a free coupling current trades off against R_eff (with C₁
   trading off to preserve the time constant), so only the *trajectory* and the RC time constant are
   constrained, not the separate constants. Adding a supervised term on a (simulated) measured V_B+
   improves the trajectory correlation (≈ 0.8 → ≈ 0.87) but does not by itself resolve the
   parameter degeneracy.

The practical consequence, adopted in §5.1, is to fix C₁ from the schematic and/or instrument the
B+ rail, and to report V_B+ as a recovered *trajectory* rather than a set of identified constants.
We consider the identifiability analysis itself a contribution: it is a concrete, reproducible
statement about what a gray-box sag model can and cannot recover, obtained before committing to
hardware. [Figures: parameter-recovery trace and latent-trajectory overlay, generated by
`scripts/validate_synthetic.py`; Euler-vs-analytic convergence, from the reporting layer.]

### 6.2 Amplifier results

*[To be completed after data collection — standard metrics and the four-part sag protocol of §5.4
for the proposed model against the three baselines of §5.3, on the held-out test set.]*

---

## 7. Discussion

*[Partial draft — to be expanded with experimental findings]*

The proposed architecture demonstrates that the explicit physical structure of the power supply
can be productively incorporated into an end-to-end neural amp model. [Results discussion TBD.]

**Generalization of the approach.** The architectural pattern introduced here — a physics ODE
layer for slow hidden state coupled to a conditioned neural audio processor — applies directly
to several other problems in audio effects modeling. Optical compressor photoresistor dynamics
[SIMIONATO2025] follow the same mathematical structure with a different governing equation.
Speaker voice coil thermal compression, output transformer saturation, and tube cathode temperature
drift are further instances of the same problem class. In each case, the slow physics is known
or can be characterized, and the learned coupling absorbs the interaction between the slow state
and the audio signal path that is too complex for first-principles modeling alone.

**Relationship to existing frameworks.** This work instantiates the Universal Differential
Equations framework [RACKAUCKAS2020] in the audio effects domain. The ODE backbone is known
from circuit physics; the neural coupling function f_θ plays the role of the "unknown term"
that UDEs are designed to learn. The result retains the interpretability benefits of a physical
model (the supply voltage trajectory is directly observable and physically meaningful) while
achieving the flexibility of a learned model for the complex coupling dynamics.

**Limitations.** [TBD: discuss specific failure modes observed in experiments, computational
cost, constraints from linearizing R_eff, etc.]

---

## 8. Conclusion

We presented a physics-informed gray-box architecture for guitar amplifier emulation that
explicitly models power supply sag as a differentiable ODE layer. The key insight is that the
power supply voltage is a slowly-varying physical state with known governing equations, and that
conditioning the neural audio path model on this state — rather than relying on the network to
implicitly encode it — produces [results TBD] improvement on dynamic playing scenarios.

More broadly, we identified a class of audio effects modeling problems characterized by coupled
fast-slow dynamics with structural physical knowledge. The architectural pattern introduced here
is applicable across this class, including optical compressors, thermal compression, and
transformer saturation.

---

## References

*[Full bibliography to be formatted for target venue. Key references listed below.]*

- [WRIGHT2020] Wright, A., Damskägg, E.-P., Juvela, L., Välimäki, V. (2020). Real-time guitar amplifier emulation with deep learning. *Applied Sciences*, 10(3), 766.
- [WRIGHT2019] Wright, A., Damskägg, E.-P., Välimäki, V. (2019). Real-time black-box modelling with recurrent neural networks. *DAFx-19*.
- [DAMSKÄGG2019] Damskägg, E.-P., Juvela, L., Thuillier, E., Välimäki, V. (2019). Deep learning for tube amplifier emulation. *ICASSP 2019*.
- [JUVELA2023] Juvela, L., et al. (2023). End-to-end amp modeling: from data to controllable guitar amplifier models. *ICASSP 2023*.
- [SIMIONATO2025] Simionato, R., Fasciani, S. (2025). Modeling time-variant responses of optical compressors with selective state space models. *JAES*, 73(3), 144–165.
- [SIMIONATO2024] Simionato, R., Fasciani, S. (2025). Comparative study of state-based neural networks for virtual analog audio effects modeling. *EURASIP JASMP*.
- [RACKAUCKAS2020] Rackauckas, C., et al. (2020). Universal differential equations for scientific machine learning. *arXiv:2001.04385*.
- [CHOWDHURY2022] Chowdhury, J., Clarke, C.J. (2022). Emulating diode circuits with differentiable wave digital filters. *SMC 2022*.
- [WRIGHT2022] Wright, A., Välimäki, V. (2022). Grey-box modelling of dynamic range compression. *DAFx-22*.
- [YIN2024] Yin, H., Cheng, G., Steinmetz, C.J., et al. (2024). Modeling analog dynamic range compressors using deep learning and state-space models. *arXiv:2403.16331*.
- [COMUNITA2023] Comunità, M., Steinmetz, C.J., Phan, H., Reiss, J.D. (2023). Modelling black-box audio effects with time-varying feature modulation. *ICASSP 2023*.
- [PAKARINEN2009] Pakarinen, J., Yeh, D.T. (2009). A review on digital guitar tube amplifier modeling techniques. *Computer Music Journal*, 33(2), 85–100.
- [VANHATALO2022] Vanhatalo, T., et al. (2022). A review of neural network-based emulation of guitar amplifiers. *Applied Sciences*, 12(12), 5894.
- [ATKINSON2023] Atkinson, S. Neural Amp Modeler. `neuralampmodeler.com`.
- [DAHMAN2024] Dahman Music (2024). 1966 Deluxe Reverb Cabinet with Jensen C12Q IR Collection. *Tone3000*.
- [FRACTAL2013] Fractal Audio Systems (2013). Multipoint Iterative Matching and Impedance Correction Technology (MIMIC). Technical white paper.
- [CARSON2024] Carson, A., Wright, A., Chowdhury, J., Välimäki, V., Bilbao, S. (2024). Sample rate independent recurrent neural networks for audio effects processing. *arXiv:2406.06293*.
- [WILCZEK2022] Wilczek, J., Wright, A., Välimäki, V., Habets, E.A.P. (2022). Virtual analog modeling of distortion circuits using neural ordinary differential equations. *DAFx-22*, arXiv:2205.01897.
- [YEH2025] Yeh, et al. (2025). DDSP guitar amp: interpretable guitar amplifier modeling. *ICASSP 2025*, arXiv:2408.11405.
- [COMUNITA2025] Comunità, M., et al. (2025). NablAFx: a framework for differentiable black-box and gray-box modeling of audio effects. *arXiv:2502.11668* (see also arXiv:2502.14405).