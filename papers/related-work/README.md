# Related-work PDFs

Source PDFs for the papers cited in `RESEARCH.md` §3 (Literature Survey) and used
in the §3.7 novelty verification. Filenames are `author<year>-<slug>.pdf`; the
arXiv IDs and citation keys are the authoritative link back to the references.

| File | arXiv | RESEARCH.md key | Role in the review |
|------|-------|-----------------|--------------------|
| `rackauckas2020-universal-differential-equations.pdf` | 2001.04385 | `RACKAUCKAS2020` | UDE framework — the theoretical home of the approach |
| `wilczek2022-neural-ode-distortion-circuits.pdf` | 2205.01897 | `WILCZEK2022` | Neural-ODE VA of a diode clipper — neural-ODE precedent (fast local nonlinearity) |
| `simionato2024-state-based-nn-virtual-analog.pdf` | 2405.04124 | `SIMIONATO2024` | Comparative study of state-based NNs for VA |
| `yeh2024-ddsp-guitar-amp.pdf` | 2408.11405 | `YEH2024` | Closest competitor — differentiable modular amp that omits the power supply |
| `simionato2025-optical-compressor-ssm.pdf` | 2408.12549 | `SIMIONATO2025` | Optical-compressor SSM — methodological sibling (learned slow state, also FiLM) |
| `comunita2025-nablafx.pdf` | 2502.11668 | `COMUNITA2025` | Differentiable gray-box effects framework |

These are large binaries kept in-repo as the project's reference library. If the
repo needs slimming later, they can be dropped from the tree (they remain in
history) without affecting any code.
