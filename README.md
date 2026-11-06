# Alysane C2+C3 pilot gate

This workspace contains the archived v1.0 specification, a corrected mathematical audit, and an executable C++ pilot gate for the A1 domain-separation hypothesis using the pinned SmolLM-135M checkpoint.

## Status

**C2+C3 is COMPLETE_WITH_EXPLICIT_NO_GO.** 12 of 13 gates pass against the pinned
SmolLM-135M on CI. The original A1 hard-partition assumption is **falsified and remains
an explicit NO-GO**; the objective is preserved by a validated replacement — emergent
overlapping density-peaks regions over continuation behaviour (63 modes, co-association
stability 0.995, coverage 0.941), with no predefined taxonomy and no fixed cluster count.

See [`docs/04_C2_C3_completion_report.md`](docs/04_C2_C3_completion_report.md) for the full
gate table, measurements, corrections and limitations.

### A1 Measurement Protocol — verdict `PARTIAL_GO`

The supplied [A1 Measurement Protocol](docs/05_A1_measurement_protocol_v1.0.md) is the
architecture-wide go/no-go gate. It is implemented in C++ (`alysane probe`) and was executed
against the pinned model on CI. **All three candidate distances exhibit a measurement-stable
plateau with H1, H2 and H3 satisfied** — structure exists — but the usable margin is far below
the certified margin on every metric:

| metric | plateau k | M usable | M required | verdict |
|---|---|---|---|---|
| `one_step_tv` | 9 | 0.0185 | 0.3683 | PARTIAL_GO |
| `rollout_js_root` | 2 | 0.0740 | 3.3302 | PARTIAL_GO |
| `continuation_mmd` | 2 | 0.2368 | 6.8955 | PARTIAL_GO |

This refines the binary A1 NO-GO: the failure is the **margin**, not the existence of
structure. The binding constraint is top-K reconstruction error (τ95 = 0.184), not geometry.
Rollout-metric error bounds are vacuous at R=4 and would need R ≈ 3.3×10⁴–8.4×10⁶ rollouts
per context to certify. Under §4, C5 competence field becomes the primary interface and
fine-tuning stays gated. Six protocol defects (C1–C6) were corrected rather than implemented
as written. Full analysis: [`docs/06_A1_probe_execution_report.md`](docs/06_A1_probe_execution_report.md),
raw evidence: `reports/probe_report.json`.

| Gate | Result | Gate | Result |
|---|---|---|---|
| G1 parity | PASS | G7 serialization | PASS |
| G2 trace τ | PASS | G8 KV parity | PASS |
| **G3 A1** | **NO-GO** | G9 continuation | PASS |
| G3b structure | PASS | G10 boundary | PASS |
| G4 ψ distortion | PASS | G11 quantisation | PASS |
| G5 routing | PASS | G12 reshape | PASS |
| G6 boundary | PASS | | |

**Key limitation:** the learned encoder does not yet generalise at N=256 (holdout
Spearman 0.020). The validated structure is computed on the behavioural metric directly,
not on the learned embedding.

C3v2-1 is complete: observables, competence, and identifiability. See `docs/02_C3v2-1_observables_competence_identifiability.md` and `reports/c3v2_1_report.md`. Minimal C++ demonstration/tests:

```bash
make c3v2_1
./build/c3v2_1_test
./build/c3v2_1_demo reports/c3v2_1_demo.json
```

## Layout

- `docs/00_original_C2_C3_spec_v1.0.md` — original supplied spec, verbatim.
- `docs/01_mathematical_audit_and_corrected_spec_v1.1.md` — corrected definitions, theorems, proofs, limitations, gate protocol.
- `include/`, `src/` — C++17 core: safetensors loader, Llama-compatible fp32 execution, traces, exact TV, MST/A1, landmarks, self-encoder, conservative energy, gates.
- `tools/prepare_data.py` — offline token conversion only; source metadata is audit-only.
- `tools/parity_reference.py` — external HuggingFace fp32 parity oracle.
- `tools/make_report.py` — generates the final HTML/Markdown report from measured JSON.
- `models/` — pinned SmolLM-135M weights/tokenizer (not persisted in lightweight snapshots if excluded by size; re-download with the pinned revision below).
- `docs/05_A1_measurement_protocol_v1.0.md` — supplied A1 gate protocol, verbatim.
- `docs/06_A1_probe_execution_report.md` — probe execution report, corrections C1–C6, certification-cost analysis.
- `reports/` — measured gate outputs.

## Pinned model

`HuggingFaceTB/SmolLM-135M` revision `1d461723eec654e65efdc40cf49301c89c0c92f4`.

## Build

Requires g++ (C++17), CMake, OpenBLAS, LAPACKE, OpenMP. On Debian/Ubuntu:

```bash
sudo apt-get install -y libopenblas-dev liblapacke-dev cmake g++
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

## Run

```bash
python3 tools/prepare_data.py
./build/alysane_unit_tests                      # 35 model-free tests
./build/alysane selftest
./build/alysane_stress 256 49152                # full-scale, no weights
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 ./build/alysane parity models data/parity.bin reports/parity_cpp.bin
OPENBLAS_NUM_THREADS=2 python3 tools/parity_reference.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 ./build/alysane gates models data reports
./build/alysane complete models data reports    # G8-G12 + G3b
python3 tools/verify_certificates.py            # independent NumPy oracle
python3 tools/make_report.py
```

Cached-trace debug mode:

```bash
ALYSANE_REUSE=1 ./build/alysane gates models data reports
ALYSANE_A1_ONLY=1 ./build/alysane gates models data8 reports8
```

## Python role

Python is used only for offline corpus/token conversion and the external parity oracle. All inference, trace production, TV, graph/A1 measurement, encoder/energy training, routing certificates, and serialization hot paths are C++.

## Release

**[v1.2-c2c3](https://github.com/nexuss0781/Alysane/releases/tag/v1.2-c2c3)** — complete
project ZIP (source, specifications, tests, scripts, verified reports, trained θ_S).
Model weights are excluded; `tools/download_model.sh` restores the pinned revision and
verifies its SHA-256.

CI: [![CI](https://github.com/nexuss0781/Alysane/actions/workflows/ci.yml/badge.svg)](https://github.com/nexuss0781/Alysane/actions/workflows/ci.yml)
[![E2E](https://github.com/nexuss0781/Alysane/actions/workflows/end_to_end.yml/badge.svg)](https://github.com/nexuss0781/Alysane/actions/workflows/end_to_end.yml)
