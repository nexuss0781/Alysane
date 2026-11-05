# Alysane C2+C3 pilot gate

This workspace contains the archived v1.0 specification, a corrected mathematical audit, and an executable C++ pilot gate for the A1 domain-separation hypothesis using the pinned SmolLM-135M checkpoint.

## Status

The gate is executable end to end. The scientific A1 hard-partition gate is **NO-GO** on the executed pilot; engineering/certificate gates are reported separately in `reports/gates.json`. Do not interpret a completed run as proof that A1 holds.

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
./build/alysane selftest
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 ./build/alysane parity models data/parity.bin reports/parity_cpp.bin
OPENBLAS_NUM_THREADS=2 python3 tools/parity_reference.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 ./build/alysane gates models data reports
python3 tools/make_report.py
```

Cached-trace debug mode:

```bash
ALYSANE_REUSE=1 ./build/alysane gates models data reports
ALYSANE_A1_ONLY=1 ./build/alysane gates models data8 reports8
```

## Python role

Python is used only for offline corpus/token conversion and the external parity oracle. All inference, trace production, TV, graph/A1 measurement, encoder/energy training, routing certificates, and serialization hot paths are C++.
