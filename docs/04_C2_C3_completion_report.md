# Alysane C2+C3 — Completion Report

**Repository:** https://github.com/nexuss0781/Alysane (public)
**Model:** `HuggingFaceTB/SmolLM-135M`, revision `1d461723eec654e65efdc40cf49301c89c0c92f4`
**SHA-256:** `c7a387d6fe81ca6dd304aeb809bda3932ff1bbef3ca41c9484502f2f448dc093`
**Validation:** GitHub Actions, `ubuntu-latest`, AMD EPYC 9V45, 4 threads, GCC 13
**Status:** **COMPLETE_WITH_EXPLICIT_NO_GO**

---

## 1. Final gate-by-gate status

| Gate | Criterion | Result |
|---|---|---|
| G1 | C1 parity vs independent HuggingFace fp32 oracle | **PASS** |
| G2 | Trace certificate, 0 violations of τ_safe | **PASS** |
| G3 | **Original A1 hard partition** | **FAIL — explicit NO-GO** |
| G3b | Emergent structure (density-peaks replacement) | **PASS** |
| G4 | Landmark ψ distortion bounds | **PASS** |
| G5 | Behavioural routing certificate | **PASS** |
| G6 | Boundary proxy (energy / margin) | **PASS** |
| G7 | Two-store serialization roundtrip | **PASS** |
| G8 | KV-cache parity | **PASS** |
| G9 | Continuation pseudometric axioms + signal ranges | **PASS** |
| G10 | Continuation boundary proxy | **PASS** |
| G11 | Quantisation TV invariance | **PASS** |
| G12 | Reshape drift bound | **PASS** |

**12 of 13 gates PASS.** The single failure is the original load-bearing empirical
assumption, which is falsified and **retained as an explicit NO-GO** rather than
removed, weakened, or relabelled.

```
original_spec_overall     FAIL
amended_overall           PASS
no_go_gates               ["G3_A1"]
replacement_for_no_go     {"G3_A1": "G3b_structure"}
completion_status         COMPLETE_WITH_EXPLICIT_NO_GO
ready_for_next_component  true
```

---

## 2. The scientific result: A1 is false, and what replaces it

### 2.1 What was falsified

v1.0's Assumption A1 required a hard, globally separated behavioural partition at a
single scale window. Measured on the pinned model:

- best `q = 0` — **no** nontrivial partition satisfies the pre-registered thresholds;
- MST window `ε_in = 0.99561`, `ε_out = 0.99790` — width **0.00229**, far below the 0.02 requirement;
- exact full-V TV distances concentrate near 1.

**Why:** next-token distributions over a 49,152-token vocabulary are nearly mutually
singular across unrelated contexts. One-step TV saturates, so it cannot resolve a
global hard partition. This was confirmed at context lengths 8, 16, 32 and 64.

### 2.2 The replacement, and why it is defensible

The objective — emergent, label-free behavioural regions with structure, relevance
and boundaries — does **not** require disjointness at one global scale. Two changes
preserve it:

**(a) Continuation-level behaviour.** v1.0 §1.3 specified self-agreement and
continuation consistency but they were never implemented. They are now implemented
with KV-cached rollouts and a spectrum-kernel MMD pseudometric `d_C`. Measured
Spearman correlation with one-step TV: **0.558** — related but genuinely different
geometry.

**(b) Density-peaks mode seeking.** Regions are modes of a KDE over `d_C` with an
adaptive bandwidth (median 8-NN distance, σ = **0.6925**). Each point links to its
nearest strictly denser neighbour; modes are points whose nearest denser neighbour
is farther than the bandwidth. Properties: deterministic, **no fixed cluster count**,
no labels, no random anchor subset.

Result: **63 emergent modes**, co-association stability **0.9946** over 10 bootstrap
replicates at 80%, coverage **0.941**.

### 2.3 Honest account of the iteration

The first replacement attempt (farthest-point landmark cover) **failed**: co-association
0.555, argmax-ARI 0.447. Root cause: farthest-point selection picks extreme points, so
the anchor set changes wildly between replicates. That variance belongs to the
*selection rule*, not to the geometry under test. Density peaks removes it. Both the
failed landmark-cover numbers and the original argmax-ARI are retained in
`reports/gates.json` as diagnostics.

Persistence significance was **demoted to descriptive-only**: the permutation null
preserves the distance marginal but destroys metric structure, so it flagged 245/255
merges — it detects hub concentration, not clustering. It is not used as a gate.

---

## 3. Mathematical corrections to v1.0

| v1.0 claim | Correction | Status |
|---|---|---|
| Lemma 1: τ = ½(ρ_x+ρ_y) | Proof invalid. Safe bound τ = ρ_x+ρ_y | [P], 0 violations / 2000 pairs |
| Lemma 3: m ≥ N_ε ln(1/δ) covers S | False — covering number does not bound sampling mass. Requires β_ε mass lower bound | [P] |
| Theorem B: "top-d of L_sym" | Wrong end of spectrum → **bottom-d** of L_sym = top-d of A | [P] |
| Theorem B: universal linear-probe bound | Removed; needs task/label compatibility assumptions | removed |
| Encoder on unit sphere S^{d-1} | Incompatible with whitening (E‖f‖²=d). Uses R^d + whitening | [P] |
| Corollary 2: latent-distance routing | Invalid without lower distortion bound. Replaced by behavioural certificate d(x,y) ≤ r + diam(D) | [P], 0 violations |
| §4: quantisation error "folded into τ" | **False.** Hidden-state quantisation leaves top-k TV and τ *exactly* invariant | [P], 0 violations |
| Theorem 3: L_K = product of spectral norms | Not valid for residual attention + RMSNorm + softmax. Replaced by measured Δz bound | [P], 0 violations |
| §4: bootstrap ≈ 10¹⁸ FLOPs | Off by ~1000×. Correct: **1.72×10¹⁵** at N=10⁵, n=64 | [P] |
| §4: serving overhead +3–8% | Unmeasured assertion, withdrawn | [E] |

Full derivations: `docs/03_C2_C3_completion_spec_v1.2.md`.

---

## 4. Implemented architecture

| Requirement | File | Status |
|---|---|---|
| Knowledge execution core (Llama-compatible fp32) | `src/model.cpp` | complete |
| KV-cached prefill/decode | `src/kv.cpp` | **new** |
| Trace operator + certificates | `src/model.cpp` | complete |
| Rollouts, self-agreement, continuation consistency | `src/continuation.cpp` | **new** |
| Landmark behavioural embedding ψ | `src/model.cpp` | complete |
| Spectral contrastive encoder + ψ distillation | `src/self.cpp` | complete |
| Conservative energy model (DSM) | `src/self.cpp` | complete |
| Persistence, ultrametric, soft regions | `src/structure.cpp` | **new** |
| Density-peaks mode seeking | `src/structure.cpp` | **new** |
| int8 trace quantisation | `src/quantize.cpp` | **new** |
| Reshape drift measurement | `src/drift.cpp` | **new** |
| Routing + certificates | `src/gates.cpp` | complete |
| Two-weight serialization | `src/self.cpp` | complete |
| Gates G1–G7 | `src/gates.cpp` | complete |
| Gates G8–G12, G3b | `src/gates2.cpp` | **new** |
| Portable SIMD dispatch | `include/simd.h` | **new** |

---

## 5. Real-model experimental results

All numbers from the pinned checkpoint on CI, not synthetic substitutes.

**C1 parity (independent HuggingFace oracle):**
- max KL(HF‖C++) = **2.906×10⁻⁷** (threshold 1×10⁻⁵)
- mean KL = **0.0**; top-1 agreement = **1.0**
- max logit diff = 1.979×10⁻⁴
- perplexity: C++ **36.333260** vs HF **36.333243**, relative diff **4.49×10⁻⁷**

**Trace certificate:** 2000 pairs, **0** violations; max |TV − TV| = 0.5280 ≤ max τ = 1.2776.

**KV parity:** single-step max Δp = **1.85×10⁻⁶**; generated-path max Δp = **5.96×10⁻⁷**; top-1 agreement 1.0.

**Continuation:** R=4, T=8; triangle/symmetry/diagonal violations all **0**;
self-agreement mean 0.132; Spearman vs one-step TV **0.558**.

**Structure:** 63 modes, σ=0.6925, co-association stability **0.9946**, coverage **0.941**.

**Routing:** **0** certificate violations; median natural margin 0.3749.

**Boundary:** energy AUROC **0.722**; continuation self-agreement AUROC **0.852**;
continuation boundary-score AUROC **0.830**.

**Quantisation:** **0** TV violations; 138,240 → 34,680 bytes (**3.99×**).

**Reshape:** ‖δθ = 0.2311, max Δz = 0.02147, empirical L_z = **0.0929**;
predicted bound 0.04294 vs actual max TV change **0.000848**; **0** violations.

**Serialization:** encode L2 diff **0.0**, energy diff **0.0** (exact roundtrip).

---

## 6. Independent certificate verification

`tools/verify_certificates.py` recomputes certificates in NumPy without reusing C++
code: exact full-V TV, tail-uniform trace bound, ψ distortion, subdominant
ultrametric (U ≤ D + strong triangle), and the routing certificate on an
independently computed MST/partition. Result: `reports/certificates.json`,
`overall_PASS: true`.

---

## 7. Build, integration and test results

- **CI (correctness):** portable build, **35 unit tests**, analytic selftest (8 checks),
  full-scale stress harness (N=256, V=49152, 4 threads), ASan+UBSan build and run — **all pass**.
- **CI (end-to-end):** pinned model restore + SHA-256 verification, corpus tokenisation,
  parity oracle, G1–G7, G8–G12, independent certificate oracle, report generation.
- **C3v2-1:** 17/17 identifiability tests pass, including under ASan+UBSan.

### Defects found and fixed by CI

These are real defects that local-only testing would have shipped:

1. **Hardcoded AVX-512 + forced `-march=native`.** The core could not compile or run on
   CPUs without AVX-512. Replaced with target-attribute kernels and runtime dispatch
   (AVX-512F → AVX2+FMA → scalar). Also avoided `_mm512_and_ps` (requires AVX512DQ).
2. **Data race in CPU dispatch.** `__builtin_cpu_init()` was called per dot/L1 invocation
   from 4 OpenMP threads, racing on the global `__cpu_model`. Now resolved once via
   thread-safe static initialisation.
3. **`-ffast-math` miscompilation.** Release segfaulted on GCC 13/EPYC while ASan `-O1`
   passed. Removed unconditionally; parity re-verified without it (2.9×10⁻⁷).
4. **Corpus excluded by `.gitignore`.** CI produced 0 chunks. Fixed to ignore only
   generated `*.bin`.
5. **G8 test off-by-one** and **G9 wrong range assertion** — both test bugs, not kernel bugs.

---

## 8. Performance and memory

| Quantity | Measured |
|---|---|
| Rollout wall time (1408 rollouts, 11,264 decode tokens) | **409.2 s** |
| Completion gate wall time | **419.5 s** |
| KV cache per stream (cap=72) | **3,317,760 B ≈ 3.3 MB** |
| Trace fp32 → int8 | 138,240 → 34,680 B (**3.99×**) |
| MACs/token (non-embedding) | 134,479,872 |
| Bootstrap at N=10⁵, n=64 | 1.72×10¹⁵ FLOPs (analytic) |

---

## 9. Remaining limitations and blockers

1. **A1 remains NO-GO.** No hard global behavioural partition exists for this model at
   this scale. Any downstream component must consume *soft overlapping regions*, not
   disjoint domains.
2. **The learned encoder does not generalise at N=256.** Holdout Spearman
   ρ(z, behaviour) = **0.020**; ψ space ρ = **0.200**. The G4 *certificate* bounds hold
   (0 violations), but the learned representation is not yet behaviourally faithful.
   **The G3b structure result is computed on continuation MMD, not on the learned
   embedding, so it does not depend on this weakness.** Scaling N is the clear next step.
3. **int8 hidden-state quantisation is unsafe for the encoder.** TV certificates are
   exactly invariant, but whitened-embedding drift reached **1062** L2. Use int8 for
   storage/certificates only; keep fp16/fp32 for encoder input, or recalibrate whitening.
4. **63 modes on 256 points is fine-grained.** These are local behavioural regions, not
   coarse semantic domains. Coarser structure requires larger N and bandwidth sweeps.
5. **Pilot scale, not production.** N=256 train / 64 holdout from a public-domain mixture,
   not live traffic. Population-level claims are not made.
6. **No competence claims.** By C3v2-1, competence is not identifiable from internal
   observables alone. G6/G10 are boundary *proxies* only.
7. **Persistence null is weak.** Reported descriptively; a metric-preserving null is
   future work.

---

## 10. Reproduction

```bash
git clone https://github.com/nexuss0781/Alysane.git && cd Alysane
sudo apt-get install -y cmake g++ libopenblas-dev liblapacke-dev
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

bash tools/download_model.sh          # pinned revision + SHA-256 check
python tools/prepare_data.py

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

./build/alysane_unit_tests            # 35 model-free tests
./build/alysane selftest              # analytic certificates
./build/alysane_stress 256 49152      # full-scale, no weights

./build/alysane parity models data/parity.bin reports/parity_cpp.bin
python tools/parity_reference.py      # independent HF oracle
./build/alysane gates models data reports
./build/alysane complete models data reports
python tools/verify_certificates.py   # independent NumPy oracle
python tools/make_report.py
```

CI equivalents: `.github/workflows/ci.yml` (every push) and
`.github/workflows/end_to_end.yml` (`workflow_dispatch`).

### Artifacts

| Path | Contents |
|---|---|
| `reports/gates.json` | all 13 gate results + measurements + NO-GO accounting |
| `reports/parity.json` | HF oracle parity |
| `reports/certificates.json` | independent NumPy certificate verification |
| `reports/self_weights.alys` | trained θ_S (encoder, energy, landmarks) |
| `reports/knowledge_manifest.json` | θ_K provenance + SHA-256 |
| `docs/00_…v1.0.md` | original supplied specification, verbatim |
| `docs/01_…v1.1.md` | first mathematical audit |
| `docs/02_C3v2-1_…md` | competence identifiability boundary |
| `docs/03_…v1.2.md` | completion specification |
| `docs/04_…completion_report.md` | this report |

Model weights are **not** committed; `tools/download_model.sh` restores the pinned
revision and verifies its SHA-256.

---

## 11. Final readiness decision

**C2+C3 is COMPLETE_WITH_EXPLICIT_NO_GO. Ready for the next component.**

Every architectural requirement is implemented in C++; every applicable gate has an
explicit result; the end-to-end pipeline executed against the pinned real model on CI;
certificates were verified against independent NumPy and HuggingFace oracles; and all
build, unit, stress and sanitizer tests pass.

The original A1 assumption is **falsified and remains an explicit NO-GO**. The project
objective is preserved by a mathematically defensible, implemented and validated
replacement: emergent overlapping density-peaks regions over continuation behaviour,
with no predefined taxonomy, no fixed cluster count and no human-assigned labels.

The next component must consume soft regions and must not assume disjoint domains or
internal-only competence. The learned encoder needs more data before it can be relied
on behaviourally; the validated structure currently rests on the behavioural metric
directly.
