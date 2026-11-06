# A1 Measurement Protocol — Execution Report

**Protocol:** [`05_A1_measurement_protocol_v1.0.md`](05_A1_measurement_protocol_v1.0.md) (authoritative, verbatim)
**Implementation:** `include/probe.h`, `src/probe.cpp` — CLI `alysane probe`
**Run:** GitHub Actions **42181955117**, AMD EPYC 9V45 (avx2 + avx512f + fma), GCC 13, 4 threads
**Artifact:** `reports/probe_report.json`
**Configuration:** N=256 training contexts, n=64 hold-out, V=49152, 48 ε-thresholds, B=20 subsamples, M=20000 pairs, 64 landmark anchors, top-K=128, R=4, T=8

## Final verdict

> ## `PARTIAL_GO`
>
> A measurement-stable regime **exists** on all three candidate distances — H1, H2
> and H3 all pass — but the usable margin is far below the certified margin on
> every metric. Per protocol §4: **architecture proceeds; C5 competence field
> becomes the primary interface; fine-tuning is gated until the margin widens;
> the plateau is monitored as telemetry.** `final_branch` is empty: none of the
> five GO branches (B1–B5) is reachable.

| metric | plateau | k | M usable | M required | H1 | H2 | H3 | verdict |
|---|---|---|---|---|---|---|---|---|
| `one_step_tv` | [0.995612, 0.997900) | **9** | 0.018538 | 0.368281 | ✅ | ✅ | ✅ | PARTIAL_GO |
| `rollout_js_root` | [0.704335, 0.728743) | 2 | 0.073977 | 3.330219 | ✅ | ✅ | ✅ | PARTIAL_GO |
| `continuation_mmd` | [0.976442, 1.056805) | 2 | 0.236761 | 6.895494 | ✅ | ✅ | ✅ | PARTIAL_GO |

**This refines the earlier "A1 NO-GO" conclusion.** The prior gate tested only
the composite margin requirement and reported a binary failure. The protocol
separates *existence of structure* from *usability of the margin*, and the
measurement shows the former holds while the latter does not. That is a
different and more actionable statement.

## Phase 0 — metric audit

All three distances pass the metric axioms (`probe::run_probe_tests`, CI-green):

| | symmetric | triangle | identity on empirical laws | bounded |
|---|---|---|---|---|
| `one_step_tv` | ✅ | ✅ | ✅ | [0,1] |
| `rollout_js_root` | ✅ | ✅ | ✅ | [0,√ln2] |
| `continuation_mmd` | ✅ (Gaussian kernel) | ✅ | — (kernel mean embedding) | [0,2] |

`one_step_tv` is computed from the model's **own** top-K distributions, not from
the reconstruction — the reconstruction error is carried separately as τ.
`rollout_js_root` compares the rollout-averaged next-token laws
q_x = (1/RT)Σ_t p_K(·|x+s_<t). `continuation_mmd` uses the Gaussian kernel on
rollout-averaged hidden states.

## Corrections to the protocol

Six defects were found and corrected rather than implemented as written. Each is
preserved in the code with a `C1`–`C6` comment and a unit test where testable.

**C1 — the λ₂ criterion is unsatisfiable.** §4 requires λ₂ ≥ λ_min with k ≥ 2.
For a connected graph the normalised Laplacian has λ₁ = 0 exactly; with k ≥ 2
components λ₂ = 0 as well. The intended quantity is the **(k+1)-th** eigenvalue
(the first non-zero one). Unit test `CORRECTION C1: lambda_2 == 0 when k=2`
asserts this on a two-clique graph and passes on CI.

**C2 — power iteration returns λ_max, not λ₂.** §4 specifies power iteration on
L_sym. Power iteration converges to the dominant eigenvalue. Replaced by exact
dense eigendecomposition (LAPACK `ssyevd`), with an explicit kernel-multiplicity
check `#{λ < 1e-5} == k`.

**C3 — Lemma M2 does not transfer.** The claim that TV-specific certificates
transfer under bi-Lipschitz equivalence fails for root-JS: the verified bounds
are TV/√2 ≤ root-JS ≤ √(ln2·TV). The upper bound is **√TV**, so the metrics are
*not* bi-Lipschitz equivalent and one-step certificates do not transfer. Each
metric therefore carries its own error term.

**C4 — the capture radius is vacuous.** √m·ε with ε → 1 gives a radius ≥ the
diameter for m ≥ 1. Reported as `capture_vacuous: true` on all metrics rather
than presented as a guarantee.

**C5 — the Hoeffding interval is sample-conditional.** It bounds deviation of the
empirical mean on the *drawn* contexts, not a population claim. Reported as such.

**C6 — H1 needs ≥ ε_hi, not > ε_hi.** k(ε) constant on the half-open interval
[ε_lo, ε_hi) implies every cross-component distance is **≥ ε_hi**. The merge *at*
ε_hi is itself a cross pair at exactly that height, so testing `d ≤ ε_hi` flags
the boundary and makes H1 fail on a genuine plateau. The violation test is
strict.

## Three defects in my own implementation

Found by running the protocol, not by inspection — recorded because they are
the kind of error that manufactures a pass.

1. **H3 counted within-component merges as bridges.** The original wording
   ("any two subtrees each carrying ≥1% mass") flags legitimate within-domain
   cohesion. At k=9, most high merges are *inside* a component. Corrected to
   count only merges joining distinct components of the plateau partition
   (Kruskal over component labels). Before: H3 = false on all metrics. After:
   H3 = true.
2. **H1 used an unpowerable sequential test as the criterion.** Bonferroni over
   32640 pairs at δ=0.05 requires **6.65 × 10⁹** samples per pair for
   `one_step_tv`. Reporting "H1 fail" from that test would have been reporting a
   power failure as a structural failure. The criterion is now **exact empirical
   cross-pair separation** (2012 cross pairs checked, 0 violations,
   min cross = 0.997900 = ε_hi); the population test is reported separately as
   `sequential_test_powered: false` with the required sample count.
3. **τ = 0 was assumed for the rollout metrics.** They carry Monte Carlo error.
   Now bounded by McDiarmid: TV(q̂,q) ≤ 1/√R + √(ln(1/δ)/(2R)), propagated through
   root-JS ≤ √(ln2·TV) and the MMD triangle inequality (×2).

## The certification-cost finding

The McDiarmid bounds make the rollout metrics' error explicit, and the result is
the most consequential number in this report:

| metric | τ95 at R=4 | R required for τ ≤ 0.05 |
|---|---|---|
| `one_step_tv` | 0.184141 | 0 — no sampling, τ is reconstruction error |
| `rollout_js_root` | 1.665109 (**vacuous**, bound is 0.829) | **8 388 608** |
| `continuation_mmd` | 3.447747 (**vacuous**, bound is 2.0) | **32 768** |

At R=4 the rollout error bounds exceed the diameter of the metric space: they
say nothing. Certifying `continuation_mmd` needs ~3.3 × 10⁴ rollouts per context;
certifying `rollout_js_root` needs ~8.4 × 10⁶. At the measured 0.296 s per rollout
(417.2 s / 1408 rollouts) that is 2.7 h and 29 days **per context**. **Only `one_step_tv` is certifiable at
feasible cost**, and even its τ95 = 0.184 is what makes M_required = 0.368 ≫
M_usable = 0.0185.

The binding constraint is therefore not the geometry — it is **top-K
reconstruction error**. Reducing τ is the single lever that could move
`one_step_tv` from PARTIAL_GO to GO: at τ95 ≤ 0.0009 the required margin would
fall below the usable margin. That means K large enough to capture essentially
all probability mass, which is a memory/bandwidth trade, not a modelling one.

## Detailed results — `one_step_tv`

| quantity | value |
|---|---|
| plateau interval | [0.995612, 0.997900) |
| plateau width | 0.002288 |
| components k | **9** |
| cross pairs checked | 2012 |
| cross violations | **0** |
| min cross distance | 0.997900 (= ε_hi) |
| kernel multiplicity | matches k |
| overlap fraction | 0.175 |
| gap mass at plateau | 0.031 |
| routing rate | 0.792 |
| routing-bound satisfaction | **1.000** |
| τ95 | 0.184141 |
| M usable / required | 0.018538 / 0.368281 |

The routing certificate `d_B(x,y) ≤ d_B(x,x*) + diam(D_i)` holds on **100%** of
routed contexts, and 79.2% of contexts route at all (target 0.80 — marginally
short). Note the granularity difference: this hard plateau partition has **9**
components, whereas the C3 density-peaks structure reports **63** soft overlapping
modes. These are not the same object and should not be conflated — the plateau
partition is a threshold cut of the same geometry at a much coarser resolution,
and 63 modes nest inside 9 components. Which granularity is useful downstream is
an open question, not a result.

## What is proven, assumed, hypothesised, measured

- **[P]** Metric axioms for all three distances; TV/√2 ≤ root-JS ≤ √(ln2·TV);
  λ₂ = 0 for k ≥ 2; McDiarmid bounds; the C6 boundary argument.
- **[C]** Corrections C1–C6 and the three implementation fixes, each with code
  and (where testable) a CI-asserted unit test.
- **[E]** Every number in this report, from run 42181955117 on the pinned
  SmolLM-135M (`1d461723eec654e65efdc40cf49301c89c0c92f4`) with HF parity
  max-KL 2.9 × 10⁻⁷.
- **[H]** That reducing τ95 would move `one_step_tv` to GO — arithmetic on the
  margin formula, **not** an executed experiment.
- **Not claimed:** population-level separation (the sequential test is
  underpowered), capture-radius guarantees (vacuous), rollout-metric
  certification (vacuous at R=4), or competence (the probe measures behavioral
  geometry, never correctness).

## Reproduction

```bash
# restore pinned weights
bash tools/download_model.sh     # rev 1d461723eec654e65efdc40cf49301c89c0c92f4
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
./build/alysane prepare-data data
./build/alysane gates models data reports
ALYSANE_PROBE_GRID=48 ALYSANE_PROBE_B=20 ALYSANE_PROBE_M=20000 \
  ALYSANE_PROBE_LANDMARKS=64 OMP_NUM_THREADS=4 \
  ./build/alysane probe models data reports
# exit 0 = GO/PARTIAL_GO, 3 = NO_GO
```

`probe_report.json` contains every threshold, every hypothesis verdict with its
evidence, the full ε-sweep, the spectrum, and the error bounds — sufficient to
recompute any verdict independently.

## Recommendation

Adopt **PARTIAL_GO** as the gate outcome and proceed under its §4 conditions:

1. **C5 competence field is the primary interface.** The behavioral geometry is
   telemetry, not a certification surface. This is consistent with the C3v2-1
   two-world barrier result (internal-only competence estimation has worst-case
   error ≥ 0.5).
2. **Fine-tuning stays gated.** Do not fine-tune on the strength of a 0.0185
   margin against a 0.368 requirement.
3. **Investigate τ reduction before any margin claim.** Sweep K upward and
   re-run the probe; this is the only feasible path to GO and it is a
   memory/bandwidth decision.
4. **Do not certify rollout metrics at R=4.** Either raise R to ≥3.3 × 10⁴ or
   report them as descriptive only.

I have not proceeded to C4. The next action is yours.
