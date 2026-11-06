# Alysane — A1 Measurement Protocol (Go/No-Go Gate)

**Precedent.** With the C2+C3 spec fixed, the load-bearing fact of the system is Assumption A1
(robust separation of the behavioral metric). Theorem 1 makes the architecture's core claim —
*domains are discovered, not written* — equivalent to A1 holding with a computable margin on a
real model. This document is the executable, statistically guarded measurement of A1. Everything
estimable is given an estimator with a confidence guarantee; every pass/fail criterion is a
numeric threshold, not a judgment call.

Status legend: **[P]** proven, **[H]** hypothesis, **[C]** conjecture, **[E]** experimental.

---

## §1. What Exactly We Measure

A1 decomposes into three testable sub-hypotheses on the context distribution μ and behavioral
metric d_B:

- **H1 (Separation).** There exist k ≥ 2 groups with all cross-group distances > ε_out.
- **H2 (Cohesion).** Each group is ε_in-connected, ε_in < ε_out.
- **H3 (No bridges).** No chain of ε_in-edges joins two groups (equivalently: the merge structure
  between groups happens only at height > ε_in).

We do not know the groups in advance — that is the point. So the measurement object is not
"verify a given partition" but **find the scale spectrum of d_B and test for a plateau**: an
interval [ε_in, ε_out) in which the coarse-graining of μ induced by G_ε is constant, non-trivial,
and stable.

**Definition M1 (Scale signature).** For scale ε: k(ε) = number of connected components of G_ε on
S; λ₂(ε) = Fiedler value of its normalized Laplacian; β(ε) = fraction of μ-mass living in
components covering < 1% of mass ("gap/novelty mass"). A **plateau** is a maximal interval
W = [ε_lo, ε_hi) where k is constant ≥ 2, λ₂ ≥ λ_min, and β ≤ β_max.

**Why a plateau is exactly A1.** Inside W: H2 holds by connectivity at ε_lo and constancy of k;
H1 holds with ε_out = ε_hi (no cross-edges below it, else k would drop); H3 holds because the first
merge of two large components *is* the endpoint ε_hi — any bridge would produce a merge inside the
window. The plateau is not a diagnostic *of* A1; it **is** A1, made measurable. **[P]**

---

## §2. Estimators and Their Guarantees

Computing all C(N,2) pairwise distances is infeasible (N = 10⁵ ⇒ 5×10⁹ pairs). The protocol rests
on a two-stage edge filter with a rigorous false-negative bound.

**Stage 1 — candidate generation (cheap).** From the bootstrap embedding ψ_A (Lemma 2/3), generate
candidate edges as pairs within radius r in ψ-space via approximate nearest-neighbor search
(Euclidean, HNSW-type index).

**Stage 2 — exact verification.** For each candidate pair, compute d_B from traces (Lemma 1, error
≤ τ). Keep edges with d̂_TV ≤ ε.

**Lemma M1 (Capture guarantee, [P]).** Let landmarks cover S with radius ε_cov (Lemma 3). Then for
covered points, every true edge {d_B ≤ ε} is captured by candidate radius r = √m · ε.

*Proof.* Lemma 2 upper bound: ‖ψ(x) − ψ(x')‖ ≤ √m · d_B(x,x') ≤ √m · ε. □

*Practical note.* The √m slack is the rigorous fallback; the realized distortion is far tighter
(Corollary 1's margin is what actually matters), and the probe measures it directly — the audit
"realized vs. worst-case capture radius" is itself a reported quantity. **[E]**

**Estimator 1 — correlation integral.** C(ε) = Pr_{x,x'∼μ}[d_B ≤ ε], estimated from M uniformly
sampled pairs: Ĉ(ε) = (1/M) Σ 1[d̂_TV ≤ ε]. Hoeffding: width √(ln(2/δ)/(2M)) at confidence 1−δ. A
stepped (non-smooth) Ĉ is the first cluster signature.

**Estimator 2 — merge-height spectrum.** Run single-linkage (Kruskal sweep) over the *verified*
edge list sorted by d̂_TV. Output: for each merge event, its height h and the sizes of the merged
subtrees. From this one object read off: k(ε) (count merges with h ≤ ε), the plateau window W, and
ε_out (height at which the first merge of two large subtrees occurs). Given the verified edges, the
sweep is **exact** — the only statistical content is the edge capture of Lemma M1, plus a
quantitative false-negative analysis if candidates are subsampled at rate q:
Pr[miss a specific bridge] ≤ (1−q)^ℓ for a bridge of ℓ candidate pairs; Chernoff controls edge-count
errors. Default: q = 1 (verify all candidates); subsampling only as a budget valve.

**Estimator 3 — cohesion and the lower window edge.** Within each plateau component, scan ε
downward from ε_lo and find the smallest ε at which the component's giant-component fraction
reaches 1 − η: that is ε̂_in per component. Report the window [ε̂_in, ε_hi] with per-endpoint
confidence intervals from the estimator CIs.

**Estimator 4 — Fiedler value.** Power iteration on the sparse normalized Laplacian of the verified
graph at plateau scale: λ̂₂, convergence-certified (Rayleigh quotient residual bound — **[P]**
standard).

**Estimator 5 — stability across resampling.** Bootstrap the context sample B = 20 times; re-run
the full pipeline; report the overlap of the plateau windows. Stable windows across resamples = A1
is a property of μ, not of the sample. **[P]** interpretation, von Luxburg-type consistency; the
realized overlap is **[E]**.

**Estimator 6 — trace-error audit.** On a subsample, compare d̂_TV against exact full-vocabulary TV
to measure the realized τ distribution (not just the bound of Lemma 1); report τ̂₉₅, the 95th
percentile, used in all margin tests below.

---

## §3. Protocol Phases

**Phase 0 — sample construction.** N = 10⁵ contexts from target traffic; dedupe; length filter
8 ≤ n ≤ 512. Separately, build the **shift battery**: synthetic families of context perturbations
(self-paraphrases generated by K, format shifts, register shifts, topic-adjacent substitutions) —
this is the Corollary-2 routing test set, no human labels.

**Phase 1 — landmarks and bootstrap geometry.** Greedy farthest-point (k-center) selection over ψ
iterates: each new landmark costs N TV computations; the k-center radius curve r(m) is itself
diagnostic — plateaus in r(m) indicate discrete distance scales **[H, measured]**. Deliver
covering-number estimate N̂_ε.

**Phase 2 — scale scan.** Candidate generation at r = √m · ε_hi^max; verify; Kruskal sweep; extract
k(ε), merge spectrum, Ĉ(ε), λ̂₂(ε), β̂(ε).

**Phase 3 — hypothesis tests.** At the detected plateau:
- H1: for each pair of large components, sample cross pairs sequentially with early stopping:
  accept separation when the upper confidence bound on Pr[d_B ≤ ε_hi] falls below δ_sep; spend the
  full budget otherwise and report failure.
- H2: cohesion scan per component (Estimator 3).
- H3: bridge audit on the merge spectrum: any merge of two subtrees each ≥ 1% mass at height
  ≤ ε_in fails H3.
- Multiple-comparisons control: Bonferroni over the ε grid and the C(k,2) component pairs.

**Phase 4 — Corollary-2 routing self-test.** Hold out the shift battery; route each shifted context
to its nearest plateau region in ψ; measure (a) correct-origin routing rate, (b) the realized bound:
Pr[d_TV(p_K(·|x̃), p_K(·|D_routed)) ≤ 2ε* + 2τ̂₉₅].

---

## §4. Decision Rule

Define the **usable margin** M = ε̂_out − ε̂_in and the **required margin**
M_req = 2(τ̂₉₅ + Δ_budget), where Δ_budget = L_K ‖δθ_budget is the fine-tune drift budget from
Theorem 3 (C8 fixes the constant; for the gate, use the conservative estimate and mark **[E]**) —
this is the Corollary-3 stability precondition, checked *before* any fine-tune is permitted.

| Outcome | Criteria | Verdict |
|---|---|---|
| **GO** | Plateau W exists; k ≥ 2; M > M_req; window overlap across bootstraps ≥ 90% of length; covered mass ≥ 80%; routing rate ≥ target | Full architecture proceeds to C4/C5 |
| **PARTIAL GO** | Plateau exists but M ≤ M_req, or 15–40% gap mass | Architecture proceeds; C5 competence field becomes primary interface; fine-tuning gated until margin widens; plateau monitored as telemetry |
| **NO-GO** | No plateau; Ĉ(ε) smooth; k(ε) = 1 up to ε_hi^max | Branch decision (§5) |

Report template (machine-readable JSON, versioned): all estimates + CIs, the four spectra curves,
margin accounting τ̂₉, Δ_budget, M, M_req, and the verdict with per-hypothesis pass/fail.

---

## §5. NO-GO Branches (Pre-Committed)

- **B1 — wrong metric, right structure.** The single-step d_TV may entangle domains that multi-step
  behavior separates. Replace the metric by d_JS^{1/2} over r-step self-rollouts (Jensen–Shannon
  root is a metric) or by continuation-consistency κ. **Lemma M2 [P]:** every result of C2+C3
  (Lemmas 1–3, Theorems 1–3, Corollaries 1–3) holds verbatim for any metric d on the output space
  with the triangle inequality; only the cost model changes (rollout factor 4–8×). No mathematics
  is lost in this branch.
- **B2 — degenerate structure.** Plateau only at trivial ε or k=1: domains collapse; the honest
  conclusion is that competence structure lives in the *field* over Z (C5) rather than in regions.
  Architecture proceeds with C5-first ordering; Theorem 1 unused.
- **B3 — entangled metric.** No plateau under any tested metric: this would falsify the research
  thesis as stated. The pre-committed response is architectural, not denial: hierarchical *soft*
  domains (mixture membership instead of partition), which is exactly the mathematics C4 must
  develop in its soft-regime section either way.

---

## §6. Complexity and Implementation (offline tool `alysane-probe`, pure C++)

| Phase | Dominant cost | Budget |
|---|---|---|
| 1: traces + ψ_A | N·m TV evals, Θ(V) each | 10⁵ × 512 × 49k ≈ GPU-hours / CPU-days |
| 2: candidates + verify | ANN build Θ(Nm); verify Θ(E_cand·V) | E_cand controlled by r audit |
| 3: sweep + spectra | Kruskal Θ(E log E); power iteration Θ(E·iters) | trivial |
| 4: routing test | shift battery × routing | trivial |

```cpp
// alysane-probe (offline binary; reads (θ_K, θ_S) stores, writes report JSON)
struct ProbeReport {
    std::vector<float> eps_grid, corr_integral, fiedler, gap_mass;
    std::vector<MergeEvent> merge_spectrum;      // (height, size_a, size_b)
    Interval plateau;                            // [eps_in, eps_out] + CIs
    float tau95, margin_usable, margin_required;
    float covered_mass, routing_rate, routing_bound_rate;
    Verdict verdict;                             // GO | PARTIAL_GO | NO_GO(+branch)
};

// modules: TraceSpooler · TVKernel(AVX-512) · KCenterLandmarks · HNSWIndex
//          UnionFindSweep · PowerIteration · SequentialTester(CI early-stop)
//          ShiftBattery · ReportWriter
```

Sequencing dependency: the probe runs against the **retargeted** K, so C1's weight-map loader and
perplexity-parity gate are the only engineering prerequisites. The probe is also retained as the
permanent telemetry tool: replay windows from traffic re-run through it to detect margin erosion
after any fine-tune (Corollary-3 monitoring).
