# Alysane C2+C3 — Completion Specification v1.2

**Status legend:** **[P]** proven here or classical under stated assumptions; **[A]** assumption-dependent; **[H]** hypothesis; **[E]** experimental/measured; **[F]** falsified in an executed run.

This document completes the supplied v1.0 specification. v1.0 is archived verbatim in `00_original_C2_C3_spec_v1.0.md`; the first audit is `01_..._v1.1.md`; the identifiability boundary is `02_C3v2-1_...md`. **The project objective is unchanged**: an internal sub-symbolic self-knowledge representation that discovers behavioural knowledge regions, their structure, relevance and boundaries without predefined domain labels.

---

## 0. What changed and why

v1.0 made one load-bearing empirical assumption, **A1** (a hard, globally separated behavioural partition at a single scale window). It was measured and **falsified [F]** for the pinned SmolLM-135M on natural contexts at n = 8, 16, 32, 64. The failure is not an implementation defect: exact full-V TV distances concentrate near 1, so single-linkage merge heights have no useful gap.

Per the critical scientific requirement, A1 was **not** deleted, weakened, or hidden. Instead:

1. G3 (hard A1) is retained and still reports **FAIL / NO-GO**.
2. A mathematically different structure criterion is derived, implemented and gated: **calibrated multiscale persistence + overlapping soft regions** (§3, gate G3b).
3. v1.0 §1.3 signals that were specified but never implemented — self-agreement and continuation consistency — are now implemented and gated (§2, gate G9).

The objective survives because *soft, overlapping, scale-calibrated regions* still provide emergent sub-symbolic structure, relevance and boundaries. What is abandoned is only the claim that this structure must be a **disjoint hard partition at one global scale**.

---

## 1. Execution core and trace certificates

### 1.1 KV-cached decode (implementation requirement)

v1.0 §5 requires trace production "fused into the inference loop, zero extra passes", and §1.3 requires sampled continuations. Sampling continuations with a full re-forward per token is Θ(T·n·P); a KV cache reduces it to Θ(n·P + T·P).

**Proposition 1.1 (cache equivalence). [P]**
For a causal transformer with absolute-position rotary embeddings, let `z_full(x_{1..n+t})` be the next-token logits from a full forward pass and `z_cache` the logits from prefill `x_{1..n}` followed by `t` cached decode steps. Then `z_full = z_cache` exactly in exact arithmetic.

*Proof.* Causality implies the layer-ℓ hidden state at position i depends only on tokens ≤ i. The cache stores K/V for positions < n+t unchanged; RoPE is applied at absolute position, so cached keys carry the same rotation as in a full pass. Attention at the final position therefore sums identical terms in identical order. Induction on layers gives equality. ∎

In floating point the two paths differ only by summation order. Gate **G8** enforces max |Δp| ≤ 1e-5 on both single-step and multi-token generated continuations, and top-1 agreement = 1.

### 1.2 Corrected trace bound (replaces v1.0 Lemma 1)

**Theorem 1.2. [P]** With tail-uniform reconstruction on each trace's own top-k support,

|TV(p_x, p_y) − d̂_TV(x,y)| ≤ ρ_x + _y =: τ_safe(x,y).

*Proof.* TV(p_x, p̂_x) ≤ ½(ρ_x + ρ_x) = ρ_x, since mass ρ_x is redistributed over the tail. Same for y. Reverse triangle inequality for the metric TV gives the claim. ∎

v1.0's τ = ½(ρ_x+ρ_y) has an invalid proof. The gate enforces τ_safe; the half-sum variant is *reported* as an observation, never assumed. Verified independently by `tools/verify_certificates.py` (gate G2 + oracle).

### 1.3 Quantisation (corrects v1.0 §4)

v1.0 claimed int8 hidden-state quantisation error is "folded into τ". **This is false.**

**Proposition 1.3. [P]** Let Q quantise only pooled hidden-state statistics, leaving top-k IDs, top-k probabilities and tail mass unchanged. Then d̂_TV and τ_safe are **exactly invariant** under Q.

*Proof.* d̂_TV and τ_safe are functions of (top-k IDs, top-k probs, ρ) only. Q acts as the identity on those fields. ∎

**Corollary 1.4. [P]** Per-layer symmetric int8 with scale s_ℓ = max|·|/127 gives dequantisation error ≤ s_ℓ/2 per coordinate. This perturbs *encoder input*, requiring a separate encoder-Lipschitz bound; it does not enter the distributional certificate.

Gate **G11** verifies exact TV invariance (0 violations) and measures the induced encoder drift.

---

## 2. Continuation behaviour (v1.0 §1.3, now implemented)

### 2.1 Rollout kernel

Let K generate continuations s = (s_1..s_T) by ancestral sampling. Define the **spectrum kernel** of order ≤ 3 via the signed-hash feature map φ: V^T → R^D (D = 512), L2-normalised so k(s,s) = 1.

**Proposition 2.1. [P]** k(s,t) = ⟨φ(s), φ(t)⟩ is a positive-semidefinite kernel, being an inner product in R^D.

The **continuation mean embedding** is μ(x) = (1/R) Σ_r φ(s_r), and

d_C(x,y) := ‖μ(x) − μ(y)‖₂.

**Theorem 2.2. [P]** d_C is a pseudometric on contexts, bounded in [0,2].

*Proof.* It is the pullback of the Euclidean metric under x ↦ μ(x), hence symmetric, zero on the diagonal, and satisfies the triangle inequality. Each φ is unit norm, so ‖μ‖ ≤ 1 and d_C ≤ 2. It is only a *pseudo*metric because distinct contexts can share a mean embedding (hash collisions, finite R). ∎

**Proposition 2.3 (self-agreement and consistency). [P]**
a(x) = mean_{r<r'} k(s_r, s_r') ∈ [−1,1] measures internal continuation agreement;
κ(x,y) = ⟨μ(x), μ(y)⟩ is the two-sided cross-kernel. Both are computed from K alone with no labels.

Gate **G9** verifies all pseudometric axioms exactly (0 triangle/symmetry/diagonal violations) and the agreement range.

**Status.** d_C is a *different* behavioural geometry from one-step TV. Whether it yields useful structure is **[E]**, measured by G3b — not assumed.

---

## 3. Calibrated multiscale structure (A1 replacement)

### 3.1 Why a null calibration is required

Single-linkage merge heights always produce a dendrogram, on pure noise as well as on structured data. Reporting a dendrogram is therefore **not** evidence of structure. v1.0's A1 tried to fix this with an absolute gap threshold, which is scale-dependent and was falsified.

### 3.2 Persistence and the permutation null

Let D be a behavioural distance matrix on N sampled contexts. The 0-dimensional persistence of the single-linkage filtration has deaths equal to the sorted MST edge weights.

**Null model.** Permute the off-diagonal entries of D uniformly among pairs. This preserves the *exact marginal distribution* of distances while destroying metric geometry (triangle inequality, block structure).

**Definition 3.1 (significant merge).** For death index i, let q_i(α) be the (1−α) quantile of that order statistic under B null permutations. Merge i is **significant at level α** if deaths_i > q_i(α).

**Proposition 3.2. [P]** Under the null hypothesis "D is an exchangeable assignment of its own distance values to pairs", the per-index test has type-I error ≤ α + O(1/B) by construction of the empirical quantile.

*Limitation.* This is a **per-index** test, not a family-wise correction; and exchangeability is a modelling choice, not a property of real data. Significance is therefore evidence against *that specific null*, not proof of semantic domains. Reported as **[E]** with the null explicitly stated.

### 3.3 Subdominant ultrametric

**Theorem 3.3. [P]** The single-linkage merge height U(x,y) = min over paths of max edge weight is the **subdominant ultrametric**: U ≤ D, U satisfies the strong triangle inequality U(x,z) ≤ max(U(x,y), U(y,z)), and U is maximal among ultrametrics bounded by D.

*Proof.* U ≤ D because the direct edge is a candidate path. Strong triangle: concatenating optimal paths for (x,y) and (y,z) gives a path for (x,z) with max edge ≤ max(U(x,y),U(y,z)). Maximality: any ultrametric U' ≤ D must satisfy U'(x,y) ≤ max edge on every path, hence U' ≤ U. ∎

Verified in unit tests (including maximality) and by the independent NumPy oracle.

### 3.4 Overlapping soft regions

**Definition 3.4.** Given anchors A = {a_1..a_K} and radii r_a, define the partition of unity

w_a(x) = b(d(x,a)/r_a) / Σ_b b(d(x,b)/r_b),  b(u) = (1−u²)² for u<1, else 0.

If no anchor covers x, assign w = 1 to the nearest anchor and flag x as uncovered.

**Proposition 3.5. [P]** w_a(x) ≥ 0, Σ_a w_a(x) = 1, and w is C¹ in the distances on the covered set. The **boundary score** β(x) = H(w(x))/log K ∈ [0,1] is 0 for a point inside exactly one anchor ball and approaches 1 for uniform membership.

This gives **emergent, overlapping, label-free regions with an explicit boundary signal** — the original objective — without asserting disjointness or a fixed cluster count. K is a *cover resolution*, not a domain count.

### 3.5 Gate G3b (pre-registered)

G3b PASS requires all of:
- ≥ 1 significant merge under the permutation null (α = 0.05, B = 20);
- coverage ≥ 0.80 (fraction of points inside at least one anchor ball);
- bootstrap stability mean ARI ≥ 0.50 over B = 10 subsamples at 80% (anchors and radii **re-derived** per subsample).

These thresholds were fixed before the completion run and are not tuned post hoc.

---

## 4. Learned space, energy, routing (carried forward from v1.1)

- **Spectral contrastive (corrected Theorem B).** Under GᵀG = I, minimisers span the top-d eigenspace of the normalised adjacency A = D^{-1/2}WD^{-1/2}, i.e. the **bottom-d** of L_sym. v1.0 said "top-d of L_sym" — wrong end. The universal downstream linear-probe bound is **removed**: it needs task/label compatibility assumptions.
- **Whitening vs sphere.** Whitening gives E‖f² = d, incompatible with a unit-sphere constraint for d>1. The implementation uses R^d + whitening.
- **Energy.** E_φ(z) = ½‖z‖² + aᵀtanh(Wz+b) is conservative; ∇E is exact (finite-difference tested). Trained by denoising score matching. Traversal is gradient **descent** on E (density ascent). Population consistency is classical **[P]**; the finite-data pilot is **[E]**.
- **Routing certificate (corrected Corollary 2).** If ∃x*∈D_i with d_B(x,x*) ≤ r then d_B(x,y) ≤ r + diam(D_i) for all y∈D_i. Latent-only nearest-region routing is **not** certified without a lower distortion bound. Gate G5 checks the behavioural certificate.

---

## 5. Reshape under perturbation

**Theorem 5.1. [P]** For any two parameter settings, with Δz := max_x ‖z'(x) − z(x)‖_∞,

|d_B'(x,y) − d_B(x,y)| ≤ 2·Δz.

*Proof.* TV(softmax(a), softmax(b)) ≤ ‖a−b‖_∞ (safe softmax bound). Apply the reverse triangle inequality to |TV(p'_x,p'_y) − TV(p_x,p_y)| ≤ TV(p_x,p'_x) + TV(p_y,p'_y) ≤ 2Δz. ∎

This is a **measurable, assumption-free** bound: Δz is observed directly, no product-of-spectral-norms constant is required. Gate **G12** perturbs a weight tensor by a known relative norm, measures Δz and the actual TV drift, and verifies 0 bound violations. The empirical local Lipschitz constant L_z = Δz/‖δθ is reported as **[E]**.

*Limitation.* This certifies drift for the *executed* perturbation, not for all fine-tunes. A global L_K remains **[E]** and is not claimed.

---

## 6. Competence boundary (from C3v2-1, binding)

Self-agreement, continuation consistency, entropy and boundary score are **proxies**. By the C3v2-1 two-world theorem, actual competence is **not identifiable from internal observables alone** (worst-case error ≥ 0.5). Every competence claim must declare an evidence contract (internal sound verifier / verified pointwise bit / population outcomes / explicit transfer assumptions). C2+C3 ships **no** competence claims; gates G6/G10 are boundary *proxy* gates only.

---

## 7. Complexity and memory (measured architecture)

Pinned SmolLM-135M: d=576, L=30, H=9, KVH=3, FF=1536, V=49152.

- Non-embedding MACs/token: L(2d² + 2d·d_kv + 3d·FF) + Vd = **134,479,872**.
- n=64 context ≈ 8.61 GMACs ≈ 17.2 GFLOPs. N=10⁵ ≈ **1.72×10¹⁵ FLOPs** (v1.0's 10¹⁸ was off by ~1000×).
- KV cache per stream: 2·L·cap·KVH·hd·4 bytes. At cap=72: **≈ 3.3 MB** fp32.
- Rollout cost with cache: Θ((n + R·T)·P) per context vs Θ(R·T·n·P) without — a ~n× saving on the generated portion.
- Trace: L·2d = 34,560 floats = 138 KB fp32 → **34.6 KB int8** + L scales (ratio ≈ 3.98×).
- Full-V TV: Θ(V) per pair; exact N×N matrix Θ(N²V).
- Persistence null: B × O(N²) MST. Soft regions: O(NK).

---

## 8. Gate inventory (final)

| Gate | Criterion | Type |
|---|---|---|
| G1 | C1 parity vs independent HF fp32 oracle | engineering |
| G2 | trace certificate, 0 violations of τ_safe | certificate |
| G3 | **hard A1 partition — retained, falsified** | scientific **[F]** |
| G3b | calibrated persistence + soft-region coverage/stability | scientific **[E]** |
| G4 | landmark ψ distortion bounds | certificate |
| G5 | behavioural routing certificate | certificate |
| G6 | boundary proxy (energy/margin) | proxy |
| G7 | two-store serialization roundtrip | engineering |
| G8 | KV-cache parity | engineering |
| G9 | continuation pseudometric axioms + signal ranges | certificate |
| G10 | continuation boundary proxy | proxy |
| G11 | quantisation TV invariance | certificate |
| G12 | reshape drift bound | certificate |

**Interpretation rule.** G3 remains NO-GO and is never silently dropped. The component's readiness decision is reported against the *full* gate set, with G3's falsification and G3b's replacement stated explicitly.
