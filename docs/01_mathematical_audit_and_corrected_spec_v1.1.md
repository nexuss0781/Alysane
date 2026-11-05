# Alysane C2+C3 — Mathematical Audit, Corrected Specification, and Gate Protocol v1.1

**Status legend:** **[P]** proven here or classical under stated assumptions; **[H]** hypothesis; **[C]** conjecture; **[E]** experimental/measured; **[F]** falsified or failed gate in the executed pilot.

This document preserves the project goal: a sub-symbolic self-knowledge space that discovers domain structure from the model's own behavior, supports inspection/traversal/editing, and exposes competence boundaries without handwritten domain labels. It does **not** preserve incorrect proofs from v1.0. The original v1.0 text is archived verbatim in `00_original_C2_C3_spec_v1.0.md`.

---

## 0. Component map and scope

| Component | Responsibility | C2+C3 pilot dependency |
|---|---|---|
| C0 | Weight-store contracts, serialization, provenance | knowledge manifest + self-weight binary |
| C1 | Retarget pretrained SmolLM-135M into Alysane C++ execution core | fp32 parity gate |
| C2 | Trace operator, behavioral distance, trace certificates | trace + TV + top-k audit |
| C3 | Domain emergence, similarity graphs, A1 separation gate | MST / component / bootstrap audit |
| C4 | Hierarchy / sub-domains / cluster tree | blocked if A1 is required as a hard global partition |
| C5 | Competence calibration | entropy/agreement are proxies only |
| C6 | Long-term self-knowledge memory | landmark store is pilot precursor |
| C7 | Reshape under fine-tuning | local drift machinery; full constants deferred |
| C8 | Numerical analysis / full proofs / production scaling | interval bounds, population limits, CI |

The executed deliverable is a **pilot go/no-go gate**, not a production sign-off. Production requires live traffic, larger N, multiple seeds, and the C1 architectural-swap specification beyond exact Llama-compatible execution.

---

## 1. Formal model

### 1.1 Contexts and knowledge kernel

Let V be a finite vocabulary, |V|=49152 for the pinned SmolLM-135M checkpoint. Let X = V^{≥1} be nonempty token contexts. (If empty contexts are desired, define K(∅) by a fixed BOS convention; v1.0's X=V* included ∅ without specifying K(∅).) X is countable; with the power-set σ-algebra every function on X is measurable. No topological continuity claim is needed.

K_θ maps x∈X to p_θ(·|x)∈Δ(V). Behavioral distance is

d_B^θ(x,y) = TV(p_θ(·|x), p_θ(·|y)) = ½ ∑_{v∈V} |p_θ(v|x)-p_θ(v|y)|.

**[P]** d_B is a pseudometric on X (metric on behavior equivalence classes). It is bounded in [0,1].

### 1.2 Trace operator

The pilot trace is

T(x) = (mean_ℓ, max_ℓ)_{ℓ=1..L}; α(x); Λ(x); H(x)

where:
- mean_ℓ, max_ℓ ∈ R^d are tokenwise mean/max of the post-block residual stream at layer ℓ;
- α(x)∈R^H is per-head mean causal attention entropy over query positions;
- Λ(x) = ((id_1,p_1),…,(id_k,p_k), ρ) contains **top-k token IDs**, probabilities, and tail mass ρ=1-∑p_i;
- H(x) is next-token entropy.

v1.0 omitted top-k IDs. Without IDs, a tail-uniform TV estimator is not computable from probabilities alone.

### 1.3 Corrected trace certificate (replaces Lemma 1)

Define the tail-uniform reconstruction

p̂_x(v) = p_x(v) if v∈S_x, else ρ_x/(|V|-k),

where S_x is the top-k support. Let d̂_TV(x,y)=TV(p̂_x,p̂_y).

**Theorem C2.1 (safe trace bound). [P]**

|TV(p_x,p_y) - d̂_TV(x,y)| ≤ ρ_x + _y =: τ_safe(x,y).

*Proof.* TV(p_x,p̂_x) = ½∑_{vS_x}|p_x(v)-ρ_x/(|V|-k)| ≤ ½(ρ_x+ρ_x)=ρ_x. Similarly for y. The reverse triangle inequality for TV gives the claim. ∎

v1.0 claimed τ=½(ρ_x+ρ_y) with a proof that bounds moved mass incorrectly. The safe bound above is what the C++ gate enforces. In the executed audit, the original half-sum bound happened not to be violated on the sampled pairs, but that is an **[E]** observation, not a proof.

**Quantization note.** Quantizing hidden-state trace statistics does not change the output-distribution TV bound if p and top-k probabilities are unchanged. It changes encoder input geometry and requires a separate encoder-Lipschitz perturbation bound. If output probabilities/logits are quantized, add an explicit distributional perturbation term; do not fold hidden-state quantization into τ.

---

## 2. Bootstrap self-knowledge space

Let A={a_1,…,a_M} be landmarks. Define ψ_A(x)=(d_B(x,a_1),…,d_B(x,a_M)).

### 2.1 Distortion

**Theorem C2.2 (one-sided landmark distortion). [P]**
If min_i d_B(x,a_i)≤r, then

d_B(x,y)-2r ≤ ||ψ_A(x)-ψ_A(y)||_∞ ≤ ||ψ_A(x)-ψ_A(y)||_2 ≤ √M d_B(x,y).

*Proof.* Upper: each coordinate is 1-Lipschitz by reverse triangle. Lower: choose i with d_B(x,a_i)≤r. If d_B(x,y)≤2r the lower bound is trivial. Otherwise d_B(y,a_i)≥d_B(x,y)-d_B(x,a_i)>r≥d_B(x,a_i), so |d_B(x,a_i)-d_B(y,a_i)|≥d_B(x,y)-2r. ∎

**Corollary C2.3 (cluster preservation, ℓ∞ version). [P]**
Under A1, if landmarks cover S with radius r and max_i diam_B(D_i) < ε_out - 2r, then every within-domain pair has smaller ψ_∞ distance than every cross-domain pair.

The ℓ2 version has the extra √M factor and is often vacuous for large M. The pilot reports ℓ2 audits but uses the ℓ∞ certificate as the cleaner mathematical statement.

### 2.2 Random covering (replaces Lemma 3)

v1.0's claim that m≥N_ε ln(1/δ) covers S from covering number alone is false: covering number does not lower-bound sampling mass.

**Theorem C2.4 (mass-weighted random cover). [P]**
Let {z_j}_{j=1..N_ε} be an ε-net of S and β_ε = min_j μ(B_ε(z_j))>0. If landmarks are sampled i.i.d. from μ and

m ≥ (1/β_ε) ln(N_ε/δ),

then with probability ≥1-δ every net point is within ε of a landmark, hence S is covered at radius 2ε.

*Proof.* Union bound and (1-β)^m≤e^{-βm}. ∎

For empirical finite support, β_ε can be estimated from empirical ball masses; for population continuous-like behavior, β_ε may be zero and the guarantee is void.

---

## 3. Domain emergence and A1

### 3.1 Similarity graph and components

G_ε=(S,E_ε), edge iff d_B≤ε. Domains at scale ε are connected components.

**Assumption A1 (robust separation). [H]**
There exists a partition {D_i}_{i=1..q} of S and 0≤ε_in<ε_out such that:
1. each D_i is ε_in-connected;
2. every cross-domain pair has d_B>ε_out.

v1.0's S3 (no bridge chains) is redundant given S2 and ε_in<ε_out.

**Theorem C3.1 (component stability). [P] under A1**
For every ε∈[ε_in,ε_out), components of G_ε on S are exactly {D_i}.

*Proof.* ε≥ε_in preserves internal paths; ε<ε_out forbids cross edges. ∎

**Limitation.** This is uniqueness only inside the chosen scale window and relative to μ. It is not a canonical, label-free, scale-free partition of all possible contexts.

### 3.2 Executed A1 gate

The pilot estimates A1 from the exact full-V TV matrix using the MST. For q components, the single-linkage window is

ε_in(q) = largest MST edge retained when N-q edges are kept,
ε_out(q) = next MST edge.

Pilot usefulness thresholds (pre-registered for the final gate):
- 2≤q≤16;
- minimum component size ≥ max(8, floor(N/20));
- width ε_out-ε_in > 0.02;
- ratio ε_out/ε_in > 1.10;
- 20 bootstrap subsamples (80% vertices), mean ARI ≥ 0.8 at the midpoint cut.

These thresholds are engineering criteria for a **useful hard partition**, not a mathematical necessity. A1 may be false while a softer hierarchy remains useful.

**Pilot result: [F] A1 is not supported** for the pinned SmolLM-135M on the executed natural-context mixture, at context lengths 8, 16, 32, and 64. Exact TV distances are concentrated near 1; MST gaps are tiny relative to the distance scale; no useful nontrivial hard partition satisfies the thresholds. See `reports/gates.json` and `reports{8,16,32}/gates.json`.

This is the load-bearing scientific outcome: **one-step next-token TV, as specified in v1.0, does not by itself yield robust hard domains on this pilot.** The goal is retained; the mathematical criterion must be amended (e.g., continuation-level behavior, trace predictive geometry, density-connected soft regions, or persistent multiscale components) before C4 can be built on A1 as a hard partition theorem.

---

## 4. Learned self-knowledge space

### 4.1 Finite spectral contrastive objective (corrected Theorem B)

Let W be a symmetric nonnegative similarity matrix with positive degrees d_i, vol=∑d_i, μ_i=d_i/vol, and A=D^{-1/2}WD^{-1/2}. Let f_i∈R^d and g_i=√μ_i f_i. Define P_pos(i,j)=W_ij/vol. The finite objective is

L_sp(f) = -2 E_{P_pos}[f_i·f_j] + E_{μ×μ}[(f_i·f_j)^2]
        = -2 Tr(G^T A G) + ||G^T G||_F².

**Theorem C3.2 (finite variational characterization). [P]**
Under G^T G=I_d, minimizers span the top-d eigenspace of A, equivalently the bottom-d eigenspace of L_sym=I-A. The minimum value is d - 2∑_{i=1..d}λ_i(A).

*Proof.* Under the constraint the second term is d. Ky Fan's maximum principle gives the first term. ∎

v1.0 said "top-d eigenspace of L_sym"; that is the wrong end of the spectrum. v1.0 also attached a universal downstream linear-probe bound; no such bound follows without task/label compatibility assumptions. It is removed.

**Measure mismatch.** If μ is uniform over samples rather than degree/vol, the normalized-adjacency statement requires reweighting. The pilot explicitly uses degree/vol for the spectral term.

**Sphere mismatch.** Whitening implies E_μ||f||²=d, so a unit-sphere constraint is incompatible for d>1. The pilot uses R^d and whitening; optional cosine normalization is a separate downstream operation.

### 4.2 Disconnected-graph recovery

**Theorem C3.3 (exact population/finite disconnected recovery). [P]**
If W has q connected components and d≥q, the top-q eigenspace of A is spanned by vectors v_C(i)=√d_i/√vol(C) for i∈C. Any whitened minimizer restricted to this subspace is constant on each component: f_i=c_C for i∈C, with ∑_C μ(C)c_C c_C^T=I. Hence components are separated. The margin depends on component masses and the chosen orthonormal basis, not directly on ε_out-ε_in.

The learned MLP's optimization path is **[H]/[E]**, not proved by the spectral theorem.

### 4.3 Pilot positive graph

If A1 supplies a window, use threshold ε_pos=(ε_in+ε_out)/2. If A1 fails, the pilot uses a self-generated symmetrized kNN graph (k=8) on exact TV distances. This fallback is an empirical representation-learning device, not evidence for A1.

### 4.4 Bootstrap distillation

The pilot adds a scale-invariant pairwise distillation term between squared encoder distances and squared ψ_A distances:

L_dist = E_{(i,j)} [ (||f_i-f_j||² / mean||f||²) - (||ψ_i-ψ_j||² / mean||ψ||²) ]².

This anchors relative geometry while respecting whitening scale. It does not create a bi-Lipschitz certificate for f.

### 4.5 Energy geometry

The pilot energy is conservative:

E_φ(z)=½||z||² + a^T tanh(Wz+b),
∇E_φ(z)=z + W^T(a ⊙ (1-tanh²(Wz+b))).

It is trained by denoising score matching on whitened embeddings:

L_DSM = E_{z,ε} ||∇E_φ(z+σε) - ε/σ||².

**[P] classical limit:** the optimum recovers the score of the Gaussian-smoothed density q_σ; as σ→0, under regularity, it approaches ∇log q. **[E] pilot:** finite σ, finite data, finite capacity. Traversal is gradient descent on E (density ascent), not gradient ascent on E.

---

## 5. Routing and reshape

### 5.1 Corrected routing certificate (replaces Corollary 2)

v1.0's latent-distance routing claim is invalid without a distortion certificate, and p(·|D_i) is undefined.

**Theorem C5.1 (behavioral routing certificate). [P]**
Let D_i have behavioral diameter Δ_i. If there exists x*∈D_i with d_B(x,x*)≤r, then for all y∈D_i,

d_B(x,y) ≤ r + Δ_i.

If only traces are available and d̂_TV(x,x*)+τ_safe(x,x*)≤r, the same conclusion holds with r as a certified upper bound.

A latent-space nearest-region rule is a heuristic unless f has a certified lower distortion bound L_low d_B(x,y)≤||f(x)-f(y)||. The pilot gate checks the behavioral certificate, not latent-only routing.

### 5.2 Reshape under fine-tuning

**Theorem C7.1 (local drift machinery). [P] under a local logit-Lipschitz assumption**
Suppose for all x∈S and θ on the segment [θ,θ+δθ],

||z_{θ+δθ}(x)-z_θ(x)||_∞ ≤ L_z ||δθ||.

Using the safe softmax bound TV(softmax(z),softmax(z'))≤||z-z'||_∞,

|d_B^{θ+δθ}(x,y)-d_B^θ(x,y)| ≤ 2 L_z ||δθ|| =: Δ.

If A1 holds with ε_out-ε_in>2Δ, then the same partition is stable for any ε∈[ε_in+Δ, ε_out-Δ).

**Limitations.** L_z for a full transformer is not merely a product of weight spectral norms; residual attention, RMSNorm, and softmax require explicit local Jacobian/interval bounds. The pilot does not claim a production fine-tune stability constant. Davis–Kahan eigen-embedding drift requires an operator-norm perturbation bound and an eigengap; it does not follow from TV margin alone.

---

## 6. Complexity and memory (corrected)

For the pinned architecture (d=576, L=30, H=9, KVH=3, FF=1536, V=49152):

- Per-token non-embedding MACs:
  L(2d² + 2d·d_kv + 3d·FF) + Vd
  = 30(663552 + 221184 + 2654208) + 28311552
  = 134,479,872 MACs/token.
- Causal attention adds Θ(L d n²) MACs/context.
- One context of n=64: ≈8.61 GMACs ≈17.2 GFLOPs (multiply-add counted as 2 FLOPs).
- N=10^5 contexts: ≈1.72×10^15 FLOPs, not 10^18.
- Full-V TV: Θ(V) per pair; exact N×N pilot matrix is Θ(N²V).
- Landmark embedding: Θ(MV) per context.
- Trace memory: L·2d=34560 floats. fp16 ≈69.1 KB/context; int8 ≈34.6 KB/context plus scales/metadata.
- 10^6 int8 traces ≈35 GB plus 256-d fp32 embeddings ≈1 GB. Compression is workload-dependent.
- Serving overhead is **[E]**, not the asserted 3–8% from v1.0.

---

## 7. Final pilot gate protocol

| Gate | Criterion | Status source |
|---|---|---|
| G1 C1 parity | max KL(HF||C++)≤1e-5, mean KL≤1e-6, top-1 agreement=1, perplexity relative diff≤1e-3 | `reports/parity.json` |
| G2 trace certificate | 0 violations of |TV-d̂TV|≤ρ_i+ρ_j on 2000 sampled pairs | `reports/gates.json` |
| G3 A1 | useful hard partition thresholds + bootstrap ARI≥0.8 | `reports/gates.json` |
| G4 bootstrap distortion | 0 violations of ψ upper/lower bounds on holdout pairs; learned Spearman reported as [E] | `reports/gates.json` |
| G5 routing certificate | 0 violations of d(x,y)≤r+diam(D) on natural + synthetic shifted contexts | `reports/gates.json` |
| G6 boundary gap | positive median natural nearest-region margin; energy AUROC(synthetic vs natural)≥0.6 | `reports/gates.json` |
| G7 serialization | self-weight save/load encode L2 diff≤1e-5, energy diff≤1e-4 | `reports/gates.json` |

**Interpretation rule:** G1, G2, G4–G7 are engineering/certificate gates. G3 is the scientific go/no-go gate for hard A1 domains. A failure of G3 is not an implementation failure; it falsifies the load-bearing assumption for the specified behavioral metric and pilot distribution.

---

## 8. Required amendments before C4

1. Do not build C4 as a theorem-dependent hard hierarchy on A1 until A1 is replaced or validated on live traffic at production scale.
2. Candidate replacements to investigate, each with its own definitions and gates:
   - continuation-distribution TV / self-agreement over rollouts;
   - trace predictive geometry (hidden-state transitions, not only final next-token distribution);
   - density-connected soft regions in Z with persistence thresholds;
   - local competence charts with overlapping neighborhoods instead of a global partition.
3. Keep the product goal unchanged: emergent sub-symbolic self-knowledge, inspectable/traversable/editable, no handwritten domain taxonomy.
4. Promote the pilot C++ core to the standardized Alysane architecture only after C1 architectural-swap parity and production-scale performance gates are specified and passed.
