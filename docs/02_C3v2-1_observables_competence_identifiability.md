# Alysane — C3v2-1: Observables, Competence, and Identifiability

**Status legend:** `[D]` definition, `[A]` assumption, `[P]` proven result, `[H]` hypothesis, `[E]` experimental/executed check, `[I]` identifiability failure under stated assumptions.

**Scope boundary:** This document completes C3v2-1 only. It does not construct the self-knowledge geometry, local regions, competence estimators, transfer scores, or C4 hierarchy. It establishes the observable/evidence contract that those later components must obey.

---

## 1. Formal setup

### 1.1 Probability space and task instances

Let `(Ω, F, P)` be a probability space. A **task instance** is a random element

```text
T : Ω → 𝒯
```

where `𝒯` is a measurable space of evaluated tasks. A task instance includes everything needed to judge success: prompt, hidden answer key, environment state, evaluator configuration, or executable tests. The model may observe only a projection of `T`, not all of `T`.

A **knowledge system** `K` produces an answer

```text
A = K(T, R)
```

where `R` represents internal randomness: sampling, dropout, nondeterministic execution, or other internal stochasticity.

### 1.2 Internal observables

**Definition 1.1 (Internal observation). [D]**
An internal observation is a measurable map

```text
O = Φ_K(T, R)
```

taking values in a measurable space `𝒪`. It may include:

- input tokens or prompt projection;
- hidden states;
- attention summaries;
- logits;
- next-token or answer distribution;
- predictive entropy;
- sampled answer;
- self-consistency statistics over repeated internal samples;
- outputs of an internal verifier, if that verifier is actually executed.

The internal σ-algebra is

```text
ℐ = σ(O).
```

An **internal-only self-knowledge estimator** is any `ℐ`-measurable random variable

```text
ĉ_int = f(O).
```

It may use the self-knowledge weights `θ_S`, but it may not use external outcome evidence unless that evidence is included in `O`.

### 1.3 Success and actual competence

Let `W` denote the evaluation world or evaluator state. A success predicate is

```text
S : 𝒯 ×  ×  → {0, 1}.
```

The realized success variable is

```text
Y = S(T, A, W).
```

**Definition 1.2 (Pointwise actual competence). [D]**
For a fixed task instance `t` and world `w`, actual competence is

```text
c_K(t, w) = P(Y = 1 | T = t, W = w).
```

If the system and evaluator are deterministic, `c_K(t, w) ∈ {0, 1}`. If either is stochastic, `c_K(t, w) ∈ [0, 1]`.

**Definition 1.3 (Population competence). [D]**
For a target task distribution `μ` and world distribution `ν`,

```text
C_K(μ, ν) = E_{T∼μ, W∼ν}[ c_K(T, W) ].
```

For binary success, this is the expected success rate. For a loss `ℓ ∈ [0, 1]`, the analogous quantity is expected loss or risk.

**Definition 1.4 (Correctness vs competence). [D]**
Correctness is the realized event `Y = 1` on a particular execution. Competence is a probability or expectation over the relevant randomness. They are not the same object.

---

## 2. Identifiability from internal observables

### 2.1 Identifiability definition

**Definition 2.1 (Internal identifiability). [D]**
Let `𝒞` be a class of admissible configurations `(K, S, μ, ν)`. Actual competence is **identifiable from internal observables over `𝒞`** if whenever two configurations induce the same law of `O`, they also induce the same competence target.

For pointwise competence, the requirement is:

```text
Law(O | t, w) = Law(O | t, w')  ⇒  c_K(t, w) = c_K(t, w').
```

For population competence:

```text
Law(O) = Law(O')  ⇒  C_K = C_K'.
```

If this fails, no internal-only estimator can uniformly recover actual competence over the class.

### 2.2 Observational-equivalence barrier

**Theorem 2.1 (Two-world barrier). [P]**
Suppose there exist two admissible worlds `w_0, w_1` and a system `K` such that for some task `t`:

```text
O(t, w_0) = O(t, w_1)
```

or, more generally, the internal observation laws are identical, but

```text
c_K(t, w_0) = 1,
c_K(t, w_1) = 0.
```

Then no internal-only estimator can be correct in both worlds. For any deterministic internal-only estimate `v = f(O)`,

```text
max(|v - 1|, |v - 0|) ≥ 1/2.
```

For squared error,

```text
max((v - 1)^2, v^2) ≥ 1/4.
```

The same lower bound holds for randomized internal-only estimators in worst-case expected error.

**Proof.**
Because the observation is identical, every internal-only estimator produces the same value or same distribution in both worlds. Let the estimate be `v ∈ [0, 1]`. If `v ≤ 1/2`, then `|v - 1| ≥ 1/2`. If `v ≥ 1/2`, then `|v - 0| ≥ 1/2`. Therefore the worst-case absolute error is at least `1/2`. The squared-error bound follows similarly, with minimum at `v = 1/2`.

For a randomized estimator `Z` whose law depends only on `O`, the law of `Z` is the same in both worlds. By the triangle inequality,

```text
E|Z - 0| + E|Z - 1| ≥ E|1 - 0| = 1.
```

Hence at least one world has expected absolute error at least `1/2`. The squared-error case follows from

```text
E[Z^2] + E[(Z - 1)^2] ≥ 1/2,
```

so at least one world has expected squared error at least `1/4`. ∎

**Corollary 2.2 (No asymptotic internal-only identification). [P]**
If two worlds induce identical laws for all internal observations, no sequence of internal-only estimators can converge in probability to two different competence values. An estimator sequence has the same distribution under both worlds, so it cannot converge to `0` under one and `1` under the other. ∎

### 2.3 Concrete counterexample: sealed evaluator key

Let the task be:

```text
t = "What is the sealed code?"
```

The model deterministically produces:

```text
answer = 42
entropy = 0
trace_hash = fixed
```

The internal observation is identical in two worlds:

```text
w_0: evaluator accepts 42
w_1: evaluator accepts 43
```

Then:

```text
c_K(t, w_0) = 1
c_K(t, w_1) = 0
```

but `O` is identical. Therefore actual competence is not identifiable from internal observables alone.

This counterexample is intentionally strong: the model is deterministic, confident, and has a fixed internal trace. The failure is not caused by noisy hidden states or insufficient trace detail. It is caused by the success predicate depending on external evaluator state not determined by the internal observation.

**Executed check. [E]**
The C++ demonstration `build/c3v2_1_demo` constructs this exact counterexample and verifies:

- identical observations;
- competence `1` in one world and `0` in the other;
- internal-only minimax absolute error lower bound `0.5`;
- internal-only minimax squared error lower bound `0.25`;
- outcome evidence separates the worlds.

### 2.4 Two-system variant

The barrier is not limited to different evaluators. Suppose two systems `K_0` and `K_1` produce identical internal observations on the observed task distribution, but their answers differ on tasks outside that distribution. If the internal observation does not include the full answer or full behavioral output, then competence on the larger task space can differ while internal observations remain identical.

Even if the internal observation includes the full answer on the observed tasks, competence on unobserved tasks remains unidentifiable without coverage or structural assumptions. This is the no-extrapolation barrier in Section 4.

---

## 3. When internal-only competence is possible

The barrier is conditional, not absolute. Internal-only competence estimation is possible when success is determined by internal observables.

**Theorem 3.1 (Internal verifier sufficiency). [P]**
Suppose there exists a known measurable function

```text
g : 𝒪 → {0, 1}
```

such that

```text
Y = g(O)  almost surely.
```

Then realized success is internally identifiable. If `O` is deterministic given `t`, actual competence is

```text
c_K(t) = g(O(t)).
```

If `O` is stochastic given `t`, then

```text
c_K(t) = E[g(O) | T = t],
```

which can be estimated by internal sampling if the conditional law of `O` is accessible.

**Proof.**
If `Y = g(O)` almost surely, then `Y` is `σ(O)`-measurable. The estimator `ĉ = g(O)` is internal-only and equals realized success. Taking conditional expectations gives competence. ∎

### 3.1 Example: executable internal verification

For an arithmetic task, suppose the internal observation includes:

```text
a = 6
b = 7
answer = 42
```

and the known success predicate is:

```text
g(O) = 1[answer = a * b].
```

Then competence is internally computable. The C++ demonstration verifies:

```text
arithmetic_competence_internal({6, 7, 42}) = 1
arithmetic_competence_internal({6, 7, 43}) = 0
```

### 3.2 Important limitation: verifier soundness

An internal verifier identifies **verifier-success**, not necessarily **actual task-success**, unless the verifier is sound for the target success predicate.

If the target success predicate is `S` and the internal verifier is `V_int`, internal-only estimation identifies:

```text
P(V_int = 1)
```

not necessarily:

```text
P(S = 1).
```

To equate them, one needs evidence or a proof that:

```text
V_int = S
```

on the target task class. That evidence is additional evidence, even if it is generated internally by a trusted proof checker or executable test suite.

---

## 4. Minimum outcome-evidence contract

### 4.1 What outcome evidence is

**Definition 4.1 (Outcome evidence). [D]**
Outcome evidence is a record

```text
E_i = (task_id, answer_id, evaluator_id, y_i, metadata)
```

where

```text
y_i = S(t_i, a_i, w_i) ∈ {0, 1}
```

is a verified success bit, or more generally a verified loss `ℓ_i ∈ [0, 1]`.

Outcome evidence is not a domain label. It does not say “math,” “code,” or “medicine.” It says whether a particular answer succeeded under a particular evaluator.

### 4.2 Pointwise deterministic competence

**Proposition 4.1 (One-bit pointwise evidence). [P]**
If the system and evaluator are deterministic for a fixed task instance, then one verified success bit for that exact task, answer, and evaluator identifies pointwise competence:

```text
c_K(t, w) = y ∈ {0, 1}.
```

**Proof.**
Determinism implies competence is either `0` or `1`. The verified outcome reveals which. ∎

If the task, answer, policy, or evaluator changes, the old bit no longer identifies the new competence.

### 4.3 Stochastic pointwise competence

If success is stochastic, one bit is insufficient to identify the exact probability. For `n` i.i.d. verified outcomes with mean `c`, the empirical mean

```text
ĉ_n = (1/n) ∑ y_i
```

satisfies Hoeffding's inequality:

```text
P(|ĉ_n - c| ≥ ε) ≤ 2 exp(-2nε²).
```

Thus a sufficient sample size for absolute error `ε` and failure probability `δ` is

```text
n ≥ ceil( log(2/δ) / (2ε²) ).
```

For `ε = 0.05` and `δ = 0.05`:

```text
n = 738.
```

The C++ demonstration verifies this value and the corresponding half-width:

```text
hoeffding_width(738, 0.05) ≈ 0.0499924 ≤ 0.05.
```

This is a distribution-free sufficient bound for binary outcomes, not a claim that 738 samples are always necessary.

### 4.4 Population competence

For population competence under a target task distribution `μ`, the minimum practical contract is:

1. Sample tasks from the target distribution, or use known importance weights.
2. Execute the current knowledge system and policy.
3. Obtain verified outcomes from a trusted evaluator.
4. Estimate the mean outcome.
5. Report uncertainty using a valid confidence bound.

For i.i.d. binary outcomes, the same Hoeffding bound applies:

```text
P(|Ĉ_n - C| ≥ ε) ≤ 2 exp(-2nε²).
```

No human-assigned domain labels are required.

### 4.5 No extrapolation without coverage or assumptions

**Theorem 4.2 (Finite-evidence no-extrapolation barrier). [P]**
Let outcome evidence be available only on a finite set of task instances `T_obs`. Let `t*` be a task instance outside `T_obs`. Suppose internal observations are identical across two worlds on all observed tasks, and all observed outcomes agree. Then there exist two admissible worlds that agree on all evidence but differ in competence at `t*`.

Therefore, pointwise competence at `t*` is not identifiable from the finite evidence alone.

**Proof.**
Construct world `w_0` where the evaluator accepts the system's answer at `t*`, and world `w_1` where it rejects that answer. Leave all observed tasks and internal observations unchanged. The evidence transcript is identical, but competence at `t*` differs. ∎

**Corollary 4.3 (Population limitation). [P]**
If the unobserved task set has positive target probability mass, population competence can differ between the two worlds by up to that mass. Finite evidence cannot identify population competence exactly without coverage, smoothness, parametric structure, or other assumptions. ∎

### 4.6 Minimum-evidence contract for C3 v2

A competence claim in Alysane must be backed by one of the following evidence contracts.

#### Contract A: Internal-verifier competence

Allowed when:

```text
Y = g(O)
```

for a known, trusted, sound verifier `g`.

Claims:

- internally computable verifier-success;
- actual competence only if verifier soundness is established.

Evidence required:

- executable verifier or proof of soundness;
- versioned verifier identity;
- task/answer binding.

#### Contract B: Pointwise verified competence

Allowed when:

- task instance is fixed;
- answer/policy is fixed;
- evaluator is fixed;
- at least one verified outcome is available.

Claims:

- deterministic pointwise competence exactly;
- stochastic pointwise competence only with repeated outcomes and a confidence bound.

Evidence required:

```text
(task_id, answer_id, evaluator_id, success_bit, policy_version, timestamp)
```

#### Contract C: Population verified competence

Allowed when:

- outcomes are sampled from the target distribution or reweighted to it;
- evaluator is trusted;
- sample size and confidence level are declared.

Claims:

```text
|Ĉ - C| ≤ ε
```

with confidence at least `1 - δ`.

Evidence required:

- outcome sample;
- sampling protocol;
- evaluator identity;
- confidence calculation.

#### Contract D: Transfer or generalization competence

Allowed only with additional assumptions or outcome coverage on the target region.

Claims must be labeled as:

- verified on covered tasks;
- extrapolated under stated assumptions;
- unverified outside coverage.

No internal-only geometry can upgrade an unverified extrapolation into verified competence.

---

## 5. Uncertainty is not competence

### 5.1 Predictive uncertainty

**Definition 5.1 (Predictive uncertainty). [D]**
Predictive uncertainty is a functional of the model's predictive distribution, for example:

```text
H[p_K(· | t)]
```

or margin, variance, top-k mass, or self-consistency. It is internally observable if the predictive distribution or samples are accessible.

Predictive uncertainty measures dispersion of the model's predictions. It does not measure whether the prediction is correct.

### 5.2 Uncertainty-competence separation

**Proposition 5.1 (Same uncertainty, different competence). [P]**
There exist systems and evaluators with identical predictive distributions and identical predictive entropy but different actual competence.

**Proof by counterexample.**
Let the model predict answer `0` with probability `1`. Predictive entropy is `0`.

- If the evaluator accepts answer `0`, competence is `1`.
- If the evaluator rejects answer `0`, competence is `0`.

Now let the model predict answers `0` and `1` with probability `1/2` each. Predictive entropy is `1` bit.

- If the evaluator accepts both answers, competence is `1`.
- If the evaluator accepts only answer `0`, competence is `1/2`.

Thus predictive uncertainty does not determine competence. ∎

**Executed check. [E]**
The C++ demonstration verifies:

```text
certain_entropy_bits = 0
certain_competence_if_correct = 1
certain_competence_if_wrong = 0

uniform_entropy_bits = 1
uniform_competence_all_accepted = 1
uniform_competence_half_accepted = 0.5
```

### 5.3 Epistemic and aleatoric uncertainty

Let `Θ` be a latent competence parameter. For binary success, suppose:

```text
Y | Θ ∼ Bernoulli(Θ).
```

Let predictive competence be:

```text
c = E[Θ].
```

The total outcome variance decomposes as:

```text
Var(Y) = E[Θ(1 - Θ)] + Var(Θ).
```

A common interpretation is:

```text
aleatoric variance = E[Θ(1 - Θ)]
epistemic variance = Var(Θ)
```

but this decomposition depends on the latent-variable model and posterior.

**Proposition 5.2 (Epistemic/aleatoric non-identifiability from predictive competence alone). [P]**
The same predictive competence and total outcome variance can arise from different epistemic/aleatoric decompositions.

**Proof.**
Let `c = 1/2`.

Model A: `Θ = 1/2` almost surely. Then:

```text
epistemic variance = Var(Θ) = 0
aleatoric variance = E[Θ(1 - Θ)] = 1/4
total variance = 1/4
```

Model B: `Θ ∈ {0, 1}` with `P(Θ = 1) = 1/2`. Then:

```text
epistemic variance = Var(Θ) = 1/4
aleatoric variance = E[Θ(1 - Θ)] = 0
total variance = 1/4
```

Both models have the same predictive competence `1/2` and total variance `1/4`, but different epistemic/aleatoric splits. Therefore the split is not identified by predictive competence alone. ∎

**Executed check. [E]**
The C++ demonstration verifies both decompositions and their equality in total variance.

### 5.4 Classification of uncertainty quantities

| Quantity | Internal-only? | Outcome-dependent? | Notes |
|---|---:|---:|---|
| Predictive entropy | Yes | No | Functional of predictive distribution |
| Top-k mass / margin | Yes | No | Predictive dispersion |
| Self-consistency | Yes | No | Requires repeated internal samples |
| Realized correctness | No, unless internally verifiable | Yes | Requires success predicate |
| Actual competence | Not in general | Yes | Requires outcome evidence or sound internal verifier |
| Calibration | No | Yes | Requires outcomes across confidence levels |
| Epistemic variance | Not without assumptions | Usually | Requires posterior/latent model and evidence |
| Aleatoric variance | Not without assumptions | Usually | Requires outcome model and evidence |

---

## 6. Classification of self-knowledge quantities

### 6.1 Internally observable quantities

These can be extracted from the knowledge model's computation, assuming white-box access:

- input tokens;
- hidden states;
- attention summaries;
- logits;
- predictive distributions;
- predictive entropy;
- sampled answers;
- self-consistency across internal samples;
- trace norms, Jacobians, or other computational summaries;
- internal verifier outputs, if the verifier is executed.

These quantities can support geometry, novelty, relevance, and uncertainty proxies. They do not by themselves establish actual competence.

### 6.2 Outcome-dependent quantities

These require verified outcomes or a sound internal verifier:

- realized correctness;
- actual competence;
- expected loss;
- calibration;
- transfer success;
- competence boundaries;
- validated competence estimates.

### 6.3 Not identifiable from internal observables alone under external success

Under the assumption that success depends on external evaluator or environment state not determined by internal observables:

- actual competence in general;
- correctness of arbitrary outputs;
- epistemic/aleatoric decomposition without additional assumptions;
- future generalization to uncovered tasks;
- competence on tasks outside outcome coverage.

### 6.4 Conditionally identifiable quantities

These become identifiable under explicit contracts:

- verifier-success, if the verifier is executed internally;
- actual competence, if the verifier is sound for the target success predicate;
- deterministic pointwise competence, if one verified outcome is available;
- population competence, if outcome samples are drawn from the target distribution;
- local competence, if outcome coverage and regularity assumptions are stated.

---

## 7. Consequences for later C3 v2 components

C3v2-2 through C3v2-9 must obey the following rules.

1. **Trace geometry is internal-only.**
   It may organize internal observables, but it cannot claim actual competence without outcome evidence.

2. **Behavioral distances are not competence distances.**
   One-step TV, continuation distance, or trace distance may be useful relevance signals, but they do not identify success.

3. **Competence estimators require an evidence contract.**
   Every competence estimate must declare whether it is:
   - internal-verifier-based;
   - pointwise verified;
   - population verified;
   - extrapolated under assumptions;
   - unverified.

4. **Uncertainty must not be relabeled as competence.**
   Predictive entropy, epistemic proxies, and aleatoric proxies must remain distinct from verified success probability.

5. **Boundaries require outcome evidence or explicit assumptions.**
   A geometric boundary can mark novelty or low density. A competence boundary requires evidence that success changes, or a stated extrapolation assumption.

6. **No domain labels are required.**
   Outcome evidence is task/answer/evaluator success, not a human-assigned domain taxonomy.

---

## 8. Executed implementation and tests

The workspace contains a minimal C++17 demonstration:

```text
include/c3v2_1.h
src/c3v2_1.cpp
src/c3v2_1_demo.cpp
tests/c3v2_1_test.cpp
Makefile
tools/run_c3v2_1.sh
```

Build and run:

```bash
make c3v2_1
./build/c3v2_1_test
./build/c3v2_1_demo reports/c3v2_1_demo.json
```

Or:

```bash
./tools/run_c3v2_1.sh
```

Verified results:

```text
ALL TESTS PASSED
AddressSanitizer + UndefinedBehaviorSanitizer: ALL TESTS PASSED
```

The demonstration verifies:

- identical internal observations with different competence;
- internal-only minimax absolute error lower bound `0.5`;
- internal-only minimax squared error lower bound `0.25`;
- outcome evidence separates the worlds;
- positive internal-verifier case;
- same predictive uncertainty with different competence;
- same predictive competence with different epistemic/aleatoric decomposition;
- Hoeffding sample size `738` for `ε = δ = 0.05`;
- Hoeffding half-width bound.

---

## 9. Final identifiability statement

**Central result. [P]**
Actual competence is not identifiable from internal observables alone when success depends on evaluator or environment state not determined by those observables. Two worlds can produce identical internal observations, including identical answers, traces, and predictive distributions, while having different actual competence.

**Positive result. [P]**
Actual competence is internally identifiable when success is a known measurable function of internal observables, for example through a sound internal verifier. Otherwise, competence estimation requires outcome evidence.

**Minimum-evidence result. [P]**
For deterministic pointwise competence, one verified success bit for the exact task, answer, and evaluator suffices. For stochastic or population competence, verified outcome samples and a confidence bound are required. For unobserved tasks, no finite evidence identifies competence without coverage or structural assumptions.

**Uncertainty result. [P]**
Predictive uncertainty is not correctness or competence. Epistemic/aleatoric decomposition is not identified by predictive competence alone.

These results define the evidence contract for the rest of C3 v2.
