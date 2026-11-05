#pragma once
// C3v2-1: Observables, Competence, and Identifiability.
// Minimal, dependency-free demonstration of the central identifiability result.
// This is NOT the self-knowledge geometry (C3v2-2+) and does not train a model.
#include <cstdint>
#include <string>
#include <vector>

namespace alysane::c3v2_1 {

// A task instance. The sealed key is NOT part of the model's internal observation.
struct Task {
    int id = 0;
    std::string prompt;
    int sealed_key = 0;
};

// Internal observation produced by the knowledge model's computation.
// In the counterexample this is identical across worlds.
struct Observation {
    int task_id = 0;
    int answer = 0;
    double entropy_bits = 0.0;
    uint64_t trace_hash = 0;
    bool operator==(const Observation& o) const {
        return task_id == o.task_id && answer == o.answer && entropy_bits == o.entropy_bits &&
               trace_hash == o.trace_hash;
    }
};

// An evaluator/world. Success depends on a sealed key external to internal observables.
struct World {
    std::string name;
    int accepted_key = 0;
};

// Outcome evidence: one verified success bit. No domain label is present.
struct OutcomeEvidence {
    int task_id = 0;
    int answer = 0;
    bool success = false;
};

// Deterministic knowledge model: same computation in every world.
Observation run_model(const Task& t);

// Actual success in a world.
bool success(const World& w, const Task& t, const Observation& o);

// Actual competence for a deterministic model/evaluator on one task (0 or 1).
double competence(const World& w, const Task& t);

// An internal-only estimator sees only Observation.
using InternalEstimator = double (*)(const Observation&);

// Outcome-aware estimator sees verified evidence.
double outcome_estimator(const OutcomeEvidence& e);

// Lower bound on worst-case absolute error of any internal-only estimator
// when two worlds share the same observation but have competences 0 and 1.
double internal_only_worst_case_lower_bound();
// Lower bound on worst-case squared error in the same two-world problem.
double internal_only_worst_case_squared_lower_bound();

// Hoeffding sample size for binary outcome mean estimation.
// Returns smallest integer n such that P(|mean - C| >= eps) <= delta.
int hoeffding_n(double eps, double delta);

// Hoeffding half-width for n verified binary outcomes.
double hoeffding_width(int n, double delta);

// Positive internal-verifier case: success is a known function of observables.
struct ArithmeticObservation {
    int a = 0, b = 0, answer = 0;
};
bool arithmetic_verifier(const ArithmeticObservation& o);
double arithmetic_competence_internal(const ArithmeticObservation& o);

// Uncertainty/competence counterexample: same predictive distribution, different competence.
struct PredictiveDistribution {
    double p0 = 0.5; // probability of answer 0; answer 1 has probability 1-p0
    double entropy_bits() const;
};
double competence_under_accept_set(const PredictiveDistribution& p, bool accept0, bool accept1);

// Epistemic/aleatoric decomposition is not identified by predictive competence alone.
// For binary success with latent competence Theta, total variance is
// Var(Y) = E[Theta(1-Theta)] + Var(Theta).
struct UncertaintyDecomposition {
    double predictive_competence = 0.0;
    double epistemic_variance = 0.0;
    double aleatoric_variance = 0.0;
    double total_variance() const { return epistemic_variance + aleatoric_variance; }
};
UncertaintyDecomposition deterministic_latent(double c);
UncertaintyDecomposition bernoulli_latent(double c);

// Run all mathematical checks. Throws std::runtime_error on failure.
void run_checks();

// Write a JSON demonstration report.
void write_demo_report(const std::string& path);

} // namespace alysane::c3v2_1
