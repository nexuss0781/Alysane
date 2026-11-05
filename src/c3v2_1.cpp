#include "c3v2_1.h"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace alysane::c3v2_1 {
using json = nlohmann::json;

Observation run_model(const Task& t) {
    // Deterministic model: always answers 42 with a fixed trace and zero entropy.
    Observation o;
    o.task_id = t.id;
    o.answer = 42;
    o.entropy_bits = 0.0;
    o.trace_hash = 0xA17A5E1Full;
    return o;
}

bool success(const World& w, const Task& t, const Observation& o) {
    (void)t;
    return o.answer == w.accepted_key;
}

double competence(const World& w, const Task& t) {
    return success(w, t, run_model(t)) ? 1.0 : 0.0;
}

double outcome_estimator(const OutcomeEvidence& e) {
    return e.success ? 1.0 : 0.0;
}

double internal_only_worst_case_lower_bound() {
    // Targets are 0 and 1 under identical observations. Any estimate x in [0,1]
    // has max(|x-0|, |x-1|) >= 1/2, minimized at x=1/2.
    return 0.5;
}

double internal_only_worst_case_squared_lower_bound() {
    // max(x^2, (1-x)^2) >= 1/4, minimized at x=1/2.
    return 0.25;
}

int hoeffding_n(double eps, double delta) {
    if (eps <= 0 || delta <= 0 || delta >= 1)
        throw std::invalid_argument("eps>0, delta in (0,1)");
    double n = std::log(2.0 / delta) / (2.0 * eps * eps);
    return std::max(1, (int)std::ceil(n));
}

double hoeffding_width(int n, double delta) {
    if (n <= 0 || delta <= 0 || delta >= 1)
        throw std::invalid_argument("n>0, delta in (0,1)");
    return std::sqrt(std::log(2.0 / delta) / (2.0 * n));
}

bool arithmetic_verifier(const ArithmeticObservation& o) {
    return o.answer == o.a * o.b;
}

double arithmetic_competence_internal(const ArithmeticObservation& o) {
    return arithmetic_verifier(o) ? 1.0 : 0.0;
}

double PredictiveDistribution::entropy_bits() const {
    auto h = [](double p) { return p <= 0.0 || p >= 1.0 ? 0.0 : -p * std::log2(p); };
    return h(p0) + h(1.0 - p0);
}

double competence_under_accept_set(const PredictiveDistribution& p, bool accept0, bool accept1) {
    return (accept0 ? p.p0 : 0.0) + (accept1 ? (1.0 - p.p0) : 0.0);
}

UncertaintyDecomposition deterministic_latent(double c) {
    // Theta = c almost surely: no epistemic variance, all outcome variance is aleatoric.
    return {c, 0.0, c * (1.0 - c)};
}

UncertaintyDecomposition bernoulli_latent(double c) {
    // Theta in {0,1} with P(Theta=1)=c: outcome is deterministic given Theta,
    // all variance is epistemic.
    return {c, c * (1.0 - c), 0.0};
}

static void require(bool ok, const std::string& msg) {
    if (!ok)
        throw std::runtime_error("C3v2-1 check failed: " + msg);
}

void run_checks() {
    Task t{7, "What is the sealed code?", 42};
    World w_pass{"accepts_42", 42};
    World w_fail{"accepts_43", 43};

    Observation o1 = run_model(t);
    Observation o2 = run_model(t);
    require(o1 == o2, "identical internal observations");
    require(competence(w_pass, t) == 1.0, "world 0 competence 1");
    require(competence(w_fail, t) == 0.0, "world 1 competence 0");

    // Any internal-only estimator returns the same value in both worlds.
    auto est = [](const Observation&) { return 0.5; };
    double v1 = est(o1), v2 = est(o2);
    require(v1 == v2, "internal estimator cannot distinguish worlds");
    double err = std::max(std::abs(v1 - 1.0), std::abs(v2 - 0.0));
    require(err >= internal_only_worst_case_lower_bound() - 1e-12, "worst-case lower bound");

    // Outcome evidence separates the worlds.
    OutcomeEvidence e_pass{t.id, o1.answer, success(w_pass, t, o1)};
    OutcomeEvidence e_fail{t.id, o2.answer, success(w_fail, t, o2)};
    require(outcome_estimator(e_pass) == 1.0, "outcome evidence pass");
    require(outcome_estimator(e_fail) == 0.0, "outcome evidence fail");

    // Positive internal-verifier case.
    ArithmeticObservation good{6, 7, 42};
    ArithmeticObservation bad{6, 7, 43};
    require(arithmetic_competence_internal(good) == 1.0, "internal verifier good");
    require(arithmetic_competence_internal(bad) == 0.0, "internal verifier bad");

    // Uncertainty is not correctness/competence.
    PredictiveDistribution certain{1.0};
    require(certain.entropy_bits() == 0.0, "certain entropy 0");
    require(competence_under_accept_set(certain, false, true) == 0.0, "certain wrong");
    require(competence_under_accept_set(certain, true, false) == 1.0, "certain correct");

    PredictiveDistribution uniform{0.5};
    require(std::abs(uniform.entropy_bits() - 1.0) < 1e-12, "uniform entropy 1 bit");
    require(competence_under_accept_set(uniform, true, true) == 1.0, "uniform all accepted");
    require(competence_under_accept_set(uniform, true, false) == 0.5, "uniform half accepted");

    // Same predictive competence, different epistemic/aleatoric decomposition.
    auto det = deterministic_latent(0.5);
    auto ber = bernoulli_latent(0.5);
    require(std::abs(det.total_variance() - 0.25) < 1e-12, "deterministic latent total variance");
    require(std::abs(ber.total_variance() - 0.25) < 1e-12, "bernoulli latent total variance");
    require(det.epistemic_variance == 0.0 && ber.epistemic_variance == 0.25, "epistemic split differs");
    require(det.aleatoric_variance == 0.25 && ber.aleatoric_variance == 0.0, "aleatoric split differs");

    // Minimum outcome evidence: Hoeffding.
    require(hoeffding_n(0.05, 0.05) == 738, "Hoeffding n(0.05,0.05)=738");
    require(hoeffding_width(738, 0.05) <= 0.05 + 1e-12, "Hoeffding width bound");
    require(hoeffding_width(100, 0.05) > hoeffding_width(1000, 0.05), "width decreases with n");
}

void write_demo_report(const std::string& path) {
    run_checks();
    Task t{7, "What is the sealed code?", 42};
    World w_pass{"accepts_42", 42};
    World w_fail{"accepts_43", 43};
    Observation o = run_model(t);

    PredictiveDistribution certain{1.0}, uniform{0.5};

    json j = {
        {"component", "C3v2-1"},
        {"scope", "observables, competence, identifiability, minimum outcome evidence"},
        {"central_result",
         "Actual competence is not identifiable from internal observables alone when success depends on an "
         "external evaluator/world state not determined by those observables."},
        {"counterexample",
         {{"task", t.prompt},
          {"observation",
           {{"task_id", o.task_id},
            {"answer", o.answer},
            {"entropy_bits", o.entropy_bits},
            {"trace_hash", o.trace_hash}}},
          {"world_pass", {{"name", w_pass.name}, {"competence", competence(w_pass, t)}}},
          {"world_fail", {{"name", w_fail.name}, {"competence", competence(w_fail, t)}}},
          {"internal_only_worst_case_absolute_error_lower_bound", internal_only_worst_case_lower_bound()},
          {"internal_only_worst_case_squared_error_lower_bound",
           internal_only_worst_case_squared_lower_bound()}}},
        {"positive_internal_case",
         {{"condition", "success predicate is a known measurable function of internal observables"},
          {"example", "arithmetic verifier answer == a*b"},
          {"good", arithmetic_competence_internal({6, 7, 42})},
          {"bad", arithmetic_competence_internal({6, 7, 43})}}},
        {"uncertainty_not_competence",
         {{"certain_entropy_bits", certain.entropy_bits()},
          {"certain_competence_if_wrong", competence_under_accept_set(certain, false, true)},
          {"certain_competence_if_correct", competence_under_accept_set(certain, true, false)},
          {"uniform_entropy_bits", uniform.entropy_bits()},
          {"uniform_competence_all_accepted", competence_under_accept_set(uniform, true, true)},
          {"uniform_competence_half_accepted", competence_under_accept_set(uniform, true, false)}}},
        {"epistemic_aleatoric_nonidentifiability",
         {{"predictive_competence", 0.5},
          {"deterministic_latent",
           {{"epistemic_variance", deterministic_latent(0.5).epistemic_variance},
            {"aleatoric_variance", deterministic_latent(0.5).aleatoric_variance}}},
          {"bernoulli_latent",
           {{"epistemic_variance", bernoulli_latent(0.5).epistemic_variance},
            {"aleatoric_variance", bernoulli_latent(0.5).aleatoric_variance}}},
          {"same_total_variance",
           deterministic_latent(0.5).total_variance() == bernoulli_latent(0.5).total_variance()}}},
        {"minimum_outcome_evidence",
         {{"deterministic_pointwise", "one verified success bit for the exact task/answer/evaluator"},
          {"population_hoeffding_n_eps0.05_delta0.05", hoeffding_n(0.05, 0.05)},
          {"population_hoeffding_width_n738_delta0.05", hoeffding_width(738, 0.05)},
          {"no_domain_labels_required", true}}},
        {"evidence_contract",
         {{"A_internal_verifier",
           {{"condition", "Y = g(O) for known trusted sound verifier g"},
            {"claims", "verifier-success; actual competence only if soundness established"},
            {"required",
             {"executable verifier or soundness proof", "verifier version", "task/answer binding"}}}},
          {"B_pointwise_verified",
           {{"condition", "fixed task, answer/policy, evaluator, and verified outcome"},
            {"claims", "deterministic pointwise competence exactly; stochastic only with repeated outcomes "
                       "and confidence bound"},
            {"required",
             {"task_id", "answer_id", "evaluator_id", "success_bit", "policy_version", "timestamp"}}}},
          {"C_population_verified",
           {{"condition", "outcomes sampled from target distribution or reweighted; trusted evaluator; "
                          "declared n, eps, delta"},
            {"claims", "|C_hat - C| <= eps with confidence >= 1-delta"},
            {"required",
             {"outcome sample", "sampling protocol", "evaluator identity", "confidence calculation"}}}},
          {"D_transfer_generalization",
           {{"condition", "additional assumptions or outcome coverage on target region"},
            {"claims", "must be labeled verified-on-covered, extrapolated-under-assumptions, or unverified"},
            {"required", {"coverage map or explicit regularity/transfer assumptions"}}}}}},
        {"classification",
         {{"internally_observable",
           {"input tokens", "hidden states", "attention", "logits", "predictive distribution",
            "predictive entropy", "sampled answer", "self-consistency",
            "internal verifier output if executed"}},
          {"outcome_dependent",
           {"correctness", "actual competence", "calibration", "expected loss", "transfer success"}},
          {"not_identifiable_internal_only_under_external_success",
           {"actual competence in general", "epistemic/aleatoric decomposition without assumptions",
            "future generalization to uncovered tasks"}}}},
        {"checks_passed", true}};
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("cannot write " + path);
    f << j.dump(2);
}

} // namespace alysane::c3v2_1
