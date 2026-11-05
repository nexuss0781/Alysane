#include "c3v2_1.h"
#include <cmath>
#include <iostream>

using namespace alysane::c3v2_1;

static int failures = 0;
static void check(bool ok, const char* name) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok)
        ++failures;
}

int main() {
    try {
        run_checks();
        check(true, "run_checks");
    } catch (const std::exception& e) {
        check(false, e.what());
    }

    Task t{1, "sealed", 42};
    World w0{"w0", 42}, w1{"w1", 43};
    auto o = run_model(t);
    check(o == run_model(t), "deterministic observation");
    check(competence(w0, t) != competence(w1, t), "observational equivalence with different competence");

    // Exhaust over a small grid of internal-only estimators to illustrate lower bound.
    double best = 1e9;
    for (int i = 0; i <= 100; ++i) {
        double x = i / 100.0;
        double err = std::max(std::abs(x - competence(w0, t)), std::abs(x - competence(w1, t)));
        best = std::min(best, err);
    }
    check(std::abs(best - 0.5) < 1e-12, "grid minimax internal-only error = 0.5");

    OutcomeEvidence ep{t.id, o.answer, true}, ef{t.id, o.answer, false};
    check(outcome_estimator(ep) == 1.0 && outcome_estimator(ef) == 0.0, "outcome evidence separates worlds");

    check(arithmetic_competence_internal({6, 7, 42}) == 1.0, "internal verifier positive");
    check(arithmetic_competence_internal({6, 7, 43}) == 0.0, "internal verifier negative");

    PredictiveDistribution p{0.5};
    check(std::abs(p.entropy_bits() - 1.0) < 1e-12, "entropy bits");
    check(competence_under_accept_set(p, true, true) == 1.0, "same uncertainty competence 1");
    check(competence_under_accept_set(p, true, false) == 0.5, "same uncertainty competence 0.5");

    auto det = deterministic_latent(0.5);
    auto ber = bernoulli_latent(0.5);
    check(std::abs(det.total_variance() - ber.total_variance()) < 1e-12, "same total variance");
    check(det.epistemic_variance != ber.epistemic_variance, "different epistemic variance");
    check(det.aleatoric_variance != ber.aleatoric_variance, "different aleatoric variance");
    check(std::abs(internal_only_worst_case_squared_lower_bound() - 0.25) < 1e-12, "squared lower bound");

    check(hoeffding_n(0.05, 0.05) == 738, "hoeffding n");
    check(hoeffding_width(738, 0.05) <= 0.05, "hoeffding width");
    bool threw = false;
    try {
        hoeffding_n(0.0, 0.05);
    } catch (...) {
        threw = true;
    }
    check(threw, "hoeffding invalid args");

    std::cout << (failures == 0 ? "ALL TESTS PASSED" : "TEST FAILURES") << "\n";
    return failures == 0 ? 0 : 1;
}
