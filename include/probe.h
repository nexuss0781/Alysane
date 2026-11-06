#pragma once
// alysane-probe — executable A1 measurement protocol (docs/05).
//
// Implements Definition M1, Estimators 1-6, Phases 0-4, the §4 decision rule and
// the §5 pre-committed NO-GO branches. Three mathematical corrections to the
// protocol as supplied are implemented and documented in docs/06:
//
//  (C1) lambda_2 of a graph with k>=2 components is identically 0, so the
//       criterion "k>=2 AND lambda_2 >= lambda_min" is unsatisfiable. The
//       meaningful quantity is lambda_{k+1}: the spectral gap above the
//       k-dimensional kernel, equal to the min within-component Fiedler value.
//  (C2) Power iteration converges to lambda_max, not lambda_2. We use an exact
//       dense symmetric eigensolver (LAPACK) at probe scale, which also certifies
//       the kernel multiplicity.
//  (C3) Lemma M2 does NOT transfer TV-specific certificates to other metrics.
//       Metric-generic results transfer; the trace bound and drift bound are
//       re-derived per metric.
#include "alysane.h"

namespace alysane {
namespace probe {

enum class Metric { OneStepTV, RolloutJsRoot, ContinuationMmd };
const char* metric_name(Metric m);

/// Working thresholds from §4/§3 of the protocol. All parameterisable.
struct Config {
    int M_pairs = 20000;          ///< Estimator 1 sample size
    int landmark_m = 128;         ///< Phase 1 k-center landmarks
    int eps_grid = 64;            ///< scale grid resolution
    int bootstrap_B = 20;         ///< Estimator 5
    float bootstrap_frac = 0.8f;
    float lambda_min = 0.01f;     ///< corrected: applies to lambda_{k+1}
    float beta_max_go = 0.15f;    ///< §4 GO gap-mass ceiling
    float beta_max_partial = 0.40f;
    float delta = 0.05f;          ///< confidence level
    float min_comp_mass = 0.01f;  ///< "large" component = >=1% mass
    int k_max = 16;               ///< non-trivial coarse-graining ceiling
    int eta_cohesion_pct = 95;    ///< Estimator 3 giant-component fraction
    float window_overlap_req = 0.90f;
    float covered_mass_req = 0.80f;
    float routing_rate_target = 0.80f;
    float delta_budget = 0.0f;    ///< Delta_budget, filled from measured drift
    int rollout_R = 4;
    int rollout_T = 8;
    int h1_budget = 4096;         ///< sequential cross-pair budget per pair
    uint64_t seed = 20261106ULL;
};

struct Interval {
    float lo = 0.f, hi = 0.f;
    int k = 0;
    bool valid = false;
    float width() const { return hi - lo; }
};

struct MergeEvent {
    float height;
    int size_a, size_b;
};

struct ScalePoint {
    float eps;
    int k;
    float lambda_gap;   ///< lambda_{k+1}, corrected
    int kernel_mult;    ///< observed multiplicity of eigenvalue 0
    float gap_mass;     ///< beta(eps)
    float corr;         ///< C_hat(eps)
    float corr_ci;      ///< Hoeffding half-width
};

struct HypothesisResult {
    bool pass = false;
    std::string detail;
    json evidence;
};

struct MetricResult {
    Metric metric;
    std::vector<ScalePoint> signature;
    std::vector<MergeEvent> merge_spectrum;
    Interval plateau;
    float eps_in_cohesion = 0.f;
    float margin_usable = 0.f;
    float margin_required = 0.f;
    float tau95 = 0.f;             ///< metric-specific error term (see metric_error)
    bool metric_error_vacuous = false;
    int metric_error_required_R = 0;
    float metric_diameter = 1.f;
    std::string metric_error_derivation;
    float covered_mass = 0.f;
    float window_overlap = 0.f;
    float routing_rate = 0.f;
    float routing_bound_rate = 0.f;
    float gap_mass_at_plateau = 0.f;
    // Phase 1 diagnostics
    std::vector<float> kcenter_radius_curve;
    float covering_number_est = 0.f;
    // Lemma M1 capture audit
    float capture_radius_worst_case = 0.f;
    float capture_radius_realized = 0.f;
    double candidate_fraction = 0.0;
    bool capture_filter_vacuous = false;
    HypothesisResult H1, H2, H3;
    std::string verdict;   ///< GO | PARTIAL_GO | NO_GO
    std::string branch;    ///< B1 | B2 | B3 | ""
    json bootstrap_windows;
};

struct Report {
    std::string version = "alysane-probe/1.0";
    int N = 0, n = 0, V = 0;
    std::string model_sha256;
    Config cfg;
    std::vector<MetricResult> metrics;
    std::string final_verdict;
    std::string final_branch;
    std::string rationale;
    json corrections;
    json timing;
};

/// Root Jensen-Shannon divergence (nats) and its metric square root.
float js_divergence(const float* p, const float* q, int V);
float js_root(const float* p, const float* q, int V);

/// Exact spectrum of the normalized Laplacian of the eps-threshold graph.
/// Returns sorted eigenvalues; kernel_mult counts eigenvalues < 1e-6.
std::vector<float> normalized_laplacian_spectrum(const DistMatrix& D, float eps, int* kernel_mult);

/// Run the full protocol against the pinned model. Writes probe_report.json.
Report run(const std::string& model_dir, const std::string& data_dir, const std::string& out_dir,
           const Config& cfg);

void write_report(const Report& r, const std::string& path);

} // namespace probe
} // namespace alysane
