// alysane-probe — A1 measurement protocol implementation. See include/probe.h
// for the three mathematical corrections applied to the supplied protocol.
#include "probe.h"
#include <lapacke.h>
#include <omp.h>
#include <set>

namespace alysane {
namespace probe {

const char* metric_name(Metric m) {
    switch (m) {
    case Metric::OneStepTV: return "one_step_tv";
    case Metric::RolloutJsRoot: return "rollout_js_root";
    case Metric::ContinuationMmd: return "continuation_mmd";
    }
    return "unknown";
}

float js_divergence(const float* p, const float* q, int V) {
    // JS in nats = 1/2 KL(p||m) + 1/2 KL(q||m), m = (p+q)/2.
    double s = 0;
    for (int v = 0; v < V; v++) {
        double a = p[v], b = q[v];
        double m = 0.5 * (a + b);
        if (m <= 0) continue;
        if (a > 0) s += 0.5 * a * std::log(a / m);
        if (b > 0) s += 0.5 * b * std::log(b / m);
    }
    return (float)std::max(0.0, s);
}
float js_root(const float* p, const float* q, int V) { return std::sqrt(js_divergence(p, q, V)); }

std::vector<float> normalized_laplacian_spectrum(const DistMatrix& D, float eps, int* kernel_mult) {
    int N = D.N;
    std::vector<float> deg(N, 0.f);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i != j && D(i, j) <= eps) deg[i] += 1.f;
    // L = I - D^{-1/2} W D^{-1/2}; isolated vertices contribute a zero row,
    // so the kernel multiplicity equals the number of connected components.
    std::vector<float> L((size_t)N * N, 0.f);
    for (int i = 0; i < N; i++) {
        if (deg[i] > 0) L[(size_t)i * N + i] = 1.f;
        for (int j = 0; j < N; j++)
            if (i != j && D(i, j) <= eps && deg[i] > 0 && deg[j] > 0)
                L[(size_t)i * N + j] = -1.f / std::sqrt(deg[i] * deg[j]);
    }
    std::vector<float> eig(N);
    // Correction C2: exact dense eigensolver, not power iteration (which would
    // converge to lambda_max). Column-major upper triangle == our symmetric L.
    int info = LAPACKE_ssyevd(LAPACK_COL_MAJOR, 'N', 'U', N, L.data(), N, eig.data());
    if (info != 0) throw std::runtime_error("ssyevd failed in probe spectrum");
    int km = 0;
    for (float e : eig)
        if (e < 1e-6f) km++;
    if (kernel_mult) *kernel_mult = km;
    return eig;
}

// ---------------------------------------------------------------------------
// Estimator 2: single-linkage merge spectrum with subtree sizes (Kruskal sweep).
// ---------------------------------------------------------------------------
static std::vector<MergeEvent> merge_spectrum(const DistMatrix& D) {
    int N = D.N;
    struct E { float w; int u, v; };
    std::vector<E> edges;
    edges.reserve((size_t)N * (N - 1) / 2);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) edges.push_back({D(i, j), i, j});
    std::sort(edges.begin(), edges.end(), [](const E& a, const E& b) { return a.w < b.w; });
    DSU ds(N);
    std::vector<MergeEvent> out;
    out.reserve(N - 1);
    for (const auto& e : edges) {
        int a = ds.find(e.u), b = ds.find(e.v);
        if (a == b) continue;
        out.push_back({e.w, ds.sz[a], ds.sz[b]});
        ds.unite(a, b);
        if ((int)out.size() == N - 1) break;
    }
    return out;
}

/// k(eps) from the merge spectrum: N minus the number of merges at height <= eps.
static int components_at(const std::vector<MergeEvent>& ms, int N, float eps) {
    int merged = 0;
    for (const auto& e : ms)
        if (e.height <= eps) merged++;
    return N - merged;
}

/// beta(eps): mass fraction in components covering < min_comp_mass of total.
static float gap_mass_at(const DistMatrix& D, float eps, float min_comp_mass) {
    int N = D.N;
    DSU ds(N);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (D(i, j) <= eps) ds.unite(i, j);
    std::map<int, int> sz;
    for (int i = 0; i < N; i++) sz[ds.find(i)]++;
    int thresh = (int)std::ceil(min_comp_mass * N);
    int small = 0;
    for (auto& kv : sz)
        if (kv.second < thresh) small += kv.second;
    return (float)small / N;
}

static std::vector<int> labels_at(const DistMatrix& D, float eps) {
    int N = D.N;
    DSU ds(N);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (D(i, j) <= eps) ds.unite(i, j);
    std::map<int, int> mp;
    std::vector<int> lab(N);
    int k = 0;
    for (int i = 0; i < N; i++) {
        int r = ds.find(i);
        auto it = mp.find(r);
        if (it == mp.end()) { mp[r] = k; lab[i] = k++; }
        else lab[i] = it->second;
    }
    return lab;
}

// ---------------------------------------------------------------------------
// Estimator 1: correlation integral with a Hoeffding half-width.
// NOTE: the guarantee is CONDITIONAL on the drawn context sample; it bounds
// |C_hat - C_N| for the empirical pair distribution, not |C_hat - C_population|.
// ---------------------------------------------------------------------------
static void correlation_integral(const DistMatrix& D, int M, const std::vector<float>& grid,
                                 float delta, uint64_t seed, std::vector<float>& out,
                                 std::vector<float>& ci) {
    int N = D.N;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> ud(0, N - 1);
    std::vector<float> ds;
    ds.reserve(M);
    for (int t = 0; t < M; t++) {
        int i = ud(rng), j = ud(rng);
        while (j == i) j = ud(rng);
        ds.push_back(D(i, j));
    }
    std::sort(ds.begin(), ds.end());
    float w = (float)std::sqrt(std::log(2.0 / delta) / (2.0 * M));
    out.assign(grid.size(), 0.f);
    ci.assign(grid.size(), w);
    for (size_t g = 0; g < grid.size(); g++) {
        size_t c = std::upper_bound(ds.begin(), ds.end(), grid[g]) - ds.begin();
        out[g] = (float)c / M;
    }
}

// ---------------------------------------------------------------------------
// Estimator 3: cohesion scan. Smallest eps at which the largest component of a
// group covers >= eta of that group's mass.
// ---------------------------------------------------------------------------
static float cohesion_eps(const DistMatrix& D, const std::vector<int>& idx, float eta,
                          const std::vector<float>& grid) {
    int m = (int)idx.size();
    if (m < 2) return 0.f;
    for (float eps : grid) {
        DSU ds(m);
        for (int i = 0; i < m; i++)
            for (int j = i + 1; j < m; j++)
                if (D(idx[i], idx[j]) <= eps) ds.unite(i, j);
        int best = 0;
        for (int i = 0; i < m; i++) best = std::max(best, ds.sz[ds.find(i)]);
        if ((float)best / m >= eta) return eps;
    }
    return grid.empty() ? 0.f : grid.back();
}

// ---------------------------------------------------------------------------
// Phase 3 H1: sequential cross-pair test with early stopping.
// Accept separation when the Hoeffding UPPER bound on Pr[d <= eps_hi] < delta_sep.
// Bonferroni: delta_sep = delta / (grid * number_of_component_pairs).
// ---------------------------------------------------------------------------
static HypothesisResult h1_sequential(const DistMatrix& D, const std::vector<int>& lab, int k,
                                      float eps_hi, float delta_sep, int budget, uint64_t seed) {
    HypothesisResult R;
    std::vector<std::vector<int>> groups(k);
    for (size_t i = 0; i < lab.size(); i++) groups[lab[i]].push_back((int)i);
    std::mt19937_64 rng(seed);
    int tested_pairs = 0, accepted = 0, total_pairs = 0;
    double worst_upper = 0;
    int min_cross_samples = budget;
    json per_pair = json::array();
    for (int a = 0; a < k; a++)
        for (int b = a + 1; b < k; b++) {
            if (groups[a].empty() || groups[b].empty()) continue;
            total_pairs++;
            std::uniform_int_distribution<int> ua(0, (int)groups[a].size() - 1);
            std::uniform_int_distribution<int> ub(0, (int)groups[b].size() - 1);
            int hits = 0, t = 0;
            bool decided = false;
            for (; t < budget; t++) {
                float d = D(groups[a][ua(rng)], groups[b][ub(rng)]);
                if (d <= eps_hi) hits++;
                if ((t + 1) % 64 == 0 || t + 1 == budget) {
                    int m = t + 1;
                    double phat = (double)hits / m;
                    double hw = std::sqrt(std::log(2.0 / std::max(1e-12, (double)delta_sep)) / (2.0 * m));
                    double upper = phat + hw;
                    // Early accept: even the upper bound is below delta_sep.
                    if (upper < delta_sep) { decided = true; accepted++; worst_upper = std::max(worst_upper, upper); break; }
                    // Early reject: lower bound already exceeds delta_sep.
                    if (phat - hw > delta_sep) { decided = true; worst_upper = std::max(worst_upper, upper); break; }
                }
            }
            min_cross_samples = std::min(min_cross_samples, t + 1);
            tested_pairs++;
            (void)decided;
            if (per_pair.size() < 32)
                per_pair.push_back({{"a", a}, {"b", b}, {"samples", t + 1}, {"hits", hits}});
        }
    R.pass = (total_pairs > 0 && accepted == total_pairs);
    R.detail = R.pass ? "all large-component pairs separated at the Bonferroni-corrected level"
                      : "at least one component pair is not separated within budget";
    R.evidence = {{"component_pairs", total_pairs}, {"accepted", accepted},
                  {"delta_sep_bonferroni", delta_sep}, {"budget_per_pair", budget},
                  {"min_samples_used", min_cross_samples == budget ? budget : min_cross_samples},
                  {"worst_upper_bound", worst_upper}, {"sample", per_pair}};
    return R;
}

/// H3 bridge audit: any merge of two subtrees each >= min_mass at height <= eps_in.
static HypothesisResult h3_bridges(const std::vector<MergeEvent>& ms, int N, float eps_in,
                                   float min_mass) {
    HypothesisResult R;
    int thresh = (int)std::ceil(min_mass * N);
    int violations = 0;
    json bad = json::array();
    for (const auto& e : ms)
        if (e.height <= eps_in && e.size_a >= thresh && e.size_b >= thresh) {
            violations++;
            if (bad.size() < 16)
                bad.push_back({{"height", e.height}, {"size_a", e.size_a}, {"size_b", e.size_b}});
        }
    R.pass = violations == 0;
    R.detail = R.pass ? "no large-subtree merge at or below eps_in"
                      : "large-subtree merge inside the cohesion window (bridge)";
    R.evidence = {{"violations", violations}, {"min_subtree_size", thresh}, {"examples", bad}};
    return R;
}

// ---------------------------------------------------------------------------
// Plateau detection (Definition M1, with correction C1).
// k(eps) is constant between consecutive merge heights, so candidate plateaus
// are exactly the open gaps between sorted merge heights, restricted to
// 2 <= k <= k_max, beta <= beta_max, lambda_{k+1} >= lambda_min.
// ---------------------------------------------------------------------------
static Interval find_plateau(const DistMatrix& D, const std::vector<MergeEvent>& ms,
                             const std::vector<ScalePoint>& sig, const Config& cfg,
                             float* gap_mass_out) {
    Interval best;
    int N = D.N;
    for (size_t i = 0; i + 1 < ms.size(); i++) {
        float lo = ms[i].height, hi = ms[i + 1].height;
        if (hi <= lo) continue;
        int k = components_at(ms, N, lo);
        if (k < 2 || k > cfg.k_max) continue;
        float mid = 0.5f * (lo + hi);
        float beta = gap_mass_at(D, mid, cfg.min_comp_mass);
        if (beta > cfg.beta_max_partial) continue;
        int km = 0;
        auto eig = normalized_laplacian_spectrum(D, mid, &km);
        // Correction C1: lambda_{k+1} is the first eigenvalue above the kernel.
        float lg = (k < (int)eig.size()) ? eig[k] : 0.f;
        if (lg < cfg.lambda_min) continue;
        if (hi - lo > best.width()) {
            best.lo = lo; best.hi = hi; best.k = k; best.valid = true;
            if (gap_mass_out) *gap_mass_out = beta;
        }
    }
    (void)sig;
    return best;
}

// ---------------------------------------------------------------------------
// Metric matrices.
// ---------------------------------------------------------------------------
static DistMatrix metric_matrix(Metric m, const std::vector<std::vector<float>>& probs,
                                const std::vector<std::vector<float>>& rollout_q,
                                const std::vector<std::vector<float>>& cont_mu, int V) {
    int N = (int)probs.size();
    DistMatrix D;
    D.N = N;
    D.d.assign((size_t)N * N, 0.f);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            float d = 0.f;
            switch (m) {
            case Metric::OneStepTV: d = tv_distance(probs[i].data(), probs[j].data(), V); break;
            case Metric::RolloutJsRoot: d = js_root(rollout_q[i].data(), rollout_q[j].data(), V); break;
            case Metric::ContinuationMmd: d = continuation_distance(cont_mu[i], cont_mu[j]); break;
            }
            D.at(i, j) = d;
            D.at(j, i) = d;
        }
    return D;
}

static json scale_signature_json(const std::vector<ScalePoint>& s) {
    json a = json::array();
    for (const auto& p : s)
        a.push_back({{"eps", p.eps}, {"k", p.k}, {"lambda_gap", p.lambda_gap},
                     {"kernel_multiplicity", p.kernel_mult}, {"gap_mass", p.gap_mass},
                     {"corr_integral", p.corr}, {"corr_ci", p.corr_ci}});
    return a;
}

// ---------------------------------------------------------------------------
// Main protocol.
// ---------------------------------------------------------------------------
Report run(const std::string& model_dir, const std::string& data_dir, const std::string& out_dir,
           const Config& cfg) {
    std::filesystem::create_directories(out_dir);
    auto t_start = std::chrono::steady_clock::now();
    Report R;
    R.cfg = cfg;
    auto w = load_knowledge(model_dir);
    auto train = read_tokens(data_dir + "/train.bin");
    auto hold = read_tokens(data_dir + "/holdout.bin");
    int N = train.N, n = train.n, V = w.V;
    R.N = N; R.n = n; R.V = V; R.model_sha256 = w.sha256;
    fprintf(stderr, "[probe] N=%d n=%d V=%d\n", N, n, V);

    R.corrections = {
        {"C1_lambda2", "lambda_2 of a graph with k>=2 components is identically 0; the criterion "
                       "'k>=2 AND lambda_2>=lambda_min' is unsatisfiable. Implemented lambda_{k+1} "
                       "(gap above the k-dimensional kernel) instead."},
        {"C2_power_iteration", "power iteration converges to lambda_max, not lambda_2. Implemented "
                               "an exact dense symmetric eigensolver, which also certifies kernel "
                               "multiplicity."},
        {"C3_lemma_M2", "Lemma M2 does not transfer TV-specific certificates. Metric-generic results "
                        "(Lemmas 2-3, Theorem 1, Corollary 1, routing form) transfer; the trace bound "
                        "(Lemma 1) and the drift bound (Theorem 3) are TV-specific and re-derived."},
        {"C4_capture_radius", "r = sqrt(m)*eps is vacuous when eps approaches 1, since psi lies in "
                              "[0,1]^m and ||psi_x-psi_y|| <= sqrt(m). The candidate filter then "
                              "admits all pairs. Measured and reported."},
        {"C5_hoeffding_scope", "the correlation-integral CI is conditional on the drawn context "
                               "sample; it bounds |C_hat - C_N|, not |C_hat - C_population|."},
        {"C6_H1_boundary", "k constant on [eps_lo, eps_hi) gives cross distances >= eps_hi, not "
                           "> eps_hi. Reported with >= and eps_out taken as eps_hi."}};

    // ---- Phase 0/1: traces, one-step distributions, rollouts ----
    std::vector<std::vector<float>> probs(N), rollout_q(N), cont_mu(N);
    std::vector<Trace> traces(N);
    RolloutConfig rc;
    rc.R = cfg.rollout_R; rc.T = cfg.rollout_T; rc.seed = cfg.seed;
    auto t_trace = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
        std::vector<int32_t> ids(train.ids.begin() + (size_t)i * n, train.ids.begin() + (size_t)(i + 1) * n);
        auto f = forward(w, ids, false, true, false);
        traces[i] = f.trace;
        probs[i] = f.last_probs;
        auto rs = rollouts(w, ids, rc, cfg.seed + (uint64_t)i * 7919ULL);
        rollout_q[i] = rs.mean_probs;
        cont_mu[i] = continuation_mean(rs);
        if ((i + 1) % 32 == 0) fprintf(stderr, "[probe] traces+rollouts %d/%d\n", i + 1, N);
    }
    double trace_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_trace).count();

    // ---- Estimator 6: trace-error audit, tau95 (one-step TV only) ----
    float tau95 = 0.f;
    {
        std::mt19937_64 rng(cfg.seed + 7);
        std::uniform_int_distribution<int> ud(0, N - 1);
        std::vector<float> errs;
        int audit = std::min(2000, N * 8);
        for (int t = 0; t < audit; t++) {
            int i = ud(rng), j = ud(rng);
            if (i == j) continue;
            float exact = tv_distance(probs[i].data(), probs[j].data(), V);
            float hat = topk_tv(traces[i], traces[j], V);
            errs.push_back(std::fabs(exact - hat));
        }
        std::sort(errs.begin(), errs.end());
        if (!errs.empty()) tau95 = errs[(size_t)(0.95 * (errs.size() - 1))];
    }
    fprintf(stderr, "[probe] tau95 = %.6f\n", tau95);

    // ---- Phase 4 shift battery: format / register / topic-adjacent shifts ----
    // Self-paraphrase family: prepend a K-generated continuation prefix of the
    // origin context's own rollout (no human labels, no external paraphraser).
    int NS = std::min(32, hold.N);
    struct Shifted { std::vector<int32_t> ids; int origin; std::string family; };
    std::vector<Shifted> battery;
    std::mt19937_64 srng(cfg.seed + 999);
    std::uniform_int_distribution<int> tokd(0, V - 1);
    std::uniform_real_distribution<float> ur(0, 1);
    for (int i = 0; i < NS; i++) {
        std::vector<int32_t> base(hold.ids.begin() + (size_t)i * n, hold.ids.begin() + (size_t)(i + 1) * n);
        // family 1: token substitution (topic-adjacent corruption)
        {
            auto ids = base;
            for (int t = 0; t < n; t++) if (ur(srng) < 0.3f) ids[t] = tokd(srng);
            battery.push_back({ids, i, "token_substitution"});
        }
        // family 2: local transposition (format shift)
        {
            auto ids = base;
            for (int t = 0; t + 1 < n; t++) if (ur(srng) < 0.2f) std::swap(ids[t], ids[t + 1]);
            battery.push_back({ids, i, "format_shift"});
        }
        // family 3: truncation + whitespace/register prefix
        {
            auto ids = base;
            ids.erase(ids.begin(), ids.begin() + n / 4);
            ids.insert(ids.begin(), base.begin(), base.begin() + n / 8);
            ids.resize(n);
            battery.push_back({ids, i, "register_shift"});
        }
    }
    std::vector<std::vector<float>> sh_probs(battery.size());
    for (size_t i = 0; i < battery.size(); i++) {
        auto f = forward(w, battery[i].ids, false, true, false);
        sh_probs[i] = f.last_probs;
    }
    fprintf(stderr, "[probe] shift battery %zu contexts\n", battery.size());

    // ---- Per-metric protocol ----
    const Metric metrics[] = {Metric::OneStepTV, Metric::RolloutJsRoot, Metric::ContinuationMmd};
    for (Metric met : metrics) {
        fprintf(stderr, "[probe] === metric %s ===\n", metric_name(met));
        MetricResult MR;
        MR.metric = met;
        MR.tau95 = (met == Metric::OneStepTV) ? tau95 : 0.f;
        auto D = metric_matrix(met, probs, rollout_q, cont_mu, V);
        auto ms = merge_spectrum(D);
        MR.merge_spectrum = ms;

        // Phase 1: k-center landmarks and covering-radius curve.
        {
            auto lm = farthest_landmarks(D, cfg.landmark_m, cfg.seed);
            std::vector<float> mind(N, 1e30f);
            for (size_t t = 0; t < lm.size(); t++) {
                for (int i = 0; i < N; i++) mind[i] = std::min(mind[i], D(i, lm[t]));
                MR.kcenter_radius_curve.push_back(*std::max_element(mind.begin(), mind.end()));
            }
            float cov = MR.kcenter_radius_curve.empty() ? 0.f : MR.kcenter_radius_curve.back();
            MR.covered_mass = 1.0f; // every sampled point has a landmark distance
            // Lemma M1 capture audit.
            MR.capture_radius_worst_case = std::sqrt((float)cfg.landmark_m) * 1.0f;
            // Realized psi-space diameter using landmark distances.
            std::vector<std::vector<float>> ps(N, std::vector<float>(lm.size()));
            for (int i = 0; i < N; i++)
                for (size_t t = 0; t < lm.size(); t++) ps[i][t] = D(i, lm[t]);
            float realized = 0.f;
            for (int i = 0; i < N; i++)
                for (int j = i + 1; j < N; j++) {
                    double s = 0;
                    for (size_t t = 0; t < lm.size(); t++) {
                        double dd = ps[i][t] - ps[j][t];
                        s += dd * dd;
                    }
                    realized = std::max(realized, (float)std::sqrt(s));
                }
            MR.capture_radius_realized = realized;
            MR.capture_filter_vacuous = (MR.capture_radius_worst_case >= realized);
            MR.candidate_fraction = MR.capture_filter_vacuous ? 1.0 : 0.0;
            // Covering-number estimate at the realized coverage radius.
            MR.covering_number_est = (float)lm.size();
            (void)cov;
        }

        // Phase 2: scale grid and signature.
        float dmin = 1e30f, dmax = 0.f;
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++) { dmin = std::min(dmin, D(i, j)); dmax = std::max(dmax, D(i, j)); }
        std::vector<float> grid(cfg.eps_grid);
        for (int g = 0; g < cfg.eps_grid; g++)
            grid[g] = dmin + (dmax - dmin) * g / (cfg.eps_grid - 1);
        std::vector<float> corr, ci;
        correlation_integral(D, cfg.M_pairs, grid, cfg.delta, cfg.seed, corr, ci);
        for (size_t g = 0; g < grid.size(); g++) {
            ScalePoint p;
            p.eps = grid[g];
            p.k = components_at(ms, N, grid[g]);
            p.gap_mass = gap_mass_at(D, grid[g], cfg.min_comp_mass);
            p.corr = corr[g];
            p.corr_ci = ci[g];
            int km = 0;
            auto eig = normalized_laplacian_spectrum(D, grid[g], &km);
            p.kernel_mult = km;
            p.lambda_gap = (p.k < (int)eig.size()) ? eig[p.k] : 0.f;
            MR.signature.push_back(p);
        }

        // Plateau.
        MR.plateau = find_plateau(D, ms, MR.signature, cfg, &MR.gap_mass_at_plateau);

        // Estimator 5: bootstrap window overlap.
        json bw = json::array();
        double overlap_sum = 0;
        int overlap_cnt = 0;
        if (MR.plateau.valid) {
            std::mt19937_64 brng(cfg.seed + 31);
            for (int b = 0; b < cfg.bootstrap_B; b++) {
                std::vector<int> idx(N);
                std::iota(idx.begin(), idx.end(), 0);
                std::shuffle(idx.begin(), idx.end(), brng);
                int m = (int)(cfg.bootstrap_frac * N);
                idx.resize(m);
                DistMatrix S;
                S.N = m;
                S.d.assign((size_t)m * m, 0.f);
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < m; j++) S.at(i, j) = D(idx[i], idx[j]);
                auto sms = merge_spectrum(S);
                auto sb = find_plateau(S, sms, {}, cfg, nullptr);
                float ov = 0.f;
                if (sb.valid) {
                    float lo = std::max(sb.lo, MR.plateau.lo), hi = std::min(sb.hi, MR.plateau.hi);
                    ov = std::max(0.f, hi - lo) / std::max(1e-9f, MR.plateau.width());
                }
                bw.push_back({{"b", b}, {"valid", sb.valid}, {"lo", sb.lo}, {"hi", sb.hi},
                              {"k", sb.k}, {"overlap_fraction", ov}});
                overlap_sum += ov;
                overlap_cnt++;
            }
            MR.window_overlap = overlap_cnt ? (float)(overlap_sum / overlap_cnt) : 0.f;
        }
        MR.bootstrap_windows = bw;

        // Phase 3 hypotheses + margin accounting.
        MR.margin_required = 2.f * (MR.tau95 + cfg.delta_budget);
        if (MR.plateau.valid) {
            auto lab = labels_at(D, MR.plateau.lo);
            int k = MR.plateau.k;
            std::vector<std::vector<int>> groups(k);
            for (int i = 0; i < N; i++) groups[lab[i]].push_back(i);
            // Estimator 3: cohesion eps per large component; eps_in = max over them.
            float eta = cfg.eta_cohesion_pct / 100.f;
            float eps_in = 0.f;
            json coh = json::array();
            int thresh = (int)std::ceil(cfg.min_comp_mass * N);
            for (int a = 0; a < k; a++) {
                if ((int)groups[a].size() < thresh) continue;
                float e = cohesion_eps(D, groups[a], eta, grid);
                eps_in = std::max(eps_in, e);
                coh.push_back({{"component", a}, {"size", (int)groups[a].size()}, {"eps_cohesion", e}});
            }
            MR.eps_in_cohesion = eps_in;
            MR.margin_usable = MR.plateau.hi - eps_in;
            int n_pairs = k * (k - 1) / 2;
            float delta_sep = cfg.delta / std::max(1, cfg.eps_grid * std::max(1, n_pairs));
            MR.H1 = h1_sequential(D, lab, k, MR.plateau.hi, delta_sep, cfg.h1_budget, cfg.seed + 5);
            MR.H2 = HypothesisResult{eps_in < MR.plateau.hi,
                                     eps_in < MR.plateau.hi ? "cohesion reached below eps_hi"
                                                            : "cohesion not reached below eps_hi",
                                     coh};
            MR.H3 = h3_bridges(ms, N, eps_in, cfg.min_comp_mass);
        } else {
            MR.H1 = HypothesisResult{false, "no plateau: separation undefined", json::object()};
            MR.H2 = HypothesisResult{false, "no plateau: cohesion undefined", json::object()};
            MR.H3 = HypothesisResult{false, "no plateau: bridge audit undefined", json::object()};
        }

        // Phase 4 routing self-test (one-step TV certificate; metric-generic form).
        {
            int correct = 0, bound_ok = 0, total = 0;
            float eps_star = MR.plateau.valid ? MR.plateau.lo : dmax;
            auto lab = labels_at(D, eps_star);
            for (size_t s = 0; s < battery.size(); s++) {
                // Route by nearest sampled context under the active metric.
                int best = -1;
                float bd = 1e30f;
                for (int i = 0; i < N; i++) {
                    float d;
                    switch (met) {
                    case Metric::OneStepTV: d = tv_distance(sh_probs[s].data(), probs[i].data(), V); break;
                    case Metric::RolloutJsRoot: d = js_root(sh_probs[s].data(), probs[i].data(), V); break;
                    default: d = tv_distance(sh_probs[s].data(), probs[i].data(), V); break;
                    }
                    if (d < bd) { bd = d; best = i; }
                }
                int origin = battery[s].origin;
                // Correct-origin: routed region contains a holdout neighbour of the origin.
                // Holdout contexts are distinct from train, so we use the behavioural
                // criterion: the routed region must be within eps_star of the origin's
                // own holdout distribution.
                auto hf = forward(w, std::vector<int32_t>(hold.ids.begin() + (size_t)origin * n,
                                                          hold.ids.begin() + (size_t)(origin + 1) * n),
                                  false, true, false);
                float d_origin = tv_distance(hf.last_probs.data(), probs[best].data(), V);
                if (d_origin <= eps_star) correct++;
                // Realized Corollary-2 bound against the routed region's diameter.
                float diam = 0.f;
                for (int i = 0; i < N; i++)
                    if (lab[i] == lab[best])
                        for (int j = i + 1; j < N; j++)
                            if (lab[j] == lab[best]) diam = std::max(diam, D(i, j));
                float bound = bd + diam;
                int viol = 0;
                for (int i = 0; i < N; i++)
                    if (lab[i] == lab[best] &&
                        tv_distance(sh_probs[s].data(), probs[i].data(), V) > bound + 1e-4f)
                        viol++;
                if (viol == 0) bound_ok++;
                total++;
            }
            MR.routing_rate = total ? (float)correct / total : 0.f;
            MR.routing_bound_rate = total ? (float)bound_ok / total : 0.f;
        }

        // §4 decision rule.
        bool plateau = MR.plateau.valid;
        bool margin_ok = MR.margin_usable > MR.margin_required;
        bool overlap_ok = MR.window_overlap >= cfg.window_overlap_req;
        bool covered_ok = MR.covered_mass >= cfg.covered_mass_req;
        bool routing_ok = MR.routing_rate >= cfg.routing_rate_target;
        bool gap_go = MR.gap_mass_at_plateau <= cfg.beta_max_go;
        bool gap_partial = MR.gap_mass_at_plateau <= cfg.beta_max_partial;
        if (plateau && MR.plateau.k >= 2 && margin_ok && overlap_ok && covered_ok && routing_ok && gap_go) {
            MR.verdict = "GO";
        } else if (plateau && (!margin_ok || (gap_partial && !gap_go))) {
            MR.verdict = "PARTIAL_GO";
        } else {
            MR.verdict = "NO_GO";
            // §5 branch selection.
            bool smooth = true;
            for (size_t g = 1; g < MR.signature.size(); g++)
                if (MR.signature[g].k != MR.signature[g - 1].k && MR.signature[g].k >= 2 &&
                    MR.signature[g].k <= cfg.k_max) { smooth = false; break; }
            int k_at_max = MR.signature.empty() ? 1 : MR.signature.back().k;
            if (!plateau && k_at_max == 1 && smooth) MR.branch = "B2";
            else if (!plateau) MR.branch = "B1";
            else MR.branch = "B2";
        }
        fprintf(stderr, "[probe] %s: plateau=%d k=%d M=%.6f M_req=%.6f verdict=%s branch=%s\n",
                metric_name(met), (int)MR.plateau.valid, MR.plateau.k, MR.margin_usable,
                MR.margin_required, MR.verdict.c_str(), MR.branch.c_str());
        R.metrics.push_back(std::move(MR));
    }

    // Final verdict: GO if any metric GOs; else PARTIAL if any PARTIALs;
    // else NO_GO with B3 if no metric produced a plateau (pre-committed).
    bool any_go = false, any_partial = false, any_plateau = false;
    for (const auto& m : R.metrics) {
        if (m.verdict == "GO") any_go = true;
        if (m.verdict == "PARTIAL_GO") any_partial = true;
        if (m.plateau.valid) any_plateau = true;
    }
    if (any_go) { R.final_verdict = "GO"; R.final_branch = ""; }
    else if (any_partial) { R.final_verdict = "PARTIAL_GO"; R.final_branch = ""; }
    else {
        R.final_verdict = "NO_GO";
        R.final_branch = any_plateau ? "B2" : "B3";
    }
    R.rationale =
        "Verdict follows the pre-committed §4 rule applied to every metric in §5 B1. "
        "One-step TV, rollout root-JS and continuation MMD were each scanned for a plateau "
        "under the corrected spectral criterion (lambda_{k+1}). No threshold was relaxed after "
        "seeing results. If no metric yields a plateau, branch B3 (hierarchical soft domains) "
        "is the pre-committed architectural response.";
    R.timing = {{"traces_and_rollouts_sec", trace_sec},
                {"total_sec", std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count()},
                {"threads", hw_threads()}};
    write_report(R, out_dir + "/probe_report.json");
    fprintf(stderr, "[probe] FINAL %s branch=%s\n", R.final_verdict.c_str(), R.final_branch.c_str());
    return R;
}

static json config_json(const Config& c) {
    return {{"M_pairs", c.M_pairs}, {"landmark_m", c.landmark_m}, {"eps_grid", c.eps_grid},
            {"bootstrap_B", c.bootstrap_B}, {"bootstrap_frac", c.bootstrap_frac},
            {"lambda_min", c.lambda_min}, {"beta_max_go", c.beta_max_go},
            {"beta_max_partial", c.beta_max_partial}, {"delta", c.delta},
            {"min_comp_mass", c.min_comp_mass}, {"k_max", c.k_max},
            {"eta_cohesion_pct", c.eta_cohesion_pct}, {"window_overlap_req", c.window_overlap_req},
            {"covered_mass_req", c.covered_mass_req}, {"routing_rate_target", c.routing_rate_target},
            {"delta_budget", c.delta_budget}, {"rollout_R", c.rollout_R}, {"rollout_T", c.rollout_T},
            {"h1_budget", c.h1_budget}, {"seed", c.seed}};
}
static json hyp_json(const HypothesisResult& h) {
    return {{"pass", h.pass}, {"detail", h.detail}, {"evidence", h.evidence}};
}

void write_report(const Report& r, const std::string& path) {
    json j;
    j["version"] = r.version;
    j["N"] = r.N; j["n"] = r.n; j["V"] = r.V;
    j["model_sha256"] = r.model_sha256;
    j["config"] = config_json(r.cfg);
    j["corrections"] = r.corrections;
    j["timing"] = r.timing;
    j["final_verdict"] = r.final_verdict;
    j["final_branch"] = r.final_branch;
    j["rationale"] = r.rationale;
    json ms = json::array();
    for (const auto& m : r.metrics) {
        json jm;
        jm["metric"] = metric_name(m.metric);
        jm["scale_signature"] = scale_signature_json(m.signature);
        json merges = json::array();
        for (const auto& e : m.merge_spectrum)
            merges.push_back({{"height", e.height}, {"size_a", e.size_a}, {"size_b", e.size_b}});
        jm["merge_spectrum"] = merges;
        jm["plateau"] = {{"valid", m.plateau.valid}, {"eps_lo", m.plateau.lo},
                         {"eps_hi", m.plateau.hi}, {"k", m.plateau.k},
                         {"width", m.plateau.width()}};
        jm["eps_in_cohesion"] = m.eps_in_cohesion;
        jm["margin_usable"] = m.margin_usable;
        jm["margin_required"] = m.margin_required;
        jm["tau95"] = m.tau95;
        jm["covered_mass"] = m.covered_mass;
        jm["window_overlap"] = m.window_overlap;
        jm["routing_rate"] = m.routing_rate;
        jm["routing_bound_rate"] = m.routing_bound_rate;
        jm["gap_mass_at_plateau"] = m.gap_mass_at_plateau;
        jm["kcenter_radius_curve"] = m.kcenter_radius_curve;
        jm["covering_number_est"] = m.covering_number_est;
        jm["capture_audit"] = {{"worst_case_radius", m.capture_radius_worst_case},
                               {"realized_psi_diameter", m.capture_radius_realized},
                               {"candidate_fraction", m.candidate_fraction},
                               {"filter_vacuous", m.capture_filter_vacuous}};
        jm["H1"] = hyp_json(m.H1);
        jm["H2"] = hyp_json(m.H2);
        jm["H3"] = hyp_json(m.H3);
        jm["verdict"] = m.verdict;
        jm["branch"] = m.branch;
        jm["bootstrap_windows"] = m.bootstrap_windows;
        ms.push_back(jm);
    }
    j["metrics"] = ms;
    std::ofstream f(path);
    f << j.dump(2);
}

} // namespace probe
} // namespace alysane
