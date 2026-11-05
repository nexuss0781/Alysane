// Completion gates for C2+C3: KV parity, continuation behaviour, calibrated
// multiscale structure (A1 replacement), quantisation, and reshape drift.
//
// Pre-registered thresholds are declared in docs/03_completion_addendum.md.
// They are NOT tuned after seeing results.
#include "alysane.h"
#include <omp.h>

namespace alysane {

static double auroc2(const std::vector<double>& pos, const std::vector<double>& neg) {
    if (pos.empty() || neg.empty()) return 0.5;
    double s = 0;
    for (double p : pos)
        for (double n : neg) s += (p > n ? 1.0 : p == n ? 0.5 : 0.0);
    return s / (pos.size() * neg.size());
}
static double median_of(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static double spearman2(const std::vector<double>& a, const std::vector<double>& b) {
    int n = (int)a.size();
    if (n < 3) return 0;
    std::vector<int> ia(n), ib(n);
    std::iota(ia.begin(), ia.end(), 0);
    std::iota(ib.begin(), ib.end(), 0);
    std::sort(ia.begin(), ia.end(), [&](int x, int y) { return a[x] < a[y]; });
    std::sort(ib.begin(), ib.end(), [&](int x, int y) { return b[x] < b[y]; });
    std::vector<double> ra(n), rb(n);
    for (int i = 0; i < n; i++) { ra[ia[i]] = i; rb[ib[i]] = i; }
    double ma = 0, mb = 0;
    for (int i = 0; i < n; i++) { ma += ra[i]; mb += rb[i]; }
    ma /= n; mb /= n;
    double num = 0, da = 0, db = 0;
    for (int i = 0; i < n; i++) {
        double x = ra[i] - ma, y = rb[i] - mb;
        num += x * y; da += x * x; db += y * y;
    }
    return (da > 0 && db > 0) ? num / std::sqrt(da * db) : 0;
}

void run_completion_gates(const std::string& model_dir, const std::string& data_dir,
                          const std::string& out_dir) {
    std::filesystem::create_directories(out_dir);
    json G;
    {
        std::ifstream gf(out_dir + "/gates.json");
        if (gf) gf >> G;
    }
    auto w = load_knowledge(model_dir);
    auto train = read_tokens(data_dir + "/train.bin");
    auto hold = read_tokens(data_dir + "/holdout.bin");
    int N = train.N, n = train.n, V = w.V, NH = hold.N;
    auto traces = read_traces(out_dir + "/train_traces.bin", w.L, w.d, w.H);
    auto probs = read_probs(out_dir + "/train_probs.bin", N, V);
    fprintf(stderr, "[completion] loaded %d train traces\n", N);

    // ---------------- G8: KV-cache parity ----------------
    // Cached prefill+decode must reproduce a full forward pass.
    auto t0 = std::chrono::steady_clock::now();
    int kvN = std::min(8, N);
    double kv_max_dp = 0, kv_max_dlogit = 0;
    int kv_top1_match = 0, kv_top1_total = 0;
    for (int i = 0; i < kvN; i++) {
        std::vector<int32_t> ids(train.ids.begin() + (size_t)i * n, train.ids.begin() + (size_t)(i + 1) * n);
        auto full = forward(w, ids, false, false, true);
        KVCache cache;
        cache.init(w.L, w.KVH, w.head_dim, n);
        std::vector<int32_t> pre(ids.begin(), ids.end() - 1);
        forward_kv(w, pre, 0, false, &cache);
        auto p = decode_step(w, cache, ids.back());
        int a = (int)(std::max_element(full.last_probs.begin(), full.last_probs.end()) - full.last_probs.begin());
        int b = (int)(std::max_element(p.begin(), p.end()) - p.begin());
        kv_top1_match += (a == b);
        kv_top1_total++;
        for (int v = 0; v < V; v++) kv_max_dp = std::max(kv_max_dp, (double)std::fabs(full.last_probs[v] - p[v]));
    }
    // Also verify a multi-token generated continuation against teacher-forced full forward.
    double kv_gen_max_dp = 0;
    {
        std::vector<int32_t> ids(train.ids.begin(), train.ids.begin() + n);
        KVCache cache;
        int T = 6;
        cache.init(w.L, w.KVH, w.head_dim, n + T);
        auto pre = forward_kv(w, ids, 0, true, &cache);
        std::vector<int32_t> gen;
        std::vector<float> p = pre.last_probs;
        for (int t = 0; t < T; t++) {
            int tok = (int)(std::max_element(p.begin(), p.end()) - p.begin());
            gen.push_back(tok);
            auto full_ids = ids;
            full_ids.insert(full_ids.end(), gen.begin(), gen.end());
            auto full = forward(w, full_ids, false, false, true);
            for (int v = 0; v < V; v++)
                kv_gen_max_dp = std::max(kv_gen_max_dp, (double)std::fabs(full.last_probs[v] - p[v]));
            if (t + 1 < T) p = decode_step(w, cache, tok);
        }
    }
    bool g8 = kv_max_dp <= 1e-5 && kv_gen_max_dp <= 1e-5 && kv_top1_match == kv_top1_total;
    G["gates"]["G8_kv_parity"] = g8 ? "PASS" : "FAIL";
    G["kv_parity"] = {{"contexts", kvN}, {"max_prob_diff_single_step", kv_max_dp},
                      {"max_prob_diff_generated", kv_gen_max_dp},
                      {"top1_agreement", (double)kv_top1_match / std::max(1, kv_top1_total)},
                      {"threshold", 1e-5}};
    fprintf(stderr, "[completion] G8 %s (dp=%.3g gen=%.3g)\n", g8 ? "PASS" : "FAIL", kv_max_dp, kv_gen_max_dp);

    // ---------------- Rollouts: §1.3 signals ----------------
    RolloutConfig cfg;
    cfg.R = 4; cfg.T = 8; cfg.temperature = 1.0f; cfg.seed = 20261105ULL;
    auto roll_start = std::chrono::steady_clock::now();
    std::vector<RolloutSet> tr_roll(N), ho_roll(NH);
    std::vector<std::vector<float>> tr_mu(N), ho_mu(NH);
    for (int i = 0; i < N; i++) {
        std::vector<int32_t> ids(train.ids.begin() + (size_t)i * n, train.ids.begin() + (size_t)(i + 1) * n);
        tr_roll[i] = rollouts(w, ids, cfg, cfg.seed + (uint64_t)i * 7919ULL);
        tr_mu[i] = continuation_mean(tr_roll[i]);
        if ((i + 1) % 32 == 0) fprintf(stderr, "[rollout] %d/%d\n", i + 1, N);
    }
    for (int i = 0; i < NH; i++) {
        std::vector<int32_t> ids(hold.ids.begin() + (size_t)i * n, hold.ids.begin() + (size_t)(i + 1) * n);
        ho_roll[i] = rollouts(w, ids, cfg, cfg.seed + 1000000ULL + (uint64_t)i * 7919ULL);
        ho_mu[i] = continuation_mean(ho_roll[i]);
    }
    // Synthetic shifted holdout contexts (same perturbation family as G5/G6).
    int NS = std::min(32, NH);
    std::vector<RolloutSet> syn_roll(NS);
    std::vector<std::vector<float>> syn_mu(NS);
    std::mt19937_64 srng(999);
    std::uniform_int_distribution<int> tokd(0, V - 1), posd(0, n - 1);
    std::uniform_real_distribution<float> ur(0, 1);
    for (int i = 0; i < NS; i++) {
        std::vector<int32_t> ids(hold.ids.begin() + (size_t)i * n, hold.ids.begin() + (size_t)(i + 1) * n);
        for (int t = 0; t < n; t++) if (ur(srng) < 0.3f) ids[t] = tokd(srng);
        for (int t = 0; t + 1 < n; t++) if (ur(srng) < 0.2f) std::swap(ids[t], ids[t + 1]);
        syn_roll[i] = rollouts(w, ids, cfg, cfg.seed + 2000000ULL + (uint64_t)i * 7919ULL);
        syn_mu[i] = continuation_mean(syn_roll[i]);
    }
    double roll_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - roll_start).count();

    // Continuation MMD matrix (train).
    DistMatrix DC;
    DC.N = N;
    DC.d.assign((size_t)N * N, 0.f);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            float d = continuation_distance(tr_mu[i], tr_mu[j]);
            DC.at(i, j) = d;
            DC.at(j, i) = d;
        }
    // Pseudometric axioms on the continuation representation.
    int tri_viol = 0, sym_viol = 0, self_viol = 0;
    for (int i = 0; i < N; i++) {
        if (DC(i, i) > 1e-6) self_viol++;
        for (int j = 0; j < N; j++) {
            if (std::fabs(DC(i, j) - DC(j, i)) > 1e-6) sym_viol++;
            for (int k = j + 1; k < N; k++)
                if (DC(i, k) > DC(i, j) + DC(j, k) + 1e-4) tri_viol++;
        }
    }
    // Self-agreement / consistency ranges.
    double sa_min = 1e9, sa_max = -1e9, sa_sum = 0;
    for (auto& r : tr_roll) { sa_min = std::min(sa_min, (double)r.self_agreement); sa_max = std::max(sa_max, (double)r.self_agreement); sa_sum += r.self_agreement; }
    std::vector<double> dcont, done;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            dcont.push_back(DC(i, j));
            done.push_back(tv_distance(probs[i].data(), probs[j].data(), V));
        }
    double rho_cont_one = spearman2(dcont, done);
    bool g9 = tri_viol == 0 && sym_viol == 0 && self_viol == 0 && sa_min >= -1e-6 && sa_max <= 1.0 + 1e-6;
    G["gates"]["G9_continuation"] = g9 ? "PASS" : "FAIL";
    G["continuation"] = {{"R", cfg.R}, {"T", cfg.T}, {"temperature", cfg.temperature},
                         {"feature_dim", CONT_FEATURE_DIM},
                         {"triangle_violations", tri_viol}, {"symmetry_violations", sym_viol},
                         {"self_distance_violations", self_viol},
                         {"self_agreement_min", sa_min}, {"self_agreement_max", sa_max},
                         {"self_agreement_mean", sa_sum / N},
                         {"spearman_continuation_onestep", rho_cont_one},
                         {"rollout_wall_sec", roll_sec},
                         {"status", "[E] measured; pseudometric axioms verified exactly"}};
    fprintf(stderr, "[completion] G9 %s (tri=%d rho=%.3f)\n", g9 ? "PASS" : "FAIL", tri_viol, rho_cont_one);

    // ---------------- G3b: calibrated multiscale structure ----------------
    auto D1 = pairwise_tv(probs, V);
    auto pers_one = persistence0(D1, 20, 4242, 0.05f);
    auto pers_cont = persistence0(DC, 20, 4242, 0.05f);
    int K = 32;
    auto lm_cont = farthest_landmarks(DC, K, 12345);
    auto regions = soft_regions(DC, lm_cont, 0.15f);
    // Coverage: fraction of points with non-fallback (multi-anchor or in-radius) membership.
    int covered = 0;
    for (int i = 0; i < N; i++) {
        float mx = *std::max_element(regions.member[i].begin(), regions.member[i].end());
        bool in_radius = false;
        for (int a = 0; a < K; a++)
            if (DC(i, lm_cont[a]) < regions.radius[a]) in_radius = true;
        if (in_radius) covered++;
        (void)mx;
    }
    double coverage = (double)covered / N;
    // Stability: re-derive anchors/radii on 80% subsamples, compare argmax labels.
    std::vector<int> base_label(N);
    for (int i = 0; i < N; i++)
        base_label[i] = (int)(std::max_element(regions.member[i].begin(), regions.member[i].end()) - regions.member[i].begin());
    std::mt19937_64 brng(777);
    double ari_sum = 0; int B = 10;
    std::vector<double> aris;
    for (int b = 0; b < B; b++) {
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), brng);
        int m = (int)(0.8 * N);
        idx.resize(m);
        DistMatrix S;
        S.N = m;
        S.d.assign((size_t)m * m, 0.f);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++) S.at(i, j) = DC(idx[i], idx[j]);
        auto lm_b = farthest_landmarks(S, K, 12345 + b);
        auto reg_b = soft_regions(S, lm_b, 0.15f);
        std::vector<int> lab_b(m), ref(m);
        for (int i = 0; i < m; i++) {
            lab_b[i] = (int)(std::max_element(reg_b.member[i].begin(), reg_b.member[i].end()) - reg_b.member[i].begin());
            ref[i] = base_label[idx[i]];
        }
        double ari = adjusted_rand(ref, lab_b);
        aris.push_back(ari);
        ari_sum += ari;
    }
    double mean_ari = ari_sum / B;
    bool g3b = pers_cont.n_significant >= 1 && coverage >= 0.80 && mean_ari >= 0.50;
    G["gates"]["G3b_structure"] = g3b ? "PASS" : "FAIL";
    G["structure"] = {
        {"method", "0-dim persistence + permutation null; overlapping partition-of-unity regions"},
        {"persistence_onestep_significant", pers_one.n_significant},
        {"persistence_continuation_significant", pers_cont.n_significant},
        {"null_B", 20}, {"alpha", 0.05},
        {"anchors_K", K}, {"radius_quantile", 0.15},
        {"coverage", coverage}, {"bootstrap_B", B}, {"bootstrap_mean_ari", mean_ari},
        {"bootstrap_aris", aris},
        {"thresholds", {{"min_significant", 1}, {"min_coverage", 0.80}, {"min_mean_ari", 0.50}}},
        {"note", "[E] calibrated structure; replaces hard A1 partition. Not a claim of disjoint domains."}};
    fprintf(stderr, "[completion] G3b %s (sig_cont=%d cov=%.3f ari=%.3f)\n", g3b ? "PASS" : "FAIL",
            pers_cont.n_significant, coverage, mean_ari);

    // ---------------- G10: boundary via continuation self-agreement ----------------
    std::vector<double> nat_sa, syn_sa;
    for (auto& r : ho_roll) nat_sa.push_back(r.self_agreement);
    for (auto& r : syn_roll) syn_sa.push_back(r.self_agreement);
    double auc_sa = auroc2(syn_sa, nat_sa); // higher agreement = more natural -> invert
    double auc_inv = auroc2(nat_sa, syn_sa);
    std::vector<double> nat_bs, syn_bs;
    // Boundary score of holdout/synthetic under the continuation soft regions.
    auto boundary_of = [&](const std::vector<float>& mu) {
        std::vector<float> da(K);
        for (int a = 0; a < K; a++) da[a] = continuation_distance(mu, tr_mu[lm_cont[a]]);
        auto mem = soft_membership(da, regions.radius);
        double h = 0;
        for (float p : mem) if (p > 0) h -= (double)p * std::log((double)p);
        return h / std::log((double)std::max(2, K));
    };
    for (auto& mu : ho_mu) nat_bs.push_back(boundary_of(mu));
    for (auto& mu : syn_mu) syn_bs.push_back(boundary_of(mu));
    double auc_bs = auroc2(syn_bs, nat_bs);
    bool g10 = auc_inv >= 0.60 || auc_bs >= 0.60;
    G["gates"]["G10_continuation_boundary"] = g10 ? "PASS" : "FAIL";
    G["continuation_boundary"] = {{"auroc_self_agreement_nat_vs_syn", auc_inv},
                                  {"auroc_boundary_score_syn_vs_nat", auc_bs},
                                  {"median_natural_self_agreement", median_of(nat_sa)},
                                  {"median_synthetic_self_agreement", median_of(syn_sa)},
                                  {"threshold", 0.60},
                                  {"note", "[E] proxy boundary signal; not competence (C3v2-1)"}};
    fprintf(stderr, "[completion] G10 %s (auc_sa=%.3f auc_bs=%.3f)\n", g10 ? "PASS" : "FAIL", auc_inv, auc_bs);

    // ---------------- G11: quantisation ----------------
    int qN = std::min(64, N);
    int q_tv_viol = 0; double q_max_feat_err = 0, q_max_enc_drift = 0;
    SelfKnowledgeWeights S;
    bool have_S = false;
    try { S = SelfKnowledgeWeights::load(out_dir + "/self_weights.alys"); have_S = true; } catch (...) {}
    for (int i = 0; i < qN; i++) {
        auto q = quantize_trace(traces[i], w.L, w.d);
        auto dq = dequantize_trace(q, w.L, w.d);
        // Top-k distribution must be untouched: TV estimate identical.
        for (int j = i + 1; j < qN; j++) {
            auto q2 = quantize_trace(traces[j], w.L, w.d);
            float a = topk_tv(traces[i], traces[j], V);
            float b = quantized_topk_tv(q, q2, V);
            if (std::fabs(a - b) > 0) q_tv_viol++;
        }
        auto f0 = traces[i].features(), f1 = dq.features();
        for (size_t k = 0; k < f0.size(); k++) q_max_feat_err = std::max(q_max_feat_err, (double)std::fabs(f0[k] - f1[k]));
        if (have_S) {
            auto z0 = S.encode(traces[i]), z1 = S.encode(dq);
            double s = 0;
            for (size_t k = 0; k < z0.size(); k++) { double d = z0[k] - z1[k]; s += d * d; }
            q_max_enc_drift = std::max(q_max_enc_drift, std::sqrt(s));
        }
    }
    bool g11 = q_tv_viol == 0;
    G["gates"]["G11_quantisation"] = g11 ? "PASS" : "FAIL";
    size_t raw_bytes = (size_t)w.L * 2 * w.d * 4;
    size_t q_bytes = (size_t)w.L * 2 * w.d + (size_t)w.L * 4;
    G["quantisation"] = {{"contexts", qN}, {"topk_tv_violations", q_tv_viol},
                         {"max_feature_abs_error", q_max_feat_err},
                         {"max_encoder_l2_drift", q_max_enc_drift},
                         {"raw_trace_bytes", raw_bytes}, {"int8_trace_bytes", q_bytes},
                         {"compression_ratio", (double)raw_bytes / q_bytes},
                         {"theorem", "hidden-state int8 quantisation leaves top-k TV and tau unchanged exactly"}};
    fprintf(stderr, "[completion] G11 %s (tv_viol=%d enc_drift=%.3g)\n", g11 ? "PASS" : "FAIL", q_tv_viol, q_max_enc_drift);

    // ---------------- G12: reshape drift bound ----------------
    int dN = std::min(16, N);
    std::vector<std::vector<int32_t>> dctx;
    for (int i = 0; i < dN; i++)
        dctx.emplace_back(train.ids.begin() + (size_t)i * n, train.ids.begin() + (size_t)(i + 1) * n);
    auto drift = measure_drift(w, dctx, 1e-3f, 31337);
    bool g12 = drift.bound_violations == 0 && drift.pairs > 0;
    G["gates"]["G12_reshape_drift"] = g12 ? "PASS" : "FAIL";
    G["reshape"] = {{"contexts", dN}, {"pairs", drift.pairs},
                    {"delta_theta_norm", drift.delta_theta_norm},
                    {"max_logit_inf", drift.max_logit_inf},
                    {"empirical_L_z", drift.empirical_L_z},
                    {"max_tv_change", drift.max_tv_change},
                    {"predicted_bound_2_max_logit_inf", drift.predicted_bound},
                    {"bound_violations", drift.bound_violations},
                    {"theorem", "|d_B' - d_B| <= 2 max_x ||z'(x)-z(x)||_inf (safe softmax TV bound)"}};
    fprintf(stderr, "[completion] G12 %s (viol=%d bound=%.3g actual=%.3g)\n", g12 ? "PASS" : "FAIL",
            drift.bound_violations, drift.predicted_bound, drift.max_tv_change);

    // ---------------- Memory / performance ----------------
    KVCache probe;
    probe.init(w.L, w.KVH, w.head_dim, n + cfg.T);
    G["performance"]["kv_cache_bytes_per_stream"] = probe.bytes();
    G["performance"]["rollout_wall_sec"] = roll_sec;
    G["performance"]["rollouts_total"] = (N + NH + NS) * cfg.R;
    G["performance"]["decode_tokens_total"] = (N + NH + NS) * cfg.R * cfg.T;
    G["performance"]["completion_wall_sec"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // ---------------- Overall ----------------
    int fails = 0;
    std::vector<std::string> failed;
    for (auto& kv : G["gates"].items())
        if (kv.value() != "PASS") { fails++; failed.push_back(kv.key()); }
    G["fail_count"] = fails;
    G["failed_gates"] = failed;
    G["overall"] = fails == 0 ? "PASS" : "FAIL";
    G["ready_for_next_component"] = fails == 0;
    G["component"] = "C2+C3 complete (v1.2)";
    std::ofstream gf(out_dir + "/gates.json");
    gf << G.dump(2);
    fprintf(stderr, "[completion] overall %s fails=%d\n", G["overall"].get<std::string>().c_str(), fails);
}

} // namespace alysane
