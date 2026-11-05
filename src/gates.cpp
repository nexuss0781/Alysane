#include "alysane.h"
#include <iomanip>
#include <map>
#include <omp.h>

namespace alysane {
static double spearman(const std::vector<double>& a, const std::vector<double>& b) {
    int n = (int)a.size();
    std::vector<int> ia(n), ib(n);
    std::iota(ia.begin(), ia.end(), 0);
    std::iota(ib.begin(), ib.end(), 0);
    std::sort(ia.begin(), ia.end(), [&](int x, int y) { return a[x] < a[y]; });
    std::sort(ib.begin(), ib.end(), [&](int x, int y) { return b[x] < b[y]; });
    std::vector<double> ra(n), rb(n);
    for (int i = 0; i < n; i++) {
        ra[ia[i]] = i;
        rb[ib[i]] = i;
    }
    double ma = 0, mb = 0;
    for (int i = 0; i < n; i++) {
        ma += ra[i];
        mb += rb[i];
    }
    ma /= n;
    mb /= n;
    double num = 0, da = 0, db = 0;
    for (int i = 0; i < n; i++) {
        double x = ra[i] - ma, y = rb[i] - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    return num / std::sqrt(da * db);
}
static double auroc(const std::vector<double>& pos, const std::vector<double>& neg) {
    double s = 0;
    for (double p : pos)
        for (double n : neg)
            s += (p > n ? 1.0 : p == n ? 0.5 : 0.0);
    return s / (pos.size() * neg.size());
}
static float euclid(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return (float)std::sqrt(s);
}

void run_parity(const std::string& model_dir, const std::string& tokens, const std::string& out) {
    auto w = load_knowledge(model_dir);
    auto c = read_tokens(tokens);
    int N = c.N, n = c.n, V = w.V;
    std::vector<float> logits((size_t)N * n * V);
    std::vector<float> nll(N);
    std::vector<int> top1((size_t)N * n);
    for (int i = 0; i < N; i++) {
        std::vector<int32_t> ids(c.ids.begin() + (size_t)i * n, c.ids.begin() + (size_t)(i + 1) * n);
        auto r = forward(w, ids, true, true, true);
        std::copy(r.all_logits.begin(), r.all_logits.end(), logits.begin() + (size_t)i * n * V);
        nll[i] = r.nll;
        for (int t = 0; t < n; t++) {
            const float* z = r.all_logits.data() + (size_t)t * V;
            top1[(size_t)i * n + t] = (int)(std::max_element(z, z + V) - z);
        }
        fprintf(stderr, "[parity] %d/%d nll %.6f\n", i + 1, N, r.nll);
    }
    std::ofstream f(out, std::ios::binary);
    f.write((char*)&N, 4);
    f.write((char*)&n, 4);
    f.write((char*)&V, 4);
    f.write((char*)logits.data(), (std::streamsize)logits.size() * 4);
    f.write((char*)nll.data(), N * 4);
    f.write((char*)top1.data(), (std::streamsize)top1.size() * 4);
    json j = {{"N", N}, {"n", n}, {"V", V}, {"sha256", w.sha256}, {"params", w.param_count}};
    std::ofstream m(out + ".json");
    m << j.dump(2);
}

void run_gates(const std::string& model_dir, const std::string& data_dir, const std::string& out_dir) {
    std::filesystem::create_directories(out_dir);
    json G;
    G["project"] = "Alysane";
    G["component"] = "C2+C3 pilot gate";
    G["status_legend"] = {{"P", "proven/classical under stated assumptions"},
                          {"H", "hypothesis"},
                          {"E", "measured"},
                          {"F", "failed gate"}};
    auto w = load_knowledge(model_dir);
    G["knowledge"] = {{"V", w.V},          {"d", w.d},     {"L", w.L},
                      {"H", w.H},          {"KVH", w.KVH}, {"params", w.param_count},
                      {"sha256", w.sha256}};
    auto train = read_tokens(data_dir + "/train.bin");
    auto hold = read_tokens(data_dir + "/holdout.bin");
    int N = train.N, n = train.n, V = w.V;
    std::vector<Trace> traces;
    std::vector<std::vector<float>> probs;
    auto t0 = std::chrono::steady_clock::now();
    if (std::getenv("ALYSANE_REUSE")) {
        fprintf(stderr, "[gates] reusing cached traces/probs\n");
        traces = read_traces(out_dir + "/train_traces.bin", w.L, w.d, w.H);
        probs = read_probs(out_dir + "/train_probs.bin", N, V);
    } else
        for (int i = 0; i < N; i++) {
            std::vector<int32_t> ids(train.ids.begin() + (size_t)i * n,
                                     train.ids.begin() + (size_t)(i + 1) * n);
            auto r = forward(w, ids, false, true, false);
            traces.push_back(std::move(r.trace));
            probs.push_back(std::move(r.last_probs));
            if ((i + 1) % 32 == 0)
                fprintf(stderr, "[trace] %d/%d\n", i + 1, N);
        }
    auto t1 = std::chrono::steady_clock::now();
    double trace_sec = std::chrono::duration<double>(t1 - t0).count();
    write_traces(out_dir + "/train_traces.bin", traces, w.L, w.d, w.H);
    write_probs(out_dir + "/train_probs.bin", probs);
    auto D = pairwise_tv(probs, V);
    auto t2 = std::chrono::steady_clock::now();
    double dist_sec = std::chrono::duration<double>(t2 - t1).count();
    int min_size_target = std::max(8, N / 20);
    auto a1 = measure_a1(D, min_size_target, 16, 20, 12345);
    G["timing"] = {
        {"train_forward_sec", trace_sec}, {"pairwise_tv_sec", dist_sec}, {"N", N}, {"n", n}, {"threads", 2}};
    G["A1"] = {{"strict_separated", a1.strict_separated},
               {"useful", a1.useful},
               {"min_size_target", min_size_target},
               {"best",
                {{"q", a1.best.q},
                 {"eps_in", a1.best.eps_in},
                 {"eps_out", a1.best.eps_out},
                 {"width", a1.best.width},
                 {"ratio", a1.best.ratio},
                 {"min_size", a1.best.min_size},
                 {"max_size", a1.best.max_size}}},
               {"bootstrap", a1.bootstrap},
               {"curve", json::array()}};
    for (auto& c : a1.curve)
        G["A1"]["curve"].push_back({{"q", c.q},
                                    {"eps_in", c.eps_in},
                                    {"eps_out", c.eps_out},
                                    {"width", c.width},
                                    {"ratio", c.ratio},
                                    {"min_size", c.min_size},
                                    {"max_size", c.max_size}});
    bool a1_pass = a1.useful && a1.bootstrap.value("mean_ari", 0.0) >= 0.8;
    G["gates"]["G3_A1"] = a1_pass ? "PASS" : "FAIL";
    if (std::getenv("ALYSANE_A1_ONLY")) {
        G["overall"] = a1_pass ? "PASS" : "FAIL";
        G["ready_for_next_component"] = a1_pass;
        std::ofstream gf(out_dir + "/gates.json");
        gf << G.dump(2);
        fprintf(stderr, "[a1-only] overall %s\n", G["overall"].get<std::string>().c_str());
        return;
    }
    // Lemma-1 audit (corrected safe tau = rho+rho').
    std::mt19937_64 rng(777);
    std::uniform_int_distribution<int> ud(0, N - 1);
    int audit = 2000, viol = 0, viol_half = 0;
    double max_err = 0, max_tau = 0;
    for (int s = 0; s < audit; s++) {
        int i = ud(rng), j = ud(rng);
        if (i == j) {
            s--;
            continue;
        }
        float exact = D(i, j), hat = topk_tv(traces[i], traces[j], V), tau = topk_tau(traces[i], traces[j]);
        double err = std::fabs(exact - hat);
        max_err = std::max(max_err, err);
        max_tau = std::max(max_tau, (double)tau);
        if (err > tau + 1e-5)
            viol++;
        if (err > 0.5 * tau + 1e-5)
            viol_half++;
    }
    G["trace_audit"] = {
        {"pairs", audit},
        {"max_abs_error", max_err},
        {"max_tau_corrected", max_tau},
        {"violations_corrected", viol},
        {"violations_original_half_tau", viol_half},
        {"bound", "|TV - hatTV| <= rho_i + rho_j (safe); original half-sum proof not accepted"}};
    G["gates"]["G2_trace_tau"] = viol == 0 ? "PASS" : "FAIL";
    // Landmarks / psi.
    auto landmarks = farthest_landmarks(D, 128, 12345);
    auto psi = psi_embedding(probs, landmarks, V);
    float cov = 0;
    for (int i = 0; i < N; i++) {
        float m = 1e30f;
        for (int l : landmarks)
            m = std::min(m, D(i, l));
        cov = std::max(cov, m);
    }
    G["landmarks"] = {{"M", (int)landmarks.size()}, {"max_coverage_radius_train", cov}};
    // Train self weights.
    fprintf(stderr, "[gates] training self\n");
    auto S = train_self(traces, D, probs, a1, 256, 80, 12345);
    fprintf(stderr, "[gates] training energy\n");
    S = train_energy(std::move(S), traces, 200, 12345);
    fprintf(stderr, "[gates] saving self\n");
    S.save(out_dir + "/self_weights.alys");
    fprintf(stderr, "[gates] loading self\n");
    auto S2 = SelfKnowledgeWeights::load(out_dir + "/self_weights.alys");
    fprintf(stderr, "[gates] roundtrip encode\n");
    double enc_diff = 0, e_diff = 0;
    for (int i = 0; i < std::min(N, 32); i++) {
        auto z = S.encode(traces[i]);
        auto z2 = S2.encode(traces[i]);
        enc_diff = std::max(enc_diff, (double)euclid(z, z2));
        e_diff = std::max(e_diff, (double)std::fabs(S.energy(z) - S2.energy(z2)));
    }
    fprintf(stderr, "[gates] roundtrip done\n");
    G["gates"]["G7_serialization"] = (enc_diff < 1e-5 && e_diff < 1e-4) ? "PASS" : "FAIL";
    G["serialization"] = {{"max_encode_l2_diff", enc_diff}, {"max_energy_diff", e_diff}};
    // Holdout forward + audits.
    fprintf(stderr, "[gates] holdout forward\n");
    int NH = hold.N;
    std::vector<Trace> htr;
    std::vector<std::vector<float>> hp;
    for (int i = 0; i < NH; i++) {
        std::vector<int32_t> ids(hold.ids.begin() + (size_t)i * n, hold.ids.begin() + (size_t)(i + 1) * n);
        auto r = forward(w, ids, false, true, false);
        htr.push_back(std::move(r.trace));
        hp.push_back(std::move(r.last_probs));
        if ((i + 1) % 16 == 0)
            fprintf(stderr, "[hold] %d/%d\n", i + 1, NH);
    }
    fprintf(stderr, "[gates] holdout psi\n");
    auto hpsi = psi_embedding_to(hp, S.landmark_probs, V);
    fprintf(stderr, "[gates] holdout encode\n");
    std::vector<std::vector<float>> hz(NH);
    for (int i = 0; i < NH; i++) {
        hz[i] = S.encode(htr[i]);
        if ((i + 1) % 16 == 0)
            fprintf(stderr, "[henc] %d/%d\n", i + 1, NH);
    }
    fprintf(stderr, "[gates] holdout audits\n");
    std::vector<double> db, dz, dp;
    int psi_viol = 0;
    double psi_lower_viol = 0;
    for (int i = 0; i < NH; i++)
        for (int j = i + 1; j < NH; j++) {
            float b = tv_distance(hp[i].data(), hp[j].data(), V);
            float z = euclid(hz[i], hz[j]);
            float p = euclid(hpsi[i], hpsi[j]);
            db.push_back(b);
            dz.push_back(z);
            dp.push_back(p);
            if (p > std::sqrt((float)landmarks.size()) * b + 1e-4)
                psi_viol++;
            float lower = std::max(0.f, b - 2 * cov);
            if (p < lower - 1e-4)
                psi_lower_viol++;
        }
    double rho_z = spearman(db, dz), rho_p = spearman(db, dp);
    G["embedding_audit"] = {
        {"holdout_pairs", (int)db.size()},
        {"spearman_z_behavior", rho_z},
        {"spearman_psi_behavior", rho_p},
        {"psi_upper_violations", psi_viol},
        {"psi_lower_violations", psi_lower_viol},
        {"note", "psi bounds use train coverage radius as pilot proxy; holdout coverage not guaranteed"}};
    G["gates"]["G4_embedding"] = (psi_viol == 0 && psi_lower_viol == 0) ? "PASS" : "FAIL";
    G["embedding_audit"]["learned_spearman_status"] = rho_z >= 0.6 ? "strong" : "weak_pilot_observation";
    // Routing partition: A1 best if useful, else largest MST gap diagnostic.
    std::vector<int> lab;
    float eps_in = 0, eps_out = 0;
    int q = 0;
    if (a1.best.q) {
        lab = a1.best.labels;
        eps_in = a1.best.eps_in;
        eps_out = a1.best.eps_out;
        q = a1.best.q;
    } else {
        auto m = prim_mst(D);
        std::vector<int> ord(m.w.size());
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return m.w[a] < m.w[b]; });
        int best = -1;
        float bg = -1;
        for (int qq = 2; qq <= std::min(16, N - 1); qq++) {
            int keep = N - qq;
            float gap = m.w[ord[keep]] - m.w[ord[keep - 1]];
            if (gap > bg) {
                bg = gap;
                best = qq;
            }
        }
        if (best > 0) {
            lab = components_at(m, N, 0.5f * (m.w[ord[N - best - 1]] + m.w[ord[N - best]]));
            eps_in = m.w[ord[N - best - 1]];
            eps_out = m.w[ord[N - best]];
            q = best;
        }
    }
    std::vector<float> diam(q, 0);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (!lab.empty() && lab[i] == lab[j])
                diam[lab[i]] = std::max(diam[lab[i]], D(i, j));
    auto route = [&](const std::vector<float>& p) {
        int best = -1;
        float r = 1e30f;
        for (int i = 0; i < N; i++) {
            float d = tv_distance(p.data(), probs[i].data(), V);
            if (d < r) {
                r = d;
                best = i;
            }
        }
        int dom = lab.empty() ? 0 : lab[best];
        float second = 1e30f;
        for (int i = 0; i < N; i++)
            if (!lab.empty() && lab[i] != dom)
                second = std::min(second, tv_distance(p.data(), probs[i].data(), V));
        float bound = r + (lab.empty() ? 0 : diam[dom]);
        int cert_viol = 0;
        for (int i = 0; i < N; i++)
            if (!lab.empty() && lab[i] == dom) {
                float d = tv_distance(p.data(), probs[i].data(), V);
                if (d > bound + 1e-4)
                    cert_viol++;
            }
        return json{{"nearest", best},
                    {"domain", dom},
                    {"r", r},
                    {"second", second},
                    {"margin", second - r},
                    {"bound", bound},
                    {"certificate_violations", cert_viol}};
    };
    json natural = json::array();
    std::vector<double> nat_r, nat_m, nat_e;
    int cert_viol = 0;
    for (int i = 0; i < NH; i++) {
        auto j = route(hp[i]);
        natural.push_back(j);
        cert_viol += j["certificate_violations"].get<int>();
        nat_r.push_back(j["r"]);
        nat_m.push_back(j["margin"]);
        nat_e.push_back(S.energy(hz[i]));
    }
    // Synthetic shifts from first 32 holdout contexts: 30% random token replacement + local shuffle.
    int NS = std::min(32, NH);
    std::vector<double> syn_r, syn_m, syn_e;
    json synthetic = json::array();
    std::mt19937_64 srng(999);
    std::uniform_int_distribution<int> tok(0, V - 1), pos(0, n - 1);
    std::uniform_real_distribution<float> ur(0, 1);
    for (int i = 0; i < NS; i++) {
        std::vector<int32_t> ids(hold.ids.begin() + (size_t)i * n, hold.ids.begin() + (size_t)(i + 1) * n);
        for (int t = 0; t < n; t++)
            if (ur(srng) < 0.3f)
                ids[t] = tok(srng);
        for (int t = 0; t + 1 < n; t++)
            if (ur(srng) < 0.2f)
                std::swap(ids[t], ids[t + 1]);
        auto r = forward(w, ids, false, true, false);
        auto z = S.encode(r.trace);
        auto j = route(r.last_probs);
        synthetic.push_back(j);
        cert_viol += j["certificate_violations"].get<int>();
        syn_r.push_back(j["r"]);
        syn_m.push_back(j["margin"]);
        syn_e.push_back(S.energy(z));
    }
    double auc_r = auroc(syn_r, nat_r), auc_e = auroc(syn_e, nat_e);
    double med_nat_r = 0, med_syn_r = 0;
    {
        auto a = nat_r, b = syn_r;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        med_nat_r = a[a.size() / 2];
        med_syn_r = b[b.size() / 2];
    }
    G["routing"] = {{"partition_q", q},
                    {"eps_in", eps_in},
                    {"eps_out", eps_out},
                    {"certificate_violations", cert_viol},
                    {"median_natural_r", med_nat_r},
                    {"median_synthetic_r", med_syn_r},
                    {"auroc_distance_syn_vs_nat", auc_r},
                    {"auroc_energy_syn_vs_nat", auc_e},
                    {"natural", natural},
                    {"synthetic", synthetic},
                    {"theorem", "corrected behavioral certificate: d(x,y)<=d(x,x*)+diam(D); latent-only "
                                "routing is not certified without distortion"}};
    double med_nat_m = 0;
    {
        auto a = nat_m;
        std::sort(a.begin(), a.end());
        med_nat_m = a[a.size() / 2];
    }
    G["routing"]["median_natural_margin"] = med_nat_m;
    G["gates"]["G5_routing"] = cert_viol == 0 ? "PASS" : "FAIL";
    G["gates"]["G6_boundary"] = (auc_e >= 0.6 && med_nat_m > 0) ? "PASS" : "FAIL";
    // Knowledge manifest (two-store contract).
    json km = {
        {"format", "alysane-knowledge-manifest-v1"},
        {"source", "HuggingFaceTB/SmolLM-135M"},
        {"sha256", w.sha256},
        {"params", w.param_count},
        {"config",
         {{"V", w.V},
          {"d", w.d},
          {"L", w.L},
          {"H", w.H},
          {"KVH", w.KVH},
          {"FF", w.FF},
          {"head_dim", w.head_dim},
          {"rms_eps", w.rms_eps},
          {"rope_theta", w.rope_theta}}},
        {"note",
         "pilot retargeted execution core; architectural swap beyond exact Llama computation not claimed"}};
    std::ofstream kmf(out_dir + "/knowledge_manifest.json");
    kmf << km.dump(2);
    // Merge external C1 parity oracle if present.
    {
        std::ifstream pf(out_dir + "/parity.json");
        if (pf) {
            json p;
            pf >> p;
            G["parity"] = p;
            G["gates"]["G1_parity"] = p.value("PASS", false) ? "PASS" : "FAIL";
        } else
            G["gates"]["G1_parity"] = "FAIL";
    }
    G["reuse_cached_traces"] = (bool)std::getenv("ALYSANE_REUSE");
    int fails = 0;
    for (auto& kv : G["gates"].items())
        if (kv.value() != "PASS")
            fails++;
    G["overall"] = fails == 0 ? "PASS" : "FAIL";
    G["fail_count"] = fails;
    G["ready_for_next_component"] = fails == 0;
    std::ofstream gf(out_dir + "/gates.json");
    gf << G.dump(2);
    fprintf(stderr, "[gates] overall %s fails %d\n", G["overall"].get<std::string>().c_str(), fails);
}

void run_selftest() {
    int fails = 0;
    auto check = [&](bool ok, const std::string& name) {
        fprintf(stderr, "[selftest] %s %s\n", ok ? "PASS" : "FAIL", name.c_str());
        if (!ok)
            fails++;
    };
    // TV metric and corrected trace bound on random distributions.
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> u(0, 1);
    int V = 1000;
    auto randp = [&]() {
        std::vector<float> p(V);
        double s = 0;
        for (auto& x : p) {
            x = std::pow(u(rng), 4);
            s += x;
        }
        for (auto& x : p)
            x /= s;
        return p;
    };
    auto a = randp(), b = randp(), c = randp();
    float ab = tv_distance(a.data(), b.data(), V), bc = tv_distance(b.data(), c.data(), V),
          ac = tv_distance(a.data(), c.data(), V);
    check(ac <= ab + bc + 1e-5, "TV triangle");
    check(std::fabs(ab - tv_distance(b.data(), a.data(), V)) < 1e-6, "TV symmetry");
    // A1 synthetic: three separated cliques in 1D behavior proxy.
    int N = 30;
    DistMatrix D;
    D.N = N;
    D.d.assign(N * N, 0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float x = i / 10.f, y = j / 10.f;
            float d = std::fabs(x - y);
            if (i / 10 == j / 10)
                d = 0.05f * std::abs((i % 10) - (j % 10)) / 10.f;
            D.at(i, j) = d;
        }
    // overwrite with exact block distances
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int bi = i / 10, bj = j / 10;
            D.at(i, j) =
                bi == bj ? 0.01f * std::abs((i % 10) - (j % 10)) / 10.f : 0.5f + 0.1f * std::abs(bi - bj);
        }
    auto a1 = measure_a1(D, 5, 8, 10, 42);
    check(a1.best.q == 3 && a1.useful, "A1 synthetic recovery");
    check(std::fabs(a1.best.eps_in - 0.001f) < 1e-4, "A1 eps_in (MST connectivity threshold)");
    // components theorem
    auto m = prim_mst(D);
    auto lab = components_at(m, N, 0.01f);
    int q = 1 + *std::max_element(lab.begin(), lab.end());
    check(q == 3, "components at eps_in");
    // routing certificate synthetic
    std::vector<float> diam(3, 0);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (lab[i] == lab[j])
                diam[lab[i]] = std::max(diam[lab[i]], D(i, j));
    bool cert = true;
    for (int x = 0; x < N; x++) {
        int best = 0;
        float r = 1e30f;
        for (int i = 0; i < N; i++)
            if (D(x, i) < r) {
                r = D(x, i);
                best = i;
            }
        float bound = r + diam[lab[best]];
        for (int i = 0; i < N; i++)
            if (lab[i] == lab[best] && D(x, i) > bound + 1e-5)
                cert = false;
    }
    check(cert, "routing certificate");
    // energy gradient finite difference on a tiny conservative network.
    SelfKnowledgeWeights S;
    S.out_dim = 4;
    S.energy_hidden = 3;
    S.energy_W = {0.1f, -0.2f, 0.3f, 0.4f, 0.5f, -0.1f, 0.2f, -0.3f, 0.4f, 0.1f, -0.4f, 0.2f};
    S.energy_b = {0.1f, -0.2f, 0.3f};
    S.energy_a = {0.5f, -0.3f, 0.2f};
    std::vector<float> z = {0.2f, -0.1f, 0.4f, 0.3f};
    auto g = S.energy_grad(z);
    float h = 1e-3f;
    double ge = 0;
    for (int i = 0; i < 4; i++) {
        auto zp = z, zm = z;
        zp[i] += h;
        zm[i] -= h;
        float num = (S.energy(zp) - S.energy(zm)) / (2 * h);
        ge = std::max(ge, (double)std::fabs(num - g[i]));
    }
    check(ge < 1e-3, "energy grad finite diff");
    // serialization roundtrip
    S.in_dim = 2;
    S.hidden = 2;
    S.mean = {0, 0, 0, 0, 0, 0};
    S.std = {1, 1};
    S.W1 = {0.1f, 0.2f, -0.1f, 0.3f};
    S.b1 = {0, 0};
    S.W2 = {0.1f, -0.2f, 0.3f, 0.4f};
    S.b2 = {0, 0, 0, 0};
    S.white.assign(16, 0);
    for (int i = 0; i < 4; i++)
        S.white[i * 4 + i] = 1;
    S.landmarks = {0};
    S.landmark_probs = {{1, 0}};
    S.meta = {{"t", 1}};
    S.save("/tmp/alys_selftest.alys");
    auto T = SelfKnowledgeWeights::load("/tmp/alys_selftest.alys");
    check(T.W1 == S.W1 && T.meta == S.meta, "self serialization");
    if (fails)
        throw std::runtime_error("selftest failures: " + std::to_string(fails));
}
} // namespace alysane
