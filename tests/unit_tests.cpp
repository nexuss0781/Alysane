// Unit tests for the C2+C3 completion modules. No model weights required.
#include "alysane.h"
#include "simd.h"
#include <cstdio>

namespace alysane {

static int g_fail = 0;
static void ck(bool ok, const char* name) {
    fprintf(stderr, "[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) g_fail++;
}

void run_unit_tests() {
    g_fail = 0;

    // ---- continuation features: normalisation, kernel bounds ----
    std::vector<int32_t> s1{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int32_t> s2{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int32_t> s3{99, 100, 101, 102, 103, 104, 105, 106};
    auto f1 = continuation_features(s1), f2 = continuation_features(s2), f3 = continuation_features(s3);
    double n1 = 0;
    for (float x : f1) n1 += (double)x * x;
    ck(std::fabs(n1 - 1.0) < 1e-5, "continuation feature L2 norm = 1");
    ck(continuation_distance(f1, f2) < 1e-6, "identical sequences -> zero continuation distance");
    float d13 = continuation_distance(f1, f3);
    ck(d13 > 0.f && d13 <= 2.f + 1e-5f, "continuation distance in [0,2]");

    // ---- continuation MMD triangle inequality on random sequences ----
    std::mt19937_64 rng(5);
    std::uniform_int_distribution<int> td(0, 500);
    std::vector<std::vector<float>> fs;
    for (int i = 0; i < 12; i++) {
        std::vector<int32_t> s;
        for (int t = 0; t < 8; t++) s.push_back(td(rng));
        fs.push_back(continuation_features(s));
    }
    int tri = 0;
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
            for (int k = 0; k < 12; k++)
                if (continuation_distance(fs[i], fs[k]) >
                    continuation_distance(fs[i], fs[j]) + continuation_distance(fs[j], fs[k]) + 1e-4f)
                    tri++;
    ck(tri == 0, "continuation MMD triangle inequality");

    // ---- RolloutSet agreement / consistency ----
    RolloutSet ra, rb;
    ra.seq = {s1, s1, s1, s1};
    rb.seq = {s3, s3, s3, s3};
    auto ma = continuation_mean(ra), mb = continuation_mean(rb);
    ck(continuation_distance(ma, ma) < 1e-6, "rollout mean self-distance zero");
    float kap = continuation_consistency(ra, rb);
    ck(kap >= -1.0f - 1e-5f && kap <= 1.0f + 1e-5f, "continuation consistency in [-1,1]");
    ck(continuation_consistency(ra, ra) > 0.99f, "self-consistency ~ 1 for identical rollouts");

    // ---- soft membership: partition of unity ----
    std::vector<float> dist{0.1f, 0.5f, 0.9f, 2.0f};
    std::vector<float> rad{0.4f, 0.6f, 1.0f, 0.5f};
    auto mem = soft_membership(dist, rad);
    double s = 0;
    for (float x : mem) s += x;
    ck(std::fabs(s - 1.0) < 1e-5, "soft membership sums to 1");
    bool nonneg = true;
    for (float x : mem) if (x < 0.f) nonneg = false;
    ck(nonneg, "soft membership non-negative");
    // Uncovered point falls back to nearest anchor.
    auto mem_far = soft_membership({5.f, 6.f, 7.f, 8.f}, rad);
    double sf = 0;
    for (float x : mem_far) sf += x;
    ck(std::fabs(sf - 1.0) < 1e-5 && mem_far[0] > 0.99f, "uncovered point -> nearest anchor");

    // ---- soft regions on a synthetic 3-block metric ----
    int N = 30;
    DistMatrix D;
    D.N = N;
    D.d.assign((size_t)N * N, 0.f);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int bi = i / 10, bj = j / 10;
            D.at(i, j) = bi == bj ? 0.01f * std::abs((i % 10) - (j % 10)) : 0.6f + 0.1f * std::abs(bi - bj);
        }
    auto lm = farthest_landmarks(D, 6, 42);
    auto reg = soft_regions(D, lm, 0.25f);
    bool all_sum1 = true;
    for (int i = 0; i < N; i++) {
        double t = 0;
        for (float x : reg.member[i]) t += x;
        if (std::fabs(t - 1.0) > 1e-5) all_sum1 = false;
    }
    ck(all_sum1, "soft_regions partition of unity over all points");
    ck(reg.K == 6 && (int)reg.radius.size() == 6, "soft_regions shape");

    // ---- persistence: separated blocks give significant deaths ----
    auto pers = persistence0(D, 12, 99, 0.05f);
    ck(pers.n_significant >= 1, "persistence detects significant structure in separated blocks");
    ck((int)pers.deaths.size() == N - 1, "persistence death count = N-1");
    bool sorted = true;
    for (size_t i = 1; i < pers.deaths.size(); i++) if (pers.deaths[i] < pers.deaths[i - 1]) sorted = false;
    ck(sorted, "persistence deaths sorted");

    // ---- subdominant ultrametric: U <= D, strong triangle, and maximality ----
    // The single-linkage merge height is the LARGEST ultrametric bounded above by
    // D (subdominant), because the direct edge {i,j} is always a candidate path.
    auto mst = prim_mst(D);
    auto U = ultrametric(mst, N);
    bool below = true, ultra = true, sym = true;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            if (U(i, j) > D(i, j) + 1e-5f) below = false;
            if (std::fabs(U(i, j) - U(j, i)) > 1e-5f) sym = false;
            for (int k = 0; k < N; k++)
                if (U(i, k) > std::max(U(i, j), U(j, k)) + 1e-5f) ultra = false;
        }
    ck(below, "subdominant ultrametric U <= D");
    ck(sym, "ultrametric symmetry");
    ck(ultra, "strong triangle inequality (ultrametric)");
    // Maximality: raising any off-diagonal entry above U while staying <= D must
    // break the strong triangle inequality somewhere.
    bool maximal = true;
    for (int i = 0; i < N && maximal; i++)
        for (int j = i + 1; j < N && maximal; j++) {
            if (D(i, j) - U(i, j) < 1e-4f) continue;
            float mid = 0.5f * (U(i, j) + D(i, j));
            for (int k = 0; k < N; k++) {
                if (k == i || k == j) continue;
                if (mid > std::max(U(i, k), U(k, j)) + 1e-5f) { maximal = true; goto next_pair; }
            }
            maximal = false;
        next_pair:;
        }
    ck(maximal, "subdominant ultrametric is maximal below D");

    // ---- quantisation: top-k TV invariant, bounded hidden-state error ----
    Trace t;
    int L = 3, d = 8;
    t.pooled.resize((size_t)L * 2 * d);
    std::uniform_real_distribution<float> ud(-5.f, 5.f);
    for (auto& x : t.pooled) x = ud(rng);
    t.attn_entropy.assign(4, 0.5f);
    t.topk_ids.resize(TOPK);
    t.topk_probs.resize(TOPK);
    for (int i = 0; i < TOPK; i++) { t.topk_ids[i] = i; t.topk_probs[i] = 0.9f / TOPK; }
    t.tail_mass = 0.1f;
    auto q = quantize_trace(t, L, d);
    auto dq = dequantize_trace(q, L, d);
    double maxerr = 0, maxscale = 0;
    for (int l = 0; l < L; l++) {
        maxscale = std::max(maxscale, (double)q.scale[l]);
        for (int i = 0; i < 2 * d; i++)
            maxerr = std::max(maxerr, (double)std::fabs(t.pooled[(size_t)l * 2 * d + i] - dq.pooled[(size_t)l * 2 * d + i]));
    }
    ck(maxerr <= maxscale / 2 + 1e-6, "int8 dequant error <= half quantisation step");
    ck(q.topk_probs == t.topk_probs && q.topk_ids == t.topk_ids, "quantisation preserves top-k exactly");
    Trace t2 = t;
    for (auto& x : t2.pooled) x = ud(rng);
    auto q2 = quantize_trace(t2, L, d);
    ck(std::fabs(topk_tv(t, t2, 1000) - quantized_topk_tv(q, q2, 1000)) == 0,
       "quantised top-k TV identical to full-precision top-k TV");

    // ---- density peaks: deterministic, no fixed K, partition of unity ----
    {
        auto dp = density_peaks(D, 4, 0.5f);
        ck(dp.K >= 1 && dp.K <= N, "density peaks mode count in [1,N]");
        ck(dp.K == 3, "density peaks recovers 3 emergent modes on 3-block metric");
        bool pu = true, nonneg = true;
        for (int i = 0; i < N; i++) {
            double t = 0;
            for (float x : dp.member[i]) { t += x; if (x < 0.f) nonneg = false; }
            if (std::fabs(t - 1.0) > 1e-4) pu = false;
        }
        ck(pu, "density peaks membership sums to 1");
        ck(nonneg, "density peaks membership non-negative");
        // Determinism: same input -> identical modes and labels.
        auto dp2 = density_peaks(D, 4, 0.5f);
        ck(dp.modes == dp2.modes && dp.label == dp2.label, "density peaks is deterministic");
        // Each mode is its own ancestor.
        bool self_mode = true;
        for (int m : dp.modes) if (dp.mode_of[m] != m) self_mode = false;
        ck(self_mode, "each density mode is its own ancestor");
        // Parent links must strictly increase density (except the global max).
        bool inc = true;
        for (int i = 0; i < N; i++)
            if (dp.parent[i] >= 0 && dp.rho[dp.parent[i]] < dp.rho[i]) inc = false;
        ck(inc, "parent links increase density");
    }

    // ---- portable SIMD kernels agree with scalar reference ----
    {
        std::vector<float> a(1024), b(1024);
        for (int i = 0; i < 1024; i++) { a[i] = ud(rng); b[i] = ud(rng); }
        double ref_dot = 0, ref_l1 = 0;
        for (int i = 0; i < 1024; i++) { ref_dot += (double)a[i] * b[i]; ref_l1 += std::fabs((double)a[i] - b[i]); }
        float sd = simd::dot(a.data(), b.data(), 1024);
        float sl = simd::l1(a.data(), b.data(), 1024);
        ck(std::fabs(sd - ref_dot) < 1e-2 * std::max(1.0, std::fabs(ref_dot)), "SIMD dot matches scalar reference");
        ck(std::fabs(sl - ref_l1) < 1e-3 * std::max(1.0, ref_l1), "SIMD L1 matches scalar reference");
        // Non-multiple-of-16 length exercises the tail loop.
        float sd2 = simd::dot(a.data(), b.data(), 1001);
        double ref2 = 0; for (int i = 0; i < 1001; i++) ref2 += (double)a[i] * b[i];
        ck(std::fabs(sd2 - ref2) < 1e-2 * std::max(1.0, std::fabs(ref2)), "SIMD dot handles ragged tail");
        fprintf(stderr, "[info] SIMD ISA: %s\n", simd::isa_name());
    }

    // ---- KVCache shape / accessors ----
    KVCache c;
    c.init(2, 3, 4, 10);
    ck(c.k.size() == 2u * 10u * 3u * 4u, "KVCache allocation shape");
    c.kp(1, 3)[2] = 7.f;
    ck(c.kp(1, 3)[2] == 7.f, "KVCache accessor");
    ck(c.bytes() == 2u * 2u * 10u * 3u * 4u * sizeof(float), "KVCache byte accounting");

    if (g_fail) throw std::runtime_error("unit test failures: " + std::to_string(g_fail));
    fprintf(stderr, "ALL UNIT TESTS PASSED\n");
}

} // namespace alysane
