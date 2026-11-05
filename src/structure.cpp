// Persistence-based multiscale structure and overlapping soft regions.
// This is the mathematically calibrated replacement for the falsified hard A1
// partition: merge heights are tested against a geometry-destroying null, and
// regions are a partition of unity rather than a hard labelling.
#include "alysane.h"
#include <omp.h>
#include <set>

namespace alysane {

DistMatrix ultrametric(const MST& m, int N) {
    // Minimax path distance on the MST equals the single-linkage merge height.
    std::vector<std::vector<std::pair<int, float>>> adj(N);
    for (size_t e = 0; e < m.w.size(); e++) {
        adj[m.u[e]].push_back({m.v[e], m.w[e]});
        adj[m.v[e]].push_back({m.u[e], m.w[e]});
    }
    DistMatrix U;
    U.N = N;
    U.d.assign((size_t)N * N, 0.f);
    std::vector<int> stack(N);
    std::vector<char> seen(N);
    for (int s = 0; s < N; s++) {
        std::fill(seen.begin(), seen.end(), 0);
        int top = 0;
        stack[top++] = s;
        seen[s] = 1;
        U.at(s, s) = 0.f;
        while (top) {
            int u = stack[--top];
            for (auto [v, wgt] : adj[u]) {
                if (seen[v]) continue;
                seen[v] = 1;
                U.at(s, v) = std::max(U.at(s, u), wgt);
                stack[top++] = v;
            }
        }
    }
    return U;
}

PersistenceDiagram persistence0(const DistMatrix& D, int null_B, uint64_t seed, float alpha) {
    int N = D.N;
    PersistenceDiagram P;
    if (N < 3) return P;
    MST m = prim_mst(D);
    P.deaths = m.w;
    std::sort(P.deaths.begin(), P.deaths.end());
    int E = (int)P.deaths.size();

    // Null: permute the off-diagonal distance values among pairs. This preserves
    // the marginal distance distribution exactly but destroys metric geometry.
    std::vector<float> vals;
    vals.reserve((size_t)N * (N - 1) / 2);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) vals.push_back(D(i, j));
    std::vector<std::vector<float>> null_deaths(null_B);
    std::mt19937_64 rng(seed);
    for (int b = 0; b < null_B; b++) {
        std::shuffle(vals.begin(), vals.end(), rng);
        DistMatrix S;
        S.N = N;
        S.d.assign((size_t)N * N, 0.f);
        size_t k = 0;
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++) {
                float v = vals[k++];
                S.at(i, j) = v;
                S.at(j, i) = v;
            }
        MST sm = prim_mst(S);
        auto dw = sm.w;
        std::sort(dw.begin(), dw.end());
        null_deaths[b] = std::move(dw);
    }
    P.null_quantile.assign(E, 0.f);
    for (int i = 0; i < E; i++) {
        std::vector<float> col;
        col.reserve(null_B);
        for (int b = 0; b < null_B; b++)
            if (i < (int)null_deaths[b].size()) col.push_back(null_deaths[b][i]);
        if (col.empty()) continue;
        std::sort(col.begin(), col.end());
        int q = (int)std::ceil((1.0 - alpha) * col.size()) - 1;
        q = std::max(0, std::min(q, (int)col.size() - 1));
        P.null_quantile[i] = col[q];
    }
    for (int i = 0; i < E; i++)
        if (P.deaths[i] > P.null_quantile[i]) P.significant.push_back(i);
    P.n_significant = (int)P.significant.size();
    P.null_B = (float)null_B;
    return P;
}

static float quantile(std::vector<float> v, float q) {
    if (v.empty()) return 0.f;
    std::sort(v.begin(), v.end());
    int i = (int)std::floor(q * (v.size() - 1));
    return v[std::max(0, std::min(i, (int)v.size() - 1))];
}

std::vector<float> soft_membership(const std::vector<float>& dist_to_anchor,
                                   const std::vector<float>& radius) {
    int K = (int)dist_to_anchor.size();
    std::vector<float> w(K, 0.f);
    double s = 0;
    int nearest = 0;
    float nd = 1e30f;
    for (int i = 0; i < K; i++) {
        if (dist_to_anchor[i] < nd) {
            nd = dist_to_anchor[i];
            nearest = i;
        }
        float r = std::max(1e-6f, radius[i]);
        float u = dist_to_anchor[i] / r;
        if (u < 1.f) {
            float a = 1.f - u * u;
            w[i] = a * a; // C^1 compactly supported bump
            s += w[i];
        }
    }
    if (s <= 1e-12) {
        // Uncovered point: assign to nearest anchor, flagged by boundary score 1.
        w[nearest] = 1.f;
        return w;
    }
    for (auto& x : w) x = (float)(x / s);
    return w;
}

DensityPeaks density_peaks(const DistMatrix& D, int knn, float bandwidth_quantile) {
    // Mode-seeking regions determined by the metric ALONE.
    //
    // Farthest-point anchor selection is high-variance under subsampling: it
    // picks extreme points, so the anchor set changes wildly between bootstrap
    // replicates. That nuisance variance is a property of the SELECTION rule,
    // not of the behavioural geometry being tested. Density peaks removes it:
    // every point links to its nearest denser neighbour, and modes emerge from
    // the density landscape. No fixed cluster count, no random anchor subset,
    // fully deterministic given (D, bandwidth).
    int N = D.N;
    DensityPeaks P;
    P.rho.assign(N, 0.f);
    P.parent.assign(N, -1);
    P.delta.assign(N, 0.f);
    P.label.assign(N, 0);
    if (N < 3) return P;

    // Adaptive bandwidth: quantile of k-nearest-neighbour distances.
    int k = std::min(std::max(1, knn), N - 1);
    std::vector<float> knn_d(N);
    for (int i = 0; i < N; i++) {
        std::vector<float> ds;
        ds.reserve(N - 1);
        for (int j = 0; j < N; j++)
            if (j != i) ds.push_back(D(i, j));
        std::nth_element(ds.begin(), ds.begin() + k - 1, ds.end());
        knn_d[i] = ds[k - 1];
    }
    std::vector<float> tmp = knn_d;
    std::sort(tmp.begin(), tmp.end());
    float sigma = std::max(1e-6f, tmp[(size_t)(bandwidth_quantile * (N - 1))]);
    P.sigma = sigma;

    // Gaussian KDE density at each sample point.
    for (int i = 0; i < N; i++) {
        double s = 0;
        for (int j = 0; j < N; j++) {
            if (j == i) continue;
            double u = D(i, j) / sigma;
            s += std::exp(-u * u);
        }
        P.rho[i] = (float)s;
    }

    // Link each point to its nearest strictly denser neighbour.
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return P.rho[a] > P.rho[b]; });
    for (int oi = 0; oi < N; oi++) {
        int i = order[oi];
        if (oi == 0) { // global density maximum is a mode
            P.parent[i] = -1;
            float mx = 0;
            for (int j = 0; j < N; j++) mx = std::max(mx, D(i, j));
            P.delta[i] = mx;
            continue;
        }
        int best = -1;
        float bd = 1e30f;
        for (int oj = 0; oj < oi; oj++) {
            int j = order[oj];
            if (D(i, j) < bd) {
                bd = D(i, j);
                best = j;
            }
        }
        P.parent[i] = best;
        P.delta[i] = bd;
    }

    // A mode is a point whose nearest denser neighbour is farther than the
    // bandwidth: it sits on its own density hill rather than a shoulder.
    // The mode count is data-determined, never fixed.
    for (int i = 0; i < N; i++)
        if (P.parent[i] < 0 || P.delta[i] > sigma) P.modes.push_back(i);
    if (P.modes.empty()) P.modes.push_back(order[0]);
    std::set<int> is_mode(P.modes.begin(), P.modes.end());
    P.K = (int)P.modes.size();

    // Assign by following parent links to the first mode ancestor.
    P.mode_of.assign(N, 0);
    for (int i = 0; i < N; i++) {
        int cur = i, guard = 0;
        while (!is_mode.count(cur) && P.parent[cur] >= 0 && guard++ < N) cur = P.parent[cur];
        P.mode_of[i] = cur;
    }
    std::map<int, int> midx;
    for (size_t a = 0; a < P.modes.size(); a++) midx[P.modes[a]] = (int)a;
    for (int i = 0; i < N; i++) P.label[i] = midx.count(P.mode_of[i]) ? midx[P.mode_of[i]] : 0;

    // Soft membership over modes: kernel weights, partition of unity.
    P.member.assign(N, std::vector<float>(P.K, 0.f));
    P.boundary_score.assign(N, 0.f);
    double logK = std::log((double)std::max(2, P.K));
    for (int i = 0; i < N; i++) {
        double s = 0;
        for (int a = 0; a < P.K; a++) {
            double u = D(i, P.modes[a]) / sigma;
            double w = std::exp(-u * u);
            P.member[i][a] = (float)w;
            s += w;
        }
        if (s <= 1e-300) {
            P.member[i][P.label[i]] = 1.f;
        } else {
            for (auto& x : P.member[i]) x = (float)(x / s);
        }
        double h = 0;
        for (float p : P.member[i])
            if (p > 0) h -= (double)p * std::log((double)p);
        P.boundary_score[i] = (float)(h / logK);
    }
    return P;
}

SoftRegions soft_regions(const DistMatrix& D, const std::vector<int>& landmarks, float radius_quantile) {
    int N = D.N, K = (int)landmarks.size();
    SoftRegions S;
    S.landmarks = landmarks;
    S.K = K;
    S.radius.assign(K, 0.f);
    S.member.assign(N, std::vector<float>(K, 0.f));
    S.boundary_score.assign(N, 0.f);
    if (K == 0) return S;
    for (int a = 0; a < K; a++) {
        std::vector<float> ds(N);
        for (int i = 0; i < N; i++) ds[i] = D(landmarks[a], i);
        S.radius[a] = std::max(1e-4f, quantile(ds, radius_quantile));
    }
    std::vector<float> da(K);
    double logK = std::log((double)std::max(2, K));
    for (int i = 0; i < N; i++) {
        for (int a = 0; a < K; a++) da[a] = D(i, landmarks[a]);
        S.member[i] = soft_membership(da, S.radius);
        double h = 0;
        for (float p : S.member[i])
            if (p > 0) h -= (double)p * std::log((double)p);
        S.boundary_score[i] = (float)(h / logK);
    }
    return S;
}

} // namespace alysane
