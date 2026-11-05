// Full-scale stress harness: reproduces the post-trace gate path at production
// dimensions (N=256, V=49152) without needing model weights, so it runs under
// ASan/UBSan in CI and locally.
#include "alysane.h"
#include <cstdio>

int main(int argc, char** argv) {
    using namespace alysane;
    int N = argc > 1 ? std::atoi(argv[1]) : 256;
    int V = argc > 2 ? std::atoi(argv[2]) : 49152;
    int L = 30, d = 576, H = 9;
    fprintf(stderr, "[stress] N=%d V=%d\n", N, V);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<float> u(0, 1);
    std::vector<std::vector<float>> probs(N, std::vector<float>(V));
    std::vector<Trace> traces(N);
    for (int i = 0; i < N; i++) {
        // Sparse-ish peaked distributions, like real next-token outputs.
        double s = 0;
        for (int v = 0; v < V; v++) {
            float x = std::pow(u(rng), 40.f);
            probs[i][v] = x;
            s += x;
        }
        for (int v = 0; v < V; v++) probs[i][v] /= (float)s;
        Trace& t = traces[i];
        t.pooled.assign((size_t)L * 2 * d, 0.f);
        for (auto& x : t.pooled) x = u(rng) * 10.f - 5.f;
        t.attn_entropy.assign(H, 1.f);
        t.topk_ids.resize(TOPK);
        t.topk_probs.resize(TOPK);
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + TOPK, idx.end(),
                          [&](int a, int b) { return probs[i][a] > probs[i][b]; });
        double sum = 0;
        for (int k = 0; k < TOPK; k++) {
            t.topk_ids[k] = idx[k];
            t.topk_probs[k] = probs[i][idx[k]];
            sum += t.topk_probs[k];
        }
        t.tail_mass = (float)std::max(0.0, 1.0 - sum);
        t.entropy = 2.5f;
    }
    fprintf(stderr, "[stress] traces built\n");

    auto D = pairwise_tv(probs, V);
    fprintf(stderr, "[stress] pairwise_tv done\n");
    auto a1 = measure_a1(D, std::max(8, N / 20), 16, 20, 12345);
    fprintf(stderr, "[stress] measure_a1 done q=%d useful=%d\n", a1.best.q, (int)a1.useful);
    auto lm = farthest_landmarks(D, 128, 12345);
    fprintf(stderr, "[stress] landmarks done\n");
    auto psi = psi_embedding(probs, lm, V);
    fprintf(stderr, "[stress] psi done\n");
    auto pers = persistence0(D, 20, 4242, 0.05f);
    fprintf(stderr, "[stress] persistence done sig=%d\n", pers.n_significant);
    auto reg = soft_regions(D, farthest_landmarks(D, 32, 12345), 0.15f);
    fprintf(stderr, "[stress] soft regions done K=%d\n", reg.K);
    auto mst = prim_mst(D);
    auto U = ultrametric(mst, N);
    fprintf(stderr, "[stress] ultrametric done\n");
    auto S = train_self(traces, D, probs, a1, 256, 3, 12345);
    fprintf(stderr, "[stress] train_self done\n");
    S = train_energy(std::move(S), traces, 3, 12345);
    fprintf(stderr, "[stress] train_energy done\n");
    S.save("/tmp/stress_self.alys");
    auto S2 = SelfKnowledgeWeights::load("/tmp/stress_self.alys");
    auto z = S.encode(traces[0]), z2 = S2.encode(traces[0]);
    double diff = 0;
    for (size_t i = 0; i < z.size(); i++) diff = std::max(diff, (double)std::fabs(z[i] - z2[i]));
    fprintf(stderr, "[stress] roundtrip diff=%.3g\n", diff);
    fprintf(stderr, "STRESS OK\n");
    return 0;
}
