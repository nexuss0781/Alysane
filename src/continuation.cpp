// §1.3 self-generated outcome signals: rollouts, self-agreement, continuation
// consistency. All signals are computed from K alone; no labels, no human input.
#include "alysane.h"
#include <omp.h>

namespace alysane {

static uint64_t fnv1a(const int32_t* v, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        uint32_t x = (uint32_t)v[i];
        for (int b = 0; b < 4; b++) {
            h ^= (x >> (8 * b)) & 0xFF;
            h *= 1099511628211ULL;
        }
    }
    return h;
}

std::vector<float> continuation_features(const std::vector<int32_t>& seq) {
    // Spectrum kernel over n-grams of order 1..3, signed-hash projected to a
    // fixed width, then L2 normalised so k(s,s) = 1.
    std::vector<float> f(CONT_FEATURE_DIM, 0.f);
    int T = (int)seq.size();
    for (int order = 1; order <= 3; order++) {
        for (int i = 0; i + order <= T; i++) {
            uint64_t h = fnv1a(seq.data() + i, order);
            int idx = (int)(h % (uint64_t)CONT_FEATURE_DIM);
            float sign = (h & (1ULL << 63)) ? -1.f : 1.f;
            f[idx] += sign;
        }
    }
    double norm = 0;
    for (float x : f) norm += (double)x * x;
    norm = std::sqrt(norm);
    if (norm > 1e-12)
        for (auto& x : f) x = (float)(x / norm);
    return f;
}

std::vector<float> continuation_mean(const RolloutSet& rs) {
    std::vector<float> mu(CONT_FEATURE_DIM, 0.f);
    if (rs.seq.empty()) return mu;
    for (const auto& s : rs.seq) {
        auto f = continuation_features(s);
        for (int i = 0; i < CONT_FEATURE_DIM; i++) mu[i] += f[i];
    }
    for (auto& x : mu) x /= (float)rs.seq.size();
    return mu;
}

float continuation_distance(const std::vector<float>& a, const std::vector<float>& b) {
    // Empirical MMD with the normalised spectrum kernel. Bounded in [0, 2].
    double s = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = (double)a[i] - b[i];
        s += d * d;
    }
    return (float)std::sqrt(std::max(0.0, s));
}

float continuation_consistency(const RolloutSet& a, const RolloutSet& b) {
    // kappa(x,y) = mean_{i,j} k(s_i, t_j) = <mu_x, mu_y>.
    auto ma = continuation_mean(a), mb = continuation_mean(b);
    double s = 0;
    for (size_t i = 0; i < ma.size(); i++) s += (double)ma[i] * mb[i];
    return (float)s;
}

static float dist_entropy(const std::vector<float>& p) {
    double h = 0;
    for (float x : p)
        if (x > 0) h -= (double)x * std::log((double)x);
    return (float)h;
}

static int sample_categorical(const std::vector<float>& probs, float temperature, std::mt19937_64& rng) {
    int V = (int)probs.size();
    std::vector<double> cum(V);
    double s = 0;
    if (std::fabs(temperature - 1.f) < 1e-6f) {
        for (int i = 0; i < V; i++) {
            s += probs[i];
            cum[i] = s;
        }
    } else {
        float it = 1.f / std::max(1e-3f, temperature);
        for (int i = 0; i < V; i++) {
            s += std::pow(std::max(1e-30f, probs[i]), it);
            cum[i] = s;
        }
    }
    if (s <= 0) return (int)(rng() % (uint64_t)V);
    std::uniform_real_distribution<double> u(0, s);
    double r = u(rng);
    int lo = (int)(std::lower_bound(cum.begin(), cum.end(), r) - cum.begin());
    return std::min(lo, V - 1);
}

RolloutSet rollouts(const KnowledgeWeights& w, const std::vector<int32_t>& ids, const RolloutConfig& cfg,
                    uint64_t stream_seed) {
    int n = (int)ids.size();
    if (cfg.R < 1 || cfg.T < 1) throw std::runtime_error("rollouts: R,T must be >= 1");
    KVCache cache;
    cache.init(w.L, w.KVH, w.head_dim, n + cfg.T);
    auto pre = forward_kv(w, ids, 0, true, &cache);

    RolloutSet rs;
    rs.seq.assign(cfg.R, {});
    std::mt19937_64 rng(stream_seed);
    double ent_sum = 0;
    int ent_cnt = 0;
    for (int r = 0; r < cfg.R; r++) {
        cache.len = n; // reuse the shared prefill; overwrite generated positions
        std::vector<float> p = pre.last_probs;
        ent_sum += pre.trace.entropy;
        ent_cnt++;
        rs.seq[r].reserve(cfg.T);
        for (int t = 0; t < cfg.T; t++) {
            int tok = sample_categorical(p, cfg.temperature, rng);
            rs.seq[r].push_back(tok);
            if (t + 1 < cfg.T) {
                p = decode_step(w, cache, tok);
                ent_sum += dist_entropy(p);
                ent_cnt++;
            }
        }
    }
    rs.mean_entropy = (float)(ent_sum / std::max(1, ent_cnt));

    // Self-agreement: mean pairwise kernel among own continuations.
    std::vector<std::vector<float>> feats(cfg.R);
    for (int r = 0; r < cfg.R; r++) feats[r] = continuation_features(rs.seq[r]);
    double agree = 0;
    int pairs = 0, exact = 0;
    for (int i = 0; i < cfg.R; i++)
        for (int j = i + 1; j < cfg.R; j++) {
            double s = 0;
            for (int k = 0; k < CONT_FEATURE_DIM; k++) s += (double)feats[i][k] * feats[j][k];
            agree += s;
            pairs++;
            if (rs.seq[i] == rs.seq[j]) exact++;
        }
    rs.self_agreement = pairs ? (float)(agree / pairs) : 1.f;
    rs.exact_agreement = pairs ? (float)exact / pairs : 1.f;
    return rs;
}

} // namespace alysane
