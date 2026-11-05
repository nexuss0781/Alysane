// Theorem 3 / reshape: empirical local logit-Lipschitz measurement and the
// resulting certified TV-drift bound. We perturb one weight tensor by a known
// relative norm and measure the induced logit and behavioural drift.
#include "alysane.h"
#include <omp.h>

namespace alysane {

DriftReport measure_drift(const KnowledgeWeights& w, const std::vector<std::vector<int32_t>>& ctxs,
                          float rel_norm, uint64_t seed) {
    DriftReport R;
    int N = (int)ctxs.size();
    if (N < 2) throw std::runtime_error("measure_drift: need >= 2 contexts");

    // Clone the final-layer down-projection into mutable storage and perturb it.
    const Tensor& target = w.layers.back().down;
    std::vector<float> perturbed(target.ptr, target.ptr + target.numel);
    double base_norm = 0;
    for (int64_t i = 0; i < target.numel; i++) base_norm += (double)perturbed[i] * perturbed[i];
    base_norm = std::sqrt(base_norm);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> delta(target.numel);
    double dn = 0;
    for (int64_t i = 0; i < target.numel; i++) {
        delta[i] = nd(rng);
        dn += (double)delta[i] * delta[i];
    }
    dn = std::sqrt(dn);
    float want = (float)(rel_norm * base_norm);
    double achieved = 0;
    for (int64_t i = 0; i < target.numel; i++) {
        delta[i] *= want / (float)dn;
        perturbed[i] += delta[i];
        achieved += (double)delta[i] * delta[i];
    }
    R.delta_theta_norm = (float)std::sqrt(achieved);

    // Build a shallow copy of the weights with the perturbed tensor swapped in.
    KnowledgeWeights w2;
    w2.V = w.V; w2.d = w.d; w2.L = w.L; w2.H = w.H; w2.KVH = w.KVH; w2.FF = w.FF;
    w2.head_dim = w.head_dim; w2.rms_eps = w.rms_eps; w2.rope_theta = w.rope_theta;
    w2.embed = w.embed; w2.final_norm = w.final_norm; w2.layers = w.layers;
    w2.param_count = w.param_count;
    Tensor pt = target;
    pt.ptr = perturbed.data();
    w2.layers.back().down = pt;

    std::vector<std::vector<float>> p0(N), p1(N);
    for (int i = 0; i < N; i++) {
        auto a = forward(w, ctxs[i], false, false, true);
        auto b = forward(w2, ctxs[i], false, false, true);
        p0[i] = a.last_probs;
        p1[i] = b.last_probs;
        float mx = 0;
        for (int v = 0; v < w.V; v++) mx = std::max(mx, std::fabs(a.last_logits[v] - b.last_logits[v]));
        R.max_logit_inf = std::max(R.max_logit_inf, mx);
    }
    R.empirical_L_z = R.delta_theta_norm > 0 ? R.max_logit_inf / R.delta_theta_norm : 0.f;
    R.predicted_bound = 2.f * R.max_logit_inf;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            float d0 = tv_distance(p0[i].data(), p0[j].data(), w.V);
            float d1 = tv_distance(p1[i].data(), p1[j].data(), w.V);
            float ch = std::fabs(d1 - d0);
            R.max_tv_change = std::max(R.max_tv_change, ch);
            if (ch > R.predicted_bound + 1e-4f) R.bound_violations++;
            R.pairs++;
        }
    return R;
}

} // namespace alysane
