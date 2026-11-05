// KV-cached execution core. Mirrors src/model.cpp arithmetic exactly so that
// cached decode is numerically equivalent to a full forward pass (gate G8).
#include "alysane.h"
#include <cblas.h>
#include <immintrin.h>
#include "simd.h"
#include <omp.h>
#include <thread>
extern "C" void openblas_set_num_threads(int);

namespace alysane {

int hw_threads() {
    const char* e = std::getenv("ALYSANE_THREADS");
    int t = e ? std::atoi(e) : 2;
    unsigned hc = std::thread::hardware_concurrency();
    if (hc > 0) t = std::min(t, (int)hc);
    return std::max(1, t);
}

static void rms(int n, int d, const float* x, const float* g, float eps, float* y) {
    for (int t = 0; t < n; t++) {
        const float* xx = x + (size_t)t * d;
        float* yy = y + (size_t)t * d;
        float s = 0;
        for (int i = 0; i < d; i++) s += xx[i] * xx[i];
        float inv = 1.f / std::sqrt(s / d + eps);
        for (int i = 0; i < d; i++) yy[i] = xx[i] * inv * g[i];
    }
}
static void mm(int n, int out, int in, const float* X, const float* W, float* Y) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, out, in, 1.f, X, in, W, in, 0.f, Y, out);
}
static float dotf(const float* a, const float* b, int hd) { return alysane::simd::dot(a, b, hd); }
ForwardResult forward_kv(const KnowledgeWeights& w, const std::vector<int32_t>& ids, int pos_offset,
                         bool want_logits, KVCache* cache) {
    openblas_set_num_threads(hw_threads());
    omp_set_num_threads(hw_threads());
    int n = (int)ids.size(), d = w.d, H = w.H, KV = w.KVH, hd = w.head_dim, FF = w.FF, V = w.V;
    if (n <= 0) throw std::runtime_error("forward_kv: empty context");
    if (pos_offset < 0) throw std::runtime_error("forward_kv: negative position");

    KVCache local;
    if (!cache) {
        if (pos_offset != 0) throw std::runtime_error("forward_kv: null cache requires pos_offset 0");
        local.init(w.L, KV, hd, n);
        cache = &local;
    }
    if (cache->cap < pos_offset + n) throw std::runtime_error("forward_kv: cache capacity exceeded");

    ForwardResult R;
    Trace& tr = R.trace;
    tr.pooled.assign((size_t)w.L * 2 * d, 0.f);
    tr.attn_entropy.assign(H, 0.f);
    tr.topk_ids.resize(TOPK);
    tr.topk_probs.resize(TOPK);

    std::vector<float> X((size_t)n * d), Xn((size_t)n * d), Q((size_t)n * d), K((size_t)n * KV * hd),
        Vv((size_t)n * KV * hd), O((size_t)n * d), proj((size_t)n * d), G((size_t)n * FF), U((size_t)n * FF);
    for (int t = 0; t < n; t++)
        std::copy(w.embed.ptr + (size_t)ids[t] * d, w.embed.ptr + (size_t)(ids[t] + 1) * d,
                  X.begin() + (size_t)t * d);

    // RoPE tables at ABSOLUTE positions pos_offset + t.
    std::vector<float> cs((size_t)n * hd), sn((size_t)n * hd);
    for (int t = 0; t < n; t++)
        for (int i = 0; i < hd / 2; i++) {
            float freq = std::pow(w.rope_theta, -2.f * i / hd);
            float a = (pos_offset + t) * freq;
            cs[(size_t)t * hd + i] = cs[(size_t)t * hd + i + hd / 2] = std::cos(a);
            sn[(size_t)t * hd + i] = sn[(size_t)t * hd + i + hd / 2] = std::sin(a);
        }
    auto rope = [&](float* x, int heads) {
        for (int t = 0; t < n; t++)
            for (int h = 0; h < heads; h++) {
                float* p = x + ((size_t)t * heads + h) * hd;
                for (int i = 0; i < hd / 2; i++) {
                    float c = cs[(size_t)t * hd + i], s = sn[(size_t)t * hd + i];
                    float a = p[i], b = p[i + hd / 2];
                    p[i] = a * c - b * s;
                    p[i + hd / 2] = b * c + a * s;
                }
            }
    };

    std::vector<float> scores((size_t)(pos_offset + n)), attout((size_t)hd);
    for (int l = 0; l < w.L; l++) {
        const auto& lw = w.layers[l];
        rms(n, d, X.data(), lw.input_layernorm.ptr, w.rms_eps, Xn.data());
        mm(n, d, d, Xn.data(), lw.q.ptr, Q.data());
        mm(n, KV * hd, d, Xn.data(), lw.k.ptr, K.data());
        mm(n, KV * hd, d, Xn.data(), lw.v.ptr, Vv.data());
        rope(Q.data(), H);
        rope(K.data(), KV);
        for (int t = 0; t < n; t++) {
            std::copy(K.data() + (size_t)t * KV * hd, K.data() + (size_t)(t + 1) * KV * hd,
                      cache->kp(l, pos_offset + t));
            std::copy(Vv.data() + (size_t)t * KV * hd, Vv.data() + (size_t)(t + 1) * KV * hd,
                      cache->vp(l, pos_offset + t));
        }
        std::fill(O.begin(), O.end(), 0.f);
        float scale = 1.f / std::sqrt((float)hd);
        for (int h = 0; h < H; h++) {
            int kv = h / (H / KV);
            double ent = 0;
            for (int t = 0; t < n; t++) {
                int last = pos_offset + t;
                const float* q = Q.data() + ((size_t)t * H + h) * hd;
                float mx = -1e30f;
                for (int j = 0; j <= last; j++) {
                    float s = dotf(q, cache->kp(l, j) + (size_t)kv * hd, hd) * scale;
                    scores[j] = s;
                    mx = std::max(mx, s);
                }
                float sum = 0;
                for (int j = 0; j <= last; j++) {
                    float e = std::exp(scores[j] - mx);
                    scores[j] = e;
                    sum += e;
                }
                float inv = 1.f / sum;
                float* out = O.data() + ((size_t)t * H + h) * hd;
                for (int j = 0; j <= last; j++) {
                    float p = scores[j] * inv;
                    if (p > 0) ent -= p * std::log(p);
                    const float* v = cache->vp(l, j) + (size_t)kv * hd;
                    for (int i = 0; i < hd; i++) out[i] += p * v[i];
                }
            }
            tr.attn_entropy[h] += (float)(ent / n);
        }
        mm(n, d, d, O.data(), lw.o.ptr, proj.data());
        for (size_t i = 0; i < X.size(); i++) X[i] += proj[i];
        rms(n, d, X.data(), lw.post_attention_layernorm.ptr, w.rms_eps, Xn.data());
        mm(n, FF, d, Xn.data(), lw.gate.ptr, G.data());
        mm(n, FF, d, Xn.data(), lw.up.ptr, U.data());
        for (size_t i = 0; i < G.size(); i++) {
            float g = G[i];
            G[i] = (g / (1.f + std::exp(-g))) * U[i];
        }
        mm(n, d, FF, G.data(), lw.down.ptr, proj.data());
        for (size_t i = 0; i < X.size(); i++) X[i] += proj[i];
        float* mean = tr.pooled.data() + (size_t)l * 2 * d;
        float* mxp = tr.pooled.data() + (size_t)l * 2 * d + d;
        for (int i = 0; i < d; i++) mxp[i] = -1e30f;
        for (int t = 0; t < n; t++) {
            const float* x = X.data() + (size_t)t * d;
            for (int i = 0; i < d; i++) {
                mean[i] += x[i];
                mxp[i] = std::max(mxp[i], x[i]);
            }
        }
        for (int i = 0; i < d; i++) mean[i] /= n;
    }
    cache->len = pos_offset + n;

    rms(n, d, X.data(), w.final_norm.ptr, w.rms_eps, Xn.data());
    const float* last = Xn.data() + (size_t)(n - 1) * d;
    if (want_logits) {
        R.last_logits.resize(V);
        mm(1, V, d, last, w.embed.ptr, R.last_logits.data());
        R.last_probs.resize(V);
        float mx = *std::max_element(R.last_logits.begin(), R.last_logits.end());
        double se = 0;
        for (int v = 0; v < V; v++) {
            double e = std::exp(R.last_logits[v] - mx);
            R.last_probs[v] = (float)e;
            se += e;
        }
        float inv = (float)(1 / se);
        double ent = 0;
        for (int v = 0; v < V; v++) {
            R.last_probs[v] *= inv;
            double p = R.last_probs[v];
            if (p > 0) ent -= p * std::log(p);
        }
        tr.entropy = (float)ent;
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + TOPK, idx.end(),
                          [&](int a, int b) { return R.last_probs[a] > R.last_probs[b]; });
        double sum = 0;
        for (int i = 0; i < TOPK; i++) {
            tr.topk_ids[i] = idx[i];
            tr.topk_probs[i] = R.last_probs[idx[i]];
            sum += tr.topk_probs[i];
        }
        tr.tail_mass = (float)std::max(0.0, 1.0 - sum);
    }
    return R;
}

std::vector<float> decode_step(const KnowledgeWeights& w, KVCache& cache, int32_t token,
                               std::vector<float>* last_probs_out) {
    std::vector<int32_t> one{token};
    auto r = forward_kv(w, one, cache.len, true, &cache);
    if (last_probs_out) *last_probs_out = r.last_probs;
    return r.last_probs;
}

} // namespace alysane
