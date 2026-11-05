// §4 trace quantisation. Hidden-state statistics are quantised to int8 with a
// per-layer scale; the top-k output distribution is left exact. This proves the
// corrected claim that hidden-state quantisation does NOT enter the TV bound tau.
#include "alysane.h"

namespace alysane {

QuantizedTrace quantize_trace(const Trace& t, int L, int d) {
    QuantizedTrace q;
    int block = 2 * d;
    q.pooled_q.resize(t.pooled.size());
    q.scale.assign(L, 0.f);
    for (int l = 0; l < L; l++) {
        const float* src = t.pooled.data() + (size_t)l * block;
        float mx = 0.f;
        for (int i = 0; i < block; i++) mx = std::max(mx, std::fabs(src[i]));
        float sc = mx > 0 ? mx / 127.f : 1.f;
        q.scale[l] = sc;
        for (int i = 0; i < block; i++) {
            int v = (int)std::lround(src[i] / sc);
            v = std::max(-127, std::min(127, v));
            q.pooled_q[(size_t)l * block + i] = (int8_t)v;
        }
    }
    q.attn_entropy = t.attn_entropy;
    q.topk_ids = t.topk_ids;
    q.topk_probs = t.topk_probs;
    q.tail_mass = t.tail_mass;
    q.entropy = t.entropy;
    q.nll = t.nll;
    return q;
}

Trace dequantize_trace(const QuantizedTrace& q, int L, int d) {
    Trace t;
    int block = 2 * d;
    t.pooled.resize(q.pooled_q.size());
    for (int l = 0; l < L; l++)
        for (int i = 0; i < block; i++)
            t.pooled[(size_t)l * block + i] = q.pooled_q[(size_t)l * block + i] * q.scale[l];
    t.attn_entropy = q.attn_entropy;
    t.topk_ids = q.topk_ids;
    t.topk_probs = q.topk_probs;
    t.tail_mass = q.tail_mass;
    t.entropy = q.entropy;
    t.nll = q.nll;
    return t;
}

float quantized_topk_tv(const QuantizedTrace& a, const QuantizedTrace& b, int V) {
    Trace ta, tb;
    ta.topk_ids = a.topk_ids;
    ta.topk_probs = a.topk_probs;
    ta.tail_mass = a.tail_mass;
    tb.topk_ids = b.topk_ids;
    tb.topk_probs = b.topk_probs;
    tb.tail_mass = b.tail_mass;
    return topk_tv(ta, tb, V);
}

} // namespace alysane
