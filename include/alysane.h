#pragma once
// Alysane C2+C3 pilot core. C++17. Hot paths: C++/OpenBLAS/AVX-512/OpenMP.
#include "json.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace alysane {
using json = nlohmann::json;
constexpr int TOPK = 64;

struct Mmap {
    void* p = nullptr;
    size_t n = 0;
    int fd = -1;
    explicit Mmap(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error("open " + path);
        struct stat st{};
        fstat(fd, &st);
        n = (size_t)st.st_size;
        p = ::mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("mmap " + path);
        }
    }
    ~Mmap() {
        if (p && p != MAP_FAILED)
            ::munmap(p, n);
        if (fd >= 0)
            ::close(fd);
    }
    Mmap(const Mmap&) = delete;
    Mmap& operator=(const Mmap&) = delete;
};

struct Tensor {
    const float* ptr = nullptr;
    std::vector<int64_t> shape;
    int64_t numel = 0;
};
struct LayerWeights {
    Tensor input_layernorm, post_attention_layernorm;
    Tensor q, k, v, o, gate, up, down;
};
struct KnowledgeWeights {
    std::unique_ptr<Mmap> file;
    std::unordered_map<std::string, Tensor> tensors;
    Tensor embed, final_norm;
    std::vector<LayerWeights> layers;
    int V = 0, d = 0, L = 0, H = 0, KVH = 0, FF = 0, head_dim = 0;
    float rms_eps = 1e-5f, rope_theta = 10000.f;
    int64_t param_count = 0;
    std::string sha256;
};

struct Trace {
    std::vector<float> pooled;       // L * 2 * d: mean then max per layer
    std::vector<float> attn_entropy; // H
    std::vector<int32_t> topk_ids;   // TOPK
    std::vector<float> topk_probs;   // TOPK
    float tail_mass = 0, entropy = 0, nll = 0;
    int input_dim() const { return (int)pooled.size() + (int)attn_entropy.size() + 2 * TOPK + 2; }
    std::vector<float> features() const {
        std::vector<float> f;
        f.reserve(input_dim());
        f.insert(f.end(), pooled.begin(), pooled.end());
        f.insert(f.end(), attn_entropy.begin(), attn_entropy.end());
        for (int i = 0; i < TOPK; ++i)
            f.push_back((float)topk_ids[i]);
        f.insert(f.end(), topk_probs.begin(), topk_probs.end());
        f.push_back(tail_mass);
        f.push_back(entropy);
        return f;
    }
};
struct ForwardResult {
    Trace trace;
    std::vector<float> last_probs;
    std::vector<float> last_logits;
    std::vector<float> all_logits;
    float nll = 0;
};

KnowledgeWeights load_knowledge(const std::string& dir);
ForwardResult forward(const KnowledgeWeights& w, const std::vector<int32_t>& ids, bool want_all_logits,
                      bool want_full_last, bool want_logits);
float tv_distance(const float* a, const float* b, int V);
float topk_tv(const Trace& a, const Trace& b, int V);
float topk_tau(const Trace& a, const Trace& b);

struct Corpus {
    int N = 0, n = 0;
    std::vector<int32_t> ids;
    std::vector<std::string> meta;
};
Corpus read_tokens(const std::string& path);
void write_tokens(const std::string& path, const Corpus& c);
void write_traces(const std::string& path, const std::vector<Trace>& t, int L, int d, int H);
std::vector<Trace> read_traces(const std::string& path, int L, int d, int H);
void write_probs(const std::string& path, const std::vector<std::vector<float>>& p);
std::vector<std::vector<float>> read_probs(const std::string& path, int N, int V);

struct DistMatrix {
    int N = 0;
    std::vector<float> d;
    float operator()(int i, int j) const { return d[(size_t)i * N + j]; }
    float& at(int i, int j) { return d[(size_t)i * N + j]; }
};
DistMatrix pairwise_tv(const std::vector<std::vector<float>>& probs, int V);
struct DSU {
    std::vector<int> p, sz;
    explicit DSU(int n) : p(n), sz(n, 1) { std::iota(p.begin(), p.end(), 0); }
    int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b)
            return;
        if (sz[a] < sz[b])
            std::swap(a, b);
        p[b] = a;
        sz[a] += sz[b];
    }
};
struct MST {
    std::vector<int> u, v;
    std::vector<float> w;
};
MST prim_mst(const DistMatrix& D);
std::vector<int> components_at(const MST& m, int N, float eps);
double adjusted_rand(const std::vector<int>& a, const std::vector<int>& b);

struct A1Candidate {
    int q = 0;
    float eps_in = 0, eps_out = 0, width = 0, ratio = 0;
    int min_size = 0, max_size = 0;
    std::vector<int> labels;
};
struct A1Report {
    bool strict_separated = false;
    bool useful = false;
    A1Candidate best;
    std::vector<A1Candidate> curve;
    json bootstrap;
};
A1Report measure_a1(const DistMatrix& D, int min_size_target, int qmax, int boot_B, uint64_t seed);

std::vector<int> farthest_landmarks(const DistMatrix& D, int M, uint64_t seed);
std::vector<std::vector<float>> psi_embedding(const std::vector<std::vector<float>>& probs,
                                              const std::vector<int>& landmarks, int V);
std::vector<std::vector<float>> psi_embedding_to(const std::vector<std::vector<float>>& queries,
                                                 const std::vector<std::vector<float>>& landmark_probs,
                                                 int V);

struct SelfKnowledgeWeights {
    int in_dim = 0, hidden = 256, out_dim = 256;
    std::vector<float> mean, std, W1, b1, W2, b2, white; // white: out_dim x out_dim
    std::vector<float> energy_W, energy_b, energy_a;
    int energy_hidden = 64;
    float energy_sigma = 1.f;
    std::vector<int> landmarks;
    std::vector<std::vector<float>> landmark_probs;
    json meta;
    std::vector<float> encode_raw(const Trace& t) const;
    std::vector<float> encode(const Trace& t) const;
    float energy(const std::vector<float>& z) const;
    std::vector<float> energy_grad(const std::vector<float>& z) const;
    void save(const std::string& path) const;
    static SelfKnowledgeWeights load(const std::string& path);
};
SelfKnowledgeWeights train_self(const std::vector<Trace>& traces, const DistMatrix& D,
                                const std::vector<std::vector<float>>& probs, const A1Report& a1, int out_dim,
                                int epochs, uint64_t seed);
SelfKnowledgeWeights train_energy(SelfKnowledgeWeights S, const std::vector<Trace>& traces, int epochs,
                                  uint64_t seed);

// ============================================================================
// C2+C3 completion: KV-cached generation, continuation behaviour, persistence.
// ============================================================================

/// Per-layer key/value cache. Layout: [L][cap][KVH*hd], keys already RoPE-rotated.
struct KVCache {
    int L = 0, KVH = 0, hd = 0, len = 0, cap = 0;
    std::vector<float> k, v;
    void init(int L_, int KVH_, int hd_, int cap_) {
        L = L_; KVH = KVH_; hd = hd_; cap = cap_; len = 0;
        k.assign((size_t)L * cap * KVH * hd, 0.f);
        v.assign((size_t)L * cap * KVH * hd, 0.f);
    }
    float* kp(int l, int t) { return k.data() + ((size_t)l * cap + t) * KVH * hd; }
    float* vp(int l, int t) { return v.data() + ((size_t)l * cap + t) * KVH * hd; }
    const float* kp(int l, int t) const { return k.data() + ((size_t)l * cap + t) * KVH * hd; }
    const float* vp(int l, int t) const { return v.data() + ((size_t)l * cap + t) * KVH * hd; }
    size_t bytes() const { return (k.size() + v.size()) * sizeof(float); }
};

/// Threads used by BLAS/OpenMP hot paths (default 2, override ALYSANE_THREADS).
int hw_threads();

/// Forward with optional KV cache. `pos_offset` is the absolute position of ids[0].
/// When cache != nullptr the K/V for these positions are appended to it.
ForwardResult forward_kv(const KnowledgeWeights& w, const std::vector<int32_t>& ids, int pos_offset,
                         bool want_logits, KVCache* cache);

/// One autoregressive decode step at absolute position `pos` (cache.len must equal pos).
/// Returns the next-token distribution and appends the consumed token's K/V to the cache.
std::vector<float> decode_step(const KnowledgeWeights& w, KVCache& cache, int32_t token,
                               std::vector<float>* last_probs_out = nullptr);

// ---- §1.3 self-generated outcome signals (rollout based, no labels) ----

struct RolloutConfig {
    int R = 4;          ///< continuations per context
    int T = 8;          ///< tokens per continuation
    float temperature = 1.0f;
    uint64_t seed = 20261105ULL;
};

struct RolloutSet {
    std::vector<std::vector<int32_t>> seq; ///< R sequences of length T
    /// Self-agreement a(x): mean pairwise n-gram kernel among own continuations.
    float self_agreement = 0.f;
    /// Exact-agreement rate (strict: identical continuations).
    float exact_agreement = 0.f;
    /// Mean next-token entropy along the rollout (predictive uncertainty proxy).
    float mean_entropy = 0.f;
};

/// Generate R continuations of length T from context ids, sharing one prefill.
RolloutSet rollouts(const KnowledgeWeights& w, const std::vector<int32_t>& ids, const RolloutConfig& cfg,
                    uint64_t stream_seed);

/// Normalised n-gram (spectrum-kernel) feature map of a token sequence, n=1..3.
/// Sparse: token-id keys hashed into a fixed-width signed projection for tractability.
constexpr int CONT_FEATURE_DIM = 512;
std::vector<float> continuation_features(const std::vector<int32_t>& seq);

/// Mean feature vector over a rollout set (the empirical kernel mean embedding).
std::vector<float> continuation_mean(const RolloutSet& rs);

/// MMD pseudometric between two rollout means (bounded in [0, sqrt(2)] after normalisation).
float continuation_distance(const std::vector<float>& a, const std::vector<float>& b);

/// Continuation consistency kappa(x,y): cross kernel between the two rollout sets.
float continuation_consistency(const RolloutSet& a, const RolloutSet& b);

// ---- Persistence-based multiscale structure (A1 replacement) ----

struct PersistenceDiagram {
    /// 0-dimensional deaths = single-linkage merge heights = sorted MST edge weights.
    std::vector<float> deaths;
    /// Sublevel-set ultrametric merge height for each pair is recoverable from the MST.
    std::vector<float> null_quantile; ///< null 95th percentile per ordered death index
    std::vector<int> significant;     ///< indices of deaths exceeding the null quantile
    int n_significant = 0;
    float null_B = 0;
};

/// 0-dim persistence of the single-linkage filtration, with a permutation null
/// calibration that destroys geometry while preserving the distance distribution.
PersistenceDiagram persistence0(const DistMatrix& D, int null_B, uint64_t seed, float alpha = 0.05f);

/// Single-linkage ultrametric (minimax path distance) from the MST, O(N^2).
DistMatrix ultrametric(const MST& m, int N);

struct SoftRegions {
    std::vector<int> landmarks;              ///< region anchors (indices into the sample)
    std::vector<float> radius;               ///< per-anchor local scale
    std::vector<std::vector<float>> member;  ///< N x K partition-of-unity weights
    std::vector<float> boundary_score;       ///< membership entropy, normalised to [0,1]
    int K = 0;
};

/// Overlapping soft regions: partition of unity subordinate to a landmark cover.
/// No hard assignment, no fixed cluster count, no labels.
SoftRegions soft_regions(const DistMatrix& D, const std::vector<int>& landmarks, float radius_quantile);

/// Density-peaks mode seeking: regions determined by the metric alone, with no
/// anchor-subset selection variance. Mode count is data-determined.
struct DensityPeaks {
    std::vector<float> rho;    ///< KDE density per point
    std::vector<int> parent;   ///< nearest denser neighbour (-1 for global max)
    std::vector<float> delta;  ///< distance to that neighbour
    std::vector<int> modes;    ///< emergent mode indices
    std::vector<int> mode_of;  ///< mode ancestor per point
    std::vector<int> label;    ///< mode index per point
    std::vector<std::vector<float>> member; ///< partition of unity over modes
    std::vector<float> boundary_score;
    int K = 0;
    float sigma = 0.f;
};
DensityPeaks density_peaks(const DistMatrix& D, int knn = 8, float bandwidth_quantile = 0.5f);

/// Partition-of-unity membership for an out-of-sample point given anchor distances.
std::vector<float> soft_membership(const std::vector<float>& dist_to_anchor, const std::vector<float>& radius);

// ---- Trace quantisation (§4) ----

struct QuantizedTrace {
    std::vector<int8_t> pooled_q; ///< per-layer-block symmetric int8
    std::vector<float> scale;     ///< one scale per layer (2*d block)
    std::vector<float> attn_entropy, topk_probs;
    std::vector<int32_t> topk_ids;
    float tail_mass = 0, entropy = 0, nll = 0;
};
QuantizedTrace quantize_trace(const Trace& t, int L, int d);
Trace dequantize_trace(const QuantizedTrace& q, int L, int d);
/// Exact TV between the *unmodified* top-k distributions of a quantised trace pair.
/// Hidden-state quantisation does not enter tau (see spec §4); this proves it.
float quantized_topk_tv(const QuantizedTrace& a, const QuantizedTrace& b, int V);

// ---- Reshape / drift (Theorem 3) ----

struct DriftReport {
    float delta_theta_norm = 0.f;
    float max_logit_inf = 0.f;   ///< max_x ||z'(x)-z(x)||_inf
    float empirical_L_z = 0.f;   ///< max_logit_inf / ||delta||
    float max_tv_change = 0.f;   ///< max pair |d_B' - d_B|
    float predicted_bound = 0.f; ///< 2 * max_logit_inf
    int bound_violations = 0;
    int pairs = 0;
};

DriftReport measure_drift(const KnowledgeWeights& w, const std::vector<std::vector<int32_t>>& ctxs,
                          float rel_norm, uint64_t seed);

void run_gates(const std::string& model_dir, const std::string& data_dir, const std::string& out_dir);
void run_completion_gates(const std::string& model_dir, const std::string& data_dir, const std::string& out_dir);
void run_parity(const std::string& model_dir, const std::string& tokens, const std::string& out);
void run_selftest();
void run_unit_tests();
} // namespace alysane
