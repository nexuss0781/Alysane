#include "alysane.h"
#include <cblas.h>
extern "C" void openblas_set_num_threads(int);
#include <immintrin.h>
#include "simd.h"
#include <iomanip>
#include <map>
#include <omp.h>

namespace alysane {
static void setblas(int threads) {
    openblas_set_num_threads(threads);
}
static std::string sha256_file(const std::string& path) {
    // Minimal SHA-256 (FIPS 180-4) for provenance.
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::ifstream f(path, std::ios::binary);
    std::vector<char> buf(1 << 20);
    uint64_t total = 0;
    auto block = [&](const uint8_t* p) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 |
                   p[4 * i + 3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f0 = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f0) ^ ((~e) & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh = g;
            g = f0;
            f0 = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f0;
        h[6] += g;
        h[7] += hh;
    };
    std::vector<uint8_t> tail;
    while (f) {
        f.read(buf.data(), buf.size());
        std::streamsize got = f.gcount();
        if (got <= 0)
            break;
        total += got;
        size_t off = 0;
        if (!tail.empty()) {
            size_t need = 64 - tail.size();
            size_t take = std::min((size_t)got, need);
            tail.insert(tail.end(), buf.data(), buf.data() + take);
            off = take;
            if (tail.size() == 64) {
                block(tail.data());
                tail.clear();
            }
        }
        while (off + 64 <= (size_t)got) {
            block((const uint8_t*)buf.data() + off);
            off += 64;
        }
        if (off < (size_t)got)
            tail.insert(tail.end(), buf.data() + off, buf.data() + got);
    }
    uint64_t bits = total * 8;
    tail.push_back(0x80);
    while (tail.size() % 64 != 56)
        tail.push_back(0);
    for (int i = 7; i >= 0; i--)
        tail.push_back((bits >> (8 * i)) & 255);
    for (size_t o = 0; o < tail.size(); o += 64)
        block(tail.data() + o);
    std::ostringstream os;
    for (int i = 0; i < 8; i++)
        os << std::hex << std::setw(8) << std::setfill('0') << h[i];
    return os.str();
}

KnowledgeWeights load_knowledge(const std::string& dir) {
    KnowledgeWeights w;
    std::ifstream cf(dir + "/config.json");
    json c;
    cf >> c;
    w.V = c.at("vocab_size");
    w.d = c.at("hidden_size");
    w.L = c.at("num_hidden_layers");
    w.H = c.at("num_attention_heads");
    w.KVH = c.at("num_key_value_heads");
    w.FF = c.at("intermediate_size");
    w.head_dim = w.d / w.H;
    w.rms_eps = c.value("rms_norm_eps", 1e-5f);
    w.rope_theta = c.value("rope_theta", 10000.f);
    w.file = std::make_unique<Mmap>(dir + "/model.safetensors");
    const uint8_t* base = (const uint8_t*)w.file->p;
    uint64_t hlen;
    std::memcpy(&hlen, base, 8);
    json hdr = json::parse(std::string((const char*)base + 8, (const char*)base + 8 + hlen));
    size_t data_off = 8 + hlen;
    for (auto it = hdr.begin(); it != hdr.end(); ++it) {
        if (it.key() == "__metadata__")
            continue;
        json t = it.value();
        if (t.at("dtype") != "F32")
            throw std::runtime_error("pilot loader expects F32: " + it.key());
        auto off = t.at("data_offsets");
        Tensor ts;
        ts.ptr = (const float*)(base + data_off + off[0].get<size_t>());
        ts.shape = t.at("shape").get<std::vector<int64_t>>();
        ts.numel = 1;
        for (auto s : ts.shape)
            ts.numel *= s;
        w.tensors[it.key()] = ts;
        w.param_count += ts.numel;
    }
    auto get = [&](const std::string& n) -> Tensor {
        auto it = w.tensors.find(n);
        if (it == w.tensors.end())
            throw std::runtime_error("missing tensor " + n);
        return it->second;
    };
    w.embed = get("model.embed_tokens.weight");
    w.final_norm = get("model.norm.weight");
    for (int l = 0; l < w.L; l++) {
        std::string p = "model.layers." + std::to_string(l) + ".";
        LayerWeights lw;
        lw.input_layernorm = get(p + "input_layernorm.weight");
        lw.post_attention_layernorm = get(p + "post_attention_layernorm.weight");
        lw.q = get(p + "self_attn.q_proj.weight");
        lw.k = get(p + "self_attn.q_proj.weight");
        lw.k = get(p + "self_attn.k_proj.weight");
        lw.v = get(p + "self_attn.v_proj.weight");
        lw.o = get(p + "self_attn.o_proj.weight");
        lw.gate = get(p + "mlp.gate_proj.weight");
        lw.up = get(p + "mlp.up_proj.weight");
        lw.down = get(p + "mlp.down_proj.weight");
        w.layers.push_back(lw);
    }
    w.sha256 = sha256_file(dir + "/model.safetensors");
    return w;
}

static void rmsnorm(int n, int d, const float* x, const float* g, float eps, float* y) {
    for (int t = 0; t < n; t++) {
        const float* xx = x + (size_t)t * d;
        float* yy = y + (size_t)t * d;
        float s = 0;
        for (int i = 0; i < d; i++)
            s += xx[i] * xx[i];
        float inv = 1.f / std::sqrt(s / d + eps);
        for (int i = 0; i < d; i++)
            yy[i] = xx[i] * inv * g[i];
    }
}
static void gemm(int n, int out, int in, const float* X, const float* W, float* Y) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, out, in, 1.f, X, in, W, in, 0.f, Y, out);
}
static float dot64(const float* a, const float* b) { return alysane::simd::dot(a, b, 64); }
ForwardResult forward(const KnowledgeWeights& w, const std::vector<int32_t>& ids, bool want_all_logits,
                      bool want_full_last, bool want_logits) {
    setblas(2);
    int n = (int)ids.size(), d = w.d, H = w.H, KV = w.KVH, hd = w.head_dim, FF = w.FF, V = w.V;
    if (n <= 0 || n > 2048)
        throw std::runtime_error("bad context length");
    ForwardResult R;
    Trace& tr = R.trace;
    tr.pooled.assign((size_t)w.L * 2 * d, 0);
    tr.attn_entropy.assign(H, 0);
    tr.topk_ids.resize(TOPK);
    tr.topk_probs.resize(TOPK);
    std::vector<float> X((size_t)n * d), Xn((size_t)n * d), Q((size_t)n * d), K((size_t)n * KV * hd),
        Vv((size_t)n * KV * hd), O((size_t)n * d), proj((size_t)n * d), G((size_t)n * FF), U((size_t)n * FF);
    for (int t = 0; t < n; t++)
        std::copy(w.embed.ptr + (size_t)ids[t] * d, w.embed.ptr + (size_t)(ids[t] + 1) * d,
                  X.begin() + (size_t)t * d);
    std::vector<float> cos((size_t)n * hd), sin((size_t)n * hd);
    for (int t = 0; t < n; t++)
        for (int i = 0; i < hd / 2; i++) {
            float freq = std::pow(w.rope_theta, -2.f * i / hd);
            float a = t * freq;
            cos[(size_t)t * hd + i] = cos[(size_t)t * hd + i + hd / 2] = std::cos(a);
            sin[(size_t)t * hd + i] = sin[(size_t)t * hd + i + hd / 2] = std::sin(a);
        }
    auto rope = [&](float* x, int heads) {
        for (int t = 0; t < n; t++)
            for (int h = 0; h < heads; h++) {
                float* p = x + ((size_t)t * heads + h) * hd;
                for (int i = 0; i < hd / 2; i++) {
                    float c = cos[(size_t)t * hd + i], s = sin[(size_t)t * hd + i];
                    float a = p[i], b = p[i + hd / 2];
                    p[i] = a * c - b * s;
                    p[i + hd / 2] = b * c + a * s;
                }
            }
    };
    std::vector<float> scores((size_t)n * n), att((size_t)n * hd);
    for (int l = 0; l < w.L; l++) {
        const auto& lw = w.layers[l];
        rmsnorm(n, d, X.data(), lw.input_layernorm.ptr, w.rms_eps, Xn.data());
        gemm(n, d, d, Xn.data(), lw.q.ptr, Q.data());
        gemm(n, KV * hd, d, Xn.data(), lw.k.ptr, K.data());
        gemm(n, KV * hd, d, Xn.data(), lw.v.ptr, Vv.data());
        rope(Q.data(), H);
        rope(K.data(), KV);
        std::fill(O.begin(), O.end(), 0.f);
        for (int h = 0; h < H; h++) {
            int kv = h / (H / KV);
            float ent = 0;
            for (int t = 0; t < n; t++) {
                const float* q = Q.data() + ((size_t)t * H + h) * hd;
                float mx = -1e30f;
                for (int j = 0; j <= t; j++) {
                    float s = dot64(q, K.data() + ((size_t)j * KV + kv) * hd) * 0.125f;
                    scores[(size_t)t * n + j] = s;
                    mx = std::max(mx, s);
                }
                float sum = 0;
                for (int j = 0; j <= t; j++) {
                    float e = std::exp(scores[(size_t)t * n + j] - mx);
                    scores[(size_t)t * n + j] = e;
                    sum += e;
                }
                float inv = 1.f / sum;
                float* out = O.data() + ((size_t)t * H + h) * hd;
                for (int j = 0; j <= t; j++) {
                    float p = scores[(size_t)t * n + j] * inv;
                    scores[(size_t)t * n + j] = p;
                    if (p > 0)
                        ent -= p * std::log(p);
                    const float* v = Vv.data() + ((size_t)j * KV + kv) * hd;
                    for (int i = 0; i < hd; i++)
                        out[i] += p * v[i];
                }
            }
            tr.attn_entropy[h] += ent / n;
        }
        gemm(n, d, d, O.data(), lw.o.ptr, proj.data());
        for (size_t i = 0; i < X.size(); i++)
            X[i] += proj[i];
        rmsnorm(n, d, X.data(), lw.post_attention_layernorm.ptr, w.rms_eps, Xn.data());
        gemm(n, FF, d, Xn.data(), lw.gate.ptr, G.data());
        gemm(n, FF, d, Xn.data(), lw.up.ptr, U.data());
        for (size_t i = 0; i < G.size(); i++) {
            float g = G[i];
            G[i] = (g / (1.f + std::exp(-g))) * U[i];
        }
        gemm(n, d, FF, G.data(), lw.down.ptr, proj.data());
        for (size_t i = 0; i < X.size(); i++)
            X[i] += proj[i];
        float* mean = tr.pooled.data() + (size_t)l * 2 * d;
        float* mx = tr.pooled.data() + (size_t)l * 2 * d + d;
        for (int i = 0; i < d; i++)
            mx[i] = -1e30f;
        for (int t = 0; t < n; t++) {
            const float* x = X.data() + (size_t)t * d;
            for (int i = 0; i < d; i++) {
                mean[i] += x[i];
                mx[i] = std::max(mx[i], x[i]);
            }
        }
        for (int i = 0; i < d; i++)
            mean[i] /= n;
    }
    rmsnorm(n, d, X.data(), w.final_norm.ptr, w.rms_eps, Xn.data());
    if (want_all_logits) {
        R.all_logits.resize((size_t)n * V);
        gemm(n, V, d, Xn.data(), w.embed.ptr, R.all_logits.data());
        double nll = 0;
        for (int t = 0; t + 1 < n; t++) {
            const float* z = R.all_logits.data() + (size_t)t * V;
            float mx = *std::max_element(z, z + V);
            double se = 0;
            for (int v = 0; v < V; v++)
                se += std::exp(z[v] - mx);
            nll += -(z[ids[t + 1]] - mx - std::log(se));
        }
        R.nll = (float)(nll / (n - 1));
        tr.nll = R.nll;
    }
    const float* last = Xn.data() + (size_t)(n - 1) * d;
    if (want_all_logits) {
        R.last_logits.assign(R.all_logits.begin() + (size_t)(n - 1) * V, R.all_logits.end());
    } else if (want_logits) {
        R.last_logits.resize(V);
        gemm(1, V, d, last, w.embed.ptr, R.last_logits.data());
    }
    if (want_full_last || want_logits || want_all_logits) {
        if (R.last_logits.empty()) {
            R.last_logits.resize(V);
            gemm(1, V, d, last, w.embed.ptr, R.last_logits.data());
        }
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
            if (p > 0)
                ent -= p * std::log(p);
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

float tv_distance(const float* a, const float* b, int V) { return 0.5f * alysane::simd::l1(a, b, V); }
float topk_tau(const Trace& a, const Trace& b) {
    return a.tail_mass + b.tail_mass;
}
float topk_tv(const Trace& a, const Trace& b, int V) {
    // Uniform-tail reconstruction on each trace's own top-k support.
    std::unordered_map<int, float> pa, pb;
    for (int i = 0; i < TOPK; i++) {
        pa[a.topk_ids[i]] = a.topk_probs[i];
        pb[b.topk_ids[i]] = b.topk_probs[i];
    }
    float ta = a.tail_mass / (V - TOPK), tb = b.tail_mass / (V - TOPK);
    double s = 0;
    int u = 0;
    for (auto& kv : pa) {
        auto it = pb.find(kv.first);
        float bv = it == pb.end() ? tb : it->second;
        s += std::fabs(kv.second - bv);
        u++;
    }
    for (auto& kv : pb) {
        if (!pa.count(kv.first)) {
            s += std::fabs(ta - kv.second);
            u++;
        }
    }
    s += (V - u) * std::fabs(ta - tb);
    return (float)(0.5 * s);
}

Corpus read_tokens(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("open " + path);
    Corpus c;
    f.read((char*)&c.N, 4);
    f.read((char*)&c.n, 4);
    c.ids.resize((size_t)c.N * c.n);
    f.read((char*)c.ids.data(), (std::streamsize)c.ids.size() * 4);
    uint32_t m;
    f.read((char*)&m, 4);
    for (uint32_t i = 0; i < m; i++) {
        uint32_t l;
        f.read((char*)&l, 4);
        std::string s(l, '\0');
        f.read(s.data(), l);
        c.meta.push_back(s);
    }
    return c;
}
void write_tokens(const std::string& path, const Corpus& c) {
    std::ofstream f(path, std::ios::binary);
    f.write((char*)&c.N, 4);
    f.write((char*)&c.n, 4);
    f.write((char*)c.ids.data(), (std::streamsize)c.ids.size() * 4);
    uint32_t m = (uint32_t)c.meta.size();
    f.write((char*)&m, 4);
    for (auto& s : c.meta) {
        uint32_t l = (uint32_t)s.size();
        f.write((char*)&l, 4);
        f.write(s.data(), l);
    }
}
static void wv(std::ostream& f, const std::vector<float>& v) {
    uint32_t n = (uint32_t)v.size();
    f.write((char*)&n, 4);
    f.write((char*)v.data(), (std::streamsize)n * 4);
}
static std::vector<float> rv(std::istream& f) {
    uint32_t n;
    f.read((char*)&n, 4);
    std::vector<float> v(n);
    f.read((char*)v.data(), (std::streamsize)n * 4);
    return v;
}
void write_traces(const std::string& path, const std::vector<Trace>& t, int L, int d, int H) {
    std::ofstream f(path, std::ios::binary);
    uint32_t N = (uint32_t)t.size();
    f.write((char*)&N, 4);
    f.write((char*)&L, 4);
    f.write((char*)&d, 4);
    f.write((char*)&H, 4);
    for (auto& tr : t) {
        wv(f, tr.pooled);
        wv(f, tr.attn_entropy);
        uint32_t k = TOPK;
        f.write((char*)&k, 4);
        f.write((char*)tr.topk_ids.data(), k * 4);
        wv(f, tr.topk_probs);
        f.write((char*)&tr.tail_mass, 4);
        f.write((char*)&tr.entropy, 4);
        f.write((char*)&tr.nll, 4);
    }
}
std::vector<Trace> read_traces(const std::string& path, int L, int d, int H) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("open " + path);
    uint32_t N, l, dd, h;
    f.read((char*)&N, 4);
    f.read((char*)&l, 4);
    f.read((char*)&dd, 4);
    f.read((char*)&h, 4);
    if ((int)l != L || (int)dd != d || (int)h != H)
        throw std::runtime_error("trace shape mismatch");
    std::vector<Trace> t(N);
    for (auto& tr : t) {
        tr.pooled = rv(f);
        tr.attn_entropy = rv(f);
        uint32_t k;
        f.read((char*)&k, 4);
        tr.topk_ids.resize(k);
        f.read((char*)tr.topk_ids.data(), k * 4);
        tr.topk_probs = rv(f);
        f.read((char*)&tr.tail_mass, 4);
        f.read((char*)&tr.entropy, 4);
        f.read((char*)&tr.nll, 4);
    }
    return t;
}
void write_probs(const std::string& path, const std::vector<std::vector<float>>& p) {
    std::ofstream f(path, std::ios::binary);
    uint32_t N = (uint32_t)p.size(), V = (uint32_t)p[0].size();
    f.write((char*)&N, 4);
    f.write((char*)&V, 4);
    for (auto& v : p)
        f.write((char*)v.data(), (std::streamsize)V * 4);
}
std::vector<std::vector<float>> read_probs(const std::string& path, int N, int V) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("open " + path);
    uint32_t n, v;
    f.read((char*)&n, 4);
    f.read((char*)&v, 4);
    if ((int)n != N || (int)v != V)
        throw std::runtime_error("probs shape");
    std::vector<std::vector<float>> p(N, std::vector<float>(V));
    for (auto& x : p)
        f.read((char*)x.data(), (std::streamsize)V * 4);
    return p;
}

DistMatrix pairwise_tv(const std::vector<std::vector<float>>& probs, int V) {
    int N = (int)probs.size();
    DistMatrix D;
    D.N = N;
    D.d.assign((size_t)N * N, 0);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            float d = tv_distance(probs[i].data(), probs[j].data(), V);
            D.at(i, j) = d;
            D.at(j, i) = d;
        }
    return D;
}
MST prim_mst(const DistMatrix& D) {
    int N = D.N;
    MST m;
    std::vector<float> key(N, 1e30f);
    std::vector<int> par(N, -1);
    std::vector<char> in(N, false);
    key[0] = 0;
    for (int it = 0; it < N; it++) {
        int u = -1;
        float best = 1e30f;
        for (int i = 0; i < N; i++)
            if (!in[i] && key[i] < best) {
                best = key[i];
                u = i;
            }
        if (u < 0)
            break;
        in[u] = true;
        if (par[u] >= 0) {
            m.u.push_back(par[u]);
            m.v.push_back(u);
            m.w.push_back(key[u]);
        }
        for (int v = 0; v < N; v++)
            if (!in[v] && D(u, v) < key[v]) {
                key[v] = D(u, v);
                par[v] = u;
            }
    }
    return m;
}
std::vector<int> components_at(const MST& m, int N, float eps) {
    DSU d(N);
    for (size_t i = 0; i < m.w.size(); i++)
        if (m.w[i] <= eps)
            d.unite(m.u[i], m.v[i]);
    std::vector<int> lab(N);
    std::unordered_map<int, int> mp;
    int k = 0;
    for (int i = 0; i < N; i++) {
        int r = d.find(i);
        auto it = mp.find(r);
        if (it == mp.end()) {
            mp[r] = k;
            lab[i] = k++;
        } else
            lab[i] = it->second;
    }
    return lab;
}
double adjusted_rand(const std::vector<int>& a, const std::vector<int>& b) {
    int n = (int)a.size();
    std::map<std::pair<int, int>, int> cnt;
    std::unordered_map<int, int> ca, cb;
    for (int i = 0; i < n; i++) {
        cnt[{a[i], b[i]}]++;
        ca[a[i]]++;
        cb[b[i]]++;
    }
    auto c2 = [](double x) { return x * (x - 1) / 2; };
    double sum = 0;
    for (auto& kv : cnt)
        sum += c2(kv.second);
    double sa = 0, sb = 0;
    for (auto& kv : ca)
        sa += c2(kv.second);
    for (auto& kv : cb)
        sb += c2(kv.second);
    double tot = c2(n);
    double exp = sa * sb / tot;
    double mx = (sa + sb) / 2;
    return std::fabs(mx - exp) < 1e-15 ? (std::fabs(sum - exp) < 1e-15 ? 1 : 0) : (sum - exp) / (mx - exp);
}

A1Report measure_a1(const DistMatrix& D, int min_size_target, int qmax, int boot_B, uint64_t seed) {
    int N = D.N;
    MST m = prim_mst(D);
    std::vector<int> ord(m.w.size());
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return m.w[a] < m.w[b]; });
    A1Report R;
    R.strict_separated = N > 1 && m.w.back() > 0;
    for (int q = 2; q <= std::min(qmax, N - 1); q++) {
        int keep = N - q;
        if (keep < 1 || keep > (int)ord.size())
            continue;
        A1Candidate c;
        c.q = q;
        c.eps_in = m.w[ord[keep - 1]];
        c.eps_out = m.w[ord[keep]];
        c.width = c.eps_out - c.eps_in;
        c.ratio = c.eps_in > 1e-12f ? c.eps_out / c.eps_in : 1e30f;
        DSU d(N);
        for (int i = 0; i < keep; i++)
            d.unite(m.u[ord[i]], m.v[ord[i]]);
        std::vector<int> lab(N);
        std::unordered_map<int, int> mp;
        int k = 0;
        for (int i = 0; i < N; i++) {
            int r = d.find(i);
            auto it = mp.find(r);
            if (it == mp.end()) {
                mp[r] = k;
                lab[i] = k++;
            } else
                lab[i] = it->second;
        }
        c.labels = lab;
        std::vector<int> sz(k);
        for (int x : lab)
            sz[x]++;
        c.min_size = *std::min_element(sz.begin(), sz.end());
        c.max_size = *std::max_element(sz.begin(), sz.end());
        R.curve.push_back(c);
    }
    float best_score = -1;
    for (auto& c : R.curve) {
        if (c.min_size < min_size_target)
            continue;
        float score = c.width * (1.f - 0.01f * c.q);
        if (score > best_score) {
            best_score = score;
            R.best = c;
        }
    }
    R.useful =
        R.best.q >= 2 && R.best.width > 0.02f && R.best.ratio > 1.10f && R.best.min_size >= min_size_target;
    std::mt19937_64 rng(seed);
    json boot = json::array();
    double ari_sum = 0;
    int B = 0;
    for (int b = 0; b < boot_B; b++) {
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        int n = (int)(0.8 * N);
        idx.resize(n);
        DistMatrix S;
        S.N = n;
        S.d.assign((size_t)n * n, 0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                S.at(i, j) = D(idx[i], idx[j]);
        MST sm = prim_mst(S);
        float eps = (R.best.eps_in + R.best.eps_out) / 2;
        if (R.best.q == 0)
            eps = 0.5f;
        auto lab = components_at(sm, n, eps);
        std::vector<int> ref(n);
        for (int i = 0; i < n; i++)
            ref[i] = R.best.q ? R.best.labels[idx[i]] : 0;
        double ari = adjusted_rand(ref, lab);
        boot.push_back(ari);
        ari_sum += ari;
        B++;
    }
    R.bootstrap = {
        {"B", B},
        {"mean_ari", B ? ari_sum / B : 0},
        {"subsample_fraction", 0.8},
        {"method", "80% vertex subsample; cut at pilot midpoint; ARI vs full-sample pilot labels"}};
    return R;
}

std::vector<int> farthest_landmarks(const DistMatrix& D, int M, uint64_t seed) {
    int N = D.N;
    M = std::min(M, N);
    std::mt19937_64 rng(seed);
    std::vector<int> L;
    std::vector<float> mind(N, 1e30f);
    int first = (int)(rng() % N);
    L.push_back(first);
    for (int i = 0; i < N; i++)
        mind[i] = D(i, first);
    while ((int)L.size() < M) {
        int best = -1;
        float bv = -1;
        for (int i = 0; i < N; i++)
            if (mind[i] > bv) {
                bv = mind[i];
                best = i;
            }
        L.push_back(best);
        for (int i = 0; i < N; i++)
            mind[i] = std::min(mind[i], D(i, best));
    }
    return L;
}
std::vector<std::vector<float>> psi_embedding(const std::vector<std::vector<float>>& probs,
                                              const std::vector<int>& landmarks, int V) {
    int N = (int)probs.size(), M = (int)landmarks.size();
    std::vector<std::vector<float>> psi(N, std::vector<float>(M));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            psi[i][j] = tv_distance(probs[i].data(), probs[landmarks[j]].data(), V);
    return psi;
}
std::vector<std::vector<float>> psi_embedding_to(const std::vector<std::vector<float>>& queries,
                                                 const std::vector<std::vector<float>>& landmark_probs,
                                                 int V) {
    int N = (int)queries.size(), M = (int)landmark_probs.size();
    std::vector<std::vector<float>> psi(N, std::vector<float>(M));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            psi[i][j] = tv_distance(queries[i].data(), landmark_probs[j].data(), V);
    return psi;
}
} // namespace alysane
