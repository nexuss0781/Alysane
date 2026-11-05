#include "alysane.h"
#include <cblas.h>
extern "C" void openblas_set_num_threads(int);
#include <lapacke.h>
#include <map>
#include <omp.h>

namespace alysane {
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

std::vector<float> SelfKnowledgeWeights::encode_raw(const Trace& t) const {
    auto x = t.features();
    if ((int)x.size() != in_dim)
        throw std::runtime_error("encoder dim mismatch");
    for (int i = 0; i < in_dim; i++)
        x[i] = (x[i] - mean[i]) / std::max(1e-6f, std[i]);
    std::vector<float> h(hidden), o(out_dim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, hidden, in_dim, 1.f, x.data(), in_dim, W1.data(),
                in_dim, 0.f, h.data(), hidden);
    for (int j = 0; j < hidden; j++)
        h[j] = std::tanh(h[j] + b1[j]);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, out_dim, hidden, 1.f, h.data(), hidden, W2.data(),
                hidden, 0.f, o.data(), out_dim);
    for (int j = 0; j < out_dim; j++)
        o[j] += b2[j];
    return o;
}
std::vector<float> SelfKnowledgeWeights::encode(const Trace& t) const {
    auto r = encode_raw(t);
    std::vector<float> z(out_dim);
    // mean stores input means (in_dim) followed by output means (out_dim).
    const float* m = mean.data() + in_dim;
    for (int j = 0; j < out_dim; j++)
        r[j] -= m[j];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, out_dim, out_dim, 1.f, r.data(), out_dim,
                white.data(), out_dim, 0.f, z.data(), out_dim);
    return z;
}
float SelfKnowledgeWeights::energy(const std::vector<float>& z) const {
    int d = out_dim, h = energy_hidden;
    float e = 0.5f * cblas_sdot(d, z.data(), 1, z.data(), 1);
    std::vector<float> u(h);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, h, d, 1.f, z.data(), d, energy_W.data(), d, 0.f,
                u.data(), h);
    for (int j = 0; j < h; j++)
        e += energy_a[j] * std::tanh(u[j] + energy_b[j]);
    return e;
}
std::vector<float> SelfKnowledgeWeights::energy_grad(const std::vector<float>& z) const {
    int d = out_dim, h = energy_hidden;
    std::vector<float> u(h), t(h), g = z;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, h, d, 1.f, z.data(), d, energy_W.data(), d, 0.f,
                u.data(), h);
    for (int j = 0; j < h; j++) {
        t[j] = std::tanh(u[j] + energy_b[j]);
        float c = energy_a[j] * (1 - t[j] * t[j]);
        for (int i = 0; i < d; i++)
            g[i] += energy_W[(size_t)j * d + i] * c;
    }
    return g;
}
void SelfKnowledgeWeights::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    uint32_t magic = 0x414C5953;
    f.write((char*)&magic, 4);
    f.write((char*)&in_dim, 4);
    f.write((char*)&hidden, 4);
    f.write((char*)&out_dim, 4);
    f.write((char*)&energy_hidden, 4);
    f.write((char*)&energy_sigma, 4);
    wv(f, mean);
    wv(f, std);
    wv(f, W1);
    wv(f, b1);
    wv(f, W2);
    wv(f, b2);
    wv(f, white);
    wv(f, energy_W);
    wv(f, energy_b);
    wv(f, energy_a);
    uint32_t M = (uint32_t)landmarks.size();
    f.write((char*)&M, 4);
    f.write((char*)landmarks.data(), M * 4);
    uint32_t V = (uint32_t)landmark_probs[0].size();
    f.write((char*)&V, 4);
    for (auto& p : landmark_probs)
        f.write((char*)p.data(), (std::streamsize)V * 4);
    std::string s = meta.dump();
    uint32_t l = (uint32_t)s.size();
    f.write((char*)&l, 4);
    f.write(s.data(), l);
}
SelfKnowledgeWeights SelfKnowledgeWeights::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    uint32_t magic;
    f.read((char*)&magic, 4);
    if (magic != 0x414C5953)
        throw std::runtime_error("bad self magic");
    SelfKnowledgeWeights S;
    f.read((char*)&S.in_dim, 4);
    f.read((char*)&S.hidden, 4);
    f.read((char*)&S.out_dim, 4);
    f.read((char*)&S.energy_hidden, 4);
    f.read((char*)&S.energy_sigma, 4);
    S.mean = rv(f);
    S.std = rv(f);
    S.W1 = rv(f);
    S.b1 = rv(f);
    S.W2 = rv(f);
    S.b2 = rv(f);
    S.white = rv(f);
    S.energy_W = rv(f);
    S.energy_b = rv(f);
    S.energy_a = rv(f);
    uint32_t M;
    f.read((char*)&M, 4);
    S.landmarks.resize(M);
    f.read((char*)S.landmarks.data(), M * 4);
    uint32_t V;
    f.read((char*)&V, 4);
    S.landmark_probs.assign(M, std::vector<float>(V));
    for (auto& p : S.landmark_probs)
        f.read((char*)p.data(), (std::streamsize)V * 4);
    uint32_t l;
    f.read((char*)&l, 4);
    std::string s(l, '\0');
    f.read(s.data(), l);
    S.meta = json::parse(s);
    return S;
}

static void adam(std::vector<float>& p, std::vector<float>& g, std::vector<float>& m, std::vector<float>& v,
                 int t, float lr, float wd) {
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, c1 = 1 - std::pow(b1, t), c2 = 1 - std::pow(b2, t);
    for (size_t i = 0; i < p.size(); i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        float mh = m[i] / c1, vh = v[i] / c2;
        p[i] -= lr * (mh / (std::sqrt(vh) + eps) + wd * p[i]);
        g[i] = 0;
    }
}

SelfKnowledgeWeights train_self(const std::vector<Trace>& traces, const DistMatrix& D,
                                const std::vector<std::vector<float>>& probs, const A1Report& a1, int out_dim,
                                int epochs, uint64_t seed) {
    int N = (int)traces.size();
    int in_dim = traces[0].input_dim();
    int hidden = 256;
    SelfKnowledgeWeights S;
    S.in_dim = in_dim;
    S.hidden = hidden;
    S.out_dim = out_dim;
    S.energy_hidden = 64;
    std::vector<float> X((size_t)N * in_dim);
    for (int i = 0; i < N; i++) {
        auto f = traces[i].features();
        std::copy(f.begin(), f.end(), X.begin() + (size_t)i * in_dim);
    }
    S.mean.assign(in_dim, 0);
    S.std.assign(in_dim, 0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < in_dim; j++)
            S.mean[j] += X[(size_t)i * in_dim + j];
    for (auto& x : S.mean)
        x /= N;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < in_dim; j++) {
            float d = X[(size_t)i * in_dim + j] - S.mean[j];
            S.std[j] += d * d;
        }
    for (auto& x : S.std)
        x = std::sqrt(x / N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < in_dim; j++)
            X[(size_t)i * in_dim + j] = (X[(size_t)i * in_dim + j] - S.mean[j]) / std::max(1e-6f, S.std[j]);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0, 1);
    S.W1.resize((size_t)hidden * in_dim);
    S.b1.assign(hidden, 0);
    S.W2.resize((size_t)out_dim * hidden);
    S.b2.assign(out_dim, 0);
    float s1 = 1.f / std::sqrt(in_dim), s2 = 1.f / std::sqrt(hidden);
    for (auto& x : S.W1)
        x = nd(rng) * s1;
    for (auto& x : S.W2)
        x = nd(rng) * s2;
    // Positive graph and degree measure. If A1 gives a window, use its midpoint;
    // otherwise use a self-generated mutual-kNN graph (k=8) so the pilot encoder
    // still receives local behavioral structure. This fallback is [E], not A1.
    float eps_pos = a1.best.q ? 0.5f * (a1.best.eps_in + a1.best.eps_out) : -1.f;
    std::vector<float> W((size_t)N * N, 0), deg(N, 0);
    if (a1.best.q) {
        for (int i = 0; i < N; i++) {
            W[(size_t)i * N + i] = 1.f;
            deg[i] += 1.f;
            for (int j = i + 1; j < N; j++)
                if (D(i, j) <= eps_pos) {
                    W[(size_t)i * N + j] = W[(size_t)j * N + i] = 1.f;
                    deg[i]++;
                    deg[j]++;
                }
        }
    } else {
        int k = std::min(8, N - 1);
        std::vector<int> idx(N);
        for (int i = 0; i < N; i++) {
            std::iota(idx.begin(), idx.end(), 0);
            idx.erase(idx.begin() + i);
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [&](int a, int b) { return D(i, a) < D(i, b); });
            for (int t = 0; t < k; t++) {
                int j = idx[t];
                W[(size_t)i * N + j] = 1.f;
            }
        }
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++) {
                float w = std::max(W[(size_t)i * N + j], W[(size_t)j * N + i]);
                W[(size_t)i * N + j] = W[(size_t)j * N + i] = w;
            }
        for (int i = 0; i < N; i++) {
            W[(size_t)i * N + i] = 1.f;
            deg[i] = std::accumulate(W.begin() + (size_t)i * N, W.begin() + (size_t)(i + 1) * N, 0.f);
        }
    }
    float vol = std::accumulate(deg.begin(), deg.end(), 0.f);
    std::vector<float> mu(N), smu(N);
    for (int i = 0; i < N; i++) {
        mu[i] = deg[i] / vol;
        smu[i] = std::sqrt(mu[i]);
    }
    std::vector<float> A((size_t)N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[(size_t)i * N + j] = W[(size_t)i * N + j] / std::sqrt(deg[i] * deg[j]);
    S.landmarks = farthest_landmarks(D, 128, seed);
    auto psi = psi_embedding(probs, S.landmarks, (int)probs[0].size());
    S.landmark_probs.resize(S.landmarks.size());
    for (size_t i = 0; i < S.landmarks.size(); i++)
        S.landmark_probs[i] = probs[S.landmarks[i]];
    std::vector<float> H((size_t)N * hidden), F((size_t)N * out_dim), G((size_t)N * out_dim),
        C((size_t)out_dim * out_dim), dF((size_t)N * out_dim), dH((size_t)N * hidden),
        AG((size_t)N * out_dim);
    std::vector<float> mW1(S.W1.size()), vW1(S.W1.size()), mb1(hidden), vb1(hidden), mW2(S.W2.size()),
        vW2(S.W2.size()), mb2(out_dim), vb2(out_dim);
    std::vector<float> gW1(S.W1.size()), gb1(hidden), gW2(S.W2.size()), gb2(out_dim);
    std::vector<int> pi(8192), pj(8192);
    std::vector<float> psi_b(8192);
    std::uniform_int_distribution<int> ud(0, N - 1);
    for (size_t p = 0; p < pi.size(); p++) {
        pi[p] = ud(rng);
        pj[p] = ud(rng);
        float s = 0;
        for (int k = 0; k < (int)psi[0].size(); k++) {
            float d = psi[pi[p]][k] - psi[pj[p]][k];
            s += d * d;
        }
        psi_b[p] = s;
    }
    double loss_last = 0;
    for (int ep = 1; ep <= epochs; ep++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, hidden, in_dim, 1.f, X.data(), in_dim,
                    S.W1.data(), in_dim, 0.f, H.data(), hidden);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < hidden; j++)
                H[(size_t)i * hidden + j] = std::tanh(H[(size_t)i * hidden + j] + S.b1[j]);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, out_dim, hidden, 1.f, H.data(), hidden,
                    S.W2.data(), hidden, 0.f, F.data(), out_dim);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < out_dim; j++) {
                F[(size_t)i * out_dim + j] += S.b2[j];
                G[(size_t)i * out_dim + j] = smu[i] * F[(size_t)i * out_dim + j];
            }
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, out_dim, out_dim, N, 1.f, G.data(), out_dim,
                    G.data(), out_dim, 0.f, C.data(), out_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, out_dim, N, 1.f, A.data(), N, G.data(),
                    out_dim, 0.f, AG.data(), out_dim);
        double tr = 0, norm = 0;
        for (int j = 0; j < out_dim; j++) {
            tr += C[(size_t)j * out_dim + j];
            for (int k = 0; k < out_dim; k++)
                norm += C[(size_t)j * out_dim + k] * C[(size_t)j * out_dim + k];
        }
        double lsp = -2 * tr + norm;
        std::fill(dF.begin(), dF.end(), 0.f);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < out_dim; j++) {
                float gc = -4 * AG[(size_t)i * out_dim + j];
                float cc = 0;
                for (int k = 0; k < out_dim; k++)
                    cc += G[(size_t)i * out_dim + k] * C[(size_t)k * out_dim + j];
                gc += 4 * cc;
                float cw = 0;
                for (int k = 0; k < out_dim; k++)
                    cw += G[(size_t)i * out_dim + k] * (C[(size_t)k * out_dim + j] - (k == j ? 1.f : 0.f));
                gc += 4.f * cw;
                dF[(size_t)i * out_dim + j] = smu[i] * gc;
            }
        // Scale-invariant pairwise distillation (stop-grad through batch means).
        double ma = 0, mb = 0;
        std::vector<float> pa(pi.size());
        for (size_t p = 0; p < pi.size(); p++) {
            float s = 0;
            for (int j = 0; j < out_dim; j++) {
                float d = F[(size_t)pi[p] * out_dim + j] - F[(size_t)pj[p] * out_dim + j];
                s += d * d;
            }
            pa[p] = s;
            ma += s;
            mb += psi_b[p];
        }
        ma /= pi.size();
        mb /= pi.size();
        ma = std::max(ma, 1e-9);
        mb = std::max(mb, 1e-9);
        double ld = 0;
        float dw = 1.0f;
        for (size_t p = 0; p < pi.size(); p++) {
            float an = pa[p] / ma, bn = psi_b[p] / mb, err = an - bn;
            ld += err * err;
            float c = dw * 2 * err / ma / pi.size();
            for (int j = 0; j < out_dim; j++) {
                float d = F[(size_t)pi[p] * out_dim + j] - F[(size_t)pj[p] * out_dim + j];
                dF[(size_t)pi[p] * out_dim + j] += 2 * c * d;
                dF[(size_t)pj[p] * out_dim + j] -= 2 * c * d;
            }
        }
        ld /= pi.size();
        // backprop
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, hidden, out_dim, 1.f, dF.data(), out_dim,
                    S.W2.data(), out_dim, 0.f, dH.data(), hidden);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < hidden; j++) {
                float h = H[(size_t)i * hidden + j];
                dH[(size_t)i * hidden + j] *= (1 - h * h);
            }
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, out_dim, hidden, N, 1.f, dF.data(), out_dim,
                    H.data(), hidden, 0.f, gW2.data(), hidden);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, hidden, in_dim, N, 1.f, dH.data(), hidden,
                    X.data(), in_dim, 0.f, gW1.data(), in_dim);
        std::fill(gb1.begin(), gb1.end(), 0);
        std::fill(gb2.begin(), gb2.end(), 0);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < hidden; j++)
                gb1[j] += dH[(size_t)i * hidden + j];
            for (int j = 0; j < out_dim; j++)
                gb2[j] += dF[(size_t)i * out_dim + j];
        }
        float gn = 0;
        for (auto x : gW1)
            gn += x * x;
        for (auto x : gW2)
            gn += x * x;
        for (auto x : gb1)
            gn += x * x;
        for (auto x : gb2)
            gn += x * x;
        gn = std::sqrt(gn);
        float scale = gn > 1.f ? 1.f / gn : 1.f;
        if (scale < 1) {
            for (auto& x : gW1)
                x *= scale;
            for (auto& x : gW2)
                x *= scale;
            for (auto& x : gb1)
                x *= scale;
            for (auto& x : gb2)
                x *= scale;
        }
        float lr = 1e-3f;
        adam(S.W1, gW1, mW1, vW1, ep, lr, 1e-4f);
        adam(S.b1, gb1, mb1, vb1, ep, lr, 0);
        adam(S.W2, gW2, mW2, vW2, ep, lr, 1e-4f);
        adam(S.b2, gb2, mb2, vb2, ep, lr, 0);
        loss_last = lsp + ld;
        if (ep % 10 == 0)
            fprintf(stderr, "[self] ep %d loss %.4f spectral %.4f distill %.4f\n", ep, loss_last, lsp, ld);
    }
    // Output centering and whitening under degree measure mu.
    // Recompute final raw with b2.
    std::vector<float> mo(out_dim, 0);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, hidden, in_dim, 1.f, X.data(), in_dim,
                S.W1.data(), in_dim, 0.f, H.data(), hidden);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < hidden; j++)
            H[(size_t)i * hidden + j] = std::tanh(H[(size_t)i * hidden + j] + S.b1[j]);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, out_dim, hidden, 1.f, H.data(), hidden,
                S.W2.data(), hidden, 0.f, F.data(), out_dim);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < out_dim; j++)
            F[(size_t)i * out_dim + j] += S.b2[j];
    std::fill(mo.begin(), mo.end(), 0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < out_dim; j++)
            mo[j] += mu[i] * F[(size_t)i * out_dim + j];
    S.mean.insert(S.mean.end(), mo.begin(), mo.end());
    for (int i = 0; i < N; i++)
        for (int j = 0; j < out_dim; j++)
            G[(size_t)i * out_dim + j] = smu[i] * (F[(size_t)i * out_dim + j] - mo[j]);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, out_dim, out_dim, N, 1.f, G.data(), out_dim,
                G.data(), out_dim, 0.f, C.data(), out_dim);
    // C is symmetric; interpret the same buffer as column-major so eigenvectors are columns.
    std::vector<float> eig(out_dim);
    int info = LAPACKE_ssyevd(LAPACK_COL_MAJOR, 'V', 'U', out_dim, C.data(), out_dim, eig.data());
    if (info != 0)
        throw std::runtime_error("ssyevd failed");
    S.white.assign((size_t)out_dim * out_dim, 0);
    // row-vector z = r * white, white[i,j] = evec_j[i] / sqrt(eig_j) = C[i + j*d] / sqrt(eig_j)
    for (int i = 0; i < out_dim; i++)
        for (int j = 0; j < out_dim; j++)
            S.white[(size_t)i * out_dim + j] =
                C[(size_t)i + (size_t)j * out_dim] / std::sqrt(std::max(eig[j], 1e-6f));
    S.meta = {{"objective", "finite normalized-adjacency spectral contrastive + degree whitening + "
                            "scale-invariant psi distillation"},
              {"eps_pos", eps_pos},
              {"mu", "degree/vol"},
              {"epochs", epochs},
              {"note", "pilot; optimization path empirical"}};
    return S;
}

SelfKnowledgeWeights train_energy(SelfKnowledgeWeights S, const std::vector<Trace>& traces, int epochs,
                                  uint64_t seed) {
    int N = (int)traces.size(), d = S.out_dim, h = S.energy_hidden;
    std::vector<std::vector<float>> Z(N);
    for (int i = 0; i < N; i++)
        Z[i] = S.encode(traces[i]);
    float sigma = 1.0f;
    S.energy_sigma = sigma;
    S.energy_W.resize((size_t)h * d);
    S.energy_b.assign(h, 0);
    S.energy_a.assign(h, 0);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0, 1);
    float sw = 1.f / std::sqrt(d);
    for (auto& x : S.energy_W)
        x = nd(rng) * sw;
    for (auto& x : S.energy_a)
        x = nd(rng) * 0.01f;
    std::vector<float> mW(S.energy_W.size()), vW(S.energy_W.size()), mb(h), vb(h), ma(h), va(h),
        gW(S.energy_W.size()), gb(h), ga(h);
    std::vector<float> u(h), t(h), g(d), r(d), Wr(h);
    for (int ep = 1; ep <= epochs; ep++) {
        std::fill(gW.begin(), gW.end(), 0);
        std::fill(gb.begin(), gb.end(), 0);
        std::fill(ga.begin(), ga.end(), 0);
        double loss = 0;
        for (int i = 0; i < N; i++) {
            std::vector<float> eps(d), zn(d);
            for (int j = 0; j < d; j++) {
                eps[j] = nd(rng);
                zn[j] = Z[i][j] + sigma * eps[j];
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, h, d, 1.f, zn.data(), d,
                        S.energy_W.data(), d, 0.f, u.data(), h);
            for (int j = 0; j < h; j++)
                t[j] = std::tanh(u[j] + S.energy_b[j]);
            g = zn;
            for (int j = 0; j < h; j++) {
                float c = S.energy_a[j] * (1 - t[j] * t[j]);
                for (int k = 0; k < d; k++)
                    g[k] += S.energy_W[(size_t)j * d + k] * c;
            }
            for (int k = 0; k < d; k++) {
                r[k] = g[k] - eps[k] / sigma;
                loss += 0.5 * r[k] * r[k];
            }
            cblas_sgemv(CblasRowMajor, CblasNoTrans, h, d, 1.f, S.energy_W.data(), d, r.data(), 1, 0.f,
                        Wr.data(), 1);
            for (int j = 0; j < h; j++) {
                float sech2 = 1 - t[j] * t[j];
                ga[j] += Wr[j] * sech2;
                float common = -2 * S.energy_a[j] * t[j] * sech2 * Wr[j];
                gb[j] += common;
                for (int k = 0; k < d; k++)
                    gW[(size_t)j * d + k] += S.energy_a[j] * sech2 * r[k] + common * zn[k];
            }
        }
        float inv = 1.f / N;
        for (auto& x : gW)
            x *= inv;
        for (auto& x : gb)
            x *= inv;
        for (auto& x : ga)
            x *= inv;
        adam(S.energy_W, gW, mW, vW, ep, 1e-3f, 1e-5f);
        adam(S.energy_b, gb, mb, vb, ep, 1e-3f, 0);
        adam(S.energy_a, ga, ma, va, ep, 1e-3f, 0);
        if (ep % 50 == 0)
            fprintf(stderr, "[energy] ep %d dsm loss %.4f\n", ep, loss / N);
    }
    S.meta["energy"] = {{"method", "denoising score matching on conservative tanh energy"},
                        {"sigma", sigma},
                        {"epochs", epochs},
                        {"target", "grad E(z+sigma eps) ~ eps/sigma"},
                        {"status", "pilot empirical; not a population consistency claim"}};
    return S;
}
} // namespace alysane
