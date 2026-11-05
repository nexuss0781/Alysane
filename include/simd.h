#pragma once
// Portable SIMD kernels with runtime dispatch.
//
// The original core hardcoded AVX-512 intrinsics, which fails to compile (and
// would fail to run) on CPUs without AVX-512 — including standard cloud CI
// runners and most commercial serving fleets. This header provides:
//   * an AVX-512 path, compiled only when the compiler targets it,
//   * an AVX2+FMA path, compiled only when available,
//   * a portable scalar fallback that is always correct.
// Dispatch is resolved once via __builtin_cpu_supports.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define ALYSANE_X86 1
#endif

namespace alysane {
namespace simd {

enum class Isa { Scalar, Avx2, Avx512 };

inline Isa detect() {
    // Resolved exactly once. C++11 guarantees thread-safe static initialisation.
    // Calling __builtin_cpu_init() concurrently from OpenMP threads is a data
    // race on the global __cpu_model, so it must not run per-call.
    static const Isa cached = []() -> Isa {
        // Diagnostic override: ALYSANE_SIMD=scalar|avx2|avx512
        if (const char* e = std::getenv("ALYSANE_SIMD")) {
            if (!std::strcmp(e, "scalar")) return Isa::Scalar;
            if (!std::strcmp(e, "avx2")) return Isa::Avx2;
            if (!std::strcmp(e, "avx512")) return Isa::Avx512;
        }
#if ALYSANE_X86
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx512f")) return Isa::Avx512;
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return Isa::Avx2;
#endif
        return Isa::Scalar;
    }();
    return cached;
}

inline const char* isa_name() {
    switch (detect()) {
    case Isa::Avx512: return "avx512f";
    case Isa::Avx2: return "avx2+fma";
    default: return "scalar";
    }
}

// ---- dot product ----
#if ALYSANE_X86
__attribute__((target("avx512f")))
inline float dot_avx512(const float* a, const float* b, int n) {
    __m512 s = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        s = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s);
    float r = _mm512_reduce_add_ps(s);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
}
#endif

#if ALYSANE_X86
__attribute__((target("avx2,fma")))
inline float dot_avx2(const float* a, const float* b, int n) {
    __m256 s = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8)
        s = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s);
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 q = _mm_add_ps(lo, hi);
    q = _mm_hadd_ps(q, q);
    q = _mm_hadd_ps(q, q);
    float r = _mm_cvtss_f32(q);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
}
#endif

inline float dot_scalar(const float* a, const float* b, int n) {
    float r = 0.f;
    for (int i = 0; i < n; i++) r += a[i] * b[i];
    return r;
}

inline float dot(const float* a, const float* b, int n) {
#if ALYSANE_X86
    if (detect() == Isa::Avx512) return dot_avx512(a, b, n);
#endif
#if ALYSANE_X86
    if (detect() == Isa::Avx2) return dot_avx2(a, b, n);
#endif
    return dot_scalar(a, b, n);
}

// ---- L1 distance (sum of absolute differences) ----
#if ALYSANE_X86
__attribute__((target("avx512f")))
inline float l1_avx512(const float* a, const float* b, int n) {
    const __m512i mask = _mm512_set1_epi32(0x7fffffff);
    __m512 s = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        // Integer AND is AVX512F; the float AND intrinsic would require AVX512DQ.
        s = _mm512_add_ps(s, _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(d), mask)));
    }
    float r = _mm512_reduce_add_ps(s);
    for (; i < n; i++) r += std::fabs(a[i] - b[i]);
    return r;
}
#endif

#if ALYSANE_X86
__attribute__((target("avx2")))
inline float l1_avx2(const float* a, const float* b, int n) {
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 s = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        s = _mm256_add_ps(s, _mm256_and_ps(d, mask));
    }
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 q = _mm_add_ps(lo, hi);
    q = _mm_hadd_ps(q, q);
    q = _mm_hadd_ps(q, q);
    float r = _mm_cvtss_f32(q);
    for (; i < n; i++) r += std::fabs(a[i] - b[i]);
    return r;
}
#endif

inline float l1_scalar(const float* a, const float* b, int n) {
    float r = 0.f;
    for (int i = 0; i < n; i++) r += std::fabs(a[i] - b[i]);
    return r;
}

inline float l1(const float* a, const float* b, int n) {
#if ALYSANE_X86
    if (detect() == Isa::Avx512) return l1_avx512(a, b, n);
#endif
#if ALYSANE_X86
    if (detect() == Isa::Avx2) return l1_avx2(a, b, n);
#endif
    return l1_scalar(a, b, n);
}

} // namespace simd
} // namespace alysane
