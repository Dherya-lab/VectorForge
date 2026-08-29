#include "simd_math.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(_MSC_VER)
    #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
    #include <cpuid.h>
    #include <x86intrin.h>
#endif

namespace vectorforge::simd {

namespace {

// =========================================================================
// CPUID & Hardware Feature Detection
// =========================================================================

void execute_cpuid(int cpu_info[4], int function_id, int subfunction_id = 0) {
#if defined(_MSC_VER)
    __cpuidex(cpu_info, function_id, subfunction_id);
#elif defined(__GNUC__) || defined(__clang__)
    __cpuid_count(function_id, subfunction_id, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#else
    cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;
#endif
}

uint64_t read_xcr0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<uint64_t>(edx) << 32) | eax;
#else
    return 0;
#endif
}

CPUFeatures detect_features_internal() {
    CPUFeatures f;
    int info[4] = {0};

    execute_cpuid(info, 0);
    const int max_basic_leaf = info[0];

    // Read CPU Brand String (Leaves 0x80000002 .. 0x80000004)
    execute_cpuid(info, static_cast<int>(0x80000000));
    const unsigned int max_extended_leaf = static_cast<unsigned int>(info[0]);
    if (max_extended_leaf >= 0x80000004) {
        char brand[49] = {0};
        for (unsigned int i = 0; i < 3; ++i) {
            execute_cpuid(info, static_cast<int>(0x80000002 + i));
            std::memcpy(brand + (i * 16), info, sizeof(info));
        }
        f.cpu_brand = std::string(brand);
        // Trim leading spaces
        const auto first_char = f.cpu_brand.find_first_not_of(' ');
        if (first_char != std::string::npos) {
            f.cpu_brand = f.cpu_brand.substr(first_char);
        }
    } else {
        f.cpu_brand = "x86_64 Processor";
    }

    if (max_basic_leaf >= 1) {
        execute_cpuid(info, 1);
        const int ecx = info[2];

        const bool osxsave = (ecx & (1 << 27)) != 0;
        const bool cpu_avx = (ecx & (1 << 28)) != 0;
        const bool cpu_fma = (ecx & (1 << 12)) != 0;

        if (osxsave) {
            const uint64_t xcr0 = read_xcr0();
            const bool avx_os_enabled = (xcr0 & 0x6) == 0x6; // XMM and YMM state enabled by OS
            const bool avx512_os_enabled = (xcr0 & 0xE6) == 0xE6; // Opmask, ZMM_Hi256, Hi16_ZMM

            if (avx_os_enabled && cpu_avx) {
                f.has_avx = true;
                f.has_fma = cpu_fma;

                if (max_basic_leaf >= 7) {
                    execute_cpuid(info, 7, 0);
                    const int ebx = info[1];

                    f.has_avx2 = (ebx & (1 << 5)) != 0;

                    if (avx512_os_enabled) {
                        f.has_avx512f = (ebx & (1 << 16)) != 0;
                        f.has_avx512dq = (ebx & (1 << 17)) != 0;
                        f.has_avx512bw = (ebx & (1 << 30)) != 0;
                        f.has_avx512vl = (ebx & (1 << 31)) != 0;
                    }
                }
            }
        }
    }

    return f;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
inline float horizontal_sum_avx2(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 h = _mm_add_ps(lo, hi);
    h = _mm_hadd_ps(h, h);
    h = _mm_hadd_ps(h, h);
    return _mm_cvtss_f32(h);
}

} // anonymous namespace

const CPUFeatures& get_cpu_features() noexcept {
    static const CPUFeatures features = detect_features_internal();
    return features;
}

// =========================================================================
// Scalar Implementations
// =========================================================================

float distance_scalar(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions mismatch for distance: " +
                                    std::to_string(a.size()) + " != " + std::to_string(b.size()));
    }

    float sum_sq = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float diff = a[i] - b[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}

float cosine_similarity_scalar(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions mismatch for cosine similarity: " +
                                    std::to_string(a.size()) + " != " + std::to_string(b.size()));
    }

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    const float denominator = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denominator <= 1e-12f) {
        return 0.0f; // Safe handling for zero-norm vectors
    }
    return dot / denominator;
}

// =========================================================================
// AVX2 Implementations (256-bit SIMD)
// =========================================================================

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
float distance_avx2(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions mismatch for AVX2 distance: " +
                                    std::to_string(a.size()) + " != " + std::to_string(b.size()));
    }

    const std::size_t n = a.size();
    const float* ptr_a = a.data();
    const float* ptr_b = b.data();

    std::size_t i = 0;
    __m256 sum_vec = _mm256_setzero_ps();

    for (; i + 8 <= n; i += 8) {
        const __m256 va = _mm256_loadu_ps(ptr_a + i);
        const __m256 vb = _mm256_loadu_ps(ptr_b + i);
        const __m256 diff = _mm256_sub_ps(va, vb);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }

    float total = horizontal_sum_avx2(sum_vec);

    // Scalar tail for remainder
    for (; i < n; ++i) {
        const float diff = ptr_a[i] - ptr_b[i];
        total += diff * diff;
    }

    return std::sqrt(total);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
float cosine_similarity_avx2(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions mismatch for AVX2 cosine similarity: " +
                                    std::to_string(a.size()) + " != " + std::to_string(b.size()));
    }

    const std::size_t n = a.size();
    const float* ptr_a = a.data();
    const float* ptr_b = b.data();

    std::size_t i = 0;
    __m256 dot_vec = _mm256_setzero_ps();
    __m256 norm_a_vec = _mm256_setzero_ps();
    __m256 norm_b_vec = _mm256_setzero_ps();

    for (; i + 8 <= n; i += 8) {
        const __m256 va = _mm256_loadu_ps(ptr_a + i);
        const __m256 vb = _mm256_loadu_ps(ptr_b + i);
        dot_vec = _mm256_fmadd_ps(va, vb, dot_vec);
        norm_a_vec = _mm256_fmadd_ps(va, va, norm_a_vec);
        norm_b_vec = _mm256_fmadd_ps(vb, vb, norm_b_vec);
    }

    float dot = horizontal_sum_avx2(dot_vec);
    float norm_a = horizontal_sum_avx2(norm_a_vec);
    float norm_b = horizontal_sum_avx2(norm_b_vec);

    for (; i < n; ++i) {
        dot += ptr_a[i] * ptr_b[i];
        norm_a += ptr_a[i] * ptr_a[i];
        norm_b += ptr_b[i] * ptr_b[i];
    }

    const float denominator = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denominator <= 1e-12f) {
        return 0.0f;
    }
    return dot / denominator;
}

// =========================================================================
// AVX-512 Implementations (512-bit SIMD)
// =========================================================================

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512dq,fma")))
#endif
float distance_avx512(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions mismatch for AVX-512 distance: " +
                                    std::to_string(a.size()) + " != " + std::to_string(b.size()));
    }

    const std::size_t n = a.size();
    const float* ptr_a = a.data();
    const float* ptr_b = b.data();

    std::size_t i = 0;
    __m512 sum_vec = _mm512_setzero_ps();

    for (; i + 16 <= n; i += 16) {
        const __m512 va = _mm512_loadu_ps(ptr_a + i);
        const __m512 vb = _mm512_loadu_ps(ptr_b + i);
        const __m512 diff = _mm512_sub_ps(va, vb);
        sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
    }

    float total = _mm512_reduce_add_ps(sum_vec);

    // Scalar tail for remainder
    for (; i < n; ++i) {
        const float diff = ptr_a[i] - ptr_b[i];
        total += diff * diff;
    }

    return std::sqrt(total);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512dq,fma")))
#endif
float cosine_similarity_avx512(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions mismatch for AVX-512 cosine similarity: " +
                                    std::to_string(a.size()) + " != " + std::to_string(b.size()));
    }

    const std::size_t n = a.size();
    const float* ptr_a = a.data();
    const float* ptr_b = b.data();

    std::size_t i = 0;
    __m512 dot_vec = _mm512_setzero_ps();
    __m512 norm_a_vec = _mm512_setzero_ps();
    __m512 norm_b_vec = _mm512_setzero_ps();

    for (; i + 16 <= n; i += 16) {
        const __m512 va = _mm512_loadu_ps(ptr_a + i);
        const __m512 vb = _mm512_loadu_ps(ptr_b + i);
        dot_vec = _mm512_fmadd_ps(va, vb, dot_vec);
        norm_a_vec = _mm512_fmadd_ps(va, va, norm_a_vec);
        norm_b_vec = _mm512_fmadd_ps(vb, vb, norm_b_vec);
    }

    float dot = _mm512_reduce_add_ps(dot_vec);
    float norm_a = _mm512_reduce_add_ps(norm_a_vec);
    float norm_b = _mm512_reduce_add_ps(norm_b_vec);

    for (; i < n; ++i) {
        dot += ptr_a[i] * ptr_b[i];
        norm_a += ptr_a[i] * ptr_a[i];
        norm_b += ptr_b[i] * ptr_b[i];
    }

    const float denominator = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denominator <= 1e-12f) {
        return 0.0f;
    }
    return dot / denominator;
}

// =========================================================================
// Runtime Dynamic Dispatch
// =========================================================================

using DistanceFn = float (*)(std::span<const float>, std::span<const float>);
using CosineFn = float (*)(std::span<const float>, std::span<const float>);

static DistanceFn select_distance_kernel() noexcept {
    const auto& features = get_cpu_features();
    if (features.supports_avx512()) {
        return &distance_avx512;
    }
    if (features.supports_avx2()) {
        return &distance_avx2;
    }
    return &distance_scalar;
}

static CosineFn select_cosine_kernel() noexcept {
    const auto& features = get_cpu_features();
    if (features.supports_avx512()) {
        return &cosine_similarity_avx512;
    }
    if (features.supports_avx2()) {
        return &cosine_similarity_avx2;
    }
    return &cosine_similarity_scalar;
}

float distance(std::span<const float> a, std::span<const float> b) {
    static const DistanceFn kernel = select_distance_kernel();
    return kernel(a, b);
}

float cosine_similarity(std::span<const float> a, std::span<const float> b) {
    static const CosineFn kernel = select_cosine_kernel();
    return kernel(a, b);
}

} // namespace vectorforge::simd
