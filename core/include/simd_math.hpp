#pragma once

#include "vector.hpp"

#include <cstddef>
#include <span>
#include <string>

namespace vectorforge::simd {

/**
 * @brief Detected CPU SIMD instruction set capabilities.
 */
struct CPUFeatures {
    bool has_avx{false};
    bool has_avx2{false};
    bool has_fma{false};
    bool has_avx512f{false};
    bool has_avx512dq{false};
    bool has_avx512vl{false};
    bool has_avx512bw{false};
    std::string cpu_brand;

    /// Returns true if AVX2 and FMA are fully supported by CPU and OS
    [[nodiscard]] bool supports_avx2() const noexcept {
        return has_avx2 && has_fma;
    }

    /// Returns true if AVX-512 Foundation and DQ are fully supported by CPU and OS
    [[nodiscard]] bool supports_avx512() const noexcept {
        return has_avx512f && has_avx512dq;
    }
};

/**
 * @brief Returns the cached CPU features detected at runtime.
 */
[[nodiscard]] const CPUFeatures& get_cpu_features() noexcept;

// =========================================================================
// Euclidean Distance (L2) Functions
// =========================================================================

/// Scalar Euclidean distance (Phase 1 ground-truth reference)
[[nodiscard]] float distance_scalar(std::span<const float> a, std::span<const float> b);

/// AVX2 + FMA 256-bit SIMD Euclidean distance (8 floats/instruction + scalar tail)
[[nodiscard]] float distance_avx2(std::span<const float> a, std::span<const float> b);

/// AVX-512 512-bit SIMD Euclidean distance (16 floats/instruction + scalar tail)
[[nodiscard]] float distance_avx512(std::span<const float> a, std::span<const float> b);

/// Automatically selects and executes the fastest supported Euclidean distance kernel
[[nodiscard]] float distance(std::span<const float> a, std::span<const float> b);

/// Vector overload for dispatched Euclidean distance
[[nodiscard]] inline float distance(const Vector& a, const Vector& b) {
    return distance(std::span<const float>(a.data(), a.dimension()),
                    std::span<const float>(b.data(), b.dimension()));
}

// =========================================================================
// Cosine Similarity Functions
// =========================================================================

/// Scalar Cosine similarity: dot(a, b) / (||a|| * ||b||)
[[nodiscard]] float cosine_similarity_scalar(std::span<const float> a, std::span<const float> b);

/// AVX2 + FMA 256-bit SIMD Cosine similarity
[[nodiscard]] float cosine_similarity_avx2(std::span<const float> a, std::span<const float> b);

/// AVX-512 512-bit SIMD Cosine similarity
[[nodiscard]] float cosine_similarity_avx512(std::span<const float> a, std::span<const float> b);

/// Automatically selects and executes the fastest supported Cosine similarity kernel
[[nodiscard]] float cosine_similarity(std::span<const float> a, std::span<const float> b);

/// Vector overload for dispatched Cosine similarity
[[nodiscard]] inline float cosine_similarity(const Vector& a, const Vector& b) {
    return cosine_similarity(std::span<const float>(a.data(), a.dimension()),
                             std::span<const float>(b.data(), b.dimension()));
}

} // namespace vectorforge::simd
