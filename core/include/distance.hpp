#pragma once

#include "vector.hpp"
#include "simd_math.hpp"

#include <cstddef>
#include <span>

namespace vectorforge {

/**
 * @brief Computes Euclidean distance (L2 distance) using runtime SIMD dispatch (AVX-512 -> AVX2 -> Scalar).
 * 
 * Formula: distance(A, B) = sqrt(sum((A[i] - B[i])^2))
 * 
 * @param a First float span
 * @param b Second float span
 * @return float Euclidean distance
 * @throws std::invalid_argument if dimensions do not match
 */
[[nodiscard]] float euclidean_distance(std::span<const float> a, std::span<const float> b);

/**
 * @brief Computes Euclidean distance (L2 distance) between two Vector objects using runtime SIMD dispatch.
 * 
 * @param a First Vector
 * @param b Second Vector
 * @return float Euclidean distance
 * @throws std::invalid_argument if dimensions do not match
 */
[[nodiscard]] float euclidean_distance(const Vector& a, const Vector& b);

/**
 * @brief Explicit scalar Euclidean distance calculation (Phase 1 ground-truth reference).
 */
[[nodiscard]] float euclidean_distance_scalar(std::span<const float> a, std::span<const float> b);

/**
 * @brief Explicit scalar Euclidean distance between two Vector objects.
 */
[[nodiscard]] float euclidean_distance_scalar(const Vector& a, const Vector& b);

} // namespace vectorforge
