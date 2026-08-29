#include "distance.hpp"
#include "simd_math.hpp"

namespace vectorforge {

float euclidean_distance(std::span<const float> a, std::span<const float> b) {
    return simd::distance(a, b);
}

float euclidean_distance(const Vector& a, const Vector& b) {
    return simd::distance(a, b);
}

float euclidean_distance_scalar(std::span<const float> a, std::span<const float> b) {
    return simd::distance_scalar(a, b);
}

float euclidean_distance_scalar(const Vector& a, const Vector& b) {
    return simd::distance_scalar(std::span<const float>(a.data(), a.dimension()),
                                std::span<const float>(b.data(), b.dimension()));
}

} // namespace vectorforge
