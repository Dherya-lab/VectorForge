#include "product_quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace vectorforge {

ProductQuantizer::ProductQuantizer(PQConfig config)
    : config_(config) {
    validate_config();
}

void ProductQuantizer::validate_config() {
    if (config_.dimension == 0) {
        throw std::invalid_argument("PQ dimension D must be greater than 0");
    }
    if (config_.num_subspaces == 0) {
        throw std::invalid_argument("PQ num_subspaces m must be greater than 0");
    }
    if (config_.num_centroids == 0 || config_.num_centroids > 256) {
        throw std::invalid_argument("PQ num_centroids k must be between 1 and 256 (for 8-bit codes)");
    }
    if (config_.dimension % config_.num_subspaces != 0) {
        throw std::invalid_argument("Vector dimension D (" + std::to_string(config_.dimension) +
                                    ") must be divisible by num_subspaces m (" +
                                    std::to_string(config_.num_subspaces) + ")");
    }
    subvector_dim_ = config_.dimension / config_.num_subspaces;
}

void ProductQuantizer::train(std::span<const Vector> training_data) {
    if (training_data.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }
    if (training_data.size() < config_.num_centroids) {
        throw std::invalid_argument("Insufficient training data: requires at least " +
                                    std::to_string(config_.num_centroids) +
                                    " vectors, got " + std::to_string(training_data.size()));
    }

    validate_config();

    for (std::size_t i = 0; i < training_data.size(); ++i) {
        if (training_data[i].dimension() != config_.dimension) {
            throw std::invalid_argument("Training vector " + std::to_string(i) +
                                        " dimension mismatch: expected " +
                                        std::to_string(config_.dimension) + ", got " +
                                        std::to_string(training_data[i].dimension()));
        }
    }

    const std::size_t N = training_data.size();
    const std::size_t m = config_.num_subspaces;
    const std::size_t k = config_.num_centroids;
    const std::size_t ds = subvector_dim_;

    codebooks_.resize(m);

    // Train K-means codebook independently for each subspace s
    for (std::size_t s = 0; s < m; ++s) {
        std::mt19937 rng(config_.random_seed + static_cast<uint32_t>(s * 10007));

        // 1. Extract subspace data: flat array of size [N * ds]
        std::vector<float> subspace_data(N * ds);
        for (std::size_t i = 0; i < N; ++i) {
            const float* src = training_data[i].data() + (s * ds);
            std::copy(src, src + ds, subspace_data.data() + (i * ds));
        }

        // 2. Initialize k centroids randomly from the training subvectors
        std::vector<std::size_t> sample_indices(N);
        std::iota(sample_indices.begin(), sample_indices.end(), 0);
        std::shuffle(sample_indices.begin(), sample_indices.end(), rng);

        std::vector<float> centroids(k * ds);
        for (std::size_t c = 0; c < k; ++c) {
            const std::size_t sample_idx = sample_indices[c];
            const float* src = subspace_data.data() + (sample_idx * ds);
            std::copy(src, src + ds, centroids.data() + (c * ds));
        }

        std::vector<std::size_t> assignments(N, 0);
        std::vector<std::size_t> cluster_counts(k, 0);
        std::vector<float> centroid_sums(k * ds, 0.0f);

        // 3. K-Means Lloyd iterations
        for (std::size_t iter = 0; iter < config_.max_iterations; ++iter) {
            bool changed = false;

            // Assignment step
            for (std::size_t i = 0; i < N; ++i) {
                const float* x = subspace_data.data() + (i * ds);
                float best_dist = std::numeric_limits<float>::max();
                std::size_t best_c = 0;

                for (std::size_t c = 0; c < k; ++c) {
                    const float* cent = centroids.data() + (c * ds);
                    float d = 0.0f;
                    for (std::size_t d_idx = 0; d_idx < ds; ++d_idx) {
                        const float diff = x[d_idx] - cent[d_idx];
                        d += diff * diff;
                    }
                    if (d < best_dist) {
                        best_dist = d;
                        best_c = c;
                    }
                }

                if (assignments[i] != best_c) {
                    assignments[i] = best_c;
                    changed = true;
                }
            }

            if (!changed && iter > 0) {
                break; // Converged
            }

            // Update step
            std::fill(cluster_counts.begin(), cluster_counts.end(), 0);
            std::fill(centroid_sums.begin(), centroid_sums.end(), 0.0f);

            for (std::size_t i = 0; i < N; ++i) {
                const std::size_t c = assignments[i];
                ++cluster_counts[c];
                const float* x = subspace_data.data() + (i * ds);
                float* sum_ptr = centroid_sums.data() + (c * ds);
                for (std::size_t d_idx = 0; d_idx < ds; ++d_idx) {
                    sum_ptr[d_idx] += x[d_idx];
                }
            }

            for (std::size_t c = 0; c < k; ++c) {
                float* cent = centroids.data() + (c * ds);
                if (cluster_counts[c] > 0) {
                    const float inv_count = 1.0f / static_cast<float>(cluster_counts[c]);
                    const float* sum_ptr = centroid_sums.data() + (c * ds);
                    for (std::size_t d_idx = 0; d_idx < ds; ++d_idx) {
                        cent[d_idx] = sum_ptr[d_idx] * inv_count;
                    }
                } else {
                    // Empty cluster fallback: reinitialize to random training vector
                    std::uniform_int_distribution<std::size_t> dist_n(0, N - 1);
                    const std::size_t rand_i = dist_n(rng);
                    const float* x = subspace_data.data() + (rand_i * ds);
                    std::copy(x, x + ds, cent);
                }
            }
        }

        codebooks_[s] = std::move(centroids);
    }

    is_trained_ = true;
}

PQCode ProductQuantizer::encode(const Vector& vector) const {
    if (!is_trained_) {
        throw std::runtime_error("ProductQuantizer must be trained before encoding");
    }
    if (vector.dimension() != config_.dimension) {
        throw std::invalid_argument("Vector dimension mismatch: expected " +
                                    std::to_string(config_.dimension) + ", got " +
                                    std::to_string(vector.dimension()));
    }

    const std::size_t m = config_.num_subspaces;
    const std::size_t k = config_.num_centroids;
    const std::size_t ds = subvector_dim_;

    PQCode code(m);

    for (std::size_t s = 0; s < m; ++s) {
        const float* sub_vec = vector.data() + (s * ds);
        const float* centroids = codebooks_[s].data();

        float min_dist = std::numeric_limits<float>::max();
        std::size_t best_c = 0;

        for (std::size_t c = 0; c < k; ++c) {
            const float* cent = centroids + (c * ds);
            float dist = 0.0f;
            for (std::size_t d = 0; d < ds; ++d) {
                const float diff = sub_vec[d] - cent[d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_c = c;
            }
        }

        code[s] = static_cast<uint8_t>(best_c);
    }

    return code;
}

Vector ProductQuantizer::decode(const PQCode& code) const {
    if (!is_trained_) {
        throw std::runtime_error("ProductQuantizer must be trained before decoding");
    }
    if (code.size() != config_.num_subspaces) {
        throw std::invalid_argument("PQ code size mismatch: expected " +
                                    std::to_string(config_.num_subspaces) + ", got " +
                                    std::to_string(code.size()));
    }

    const std::size_t m = config_.num_subspaces;
    const std::size_t ds = subvector_dim_;

    std::vector<float> reconstructed(config_.dimension);

    for (std::size_t s = 0; s < m; ++s) {
        const std::size_t c = code[s];
        const float* cent = codebooks_[s].data() + (c * ds);
        float* dst = reconstructed.data() + (s * ds);
        std::copy(cent, cent + ds, dst);
    }

    return Vector(std::move(reconstructed));
}

ADCLookupTable ProductQuantizer::compute_adc_table(const Vector& query) const {
    if (!is_trained_) {
        throw std::runtime_error("ProductQuantizer must be trained before computing ADC table");
    }
    if (query.dimension() != config_.dimension) {
        throw std::invalid_argument("Query dimension mismatch for ADC: expected " +
                                    std::to_string(config_.dimension) + ", got " +
                                    std::to_string(query.dimension()));
    }

    const std::size_t m = config_.num_subspaces;
    const std::size_t k = config_.num_centroids;
    const std::size_t ds = subvector_dim_;

    ADCLookupTable table{
        .num_subspaces = m,
        .num_centroids = k,
        .data = std::vector<float>(m * k)
    };

    for (std::size_t s = 0; s < m; ++s) {
        const float* query_sub = query.data() + (s * ds);
        const float* centroids = codebooks_[s].data();
        float* table_row = table.data.data() + (s * k);

        for (std::size_t c = 0; c < k; ++c) {
            const float* cent = centroids + (c * ds);
            float dist_sq = 0.0f;
            for (std::size_t d = 0; d < ds; ++d) {
                const float diff = query_sub[d] - cent[d];
                dist_sq += diff * diff;
            }
            table_row[c] = dist_sq;
        }
    }

    return table;
}

float ProductQuantizer::approximate_distance(const ADCLookupTable& lut, const PQCode& code) const {
    if (code.size() != lut.num_subspaces) {
        throw std::invalid_argument("PQ code size mismatch with ADC LUT: expected " +
                                    std::to_string(lut.num_subspaces) + ", got " +
                                    std::to_string(code.size()));
    }

    float total_sq_dist = 0.0f;
    for (std::size_t s = 0; s < lut.num_subspaces; ++s) {
        total_sq_dist += lut.get(s, code[s]);
    }

    return std::sqrt(total_sq_dist);
}

float ProductQuantizer::approximate_distance(const Vector& query, const PQCode& code) const {
    const auto lut = compute_adc_table(query);
    return approximate_distance(lut, code);
}

bool ProductQuantizer::is_trained() const noexcept {
    return is_trained_;
}

const PQConfig& ProductQuantizer::config() const noexcept {
    return config_;
}

std::size_t ProductQuantizer::subvector_dim() const noexcept {
    return subvector_dim_;
}

std::size_t ProductQuantizer::original_bytes_per_vector() const noexcept {
    return config_.dimension * sizeof(float);
}

std::size_t ProductQuantizer::compressed_bytes_per_vector() const noexcept {
    return config_.num_subspaces * sizeof(uint8_t);
}

double ProductQuantizer::compression_ratio() const noexcept {
    return static_cast<double>(original_bytes_per_vector()) /
           static_cast<double>(compressed_bytes_per_vector());
}

const std::vector<float>& ProductQuantizer::subspace_centroids(std::size_t subspace) const {
    if (subspace >= codebooks_.size()) {
        throw std::out_of_range("Subspace index out of range: " + std::to_string(subspace));
    }
    return codebooks_[subspace];
}

} // namespace vectorforge
