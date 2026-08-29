#pragma once

#include "vector.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vectorforge {

/**
 * @brief Configuration parameters for Product Quantization.
 */
struct PQConfig {
    std::size_t dimension{128};        ///< Total vector dimensionality D
    std::size_t num_subspaces{16};     ///< Number of subvectors m (D must be divisible by m)
    std::size_t num_centroids{256};    ///< Number of centroids k per subspace (default: 256 for 8-bit codes)
    std::size_t max_iterations{25};    ///< Maximum K-means clustering iterations
    uint32_t random_seed{42};          ///< Deterministic seed for centroid initialization
};

/// Type alias for encoded quantized vector (m bytes for k <= 256)
using PQCode = std::vector<uint8_t>;

/**
 * @brief Precomputed lookup table for Asymmetric Distance Computation (ADC).
 * 
 * Stores squared Euclidean distances from query subvectors to all centroids across all subspaces.
 */
struct ADCLookupTable {
    std::size_t num_subspaces{0};
    std::size_t num_centroids{0};
    std::vector<float> data; ///< Flat array of size [num_subspaces * num_centroids]

    [[nodiscard]] inline float get(std::size_t subspace, std::size_t centroid_idx) const noexcept {
        return data[subspace * num_centroids + centroid_idx];
    }
};

/**
 * @brief Product Quantizer for lossy vector compression and fast ADC approximate retrieval.
 */
class ProductQuantizer {
public:
    /// Default constructor
    ProductQuantizer() = default;

    /// Constructor with configuration
    explicit ProductQuantizer(PQConfig config);

    /**
     * @brief Trains codebooks for all subspaces using K-means clustering.
     * 
     * @param training_data Collection of vectors for learning subspace centroids
     * @throws std::invalid_argument If training data is empty, dimensions mismatch, or size < k
     */
    void train(std::span<const Vector> training_data);

    /**
     * @brief Encodes a D-dimensional float vector into a compact m-byte PQ code.
     * 
     * @param vector Input floating-point vector
     * @return PQCode Sequence of centroid indices (one uint8_t per subspace)
     * @throws std::runtime_error If quantizer is not trained
     * @throws std::invalid_argument If vector dimension does not match
     */
    [[nodiscard]] PQCode encode(const Vector& vector) const;

    /**
     * @brief Decodes an m-byte PQ code back to an approximate D-dimensional float vector.
     * 
     * @param code Compressed PQ code
     * @return Vector Reconstructed floating-point vector
     * @throws std::runtime_error If quantizer is not trained
     * @throws std::invalid_argument If code size != m
     */
    [[nodiscard]] Vector decode(const PQCode& code) const;

    /**
     * @brief Precomputes an ADC lookup table for a given query vector.
     * 
     * @param query Unquantized floating-point query vector
     * @return ADCLookupTable Precalculated squared distances to all centroids
     * @throws std::runtime_error If quantizer is not trained
     * @throws std::invalid_argument If query dimension does not match
     */
    [[nodiscard]] ADCLookupTable compute_adc_table(const Vector& query) const;

    /**
     * @brief Computes approximate Euclidean distance between a query and a PQ code using a precomputed LUT.
     * 
     * Formula: sqrt(sum_{s=0}^{m-1} LUT[s][code[s]])
     * 
     * @param lut Precomputed ADC lookup table
     * @param code Compressed database vector PQ code
     * @return float Approximate Euclidean distance
     */
    [[nodiscard]] float approximate_distance(const ADCLookupTable& lut, const PQCode& code) const;

    /**
     * @brief Computes approximate Euclidean distance directly from query to PQ code (convenience overload).
     * 
     * @param query Unquantized query vector
     * @param code Compressed database vector PQ code
     * @return float Approximate Euclidean distance
     */
    [[nodiscard]] float approximate_distance(const Vector& query, const PQCode& code) const;

    /// Returns true if codebooks have been trained
    [[nodiscard]] bool is_trained() const noexcept;

    /// Returns the configuration
    [[nodiscard]] const PQConfig& config() const noexcept;

    /// Returns the subvector dimensionality d_s = D / m
    [[nodiscard]] std::size_t subvector_dim() const noexcept;

    /// Returns uncompressed byte footprint per vector (D * sizeof(float))
    [[nodiscard]] std::size_t original_bytes_per_vector() const noexcept;

    /// Returns compressed byte footprint per vector (m * sizeof(uint8_t))
    [[nodiscard]] std::size_t compressed_bytes_per_vector() const noexcept;

    /// Returns memory compression ratio (original_bytes / compressed_bytes)
    [[nodiscard]] double compression_ratio() const noexcept;

    /// Access raw centroids of subspace s (shape: num_centroids * subvector_dim)
    [[nodiscard]] const std::vector<float>& subspace_centroids(std::size_t subspace) const;

private:
    PQConfig config_;
    std::size_t subvector_dim_{0};
    bool is_trained_{false};

    // codebooks_[s] stores the flat float centroids for subspace s (size: k * subvector_dim)
    std::vector<std::vector<float>> codebooks_;

    void validate_config();
};

} // namespace vectorforge
