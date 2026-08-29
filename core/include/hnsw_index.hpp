#pragma once

#include "vector.hpp"
#include "distance.hpp"
#include "vector_index.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace vectorforge {

/**
 * @brief Configuration parameters for HNSW index.
 */
struct HNSWConfig {
    std::size_t M{16};                 ///< Maximum outgoing connections per node per layer (> 0)
    std::size_t ef_construction{100};  ///< Size of dynamic candidate list during construction
    std::size_t ef_search{50};         ///< Size of dynamic candidate list during search
    uint32_t random_seed{42};          ///< Seed for deterministic level generation
};

/**
 * @brief Performance and diagnostics telemetry captured during search queries.
 */
struct HNSWStats {
    std::size_t nodes_visited{0};          ///< Number of distinct graph nodes evaluated
    std::size_t distance_computations{0};  ///< Total Euclidean distance calculations performed
};

/**
 * @brief Hierarchical Navigable Small World (HNSW) graph-based approximate nearest neighbor index.
 */
class HNSWIndex {
public:
    /// Constructs an HNSW index with optional custom configuration and dimension constraint
    explicit HNSWIndex(HNSWConfig config = {}, std::size_t dimension = 0);

    /**
     * @brief Inserts a vector into the HNSW graph.
     * 
     * @param vector Dense float vector to insert
     * @return std::size_t Assigned sequential vector ID (0, 1, 2, ...)
     * @throws std::invalid_argument If vector dimension does not match index dimension or is empty
     */
    std::size_t insert(const Vector& vector);

    /**
     * @brief Inserts a vector into the HNSW graph (move overload).
     * 
     * @param vector Dense float vector to move into the index
     * @return std::size_t Assigned sequential vector ID (0, 1, 2, ...)
     * @throws std::invalid_argument If vector dimension does not match index dimension or is empty
     */
    std::size_t insert(Vector&& vector);

    /**
     * @brief Searches for the K nearest neighbors to the query vector.
     * 
     * @param query Query vector
     * @param k Number of nearest neighbors to retrieve
     * @return std::vector<SearchResult> Nearest neighbors sorted closest to farthest
     * @throws std::invalid_argument If query dimension does not match index dimension
     */
    [[nodiscard]] std::vector<SearchResult> search(const Vector& query, std::size_t k) const;

    /**
     * @brief Searches for the K nearest neighbors while tracking search statistics.
     * 
     * @param query Query vector
     * @param k Number of nearest neighbors to retrieve
     * @param stats Output telemetry structure receiving node visits and distance computation counts
     * @return std::vector<SearchResult> Nearest neighbors sorted closest to farthest
     * @throws std::invalid_argument If query dimension does not match index dimension
     */
    [[nodiscard]] std::vector<SearchResult> search_with_stats(const Vector& query, std::size_t k, HNSWStats& stats) const;

    /// Configures the runtime ef_search parameter
    void set_ef_search(std::size_t ef_search) noexcept;

    /// Returns current ef_search parameter
    [[nodiscard]] std::size_t ef_search() const noexcept;

    /// Returns the HNSW configuration
    [[nodiscard]] const HNSWConfig& config() const noexcept;

    /// Returns the number of vectors stored in the index
    [[nodiscard]] std::size_t size() const noexcept;

    /// Returns true if the index contains no vectors
    [[nodiscard]] bool empty() const noexcept;

    /// Returns vector dimension (or 0 if unconstrained)
    [[nodiscard]] std::size_t dimension() const noexcept;

    /// Retrieves a stored vector by its ID
    [[nodiscard]] const Vector& get(std::size_t id) const;

    /// Returns the maximum level in the HNSW graph (-1 if empty)
    [[nodiscard]] int max_level() const noexcept;

    /// Returns the global entry point node ID (std::nullopt if index is empty)
    [[nodiscard]] std::optional<std::size_t> entry_point() const noexcept;

    /// Returns the assigned level for a specific node ID
    [[nodiscard]] std::size_t node_level(std::size_t id) const;

    /// Returns the neighbor connections for a specific node ID at a specific layer
    [[nodiscard]] const std::vector<std::size_t>& node_neighbors(std::size_t id, std::size_t level) const;

    /// Clears all vectors and graph layers from the index
    void clear() noexcept;

private:
    struct Node {
        std::size_t id{0};
        std::size_t level{0};
        // friends_[l] contains neighbor IDs at layer l (from 0 to level)
        std::vector<std::vector<std::size_t>> friends;
    };

    /// Internal random level generation: floor(-ln(uniform(0,1)) * mL)
    [[nodiscard]] std::size_t generate_random_level();

    /// Greedy search at layer lc: returns closest node to query
    [[nodiscard]] std::size_t greedy_search_layer(const Vector& query, std::size_t ep, std::size_t lc, HNSWStats* stats = nullptr) const;

    /// Search layer lc with dynamic candidate list ef: returns nearest candidates
    [[nodiscard]] std::vector<SearchResult> search_layer(const Vector& query,
                                                         const std::vector<std::size_t>& ep_list,
                                                         std::size_t ef,
                                                         std::size_t lc,
                                                         HNSWStats* stats = nullptr) const;

    /// Selects up to M closest neighbors from candidate list
    [[nodiscard]] std::vector<std::size_t> select_neighbors(const Vector& query,
                                                            std::vector<SearchResult>& candidates,
                                                            std::size_t max_m) const;

    /// Prunes excess outgoing connections on node_id at layer lc to max_m
    void prune_neighbors(std::size_t node_id, std::size_t lc, std::size_t max_m);

    HNSWConfig config_;
    std::size_t dimension_{0};
    double m_l_{1.0};
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};

    int max_level_{-1};
    std::optional<std::size_t> entry_point_{std::nullopt};

    std::vector<Vector> vectors_;
    std::vector<Node> nodes_;
};

} // namespace vectorforge
