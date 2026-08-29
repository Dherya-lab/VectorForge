#pragma once

#include "vector.hpp"
#include "distance.hpp"

#include <cstddef>
#include <vector>

namespace vectorforge {

/**
 * @brief Search result structure representing a nearest neighbor match.
 */
struct SearchResult {
    std::size_t id{0};      ///< Identifier of the stored vector
    float distance{0.0f};   ///< Euclidean distance from the query vector

    /// Three-way comparison operator for sorting and equality
    auto operator<=>(const SearchResult& other) const = default;
};

/**
 * @brief In-memory brute-force vector index providing exact k-NN search.
 */
class VectorIndex {
public:
    /// Default constructor creates an empty index with unconstrained dimension
    VectorIndex() = default;

    /// Constructs an index with a predefined vector dimension
    explicit VectorIndex(std::size_t dimension);

    /**
     * @brief Inserts a vector into the index.
     * 
     * Assigns a sequential 0-indexed ID (0, 1, 2, ...).
     * 
     * @param vector Vector to insert
     * @return std::size_t Assigned vector ID
     * @throws std::invalid_argument If vector dimension does not match index dimension or is empty
     */
    std::size_t insert(const Vector& vector);

    /**
     * @brief Inserts a vector into the index (move overload).
     * 
     * @param vector Vector to move into the index
     * @return std::size_t Assigned vector ID
     * @throws std::invalid_argument If vector dimension does not match index dimension or is empty
     */
    std::size_t insert(Vector&& vector);

    /**
     * @brief Searches for the K nearest vectors ordered by increasing Euclidean distance.
     * 
     * @param query Query vector
     * @param k Maximum number of nearest neighbors to return
     * @return std::vector<SearchResult> Nearest neighbors sorted from closest to farthest
     * @throws std::invalid_argument If query dimension does not match index dimension
     */
    [[nodiscard]] std::vector<SearchResult> search(const Vector& query, std::size_t k) const;

    /// Returns the number of vectors stored in the index
    [[nodiscard]] std::size_t size() const noexcept;

    /// Returns true if the index contains no vectors
    [[nodiscard]] bool empty() const noexcept;

    /// Returns the dimension of the indexed vectors (or 0 if empty and unconstrained)
    [[nodiscard]] std::size_t dimension() const noexcept;

    /**
     * @brief Retrieves a stored vector by its ID.
     * 
     * @param id Vector ID
     * @return const Vector& Reference to the stored vector
     * @throws std::out_of_range If ID is invalid
     */
    [[nodiscard]] const Vector& get(std::size_t id) const;

    /// Clears all vectors from the index
    void clear() noexcept;

private:
    std::size_t dimension_{0};
    std::vector<Vector> vectors_;
};

} // namespace vectorforge
