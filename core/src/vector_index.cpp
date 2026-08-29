#include "vector_index.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace vectorforge {

VectorIndex::VectorIndex(std::size_t dimension)
    : dimension_(dimension) {}

std::size_t VectorIndex::insert(const Vector& vector) {
    if (vector.empty()) {
        throw std::invalid_argument("Cannot insert an empty vector into the index");
    }

    if (dimension_ == 0) {
        dimension_ = vector.dimension();
    } else if (vector.dimension() != dimension_) {
        throw std::invalid_argument("Vector dimension mismatch: index expects " +
                                    std::to_string(dimension_) + ", got " +
                                    std::to_string(vector.dimension()));
    }

    const std::size_t id = vectors_.size();
    vectors_.push_back(vector);
    return id;
}

std::size_t VectorIndex::insert(Vector&& vector) {
    if (vector.empty()) {
        throw std::invalid_argument("Cannot insert an empty vector into the index");
    }

    if (dimension_ == 0) {
        dimension_ = vector.dimension();
    } else if (vector.dimension() != dimension_) {
        throw std::invalid_argument("Vector dimension mismatch: index expects " +
                                    std::to_string(dimension_) + ", got " +
                                    std::to_string(vector.dimension()));
    }

    const std::size_t id = vectors_.size();
    vectors_.push_back(std::move(vector));
    return id;
}

std::vector<SearchResult> VectorIndex::search(const Vector& query, std::size_t k) const {
    if (dimension_ != 0 && query.dimension() != dimension_) {
        throw std::invalid_argument("Query dimension mismatch: index expects " +
                                    std::to_string(dimension_) + ", got " +
                                    std::to_string(query.dimension()));
    }

    if (k == 0 || vectors_.empty()) {
        return {};
    }

    std::vector<SearchResult> all_results;
    all_results.reserve(vectors_.size());

    for (std::size_t i = 0; i < vectors_.size(); ++i) {
        const float dist = euclidean_distance(query, vectors_[i]);
        all_results.push_back(SearchResult{
            .id = i,
            .distance = dist
        });
    }

    const std::size_t result_count = std::min(k, all_results.size());

    // Partial sort to find the k smallest distances
    std::partial_sort(
        all_results.begin(),
        all_results.begin() + result_count,
        all_results.end(),
        [](const SearchResult& a, const SearchResult& b) noexcept {
            if (a.distance != b.distance) {
                return a.distance < b.distance;
            }
            return a.id < b.id; // Deterministic tie-breaker
        }
    );

    all_results.resize(result_count);
    return all_results;
}

std::size_t VectorIndex::size() const noexcept {
    return vectors_.size();
}

bool VectorIndex::empty() const noexcept {
    return vectors_.empty();
}

std::size_t VectorIndex::dimension() const noexcept {
    return dimension_;
}

const Vector& VectorIndex::get(std::size_t id) const {
    if (id >= vectors_.size()) {
        throw std::out_of_range("Vector ID out of range: " + std::to_string(id) +
                                " (index size is " + std::to_string(vectors_.size()) + ")");
    }
    return vectors_[id];
}

void VectorIndex::clear() noexcept {
    vectors_.clear();
}

} // namespace vectorforge
