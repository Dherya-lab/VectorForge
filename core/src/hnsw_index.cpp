#include "hnsw_index.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace vectorforge {

HNSWIndex::HNSWIndex(HNSWConfig config, std::size_t dimension)
    : config_(config),
      dimension_(dimension),
      rng_(config.random_seed) {
    if (config_.M == 0) {
        throw std::invalid_argument("HNSW parameter M must be greater than 0");
    }
    if (config_.ef_construction == 0) {
        throw std::invalid_argument("HNSW parameter efConstruction must be greater than 0");
    }
    if (config_.ef_search == 0) {
        throw std::invalid_argument("HNSW parameter efSearch must be greater than 0");
    }
    // Normalization factor mL = 1 / ln(M)
    m_l_ = 1.0 / std::log(static_cast<double>(config_.M > 1 ? config_.M : 2));
}

std::size_t HNSWIndex::generate_random_level() {
    double r = uniform_dist_(rng_);
    while (r <= 0.0 || r >= 1.0) {
        r = uniform_dist_(rng_);
    }
    const double level = -std::log(r) * m_l_;
    return static_cast<std::size_t>(level);
}

std::size_t HNSWIndex::insert(const Vector& vector) {
    Vector copy = vector;
    return insert(std::move(copy));
}

std::size_t HNSWIndex::insert(Vector&& vector) {
    if (vector.empty()) {
        throw std::invalid_argument("Cannot insert an empty vector into HNSWIndex");
    }

    if (dimension_ == 0) {
        dimension_ = vector.dimension();
    } else if (vector.dimension() != dimension_) {
        throw std::invalid_argument("Vector dimension mismatch: HNSWIndex expects " +
                                    std::to_string(dimension_) + ", got " +
                                    std::to_string(vector.dimension()));
    }

    const std::size_t new_id = vectors_.size();
    const std::size_t node_level = generate_random_level();

    Node new_node{
        .id = new_id,
        .level = node_level,
        .friends = std::vector<std::vector<std::size_t>>(node_level + 1)
    };

    vectors_.push_back(std::move(vector));
    nodes_.push_back(std::move(new_node));

    // First node inserted into the graph
    if (new_id == 0) {
        entry_point_ = new_id;
        max_level_ = static_cast<int>(node_level);
        return new_id;
    }

    std::size_t curr_obj = *entry_point_;
    const int top_level = max_level_;
    const int insert_level = static_cast<int>(node_level);

    // 1. Greedy search from max_level down to insert_level + 1
    for (int lc = top_level; lc > insert_level; --lc) {
        curr_obj = greedy_search_layer(vectors_[new_id], curr_obj, static_cast<std::size_t>(lc));
    }

    // 2. Search and connect on layers from min(insert_level, top_level) down to 0
    for (int lc = std::min(insert_level, top_level); lc >= 0; --lc) {
        const std::size_t layer_idx = static_cast<std::size_t>(lc);
        const std::size_t max_m = (layer_idx == 0) ? (2 * config_.M) : config_.M;

        auto candidates = search_layer(vectors_[new_id], {curr_obj}, config_.ef_construction, layer_idx);
        auto neighbors = select_neighbors(vectors_[new_id], candidates, max_m);

        // Connect new node to chosen neighbors
        nodes_[new_id].friends[layer_idx] = neighbors;

        // Add reverse connections and prune if exceeding max_m
        for (std::size_t neighbor_id : neighbors) {
            nodes_[neighbor_id].friends[layer_idx].push_back(new_id);
            if (nodes_[neighbor_id].friends[layer_idx].size() > max_m) {
                prune_neighbors(neighbor_id, layer_idx, max_m);
            }
        }

        if (!candidates.empty()) {
            curr_obj = candidates.front().id;
        }
    }

    // 3. Update entry point if newly inserted node reached a new highest level
    if (insert_level > max_level_) {
        max_level_ = insert_level;
        entry_point_ = new_id;
    }

    return new_id;
}

std::size_t HNSWIndex::greedy_search_layer(const Vector& query,
                                           std::size_t ep,
                                           std::size_t lc,
                                           HNSWStats* stats) const {
    std::size_t curr = ep;
    float cur_dist = euclidean_distance(query, vectors_[curr]);
    if (stats) {
        ++stats->nodes_visited;
        ++stats->distance_computations;
    }

    bool changed = true;
    while (changed) {
        changed = false;
        if (lc >= nodes_[curr].friends.size()) {
            break;
        }
        for (std::size_t neighbor : nodes_[curr].friends[lc]) {
            const float dist = euclidean_distance(query, vectors_[neighbor]);
            if (stats) {
                ++stats->nodes_visited;
                ++stats->distance_computations;
            }
            if (dist < cur_dist) {
                cur_dist = dist;
                curr = neighbor;
                changed = true;
            }
        }
    }

    return curr;
}

std::vector<SearchResult> HNSWIndex::search_layer(const Vector& query,
                                                  const std::vector<std::size_t>& ep_list,
                                                  std::size_t ef,
                                                  std::size_t lc,
                                                  HNSWStats* stats) const {
    // Visited set to prevent evaluating nodes multiple times
    std::unordered_set<std::size_t> visited;
    visited.reserve(ef * 4);

    // Min-heap for candidate exploration: nearest candidate at top
    auto candidate_cmp = [](const SearchResult& a, const SearchResult& b) noexcept {
        return a.distance > b.distance;
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(candidate_cmp)> candidates(candidate_cmp);

    // Max-heap for keeping the ef best results: farthest result at top
    auto result_cmp = [](const SearchResult& a, const SearchResult& b) noexcept {
        return a.distance < b.distance;
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(result_cmp)> best_results(result_cmp);

    for (std::size_t ep : ep_list) {
        visited.insert(ep);
        const float dist = euclidean_distance(query, vectors_[ep]);
        if (stats) {
            ++stats->nodes_visited;
            ++stats->distance_computations;
        }

        SearchResult sr{.id = ep, .distance = dist};
        candidates.push(sr);
        best_results.push(sr);
    }

    while (!candidates.empty()) {
        const SearchResult curr = candidates.top();
        candidates.pop();

        const SearchResult farthest_result = best_results.top();
        if (curr.distance > farthest_result.distance) {
            break;
        }

        if (lc >= nodes_[curr.id].friends.size()) {
            continue;
        }

        for (std::size_t neighbor : nodes_[curr.id].friends[lc]) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);

                const float dist = euclidean_distance(query, vectors_[neighbor]);
                if (stats) {
                    ++stats->nodes_visited;
                    ++stats->distance_computations;
                }

                const SearchResult neighbor_res{.id = neighbor, .distance = dist};

                if (dist < best_results.top().distance || best_results.size() < ef) {
                    candidates.push(neighbor_res);
                    best_results.push(neighbor_res);

                    if (best_results.size() > ef) {
                        best_results.pop();
                    }
                }
            }
        }
    }

    std::vector<SearchResult> sorted_results;
    sorted_results.reserve(best_results.size());
    while (!best_results.empty()) {
        sorted_results.push_back(best_results.top());
        best_results.pop();
    }
    std::reverse(sorted_results.begin(), sorted_results.end());

    return sorted_results;
}

std::vector<std::size_t> HNSWIndex::select_neighbors(const Vector& /*query*/,
                                                     std::vector<SearchResult>& candidates,
                                                     std::size_t max_m) const {
    // Simple nearest-neighbor heuristic: sort candidates by distance and take up to max_m
    std::sort(candidates.begin(), candidates.end(), [](const SearchResult& a, const SearchResult& b) noexcept {
        if (a.distance != b.distance) {
            return a.distance < b.distance;
        }
        return a.id < b.id;
    });

    std::vector<std::size_t> selected;
    selected.reserve(std::min(max_m, candidates.size()));
    for (std::size_t i = 0; i < candidates.size() && selected.size() < max_m; ++i) {
        selected.push_back(candidates[i].id);
    }
    return selected;
}

void HNSWIndex::prune_neighbors(std::size_t node_id, std::size_t lc, std::size_t max_m) {
    auto& neighbors = nodes_[node_id].friends[lc];
    if (neighbors.size() <= max_m) {
        return;
    }

    const Vector& node_vec = vectors_[node_id];
    std::vector<SearchResult> candidate_distances;
    candidate_distances.reserve(neighbors.size());

    for (std::size_t n : neighbors) {
        candidate_distances.push_back(SearchResult{
            .id = n,
            .distance = euclidean_distance(node_vec, vectors_[n])
        });
    }

    auto selected = select_neighbors(node_vec, candidate_distances, max_m);
    neighbors = std::move(selected);
}

std::vector<SearchResult> HNSWIndex::search(const Vector& query, std::size_t k) const {
    HNSWStats dummy_stats;
    return search_with_stats(query, k, dummy_stats);
}

std::vector<SearchResult> HNSWIndex::search_with_stats(const Vector& query,
                                                       std::size_t k,
                                                       HNSWStats& stats) const {
    if (dimension_ != 0 && query.dimension() != dimension_) {
        throw std::invalid_argument("Query dimension mismatch: HNSWIndex expects " +
                                    std::to_string(dimension_) + ", got " +
                                    std::to_string(query.dimension()));
    }

    if (k == 0 || vectors_.empty() || !entry_point_.has_value()) {
        return {};
    }

    std::size_t curr_obj = *entry_point_;
    const int top_level = max_level_;

    // 1. Greedy search from top_level down to layer 1
    for (int lc = top_level; lc >= 1; --lc) {
        curr_obj = greedy_search_layer(query, curr_obj, static_cast<std::size_t>(lc), &stats);
    }

    // 2. Layer 0 candidate search with ef_search
    const std::size_t ef = std::max(config_.ef_search, k);
    auto candidates = search_layer(query, {curr_obj}, ef, 0, &stats);

    // 3. Return top-k results
    if (candidates.size() > k) {
        candidates.resize(k);
    }

    return candidates;
}

void HNSWIndex::set_ef_search(std::size_t ef_search) noexcept {
    config_.ef_search = ef_search;
}

std::size_t HNSWIndex::ef_search() const noexcept {
    return config_.ef_search;
}

const HNSWConfig& HNSWIndex::config() const noexcept {
    return config_;
}

std::size_t HNSWIndex::size() const noexcept {
    return vectors_.size();
}

bool HNSWIndex::empty() const noexcept {
    return vectors_.empty();
}

std::size_t HNSWIndex::dimension() const noexcept {
    return dimension_;
}

const Vector& HNSWIndex::get(std::size_t id) const {
    if (id >= vectors_.size()) {
        throw std::out_of_range("Vector ID out of range: " + std::to_string(id) +
                                " (HNSWIndex size is " + std::to_string(vectors_.size()) + ")");
    }
    return vectors_[id];
}

int HNSWIndex::max_level() const noexcept {
    return max_level_;
}

std::optional<std::size_t> HNSWIndex::entry_point() const noexcept {
    return entry_point_;
}

std::size_t HNSWIndex::node_level(std::size_t id) const {
    if (id >= nodes_.size()) {
        throw std::out_of_range("Node ID out of range: " + std::to_string(id));
    }
    return nodes_[id].level;
}

const std::vector<std::size_t>& HNSWIndex::node_neighbors(std::size_t id, std::size_t level) const {
    if (id >= nodes_.size()) {
        throw std::out_of_range("Node ID out of range: " + std::to_string(id));
    }
    if (level >= nodes_[id].friends.size()) {
        throw std::out_of_range("Level out of range for node " + std::to_string(id) +
                                ": requested " + std::to_string(level) +
                                ", max level is " + std::to_string(nodes_[id].level));
    }
    return nodes_[id].friends[level];
}

void HNSWIndex::clear() noexcept {
    vectors_.clear();
    nodes_.clear();
    max_level_ = -1;
    entry_point_ = std::nullopt;
}

} // namespace vectorforge
