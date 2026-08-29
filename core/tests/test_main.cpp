#include "vector.hpp"
#include "distance.hpp"
#include "vector_index.hpp"
#include "hnsw_index.hpp"
#include "simd_math.hpp"
#include "product_quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

int total_tests = 0;
int passed_tests = 0;
int failed_tests = 0;

void record_test(bool condition, const std::string& test_name, const std::string& details = "") {
    ++total_tests;
    if (condition) {
        ++passed_tests;
        std::cout << "  [PASS] " << test_name << std::endl;
    } else {
        ++failed_tests;
        std::cerr << "  [FAIL] " << test_name;
        if (!details.empty()) {
            std::cerr << " -> " << details;
        }
        std::cerr << std::endl;
    }
}

bool approx_equal(float a, float b, float epsilon = 1e-4f) {
    const float diff = std::fabs(a - b);
    if (diff <= epsilon) return true;
    const float max_val = std::max(std::fabs(a), std::fabs(b));
    return (max_val > 0.0f) && ((diff / max_val) <= 1e-4f);
}

// =========================================================================
// PHASE 1 TESTS (Brute-Force VectorIndex)
// =========================================================================

void test_empty_index() {
    vectorforge::VectorIndex index;
    record_test(index.empty(), "Empty Index: index.empty() is true");
    record_test(index.size() == 0, "Empty Index: index.size() == 0");
    record_test(index.dimension() == 0, "Empty Index: unconstrained index.dimension() == 0");

    const vectorforge::Vector query{1.0f, 2.0f, 3.0f};
    const auto results = index.search(query, 5);
    record_test(results.empty(), "Empty Index: search returns empty vector");
}

void test_single_vector() {
    vectorforge::VectorIndex index;
    const std::size_t id = index.insert(vectorforge::Vector{1.0f, 2.0f, 3.0f});
    record_test(id == 0, "Single Vector: first inserted vector ID is 0");
    record_test(index.size() == 1, "Single Vector: index.size() == 1");
    record_test(!index.empty(), "Single Vector: index.empty() is false");
    record_test(index.dimension() == 3, "Single Vector: index.dimension() == 3");

    const vectorforge::Vector query{1.0f, 2.0f, 3.0f};
    const auto results = index.search(query, 1);
    record_test(results.size() == 1, "Single Vector: search returns 1 result");
    if (!results.empty()) {
        record_test(results[0].id == 0, "Single Vector: result ID matches inserted vector");
        record_test(approx_equal(results[0].distance, 0.0f), "Single Vector: distance is 0 for identical vector");
    }
}

void test_multiple_vectors() {
    vectorforge::VectorIndex index;
    const std::size_t id0 = index.insert(vectorforge::Vector{1.0f, 0.0f});
    const std::size_t id1 = index.insert(vectorforge::Vector{0.0f, 1.0f});
    const std::size_t id2 = index.insert(vectorforge::Vector{-1.0f, 0.0f});
    const std::size_t id3 = index.insert(vectorforge::Vector{0.0f, -1.0f});

    record_test(id0 == 0 && id1 == 1 && id2 == 2 && id3 == 3, "Multiple Vectors: sequential IDs assigned (0..3)");
    record_test(index.size() == 4, "Multiple Vectors: index.size() == 4");

    const vectorforge::Vector query{0.0f, 0.0f};
    const auto results = index.search(query, 4);
    record_test(results.size() == 4, "Multiple Vectors: returns all 4 vectors when k=4");
}

void test_euclidean_distance_calculation() {
    const vectorforge::Vector a{0.0f, 0.0f};
    const vectorforge::Vector b{3.0f, 4.0f};
    const float dist_ab = vectorforge::euclidean_distance(a, b);
    record_test(approx_equal(dist_ab, 5.0f), "Euclidean Distance: (0,0) to (3,4) is 5.0",
                "Got " + std::to_string(dist_ab));

    const vectorforge::Vector c{1.0f, 2.0f, 3.0f};
    const vectorforge::Vector d{4.0f, 6.0f, 3.0f};
    const float dist_cd = vectorforge::euclidean_distance(c, d);
    record_test(approx_equal(dist_cd, 5.0f), "Euclidean Distance: (1,2,3) to (4,6,3) is 5.0",
                "Got " + std::to_string(dist_cd));

    const vectorforge::Vector origin{0.0f, 0.0f, 0.0f};
    const vectorforge::Vector unit{1.0f, 1.0f, 1.0f};
    const float dist_unit = vectorforge::euclidean_distance(origin, unit);
    record_test(approx_equal(dist_unit, std::sqrt(3.0f)), "Euclidean Distance: (0,0,0) to (1,1,1) is sqrt(3)",
                "Got " + std::to_string(dist_unit));
}

void test_nearest_neighbor_ordering() {
    vectorforge::VectorIndex index;
    index.insert(vectorforge::Vector{0.0f, 0.0f});
    index.insert(vectorforge::Vector{0.0f, 5.0f});
    index.insert(vectorforge::Vector{0.0f, 2.0f});
    index.insert(vectorforge::Vector{0.0f, 8.0f});
    index.insert(vectorforge::Vector{0.0f, 1.0f});

    const vectorforge::Vector query{0.0f, 0.0f};
    const auto results = index.search(query, 5);

    record_test(results.size() == 5, "Ordering: returned 5 results");
    if (results.size() == 5) {
        const bool ids_correct = (results[0].id == 0 &&
                                  results[1].id == 4 &&
                                  results[2].id == 2 &&
                                  results[3].id == 1 &&
                                  results[4].id == 3);
        record_test(ids_correct, "Ordering: results ordered by increasing distance (IDs 0, 4, 2, 1, 3)");

        const bool monotonic = (results[0].distance <= results[1].distance &&
                                results[1].distance <= results[2].distance &&
                                results[2].distance <= results[3].distance &&
                                results[3].distance <= results[4].distance);
        record_test(monotonic, "Ordering: distances are monotonically increasing");
    }
}

void test_k_equals_one() {
    vectorforge::VectorIndex index;
    index.insert(vectorforge::Vector{10.0f, 10.0f});
    index.insert(vectorforge::Vector{1.0f, 1.0f});
    index.insert(vectorforge::Vector{5.0f, 5.0f});

    const vectorforge::Vector query{0.0f, 0.0f};
    const auto results = index.search(query, 1);

    record_test(results.size() == 1, "k=1: returns exactly 1 result");
    if (!results.empty()) {
        record_test(results[0].id == 1, "k=1: returned closest vector ID 1");
        record_test(approx_equal(results[0].distance, std::sqrt(2.0f)), "k=1: distance is sqrt(2)");
    }
}

void test_k_greater_than_size() {
    vectorforge::VectorIndex index;
    index.insert(vectorforge::Vector{1.0f, 2.0f});
    index.insert(vectorforge::Vector{3.0f, 4.0f});
    index.insert(vectorforge::Vector{5.0f, 6.0f});

    const vectorforge::Vector query{0.0f, 0.0f};
    const auto results = index.search(query, 100);

    record_test(results.size() == 3, "k > size: returns all 3 available vectors without crashing");
    if (results.size() == 3) {
        record_test(results[0].id == 0 && results[1].id == 1 && results[2].id == 2,
                    "k > size: results correctly ordered");
    }
}

void test_dimension_mismatch() {
    bool distance_threw = false;
    try {
        const vectorforge::Vector v2{1.0f, 2.0f};
        const vectorforge::Vector v3{1.0f, 2.0f, 3.0f};
        (void)vectorforge::euclidean_distance(v2, v3);
    } catch (const std::invalid_argument&) {
        distance_threw = true;
    }
    record_test(distance_threw, "Dimension Mismatch: euclidean_distance throws on dimension mismatch");

    bool insert_threw = false;
    try {
        vectorforge::VectorIndex index;
        index.insert(vectorforge::Vector{1.0f, 2.0f});
        (void)index.insert(vectorforge::Vector{1.0f, 2.0f, 3.0f});
    } catch (const std::invalid_argument&) {
        insert_threw = true;
    }
    record_test(insert_threw, "Dimension Mismatch: index.insert throws on dimension mismatch");

    bool query_threw = false;
    try {
        vectorforge::VectorIndex index;
        index.insert(vectorforge::Vector{1.0f, 2.0f});
        const vectorforge::Vector bad_query{1.0f, 2.0f, 3.0f};
        (void)index.search(bad_query, 1);
    } catch (const std::invalid_argument&) {
        query_threw = true;
    }
    record_test(query_threw, "Dimension Mismatch: index.search throws on query dimension mismatch");

    bool empty_insert_threw = false;
    try {
        vectorforge::VectorIndex index;
        (void)index.insert(vectorforge::Vector{});
    } catch (const std::invalid_argument&) {
        empty_insert_threw = true;
    }
    record_test(empty_insert_threw, "Dimension Mismatch: index.insert throws on empty vector");
}

void test_identical_vector_distance_zero() {
    vectorforge::VectorIndex index;
    index.insert(vectorforge::Vector{42.0f, -17.5f, 0.001f, 100.0f});

    const vectorforge::Vector query{42.0f, -17.5f, 0.001f, 100.0f};
    const auto results = index.search(query, 1);

    record_test(results.size() == 1, "Identical Vector: returns 1 result");
    if (!results.empty()) {
        record_test(results[0].distance == 0.0f, "Identical Vector: produces exact distance 0.0");
    }
}

// =========================================================================
// PHASE 2 TESTS (HNSWIndex)
// =========================================================================

void test_hnsw_empty_index() {
    vectorforge::HNSWIndex index;
    record_test(index.empty(), "HNSW 1/15 [Empty Index]: index.empty() is true");
    record_test(index.size() == 0, "HNSW 1/15 [Empty Index]: index.size() == 0");
    record_test(index.dimension() == 0, "HNSW 1/15 [Empty Index]: unconstrained dimension() == 0");
    record_test(index.max_level() == -1, "HNSW 1/15 [Empty Index]: max_level() == -1");
    record_test(!index.entry_point().has_value(), "HNSW 1/15 [Empty Index]: entry_point is nullopt");

    const vectorforge::Vector query{1.0f, 2.0f, 3.0f};
    const auto results = index.search(query, 5);
    record_test(results.empty(), "HNSW 1/15 [Empty Index]: search returns empty vector");
}

void test_hnsw_single_vector() {
    vectorforge::HNSWIndex index;
    const std::size_t id = index.insert(vectorforge::Vector{1.0f, 2.0f, 3.0f});
    record_test(id == 0, "HNSW 2/15 [Single Vector]: first inserted vector ID is 0");
    record_test(index.size() == 1, "HNSW 2/15 [Single Vector]: index.size() == 1");
    record_test(!index.empty(), "HNSW 2/15 [Single Vector]: index.empty() is false");
    record_test(index.dimension() == 3, "HNSW 2/15 [Single Vector]: dimension is 3");
    record_test(index.entry_point() == 0, "HNSW 2/15 [Single Vector]: entry point is 0");
    record_test(index.max_level() >= 0, "HNSW 2/15 [Single Vector]: max_level >= 0");

    const vectorforge::Vector query{1.0f, 2.0f, 3.0f};
    const auto results = index.search(query, 1);
    record_test(results.size() == 1, "HNSW 2/15 [Single Vector]: search returns 1 result");
    if (!results.empty()) {
        record_test(results[0].id == 0, "HNSW 2/15 [Single Vector]: result ID is 0");
        record_test(approx_equal(results[0].distance, 0.0f), "HNSW 2/15 [Single Vector]: distance is 0.0");
    }
}

void test_hnsw_multiple_vectors() {
    vectorforge::HNSWIndex index;
    const std::size_t id0 = index.insert(vectorforge::Vector{1.0f, 0.0f});
    const std::size_t id1 = index.insert(vectorforge::Vector{0.0f, 1.0f});
    const std::size_t id2 = index.insert(vectorforge::Vector{-1.0f, 0.0f});
    const std::size_t id3 = index.insert(vectorforge::Vector{0.0f, -1.0f});

    record_test(id0 == 0 && id1 == 1 && id2 == 2 && id3 == 3, "HNSW 3/15 [Multiple Vectors]: sequential IDs 0..3");
    record_test(index.size() == 4, "HNSW 3/15 [Multiple Vectors]: index.size() == 4");

    const vectorforge::Vector query{0.0f, 0.0f};
    const auto results = index.search(query, 4);
    record_test(results.size() == 4, "HNSW 3/15 [Multiple Vectors]: returns all 4 vectors");
}

void test_hnsw_insert_sequence() {
    vectorforge::HNSWIndex index;
    constexpr std::size_t N = 30;
    for (std::size_t i = 0; i < N; ++i) {
        const float val = static_cast<float>(i);
        const std::size_t assigned_id = index.insert(vectorforge::Vector{val, val * 2.0f});
        if (assigned_id != i) {
            record_test(false, "HNSW 4/15 [Insert Sequence]: ID mismatch", "Expected " + std::to_string(i) + " got " + std::to_string(assigned_id));
            return;
        }
    }
    record_test(index.size() == N, "HNSW 4/15 [Insert Sequence]: index size matches N=30");
}

void test_hnsw_k_equals_one() {
    vectorforge::HNSWIndex index;
    index.insert(vectorforge::Vector{10.0f, 10.0f});
    index.insert(vectorforge::Vector{1.0f, 1.0f});
    index.insert(vectorforge::Vector{5.0f, 5.0f});

    const vectorforge::Vector query{0.0f, 0.0f};
    const auto results = index.search(query, 1);

    record_test(results.size() == 1, "HNSW 5/15 [k=1]: returns exactly 1 result");
    if (!results.empty()) {
        record_test(results[0].id == 1, "HNSW 5/15 [k=1]: returns closest ID 1");
        record_test(approx_equal(results[0].distance, std::sqrt(2.0f)), "HNSW 5/15 [k=1]: distance is sqrt(2)");
    }
}

void test_hnsw_k_greater_one() {
    vectorforge::HNSWIndex index;
    index.insert(vectorforge::Vector{0.0f, 0.0f});
    index.insert(vectorforge::Vector{1.0f, 0.0f});
    index.insert(vectorforge::Vector{2.0f, 0.0f});
    index.insert(vectorforge::Vector{3.0f, 0.0f});
    index.insert(vectorforge::Vector{4.0f, 0.0f});

    const vectorforge::Vector query{0.0f, 0.0f};
    const auto results = index.search(query, 3);

    record_test(results.size() == 3, "HNSW 6/15 [k>1]: returns exactly 3 results");
    if (results.size() == 3) {
        record_test(results[0].id == 0 && results[1].id == 1 && results[2].id == 2,
                    "HNSW 6/15 [k>1]: returned correct closest 3 vectors (IDs 0, 1, 2)");
    }
}

void test_hnsw_k_greater_than_size() {
    vectorforge::HNSWIndex index;
    index.insert(vectorforge::Vector{1.0f, 1.0f});
    index.insert(vectorforge::Vector{2.0f, 2.0f});
    index.insert(vectorforge::Vector{3.0f, 3.0f});

    const vectorforge::Vector query{0.0f, 0.0f};
    const auto results = index.search(query, 100);

    record_test(results.size() == 3, "HNSW 7/15 [k > size]: returns all 3 available vectors without crashing");
}

void test_hnsw_identical_vector() {
    vectorforge::HNSWIndex index;
    index.insert(vectorforge::Vector{3.14159f, 2.71828f, 1.41421f});

    const vectorforge::Vector query{3.14159f, 2.71828f, 1.41421f};
    const auto results = index.search(query, 1);

    record_test(results.size() == 1, "HNSW 8/15 [Identical Vector]: returns 1 result");
    if (!results.empty()) {
        record_test(results[0].distance == 0.0f, "HNSW 8/15 [Identical Vector]: exact distance is 0.0");
    }
}

void test_hnsw_dimension_mismatch() {
    bool insert_threw = false;
    try {
        vectorforge::HNSWIndex index;
        index.insert(vectorforge::Vector{1.0f, 2.0f});
        (void)index.insert(vectorforge::Vector{1.0f, 2.0f, 3.0f});
    } catch (const std::invalid_argument&) {
        insert_threw = true;
    }
    record_test(insert_threw, "HNSW 9/15 [Dimension Mismatch]: insert throws on dimension mismatch");

    bool query_threw = false;
    try {
        vectorforge::HNSWIndex index;
        index.insert(vectorforge::Vector{1.0f, 2.0f});
        const vectorforge::Vector bad_query{1.0f, 2.0f, 3.0f};
        (void)index.search(bad_query, 1);
    } catch (const std::invalid_argument&) {
        query_threw = true;
    }
    record_test(query_threw, "HNSW 9/15 [Dimension Mismatch]: search throws on query dimension mismatch");

    bool empty_insert_threw = false;
    try {
        vectorforge::HNSWIndex index;
        (void)index.insert(vectorforge::Vector{});
    } catch (const std::invalid_argument&) {
        empty_insert_threw = true;
    }
    record_test(empty_insert_threw, "HNSW 9/15 [Dimension Mismatch]: insert throws on empty vector");
}

void test_hnsw_entry_point() {
    vectorforge::HNSWIndex index(vectorforge::HNSWConfig{.M = 4, .ef_construction = 16, .ef_search = 16, .random_seed = 123});
    record_test(!index.entry_point().has_value(), "HNSW 10/15 [Entry Point]: initial entry point is nullopt");

    index.insert(vectorforge::Vector{0.0f, 0.0f});
    record_test(index.entry_point() == 0, "HNSW 10/15 [Entry Point]: entry point is 0 after first insert");

    for (std::size_t i = 1; i <= 20; ++i) {
        index.insert(vectorforge::Vector{static_cast<float>(i), static_cast<float>(i)});
        record_test(index.entry_point().has_value() && *index.entry_point() < index.size(),
                    "HNSW 10/15 [Entry Point]: valid entry point during insertion step " + std::to_string(i));
    }
}

void test_hnsw_multiple_layers() {
    vectorforge::HNSWIndex index(vectorforge::HNSWConfig{.M = 4, .ef_construction = 32, .ef_search = 16, .random_seed = 42});
    for (std::size_t i = 0; i < 50; ++i) {
        index.insert(vectorforge::Vector{static_cast<float>(i * 2), static_cast<float>(i * 3)});
    }

    record_test(index.max_level() >= 1, "HNSW 11/15 [Multiple Layers]: graph has multiple layers (max_level >= 1)",
                "Got max_level = " + std::to_string(index.max_level()));

    std::size_t layer0_count = 0;
    std::size_t higher_layer_count = 0;
    for (std::size_t i = 0; i < index.size(); ++i) {
        ++layer0_count;
        if (index.node_level(i) > 0) {
            ++higher_layer_count;
        }
    }
    record_test(layer0_count == 50, "HNSW 11/15 [Multiple Layers]: all 50 nodes exist on layer 0");
    record_test(higher_layer_count > 0 && higher_layer_count < 50,
                "HNSW 11/15 [Multiple Layers]: subset of nodes exist on higher layers (" + std::to_string(higher_layer_count) + " nodes)");
}

void test_hnsw_neighbor_connections() {
    vectorforge::HNSWIndex index(vectorforge::HNSWConfig{.M = 4, .ef_construction = 16, .ef_search = 16, .random_seed = 42});
    for (std::size_t i = 0; i < 10; ++i) {
        index.insert(vectorforge::Vector{static_cast<float>(i), static_cast<float>(i)});
    }

    bool all_connected = true;
    for (std::size_t i = 0; i < index.size(); ++i) {
        const auto& neighbors = index.node_neighbors(i, 0);
        if (neighbors.empty()) {
            all_connected = false;
            break;
        }
    }
    record_test(all_connected, "HNSW 12/15 [Connections Exist]: all nodes have neighbor connections on layer 0");
}

void test_hnsw_neighbor_degree_limits() {
    constexpr std::size_t M = 4;
    constexpr std::size_t M0 = 2 * M;
    vectorforge::HNSWIndex index(vectorforge::HNSWConfig{.M = M, .ef_construction = 32, .ef_search = 16, .random_seed = 42});

    for (std::size_t i = 0; i < 40; ++i) {
        index.insert(vectorforge::Vector{static_cast<float>(i * 3), static_cast<float>(i * 7)});
    }

    bool limits_respected = true;
    for (std::size_t i = 0; i < index.size(); ++i) {
        const std::size_t node_lvl = index.node_level(i);
        for (std::size_t l = 0; l <= node_lvl; ++l) {
            const std::size_t max_allowed = (l == 0) ? M0 : M;
            const auto& neighbors = index.node_neighbors(i, l);
            if (neighbors.size() > max_allowed) {
                limits_respected = false;
                break;
            }
        }
        if (!limits_respected) break;
    }
    record_test(limits_respected, "HNSW 13/15 [Degree Limits]: connections per node per layer do not exceed M / M0 limits");
}

void test_hnsw_distance_monotonicity() {
    vectorforge::HNSWIndex index(vectorforge::HNSWConfig{.M = 8, .ef_construction = 32, .ef_search = 32, .random_seed = 42});
    for (std::size_t i = 0; i < 20; ++i) {
        index.insert(vectorforge::Vector{static_cast<float>(i * 5), static_cast<float>(i * 5)});
    }

    const vectorforge::Vector query{22.0f, 22.0f};
    const auto results = index.search(query, 10);

    bool monotonic = true;
    for (std::size_t i = 1; i < results.size(); ++i) {
        if (results[i].distance < results[i - 1].distance) {
            monotonic = false;
            break;
        }
    }
    record_test(monotonic, "HNSW 14/15 [Monotonic Ordering]: search results are strictly ordered by increasing distance");
}

void test_hnsw_ground_truth_comparison() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    vectorforge::VectorIndex bf_index;
    vectorforge::HNSWIndex hnsw_index(vectorforge::HNSWConfig{.M = 8, .ef_construction = 64, .ef_search = 32, .random_seed = 42});

    for (std::size_t i = 0; i < 50; ++i) {
        vectorforge::Vector vec{dist(rng), dist(rng), dist(rng), dist(rng)};
        bf_index.insert(vec);
        hnsw_index.insert(vec);
    }

    std::size_t total_matches = 0;
    constexpr std::size_t K = 5;
    constexpr std::size_t NUM_QUERIES = 10;

    for (std::size_t q = 0; q < NUM_QUERIES; ++q) {
        const vectorforge::Vector query{dist(rng), dist(rng), dist(rng), dist(rng)};
        const auto bf_res = bf_index.search(query, K);
        const auto hnsw_res = hnsw_index.search(query, K);

        std::unordered_set<std::size_t> ground_truth;
        for (const auto& r : bf_res) ground_truth.insert(r.id);

        for (const auto& r : hnsw_res) {
            if (ground_truth.find(r.id) != ground_truth.end()) {
                ++total_matches;
            }
        }
    }

    const double recall = static_cast<double>(total_matches) / static_cast<double>(NUM_QUERIES * K);
    record_test(recall >= 0.90, "HNSW 15/15 [Ground Truth Recall]: Recall@" + std::to_string(K) + " >= 90% (Measured: " + std::to_string(recall * 100.0) + "%)");
}

// =========================================================================
// PHASE 3 TESTS (SIMD Math & Distance Acceleration)
// =========================================================================

// 1. Scalar vs AVX2 Euclidean Distance
void test_simd_scalar_vs_avx2() {
    const auto& cpu = vectorforge::simd::get_cpu_features();
    if (!cpu.supports_avx2()) {
        record_test(true, "SIMD 1/10 [Scalar vs AVX2]: Skipped (AVX2 unsupported on this CPU)");
        return;
    }

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    bool all_match = true;
    for (std::size_t dim : {8, 16, 64, 128, 768, 1536}) {
        std::vector<float> a(dim), b(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }
        const float scalar_d = vectorforge::simd::distance_scalar(a, b);
        const float avx2_d = vectorforge::simd::distance_avx2(a, b);

        if (!approx_equal(scalar_d, avx2_d, 1e-4f)) {
            all_match = false;
            record_test(false, "SIMD 1/10 [Scalar vs AVX2]: Mismatch on dim " + std::to_string(dim),
                        "Scalar: " + std::to_string(scalar_d) + ", AVX2: " + std::to_string(avx2_d));
            break;
        }
    }
    if (all_match) {
        record_test(true, "SIMD 1/10 [Scalar vs AVX2]: AVX2 matches Scalar Euclidean distance within 1e-4");
    }
}

// 2. Scalar vs AVX-512 Euclidean Distance
void test_simd_scalar_vs_avx512() {
    const auto& cpu = vectorforge::simd::get_cpu_features();
    if (!cpu.supports_avx512()) {
        record_test(true, "SIMD 2/10 [Scalar vs AVX-512]: Skipped (AVX-512 unsupported on this CPU)");
        return;
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    bool all_match = true;
    for (std::size_t dim : {16, 32, 64, 128, 768, 1536}) {
        std::vector<float> a(dim), b(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }
        const float scalar_d = vectorforge::simd::distance_scalar(a, b);
        const float avx512_d = vectorforge::simd::distance_avx512(a, b);

        if (!approx_equal(scalar_d, avx512_d, 1e-4f)) {
            all_match = false;
            record_test(false, "SIMD 2/10 [Scalar vs AVX-512]: Mismatch on dim " + std::to_string(dim),
                        "Scalar: " + std::to_string(scalar_d) + ", AVX-512: " + std::to_string(avx512_d));
            break;
        }
    }
    if (all_match) {
        record_test(true, "SIMD 2/10 [Scalar vs AVX-512]: AVX-512 matches Scalar Euclidean distance within 1e-4");
    }
}

// 3. Non-SIMD Dimension Sizes (Remainder / Scalar Tail Handling)
void test_simd_non_aligned_dimensions() {
    const auto& cpu = vectorforge::simd::get_cpu_features();
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    bool all_pass = true;
    for (std::size_t dim : {1, 3, 7, 9, 15, 17, 31, 33, 127, 255}) {
        std::vector<float> a(dim), b(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }
        const float scalar_d = vectorforge::simd::distance_scalar(a, b);
        const float disp_d = vectorforge::simd::distance(a, b);

        if (!approx_equal(scalar_d, disp_d, 1e-4f)) {
            all_pass = false;
            record_test(false, "SIMD 3/10 [Non-SIMD Dims]: Mismatch on dim " + std::to_string(dim));
            break;
        }

        if (cpu.supports_avx2()) {
            const float avx2_d = vectorforge::simd::distance_avx2(a, b);
            if (!approx_equal(scalar_d, avx2_d, 1e-4f)) {
                all_pass = false;
                record_test(false, "SIMD 3/10 [Non-SIMD Dims]: AVX2 tail mismatch on dim " + std::to_string(dim));
                break;
            }
        }
    }

    if (all_pass) {
        record_test(true, "SIMD 3/10 [Non-SIMD Dims]: Correct scalar tail processing across non-divisible dimensions (1..255)");
    }
}

// 4. Dimension Sizes Divisible by SIMD Width
void test_simd_aligned_dimensions() {
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    bool all_pass = true;
    for (std::size_t dim : {8, 16, 32, 64, 128, 256, 512, 1024}) {
        std::vector<float> a(dim), b(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }
        const float scalar_d = vectorforge::simd::distance_scalar(a, b);
        const float disp_d = vectorforge::simd::distance(a, b);

        if (!approx_equal(scalar_d, disp_d, 1e-4f)) {
            all_pass = false;
            break;
        }
    }
    record_test(all_pass, "SIMD 4/10 [Aligned Dims]: Correct distance computation on powers-of-two / divisible dims (8..1024)");
}

// 5. Zero Vectors
void test_simd_zero_vectors() {
    std::vector<float> z1(128, 0.0f);
    std::vector<float> z2(128, 0.0f);

    const float d = vectorforge::simd::distance(z1, z2);
    record_test(d == 0.0f, "SIMD 5/10 [Zero Vectors]: Distance between zero vectors is exactly 0.0");
}

// 6. Identical Vectors
void test_simd_identical_vectors() {
    std::vector<float> a(256, 3.14159f);
    std::vector<float> b(256, 3.14159f);

    const float d = vectorforge::simd::distance(a, b);
    record_test(d == 0.0f, "SIMD 6/10 [Identical Vectors]: Distance between identical non-zero vectors is exactly 0.0");
}

// 7. Large and Small Float Values (Dynamic Range Stability)
void test_simd_large_and_small_values() {
    std::vector<float> large_a(128, 10000.0f);
    std::vector<float> large_b(128, 10003.0f);
    const float d_large_scalar = vectorforge::simd::distance_scalar(large_a, large_b);
    const float d_large_simd = vectorforge::simd::distance(large_a, large_b);
    record_test(approx_equal(d_large_scalar, d_large_simd, 1e-3f), "SIMD 7/10 [Large Values]: Correct distance for large numbers (~10^4)");

    std::vector<float> small_a(128, 1e-4f);
    std::vector<float> small_b(128, 2e-4f);
    const float d_small_scalar = vectorforge::simd::distance_scalar(small_a, small_b);
    const float d_small_simd = vectorforge::simd::distance(small_a, small_b);
    record_test(approx_equal(d_small_scalar, d_small_simd, 1e-6f), "SIMD 7/10 [Small Values]: Correct distance for small numbers (~10^-4)");
}

// 8. Dimension Mismatch Validation in SIMD Module
void test_simd_dimension_mismatch() {
    std::vector<float> a(128, 1.0f);
    std::vector<float> b(64, 1.0f);

    bool scalar_threw = false;
    try { (void)vectorforge::simd::distance_scalar(a, b); } catch (const std::invalid_argument&) { scalar_threw = true; }

    bool simd_threw = false;
    try { (void)vectorforge::simd::distance(a, b); } catch (const std::invalid_argument&) { simd_threw = true; }

    bool cos_threw = false;
    try { (void)vectorforge::simd::cosine_similarity(a, b); } catch (const std::invalid_argument&) { cos_threw = true; }

    record_test(scalar_threw && simd_threw && cos_threw, "SIMD 8/10 [Dimension Mismatch]: All SIMD functions throw std::invalid_argument on size mismatch");
}

// 9. Runtime Dispatcher Verification
void test_simd_runtime_dispatcher() {
    const auto& cpu = vectorforge::simd::get_cpu_features();
    std::vector<float> a(128, 1.0f);
    std::vector<float> b(128, 2.0f);

    const float dispatched_d = vectorforge::simd::distance(a, b);

    if (cpu.supports_avx512()) {
        const float avx512_d = vectorforge::simd::distance_avx512(a, b);
        record_test(approx_equal(dispatched_d, avx512_d, 1e-5f), "SIMD 9/10 [Dispatcher]: Dispatched calls AVX-512 kernel");
    } else if (cpu.supports_avx2()) {
        const float avx2_d = vectorforge::simd::distance_avx2(a, b);
        record_test(approx_equal(dispatched_d, avx2_d, 1e-5f), "SIMD 9/10 [Dispatcher]: Dispatched calls AVX2 kernel");
    } else {
        const float scalar_d = vectorforge::simd::distance_scalar(a, b);
        record_test(approx_equal(dispatched_d, scalar_d, 1e-5f), "SIMD 9/10 [Dispatcher]: Dispatched calls Scalar kernel");
    }
}

// 10. Cosine Similarity Correctness & Zero-Norm Safety
void test_simd_cosine_similarity() {
    std::vector<float> a{1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> b{0.0f, 1.0f, 0.0f, 0.0f};
    const float cos_ortho = vectorforge::simd::cosine_similarity(a, b);
    record_test(approx_equal(cos_ortho, 0.0f), "SIMD 10/10 [Cosine Similarity]: Orthogonal vectors have cos = 0.0");

    std::vector<float> c{2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> d{1.0f, 2.0f, 3.0f, 4.0f};
    const float cos_parallel = vectorforge::simd::cosine_similarity(c, d);
    record_test(approx_equal(cos_parallel, 1.0f), "SIMD 10/10 [Cosine Similarity]: Parallel vectors have cos = 1.0");

    std::vector<float> zero_vec{0.0f, 0.0f, 0.0f, 0.0f};
    const float cos_zero = vectorforge::simd::cosine_similarity(c, zero_vec);
    record_test(cos_zero == 0.0f, "SIMD 10/10 [Cosine Similarity]: Zero-norm vector returns 0.0 without division-by-zero crash");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> r1(128), r2(128);
    for (std::size_t i = 0; i < 128; ++i) { r1[i] = dist(rng); r2[i] = dist(rng); }
    const float cos_sc = vectorforge::simd::cosine_similarity_scalar(r1, r2);
    const float cos_disp = vectorforge::simd::cosine_similarity(r1, r2);
    record_test(approx_equal(cos_sc, cos_disp, 1e-4f), "SIMD 10/10 [Cosine Similarity]: Dispatched Cosine similarity matches Scalar within 1e-4");
}

// =========================================================================
// PHASE 4 TESTS (Product Quantization & Asymmetric Distance Computation)
// =========================================================================

// Helper to generate random training vectors
std::vector<vectorforge::Vector> generate_test_vectors(std::size_t count, std::size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<vectorforge::Vector> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::vector<float> vals(dim);
        for (std::size_t d = 0; d < dim; ++d) {
            vals[d] = dist(rng);
        }
        result.emplace_back(std::move(vals));
    }
    return result;
}

// 1. Invalid Dimensions
void test_pq_invalid_dimensions() {
    bool dim_zero_threw = false;
    try {
        vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = 0, .num_subspaces = 4, .num_centroids = 16});
        auto data = generate_test_vectors(20, 0, 1);
        pq.train(data);
    } catch (const std::invalid_argument&) {
        dim_zero_threw = true;
    }
    record_test(dim_zero_threw, "PQ 1/15 [Invalid Dimensions]: Throws on dimension D = 0");

    bool sub_zero_threw = false;
    try {
        vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = 16, .num_subspaces = 0, .num_centroids = 16});
        auto data = generate_test_vectors(20, 16, 1);
        pq.train(data);
    } catch (const std::invalid_argument&) {
        sub_zero_threw = true;
    }
    record_test(sub_zero_threw, "PQ 1/15 [Invalid Dimensions]: Throws on num_subspaces m = 0");

    bool cent_invalid_threw = false;
    try {
        vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = 16, .num_subspaces = 4, .num_centroids = 500});
        auto data = generate_test_vectors(600, 16, 1);
        pq.train(data);
    } catch (const std::invalid_argument&) {
        cent_invalid_threw = true;
    }
    record_test(cent_invalid_threw, "PQ 1/15 [Invalid Dimensions]: Throws on num_centroids k > 256");
}

// 2. Dimension Not Divisible by m
void test_pq_dimension_not_divisible() {
    bool threw = false;
    try {
        vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = 128, .num_subspaces = 15, .num_centroids = 16});
        auto data = generate_test_vectors(20, 128, 1);
        pq.train(data);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    record_test(threw, "PQ 2/15 [Divisibility]: Throws std::invalid_argument when D=128 is not divisible by m=15");
}

// 3. Training with Valid Data
void test_pq_valid_training() {
    constexpr std::size_t D = 32;
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 16;
    vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = D, .num_subspaces = M, .num_centroids = K, .max_iterations = 10, .random_seed = 42});
    record_test(!pq.is_trained(), "PQ 3/15 [Valid Training]: is_trained() is false initially");

    const auto train_data = generate_test_vectors(100, D, 123);
    pq.train(train_data);
    record_test(pq.is_trained(), "PQ 3/15 [Valid Training]: is_trained() is true after training");
}

// 4. Training with Insufficient Data
void test_pq_insufficient_training_data() {
    vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = 16, .num_subspaces = 4, .num_centroids = 32});
    const auto small_data = generate_test_vectors(10, 16, 123); // N=10 < K=32

    bool threw = false;
    try {
        pq.train(small_data);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    record_test(threw, "PQ 4/15 [Insufficient Data]: Throws error when training vector count N < k centroids");
}

// 5. Codebook Dimensions
void test_pq_codebook_dimensions() {
    constexpr std::size_t D = 64;
    constexpr std::size_t M = 8;
    constexpr std::size_t K = 16;
    constexpr std::size_t DS = D / M; // 8

    vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = D, .num_subspaces = M, .num_centroids = K, .max_iterations = 5, .random_seed = 1});
    pq.train(generate_test_vectors(80, D, 42));

    record_test(pq.subvector_dim() == DS, "PQ 5/15 [Codebook Dims]: subvector_dim() == 8");

    bool all_codebooks_valid = true;
    for (std::size_t s = 0; s < M; ++s) {
        const auto& cents = pq.subspace_centroids(s);
        if (cents.size() != K * DS) {
            all_codebooks_valid = false;
            break;
        }
    }
    record_test(all_codebooks_valid, "PQ 5/15 [Codebook Dims]: all m=8 codebooks have exact size k * ds = 16 * 8 = 128 floats");
}

// 6. Vector Encoding
void test_pq_encoding() {
    constexpr std::size_t D = 32;
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 16;

    vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = D, .num_subspaces = M, .num_centroids = K, .max_iterations = 10, .random_seed = 42});
    pq.train(generate_test_vectors(100, D, 42));

    const auto vec = generate_test_vectors(1, D, 99)[0];
    const auto code = pq.encode(vec);

    record_test(code.size() == M, "PQ 6/15 [Encoding]: Encoded code size matches num_subspaces m=4");
}

// 7. Vector Decoding
void test_pq_decoding() {
    constexpr std::size_t D = 32;
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 16;

    vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = D, .num_subspaces = M, .num_centroids = K, .max_iterations = 10, .random_seed = 42});
    pq.train(generate_test_vectors(100, D, 42));

    const auto vec = generate_test_vectors(1, D, 101)[0];
    const auto code = pq.encode(vec);
    const auto decoded = pq.decode(code);

    record_test(decoded.dimension() == D, "PQ 7/15 [Decoding]: Decoded vector dimension matches original D=32");
}

// 8. PQ Code Value Range
void test_pq_code_range() {
    constexpr std::size_t D = 32;
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 16;

    vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = D, .num_subspaces = M, .num_centroids = K, .max_iterations = 10, .random_seed = 42});
    pq.train(generate_test_vectors(100, D, 42));

    const auto test_set = generate_test_vectors(20, D, 777);
    bool in_range = true;
    for (const auto& v : test_set) {
        const auto code = pq.encode(v);
        for (uint8_t byte_val : code) {
            if (byte_val >= K) {
                in_range = false;
                break;
            }
        }
        if (!in_range) break;
    }
    record_test(in_range, "PQ 8/15 [Code Range]: All encoded centroid IDs satisfy 0 <= code[s] < k=16");
}

// 9. Reconstruction Error
void test_pq_reconstruction_error() {
    constexpr std::size_t D = 16;
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 32;

    vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = D, .num_subspaces = M, .num_centroids = K, .max_iterations = 20, .random_seed = 42});
    const auto train_data = generate_test_vectors(150, D, 42);
    pq.train(train_data);

    double total_l2_error = 0.0;
    for (std::size_t i = 0; i < 50; ++i) {
        const auto& original = train_data[i];
        const auto code = pq.encode(original);
        const auto decoded = pq.decode(code);
        total_l2_error += vectorforge::euclidean_distance(original, decoded);
    }
    const double avg_error = total_l2_error / 50.0;

    record_test(avg_error < 2.0, "PQ 9/15 [Reconstruction Error]: Average L2 reconstruction error is bounded (Got: " + std::to_string(avg_error) + ")");
}

// 10. ADC Lookup Table Correctness
void test_pq_adc_lut_correctness() {
    constexpr std::size_t D = 16;
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 8;
    constexpr std::size_t DS = D / M;

    vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = D, .num_subspaces = M, .num_centroids = K, .max_iterations = 10, .random_seed = 42});
    pq.train(generate_test_vectors(50, D, 42));

    const auto query = generate_test_vectors(1, D, 999)[0];
    const auto lut = pq.compute_adc_table(query);

    record_test(lut.num_subspaces == M && lut.num_centroids == K, "PQ 10/15 [ADC LUT]: LUT shape matches m=4, k=8");

    bool entries_exact = true;
    for (std::size_t s = 0; s < M; ++s) {
        const float* q_sub = query.data() + (s * DS);
        const auto& cents = pq.subspace_centroids(s);
        for (std::size_t c = 0; c < K; ++c) {
            const float* c_sub = cents.data() + (c * DS);
            float expected_sq = 0.0f;
            for (std::size_t d = 0; d < DS; ++d) {
                const float diff = q_sub[d] - c_sub[d];
                expected_sq += diff * diff;
            }
            if (!approx_equal(lut.get(s, c), expected_sq, 1e-4f)) {
                entries_exact = false;
                break;
            }
        }
        if (!entries_exact) break;
    }
    record_test(entries_exact, "PQ 10/15 [ADC LUT]: Every LUT entry matches exact squared subvector Euclidean distance");
}

// 11. Approximate Distance Sanity
void test_pq_approximate_distance_sanity() {
    constexpr std::size_t D = 32;
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 16;

    vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = D, .num_subspaces = M, .num_centroids = K, .max_iterations = 15, .random_seed = 42});
    pq.train(generate_test_vectors(100, D, 42));

    const auto vec = generate_test_vectors(1, D, 123)[0];
    const auto code = pq.encode(vec);

    // Approximate distance between vector and its own PQ code equals the reconstruction distance
    const float approx_d = pq.approximate_distance(vec, code);
    const auto decoded = pq.decode(code);
    const float exact_recon_d = vectorforge::euclidean_distance(vec, decoded);

    record_test(approx_equal(approx_d, exact_recon_d, 1e-4f),
                "PQ 11/15 [Distance Sanity]: ADC distance to self matches reconstruction distance (Approx: " +
                std::to_string(approx_d) + ", Recon: " + std::to_string(exact_recon_d) + ")");
}

// 12. Exact vs PQ Distance Comparison (Correlation)
void test_pq_exact_vs_pq_distance() {
    constexpr std::size_t D = 32;
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 16;

    vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = D, .num_subspaces = M, .num_centroids = K, .max_iterations = 15, .random_seed = 42});
    const auto data = generate_test_vectors(100, D, 42);
    pq.train(data);

    const auto query = generate_test_vectors(1, D, 555)[0];
    const auto lut = pq.compute_adc_table(query);

    double sum_diff = 0.0;
    for (std::size_t i = 0; i < 30; ++i) {
        const float exact_d = vectorforge::euclidean_distance(query, data[i]);
        const auto code = pq.encode(data[i]);
        const float adc_d = pq.approximate_distance(lut, code);
        sum_diff += std::fabs(exact_d - adc_d);
    }
    const double avg_diff = sum_diff / 30.0;

    record_test(avg_diff < 1.5, "PQ 12/15 [Distance Correlation]: ADC approximate distances closely track exact Euclidean distances (Avg diff: " + std::to_string(avg_diff) + ")");
}

// 13. Deterministic Training Behavior
void test_pq_deterministic_training() {
    constexpr std::size_t D = 16;
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 8;

    vectorforge::ProductQuantizer pq1(vectorforge::PQConfig{.dimension = D, .num_subspaces = M, .num_centroids = K, .max_iterations = 10, .random_seed = 12345});
    vectorforge::ProductQuantizer pq2(vectorforge::PQConfig{.dimension = D, .num_subspaces = M, .num_centroids = K, .max_iterations = 10, .random_seed = 12345});

    const auto data = generate_test_vectors(50, D, 99);
    pq1.train(data);
    pq2.train(data);

    bool identical = true;
    for (std::size_t s = 0; s < M; ++s) {
        const auto& c1 = pq1.subspace_centroids(s);
        const auto& c2 = pq2.subspace_centroids(s);
        for (std::size_t i = 0; i < c1.size(); ++i) {
            if (c1[i] != c2[i]) {
                identical = false;
                break;
            }
        }
        if (!identical) break;
    }
    record_test(identical, "PQ 13/15 [Deterministic Training]: Same seed produces bit-identical codebooks across independent runs");
}

// 14. Compression Size Calculation
void test_pq_compression_size_calculation() {
    vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = 128, .num_subspaces = 16, .num_centroids = 256});

    const std::size_t orig_bytes = pq.original_bytes_per_vector();
    const std::size_t comp_bytes = pq.compressed_bytes_per_vector();
    const double ratio = pq.compression_ratio();

    record_test(orig_bytes == 512, "PQ 14/15 [Compression Size]: original_bytes_per_vector() == 512 (128 * 4)");
    record_test(comp_bytes == 16, "PQ 14/15 [Compression Size]: compressed_bytes_per_vector() == 16 (16 * 1)");
    record_test(approx_equal(static_cast<float>(ratio), 32.0f), "PQ 14/15 [Compression Size]: compression_ratio() == 32.0x");
}

// 15. PQ + HNSW Experimental Path
void test_pq_hnsw_experimental_path() {
    constexpr std::size_t D = 16;
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 8;
    constexpr std::size_t N = 50;

    const auto dataset = generate_test_vectors(N, D, 42);

    // Train PQ
    vectorforge::ProductQuantizer pq(vectorforge::PQConfig{.dimension = D, .num_subspaces = M, .num_centroids = K, .max_iterations = 10, .random_seed = 42});
    pq.train(dataset);

    // Encode all vectors
    std::vector<vectorforge::PQCode> pq_db;
    pq_db.reserve(N);
    for (const auto& v : dataset) pq_db.push_back(pq.encode(v));

    // Build HNSW index
    vectorforge::HNSWIndex hnsw(vectorforge::HNSWConfig{.M = 4, .ef_construction = 32, .ef_search = 16, .random_seed = 42});
    for (const auto& v : dataset) hnsw.insert(v);

    // Query with candidate reranking via PQ ADC
    const auto query = generate_test_vectors(1, D, 999)[0];
    const auto candidates = hnsw.search(query, 10);
    const auto lut = pq.compute_adc_table(query);

    std::vector<vectorforge::SearchResult> reranked;
    for (const auto& c : candidates) {
        reranked.push_back({c.id, pq.approximate_distance(lut, pq_db[c.id])});
    }
    std::sort(reranked.begin(), reranked.end(), [](const auto& a, const auto& b) {
        return a.distance < b.distance;
    });

    record_test(reranked.size() == 10, "PQ 15/15 [HNSW+PQ Path]: Returns 10 reranked candidates without error");
    record_test(reranked[0].distance <= reranked.back().distance, "PQ 15/15 [HNSW+PQ Path]: Reranked candidates ordered monotonically by ADC distance");
}

} // anonymous namespace

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << " Running VectorForge Test Suite (Phases 1, 2, 3, and 4)   " << std::endl;
    std::cout << "==========================================================\n" << std::endl;

    std::cout << "--- [Phase 1: Brute-Force Baseline Unit Tests] ---" << std::endl;
    std::cout << "[Suite 1/9] Empty Index Tests:" << std::endl;
    test_empty_index();

    std::cout << "\n[Suite 2/9] Single Vector Tests:" << std::endl;
    test_single_vector();

    std::cout << "\n[Suite 3/9] Multiple Vectors Tests:" << std::endl;
    test_multiple_vectors();

    std::cout << "\n[Suite 4/9] Euclidean Distance Calculation Tests:" << std::endl;
    test_euclidean_distance_calculation();

    std::cout << "\n[Suite 5/9] Nearest Neighbor Ordering Tests:" << std::endl;
    test_nearest_neighbor_ordering();

    std::cout << "\n[Suite 6/9] k = 1 Tests:" << std::endl;
    test_k_equals_one();

    std::cout << "\n[Suite 7/9] k > Size Tests:" << std::endl;
    test_k_greater_than_size();

    std::cout << "\n[Suite 8/9] Dimension Mismatch Handling Tests:" << std::endl;
    test_dimension_mismatch();

    std::cout << "\n[Suite 9/9] Identical Vector Distance Zero Tests:" << std::endl;
    test_identical_vector_distance_zero();

    std::cout << "\n--- [Phase 2: HNSW Approximate Index Unit Tests] ---" << std::endl;
    std::cout << "[HNSW 1/15] Empty HNSW Index Tests:" << std::endl;
    test_hnsw_empty_index();

    std::cout << "\n[HNSW 2/15] Single Vector Tests:" << std::endl;
    test_hnsw_single_vector();

    std::cout << "\n[HNSW 3/15] Multiple Vectors Tests:" << std::endl;
    test_hnsw_multiple_vectors();

    std::cout << "\n[HNSW 4/15] Insert Sequence Tests:" << std::endl;
    test_hnsw_insert_sequence();

    std::cout << "\n[HNSW 5/15] k = 1 Tests:" << std::endl;
    test_hnsw_k_equals_one();

    std::cout << "\n[HNSW 6/15] k > 1 Tests:" << std::endl;
    test_hnsw_k_greater_one();

    std::cout << "\n[HNSW 7/15] k > Dataset Size Tests:" << std::endl;
    test_hnsw_k_greater_than_size();

    std::cout << "\n[HNSW 8/15] Identical Query Vector Tests:" << std::endl;
    test_hnsw_identical_vector();

    std::cout << "\n[HNSW 9/15] Dimension Mismatch Handling Tests:" << std::endl;
    test_hnsw_dimension_mismatch();

    std::cout << "\n[HNSW 10/15] Entry Point Tracking Tests:" << std::endl;
    test_hnsw_entry_point();

    std::cout << "\n[HNSW 11/15] Multi-Layer Graph Tests:" << std::endl;
    test_hnsw_multiple_layers();

    std::cout << "\n[HNSW 12/15] Neighbor Connectivity Tests:" << std::endl;
    test_hnsw_neighbor_connections();

    std::cout << "\n[HNSW 13/15] Degree Limit Enforcement Tests:" << std::endl;
    test_hnsw_neighbor_degree_limits();

    std::cout << "\n[HNSW 14/15] Monotonic Distance Ordering Tests:" << std::endl;
    test_hnsw_distance_monotonicity();

    std::cout << "\n[HNSW 15/15] Ground Truth Recall Comparison Tests:" << std::endl;
    test_hnsw_ground_truth_comparison();

    std::cout << "\n--- [Phase 3: SIMD Acceleration Unit Tests] ---" << std::endl;
    std::cout << "[SIMD 1/10] Scalar vs AVX2 Euclidean Distance:" << std::endl;
    test_simd_scalar_vs_avx2();

    std::cout << "\n[SIMD 2/10] Scalar vs AVX-512 Euclidean Distance:" << std::endl;
    test_simd_scalar_vs_avx512();

    std::cout << "\n[SIMD 3/10] Non-SIMD Dimension Sizes:" << std::endl;
    test_simd_non_aligned_dimensions();

    std::cout << "\n[SIMD 4/10] Aligned Dimension Sizes:" << std::endl;
    test_simd_aligned_dimensions();

    std::cout << "\n[SIMD 5/10] Zero Vectors:" << std::endl;
    test_simd_zero_vectors();

    std::cout << "\n[SIMD 6/10] Identical Vectors:" << std::endl;
    test_simd_identical_vectors();

    std::cout << "\n[SIMD 7/10] Large and Small Values:" << std::endl;
    test_simd_large_and_small_values();

    std::cout << "\n[SIMD 8/10] Dimension Mismatch Validation:" << std::endl;
    test_simd_dimension_mismatch();

    std::cout << "\n[SIMD 9/10] Runtime Dispatcher:" << std::endl;
    test_simd_runtime_dispatcher();

    std::cout << "\n[SIMD 10/10] Cosine Similarity & Zero-Norm Safety:" << std::endl;
    test_simd_cosine_similarity();

    std::cout << "\n--- [Phase 4: Product Quantization (PQ) Unit Tests] ---" << std::endl;
    std::cout << "[PQ 1/15] Invalid Dimension Tests:" << std::endl;
    test_pq_invalid_dimensions();

    std::cout << "\n[PQ 2/15] Dimension Divisibility Tests:" << std::endl;
    test_pq_dimension_not_divisible();

    std::cout << "\n[PQ 3/15] Valid Training Tests:" << std::endl;
    test_pq_valid_training();

    std::cout << "\n[PQ 4/15] Insufficient Training Data Tests:" << std::endl;
    test_pq_insufficient_training_data();

    std::cout << "\n[PQ 5/15] Codebook Dimensions Tests:" << std::endl;
    test_pq_codebook_dimensions();

    std::cout << "\n[PQ 6/15] Vector Encoding Tests:" << std::endl;
    test_pq_encoding();

    std::cout << "\n[PQ 7/15] Vector Decoding Tests:" << std::endl;
    test_pq_decoding();

    std::cout << "\n[PQ 8/15] PQ Code Range Tests:" << std::endl;
    test_pq_code_range();

    std::cout << "\n[PQ 9/15] Reconstruction Error Tests:" << std::endl;
    test_pq_reconstruction_error();

    std::cout << "\n[PQ 10/15] ADC Lookup Table Correctness Tests:" << std::endl;
    test_pq_adc_lut_correctness();

    std::cout << "\n[PQ 11/15] Approximate Distance Sanity Tests:" << std::endl;
    test_pq_approximate_distance_sanity();

    std::cout << "\n[PQ 12/15] Distance Correlation Tests:" << std::endl;
    test_pq_exact_vs_pq_distance();

    std::cout << "\n[PQ 13/15] Deterministic Training Tests:" << std::endl;
    test_pq_deterministic_training();

    std::cout << "\n[PQ 14/15] Compression Size Calculations Tests:" << std::endl;
    test_pq_compression_size_calculation();

    std::cout << "\n[PQ 15/15] PQ + HNSW Experimental Path Tests:" << std::endl;
    test_pq_hnsw_experimental_path();

    std::cout << "\n==========================================================" << std::endl;
    std::cout << " Test Summary: " << passed_tests << " / " << total_tests << " Passed";
    if (failed_tests > 0) {
        std::cout << " (" << failed_tests << " FAILED)";
    }
    std::cout << "\n==========================================================" << std::endl;

    return (failed_tests == 0) ? 0 : 1;
}
