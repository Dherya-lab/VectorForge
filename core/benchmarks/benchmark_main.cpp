#include "vector.hpp"
#include "vector_index.hpp"
#include "hnsw_index.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

namespace {

// Deterministic float vector generator using standard uniform distribution
std::vector<vectorforge::Vector> generate_deterministic_vectors(
    std::size_t count,
    std::size_t dimension,
    uint32_t seed = 42
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<vectorforge::Vector> dataset;
    dataset.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        std::vector<float> values(dimension);
        for (std::size_t d = 0; d < dimension; ++d) {
            values[d] = dist(rng);
        }
        dataset.emplace_back(std::move(values));
    }

    return dataset;
}

struct BenchmarkMetrics {
    double total_time_ms{0.0};
    double qps{0.0};
    double avg_latency_us{0.0};
    double min_latency_us{0.0};
    double max_latency_us{0.0};
    double p50_latency_us{0.0};
    double p95_latency_us{0.0};
    double p99_latency_us{0.0};
    double recall_at_k{0.0};
    double avg_nodes_visited{0.0};
    double avg_distance_calcs{0.0};
};

} // anonymous namespace

int main() {
    std::cout << "==========================================================================" << std::endl;
    std::cout << " VectorForge — Phase 2 Comparative Benchmark: Brute-Force vs HNSW Index   " << std::endl;
    std::cout << "==========================================================================\n" << std::endl;

    constexpr std::size_t NUM_VECTORS = 1000;
    constexpr std::size_t DIMENSION = 128;
    constexpr std::size_t NUM_QUERIES = 100;
    constexpr std::size_t K = 10;
    constexpr uint32_t DATASET_SEED = 1337;
    constexpr uint32_t QUERY_SEED = 9999;

    std::cout << "Benchmark Configuration:" << std::endl;
    std::cout << "  - Number of indexed vectors : " << NUM_VECTORS << std::endl;
    std::cout << "  - Vector dimensionality     : " << DIMENSION << " floats (" 
              << (DIMENSION * sizeof(float)) << " bytes/vector)" << std::endl;
    std::cout << "  - Number of search queries  : " << NUM_QUERIES << std::endl;
    std::cout << "  - k (nearest neighbors)     : " << K << std::endl;
    std::cout << "  - Dataset seed              : " << DATASET_SEED << std::endl;
    std::cout << "  - Query seed                : " << QUERY_SEED << "\n" << std::endl;

    // 1. Generate Dataset & Queries
    std::cout << "Generating deterministic dataset and queries..." << std::endl;
    const auto dataset = generate_deterministic_vectors(NUM_VECTORS, DIMENSION, DATASET_SEED);
    const auto queries = generate_deterministic_vectors(NUM_QUERIES, DIMENSION, QUERY_SEED);

    // ========================================================================
    // A. Phase 1 Brute-Force (Ground Truth Baseline)
    // ========================================================================
    std::cout << "\n--------------------------------------------------------------------------" << std::endl;
    std::cout << " [1/3] Benchmarking Phase 1 Brute-Force Index (Ground Truth)...            " << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;

    vectorforge::VectorIndex bf_index;
    for (const auto& vec : dataset) {
        bf_index.insert(vec);
    }

    // Warm-up
    for (std::size_t i = 0; i < 5; ++i) {
        (void)bf_index.search(queries[i], K);
    }

    std::vector<std::vector<std::size_t>> ground_truth(NUM_QUERIES);
    std::vector<double> bf_latencies_us;
    bf_latencies_us.reserve(NUM_QUERIES);

    const auto bf_bench_start = std::chrono::high_resolution_clock::now();
    for (std::size_t q = 0; q < NUM_QUERIES; ++q) {
        const auto q_start = std::chrono::high_resolution_clock::now();
        const auto results = bf_index.search(queries[q], K);
        const auto q_end = std::chrono::high_resolution_clock::now();

        bf_latencies_us.push_back(std::chrono::duration<double, std::micro>(q_end - q_start).count());
        for (const auto& res : results) {
            ground_truth[q].push_back(res.id);
        }
    }
    const auto bf_bench_end = std::chrono::high_resolution_clock::now();
    const double bf_total_ms = std::chrono::duration<double, std::milli>(bf_bench_end - bf_bench_start).count();

    std::sort(bf_latencies_us.begin(), bf_latencies_us.end());
    const double bf_avg_us = std::accumulate(bf_latencies_us.begin(), bf_latencies_us.end(), 0.0) / static_cast<double>(NUM_QUERIES);
    const double bf_p50_us = bf_latencies_us[NUM_QUERIES * 50 / 100];
    const double bf_p95_us = bf_latencies_us[NUM_QUERIES * 95 / 100];
    const double bf_qps = (static_cast<double>(NUM_QUERIES) / bf_total_ms) * 1000.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Brute-Force Average Latency : " << bf_avg_us << " us (" << (bf_avg_us / 1000.0) << " ms)" << std::endl;
    std::cout << "  Brute-Force Median (p50)    : " << bf_p50_us << " us" << std::endl;
    std::cout << "  Brute-Force 95th % (p95)    : " << bf_p95_us << " us" << std::endl;
    std::cout << "  Brute-Force Throughput      : " << bf_qps << " QPS" << std::endl;
    std::cout << "  Vectors examined per query  : " << NUM_VECTORS << " (100% full scan)" << std::endl;

    // ========================================================================
    // B. Phase 2 HNSW Index Build & Benchmark (Default Config)
    // ========================================================================
    std::cout << "\n--------------------------------------------------------------------------" << std::endl;
    std::cout << " [2/3] Building & Benchmarking Phase 2 HNSW Index (M=16, efC=100, efS=50)..." << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;

    vectorforge::HNSWConfig default_config{
        .M = 16,
        .ef_construction = 100,
        .ef_search = 50,
        .random_seed = 42
    };

    const auto build_start = std::chrono::high_resolution_clock::now();
    vectorforge::HNSWIndex hnsw_index(default_config);
    for (const auto& vec : dataset) {
        hnsw_index.insert(vec);
    }
    const auto build_end = std::chrono::high_resolution_clock::now();
    const double build_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();

    std::cout << "  HNSW Index Built in         : " << build_time_ms << " ms" << std::endl;
    std::cout << "  HNSW Max Graph Layer Level  : " << hnsw_index.max_level() << std::endl;
    std::cout << "  HNSW Entry Point Node ID    : " << *hnsw_index.entry_point() << std::endl;

    // Warm-up
    for (std::size_t i = 0; i < 5; ++i) {
        (void)hnsw_index.search(queries[i], K);
    }

    std::vector<double> hnsw_latencies_us;
    hnsw_latencies_us.reserve(NUM_QUERIES);
    std::size_t total_visited = 0;
    std::size_t total_dist_calcs = 0;
    std::size_t total_matches = 0;

    const auto hnsw_bench_start = std::chrono::high_resolution_clock::now();
    for (std::size_t q = 0; q < NUM_QUERIES; ++q) {
        vectorforge::HNSWStats stats;
        const auto q_start = std::chrono::high_resolution_clock::now();
        const auto results = hnsw_index.search_with_stats(queries[q], K, stats);
        const auto q_end = std::chrono::high_resolution_clock::now();

        hnsw_latencies_us.push_back(std::chrono::duration<double, std::micro>(q_end - q_start).count());
        total_visited += stats.nodes_visited;
        total_dist_calcs += stats.distance_computations;

        std::unordered_set<std::size_t> gt_set(ground_truth[q].begin(), ground_truth[q].end());
        for (const auto& res : results) {
            if (gt_set.find(res.id) != gt_set.end()) {
                ++total_matches;
            }
        }
    }
    const auto hnsw_bench_end = std::chrono::high_resolution_clock::now();
    const double hnsw_total_ms = std::chrono::duration<double, std::milli>(hnsw_bench_end - hnsw_bench_start).count();

    std::sort(hnsw_latencies_us.begin(), hnsw_latencies_us.end());
    const double hnsw_avg_us = std::accumulate(hnsw_latencies_us.begin(), hnsw_latencies_us.end(), 0.0) / static_cast<double>(NUM_QUERIES);
    const double hnsw_p50_us = hnsw_latencies_us[NUM_QUERIES * 50 / 100];
    const double hnsw_p95_us = hnsw_latencies_us[NUM_QUERIES * 95 / 100];
    const double hnsw_p99_us = hnsw_latencies_us[NUM_QUERIES * 99 / 100];
    const double hnsw_qps = (static_cast<double>(NUM_QUERIES) / hnsw_total_ms) * 1000.0;
    const double recall_k = static_cast<double>(total_matches) / static_cast<double>(NUM_QUERIES * K);
    const double avg_visited = static_cast<double>(total_visited) / static_cast<double>(NUM_QUERIES);
    const double avg_dist_calcs = static_cast<double>(total_dist_calcs) / static_cast<double>(NUM_QUERIES);

    std::cout << "\n  HNSW Recall@" << K << "                  : " << (recall_k * 100.0) << " %" << std::endl;
    std::cout << "  HNSW Average Latency        : " << hnsw_avg_us << " us (" << (hnsw_avg_us / 1000.0) << " ms)" << std::endl;
    std::cout << "  HNSW Median (p50)           : " << hnsw_p50_us << " us" << std::endl;
    std::cout << "  HNSW 95th % (p95)           : " << hnsw_p95_us << " us" << std::endl;
    std::cout << "  HNSW 99th % (p99)           : " << hnsw_p99_us << " us" << std::endl;
    std::cout << "  HNSW Throughput             : " << hnsw_qps << " QPS" << std::endl;
    std::cout << "  Avg Graph Nodes Visited     : " << avg_visited << " nodes / query ("
              << ((avg_visited / static_cast<double>(NUM_VECTORS)) * 100.0) << "% of dataset)" << std::endl;
    std::cout << "  Avg Distance Computations   : " << avg_dist_calcs << " / query" << std::endl;

    // ========================================================================
    // C. Parameter Experiment: efSearch Trade-off Study
    // ========================================================================
    std::cout << "\n--------------------------------------------------------------------------" << std::endl;
    std::cout << " [3/3] Parameter Experiment: Evaluating efSearch Trade-off Study           " << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;

    const std::vector<std::size_t> ef_search_values = {10, 25, 50, 100, 200};

    std::cout << "\n+----------+-----------+----------------+--------------+-----------------+" << std::endl;
    std::cout << "| efSearch | Recall@10 | Latency (avg)  | Nodes Visited| Throughput(QPS) |" << std::endl;
    std::cout << "+----------+-----------+----------------+--------------+-----------------+" << std::endl;

    for (std::size_t ef_val : ef_search_values) {
        hnsw_index.set_ef_search(ef_val);

        std::vector<double> lat_list;
        lat_list.reserve(NUM_QUERIES);
        std::size_t matches = 0;
        std::size_t nodes_sum = 0;

        const auto start_exp = std::chrono::high_resolution_clock::now();
        for (std::size_t q = 0; q < NUM_QUERIES; ++q) {
            vectorforge::HNSWStats stats;
            const auto q_start = std::chrono::high_resolution_clock::now();
            const auto results = hnsw_index.search_with_stats(queries[q], K, stats);
            const auto q_end = std::chrono::high_resolution_clock::now();

            lat_list.push_back(std::chrono::duration<double, std::micro>(q_end - q_start).count());
            nodes_sum += stats.nodes_visited;

            std::unordered_set<std::size_t> gt_set(ground_truth[q].begin(), ground_truth[q].end());
            for (const auto& res : results) {
                if (gt_set.find(res.id) != gt_set.end()) {
                    ++matches;
                }
            }
        }
        const auto end_exp = std::chrono::high_resolution_clock::now();
        const double exp_total_ms = std::chrono::duration<double, std::milli>(end_exp - start_exp).count();

        const double exp_avg_lat = std::accumulate(lat_list.begin(), lat_list.end(), 0.0) / static_cast<double>(NUM_QUERIES);
        const double exp_recall = static_cast<double>(matches) / static_cast<double>(NUM_QUERIES * K);
        const double exp_avg_nodes = static_cast<double>(nodes_sum) / static_cast<double>(NUM_QUERIES);
        const double exp_qps = (static_cast<double>(NUM_QUERIES) / exp_total_ms) * 1000.0;

        std::cout << "| " << std::setw(8) << ef_val
                  << " | " << std::setw(8) << (exp_recall * 100.0) << "%"
                  << " | " << std::setw(11) << exp_avg_lat << " us"
                  << " | " << std::setw(12) << exp_avg_nodes
                  << " | " << std::setw(12) << exp_qps << " QPS"
                  << " |" << std::endl;
    }
    std::cout << "+----------+-----------+----------------+--------------+-----------------+\n" << std::endl;

    std::cout << "==========================================================================" << std::endl;
    std::cout << " Phase 2 Benchmark Completed Successfully!" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    return 0;
}
