#include "vector.hpp"
#include "vector_index.hpp"
#include "hnsw_index.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

// Deterministic float generator using standard uniform distribution
std::vector<vectorforge::Vector> generate_deterministic_vectors(
    std::size_t count,
    std::size_t dimension,
    uint32_t seed
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

struct ScaleResult {
    std::size_t dataset_size{0};
    double bf_avg_latency_us{0.0};
    double bf_qps{0.0};
    std::size_t bf_nodes_examined{0};
    
    double hnsw_avg_latency_us{0.0};
    double hnsw_qps{0.0};
    double hnsw_nodes_visited{0.0};
    double hnsw_recall_at_10{0.0};
    
    double speedup{0.0};
    std::string comparison_verdict;
    double index_build_time_ms{0.0};
};

ScaleResult run_scale_evaluation(
    std::size_t dataset_size,
    std::size_t dimension,
    const std::vector<vectorforge::Vector>& queries,
    std::size_t k,
    const vectorforge::HNSWConfig& hnsw_config,
    uint32_t dataset_seed
) {
    std::cout << "\n==========================================================================" << std::endl;
    std::cout << " Evaluating Dataset Scale: N = " << dataset_size << " vectors (" << dimension << " dims)" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    std::cout << "  [1/4] Generating " << dataset_size << " vectors..." << std::endl;
    const auto dataset = generate_deterministic_vectors(dataset_size, dimension, dataset_seed);

    // 1. Build Brute-Force Index
    std::cout << "  [2/4] Building Brute-Force Index..." << std::endl;
    vectorforge::VectorIndex bf_index;
    for (const auto& vec : dataset) {
        bf_index.insert(vec);
    }

    // 2. Build HNSW Index
    std::cout << "  [3/4] Building HNSW Index (M=" << hnsw_config.M 
              << ", efC=" << hnsw_config.ef_construction 
              << ", efS=" << hnsw_config.ef_search << ")..." << std::endl;
    const auto build_start = std::chrono::high_resolution_clock::now();
    vectorforge::HNSWIndex hnsw_index(hnsw_config);
    for (const auto& vec : dataset) {
        hnsw_index.insert(vec);
    }
    const auto build_end = std::chrono::high_resolution_clock::now();
    const double build_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    std::cout << "        HNSW built in " << build_time_ms << " ms (Max Level: " << hnsw_index.max_level() << ")" << std::endl;

    const std::size_t num_queries = queries.size();

    // 3. Brute-Force Search (Ground Truth)
    std::cout << "  [4/4] Executing " << num_queries << " queries (k=" << k << ") on both indexes..." << std::endl;
    
    // Warm-up
    for (std::size_t i = 0; i < std::min<std::size_t>(3, num_queries); ++i) {
        (void)bf_index.search(queries[i], k);
        (void)hnsw_index.search(queries[i], k);
    }

    std::vector<std::vector<std::size_t>> ground_truth(num_queries);
    std::vector<double> bf_latencies_us;
    bf_latencies_us.reserve(num_queries);

    const auto bf_start = std::chrono::high_resolution_clock::now();
    for (std::size_t q = 0; q < num_queries; ++q) {
        const auto q_start = std::chrono::high_resolution_clock::now();
        const auto results = bf_index.search(queries[q], k);
        const auto q_end = std::chrono::high_resolution_clock::now();

        bf_latencies_us.push_back(std::chrono::duration<double, std::micro>(q_end - q_start).count());
        for (const auto& r : results) {
            ground_truth[q].push_back(r.id);
        }
    }
    const auto bf_end = std::chrono::high_resolution_clock::now();
    const double bf_total_ms = std::chrono::duration<double, std::milli>(bf_end - bf_start).count();
    const double bf_avg_lat_us = std::accumulate(bf_latencies_us.begin(), bf_latencies_us.end(), 0.0) / static_cast<double>(num_queries);
    const double bf_qps = (static_cast<double>(num_queries) / bf_total_ms) * 1000.0;

    // 4. HNSW Search
    std::vector<double> hnsw_latencies_us;
    hnsw_latencies_us.reserve(num_queries);
    std::size_t total_visited = 0;
    std::size_t total_matches = 0;

    const auto hnsw_start = std::chrono::high_resolution_clock::now();
    for (std::size_t q = 0; q < num_queries; ++q) {
        vectorforge::HNSWStats stats;
        const auto q_start = std::chrono::high_resolution_clock::now();
        const auto results = hnsw_index.search_with_stats(queries[q], k, stats);
        const auto q_end = std::chrono::high_resolution_clock::now();

        hnsw_latencies_us.push_back(std::chrono::duration<double, std::micro>(q_end - q_start).count());
        total_visited += stats.nodes_visited;

        std::unordered_set<std::size_t> gt_set(ground_truth[q].begin(), ground_truth[q].end());
        for (const auto& r : results) {
            if (gt_set.find(r.id) != gt_set.end()) {
                ++total_matches;
            }
        }
    }
    const auto hnsw_end = std::chrono::high_resolution_clock::now();
    const double hnsw_total_ms = std::chrono::duration<double, std::milli>(hnsw_end - hnsw_start).count();
    const double hnsw_avg_lat_us = std::accumulate(hnsw_latencies_us.begin(), hnsw_latencies_us.end(), 0.0) / static_cast<double>(num_queries);
    const double hnsw_qps = (static_cast<double>(num_queries) / hnsw_total_ms) * 1000.0;
    const double recall = static_cast<double>(total_matches) / static_cast<double>(num_queries * k);
    const double avg_visited = static_cast<double>(total_visited) / static_cast<double>(num_queries);

    const double speedup = bf_avg_lat_us / hnsw_avg_lat_us;
    const std::string verdict = (speedup > 1.0) ? "HNSW is FASTER" : "HNSW is SLOWER";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  --> Brute-Force : " << bf_avg_lat_us << " us (" << bf_qps << " QPS) | " << dataset_size << " nodes examined" << std::endl;
    std::cout << "  --> HNSW        : " << hnsw_avg_lat_us << " us (" << hnsw_qps << " QPS) | " << avg_visited << " nodes visited (" << ((avg_visited / static_cast<double>(dataset_size)) * 100.0) << "% of index)" << std::endl;
    std::cout << "  --> Recall@10   : " << (recall * 100.0) << " %" << std::endl;
    std::cout << "  --> Speedup     : " << speedup << "x (" << verdict << ")" << std::endl;

    return ScaleResult{
        .dataset_size = dataset_size,
        .bf_avg_latency_us = bf_avg_lat_us,
        .bf_qps = bf_qps,
        .bf_nodes_examined = dataset_size,
        .hnsw_avg_latency_us = hnsw_avg_lat_us,
        .hnsw_qps = hnsw_qps,
        .hnsw_nodes_visited = avg_visited,
        .hnsw_recall_at_10 = recall,
        .speedup = speedup,
        .comparison_verdict = verdict,
        .index_build_time_ms = build_time_ms
    };
}

} // anonymous namespace

int main() {
    std::cout << "==========================================================================" << std::endl;
    std::cout << " VectorForge — Phase 2 HNSW Scale-Validation Benchmark                    " << std::endl;
    std::cout << " Evaluating Scaling Behavior Across 1K, 10K, and 50K Vectors (128 dims)   " << std::endl;
    std::cout << "==========================================================================\n" << std::endl;

    constexpr std::size_t DIMENSION = 128;
    constexpr std::size_t NUM_QUERIES = 50;
    constexpr std::size_t K = 10;
    constexpr uint32_t QUERY_SEED = 9999;

    vectorforge::HNSWConfig config{
        .M = 16,
        .ef_construction = 100,
        .ef_search = 50,
        .random_seed = 42
    };

    std::cout << "Fixed Configuration:" << std::endl;
    std::cout << "  - Dimensionality   : " << DIMENSION << " floats" << std::endl;
    std::cout << "  - Queries          : " << NUM_QUERIES << " deterministic queries" << std::endl;
    std::cout << "  - k (Neighbors)    : " << K << std::endl;
    std::cout << "  - HNSW M           : " << config.M << std::endl;
    std::cout << "  - HNSW efConstruct : " << config.ef_construction << std::endl;
    std::cout << "  - HNSW efSearch    : " << config.ef_search << std::endl;

    std::cout << "\nGenerating " << NUM_QUERIES << " deterministic query vectors..." << std::endl;
    const auto queries = generate_deterministic_vectors(NUM_QUERIES, DIMENSION, QUERY_SEED);

    std::vector<ScaleResult> results;
    results.push_back(run_scale_evaluation(1000, DIMENSION, queries, K, config, 101));
    results.push_back(run_scale_evaluation(10000, DIMENSION, queries, K, config, 202));
    results.push_back(run_scale_evaluation(50000, DIMENSION, queries, K, config, 303));

    // Summary Comparison Table
    std::cout << "\n\n=========================================================================================================================" << std::endl;
    std::cout << "                                  FINAL MEASURED SCALE COMPARISON TABLE                                                  " << std::endl;
    std::cout << "=========================================================================================================================" << std::endl;
    std::cout << "| Dataset Size | BF Latency (avg) | HNSW Latency (avg) | BF Nodes Examined | HNSW Nodes Visited | Recall@10 | Speedup | Verdict          |" << std::endl;
    std::cout << "|:------------:|:----------------:|:------------------:|:-----------------:|:------------------:|:---------:|:-------:|:-----------------|" << std::endl;

    std::cout << std::fixed << std::setprecision(2);
    for (const auto& r : results) {
        std::cout << "| " << std::setw(12) << r.dataset_size
                  << " | " << std::setw(13) << r.bf_avg_latency_us << " us"
                  << " | " << std::setw(15) << r.hnsw_avg_latency_us << " us"
                  << " | " << std::setw(17) << r.bf_nodes_examined
                  << " | " << std::setw(18) << r.hnsw_nodes_visited
                  << " | " << std::setw(8) << (r.hnsw_recall_at_10 * 100.0) << "%"
                  << " | " << std::setw(6) << r.speedup << "x"
                  << " | " << std::setw(16) << r.comparison_verdict
                  << " |" << std::endl;
    }
    std::cout << "=========================================================================================================================\n" << std::endl;

    std::cout << "Scaling Analysis & Observations:" << std::endl;
    for (const auto& r : results) {
        const double visited_pct = (r.hnsw_nodes_visited / static_cast<double>(r.dataset_size)) * 100.0;
        std::cout << "  - N = " << std::setw(5) << r.dataset_size << ": "
                  << "HNSW visits only " << r.hnsw_nodes_visited << " nodes (" << visited_pct << "% of dataset) "
                  << "vs " << r.bf_nodes_examined << " in Brute-Force. "
                  << "Speedup = " << r.speedup << "x -> " << r.comparison_verdict << "."
                  << std::endl;
    }
    std::cout << "\n=========================================================================================================================" << std::endl;

    return 0;
}
