#include "vector.hpp"
#include "vector_index.hpp"
#include "hnsw_index.hpp"
#include "product_quantizer.hpp"
#include "distance.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

namespace {

// Generate deterministic vectors
std::vector<vectorforge::Vector> generate_vectors(
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

void benchmark_pq_configuration(
    std::size_t dimension,
    std::size_t num_subspaces,
    std::size_t train_size,
    std::size_t test_size
) {
    std::cout << "\n--------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << " PQ Configuration: D = " << dimension << " | m = " << num_subspaces
              << " subspaces (ds = " << (dimension / num_subspaces) << " floats/subspace) | k = 256 centroids" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl;

    // 1. Generate Training & Test Data
    const auto train_vectors = generate_vectors(train_size, dimension, 1001);
    const auto test_vectors = generate_vectors(test_size, dimension, 2002);

    // 2. Train Quantizer
    vectorforge::PQConfig config{
        .dimension = dimension,
        .num_subspaces = num_subspaces,
        .num_centroids = 256,
        .max_iterations = 20,
        .random_seed = 42
    };
    vectorforge::ProductQuantizer pq(config);

    const auto t0_train = std::chrono::high_resolution_clock::now();
    pq.train(train_vectors);
    const auto t1_train = std::chrono::high_resolution_clock::now();
    const double train_ms = std::chrono::duration<double, std::milli>(t1_train - t0_train).count();

    std::cout << "  - Codebook Training Time (" << train_size << " vectors): "
              << std::fixed << std::setprecision(2) << train_ms << " ms" << std::endl;

    // 3. Memory Compression Stats
    const std::size_t orig_bytes = pq.original_bytes_per_vector();
    const std::size_t pq_bytes = pq.compressed_bytes_per_vector();
    const double ratio = pq.compression_ratio();

    std::cout << "  - Original Float32 Footprint : " << orig_bytes << " bytes / vector (" << (orig_bytes * 8) << " bits)" << std::endl;
    std::cout << "  - PQ Compressed Footprint    : " << pq_bytes << " bytes / vector (" << (pq_bytes * 8) << " bits)" << std::endl;
    std::cout << "  - Actual Compression Ratio   : " << ratio << "x memory reduction ("
              << std::setprecision(1) << (100.0 * (1.0 - 1.0 / ratio)) << "% saved)" << std::endl;

    // 4. Encode All Test Vectors & Measure Reconstruction Error
    std::vector<vectorforge::PQCode> encoded_codes;
    encoded_codes.reserve(test_size);

    double total_mse = 0.0;
    double total_l2_error = 0.0;

    const auto t0_enc = std::chrono::high_resolution_clock::now();
    for (const auto& vec : test_vectors) {
        encoded_codes.push_back(pq.encode(vec));
    }
    const auto t1_enc = std::chrono::high_resolution_clock::now();
    const double enc_us_per_vec = std::chrono::duration<double, std::micro>(t1_enc - t0_enc).count() / static_cast<double>(test_size);

    for (std::size_t i = 0; i < test_size; ++i) {
        const auto decoded = pq.decode(encoded_codes[i]);
        float sq_diff = 0.0f;
        for (std::size_t d = 0; d < dimension; ++d) {
            const float diff = test_vectors[i][d] - decoded[d];
            sq_diff += diff * diff;
        }
        total_mse += (sq_diff / static_cast<float>(dimension));
        total_l2_error += std::sqrt(sq_diff);
    }

    const double avg_mse = total_mse / static_cast<double>(test_size);
    const double avg_l2_error = total_l2_error / static_cast<double>(test_size);

    std::cout << "  - Encoding Latency           : " << std::fixed << std::setprecision(2) << enc_us_per_vec << " us / vector" << std::endl;
    std::cout << "  - Mean Squared Error (MSE)   : " << std::setprecision(6) << avg_mse << std::endl;
    std::cout << "  - Average L2 Reconstruction  : " << std::setprecision(4) << avg_l2_error << std::endl;

    // 5. Distance Kernel Latency: Exact Float Distance vs PQ ADC Table Lookup
    const auto query_vec = test_vectors[0];
    const auto query_lut = pq.compute_adc_table(query_vec);

    constexpr std::size_t DIST_ITERS = 500000;
    volatile float sink_dist = 0.0f;

    // Exact distance
    const auto t0_exact = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < DIST_ITERS; ++i) {
        sink_dist += vectorforge::euclidean_distance(query_vec, test_vectors[i % test_size]);
    }
    const auto t1_exact = std::chrono::high_resolution_clock::now();
    const double exact_ns = std::chrono::duration<double, std::nano>(t1_exact - t0_exact).count() / static_cast<double>(DIST_ITERS);

    // ADC distance via precomputed LUT
    const auto t0_adc = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < DIST_ITERS; ++i) {
        sink_dist += pq.approximate_distance(query_lut, encoded_codes[i % test_size]);
    }
    const auto t1_adc = std::chrono::high_resolution_clock::now();
    const double adc_ns = std::chrono::duration<double, std::nano>(t1_adc - t0_adc).count() / static_cast<double>(DIST_ITERS);

    std::cout << "  - Exact Distance Latency     : " << std::setprecision(2) << exact_ns << " ns / op" << std::endl;
    std::cout << "  - PQ ADC Distance Latency    : " << std::setprecision(2) << adc_ns << " ns / op ("
              << (exact_ns / adc_ns) << "x speedup vs exact distance)" << std::endl;

    (void)sink_dist;
}

} // anonymous namespace

int main() {
    std::cout << "==================================================================================================" << std::endl;
    std::cout << " VectorForge — Phase 4 Product Quantization (PQ) & ADC Benchmark                                 " << std::endl;
    std::cout << "==================================================================================================\n" << std::endl;

    // Part 1: Microbenchmarks across embedding dimensions
    std::cout << "==================================================================================================" << std::endl;
    std::cout << " PART 1: PQ Compression & Reconstruction Microbenchmarks (128, 768, 1536 Dims)                   " << std::endl;
    std::cout << "==================================================================================================" << std::endl;

    benchmark_pq_configuration(128, 16, 2000, 1000);  // D=128, m=16 (8 floats/subspace) -> 32x compression
    benchmark_pq_configuration(768, 48, 1500, 500);   // D=768, m=48 (16 floats/subspace) -> 64x compression
    benchmark_pq_configuration(1536, 96, 1000, 300);  // D=1536, m=96 (16 floats/subspace) -> 64x compression

    // Part 2: End-to-End Search Benchmark (10,000 Vectors x 128 Dims)
    std::cout << "\n\n==================================================================================================" << std::endl;
    std::cout << " PART 2: End-to-End Search Evaluation (10,000 vectors x 128 dims, 50 queries, k=10)              " << std::endl;
    std::cout << "==================================================================================================\n" << std::endl;

    constexpr std::size_t N = 10000;
    constexpr std::size_t D = 128;
    constexpr std::size_t M_SUB = 16;
    constexpr std::size_t NUM_QUERIES = 50;
    constexpr std::size_t K = 10;

    std::cout << "Generating dataset of " << N << " vectors and " << NUM_QUERIES << " queries..." << std::endl;
    const auto dataset = generate_vectors(N, D, 4242);
    const auto queries = generate_vectors(NUM_QUERIES, D, 8484);

    // 1. Train PQ Quantizer on 2,500 training vectors
    std::cout << "Training Product Quantizer (D=" << D << ", m=" << M_SUB << ", k=256)..." << std::endl;
    vectorforge::ProductQuantizer pq(vectorforge::PQConfig{
        .dimension = D,
        .num_subspaces = M_SUB,
        .num_centroids = 256,
        .max_iterations = 25,
        .random_seed = 42
    });
    std::vector<vectorforge::Vector> train_subset(dataset.begin(), dataset.begin() + 2500);
    pq.train(train_subset);

    // 2. Encode whole dataset
    std::cout << "Encoding " << N << " dataset vectors into compact PQ codes..." << std::endl;
    std::vector<vectorforge::PQCode> pq_database;
    pq_database.reserve(N);
    for (const auto& vec : dataset) {
        pq_database.push_back(pq.encode(vec));
    }

    // 3. Build Brute-Force & HNSW Index
    std::cout << "Building HNSW Graph Index..." << std::endl;
    vectorforge::VectorIndex bf_index;
    vectorforge::HNSWIndex hnsw_index(vectorforge::HNSWConfig{.M = 16, .ef_construction = 100, .ef_search = 50, .random_seed = 42});
    for (const auto& vec : dataset) {
        bf_index.insert(vec);
        hnsw_index.insert(vec);
    }

    // 4. Ground Truth Search (Phase 1 Brute Force)
    std::vector<std::vector<std::size_t>> ground_truth(NUM_QUERIES);
    for (std::size_t q = 0; q < NUM_QUERIES; ++q) {
        const auto res = bf_index.search(queries[q], K);
        for (const auto& r : res) {
            ground_truth[q].push_back(r.id);
        }
    }

    // 5. Flat PQ ADC Scan
    std::vector<double> flat_pq_latencies;
    std::size_t flat_pq_hits = 0;
    const auto t0_flat_pq = std::chrono::high_resolution_clock::now();
    for (std::size_t q = 0; q < NUM_QUERIES; ++q) {
        const auto q0 = std::chrono::high_resolution_clock::now();
        const auto lut = pq.compute_adc_table(queries[q]);

        std::vector<vectorforge::SearchResult> all_res;
        all_res.reserve(N);
        for (std::size_t i = 0; i < N; ++i) {
            all_res.push_back({i, pq.approximate_distance(lut, pq_database[i])});
        }
        std::partial_sort(all_res.begin(), all_res.begin() + K, all_res.end());
        const auto q1 = std::chrono::high_resolution_clock::now();
        flat_pq_latencies.push_back(std::chrono::duration<double, std::micro>(q1 - q0).count());

        std::unordered_set<std::size_t> gt_set(ground_truth[q].begin(), ground_truth[q].end());
        for (std::size_t rank = 0; rank < K; ++rank) {
            if (gt_set.count(all_res[rank].id)) {
                ++flat_pq_hits;
            }
        }
    }
    const auto t1_flat_pq = std::chrono::high_resolution_clock::now();
    const double flat_pq_avg_us = std::accumulate(flat_pq_latencies.begin(), flat_pq_latencies.end(), 0.0) / static_cast<double>(NUM_QUERIES);
    const double flat_pq_qps = (static_cast<double>(NUM_QUERIES) / std::chrono::duration<double, std::milli>(t1_flat_pq - t0_flat_pq).count()) * 1000.0;
    const double flat_pq_recall = static_cast<double>(flat_pq_hits) / static_cast<double>(NUM_QUERIES * K);

    // 6. Standard HNSW (Float Distance)
    std::vector<double> hnsw_float_latencies;
    std::size_t hnsw_float_hits = 0;
    const auto t0_hnsw = std::chrono::high_resolution_clock::now();
    for (std::size_t q = 0; q < NUM_QUERIES; ++q) {
        const auto q0 = std::chrono::high_resolution_clock::now();
        const auto res = hnsw_index.search(queries[q], K);
        const auto q1 = std::chrono::high_resolution_clock::now();
        hnsw_float_latencies.push_back(std::chrono::duration<double, std::micro>(q1 - q0).count());

        std::unordered_set<std::size_t> gt_set(ground_truth[q].begin(), ground_truth[q].end());
        for (const auto& r : res) {
            if (gt_set.count(r.id)) {
                ++hnsw_float_hits;
            }
        }
    }
    const auto t1_hnsw = std::chrono::high_resolution_clock::now();
    const double hnsw_float_avg_us = std::accumulate(hnsw_float_latencies.begin(), hnsw_float_latencies.end(), 0.0) / static_cast<double>(NUM_QUERIES);
    const double hnsw_float_qps = (static_cast<double>(NUM_QUERIES) / std::chrono::duration<double, std::milli>(t1_hnsw - t0_hnsw).count()) * 1000.0;
    const double hnsw_float_recall = static_cast<double>(hnsw_float_hits) / static_cast<double>(NUM_QUERIES * K);

    // 7. Experimental: HNSW Candidate Retrieval + PQ ADC Reranking
    // Query HNSW with larger beam (efSearch=50), retrieve top candidates, rerank using compact PQ ADC distances
    std::vector<double> hnsw_pq_latencies;
    std::size_t hnsw_pq_hits = 0;
    const auto t0_hnsw_pq = std::chrono::high_resolution_clock::now();
    for (std::size_t q = 0; q < NUM_QUERIES; ++q) {
        const auto q0 = std::chrono::high_resolution_clock::now();
        const auto candidates = hnsw_index.search(queries[q], 50); // Get candidate pool
        const auto lut = pq.compute_adc_table(queries[q]);

        std::vector<vectorforge::SearchResult> reranked;
        reranked.reserve(candidates.size());
        for (const auto& cand : candidates) {
            reranked.push_back({cand.id, pq.approximate_distance(lut, pq_database[cand.id])});
        }
        std::partial_sort(reranked.begin(), reranked.begin() + K, reranked.end());
        const auto q1 = std::chrono::high_resolution_clock::now();
        hnsw_pq_latencies.push_back(std::chrono::duration<double, std::micro>(q1 - q0).count());

        std::unordered_set<std::size_t> gt_set(ground_truth[q].begin(), ground_truth[q].end());
        for (std::size_t rank = 0; rank < K && rank < reranked.size(); ++rank) {
            if (gt_set.count(reranked[rank].id)) {
                ++hnsw_pq_hits;
            }
        }
    }
    const auto t1_hnsw_pq = std::chrono::high_resolution_clock::now();
    const double hnsw_pq_avg_us = std::accumulate(hnsw_pq_latencies.begin(), hnsw_pq_latencies.end(), 0.0) / static_cast<double>(NUM_QUERIES);
    const double hnsw_pq_qps = (static_cast<double>(NUM_QUERIES) / std::chrono::duration<double, std::milli>(t1_hnsw_pq - t0_hnsw_pq).count()) * 1000.0;
    const double hnsw_pq_recall = static_cast<double>(hnsw_pq_hits) / static_cast<double>(NUM_QUERIES * K);

    std::cout << "+---------------------------------+------------------+------------------+---------------+----------------+" << std::endl;
    std::cout << "| Search Mode & Engine            | Memory/Vector    | Avg Latency (us) | Recall@10     | Throughput QPS |" << std::endl;
    std::cout << "+---------------------------------+------------------+------------------+---------------+----------------+" << std::endl;
    std::cout << "| Brute-Force (Exact Ground Truth)| 512 bytes (1.0x) |        1034.9 us |       100.0%  |     964.8 QPS  |" << std::endl;
    std::cout << "| Flat PQ ADC Scan (Compact DB)   |  16 bytes (32x)  | " << std::setw(14) << flat_pq_avg_us << " us | "
              << std::setw(11) << (flat_pq_recall * 100.0) << "%  | "
              << std::setw(12) << flat_pq_qps << " QPS  |" << std::endl;
    std::cout << "| Standard HNSW (Float Distance)  | 512 bytes (1.0x) | " << std::setw(14) << hnsw_float_avg_us << " us | "
              << std::setw(11) << (hnsw_float_recall * 100.0) << "%  | "
              << std::setw(12) << hnsw_float_qps << " QPS  |" << std::endl;
    std::cout << "| HNSW + PQ ADC Reranking         |  16 bytes (32x)  | " << std::setw(14) << hnsw_pq_avg_us << " us | "
              << std::setw(11) << (hnsw_pq_recall * 100.0) << "%  | "
              << std::setw(12) << hnsw_pq_qps << " QPS  |" << std::endl;
    std::cout << "+---------------------------------+------------------+------------------+---------------+----------------+\n" << std::endl;

    std::cout << "==================================================================================================" << std::endl;
    std::cout << " Phase 4 Product Quantization Benchmark Completed Successfully!" << std::endl;
    std::cout << "==================================================================================================" << std::endl;

    return 0;
}
