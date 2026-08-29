#include "vector.hpp"
#include "vector_index.hpp"
#include "hnsw_index.hpp"
#include "simd_math.hpp"
#include "distance.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

// Deterministic float generator
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

// Microbenchmark for a specific dimension
void benchmark_dimension(std::size_t dimension, std::size_t iterations = 500000) {
    std::cout << "\n--------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << " Microbenchmark: Dimension D = " << dimension << " floats (" << (dimension * sizeof(float)) << " bytes/vector) | " << iterations << " iterations" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl;

    const auto vec_a = generate_vectors(1, dimension, 101)[0];
    const auto vec_b = generate_vectors(1, dimension, 202)[0];

    const auto span_a = std::span<const float>(vec_a.data(), vec_a.dimension());
    const auto span_b = std::span<const float>(vec_b.data(), vec_b.dimension());

    const auto& cpu = vectorforge::simd::get_cpu_features();

    // 1. Scalar Distance
    volatile float sink_dist = 0.0f;
    const auto t0_scalar = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        sink_dist += vectorforge::simd::distance_scalar(span_a, span_b);
    }
    const auto t1_scalar = std::chrono::high_resolution_clock::now();
    const double scalar_ns = std::chrono::duration<double, std::nano>(t1_scalar - t0_scalar).count() / static_cast<double>(iterations);
    const double scalar_mops = (static_cast<double>(iterations) / std::chrono::duration<double>(t1_scalar - t0_scalar).count()) / 1e6;

    // 2. AVX2 Distance
    double avx2_ns = 0.0;
    double avx2_mops = 0.0;
    double avx2_speedup = 0.0;
    if (cpu.supports_avx2()) {
        const auto t0_avx2 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < iterations; ++i) {
            sink_dist += vectorforge::simd::distance_avx2(span_a, span_b);
        }
        const auto t1_avx2 = std::chrono::high_resolution_clock::now();
        avx2_ns = std::chrono::duration<double, std::nano>(t1_avx2 - t0_avx2).count() / static_cast<double>(iterations);
        avx2_mops = (static_cast<double>(iterations) / std::chrono::duration<double>(t1_avx2 - t0_avx2).count()) / 1e6;
        avx2_speedup = scalar_ns / avx2_ns;
    }

    // 3. AVX-512 Distance
    double avx512_ns = 0.0;
    double avx512_mops = 0.0;
    double avx512_speedup = 0.0;
    if (cpu.supports_avx512()) {
        const auto t0_avx512 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < iterations; ++i) {
            sink_dist += vectorforge::simd::distance_avx512(span_a, span_b);
        }
        const auto t1_avx512 = std::chrono::high_resolution_clock::now();
        avx512_ns = std::chrono::duration<double, std::nano>(t1_avx512 - t0_avx512).count() / static_cast<double>(iterations);
        avx512_mops = (static_cast<double>(iterations) / std::chrono::duration<double>(t1_avx512 - t0_avx512).count()) / 1e6;
        avx512_speedup = scalar_ns / avx512_ns;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  [Euclidean Distance]" << std::endl;
    std::cout << "    - Scalar  : " << std::setw(8) << scalar_ns << " ns/op (" << std::setw(7) << scalar_mops << " MOps/s) | 1.00x baseline" << std::endl;
    if (cpu.supports_avx2()) {
        std::cout << "    - AVX2    : " << std::setw(8) << avx2_ns << " ns/op (" << std::setw(7) << avx2_mops << " MOps/s) | " << avx2_speedup << "x speedup" << std::endl;
    } else {
        std::cout << "    - AVX2    : unavailable on this CPU" << std::endl;
    }
    if (cpu.supports_avx512()) {
        std::cout << "    - AVX-512 : " << std::setw(8) << avx512_ns << " ns/op (" << std::setw(7) << avx512_mops << " MOps/s) | " << avx512_speedup << "x speedup" << std::endl;
    } else {
        std::cout << "    - AVX-512 : unavailable on this CPU" << std::endl;
    }

    // 4. Cosine Similarity Microbenchmark
    volatile float sink_cos = 0.0f;
    const auto t0_cos_scalar = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        sink_cos += vectorforge::simd::cosine_similarity_scalar(span_a, span_b);
    }
    const auto t1_cos_scalar = std::chrono::high_resolution_clock::now();
    const double cos_scalar_ns = std::chrono::duration<double, std::nano>(t1_cos_scalar - t0_cos_scalar).count() / static_cast<double>(iterations);

    double cos_avx2_ns = 0.0;
    double cos_avx2_speedup = 0.0;
    if (cpu.supports_avx2()) {
        const auto t0_cos_avx2 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < iterations; ++i) {
            sink_cos += vectorforge::simd::cosine_similarity_avx2(span_a, span_b);
        }
        const auto t1_cos_avx2 = std::chrono::high_resolution_clock::now();
        cos_avx2_ns = std::chrono::duration<double, std::nano>(t1_cos_avx2 - t0_cos_avx2).count() / static_cast<double>(iterations);
        cos_avx2_speedup = cos_scalar_ns / cos_avx2_ns;
    }

    std::cout << "  [Cosine Similarity]" << std::endl;
    std::cout << "    - Scalar  : " << std::setw(8) << cos_scalar_ns << " ns/op | 1.00x baseline" << std::endl;
    if (cpu.supports_avx2()) {
        std::cout << "    - AVX2    : " << std::setw(8) << cos_avx2_ns << " ns/op | " << cos_avx2_speedup << "x speedup" << std::endl;
    }
    (void)sink_dist;
    (void)sink_cos;
}

} // anonymous namespace

int main() {
    std::cout << "==================================================================================================" << std::endl;
    std::cout << " VectorForge — Phase 3 SIMD Acceleration Benchmark (AVX2 / AVX-512 / Scalar)                      " << std::endl;
    std::cout << "==================================================================================================\n" << std::endl;

    // 1. Hardware Detection
    const auto& cpu = vectorforge::simd::get_cpu_features();
    std::cout << "Hardware & CPU Capabilities Detected:" << std::endl;
    std::cout << "  - CPU Model Brand : " << cpu.cpu_brand << std::endl;
    std::cout << "  - AVX Supported   : " << (cpu.has_avx ? "YES" : "NO") << std::endl;
    std::cout << "  - AVX2 Supported  : " << (cpu.has_avx2 ? "YES" : "NO") << std::endl;
    std::cout << "  - FMA Supported   : " << (cpu.has_fma ? "YES" : "NO") << std::endl;
    std::cout << "  - AVX-512F Supp.  : " << (cpu.has_avx512f ? "YES" : "NO") << std::endl;
    std::cout << "  - AVX-512DQ Supp. : " << (cpu.has_avx512dq ? "YES" : "NO") << std::endl;
    std::cout << "  - Active Dispatch : " << (cpu.supports_avx512() ? "AVX-512" : (cpu.supports_avx2() ? "AVX2 + FMA" : "Scalar Fallback")) << "\n" << std::endl;

    // 2. SIMD Microbenchmarks
    std::cout << "==================================================================================================" << std::endl;
    std::cout << " PART 1: SIMD Kernel Microbenchmarks Across Standard Embedding Dimensions                        " << std::endl;
    std::cout << "==================================================================================================" << std::endl;

    benchmark_dimension(128, 1000000);   // Standard small embedding (e.g. lightweight models)
    benchmark_dimension(768, 200000);    // BERT / MiniLM embedding size
    benchmark_dimension(1536, 100000);   // OpenAI text-embedding-3-small / Ada-002 size

    // 3. End-to-End Search Benchmark (Scalar vs SIMD)
    std::cout << "\n\n==================================================================================================" << std::endl;
    std::cout << " PART 2: End-to-End Search Benchmark (10,000 vectors x 128 dims, 50 queries, k=10)               " << std::endl;
    std::cout << "==================================================================================================\n" << std::endl;

    constexpr std::size_t N = 10000;
    constexpr std::size_t D = 128;
    constexpr std::size_t NUM_QUERIES = 50;
    constexpr std::size_t K = 10;

    std::cout << "Generating dataset of " << N << " vectors and " << NUM_QUERIES << " queries..." << std::endl;
    const auto dataset = generate_vectors(N, D, 1234);
    const auto queries = generate_vectors(NUM_QUERIES, D, 5678);

    // Build Index
    vectorforge::VectorIndex index;
    vectorforge::HNSWIndex hnsw_index(vectorforge::HNSWConfig{.M = 16, .ef_construction = 100, .ef_search = 50, .random_seed = 42});

    for (const auto& vec : dataset) {
        index.insert(vec);
        hnsw_index.insert(vec);
    }

    // Warm-up
    for (std::size_t i = 0; i < 3; ++i) {
        (void)index.search(queries[i], K);
        (void)hnsw_index.search(queries[i], K);
    }

    // Benchmark Brute-Force (Scalar baseline vs SIMD-dispatched)
    // Brute-force scalar
    std::vector<double> bf_scalar_latencies;
    const auto t0_bf_scalar = std::chrono::high_resolution_clock::now();
    for (std::size_t q = 0; q < NUM_QUERIES; ++q) {
        const auto q0 = std::chrono::high_resolution_clock::now();
        // Manually compute scalar for comparison
        std::vector<vectorforge::SearchResult> all_res;
        all_res.reserve(N);
        for (std::size_t i = 0; i < N; ++i) {
            all_res.push_back({i, vectorforge::simd::distance_scalar(
                std::span<const float>(queries[q].data(), D),
                std::span<const float>(index.get(i).data(), D))});
        }
        std::partial_sort(all_res.begin(), all_res.begin() + K, all_res.end());
        const auto q1 = std::chrono::high_resolution_clock::now();
        bf_scalar_latencies.push_back(std::chrono::duration<double, std::micro>(q1 - q0).count());
    }
    const auto t1_bf_scalar = std::chrono::high_resolution_clock::now();
    const double bf_scalar_avg_us = std::accumulate(bf_scalar_latencies.begin(), bf_scalar_latencies.end(), 0.0) / static_cast<double>(NUM_QUERIES);
    const double bf_scalar_qps = (static_cast<double>(NUM_QUERIES) / std::chrono::duration<double, std::milli>(t1_bf_scalar - t0_bf_scalar).count()) * 1000.0;

    // Brute-force SIMD (calls index.search which uses simd::distance)
    std::vector<double> bf_simd_latencies;
    const auto t0_bf_simd = std::chrono::high_resolution_clock::now();
    for (std::size_t q = 0; q < NUM_QUERIES; ++q) {
        const auto q0 = std::chrono::high_resolution_clock::now();
        const auto res = index.search(queries[q], K);
        const auto q1 = std::chrono::high_resolution_clock::now();
        bf_simd_latencies.push_back(std::chrono::duration<double, std::micro>(q1 - q0).count());
    }
    const auto t1_bf_simd = std::chrono::high_resolution_clock::now();
    const double bf_simd_avg_us = std::accumulate(bf_simd_latencies.begin(), bf_simd_latencies.end(), 0.0) / static_cast<double>(NUM_QUERIES);
    const double bf_simd_qps = (static_cast<double>(NUM_QUERIES) / std::chrono::duration<double, std::milli>(t1_bf_simd - t0_bf_simd).count()) * 1000.0;

    // HNSW SIMD search
    std::vector<double> hnsw_simd_latencies;
    const auto t0_hnsw_simd = std::chrono::high_resolution_clock::now();
    for (std::size_t q = 0; q < NUM_QUERIES; ++q) {
        const auto q0 = std::chrono::high_resolution_clock::now();
        const auto res = hnsw_index.search(queries[q], K);
        const auto q1 = std::chrono::high_resolution_clock::now();
        hnsw_simd_latencies.push_back(std::chrono::duration<double, std::micro>(q1 - q0).count());
    }
    const auto t1_hnsw_simd = std::chrono::high_resolution_clock::now();
    const double hnsw_simd_avg_us = std::accumulate(hnsw_simd_latencies.begin(), hnsw_simd_latencies.end(), 0.0) / static_cast<double>(NUM_QUERIES);
    const double hnsw_simd_qps = (static_cast<double>(NUM_QUERIES) / std::chrono::duration<double, std::milli>(t1_hnsw_simd - t0_hnsw_simd).count()) * 1000.0;

    std::cout << "+------------------------------+-------------------+--------------------+----------------+" << std::endl;
    std::cout << "| Search Algorithm & Engine    | Avg Latency (us)  | Throughput (QPS)   | SIMD Speedup   |" << std::endl;
    std::cout << "+------------------------------+-------------------+--------------------+----------------+" << std::endl;
    std::cout << "| Brute-Force (Scalar Baseline)| " << std::setw(15) << bf_scalar_avg_us << " us | " << std::setw(16) << bf_scalar_qps << " QPS | 1.00x baseline |" << std::endl;
    std::cout << "| Brute-Force (SIMD Dispatch)  | " << std::setw(15) << bf_simd_avg_us << " us | " << std::setw(16) << bf_simd_qps << " QPS | " << std::setw(12) << (bf_scalar_avg_us / bf_simd_avg_us) << "x |" << std::endl;
    std::cout << "| HNSW Index (SIMD Dispatch)   | " << std::setw(15) << hnsw_simd_avg_us << " us | " << std::setw(16) << hnsw_simd_qps << " QPS | " << std::setw(12) << (bf_scalar_avg_us / hnsw_simd_avg_us) << "x |" << std::endl;
    std::cout << "+------------------------------+-------------------+--------------------+----------------+\n" << std::endl;

    std::cout << "==================================================================================================" << std::endl;
    std::cout << " Phase 3 SIMD Benchmark Completed Successfully!" << std::endl;
    std::cout << "==================================================================================================" << std::endl;

    return 0;
}
