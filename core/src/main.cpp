#include "vector.hpp"
#include "vector_index.hpp"
#include "hnsw_index.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << " VectorForge — Phase 2 Demonstration                      " << std::endl;
    std::cout << " HNSW Approximate Nearest Neighbor Search vs Brute-Force " << std::endl;
    std::cout << "==========================================================\n" << std::endl;

    // 1. Dataset definition
    const std::vector<vectorforge::Vector> dataset = {
        vectorforge::Vector{1.0f, 2.0f},   // ID 0
        vectorforge::Vector{2.0f, 3.0f},   // ID 1
        vectorforge::Vector{10.0f, 10.0f}, // ID 2
        vectorforge::Vector{1.0f, 1.0f},   // ID 3
        vectorforge::Vector{5.0f, 5.0f},   // ID 4
        vectorforge::Vector{2.0f, 2.0f},   // ID 5
        vectorforge::Vector{8.0f, 8.0f},   // ID 6
        vectorforge::Vector{1.5f, 2.5f}    // ID 7
    };

    // 2. Build Brute-Force Index (Ground Truth)
    vectorforge::VectorIndex bf_index;
    for (const auto& vec : dataset) {
        bf_index.insert(vec);
    }

    // 3. Build HNSW Index
    vectorforge::HNSWConfig config{
        .M = 4,
        .ef_construction = 16,
        .ef_search = 16,
        .random_seed = 42
    };
    vectorforge::HNSWIndex hnsw_index(config);
    for (const auto& vec : dataset) {
        hnsw_index.insert(vec);
    }

    std::cout << "Dataset Summary:" << std::endl;
    std::cout << "  - Total Vectors       : " << dataset.size() << std::endl;
    std::cout << "  - Dimensionality      : " << bf_index.dimension() << std::endl;
    std::cout << "  - HNSW Max Level      : " << hnsw_index.max_level() << std::endl;
    std::cout << "  - HNSW Entry Point ID : " << (hnsw_index.entry_point().has_value() ? std::to_string(*hnsw_index.entry_point()) : "None") << "\n" << std::endl;

    // 4. Query vector
    const vectorforge::Vector query{1.0f, 2.0f};
    const std::size_t k = 3;

    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "Query Vector : " << query << std::endl;
    std::cout << "k (Neighbors): " << k << std::endl;
    std::cout << "----------------------------------------------------------\n" << std::endl;

    // Search Brute Force
    const auto bf_results = bf_index.search(query, k);

    // Search HNSW with telemetry
    vectorforge::HNSWStats hnsw_stats;
    const auto hnsw_results = hnsw_index.search_with_stats(query, k, hnsw_stats);

    // 5. Compare Results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== [Ground Truth] Phase 1 Brute-Force Search ===" << std::endl;
    for (std::size_t rank = 0; rank < bf_results.size(); ++rank) {
        const auto& res = bf_results[rank];
        std::cout << "  Rank " << (rank + 1) << ": ID = " << res.id
                  << ", Distance = " << res.distance
                  << ", Vector = " << bf_index.get(res.id)
                  << std::endl;
    }
    std::cout << "  Vectors examined (Brute-Force): " << bf_index.size() << "\n" << std::endl;

    std::cout << "=== [Approximate] Phase 2 HNSW Graph Search ===" << std::endl;
    for (std::size_t rank = 0; rank < hnsw_results.size(); ++rank) {
        const auto& res = hnsw_results[rank];
        std::cout << "  Rank " << (rank + 1) << ": ID = " << res.id
                  << ", Distance = " << res.distance
                  << ", Vector = " << hnsw_index.get(res.id)
                  << std::endl;
    }
    std::cout << "  Graph Nodes Visited: " << hnsw_stats.nodes_visited << std::endl;
    std::cout << "  Distance Computations: " << hnsw_stats.distance_computations << "\n" << std::endl;

    std::cout << "==========================================================" << std::endl;
    std::cout << " Phase 2 Demonstration Completed Successfully!" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
