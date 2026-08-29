import sys
import os
import numpy as np

# Ensure server package path is recognized
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import VectorForgeIndex, simd_info

def main():
    print("=" * 70)
    print(" VectorForge — Python Binding Smoke Test (pybind11)                  ")
    print("=" * 70)

    # 1. Hardware SIMD capabilities
    info = simd_info()
    print("\n[SIMD Capabilities]")
    print(f"  CPU Model Brand    : {info['cpu_brand']}")
    print(f"  AVX Supported      : {'YES' if info['avx'] else 'NO'}")
    print(f"  AVX2 Supported     : {'YES' if info['avx2'] else 'NO'}")
    print(f"  FMA Supported      : {'YES' if info['fma'] else 'NO'}")
    print(f"  AVX-512F Supported : {'YES' if info['avx512f'] else 'NO'}")
    print(f"  AVX-512DQ Supported: {'YES' if info['avx512dq'] else 'NO'}")
    print(f"  Active Dispatch    : {info['active_dispatch']}")

    # 2. Create HNSW Index (D=4)
    D = 4
    index = VectorForgeIndex(dimension=D, backend="hnsw", m=16, ef_construction=100, ef_search=50)
    print(f"\n[Index Initialized]")
    print(f"  Backend   : {index.backend()}")
    print(f"  Dimension : {index.dimension()}")
    print(f"  Size      : {index.size()} vectors")

    # 3. Deterministic test dataset (5 vectors x 4 dims)
    dataset = np.array([
        [0.0, 0.0, 0.0, 0.0],  # ID 0
        [1.0, 0.0, 0.0, 0.0],  # ID 1
        [0.0, 2.0, 0.0, 0.0],  # ID 2
        [3.0, 0.0, 0.0, 0.0],  # ID 3
        [0.0, 0.0, 4.0, 0.0],  # ID 4
    ], dtype=np.float32)

    count = index.insert(dataset)
    print(f"\n[Insertion]")
    print(f"  Vectors Inserted : {count}")
    print(f"  Current Size     : {index.size()}")

    # 4. Search Query: [0.1, 0.0, 0.0, 0.0] -> closest to ID 0, then ID 1
    query = np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    k = 3
    results = index.search(query, k=k)

    print(f"\n[Query Results for [0.1, 0.0, 0.0, 0.0], k={k}]")
    for rank, r in enumerate(results, start=1):
        print(f"  Rank {rank}: ID = {r['id']}, Distance = {r['distance']:.6f}")

    # 5. Telemetry
    tele = index.get_telemetry()
    print(f"\n[Telemetry]")
    print(f"  Total Indexed  : {tele['size']}")
    print(f"  Max Level      : {tele['max_level']}")
    print(f"  Entry Point ID : {tele['entry_point']}")
    print(f"  efSearch       : {tele['ef_search']}")

    # 6. Parameter Tuning
    tuned = index.tune(ef_search=64)
    print(f"\n[Tuned efSearch]")
    print(f"  New efSearch   : {tuned['ef_search']}")

    print("\n" + "=" * 70)
    print(" Python Binding Smoke Test Passed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
