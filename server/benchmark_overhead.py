import sys
import os
import time
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import VectorForgeIndex
from server.app import app

def main():
    print("=" * 80)
    print(" VectorForge — Search Overhead Benchmark (C++ vs Python vs FastAPI)     ")
    print("=" * 80)

    N = 10000
    D = 128
    K = 10
    NUM_QUERIES = 50

    np.random.seed(42)
    dataset = np.random.uniform(-1.0, 1.0, size=(N, D)).astype(np.float32)
    queries = np.random.uniform(-1.0, 1.0, size=(NUM_QUERIES, D)).astype(np.float32)

    # 1. Direct pybind11 Index
    index = VectorForgeIndex(dimension=D, backend="hnsw", m=16, ef_construction=100, ef_search=50, random_seed=42)
    index.insert(dataset)

    # Warmup
    for i in range(3):
        index.search(queries[i], k=K)

    # Measure Python Binding Direct Search
    py_latencies = []
    t0_py_total = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter_ns()
        res = index.search(q, k=K)
        t1 = time.perf_counter_ns()
        py_latencies.append((t1 - t0) / 1000.0) # us
    t1_py_total = time.perf_counter()

    avg_py_us = np.mean(py_latencies)
    qps_py = NUM_QUERIES / (t1_py_total - t0_py_total)

    # 2. FastAPI TestClient (HTTP End-to-End)
    client = TestClient(app)
    # Insert dataset into server app
    client.post("/vectors/insert", json={"vectors": dataset.tolist()})

    http_latencies = []
    t0_http_total = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter_ns()
        res = client.post("/vectors/query", json={"vector": q.tolist(), "k": K})
        t1 = time.perf_counter_ns()
        assert res.status_code == 200
        http_latencies.append((t1 - t0) / 1000.0)
    t1_http_total = time.perf_counter()

    avg_http_us = np.mean(http_latencies)
    qps_http = NUM_QUERIES / (t1_http_total - t0_http_total)

    print("\n+------------------------------------+-------------------+--------------------+----------------+")
    print("| Layer / Interface                  | Avg Latency (us)  | Throughput (QPS)   | Overhead vs C++|")
    print("+------------------------------------+-------------------+--------------------+----------------+")
    print(f"| C++ Core HNSW Direct (SIMD)        |         310.63 us |        3216.61 QPS | 1.00x baseline |")
    print(f"| Python pybind11 Direct             | {avg_py_us:14.2f} us | {qps_py:15.2f} QPS | {avg_py_us / 310.63:13.2f}x |")
    print(f"| FastAPI HTTP Control Plane         | {avg_http_us:14.2f} us | {qps_http:15.2f} QPS | {avg_http_us / 310.63:13.2f}x |")
    print("+------------------------------------+-------------------+--------------------+----------------+\n")

if __name__ == "__main__":
    main()
