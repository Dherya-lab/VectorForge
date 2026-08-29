import sys
import os
import time
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import VectorForgeIndex
from server.app import app
from server.telemetry import TelemetryCollector
from server.drift import DriftDetector

def main():
    print("=" * 80)
    print(" VectorForge — Search & Telemetry Overhead Benchmark ")
    print("=" * 80)

    N = 10000
    D = 128
    K = 10
    NUM_QUERIES = 100

    np.random.seed(42)
    dataset = np.random.uniform(-1.0, 1.0, size=(N, D)).astype(np.float32)
    queries = np.random.uniform(-1.0, 1.0, size=(NUM_QUERIES, D)).astype(np.float32)

    # 1. Direct pybind11 Index (Pure Core C++)
    index = VectorForgeIndex(dimension=D, backend="hnsw", m=16, ef_construction=100, ef_search=50, random_seed=42)
    index.insert(dataset)

    # Warmup
    for i in range(5):
        index.search(queries[i], k=K)

    # Measure Python Binding Direct Search (Without Telemetry)
    py_latencies_raw = []
    t0_py_total = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter_ns()
        res = index.search(q, k=K)
        t1 = time.perf_counter_ns()
        py_latencies_raw.append((t1 - t0) / 1000.0) # us
    t1_py_total = time.perf_counter()

    avg_raw_us = float(np.mean(py_latencies_raw))
    qps_raw = NUM_QUERIES / (t1_py_total - t0_py_total)

    # Measure Python Binding Search (With Telemetry & Drift Hook)
    collector = TelemetryCollector()
    drift_det = DriftDetector()
    drift_det.set_baseline(dataset[:500])

    py_latencies_telemetry = []
    t0_tele_total = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter_ns()
        res = index.search(q, k=K)
        t1 = time.perf_counter_ns()
        latency_us = (t1 - t0) / 1000.0
        latency_ms = latency_us / 1000.0
        drift_det.add_samples(q)
        collector.record_query(
            latency_ms=latency_ms,
            k=K,
            backend="hnsw",
            dimension=D,
            result_count=len(res),
            success=True
        )
        t2 = time.perf_counter_ns()
        py_latencies_telemetry.append((t2 - t0) / 1000.0)
    t1_tele_total = time.perf_counter()

    avg_tele_us = float(np.mean(py_latencies_telemetry))
    qps_tele = NUM_QUERIES / (t1_tele_total - t0_tele_total)
    overhead_pct = ((avg_tele_us - avg_raw_us) / avg_raw_us) * 100.0

    # 2. FastAPI TestClient (HTTP End-to-End with Telemetry + Drift)
    client = TestClient(app)
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

    avg_http_us = float(np.mean(http_latencies))
    qps_http = NUM_QUERIES / (t1_http_total - t0_http_total)

    print("\n+------------------------------------+-------------------+--------------------+----------------+")
    print("| Layer / Configuration              | Avg Latency (us)  | Throughput (QPS)   | Overhead       |")
    print("+------------------------------------+-------------------+--------------------+----------------+")
    print(f"| Python Direct (No Telemetry)       | {avg_raw_us:14.2f} us | {qps_raw:15.2f} QPS | Baseline (0.0%)|")
    print(f"| Python Direct (+ Telemetry & Drift)| {avg_tele_us:14.2f} us | {qps_tele:15.2f} QPS | +{overhead_pct:6.2f}%       |")
    print(f"| FastAPI HTTP (+ Full Telemetry)    | {avg_http_us:14.2f} us | {qps_http:15.2f} QPS | {avg_http_us / avg_raw_us:13.2f}x |")
    print("+------------------------------------+-------------------+--------------------+----------------+\n")

    print(f"Telemetry Latency Overhead: {avg_tele_us - avg_raw_us:.2f} us per query ({overhead_pct:.2f}% relative overhead).")

if __name__ == "__main__":
    main()
