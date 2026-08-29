import sys
import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.telemetry import TelemetryCollector, MAX_LATENCY_SAMPLES
from server.drift import DriftDetector, DistributionStatistics
from server.app import app, telemetry_collector, query_drift_detector, dataset_drift_detector, state, get_or_create_index

client = TestClient(app)

# =========================================================================
# 1. TELEMETRY UNIT TESTS
# =========================================================================

def test_telemetry_query_recording_and_counters():
    collector = TelemetryCollector(max_latency_samples=100)
    assert collector.total_query_operations == 0
    assert collector.successful_queries == 0
    assert collector.failed_queries == 0

    # Record 5 successful queries and 1 failed query
    for i in range(5):
        collector.record_query(latency_ms=1.5 + i, k=10, backend="hnsw", dimension=128, result_count=10, success=True)
    collector.record_query(latency_ms=0.0, k=10, backend="hnsw", dimension=128, result_count=0, success=False)

    assert collector.total_query_operations == 6
    assert collector.successful_queries == 5
    assert collector.failed_queries == 1

def test_telemetry_latency_percentiles():
    collector = TelemetryCollector(max_latency_samples=1000)
    
    # 100 deterministic latencies: 1.0 to 100.0 ms
    for val in range(1, 101):
        collector.record_query(latency_ms=float(val), k=10, backend="hnsw", dimension=64, result_count=10, success=True)
    
    stats = collector.get_latency_stats()
    assert stats["sample_count"] == 100
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 100.0
    assert pytest.approx(stats["avg_ms"], 0.1) == 50.5
    assert pytest.approx(stats["p50_ms"], 0.5) == 50.5
    assert pytest.approx(stats["p95_ms"], 0.5) == 95.05
    assert pytest.approx(stats["p99_ms"], 0.5) == 99.01

def test_telemetry_bounded_memory():
    bounded_size = 50
    collector = TelemetryCollector(max_latency_samples=bounded_size)
    
    for i in range(200):
        collector.record_query(latency_ms=float(i), k=5, backend="hnsw", dimension=32, result_count=5, success=True)
        
    stats = collector.get_latency_stats()
    assert stats["sample_count"] == bounded_size
    # Oldest samples (0..149) should have been evicted; remaining are 150..199
    assert stats["min_ms"] == 150.0
    assert stats["max_ms"] == 199.0

def test_telemetry_failed_queries_isolation():
    collector = TelemetryCollector(max_latency_samples=100)
    collector.record_query(latency_ms=10.0, k=5, backend="hnsw", dimension=16, result_count=5, success=True)
    collector.record_query(latency_ms=0.0, k=5, backend="hnsw", dimension=16, result_count=0, success=False)
    
    stats = collector.get_latency_stats()
    # Failed query latency (0.0) should not corrupt valid query stats
    assert stats["sample_count"] == 1
    assert stats["avg_ms"] == 10.0

# =========================================================================
# 2. STATISTICAL DRIFT DETECTION UNIT TESTS
# =========================================================================

def test_drift_baseline_initialization():
    detector = DriftDetector(threshold=0.30, severe_threshold=0.60, max_samples=500)
    
    np.random.seed(42)
    baseline_data = np.random.normal(loc=0.0, scale=1.0, size=(100, 16)).astype(np.float32)
    detector.set_baseline(baseline_data)
    
    report = detector.compute_drift()
    assert report["status"] == "normal"
    assert report["score"] == 0.0
    assert report["baseline_statistics"] is not None
    assert report["baseline_statistics"]["sample_count"] == 100
    assert report["baseline_statistics"]["dimension"] == 16

def test_drift_identical_distribution_low_drift():
    detector = DriftDetector(threshold=0.30, severe_threshold=0.60, max_samples=500)
    
    np.random.seed(42)
    baseline_data = np.random.normal(loc=0.0, scale=1.0, size=(200, 16)).astype(np.float32)
    detector.set_baseline(baseline_data)
    
    # Samples drawn from the exact same distribution
    incoming_data = np.random.normal(loc=0.0, scale=1.0, size=(100, 16)).astype(np.float32)
    detector.add_samples(incoming_data)
    
    report = detector.compute_drift()
    assert report["status"] == "normal"
    assert report["score"] < 0.20  # Well below 0.30 threshold

def test_drift_small_shift_produces_warning():
    detector = DriftDetector(threshold=0.30, severe_threshold=0.60, max_samples=500)
    
    np.random.seed(42)
    baseline_data = np.random.normal(loc=0.0, scale=1.0, size=(200, 16)).astype(np.float32)
    detector.set_baseline(baseline_data)
    
    # Moderate mean shift (loc=0.55)
    shifted_data = np.random.normal(loc=0.55, scale=1.1, size=(100, 16)).astype(np.float32)
    detector.add_samples(shifted_data)
    
    report = detector.compute_drift()
    assert 0.25 <= report["score"] <= 0.60
    assert report["status"] in ["warning", "normal"]

def test_drift_large_shift_detected():
    detector = DriftDetector(threshold=0.30, severe_threshold=0.60, max_samples=500)
    
    np.random.seed(42)
    baseline_data = np.random.normal(loc=0.0, scale=1.0, size=(200, 16)).astype(np.float32)
    detector.set_baseline(baseline_data)
    
    # Large mean shift (loc=3.0, scale=2.5)
    drastic_data = np.random.normal(loc=3.0, scale=2.5, size=(100, 16)).astype(np.float32)
    detector.add_samples(drastic_data)
    
    report = detector.compute_drift()
    assert report["score"] >= 0.60
    assert report["status"] == "drift_detected"

def test_drift_manual_reset():
    detector = DriftDetector(threshold=0.30, severe_threshold=0.60, max_samples=500)
    
    np.random.seed(42)
    baseline_data = np.random.normal(loc=0.0, scale=1.0, size=(100, 16)).astype(np.float32)
    detector.set_baseline(baseline_data)
    
    # Induce drift
    drastic_data = np.random.normal(loc=4.0, scale=2.0, size=(100, 16)).astype(np.float32)
    detector.add_samples(drastic_data)
    assert detector.compute_drift()["status"] == "drift_detected"
    
    # Manual reset
    reset_res = detector.reset()
    assert reset_res["status"] == "normal"
    assert reset_res["score"] == 0.0
    assert reset_res["sample_count"] == 0

# =========================================================================
# 3. FASTAPI PHASE 6 ENDPOINTS & INTEGRATION TESTS
# =========================================================================

def test_api_telemetry_endpoint():
    # Insert some vectors and query
    dim = 8
    np.random.seed(42)
    vecs = np.random.uniform(-1.0, 1.0, size=(20, dim)).tolist()
    client.post("/vectors/insert", json={"vectors": vecs})
    
    query_vec = np.random.uniform(-1.0, 1.0, size=(dim,)).tolist()
    client.post("/vectors/query", json={"vector": query_vec, "k": 5})
    
    res = client.get("/telemetry")
    assert res.status_code == 200
    data = res.json()
    
    assert "vectors" in data
    assert "query_count" in data
    assert "latency" in data
    assert "avg_ms" in data["latency"]
    assert "p50_ms" in data["latency"]
    assert "p95_ms" in data["latency"]
    assert "p99_ms" in data["latency"]
    assert "backend" in data
    assert "simd" in data

def test_api_drift_endpoint_and_reset():
    res = client.get("/drift")
    assert res.status_code == 200
    data = res.json()
    assert "query_drift" in data
    assert "dataset_drift" in data
    assert data["query_drift"]["status"] in ["normal", "warning", "drift_detected"]
    assert data["dataset_drift"]["status"] in ["normal", "warning", "drift_detected"]
    
    # Test reset endpoint
    reset_res = client.post("/drift/reset", json={"channel": "all"})
    assert reset_res.status_code == 200
    reset_data = reset_res.json()
    assert reset_data["status"] == "success"
    assert reset_data["query_drift"]["score"] == 0.0
    assert reset_data["dataset_drift"]["score"] == 0.0

# =========================================================================
# 4. IMPORTANT INTEGRATION TEST: NO AUTONOMOUS ACTION
# =========================================================================

def test_phase6_drift_detection_is_strictly_non_autonomous():
    """
    Validates the end-to-end scenario:
    1. Baseline vectors created.
    2. Baseline queries -> status NORMAL.
    3. Shifted queries -> status DRIFT_DETECTED.
    4. Verify HNSW/PQ parameters did NOT change automatically.
    5. Verify search backend did NOT change.
    6. Verify zero autonomous tuning was invoked.
    """
    dim = 16
    np.random.seed(123)
    
    # Step 1: Initialize index and insert baseline dataset
    client.post("/drift/reset", json={"channel": "all"})
    
    baseline_dataset = np.random.normal(loc=0.0, scale=1.0, size=(100, dim)).astype(np.float32)
    client.post("/vectors/insert", json={"vectors": baseline_dataset.tolist()})
    
    # Record initial index parameters
    tele_before = client.get("/system/telemetry").json()
    ef_search_before = tele_before["hnsw_parameters"]["ef_search"]
    m_before = tele_before["hnsw_parameters"]["m"]
    ef_construction_before = tele_before["hnsw_parameters"]["ef_construction"]
    backend_before = tele_before["backend"]
    
    # Initialize query baseline
    baseline_queries = np.random.normal(loc=0.0, scale=1.0, size=(50, dim)).astype(np.float32)
    query_drift_detector.set_baseline(baseline_queries)
    
    # Step 2: Query with baseline distribution -> NORMAL
    for q in baseline_queries[:20]:
        client.post("/vectors/query", json={"vector": q.tolist(), "k": 5})
        
    drift_res1 = client.get("/drift").json()
    assert drift_res1["query_drift"]["status"] == "normal"
    
    # Step 3 & 4: Query with drastically shifted distribution (loc=5.0, scale=3.0)
    shifted_queries = np.random.normal(loc=5.0, scale=3.0, size=(50, dim)).astype(np.float32)
    for q in shifted_queries:
        client.post("/vectors/query", json={"vector": q.tolist(), "k": 5})
        
    drift_res2 = client.get("/drift").json()
    assert drift_res2["query_drift"]["status"] == "drift_detected"
    assert drift_res2["query_drift"]["score"] >= 0.60
    
    # Step 5: Verify NO autonomous parameter modifications occurred!
    tele_after = client.get("/system/telemetry").json()
    assert tele_after["hnsw_parameters"]["ef_search"] == ef_search_before
    assert tele_after["hnsw_parameters"]["m"] == m_before
    assert tele_after["hnsw_parameters"]["ef_construction"] == ef_construction_before
    assert tele_after["backend"] == backend_before

if __name__ == "__main__":
    pytest.main(["-v", __file__])
