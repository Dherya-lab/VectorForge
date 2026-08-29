import sys
import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

# Ensure root directory is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import VectorForgeIndex, simd_info
from server.app import app

client = TestClient(app)

# =========================================================================
# 1. PYBIND11 BINDING UNIT TESTS
# =========================================================================

def test_pybind11_import_and_simd():
    info = simd_info()
    assert isinstance(info, dict)
    assert "cpu_brand" in info
    assert "avx2" in info
    assert "active_dispatch" in info

def test_pybind11_insert_and_size():
    index = VectorForgeIndex(dimension=4, backend="hnsw")
    assert index.size() == 0
    assert index.dimension() == 4

    data = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0]
    ], dtype=np.float32)

    count = index.insert(data)
    assert count == 2
    assert index.size() == 2

def test_pybind11_search_ordered():
    index = VectorForgeIndex(dimension=2, backend="hnsw")
    vectors = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 5.0],
        [0.0, 2.0]
    ], dtype=np.float32)
    index.insert(vectors)

    query = np.array([0.0, 0.0], dtype=np.float32)
    results = index.search(query, k=4)
    assert len(results) == 4

    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)
    assert results[0]["id"] == 0
    assert pytest.approx(results[0]["distance"], 1e-4) == 0.0

def test_pybind11_k_equals_one():
    index = VectorForgeIndex(dimension=2, backend="hnsw")
    index.insert(np.array([[10.0, 10.0], [1.0, 1.0]], dtype=np.float32))

    results = index.search(np.array([0.0, 0.0], dtype=np.float32), k=1)
    assert len(results) == 1
    assert results[0]["id"] == 1

def test_pybind11_k_greater_than_dataset():
    index = VectorForgeIndex(dimension=2, backend="hnsw")
    index.insert(np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32))

    results = index.search(np.array([0.0, 0.0], dtype=np.float32), k=50)
    assert len(results) == 2

def test_pybind11_dimension_mismatch_error():
    index = VectorForgeIndex(dimension=4, backend="hnsw")
    bad_data = np.array([[1.0, 2.0]], dtype=np.float32)

    with pytest.raises(Exception):
        index.insert(bad_data)

    bad_query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(Exception):
        index.search(bad_query, k=1)

def test_pybind11_bruteforce_backend():
    index = VectorForgeIndex(dimension=3, backend="bruteforce")
    assert index.backend() == "bruteforce"
    index.insert(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
    assert index.size() == 2
    res = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert res[0]["id"] == 0
    assert pytest.approx(res[0]["distance"], 1e-4) == 0.0

# =========================================================================
# 2. FASTAPI CONTROL PLANE ENDPOINT TESTS
# =========================================================================

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "VectorForge"
    assert data["status"] == "online"

def test_api_system_telemetry():
    response = client.get("/system/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "total_vectors" in data
    assert "simd" in data
    assert "active_dispatch" in data["simd"]

def test_api_vectors_insert_and_query():
    # Insert vectors
    insert_payload = {
        "vectors": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ]
    }
    insert_res = client.post("/vectors/insert", json=insert_payload)
    assert insert_res.status_code == 201
    insert_data = insert_res.json()
    assert insert_data["inserted"] == 4
    assert insert_data["dimension"] == 3
    assert insert_data["total_vectors"] >= 4

    # Query vector
    query_payload = {
        "vector": [0.0, 0.0, 0.0],
        "k": 2
    }
    query_res = client.post("/vectors/query", json=query_payload)
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert len(query_data["results"]) == 2
    assert query_data["results"][0]["id"] == 0
    assert pytest.approx(query_data["results"][0]["distance"], 1e-4) == 0.0
    assert "query_latency_us" in query_data

def test_api_index_tune():
    tune_payload = {
        "ef_search": 80,
        "m": 32,
        "ef_construction": 120
    }
    tune_res = client.post("/index/tune", json=tune_payload)
    assert tune_res.status_code == 200
    tune_data = tune_res.json()
    assert tune_data["ef_search"] == 80
    assert tune_data["m"] == 32
    assert tune_data["ef_construction"] == 120

def test_api_validation_errors():
    # 1. Empty vectors list
    bad_insert = client.post("/vectors/insert", json={"vectors": []})
    assert bad_insert.status_code in [400, 422]

    # 2. Inconsistent dimensions in insert
    inconsistent_insert = client.post("/vectors/insert", json={"vectors": [[1.0, 2.0], [1.0, 2.0, 3.0]]})
    assert inconsistent_insert.status_code == 400

    # 3. Invalid k <= 0
    bad_k_query = client.post("/vectors/query", json={"vector": [0.0, 0.0, 0.0], "k": 0})
    assert bad_k_query.status_code == 422

    # 4. Mismatched query dimension
    bad_dim_query = client.post("/vectors/query", json={"vector": [0.0, 0.0], "k": 2})
    assert bad_dim_query.status_code == 400

if __name__ == "__main__":
    pytest.main(["-v", __file__])
