import sys
import os
import time
import pytest
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.tuner import (
    VectorForgeTuner,
    TuningTarget,
    TuningRecommendation,
    ExpectedEffect,
    PARAMETER_BOUNDS
)
from server.app import app, ai_tuner, state

client = TestClient(app)

# =========================================================================
# 1. AI TUNER UNIT & VALIDATION TESTS
# =========================================================================

def test_missing_api_key_safe_fallback():
    # Ensure GEMINI_API_KEY is not set or test fallback behavior
    old_key = os.environ.get("GEMINI_API_KEY")
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

    try:
        tuner = VectorForgeTuner()
        telemetry = {"latency": {"p95_ms": 5.0}, "vectors": 1000}
        drift = {"query_drift": {"status": "normal"}}
        config = {"ef_search": 50, "m": 16, "ef_construction": 100}
        target = TuningTarget(target_recall=0.95, max_p95_latency_ms=20.0, priority="recall")

        rec = tuner.recommend(telemetry, drift, config, target)
        assert isinstance(rec, TuningRecommendation)
        assert rec.action in ["tune", "no_change"]
        assert 0.0 <= rec.confidence <= 1.0
    finally:
        if old_key is not None:
            os.environ["GEMINI_API_KEY"] = old_key

def test_valid_structured_recommendation_parsing():
    raw_dict = {
        "action": "tune",
        "reason": "Recall optimization with available latency headroom.",
        "changes": {"ef_search": 128},
        "expected_effect": {"recall": "increase", "latency": "increase", "memory": "neutral"},
        "confidence": 0.85,
        "rebuild_required": False
    }
    rec = TuningRecommendation(**raw_dict)
    assert rec.action == "tune"
    assert rec.changes["ef_search"] == 128
    assert rec.confidence == 0.85
    assert not rec.rebuild_required

def test_invalid_json_or_missing_fields_rejected():
    with pytest.raises(Exception):
        TuningRecommendation(action="tune") # missing reason & confidence

def test_unknown_or_disallowed_parameter_rejected():
    with pytest.raises(Exception):
        TuningRecommendation(
            action="tune",
            reason="Injecting malicious or invalid parameter",
            changes={"arbitrary_code": 100},
            confidence=0.5
        )

def test_out_of_bounds_ef_search_rejected():
    with pytest.raises(Exception):
        TuningRecommendation(
            action="tune",
            reason="Exceeds max limit",
            changes={"ef_search": 20000}, # max is 10000
            confidence=0.5
        )

    with pytest.raises(Exception):
        TuningRecommendation(
            action="tune",
            reason="Below min limit",
            changes={"ef_search": 0}, # min is 1
            confidence=0.5
        )

def test_out_of_bounds_m_rejected():
    with pytest.raises(Exception):
        TuningRecommendation(
            action="tune",
            reason="M too large",
            changes={"m": 256}, # max is 128
            confidence=0.5
        )

def test_rebuild_required_for_structural_parameters():
    tuner = VectorForgeTuner()
    rec = TuningRecommendation(
        action="rebuild_required",
        reason="M requires index reconstruction",
        changes={"m": 32},
        confidence=0.8,
        rebuild_required=True
    )
    valid, msg, req_rebuild = tuner.validate_recommendation(rec)
    assert valid
    assert req_rebuild is True

    # Test applying a rebuild_required change -> rejected from live apply
    result = tuner.apply(
        recommendation=rec,
        current_config={"ef_search": 50, "m": 16},
        telemetry_before={},
        apply_fn=lambda c: c
    )
    assert result["status"] == "rejected"
    assert result["applied"] is False
    assert result["rebuild_required"] is True

def test_recommendation_does_not_modify_configuration():
    initial_ef_search = state.ef_search
    res = client.post("/tune/recommend", json={"priority": "recall", "target_recall": 0.98})
    assert res.status_code == 200
    # Live configuration must remain untouched by recommendation
    assert state.ef_search == initial_ef_search

def test_apply_and_rollback_lifecycle():
    # 1. Start with known configuration
    state.ef_search = 50
    initial_config = client.get("/tune/status").json()["current_configuration"]
    assert initial_config["ef_search"] == 50

    # 2. Apply a safe tuning change
    rec_payload = {
        "recommendation": {
            "action": "tune",
            "reason": "Test tuning application",
            "changes": {"ef_search": 120},
            "expected_effect": {"recall": "increase", "latency": "increase"},
            "confidence": 0.9,
            "rebuild_required": False
        }
    }
    apply_res = client.post("/tune/apply", json=rec_payload)
    assert apply_res.status_code == 200
    apply_data = apply_res.json()
    assert apply_data["status"] == "success"
    assert apply_data["applied"] is True
    assert apply_data["configuration"]["ef_search"] == 120
    assert state.ef_search == 120

    # 3. Check /tune/status reflects the update
    status_res = client.get("/tune/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["current_configuration"]["ef_search"] == 120
    assert status_data["rollback_available"] is True

    # 4. Roll back
    rollback_res = client.post("/tune/rollback")
    assert rollback_res.status_code == 200
    rollback_data = rollback_res.json()
    assert rollback_data["status"] == "success"
    assert rollback_data["applied"] is True
    assert rollback_data["configuration"]["ef_search"] == 50
    assert state.ef_search == 50

def test_repeated_tuning_and_rollback_stack():
    state.ef_search = 90
    ai_tuner._rollback_stack.clear()
    ai_tuner.history.clear()

    # Step 1: 90 -> 120
    r1 = client.post("/tune/apply", json={"recommendation": {"action": "tune", "reason": "step 1", "changes": {"ef_search": 120}, "confidence": 1.0}}).json()
    assert r1["applied"] is True
    assert r1["configuration"]["ef_search"] == 120
    assert r1["previous_configuration"]["ef_search"] == 90

    # Step 2: 120 -> 150
    r2 = client.post("/tune/apply", json={"recommendation": {"action": "tune", "reason": "step 2", "changes": {"ef_search": 150}, "confidence": 1.0}}).json()
    assert r2["applied"] is True
    assert r2["configuration"]["ef_search"] == 150
    assert r2["previous_configuration"]["ef_search"] == 120

    s = client.get("/tune/status").json()
    assert s["current_configuration"]["ef_search"] == 150
    assert s["rollback_available"] is True
    assert s["tuning_history_count"] == 2

    # Step 3: Rollback 150 -> 120
    rb1 = client.post("/tune/rollback").json()
    assert rb1["applied"] is True
    assert rb1["configuration"]["ef_search"] == 120
    assert client.get("/tune/status").json()["rollback_available"] is True

    # Step 4: Rollback 120 -> 90
    rb2 = client.post("/tune/rollback").json()
    assert rb2["applied"] is True
    assert rb2["configuration"]["ef_search"] == 90
    assert client.get("/tune/status").json()["rollback_available"] is False

    # Step 5: Rollback when empty -> graceful error
    rb3 = client.post("/tune/rollback").json()
    assert rb3["applied"] is False
    assert rb3["status"] == "error"
    assert "No previous configuration" in rb3["message"]

# =========================================================================
# 2. FASTAPI ENDPOINT & ERROR HANDLING TESTS
# =========================================================================

def test_api_tune_recommend_endpoint():
    res = client.post("/tune/recommend", json={"target_recall": 0.95, "max_p95_latency_ms": 15.0, "priority": "balanced"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "recommendation" in data
    assert "action" in data["recommendation"]
    assert "changes" in data["recommendation"]
    assert "current_configuration" in data
    assert "proposed_configuration" in data

def test_api_tune_apply_without_prior_recommendation_error():
    # Clear last recommendation
    ai_tuner.last_recommendation = None
    res = client.post("/tune/apply", json={})
    assert res.status_code == 400
    assert "No recommendation" in res.json()["detail"]

def test_api_tune_status_never_exposes_api_key():
    res = client.get("/tune/status")
    assert res.status_code == 200
    data = res.json()
    assert "gemini_api_key_configured" in data
    # Ensure no raw API key string is exposed
    assert "api_key" not in data
    assert "key" not in data

# =========================================================================
# 3. DETERMINISTIC RECALL & LATENCY TRADE-OFF EVALUATION DEMO
# =========================================================================

def test_phase7_recall_latency_tradeoff_demonstration():
    """
    Demonstrates the performance trade-off:
    1. Create a synthetic dataset with known ground truth (brute force).
    2. Evaluate Recall@10 and P95 search latency with low efSearch (e.g. efSearch=8).
    3. AI recommends increasing efSearch (e.g. efSearch=64).
    4. Apply the recommendation.
    5. Evaluate Recall@10 and P95 search latency with tuned efSearch.
    6. Verify Recall increases as expected.
    """
    from server import VectorForgeIndex

    N = 1000
    D = 32
    K = 10
    NUM_QUERIES = 30

    np.random.seed(42)
    dataset = np.random.normal(0.0, 1.0, size=(N, D)).astype(np.float32)
    queries = np.random.normal(0.0, 1.0, size=(NUM_QUERIES, D)).astype(np.float32)

    # Build exact ground truth via Brute-Force index
    bf_index = VectorForgeIndex(dimension=D, backend="bruteforce")
    bf_index.insert(dataset)
    ground_truth = []
    for q in queries:
        gt_res = bf_index.search(q, k=K)
        ground_truth.append(set(r["id"] for r in gt_res))

    # Build HNSW index with low initial efSearch
    hnsw = VectorForgeIndex(dimension=D, backend="hnsw", m=16, ef_construction=100, ef_search=8, random_seed=42)
    hnsw.insert(dataset)

    # 1. Measure BEFORE tuning (low efSearch = 8)
    hits_before = 0
    total_items = NUM_QUERIES * K
    latencies_before = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter_ns()
        res = hnsw.search(q, k=K)
        t1 = time.perf_counter_ns()
        latencies_before.append((t1 - t0) / 1000.0) # us
        retrieved_ids = set(r["id"] for r in res)
        hits_before += len(retrieved_ids.intersection(ground_truth[i]))

    recall_before = hits_before / total_items
    p95_before = float(np.percentile(latencies_before, 95))

    # 2. Apply tuning to efSearch = 64
    hnsw.set_ef_search(64)

    # 3. Measure AFTER tuning (tuned efSearch = 64)
    hits_after = 0
    latencies_after = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter_ns()
        res = hnsw.search(q, k=K)
        t1 = time.perf_counter_ns()
        latencies_after.append((t1 - t0) / 1000.0)
        retrieved_ids = set(r["id"] for r in res)
        hits_after += len(retrieved_ids.intersection(ground_truth[i]))

    recall_after = hits_after / total_items
    p95_after = float(np.percentile(latencies_after, 95))

    # Verification: Recall@10 must improve after tuning efSearch upward
    assert recall_after >= recall_before
    assert recall_after >= 0.90

if __name__ == "__main__":
    pytest.main(["-v", __file__])
