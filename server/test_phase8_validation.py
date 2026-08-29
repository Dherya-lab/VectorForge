import pytest
import numpy as np
from fastapi.testclient import TestClient
from server.app import app, state, ai_tuner
from server.tuner import TuningRecommendation, TuningChanges, ExpectedEffect, TuningTarget

@pytest.fixture
def client():
    return TestClient(app)

def test_phase8_full_loop_validation(client):
    """
    Tests the complete 15-step Phase 8 Closed-Loop Validation flow:
    Baseline -> Recommend (Proposed) -> Apply (Snapshot) -> Benchmark -> Validate (Before vs After) -> Rollback
    """
    # STEP 1: Clear experiment history and reset to baseline
    client.post("/benchmark/clear")
    client.post("/index/tune", json={"ef_search": 50, "m": 16})
    
    # STEP 2 & 3: Run baseline benchmark & verify baseline configuration and metrics stored
    b1_res = client.post("/benchmark/run")
    assert b1_res.status_code == 200
    b1 = b1_res.json()
    assert b1["run_name"] == "Baseline"
    assert b1["configuration"]["ef_search"] == 50
    assert b1["configuration"]["m"] == 16
    assert b1["metrics"]["recall"] > 0.85
    assert b1["source"] == "baseline"
    
    # STEP 4 & 5: Request AI recommendation & verify marked PROPOSED
    rec_res = client.post("/tune/recommend", json={"priority": "recall", "target_recall": 0.98})
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert rec_data["status"] == "success"
    assert rec_data["proposed_configuration"]["ef_search"] == 96
    
    status_res = client.get("/tune/status").json()
    assert status_res["recommendation_status"] == "proposed"
    assert status_res["last_validation"] is None
    
    # STEP 6 & 7: Apply recommendation & verify rollback snapshot created
    apply_res = client.post("/tune/apply", json={"recommendation": rec_data["recommendation"]})
    assert apply_res.status_code == 200
    applied_data = apply_res.json()
    assert applied_data["applied"] is True
    assert applied_data["previous_configuration"]["ef_search"] == 50
    assert applied_data["configuration"]["ef_search"] == 96
    
    status_after_apply = client.get("/tune/status").json()
    assert status_after_apply["recommendation_status"] == "applied"
    assert status_after_apply["current_configuration"]["ef_search"] == 96
    assert status_after_apply["rollback_available"] is True
    
    # Verify no benchmark row created yet
    hist_before_bench = client.get("/benchmark/history").json()
    assert len(hist_before_bench["runs"]) == 1
    
    # STEP 8 & 9: Run post-apply benchmark
    b2_res = client.post("/benchmark/run")
    assert b2_res.status_code == 200
    b2 = b2_res.json()
    assert b2["run_name"] == "Experiment 02"
    assert b2["configuration"]["ef_search"] == 96
    assert b2["source"] == "ai_recommendation"
    
    # STEP 10 & 11: Compare BEFORE vs AFTER & verify empirical validation decision
    val = b2["validation"]
    assert val is not None
    assert val["status"] in ["improved", "acceptable"]
    assert val["decision"] == "ACCEPT"
    assert val["before_metrics"]["recall"] <= b2["metrics"]["recall"]
    assert "deltas" in val
    assert "recall_pct" in val["deltas"]
    assert val["deltas"]["recall_pct"] > 0
    
    status_validated = client.get("/tune/status").json()
    assert status_validated["recommendation_status"] in ["validated", "improved", "acceptable"]
    assert status_validated["last_validation"]["decision"] == "ACCEPT"
    
    # STEP 12 & 13: Test Rollback
    rb_res = client.post("/tune/rollback")
    assert rb_res.status_code == 200
    rb_data = rb_res.json()
    assert rb_data["applied"] is True
    assert rb_data["configuration"]["ef_search"] == 50
    
    status_rolled_back = client.get("/tune/status").json()
    assert status_rolled_back["recommendation_status"] == "rolled_back"
    assert status_rolled_back["current_configuration"]["ef_search"] == 50
    
    # STEP 14 & 15: Post-rollback benchmark run & consistency
    b3_res = client.post("/benchmark/run")
    assert b3_res.status_code == 200
    b3 = b3_res.json()
    assert b3["run_name"] == "Experiment 03"
    assert b3["configuration"]["ef_search"] == 50
    
    hist_final = client.get("/benchmark/history").json()
    assert len(hist_final["runs"]) == 3
    assert hist_final["runs"][0]["configuration"]["ef_search"] == 50
    assert hist_final["runs"][1]["configuration"]["ef_search"] == 96
    assert hist_final["runs"][2]["configuration"]["ef_search"] == 50

def test_phase8_safety_and_regression_detection(client):
    """
    Tests safety rejection of invalid parameters and proper validation handling.
    """
    # 1. Out of bounds ef_search rejected by validation
    invalid_rec = {
        "action": "tune",
        "reason": "Test out of bounds",
        "changes": {"ef_search": 999999},
        "expected_effect": {"recall": "increase", "latency": "increase", "memory": "neutral"},
        "confidence": 0.5,
        "rebuild_required": False
    }
    res = client.post("/tune/apply", json={"recommendation": invalid_rec})
    assert res.status_code == 422  # Pydantic schema validation rejects bounds > 10000

    # 2. Structural parameter rejected from live update
    structural_rec = {
        "action": "rebuild_required",
        "reason": "Test structural",
        "changes": {"m": 32},
        "expected_effect": {"recall": "increase", "latency": "increase", "memory": "increase"},
        "confidence": 0.9,
        "rebuild_required": True
    }
    res = client.post("/tune/apply", json={"recommendation": structural_rec})
    assert res.status_code == 200
    assert res.json()["applied"] is False
    assert res.json()["rebuild_required"] is True

def test_clear_benchmarks_preserves_configuration(client):
    """
    Clear benchmark history clears only benchmark runs without altering current index parameters.
    """
    client.post("/index/tune", json={"ef_search": 120})
    client.post("/benchmark/clear")
    
    status = client.get("/tune/status").json()
    assert status["current_configuration"]["ef_search"] == 120
