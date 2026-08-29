import pytest
from fastapi.testclient import TestClient
from server.app import app, state, benchmark_history

client = TestClient(app)

def test_full_review_ready_accuracy_flow():
    # Clear history for clean test run
    benchmark_history.clear()
    client.post("/index/tune", json={"ef_search": 50})

    # STEP 1: Run baseline benchmark
    b1_res = client.post("/benchmark/run")
    assert b1_res.status_code == 200
    b1 = b1_res.json()
    assert b1["run_name"] == "Baseline"
    assert b1["configuration"]["ef_search"] == 50
    assert b1["configuration"]["m"] == state.m
    assert b1["metrics"]["recall"] > 0.85
    assert b1["metrics"]["p95_latency_ms"] > 0
    assert b1["metrics"]["qps"] > 1000
    baseline_recall = b1["metrics"]["recall"]
    baseline_p95 = b1["metrics"]["p95_latency_ms"]

    # STEP 2: Verify benchmark history
    hist1 = client.get("/benchmark/history").json()
    assert len(hist1["runs"]) == 1
    assert hist1["runs"][0]["configuration"]["ef_search"] == 50

    # STEP 3: Call POST /tune/recommend
    rec_res = client.post("/tune/recommend", json={"priority": "recall", "target_recall": 0.98})
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert "recommendation" in rec_data
    rec = rec_data["recommendation"]
    assert rec["changes"]["ef_search"] == 96
    assert rec["expected_effect"]["recall"] == "increase"
    assert rec_data["current_configuration"]["ef_search"] == 50
    assert rec_data["proposed_configuration"]["ef_search"] == 96

    # STEP 4: Call POST /tune/apply
    apply_res = client.post("/tune/apply", json={"recommendation": rec})
    assert apply_res.status_code == 200
    app_data = apply_res.json()
    assert app_data["applied"] is True
    assert app_data["previous_configuration"]["ef_search"] == 50
    assert app_data["configuration"]["ef_search"] == 96
    assert state.ef_search == 96

    # STEP 5: Run benchmark after apply (MUST evaluate ef_search=96)
    b2_res = client.post("/benchmark/run")
    assert b2_res.status_code == 200
    b2 = b2_res.json()
    assert b2["run_name"] == "Experiment 02"
    assert b2["configuration"]["ef_search"] == 96
    assert b2["metrics"]["recall"] >= baseline_recall
    assert b2["deltas"]["recall_pct"] >= 0

    # STEP 6: Verify benchmark history has both runs with their distinct configurations
    hist2 = client.get("/benchmark/history").json()
    assert len(hist2["runs"]) == 2
    assert hist2["runs"][0]["configuration"]["ef_search"] == 50
    assert hist2["runs"][1]["configuration"]["ef_search"] == 96

    # STEP 7: Call GET /tune/status
    status_res = client.get("/tune/status")
    assert status_res.status_code == 200
    stat_data = status_res.json()
    assert stat_data["current_configuration"]["ef_search"] == 96
    assert stat_data["last_applied_change"]["ef_search"] == 96
    assert stat_data["rollback_available"] is True
    assert stat_data["tuning_history_count"] >= 1

    # STEP 8: Call POST /tune/rollback
    rb_res = client.post("/tune/rollback")
    assert rb_res.status_code == 200
    rb_data = rb_res.json()
    assert rb_data["applied"] is True
    assert rb_data["configuration"]["ef_search"] == 50
    assert state.ef_search == 50

    # STEP 9: Run benchmark after rollback (MUST evaluate ef_search=50)
    b3_res = client.post("/benchmark/run")
    assert b3_res.status_code == 200
    b3 = b3_res.json()
    assert b3["run_name"] == "Experiment 03"
    assert b3["configuration"]["ef_search"] == 50
    assert abs(b3["metrics"]["recall"] - baseline_recall) < 0.005
