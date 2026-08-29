import pytest
from fastapi.testclient import TestClient
from server.app import app, state

client = TestClient(app)

def test_dashboard_static_files():
    # 1. HTML index
    res = client.get("/dashboard/")
    assert res.status_code == 200
    assert "VectorForge" in res.text
    assert "Optimization Overview" in res.text
    assert "Recall vs P95 Latency" in res.text
    assert "AI Tuner" in res.text
    assert "Benchmark results" in res.text
    assert "tradeoff-canvas" in res.text

    # 2. CSS stylesheet
    res_css = client.get("/dashboard/styles.css")
    assert res_css.status_code == 200
    assert "--accent-green" in res_css.text

    # 3. JS script
    res_js = client.get("/dashboard/app.js")
    assert res_js.status_code == 200
    assert "handleRunBenchmark" in res_js.text
    assert "handleApplyConfiguration" in res_js.text
    assert "handleRollback" in res_js.text

def test_benchmark_endpoint_live_execution():
    res = client.post("/benchmark/run")
    assert res.status_code == 200
    data = res.json()
    
    assert "metrics" in data
    assert "recall" in data["metrics"]
    assert "p95_latency_ms" in data["metrics"]
    assert "qps" in data["metrics"]
    assert "memory_mb" in data["metrics"]
    assert "tradeoff_points" in data
    assert len(data["tradeoff_points"]) > 0
    assert "runs" in data
    assert len(data["runs"]) >= 1

def test_benchmark_history_endpoint():
    res = client.get("/benchmark/history")
    assert res.status_code == 200
    data = res.json()
    assert "runs" in data
    assert len(data["runs"]) >= 1
    assert "latest" in data

def test_benchmark_reproducibility_and_consistency():
    # Ensure baseline is set to 50
    client.post("/index/tune", json={"ef_search": 50})
    
    # Run 1
    r1 = client.post("/benchmark/run").json()
    # Run 2 with identical configuration
    r2 = client.post("/benchmark/run").json()
    
    # Exact same recall
    assert r1["metrics"]["recall"] == r2["metrics"]["recall"]
    # P95 latency is consistent (no cold-start distortion)
    assert abs(r1["metrics"]["p95_latency_ms"] - r2["metrics"]["p95_latency_ms"]) < 0.5
    # QPS is positive and healthy
    assert r1["metrics"]["qps"] > 1000 and r2["metrics"]["qps"] > 1000

def test_first_review_end_to_end_flow():
    # Step 1: Run baseline benchmark
    client.post("/index/tune", json={"ef_search": 50})
    bench_res = client.post("/benchmark/run")
    assert bench_res.status_code == 200
    baseline_ef = state.ef_search
    baseline_recall = bench_res.json()["metrics"]["recall"]

    # Step 2: Request AI recommendation (should recommend higher ef_search e.g. 96)
    rec_res = client.post("/tune/recommend", json={"priority": "recall", "target_recall": 0.98})
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert "recommendation" in rec_data
    rec = rec_data["recommendation"]
    assert rec["changes"]["ef_search"] != baseline_ef
    assert rec["changes"]["ef_search"] > baseline_ef

    # Step 3: Apply configuration
    apply_res = client.post("/tune/apply", json={"recommendation": rec})
    assert apply_res.status_code == 200
    apply_data = apply_res.json()
    assert apply_data["applied"] is True
    assert apply_data["previous_configuration"]["ef_search"] == baseline_ef
    assert apply_data["configuration"]["ef_search"] == rec["changes"]["ef_search"]

    # Step 4: Run post-tuning benchmark and verify measured recall improved
    post_bench = client.post("/benchmark/run").json()
    assert post_bench["metrics"]["recall"] >= baseline_recall

    # Step 5: Verify status confirms rollback availability
    status_res = client.get("/tune/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["rollback_available"] is True
    assert status_data["tuning_history_count"] >= 1

    # Step 6: Execute Rollback
    rollback_res = client.post("/tune/rollback")
    assert rollback_res.status_code == 200
    rollback_data = rollback_res.json()
    assert rollback_data["applied"] is True
    assert rollback_data["configuration"]["ef_search"] == baseline_ef

def test_benchmark_clear_endpoint():
    # Populate a run
    client.post("/benchmark/run")
    
    # Call clear
    res = client.post("/benchmark/clear")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["runs"] == []
    
    # Verify history is empty
    hist = client.get("/benchmark/history").json()
    assert hist["runs"] == []
    assert hist["latest"] is None

def test_delete_benchmark_history_persistence_and_reset():
    # Step A: Run 3 benchmarks
    r1 = client.post("/benchmark/run").json()
    r2 = client.post("/benchmark/run").json()
    r3 = client.post("/benchmark/run").json()
    
    # Step B: Confirm they appear in history
    hist = client.get("/benchmark/history").json()
    assert len(hist["runs"]) >= 3
    
    # Step C-E: Execute DELETE /benchmark/history
    del_res = client.delete("/benchmark/history")
    assert del_res.status_code == 200
    del_data = del_res.json()
    assert del_data["status"] == "success"
    assert del_data["deleted_count"] >= 3
    assert del_data["message"] == "Benchmark history cleared"
    
    # Step I-J: Directly call GET /benchmark/history and verify empty
    hist_after = client.get("/benchmark/history").json()
    assert hist_after["runs"] == []
    assert hist_after["latest"] is None
    assert hist_after["tradeoff_points"] == []
    
    # Step K-L: Run a new benchmark and verify it starts fresh as Baseline (run 1)
    fresh_run = client.post("/benchmark/run").json()
    assert fresh_run["run_id"] == "run_1"
    assert fresh_run["run_name"] == "Baseline"
    assert fresh_run["source"] == "baseline"
    assert len(fresh_run["runs"]) == 1

