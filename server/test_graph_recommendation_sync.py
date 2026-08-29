import pytest
from fastapi.testclient import TestClient
from server.app import app, state, benchmark_history, ai_tuner
from server.tuner import TuningRecommendation

def test_graph_and_tuner_recommendation_synchronization():
    client = TestClient(app)
    client.post("/index/tune", json={"ef_search": 50, "m": 16, "ef_construction": 100})
    client.delete("/benchmark/history")

    # 1. Run baseline
    b_resp = client.post("/benchmark/run").json()
    assert b_resp["run_name"] == "Baseline"
    tradeoff_pts = b_resp.get("tradeoff_points", [])
    assert len(tradeoff_pts) > 0

    # Ensure tradeoff_points have qps and no hardcoded recommendation flag
    for pt in tradeoff_pts:
        assert "qps" in pt
        assert "p95_ms" in pt
        assert "recall" in pt
        assert "ef_search" in pt
        assert "recommended" not in pt

    # 2. Get status and recommend ef200
    rec_payload = {
        "action": "tune",
        "priority": "recall",
        "changes": {
            "ef_search": 200
        },
        "target_recall": 0.99,
        "max_p95_latency_ms": 2.0,
        "rebuild_required": False,
        "confidence": 0.95,
        "reason": "Expanding beam width to 200 for peak recall."
    }
    ai_tuner.last_recommendation = TuningRecommendation(**rec_payload)

    status = client.get("/tune/status").json()
    assert status["last_recommendation"]["changes"]["ef_search"] == 200

    # 3. Simulate client canonical recommendation resolution (same logic as getCanonicalRecommendation())
    def get_canonical_rec(rec, current_config):
        rec_changes = rec.get("changes", {}) if rec else {}
        rec_ef = rec_changes.get("ef_search", 96)
        rec_m = rec_changes.get("m", current_config.get("m", 16))
        return {
            "m": rec_m,
            "ef_search": rec_ef,
            "tuner_label": f"Proposed M{rec_m} / ef{rec_ef}",
            "graph_label": f"Proposed M{rec_m} · ef{rec_ef}",
            "apply_target": rec_ef
        }

    curr_cfg = client.get("/index/config").json()
    canon200 = get_canonical_rec(status["last_recommendation"], curr_cfg)
    assert canon200["tuner_label"] == "Proposed M16 / ef200"
    assert canon200["graph_label"] == "Proposed M16 · ef200"
    assert canon200["apply_target"] == 200

    # 4. Change recommendation to M16/ef144
    rec_payload_144 = {
        "action": "tune",
        "priority": "recall",
        "changes": {
            "ef_search": 144
        },
        "target_recall": 0.985,
        "max_p95_latency_ms": 1.5,
        "rebuild_required": False,
        "confidence": 0.92,
        "reason": "Adjusting beam width to 144."
    }
    ai_tuner.last_recommendation = TuningRecommendation(**rec_payload_144)

    status_144 = client.get("/tune/status").json()
    canon144 = get_canonical_rec(status_144["last_recommendation"], curr_cfg)
    assert canon144["tuner_label"] == "Proposed M16 / ef144"
    assert canon144["graph_label"] == "Proposed M16 · ef144"
    assert canon144["apply_target"] == 144

    # 5. Verify apply target and validation
    app_res = client.post("/tune/apply", json={"recommendation": status_144["last_recommendation"]}).json()
    assert app_res["applied"] is True
    assert app_res["configuration"]["ef_search"] == 144

    bench2 = client.post("/benchmark/run").json()
    assert bench2["configuration"]["ef_search"] == 144
    val = client.get("/tune/status").json()["last_validation"]
    assert val is not None
    assert val["before_metrics"]["recall"] == b_resp["metrics"]["recall"]
    assert val["after_metrics"]["recall"] == bench2["metrics"]["recall"]
