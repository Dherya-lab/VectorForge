from fastapi.testclient import TestClient
from server.app import app, state, benchmark_history, ai_tuner

def test_validation_summary():
    client = TestClient(app)
    client.post('/index/tune', json={'ef_search': 50})
    client.delete('/benchmark/history')

    # Step 1: Run benchmark -> Baseline
    b1 = client.post('/benchmark/run').json()
    baseline_metrics = b1['metrics']
    print('Baseline metrics:', baseline_metrics)

    # Step 2: Recommend -> Apply
    rec = client.post('/tune/recommend', json={'priority': 'recall', 'target_recall': 0.98}).json()['recommendation']
    client.post('/tune/apply', json={'recommendation': rec})

    # Step 3: Run benchmark -> Experiment 02
    b2 = client.post('/benchmark/run').json()
    status_val = client.get('/tune/status').json()['last_validation']
    print('Validation summary text:', status_val['reason'])
    print('Validation before recall:', status_val['before_metrics']['recall'], 'p95:', status_val['before_metrics']['p95_latency_ms'])
    print('Validation after recall:', status_val['after_metrics']['recall'], 'p95:', status_val['after_metrics']['p95_latency_ms'])

    assert status_val['before_metrics']['recall'] == baseline_metrics['recall']
    assert status_val['before_metrics']['p95_latency_ms'] == baseline_metrics['p95_latency_ms']
    assert status_val['after_metrics']['recall'] == b2['metrics']['recall']
    assert status_val['after_metrics']['p95_latency_ms'] == b2['metrics']['p95_latency_ms']

    # Step 4: Next recommendation (ef144) -> Apply -> Run benchmark -> Experiment 03
    rec2 = client.post('/tune/recommend', json={'priority': 'recall', 'target_recall': 0.99}).json()['recommendation']
    client.post('/tune/apply', json={'recommendation': rec2})
    b3 = client.post('/benchmark/run').json()
    status_val3 = client.get('/tune/status').json()['last_validation']
    print('Exp 03 Validation summary text:', status_val3['reason'])
    print('Exp 03 Validation before recall:', status_val3['before_metrics']['recall'], 'p95:', status_val3['before_metrics']['p95_latency_ms'])
    print('Exp 03 Validation after recall:', status_val3['after_metrics']['recall'], 'p95:', status_val3['after_metrics']['p95_latency_ms'])

    assert status_val3['before_metrics']['recall'] == baseline_metrics['recall'] # MUST BE BASELINE!
    assert status_val3['before_metrics']['p95_latency_ms'] == baseline_metrics['p95_latency_ms']
    assert status_val3['after_metrics']['recall'] == b3['metrics']['recall']
    assert status_val3['after_metrics']['p95_latency_ms'] == b3['metrics']['p95_latency_ms']

    print('ALL PHASE 9 VALIDATION SUMMARY MATCHING TESTS PASSED!')
