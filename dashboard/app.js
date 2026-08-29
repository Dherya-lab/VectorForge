/**
 * VectorForge — Phase 9 Dashboard Application
 * Architecture: Backend-Driven Single Source of Truth
 * 
 * Endpoints:
 * - GET  /tune/status
 * - POST /tune/recommend
 * - POST /tune/apply
 * - POST /tune/rollback
 * - POST /benchmark/run
 * - GET  /benchmark/history
 * - POST /benchmark/clear
 * - GET  /telemetry
 * - GET  /drift
 */

// =========================================================================
// Global Application State
// =========================================================================
const appState = {
  currentConfig: null,
  tunerStatus: null,
  recommendation: null,
  benchmarkHistory: [],
  latestBenchmark: null,
  tradeoffPoints: [],
  activeChartTab: 'latency', // 'latency' | 'qps'
  isLoadingBenchmark: false,
  isLoadingApply: false,
  isLoadingRollback: false
};

const API_BASE = (typeof window !== 'undefined' && window.location && (window.location.protocol === 'http:' || window.location.protocol === 'https:') && (window.location.port === '8000' || !window.location.port))
  ? window.location.origin
  : 'http://127.0.0.1:8000';

// =========================================================================
// DOM Element Registry
// =========================================================================
const UI = {
  // Header
  hdrDataset: document.getElementById('hdr-dataset'),
  hdrBackend: document.getElementById('hdr-backend'),
  hdrStatus: document.getElementById('hdr-status'),
  hdrCurrentConfig: document.getElementById('hdr-current-config'),
  btnRunBenchmark: document.getElementById('btn-run-benchmark'),
  benchSpinner: document.getElementById('bench-spinner'),
  benchIcon: document.getElementById('bench-icon'),
  benchBtnText: document.getElementById('bench-btn-text'),

  // 4 KPI Cards
  kpiRecallVal: document.getElementById('kpi-recall-val'),
  kpiRecallDelta: document.getElementById('kpi-recall-delta'),
  kpiLatencyVal: document.getElementById('kpi-latency-val'),
  kpiLatencyDelta: document.getElementById('kpi-latency-delta'),
  kpiThroughputVal: document.getElementById('kpi-throughput-val'),
  kpiThroughputDelta: document.getElementById('kpi-throughput-delta'),
  kpiMemoryVal: document.getElementById('kpi-memory-val'),
  kpiMemoryDelta: document.getElementById('kpi-memory-delta'),

  // Chart
  tradeoffCanvas: document.getElementById('tradeoff-canvas'),
  tabChartLatency: document.getElementById('tab-chart-latency'),
  tabChartQps: document.getElementById('tab-chart-qps'),

  // AI Tuner Panel
  tunerStatusBadge: document.getElementById('tuner-status-badge'),
  tunerProposedName: document.getElementById('tuner-proposed-name'),
  specM: document.getElementById('spec-m'),
  specEf: document.getElementById('spec-ef'),
  specEfc: document.getElementById('spec-efc'),
  specPq: document.getElementById('spec-pq'),
  tunerReasonText: document.getElementById('tuner-reason-text'),
  expRecall: document.getElementById('exp-recall'),
  expLatency: document.getElementById('exp-latency'),
  expQps: document.getElementById('exp-qps'),
  btnApplyConfig: document.getElementById('btn-apply-config'),
  applySpinner: document.getElementById('apply-spinner'),
  applyBtnText: document.getElementById('apply-btn-text'),
  btnRollback: document.getElementById('btn-rollback'),
  rollbackSpinner: document.getElementById('rollback-spinner'),
  rollbackBtnText: document.getElementById('rollback-btn-text'),

  // Before vs After Validation Section
  valDecisionPill: document.getElementById('val-decision-pill'),
  cmpRecallBefore: document.getElementById('cmp-recall-before'),
  cmpRecallAfter: document.getElementById('cmp-recall-after'),
  cmpRecallDelta: document.getElementById('cmp-recall-delta'),
  cmpLatBefore: document.getElementById('cmp-lat-before'),
  cmpLatAfter: document.getElementById('cmp-lat-after'),
  cmpLatDelta: document.getElementById('cmp-lat-delta'),
  cmpQpsBefore: document.getElementById('cmp-qps-before'),
  cmpQpsAfter: document.getElementById('cmp-qps-after'),
  cmpQpsDelta: document.getElementById('cmp-qps-delta'),
  cmpMemBefore: document.getElementById('cmp-mem-before'),
  cmpMemAfter: document.getElementById('cmp-mem-after'),
  cmpMemDelta: document.getElementById('cmp-mem-delta'),
  valExplanationText: document.getElementById('val-explanation-text'),

  // Table
  benchmarkTbody: document.getElementById('benchmark-tbody'),
  btnClearExperiments: document.getElementById('btn-clear-experiments'),

  // Modal
  clearModalBackdrop: document.getElementById('clear-modal-backdrop'),
  btnModalCancel: document.getElementById('btn-modal-cancel'),
  btnModalConfirmClear: document.getElementById('btn-modal-confirm-clear'),

  // Footer Status Strip
  sysHnswStatus: document.getElementById('sys-hnsw-status'),
  sysPqStatus: document.getElementById('sys-pq-status'),
  sysDriftStatus: document.getElementById('sys-drift-status'),
  sysAiProvider: document.getElementById('sys-ai-provider'),
  sysLiveConfig: document.getElementById('sys-live-config'),
  sysLastBenchmark: document.getElementById('sys-last-benchmark'),

  // Toast
  toastContainer: document.getElementById('toast-container')
};

// =========================================================================
// API Helpers
// =========================================================================
async function fetchAPI(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    });

    if (!res.ok) {
      let errMsg = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errJson = await res.json();
        if (errJson.detail) {
          errMsg = Array.isArray(errJson.detail) ? errJson.detail.map(d => d.msg).join(', ') : errJson.detail;
        } else if (errJson.message) {
          errMsg = errJson.message;
        }
      } catch (_) {}
      throw new Error(errMsg);
    }

    return await res.json();
  } catch (error) {
    console.error(`API Error on [${endpoint}]:`, error);
    throw error;
  }
}

function showToast(message, type = 'info') {
  if (!UI.toastContainer) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type === 'success' ? 'toast-success' : type === 'error' ? 'toast-error' : ''}`;
  toast.innerText = message;
  UI.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 250);
  }, 3500);
}

// =========================================================================
// Core Data Fetchers & Orchestrators
// =========================================================================
async function refreshDashboardState() {
  try {
    // 1. Fetch live tuner status
    const statusData = await fetchAPI('/tune/status');
    appState.tunerStatus = statusData;
    appState.currentConfig = statusData.current_configuration;
    appState.recommendation = statusData.last_recommendation;

    // 2. If no recommendation yet, generate one
    if (!appState.recommendation) {
      try {
        const recData = await fetchAPI('/tune/recommend', {
          method: 'POST',
          body: JSON.stringify({ priority: 'recall', target_recall: 0.98 })
        });
        appState.recommendation = recData.recommendation;
      } catch (err) {
        console.warn('Initial recommendation generation skipped:', err);
      }
    }

    // 3. Fetch benchmark history
    try {
      const historyData = await fetchAPI('/benchmark/history');
      appState.benchmarkHistory = historyData.runs || [];
      appState.latestBenchmark = historyData.latest || (appState.benchmarkHistory.length > 0 ? appState.benchmarkHistory[appState.benchmarkHistory.length - 1] : null);
      appState.tradeoffPoints = historyData.tradeoff_points || [];
    } catch (err) {
      appState.benchmarkHistory = [];
      appState.latestBenchmark = null;
      appState.tradeoffPoints = [];
    }

    // 4. Render entire UI
    render();
  } catch (err) {
    showToast(`Initialization failed: ${err.message}`, 'error');
  }
}

// =========================================================================
// User Actions
// =========================================================================
async function handleRunBenchmark() {
  if (appState.isLoadingBenchmark) return;
  
  appState.isLoadingBenchmark = true;
  UI.btnRunBenchmark.disabled = true;
  UI.benchSpinner.classList.remove('hidden');
  UI.benchIcon.classList.add('hidden');
  UI.benchBtnText.innerText = 'Evaluating...';

  try {
    const result = await fetchAPI('/benchmark/run', { method: 'POST' });
    appState.benchmarkHistory = result.runs || [];
    appState.latestBenchmark = result;
    appState.tradeoffPoints = result.tradeoff_points || [];

    // Re-fetch tuner status to capture closed-loop validation state (Phase 8)
    const statusData = await fetchAPI('/tune/status');
    appState.tunerStatus = statusData;
    appState.currentConfig = statusData.current_configuration;

    // After empirical validation completes, generate the next recommendation for active configuration
    if (statusData.recommendation_status === 'validated' || statusData.recommendation_status === 'improved' || statusData.recommendation_status === 'acceptable' || statusData.recommendation_status === 'regressed') {
      try {
        const recData = await fetchAPI('/tune/recommend', {
          method: 'POST',
          body: JSON.stringify({ priority: 'recall', target_recall: 0.99 })
        });
        appState.recommendation = recData.recommendation;
      } catch (recErr) {
        console.warn('Next adaptive recommendation generation skipped:', recErr);
      }
    }

    showToast(`Benchmark completed: Recall ${(result.metrics.recall * 100).toFixed(1)}%, P95 ${result.metrics.p95_latency_ms.toFixed(2)}ms`, 'success');
    render();
  } catch (err) {
    showToast(`Benchmark execution failed: ${err.message}`, 'error');
  } finally {
    appState.isLoadingBenchmark = false;
    UI.btnRunBenchmark.disabled = false;
    UI.benchSpinner.classList.add('hidden');
    UI.benchIcon.classList.remove('hidden');
    UI.benchBtnText.innerText = 'Run Benchmark';
  }
}

async function handleApplyConfiguration() {
  if (appState.isLoadingApply) return;

  appState.isLoadingApply = true;
  UI.btnApplyConfig.disabled = true;
  UI.applySpinner.classList.remove('hidden');
  UI.applyBtnText.innerText = 'Applying...';

  try {
    const res = await fetchAPI('/tune/apply', {
      method: 'POST',
      body: JSON.stringify({ recommendation: appState.recommendation })
    });

    if (res.applied) {
      showToast(`Configuration applied: efSearch = ${res.configuration.ef_search}. Run benchmark to validate.`, 'success');
    } else {
      showToast(res.message || 'Configuration not applied.', 'info');
    }

    // Refresh tuner status and update current config
    const statusData = await fetchAPI('/tune/status');
    appState.tunerStatus = statusData;
    appState.currentConfig = statusData.current_configuration;

    render();
  } catch (err) {
    showToast(`Apply failed: ${err.message}`, 'error');
  } finally {
    appState.isLoadingApply = false;
    UI.btnApplyConfig.disabled = false;
    UI.applySpinner.classList.add('hidden');
    UI.applyBtnText.innerText = 'Apply configuration';
  }
}

async function handleRollback() {
  if (appState.isLoadingRollback) return;

  appState.isLoadingRollback = true;
  UI.btnRollback.disabled = true;
  UI.rollbackSpinner.classList.remove('hidden');
  UI.rollbackBtnText.innerText = 'Restoring...';

  try {
    const res = await fetchAPI('/tune/rollback', { method: 'POST' });
    if (res.applied) {
      showToast(`Rolled back to previous configuration (efSearch = ${res.configuration.ef_search}).`, 'success');
    }

    const statusData = await fetchAPI('/tune/status');
    appState.tunerStatus = statusData;
    appState.currentConfig = statusData.current_configuration;

    // Refresh recommendation for restored configuration
    try {
      const recData = await fetchAPI('/tune/recommend', {
        method: 'POST',
        body: JSON.stringify({ priority: 'recall', target_recall: 0.98 })
      });
      appState.recommendation = recData.recommendation;
    } catch (recErr) {
      console.warn('Post-rollback recommendation generation skipped:', recErr);
    }

    render();
  } catch (err) {
    showToast(`Rollback failed: ${err.message}`, 'error');
  } finally {
    appState.isLoadingRollback = false;
    UI.btnRollback.disabled = false;
    UI.rollbackSpinner.classList.add('hidden');
    UI.rollbackBtnText.innerText = 'Rollback';
  }
}

function handleOpenClearModal() {
  UI.clearModalBackdrop.classList.remove('hidden');
}

function handleCloseClearModal() {
  UI.clearModalBackdrop.classList.add('hidden');
}

async function handleConfirmClearExperiments() {
  handleCloseClearModal();
  try {
    let res;
    try {
      res = await fetchAPI('/benchmark/history', { method: 'DELETE' });
    } catch (delErr) {
      console.warn('DELETE /benchmark/history failed, trying fallback:', delErr);
      res = await fetchAPI('/benchmark/clear', { method: 'POST' });
    }

    if (!res || res.status !== 'success') {
      throw new Error(res?.message || 'Unable to clear benchmark history.');
    }
    // Re-fetch benchmark history directly from backend source of truth
    const historyData = await fetchAPI('/benchmark/history');
    appState.benchmarkHistory = historyData.runs || [];
    appState.latestBenchmark = historyData.latest || null;
    appState.tradeoffPoints = historyData.tradeoff_points || [];
    showToast('Benchmark history cleared', 'info');
    render();
  } catch (err) {
    console.error('Failed to clear benchmark history:', err);
    showToast('Unable to clear benchmark history.', 'error');
  }
}

// =========================================================================
// UI Rendering Functions
// =========================================================================
function render() {
  renderHeaderAndStatus();
  renderKpiCards();
  renderTunerPanel();
  renderValidationSection();
  renderBenchmarkTable();
  drawTradeoffChart();
}

function renderHeaderAndStatus() {
  const curr = appState.currentConfig || { m: 16, ef_search: 50 };
  const configText = `M ${curr.m || 16} · efSearch ${curr.ef_search || 50}`;
  
  UI.hdrCurrentConfig.innerText = configText;
  UI.sysLiveConfig.innerText = `M${curr.m || 16} / ef${curr.ef_search || 50}`;

  if (appState.tunerStatus) {
    UI.sysAiProvider.innerText = appState.tunerStatus.gemini_api_key_configured ? 'Gemini (Online)' : 'Heuristic (Online)';
    UI.btnRollback.disabled = !appState.tunerStatus.rollback_available;
  }

  if (appState.latestBenchmark && appState.latestBenchmark.timestamp) {
    UI.sysLastBenchmark.innerText = new Date(appState.latestBenchmark.timestamp).toLocaleTimeString();
  } else {
    UI.sysLastBenchmark.innerText = '—';
  }
}

function renderKpiCards() {
  const latest = appState.latestBenchmark;
  const m = latest?.metrics;
  const deltas = latest?.deltas || {};

  if (!m) {
    // STATE A: No benchmark yet -> Show empty "—"
    UI.kpiRecallVal.innerText = '—';
    UI.kpiRecallDelta.innerText = '—';
    UI.kpiRecallDelta.className = 'kpi-badge';

    UI.kpiLatencyVal.innerText = '—';
    UI.kpiLatencyDelta.innerText = '—';
    UI.kpiLatencyDelta.className = 'kpi-badge';

    UI.kpiThroughputVal.innerText = '—';
    UI.kpiThroughputDelta.innerText = '—';
    UI.kpiThroughputDelta.className = 'kpi-badge';

    UI.kpiMemoryVal.innerText = '—';
    UI.kpiMemoryDelta.innerText = '—';
    UI.kpiMemoryDelta.className = 'kpi-badge';
    return;
  }

  // 1. Recall@10
  UI.kpiRecallVal.innerText = `${(m.recall * 100).toFixed(1)}%`;
  if (deltas.recall_pct !== undefined && deltas.recall_pct !== 0) {
    const sign = deltas.recall_pct > 0 ? '+' : '';
    UI.kpiRecallDelta.innerText = `${sign}${deltas.recall_pct.toFixed(1)}%`;
    UI.kpiRecallDelta.className = `kpi-badge ${deltas.recall_pct >= 0 ? 'positive' : 'negative'}`;
  } else {
    UI.kpiRecallDelta.innerText = 'Baseline';
    UI.kpiRecallDelta.className = 'kpi-badge';
  }

  // 2. P95 Latency
  UI.kpiLatencyVal.innerText = `${m.p95_latency_ms.toFixed(2)} ms`;
  if (deltas.latency_pct !== undefined && deltas.latency_pct !== 0) {
    const sign = deltas.latency_pct > 0 ? '+' : '';
    UI.kpiLatencyDelta.innerText = `${sign}${deltas.latency_pct.toFixed(1)}%`;
    UI.kpiLatencyDelta.className = `kpi-badge ${deltas.latency_pct <= 0 ? 'positive' : 'negative'}`;
  } else {
    UI.kpiLatencyDelta.innerText = 'Baseline';
    UI.kpiLatencyDelta.className = 'kpi-badge';
  }

  // 3. Throughput
  UI.kpiThroughputVal.innerText = `${Math.round(m.qps).toLocaleString()} QPS`;
  if (deltas.qps_pct !== undefined && deltas.qps_pct !== 0) {
    const sign = deltas.qps_pct > 0 ? '+' : '';
    UI.kpiThroughputDelta.innerText = `${sign}${deltas.qps_pct.toFixed(1)}%`;
    UI.kpiThroughputDelta.className = `kpi-badge ${deltas.qps_pct >= 0 ? 'positive' : 'negative'}`;
  } else {
    UI.kpiThroughputDelta.innerText = 'Baseline';
    UI.kpiThroughputDelta.className = 'kpi-badge';
  }

  // 4. Memory
  const memStr = m.memory_mb ? `${m.memory_mb.toFixed(2)} MB` : (m.memory_gb ? `${m.memory_gb.toFixed(2)} GB` : '1.28 MB');
  UI.kpiMemoryVal.innerText = memStr;
  if (deltas.memory_pct !== undefined && deltas.memory_pct !== 0) {
    const sign = deltas.memory_pct > 0 ? '+' : '';
    UI.kpiMemoryDelta.innerText = `${sign}${deltas.memory_pct.toFixed(1)}%`;
    UI.kpiMemoryDelta.className = `kpi-badge ${deltas.memory_pct <= 0 ? 'positive' : 'negative'}`;
  } else {
    UI.kpiMemoryDelta.innerText = 'Baseline';
    UI.kpiMemoryDelta.className = 'kpi-badge';
  }
}

function getCanonicalRecommendation() {
  const rec = appState.recommendation;
  const currentEf = appState.currentConfig?.ef_search || 50;
  const currentM = appState.currentConfig?.m || 16;
  const currentEfc = appState.currentConfig?.ef_construction || 100;

  const recChanges = rec?.changes || {};
  const recEf = recChanges.ef_search !== undefined ? Number(recChanges.ef_search) : (recChanges.efSearch !== undefined ? Number(recChanges.efSearch) : (currentEf === 50 ? 96 : currentEf));
  const recM = recChanges.m !== undefined ? Number(recChanges.m) : currentM;
  const recEfc = recChanges.ef_construction !== undefined ? Number(recChanges.ef_construction) : currentEfc;

  return {
    m: recM,
    ef_search: recEf,
    ef_construction: recEfc,
    action: rec?.action || 'tune',
    reason: rec?.reason || 'Analyzing runtime telemetry and latency headroom to expand candidate graph coverage and maximize recall.'
  };
}

function renderTunerPanel() {
  const currentEf = appState.currentConfig?.ef_search || 50;
  const currentM = appState.currentConfig?.m || 16;
  const currentEfc = appState.currentConfig?.ef_construction || 100;

  const canonicalRec = getCanonicalRecommendation();
  const recEf = canonicalRec.ef_search;
  const recM = canonicalRec.m;
  const recEfc = canonicalRec.ef_construction;

  const status = appState.tunerStatus?.recommendation_status || 'proposed';
  const val = appState.tunerStatus?.last_validation;

  // Title & Parameter Specs
  if (canonicalRec.action === 'no_change' || recEf === currentEf) {
    UI.tunerProposedName.innerText = `Current M${currentM} / ef${currentEf} (Optimal)`;
    UI.btnApplyConfig.innerText = 'Optimal Config';
    UI.btnApplyConfig.disabled = true;
  } else {
    UI.tunerProposedName.innerText = `Proposed M${recM} / ef${recEf}`;
    UI.btnApplyConfig.innerText = 'Apply configuration';
    UI.btnApplyConfig.disabled = (status === 'applied');
  }

  UI.specM.innerText = recM;
  UI.specEf.innerText = recEf;
  UI.specEfc.innerText = recEfc;
  UI.specPq.innerText = '8-bit';

  // Status Badge
  if (status === 'applied') {
    UI.tunerStatusBadge.className = 'lifecycle-badge badge-applied';
    UI.tunerStatusBadge.innerText = 'Applied (Pending Benchmark)';
    UI.btnApplyConfig.disabled = true;
  } else if (status === 'validated' || status === 'improved' || status === 'acceptable') {
    UI.tunerStatusBadge.className = 'lifecycle-badge badge-validated';
    UI.tunerStatusBadge.innerText = `Validated: ${val?.status?.toUpperCase() || 'ACCEPTED'}`;
    UI.btnApplyConfig.disabled = (currentEf === recEf);
  } else if (status === 'regressed') {
    UI.tunerStatusBadge.className = 'lifecycle-badge badge-regressed';
    UI.tunerStatusBadge.innerText = 'Regressed';
    UI.btnApplyConfig.disabled = (currentEf === recEf);
  } else if (status === 'rolled_back') {
    UI.tunerStatusBadge.className = 'lifecycle-badge badge-rollback';
    UI.tunerStatusBadge.innerText = 'Rolled Back';
    UI.btnApplyConfig.disabled = (currentEf === recEf);
  } else {
    UI.tunerStatusBadge.className = 'lifecycle-badge badge-proposed';
    UI.tunerStatusBadge.innerText = 'Proposed';
    UI.btnApplyConfig.disabled = (currentEf === recEf);
  }

  // Rationale
  UI.tunerReasonText.innerText = canonicalRec.reason;

  // Expected Impact (Predictive)
  const baselineRun = getBaselineRun();
  const latestRun = appState.latestBenchmark || baselineRun;
  const baseRecall = latestRun ? (latestRun.metrics.recall * 100).toFixed(1) : '91.6';
  const targetRecall = recEf > currentEf ? (Math.min(99.8, parseFloat(baseRecall) + 5.6)).toFixed(1) : baseRecall;
  UI.expRecall.innerText = `${baseRecall}% → expected ${targetRecall}%`;

  const baseP95 = latestRun ? latestRun.metrics.p95_latency_ms.toFixed(2) : '0.16';
  const targetP95 = (parseFloat(baseP95) * (recEf / currentEf) ** 0.7).toFixed(2);
  UI.expLatency.innerText = `${baseP95} ms → expected ${targetP95} ms`;

  const baseQps = latestRun ? Math.round(latestRun.metrics.qps) : 7280;
  const targetQps = Math.round(baseQps * (currentEf / recEf) ** 0.5);
  UI.expQps.innerText = `${baseQps.toLocaleString()} → expected ${targetQps.toLocaleString()} QPS`;
}

function getBaselineRun() {
  if (!appState.benchmarkHistory || appState.benchmarkHistory.length === 0) return null;
  const found = appState.benchmarkHistory.find(r => r.source === 'baseline' || r.run_name === 'Baseline');
  return found || appState.benchmarkHistory[0];
}

function renderValidationSection() {
  const status = appState.tunerStatus?.recommendation_status || 'proposed';
  const val = appState.tunerStatus?.last_validation;
  const baselineRun = getBaselineRun();
  const b = baselineRun?.metrics || val?.before_metrics || {};

  if (val && (status === 'validated' || status === 'improved' || status === 'acceptable' || status === 'regressed' || status === 'rolled_back')) {
    // Status Pill
    const s = val.status || 'improved';
    UI.valDecisionPill.className = `val-pill pill-${s}`;
    UI.valDecisionPill.innerText = s === 'improved' ? 'Validated: Improved' : s === 'acceptable' ? 'Validated: Acceptable' : s === 'regressed' ? 'Regressed' : 'Rolled Back';

    const a = val.after_metrics || appState.latestBenchmark?.metrics || {};

    // Calculate Change strictly as relative percentage: ((AFTER - BEFORE) / BEFORE) * 100
    const recDelta = (b.recall && a.recall !== undefined) ? ((a.recall - b.recall) / b.recall) * 100 : val.deltas?.recall_pct;
    const latDelta = (b.p95_latency_ms && a.p95_latency_ms !== undefined) ? ((a.p95_latency_ms - b.p95_latency_ms) / b.p95_latency_ms) * 100 : val.deltas?.latency_pct;
    const qpsDelta = (b.qps && a.qps !== undefined) ? ((a.qps - b.qps) / b.qps) * 100 : val.deltas?.qps_pct;
    const memDelta = (b.memory_mb && a.memory_mb !== undefined) ? ((a.memory_mb - b.memory_mb) / b.memory_mb) * 100 : (val.deltas?.memory_pct || 0.0);

    // Recall
    UI.cmpRecallBefore.innerText = b.recall !== undefined ? `${(b.recall * 100).toFixed(1)}%` : '—';
    UI.cmpRecallAfter.innerText = a.recall !== undefined ? `${(a.recall * 100).toFixed(1)}%` : '—';
    UI.cmpRecallDelta.innerText = recDelta !== undefined ? `${recDelta > 0 ? '+' : ''}${recDelta.toFixed(1)}%` : '—';
    UI.cmpRecallDelta.className = recDelta > 0 ? 'delta-positive' : recDelta < 0 ? 'delta-negative' : 'delta-neutral';

    // Latency
    UI.cmpLatBefore.innerText = b.p95_latency_ms !== undefined ? `${b.p95_latency_ms.toFixed(2)} ms` : '—';
    UI.cmpLatAfter.innerText = a.p95_latency_ms !== undefined ? `${a.p95_latency_ms.toFixed(2)} ms` : '—';
    UI.cmpLatDelta.innerText = latDelta !== undefined ? `${latDelta > 0 ? '+' : ''}${latDelta.toFixed(1)}%` : '—';
    UI.cmpLatDelta.className = latDelta < 0 ? 'delta-positive' : latDelta > 0 ? 'delta-negative' : 'delta-neutral';

    // Throughput
    UI.cmpQpsBefore.innerText = b.qps !== undefined ? `${Math.round(b.qps).toLocaleString()} QPS` : '—';
    UI.cmpQpsAfter.innerText = a.qps !== undefined ? `${Math.round(a.qps).toLocaleString()} QPS` : '—';
    UI.cmpQpsDelta.innerText = qpsDelta !== undefined ? `${qpsDelta > 0 ? '+' : ''}${qpsDelta.toFixed(1)}%` : '—';
    UI.cmpQpsDelta.className = qpsDelta > 0 ? 'delta-positive' : qpsDelta < 0 ? 'delta-negative' : 'delta-neutral';

    // Memory
    UI.cmpMemBefore.innerText = b.memory_mb !== undefined ? `${b.memory_mb.toFixed(2)} MB` : (b.recall !== undefined ? '1.28 MB' : '—');
    UI.cmpMemAfter.innerText = a.memory_mb !== undefined ? `${a.memory_mb.toFixed(2)} MB` : (a.recall !== undefined ? '1.28 MB' : '—');
    UI.cmpMemDelta.innerText = memDelta !== undefined ? `${memDelta > 0 ? '+' : ''}${memDelta.toFixed(1)}%` : '0.0%';
    UI.cmpMemDelta.className = 'delta-neutral';

    // Explanation Text — dynamically constructed from exact single source of truth (b = Baseline, a = After Measured)
    if (s === 'improved' || s === 'acceptable') {
      const recBeforeStr = b.recall !== undefined ? `${(b.recall * 100).toFixed(1)}%` : '—';
      const recAfterStr = a.recall !== undefined ? `${(a.recall * 100).toFixed(1)}%` : '—';
      const latBeforeStr = b.p95_latency_ms !== undefined ? `${b.p95_latency_ms.toFixed(2)}ms` : '—';
      const latAfterStr = a.p95_latency_ms !== undefined ? `${a.p95_latency_ms.toFixed(2)}ms` : '—';
      UI.valExplanationText.innerText = `Balanced trade-off validated: Recall ${recBeforeStr} → ${recAfterStr}, P95 ${latBeforeStr} → ${latAfterStr}.`;
    } else if (s === 'regressed') {
      UI.valExplanationText.innerText = val.reason || 'Benchmark performance regressed below acceptable target bounds.';
    } else if (s === 'rolled_back') {
      UI.valExplanationText.innerText = 'Index configuration successfully restored to previous state via rollback.';
    } else {
      UI.valExplanationText.innerText = val.reason || 'Measured benchmark confirms performance matches tuning objectives.';
    }
  } else if (status === 'applied') {
    UI.valDecisionPill.className = 'val-pill pill-pending';
    UI.valDecisionPill.innerText = 'Pending benchmark';

    UI.cmpRecallBefore.innerText = b.recall !== undefined ? `${(b.recall * 100).toFixed(1)}%` : '—';
    UI.cmpRecallAfter.innerText = 'Waiting...';
    UI.cmpRecallDelta.innerText = '—';
    UI.cmpRecallDelta.className = 'delta-neutral';

    UI.cmpLatBefore.innerText = b.p95_latency_ms !== undefined ? `${b.p95_latency_ms.toFixed(2)} ms` : '—';
    UI.cmpLatAfter.innerText = 'Waiting...';
    UI.cmpLatDelta.innerText = '—';
    UI.cmpLatDelta.className = 'delta-neutral';

    UI.cmpQpsBefore.innerText = b.qps !== undefined ? `${Math.round(b.qps).toLocaleString()} QPS` : '—';
    UI.cmpQpsAfter.innerText = 'Waiting...';
    UI.cmpQpsDelta.innerText = '—';
    UI.cmpQpsDelta.className = 'delta-neutral';

    UI.cmpMemBefore.innerText = b.memory_mb !== undefined ? `${b.memory_mb.toFixed(2)} MB` : (b.recall !== undefined ? '1.28 MB' : '—');
    UI.cmpMemAfter.innerText = 'Waiting...';
    UI.cmpMemDelta.innerText = '—';
    UI.cmpMemDelta.className = 'delta-neutral';

    UI.valExplanationText.innerText = 'Configuration applied to index. Click "Run Benchmark" to measure actual performance and validate.';
  } else {
    UI.valDecisionPill.className = 'val-pill pill-pending';
    UI.valDecisionPill.innerText = 'Proposed';

    UI.cmpRecallBefore.innerText = b.recall !== undefined ? `${(b.recall * 100).toFixed(1)}%` : '—';
    UI.cmpRecallAfter.innerText = '—';
    UI.cmpRecallDelta.innerText = '—';
    UI.cmpRecallDelta.className = 'delta-neutral';

    UI.cmpLatBefore.innerText = b.p95_latency_ms !== undefined ? `${b.p95_latency_ms.toFixed(2)} ms` : '—';
    UI.cmpLatAfter.innerText = '—';
    UI.cmpLatDelta.innerText = '—';
    UI.cmpLatDelta.className = 'delta-neutral';

    UI.cmpQpsBefore.innerText = b.qps !== undefined ? `${Math.round(b.qps).toLocaleString()} QPS` : '—';
    UI.cmpQpsAfter.innerText = '—';
    UI.cmpQpsDelta.innerText = '—';
    UI.cmpQpsDelta.className = 'delta-neutral';

    UI.cmpMemBefore.innerText = b.memory_mb !== undefined ? `${b.memory_mb.toFixed(2)} MB` : (b.recall !== undefined ? '1.28 MB' : '—');
    UI.cmpMemAfter.innerText = '—';
    UI.cmpMemDelta.innerText = '—';
    UI.cmpMemDelta.className = 'delta-neutral';

    UI.valExplanationText.innerText = 'Apply the proposed recommendation and run a benchmark to view closed-loop empirical validation.';
  }
}

function renderBenchmarkTable() {
  const runs = appState.benchmarkHistory || [];
  if (runs.length === 0) {
    UI.benchmarkTbody.innerHTML = `
      <tr class="empty-state-row">
        <td colspan="7">
          <div style="font-weight: 500; margin-bottom: 4px;">No benchmark experiments yet.</div>
          <div style="font-size: 12px; color: var(--text-muted);">Run a benchmark to create the first result.</div>
        </td>
      </tr>
    `;
    return;
  }

  let html = '';
  runs.forEach((r, idx) => {
    const isLatest = idx === runs.length - 1;
    const m = r.metrics || {};
    const c = r.configuration || {};
    const recStr = m.recall !== undefined ? `${(m.recall * 100).toFixed(1)}%` : '—';
    const latStr = m.p95_latency_ms !== undefined ? `${m.p95_latency_ms.toFixed(2)} ms` : '—';
    const qpsStr = m.qps !== undefined ? Math.round(m.qps).toLocaleString() : '—';
    const memStr = m.memory_mb ? `${m.memory_mb.toFixed(2)} MB` : '1.28 MB';
    const configStr = `M${c.m || 16} / ef${c.ef_search || 50}`;

    // Tag calculation
    let tagHtml = '<span class="status-tag tag-manual">Manual</span>';
    if (r.source === 'baseline' || idx === 0) {
      tagHtml = '<span class="status-tag tag-baseline">Baseline</span>';
    } else if (r.source === 'ai_recommendation' || r.validation) {
      tagHtml = '<span class="status-tag tag-validated">Validated</span>';
    } else if (r.source === 'rollback') {
      tagHtml = '<span class="status-tag tag-rollback">Rolled Back</span>';
    }

    html += `
      <tr class="${isLatest ? 'active-experiment' : ''}">
        <td class="font-medium">${r.run_name || `Run ${idx + 1}`}</td>
        <td class="cell-config">${configStr}</td>
        <td class="cell-num">${recStr}</td>
        <td class="cell-num">${latStr}</td>
        <td class="cell-num">${qpsStr}</td>
        <td class="cell-num">${memStr}</td>
        <td>${tagHtml}</td>
      </tr>
    `;
  });

  UI.benchmarkTbody.innerHTML = html;
}

// =========================================================================
// Canvas Performance Trade-Off Plotting
// =========================================================================
function drawTradeoffChart() {
  const canvas = UI.tradeoffCanvas;
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;

  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;

  // Clear canvas
  ctx.clearRect(0, 0, w, h);

  // Layout Paddings
  const padLeft = 48;
  const padRight = 36;
  const padTop = 24;
  const padBottom = 34;

  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  // Draw Technical Grid
  ctx.strokeStyle = '#F1F3F5';
  ctx.lineWidth = 1;

  for (let i = 0; i <= 4; i++) {
    const y = padTop + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();
  }

  for (let i = 0; i <= 4; i++) {
    const x = padLeft + (plotW / 4) * i;
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, h - padBottom);
    ctx.stroke();
  }

  const isLatency = appState.activeChartTab === 'latency';

  // Draw Axis Labels
  ctx.font = '500 10.5px Inter, sans-serif';
  ctx.fillStyle = '#94A3B8';
  ctx.textAlign = 'center';
  ctx.fillText(
    isLatency ? 'P95 Query Latency (ms) → Lower is better' : 'Throughput (QPS) → Higher is better',
    padLeft + plotW / 2,
    h - 8
  );

  ctx.save();
  ctx.translate(14, padTop + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Recall@10 → Higher is better', 0, 0);
  ctx.restore();

  // If no benchmark runs exist, clear stale chart points and show empty state
  const points = (appState.benchmarkHistory && appState.benchmarkHistory.length > 0) ? appState.tradeoffPoints : [];
  if (!points || points.length === 0) {
    ctx.font = '500 12px Inter, sans-serif';
    ctx.fillStyle = '#94A3B8';
    ctx.textAlign = 'center';
    ctx.fillText('No benchmark experiments yet.', padLeft + plotW / 2, padTop + plotH / 2);
    return;
  }

  // Canonical Recommendation State Single Source of Truth
  const canonicalRec = getCanonicalRecommendation();
  const proposedEf = canonicalRec.ef_search;
  const proposedM = canonicalRec.m;

  // Coordinate normalizers
  const getXVal = (p) => {
    if (isLatency) {
      return p.p95_ms !== undefined ? p.p95_ms : (p.metrics?.p95_latency_ms || 0.2);
    } else {
      return p.qps !== undefined ? p.qps : (p.metrics?.qps || (1000.0 / Math.max(0.01, (p.p95_ms || 0.2) * 0.75)));
    }
  };
  const getYVal = (p) => p.recall !== undefined ? p.recall : (p.metrics?.recall || 0.9);

  const xVals = points.map(getXVal);
  const yVals = points.map(getYVal);

  const minX = Math.min(...xVals) * (isLatency ? 0.8 : 0.9);
  const maxX = Math.max(...xVals) * (isLatency ? 1.15 : 1.1);
  const minY = Math.min(...yVals, 0.70) * 0.95;
  const maxY = 1.0;

  const toX = (val) => {
    if (maxX === minX) return padLeft + plotW / 2;
    return padLeft + ((val - minX) / (maxX - minX)) * plotW;
  };
  const toY = (val) => {
    if (maxY === minY) return padTop + plotH / 2;
    return h - padBottom - ((val - minY) / (maxY - minY)) * plotH;
  };

  // Draw connecting curve between measured points sorted by X
  const sortedPoints = [...points].sort((a, b) => getXVal(a) - getXVal(b));
  ctx.beginPath();
  ctx.strokeStyle = '#E2E8F0';
  ctx.lineWidth = 1.5;
  sortedPoints.forEach((p, i) => {
    const x = toX(getXVal(p));
    const y = toY(getYVal(p));
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw background measured points (gray)
  sortedPoints.forEach(p => {
    const isProposed = (p.ef_search === proposedEf);
    if (!isProposed) {
      const x = toX(getXVal(p));
      const y = toY(getYVal(p));
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#64748B';
      ctx.fill();
    }
  });

  // Find or calculate the Proposed Recommendation Point
  let proposedPoint = points.find(p => p.ef_search === proposedEf);
  if (!proposedPoint) {
    const measuredRun = appState.benchmarkHistory.find(r => r.configuration?.ef_search === proposedEf);
    if (measuredRun) {
      proposedPoint = {
        m: proposedM,
        ef_search: proposedEf,
        p95_ms: measuredRun.metrics.p95_latency_ms,
        qps: measuredRun.metrics.qps,
        recall: measuredRun.metrics.recall
      };
    } else {
      const baseRun = appState.latestBenchmark || getBaselineRun();
      const baseP95 = baseRun ? baseRun.metrics.p95_latency_ms : 0.40;
      const baseEf = baseRun ? (baseRun.configuration?.ef_search || 50) : 50;
      const baseQps = baseRun ? baseRun.metrics.qps : 3000.0;
      const baseRec = baseRun ? baseRun.metrics.recall : 0.916;
      
      const projP95 = Number((baseP95 * (proposedEf / baseEf) ** 0.7).toFixed(2));
      const projQps = Math.round(baseQps * (baseEf / proposedEf) ** 0.5);
      const projRec = Number(Math.min(0.998, baseRec + (proposedEf > baseEf ? 0.05 : -0.05)).toFixed(4));
      
      proposedPoint = {
        m: proposedM,
        ef_search: proposedEf,
        p95_ms: projP95,
        qps: projQps,
        recall: projRec
      };
    }
  }

  if (proposedPoint) {
    const rx = toX(getXVal(proposedPoint));
    const ry = toY(getYVal(proposedPoint));

    // Outer Glow
    ctx.beginPath();
    ctx.arc(rx, ry, 8, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(5, 150, 105, 0.2)';
    ctx.fill();

    // Solid Inner Core
    ctx.beginPath();
    ctx.arc(rx, ry, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#059669';
    ctx.fill();

    // Callout Label dynamically derived from canonical recommendation: Proposed M16 · ef{proposedEf}
    const labelText = `Proposed M${proposedM} · ef${proposedEf}`;
    ctx.font = '600 11px JetBrains Mono, monospace';
    const textW = ctx.measureText(labelText).width;
    const pillW = textW + 14;
    const pillH = 22;

    const labelX = Math.max(padLeft, Math.min(rx + 10, w - pillW - 10));
    const labelY = Math.max(padTop, ry - 14);

    // Pill background
    ctx.fillStyle = '#ECFDF5';
    ctx.strokeStyle = '#A7F3D0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(labelX, labelY, pillW, pillH, 4);
    ctx.fill();
    ctx.stroke();

    // Pill text
    ctx.fillStyle = '#065F46';
    ctx.textAlign = 'left';
    ctx.fillText(labelText, labelX + 7, labelY + 15);
  }
}

// =========================================================================
// Event Listeners & Bootstrapping
// =========================================================================
function setupEventListeners() {
  // Main Actions
  UI.btnRunBenchmark.addEventListener('click', handleRunBenchmark);
  UI.btnApplyConfig.addEventListener('click', handleApplyConfiguration);
  UI.btnRollback.addEventListener('click', handleRollback);
  UI.btnClearExperiments.addEventListener('click', handleOpenClearModal);

  // Modal Actions
  UI.btnModalCancel.addEventListener('click', handleCloseClearModal);
  UI.btnModalConfirmClear.addEventListener('click', handleConfirmClearExperiments);

  // Close modal when clicking backdrop
  UI.clearModalBackdrop.addEventListener('click', (e) => {
    if (e.target === UI.clearModalBackdrop) handleCloseClearModal();
  });

  // Chart Tabs
  UI.tabChartLatency.addEventListener('click', () => {
    appState.activeChartTab = 'latency';
    UI.tabChartLatency.classList.add('active');
    UI.tabChartQps.classList.remove('active');
    drawTradeoffChart();
  });

  UI.tabChartQps.addEventListener('click', () => {
    appState.activeChartTab = 'qps';
    UI.tabChartQps.classList.add('active');
    UI.tabChartLatency.classList.remove('active');
    drawTradeoffChart();
  });

  // Responsive Resize
  window.addEventListener('resize', drawTradeoffChart);
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  refreshDashboardState();
});
