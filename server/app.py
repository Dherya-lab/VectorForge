import os
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Literal
from fastapi import FastAPI, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
import numpy as np

from server import VectorForgeIndex, simd_info
from server.telemetry import TelemetryCollector
from server.drift import DriftDetector
from server.tuner import (
    VectorForgeTuner,
    TuningTarget,
    TuningChanges,
    TuningRecommendation,
    ExpectedEffect,
    PARAMETER_BOUNDS
)

app = FastAPI(
    title="VectorForge Control Plane",
    description="FastAPI control plane, telemetry subsystem, statistical drift detection, live benchmarks, and Gemini AI-assisted adaptive index tuning for VectorForge.",
    version="0.7.0"
)

# Enable CORS for local dashboard and dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/dashboard") or request.url.path == "/ui":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Global in-memory index & telemetry state
class GlobalState:
    index: Optional[VectorForgeIndex] = None
    dimension: int = 128
    backend: str = "hnsw"
    m: int = 16
    ef_construction: int = 100
    ef_search: int = 50

state = GlobalState()
telemetry_collector = TelemetryCollector()
query_drift_detector = DriftDetector(name="query_drift")
dataset_drift_detector = DriftDetector(name="dataset_drift")
ai_tuner = VectorForgeTuner()
benchmark_history: List[Dict[str, Any]] = []

def get_or_create_index(dim: Optional[int] = None) -> VectorForgeIndex:
    if dim is not None and dim != state.dimension:
        state.dimension = dim
        state.index = VectorForgeIndex(
            dimension=dim,
            backend=state.backend,
            m=state.m,
            ef_construction=state.ef_construction,
            ef_search=state.ef_search
        )
    elif state.index is None:
        state.index = VectorForgeIndex(
            dimension=state.dimension,
            backend=state.backend,
            m=state.m,
            ef_construction=state.ef_construction,
            ef_search=state.ef_search
        )
    return state.index

def get_current_config() -> Dict[str, Any]:
    return {
        "backend": state.backend,
        "ef_search": state.ef_search,
        "m": state.m,
        "ef_construction": state.ef_construction,
        "dimension": state.dimension
    }

def apply_config_changes(changes: Dict[str, Any]) -> Dict[str, Any]:
    idx = get_or_create_index()
    if "ef_search" in changes and changes["ef_search"] is not None:
        state.ef_search = int(changes["ef_search"])
        idx.tune(ef_search=state.ef_search)
    if "m" in changes and changes["m"] is not None:
        state.m = int(changes["m"])
    if "ef_construction" in changes and changes["ef_construction"] is not None:
        state.ef_construction = int(changes["ef_construction"])
    return get_current_config()

# =========================================================================
# Pydantic Request & Response Models
# =========================================================================

class InsertRequest(BaseModel):
    vectors: List[List[float]] = Field(
        ...,
        description="List of floating-point vectors (N x D) to insert into the index",
        examples=[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]
    )

class InsertResponse(BaseModel):
    inserted: int = Field(..., description="Number of vectors inserted in this operation")
    total_vectors: int = Field(..., description="Total number of vectors currently stored in the index")
    dimension: int = Field(..., description="Vector dimensionality D")

class QueryRequest(BaseModel):
    vector: List[float] = Field(
        ...,
        description="Query floating-point vector of dimension D",
        examples=[[0.1, 0.2, 0.3]]
    )
    k: int = Field(
        default=10,
        ge=1,
        le=10000,
        description="Number of nearest neighbors to retrieve"
    )

class SearchResultItem(BaseModel):
    id: int = Field(..., description="Unique integer ID of the matched vector")
    distance: float = Field(..., description="Euclidean distance from the query vector")

class QueryResponse(BaseModel):
    results: List[SearchResultItem] = Field(..., description="Nearest neighbors ordered by increasing distance")
    query_latency_us: float = Field(..., description="Query execution latency in microseconds")
    k: int = Field(..., description="Requested top-k parameter")

class TuneRequest(BaseModel):
    ef_search: Optional[int] = Field(None, ge=1, description="Candidate beam width during HNSW query search")
    m: Optional[int] = Field(None, ge=2, description="Maximum number of outgoing graph connections per node")
    ef_construction: Optional[int] = Field(None, ge=4, description="Candidate beam width during index construction")

class TuneResponse(BaseModel):
    ef_search: int
    m: int
    ef_construction: int
    backend: str
    message: str

class SystemTelemetryResponse(BaseModel):
    status: str
    total_vectors: int
    dimension: int
    backend: str
    simd: Dict[str, Any]
    hnsw_parameters: Dict[str, Any]

class LatencySummary(BaseModel):
    avg_ms: Optional[float]
    p50_ms: Optional[float]
    p95_ms: Optional[float]
    p99_ms: Optional[float]
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    sample_count: int = 0
    max_buffer_size: int = 10000

class TelemetrySummaryResponse(BaseModel):
    vectors: int
    dimension: int
    total_insert_operations: int
    total_query_operations: int
    query_count: int
    successful_queries: int
    failed_queries: int
    successful_inserts: int
    failed_inserts: int
    latency: LatencySummary
    backend: str
    simd: str
    simd_details: Optional[Dict[str, Any]] = None
    hnsw_parameters: Optional[Dict[str, Any]] = None
    pq_parameters: Optional[Dict[str, Any]] = None

class DriftChannelStatus(BaseModel):
    status: str
    score: float
    threshold: float
    severe_threshold: float
    sample_count: int
    timestamp: str
    baseline_statistics: Optional[Dict[str, Any]] = None
    current_statistics: Optional[Dict[str, Any]] = None

class DriftSummaryResponse(BaseModel):
    query_drift: DriftChannelStatus
    dataset_drift: DriftChannelStatus

class DriftResetRequest(BaseModel):
    channel: Optional[str] = Field("all", description="Target channel to reset: 'all', 'query', or 'dataset'")

class DriftResetResponse(BaseModel):
    status: str
    channel: str
    message: str
    query_drift: DriftChannelStatus
    dataset_drift: DriftChannelStatus

# Phase 7 AI Tuning Models
class RecommendResponse(BaseModel):
    status: str
    recommendation: TuningRecommendation
    current_configuration: Dict[str, Any]
    proposed_configuration: Dict[str, Any]
    rebuild_required: bool

class ApplyRequest(BaseModel):
    recommendation: Optional[TuningRecommendation] = Field(
        None,
        description="Recommendation to apply; if omitted, last generated recommendation is applied"
    )
    action: Optional[Literal["no_change", "tune", "rebuild_required"]] = None
    reason: Optional[str] = None
    changes: Optional[TuningChanges] = None
    expected_effect: Optional[ExpectedEffect] = None
    confidence: Optional[float] = None
    rebuild_required: Optional[bool] = None

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "recommendation": {
                    "action": "tune",
                    "reason": "Recall optimization with available latency headroom.",
                    "changes": {
                        "ef_search": 120
                    },
                    "expected_effect": {
                        "recall": "increase",
                        "latency": "increase",
                        "memory": "neutral"
                    },
                    "confidence": 0.85,
                    "rebuild_required": False
                }
            }
        }
    }

    def get_recommendation(self) -> Optional[TuningRecommendation]:
        if self.recommendation is not None:
            return self.recommendation
        if self.action is not None and self.reason is not None:
            return TuningRecommendation(
                action=self.action,
                reason=self.reason,
                changes=self.changes or TuningChanges(),
                expected_effect=self.expected_effect or ExpectedEffect(),
                confidence=self.confidence if self.confidence is not None else 1.0,
                rebuild_required=self.rebuild_required or False
            )
        return None

class ApplyResponse(BaseModel):
    status: str
    applied: bool
    rebuild_required: bool
    message: str
    previous_configuration: Optional[Dict[str, Any]] = None
    configuration: Dict[str, Any]

class RollbackResponse(BaseModel):
    status: str
    applied: bool
    message: str
    configuration: Optional[Dict[str, Any]] = None

class TunerStatusResponse(BaseModel):
    tuner_status: str
    ai_provider: str
    gemini_api_key_configured: bool
    model: str
    current_configuration: Dict[str, Any]
    last_recommendation: Optional[Dict[str, Any]] = None
    recommendation_status: Optional[str] = "idle"
    last_applied_change: Optional[Dict[str, Any]] = None
    last_validation: Optional[Dict[str, Any]] = None
    active_snapshot: Optional[Dict[str, Any]] = None
    tuning_history_count: int
    rollback_available: bool
    parameter_bounds: Dict[str, Any]

# =========================================================================
# REST Endpoints
# =========================================================================

@app.get("/", tags=["Health"])
def root():
    return {
        "engine": "VectorForge",
        "version": "0.7.0",
        "status": "online",
        "docs_url": "/docs",
        "dashboard_url": "/dashboard"
    }

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.post("/vectors/insert", response_model=InsertResponse, status_code=status.HTTP_201_CREATED, tags=["Vectors"])
def insert_vectors(payload: InsertRequest):
    """
    Inserts a batch of floating-point vectors into the active index.
    Updates dataset drift tracking and telemetry.
    """
    if not payload.vectors or len(payload.vectors) == 0:
        telemetry_collector.record_insert(0, 0, success=False)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'vectors' list cannot be empty."
        )

    first_len = len(payload.vectors[0])
    if first_len == 0:
        telemetry_collector.record_insert(0, 0, success=False)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vector dimensionality cannot be 0."
        )

    for idx, v in enumerate(payload.vectors):
        if len(v) != first_len:
            telemetry_collector.record_insert(0, first_len, success=False)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Inconsistent vector dimension at index {idx}: expected {first_len}, got {len(v)}."
            )

    idx = get_or_create_index(first_len)

    try:
        np_arr = np.array(payload.vectors, dtype=np.float32)
        count = idx.insert(np_arr)

        dataset_drift_detector.add_samples(np_arr)
        telemetry_collector.record_insert(count, idx.dimension(), success=True)

        return InsertResponse(
            inserted=count,
            total_vectors=idx.size(),
            dimension=idx.dimension()
        )
    except Exception as e:
        telemetry_collector.record_insert(0, idx.dimension(), success=False)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insertion failed: {str(e)}"
        )

@app.post("/vectors/query", response_model=QueryResponse, tags=["Vectors"])
def query_vectors(payload: QueryRequest):
    """
    Executes approximate k-NN nearest neighbor search against the index.
    Updates query drift tracking and latency telemetry.
    """
    idx = get_or_create_index()

    if idx.size() == 0:
        telemetry_collector.record_query(
            latency_ms=0.0,
            k=payload.k,
            backend=state.backend,
            dimension=idx.dimension(),
            result_count=0,
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Index is empty. Please insert vectors before querying."
        )

    if len(payload.vector) != idx.dimension():
        telemetry_collector.record_query(
            latency_ms=0.0,
            k=payload.k,
            backend=state.backend,
            dimension=len(payload.vector),
            result_count=0,
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query dimension mismatch: expected {idx.dimension()}, got {len(payload.vector)}."
        )

    try:
        np_query = np.array(payload.vector, dtype=np.float32)
        t0 = time.perf_counter_ns()
        raw_results = idx.search(np_query, payload.k)
        t1 = time.perf_counter_ns()

        latency_ns = t1 - t0
        latency_us = latency_ns / 1000.0
        latency_ms = latency_ns / 1_000_000.0

        formatted_results = [
            SearchResultItem(id=r["id"], distance=float(r["distance"]))
            for r in raw_results
        ]

        query_drift_detector.add_samples(np_query)
        telemetry_collector.record_query(
            latency_ms=latency_ms,
            k=payload.k,
            backend=state.backend,
            dimension=idx.dimension(),
            result_count=len(formatted_results),
            success=True
        )

        return QueryResponse(
            results=formatted_results,
            query_latency_us=latency_us,
            k=payload.k
        )
    except Exception as e:
        telemetry_collector.record_query(
            latency_ms=0.0,
            k=payload.k,
            backend=state.backend,
            dimension=idx.dimension(),
            result_count=0,
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query failed: {str(e)}"
        )

@app.get("/telemetry", response_model=TelemetrySummaryResponse, tags=["Telemetry"])
def get_telemetry():
    """
    Returns comprehensive search and index telemetry, operational counters, and latency percentiles.
    """
    idx = get_or_create_index()
    index_tele = idx.get_telemetry()
    return telemetry_collector.get_summary(index_telemetry=index_tele)

@app.get("/drift", response_model=DriftSummaryResponse, tags=["Drift Detection"])
def get_drift():
    """
    Returns current embedding drift status for both query and dataset workloads.
    """
    return {
        "query_drift": query_drift_detector.compute_drift(),
        "dataset_drift": dataset_drift_detector.compute_drift()
    }

@app.post("/drift/reset", response_model=DriftResetResponse, tags=["Drift Detection"])
def reset_drift(payload: Optional[DriftResetRequest] = None):
    """
    Manually resets the baseline distribution for query and/or dataset drift detectors.
    """
    channel = payload.channel.lower() if payload and payload.channel else "all"

    if channel in ["all", "query"]:
        query_drift_detector.reset()
    if channel in ["all", "dataset"]:
        dataset_drift_detector.reset()

    return DriftResetResponse(
        status="success",
        channel=channel,
        message=f"Drift baseline for '{channel}' reset successfully.",
        query_drift=query_drift_detector.compute_drift(),
        dataset_drift=dataset_drift_detector.compute_drift()
    )

# =========================================================================
# AI Tuning Endpoints (Phase 7)
# =========================================================================

@app.post("/tune/recommend", response_model=RecommendResponse, tags=["AI Tuning"])
def recommend_tuning(payload: Optional[TuningTarget] = None):
    """
    Asks the AI tuner to analyze current telemetry, drift signals, and search objectives
    to recommend optimized index parameters without changing live configuration.
    """
    target = payload or TuningTarget()
    idx = get_or_create_index()
    index_tele = idx.get_telemetry()
    tele_summary = telemetry_collector.get_summary(index_telemetry=index_tele)
    drift_summary = {
        "query_drift": query_drift_detector.compute_drift(),
        "dataset_drift": dataset_drift_detector.compute_drift()
    }
    curr_config = get_current_config()

    recommendation = ai_tuner.recommend(
        telemetry=tele_summary,
        drift=drift_summary,
        current_config=curr_config,
        target=target
    )

    proposed_config = dict(curr_config)
    changes_dict = recommendation.changes.to_dict() if hasattr(recommendation.changes, "to_dict") else recommendation.changes
    proposed_config.update(changes_dict)

    return RecommendResponse(
        status="success",
        recommendation=recommendation,
        current_configuration=curr_config,
        proposed_configuration=proposed_config,
        rebuild_required=recommendation.rebuild_required
    )

def get_baseline_run() -> Optional[Dict[str, Any]]:
    if not benchmark_history:
        return None
    for run in benchmark_history:
        if run.get("source") == "baseline" or run.get("run_name") == "Baseline":
            return run
    return benchmark_history[0]

@app.post("/tune/apply", response_model=ApplyResponse, tags=["AI Tuning"])
def apply_tuning(payload: Optional[ApplyRequest] = None):
    """
    Applies a validated tuning recommendation after rigorous safety checks.
    If the parameter requires index reconstruction, rejects live application safely.
    """
    rec = None
    if payload:
        rec = payload.get_recommendation()
    if rec is None:
        rec = ai_tuner.last_recommendation

    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No recommendation provided and no previous recommendation available to apply. Call /tune/recommend first."
        )

    curr_config = get_current_config()
    idx = get_or_create_index()
    tele_before = telemetry_collector.get_summary(index_telemetry=idx.get_telemetry())
    baseline_run = get_baseline_run()
    bench_before = baseline_run["metrics"] if baseline_run else None

    result = ai_tuner.apply(
        recommendation=rec,
        current_config=curr_config,
        telemetry_before=tele_before,
        apply_fn=apply_config_changes,
        benchmark_metrics_before=bench_before
    )

    return ApplyResponse(
        status=result["status"],
        applied=result["applied"],
        rebuild_required=result.get("rebuild_required", False),
        message=result["message"],
        previous_configuration=result.get("previous_configuration"),
        configuration=result["configuration"]
    )

@app.get("/tune/status", response_model=TunerStatusResponse, tags=["AI Tuning"])
def get_tuner_status():
    """
    Returns AI tuner status, model metadata, last recommendation, validation, and history.
    """
    curr_config = get_current_config()
    return ai_tuner.get_status(current_config=curr_config)

@app.post("/tune/rollback", response_model=RollbackResponse, tags=["AI Tuning"])
def rollback_tuning():
    """
    Restores the previous configuration from the rollback stack.
    """
    result = ai_tuner.rollback(apply_fn=apply_config_changes)
    return RollbackResponse(
        status=result["status"],
        applied=result["applied"],
        message=result["message"],
        configuration=result.get("configuration")
    )

# =========================================================================
# Robust, Reproducible Benchmark Harness
# =========================================================================

class BenchmarkHarness:
    def __init__(self, num_vectors: int = 2000, dimension: int = 128, num_queries: int = 50, k: int = 10):
        self.num_vectors = num_vectors
        self.dimension = dimension
        self.num_queries = num_queries
        self.k = k
        self._dataset: Optional[np.ndarray] = None
        self._queries: Optional[np.ndarray] = None
        self._ground_truth: Optional[List[set]] = None
        self._initialized = False

    def ensure_initialized(self):
        if self._initialized and state.index is not None and state.index.size() == self.num_vectors:
            return
        
        self.dimension = state.dimension
        
        # 1. Deterministic evaluation dataset
        np.random.seed(42)
        self._dataset = np.random.normal(0.0, 1.0, size=(self.num_vectors, self.dimension)).astype(np.float32)
        
        # 2. Reset or initialize global index with clean evaluation dataset
        state.index = VectorForgeIndex(
            dimension=self.dimension,
            backend=state.backend,
            m=state.m,
            ef_construction=state.ef_construction,
            ef_search=state.ef_search
        )
        state.index.insert(self._dataset)
        telemetry_collector.record_insert(len(self._dataset), self.dimension, success=True)
            
        # 3. Deterministic evaluation query set
        np.random.seed(1337)
        self._queries = np.random.normal(0.0, 1.0, size=(self.num_queries, self.dimension)).astype(np.float32)
        
        # 4. Pre-compute brute-force ground truth once and cache in memory
        bf = VectorForgeIndex(dimension=self.dimension, backend="bruteforce")
        bf.insert(self._dataset)
        self._ground_truth = []
        for q in self._queries:
            gt = bf.search(q, k=self.k)
            self._ground_truth.append(set(r["id"] for r in gt))
            
        self._initialized = True

    def run_benchmark(self, index: VectorForgeIndex, current_config: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_initialized()
        index = state.index
        
        # --- WARMUP PHASE ---
        # Run 15 queries to warm CPU caches, TLBs, and C++ memory allocations
        for i in range(15):
            warm_q = self._queries[i % len(self._queries)]
            index.search(warm_q, k=self.k)
            
        # Timed measurement loop over 3 iterations for stable microsecond statistics
        latencies_ms: List[float] = []
        hits = 0
        total_eval_queries = len(self._queries) * 3
        
        for _ in range(3):
            for i, q in enumerate(self._queries):
                t0 = time.perf_counter_ns()
                res = index.search(q, k=self.k)
                t1 = time.perf_counter_ns()
                
                lat_ms = (t1 - t0) / 1_000_000.0
                latencies_ms.append(lat_ms)
                
                retrieved_ids = set(r["id"] for r in res)
                hits += len(retrieved_ids.intersection(self._ground_truth[i]))
                
                telemetry_collector.record_query(
                    lat_ms,
                    self.k,
                    current_config.get("backend", "hnsw"),
                    self.dimension,
                    len(res),
                    True
                )
            
        avg_lat_sec = float(np.mean(latencies_ms)) / 1000.0 if latencies_ms else 0.001
        qps = (1.0 / avg_lat_sec) if avg_lat_sec > 0 else 1500.0
        p95_ms = float(np.percentile(latencies_ms, 95))
        recall = hits / (total_eval_queries * self.k) if (total_eval_queries * self.k) > 0 else 0.98
        
        # Defensible structural index memory calculation
        N = index.size()
        D = self.dimension
        M = current_config.get("m", 16)
        vec_bytes = N * D * 4
        graph_bytes = int(N * M * 8 * 1.25)
        total_mem_mb = round((vec_bytes + graph_bytes) / (1024 * 1024), 2)
        total_mem_gb = round(total_mem_mb / 1024, 3)
        
        # Real trade-off points measured across multiple search beam widths
        saved_ef = current_config.get("ef_search", 50)
        tradeoff_points = []
        
        for ef_val in [16, 24, 32, 48, 64, 96, 128, 160, 200]:
            index.tune(ef_search=ef_val)
            sub_hits = 0
            sub_lats = []
            for i in range(12):
                t0 = time.perf_counter_ns()
                r = index.search(self._queries[i], k=self.k)
                t1 = time.perf_counter_ns()
                sub_lats.append((t1 - t0) / 1_000_000.0)
                sub_hits += len(set(x["id"] for x in r).intersection(self._ground_truth[i]))
                
            sub_recall = sub_hits / (12 * self.k)
            sub_p95 = float(np.percentile(sub_lats, 95))
            sub_mean_ms = float(np.mean(sub_lats)) if sub_lats else 0.2
            sub_qps = round(1000.0 / max(0.01, sub_mean_ms), 1)
            tradeoff_points.append({
                "p95_ms": round(sub_p95, 2),
                "qps": sub_qps,
                "recall": round(sub_recall, 4),
                "m": M,
                "ef_search": ef_val
            })
            
        # Restore active search parameters
        index.tune(ef_search=saved_ef)
        
        return {
            "recall": round(recall, 4),
            "p95_latency_ms": round(p95_ms, 2),
            "qps": round(qps, 1),
            "memory_mb": total_mem_mb,
            "memory_gb": total_mem_gb,
            "memory_type": "estimated_structure",
            "tradeoff_points": tradeoff_points
        }

benchmark_harness = BenchmarkHarness(num_vectors=2000, dimension=128, num_queries=50, k=10)

@app.post("/benchmark/run", tags=["Benchmark"])
def run_benchmark():
    """
    Executes real live k-NN benchmark queries with warmup,
    computes exact Recall@10 against a pre-computed BruteForce ground truth,
    measures P95 latency (ms), QPS throughput, and memory footprint.
    Validates empirical performance if an AI recommendation was applied (Phase 8).
    """
    benchmark_harness.ensure_initialized()
    idx = get_or_create_index()
    curr_config = get_current_config()
    
    result = benchmark_harness.run_benchmark(idx, curr_config)
    
    run_num = len(benchmark_history) + 1
    run_name = "Baseline" if run_num == 1 else f"Experiment {run_num:02d}"
    
    # Closed-loop Phase 8 Validation against proposed AI recommendations
    validation = ai_tuner.validate_benchmark(result, curr_config)
    source = "baseline" if run_num == 1 else ("ai_recommendation" if validation else "manual")
    
    # Compute relative deltas against baseline
    baseline_run = get_baseline_run()
    deltas = {}
    if baseline_run and run_num > 1:
        bm = baseline_run["metrics"]
        deltas = {
            "recall_pct": round(((result["recall"] - bm["recall"]) / bm["recall"]) * 100, 1) if bm.get("recall", 0) > 0 else 0.0,
            "latency_pct": round(((result["p95_latency_ms"] - bm["p95_latency_ms"]) / bm["p95_latency_ms"]) * 100, 1) if bm.get("p95_latency_ms", 0) > 0 else 0.0,
            "qps_pct": round(((result["qps"] - bm["qps"]) / bm["qps"]) * 100, 1) if bm.get("qps", 0) > 0 else 0.0,
            "memory_pct": round(((result["memory_mb"] - bm["memory_mb"]) / bm["memory_mb"]) * 100, 1) if bm.get("memory_mb", 0) > 0 else 0.0
        }
    
    run_record = {
        "run_id": f"run_{run_num}",
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": dict(curr_config),
        "metrics": {
            "recall": result["recall"],
            "p95_latency_ms": result["p95_latency_ms"],
            "qps": result["qps"],
            "memory_mb": result["memory_mb"],
            "memory_gb": result["memory_gb"],
            "memory_type": result.get("memory_type", "estimated_structure")
        },
        "source": source,
        "validation": validation,
        "deltas": deltas,
        "tradeoff_points": result["tradeoff_points"]
    }
    
    benchmark_history.append(run_record)
    
    return {
        **run_record,
        "runs": benchmark_history
    }

@app.post("/benchmark/clear", tags=["Benchmark"])
@app.delete("/benchmark/clear", tags=["Benchmark"])
@app.delete("/benchmark/history", tags=["Benchmark"])
@app.post("/benchmark/history/clear", tags=["Benchmark"])
def clear_benchmark_history():
    """
    Clears all historical benchmark runs from memory.
    """
    count = len(benchmark_history)
    benchmark_history.clear()
    return {
        "status": "success",
        "deleted_count": count,
        "message": "Benchmark history cleared",
        "runs": []
    }

@app.get("/benchmark/history", tags=["Benchmark"])
def get_benchmark_history():
    """
    Returns historical benchmark runs and trade-off points without automatically creating new benchmark runs when empty.
    """
    return {
        "runs": benchmark_history,
        "latest": benchmark_history[-1] if benchmark_history else None,
        "tradeoff_points": benchmark_history[-1].get("tradeoff_points", []) if benchmark_history else []
    }

# =========================================================================
# Legacy / Manual System Endpoints
# =========================================================================

@app.get("/system/telemetry", response_model=SystemTelemetryResponse, tags=["System"])
def get_system_telemetry():
    """
    Returns current index state, vector counts, active backend, and hardware SIMD capabilities (Phase 5 compatibility).
    """
    idx = get_or_create_index()
    tele = idx.get_telemetry()

    return SystemTelemetryResponse(
        status="healthy",
        total_vectors=tele["size"],
        dimension=tele["dimension"],
        backend=tele["backend"],
        simd=tele["simd"],
        hnsw_parameters={
            "ef_search": tele.get("ef_search", state.ef_search),
            "ef_construction": tele.get("ef_construction", state.ef_construction),
            "m": tele.get("m", state.m),
            "max_level": tele.get("max_level", -1),
            "entry_point": tele.get("entry_point")
        }
    )

@app.post("/index/tune", response_model=TuneResponse, tags=["Configuration"])
def tune_index(payload: TuneRequest):
    """
    Dynamically tunes index search parameters (such as efSearch).
    """
    idx = get_or_create_index()

    if payload.ef_search is not None:
        state.ef_search = payload.ef_search
    if payload.m is not None:
        state.m = payload.m
    if payload.ef_construction is not None:
        state.ef_construction = payload.ef_construction

    result = idx.tune(
        ef_search=payload.ef_search,
        m=payload.m,
        ef_construction=payload.ef_construction
    )

    return TuneResponse(
        ef_search=result["ef_search"],
        m=result["m"],
        ef_construction=result["ef_construction"],
        backend=result["backend"],
        message="Index parameters updated successfully."
    )

# =========================================================================
# Static Dashboard UI Mounting
# =========================================================================

dashboard_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")
if os.path.exists(dashboard_path):
    app.mount("/dashboard", StaticFiles(directory=dashboard_path, html=True), name="dashboard")

@app.get("/ui", include_in_schema=False)
def serve_ui():
    index_file = os.path.join(dashboard_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Dashboard UI index.html not found."}
