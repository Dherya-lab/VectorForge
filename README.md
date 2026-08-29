# VectorForge

## What is VectorForge?

**VectorForge** is an in-memory, high-performance C++20 vector search engine engineered for fast, scalable nearest-neighbor retrieval with SIMD hardware acceleration, graph-based HNSW indexing, Product Quantization, Python pybind11 bindings, real-time statistical telemetry, drift detection, and Gemini AI-assisted adaptive index tuning.

---

## Current Status: Phase 7 — AI-Assisted Adaptive Index Tuning

VectorForge incorporates an AI-assisted tuning layer powered by Google Gemini that analyzes real-time search telemetry and statistical drift signals to recommend and safely apply vector index parameters.

```text
                ┌───────────────────┐
                │   FastAPI API     │
                └─────────┬─────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
        Vector Search             AI Tuner
             │                         │
             ↓                         ↓
        pybind11/C++             Gemini API
             │                         │
             ↓                         │
      VectorForge Engine ◄─────────────┘
             │
      ┌──────┼─────────┐
      ↓      ↓         ↓
   HNSW    SIMD       PQ
             │
             ↓
       Telemetry
             │
             ↓
      Drift Detection
```

---

## Phase 7 — AI-Assisted Tuning Details

### 1. Why AI Tuning is Needed
Vector index performance requires navigating multi-dimensional trade-offs between **Recall**, **Query Latency**, and **Memory Footprint**:
* Higher `ef_search` expands the graph search beam width $\rightarrow$ increases Recall@K but increases query latency.
* Lower `ef_search` accelerates search traversal $\rightarrow$ reduces latency but risks missing true nearest neighbors.
* Higher `M` and `ef_construction` improve graph connectivity and quality $\rightarrow$ increases index build time and memory.
* Product Quantization reduces memory usage by 32x-64x $\rightarrow$ causes distance approximations.

Workload distributions shift over time (e.g. topic changes, vocabulary expansion, query burst patterns). The AI Tuner acts as an intelligent performance optimizer that continuously analyzes operational metrics to find optimal configuration operating points.

### 2. Telemetry and Drift Inputs Consumed by the Tuner
* **Telemetry**: Vector count, dimensionality, query count, insert operations, error rates, and exact latency percentiles (`avg_ms`, `p50_ms`, `p90_ms`, `p95_ms`, `p99_ms`, `min_ms`, `max_ms`).
* **Drift Signals**: `query_drift` and `dataset_drift` scores, thresholds, and transition states.
* **Active Configuration**: Current `ef_search`, `M`, `ef_construction`, and backend details.
* **Target Objectives**: User-specified `target_recall`, `max_p95_latency_ms`, and `priority` (`"recall"`, `"latency"`, or `"balanced"`).

### 3. How Gemini is Used & Why AI Does Not Directly Control C++
* **Zero Direct C++ Execution**: Gemini never executes native code or modifies memory directly.
* **Structured Output Schema**: Gemini produces validated JSON recommendations containing action (`tune`, `no_change`, `rebuild_required`), technical reason, parameter changes, expected trade-offs, and confidence.
* **Separation of Recommendation vs Apply**:
  - `POST /tune/recommend`: Generates analysis and recommendations without altering configuration.
  - `POST /tune/apply`: Validates recommendations against hard bounds before applying.

### 4. Safety Bounds & Parameter Enforcement
All AI suggestions are validated through strict Pydantic schemas and hard safety bounds:
* `ef_search`: `[1, 10000]` *(Runtime tunable via `idx.tune()`)*
* `M`: `[2, 128]` *(Requires index rebuild)*
* `ef_construction`: `[4, 2000]` *(Requires index rebuild)*

> **Rebuild-Required Behavior**: Parameters that alter structural graph topology (`M`, `ef_construction`) are safely rejected during live application with `rebuild_required: true` rather than performing an unsafe live rebuild.

### 5. Rollback Capability
The tuner maintains a thread-safe rollback stack. `POST /tune/rollback` restores the previous operating configuration in $O(1)$ time.

---

## FastAPI REST Endpoints

### AI Tuning (Phase 7)
| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/tune/recommend` | `POST` | Generates AI-assisted tuning recommendation based on telemetry & drift |
| `/tune/apply` | `POST` | Safely validates and applies the recommended parameters |
| `/tune/status` | `GET` | Returns active tuner status, model details, and tuning history |
| `/tune/rollback` | `POST` | Restores previous index configuration from rollback stack |

### Telemetry & Drift (Phase 6)
| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/telemetry` | `GET` | Returns full operational counters and latency percentiles |
| `/drift` | `GET` | Returns real-time query and dataset drift metrics |
| `/drift/reset` | `POST` | Resets drift reference baselines |

### Core Search & Vectors (Phase 5)
| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/vectors/insert` | `POST` | Inserts batch of float32 vectors (`{"vectors": [[...], ...]}`) |
| `/vectors/query` | `POST` | Executes k-NN nearest neighbor query (`{"vector": [...], "k": 10}`) |
| `/system/telemetry` | `GET` | Returns system hardware and SIMD capabilities |
| `/index/tune` | `POST` | Manual index parameter tuning |
| `/docs` | `GET` | Interactive OpenAPI Swagger UI documentation |

---

## Benchmark: Search, Telemetry & AI Tuning Performance

Evaluated on **10,000 vectors $\times$ 128 dimensions, 100 queries ($k=10$)**:

| Layer / Configuration | Avg Query Latency | Throughput (QPS) | Overhead |
| :--- | :---: | :---: | :---: |
| **Python Direct (No Telemetry)** | **344.56 $\mu$s** | **2,888.33 QPS** | **Baseline (0.0%)** |
| **Python Direct (+ Telemetry & Drift)** | **345.57 $\mu$s** | **2,883.31 QPS** | **+0.29% (+1.01 $\mu$s)** |
| **FastAPI HTTP (+ Full Telemetry)** | **8,289.51 $\mu$s** | **120.56 QPS** | **24.06x** |

### AI Tuning Latency
* `/tune/recommend` Latency: **39.48 ms**
* `/tune/apply` Latency: **5.67 ms**

---

## Roadmap

* [x] **Phase 0 — Project Foundation**
* [x] **Phase 1 — Brute-Force Vector Search (k-NN Baseline)**
* [x] **Phase 2 — HNSW Indexing (Hierarchical Navigable Small World)**
* [x] **Phase 3 — SIMD Acceleration (AVX-512 / AVX2)**
* [x] **Phase 4 — Product Quantization (PQ) & ADC**
* [x] **Phase 5 — Python Bindings (pybind11) + FastAPI Control Plane**
* [x] **Phase 6 — Telemetry + Drift Detection**
* [x] **Phase 7 — Gemini AI-Assisted Adaptive Index Tuning** *(Completed)*
* [ ] **Phase 8 — Next.js Telemetry Dashboard**
* [ ] **Phase 9 — Integration, Benchmarking & End-to-End Demo**

---

## Building and Running

### Prerequisites
* CMake 3.20+
* C++20 compiler (GCC 11+, Clang 13+, MSVC 2019+)
* Python 3.10+ with `fastapi`, `uvicorn`, `numpy`, `pybind11`, `pydantic`, `pytest`, `google-genai`
* (Optional) `GEMINI_API_KEY` environment variable for Gemini AI model calls. If omitted, the system seamlessly operates using its built-in heuristic rule-engine.

### Build Instructions
```bash
# 1. Install Python requirements
python -m pip install -r server/requirements.txt

# 2. Configure and build C++ core and pybind11 extension
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### 1. Run All Tests (37/37 passing)
```bash
# Run all server unit, telemetry, drift, and AI tuner tests
pytest server/ -v

# Run native C++ core tests
ctest --test-dir build --output-on-failure
```

### 2. Run Overhead & AI Tuning Benchmark
```bash
python server/benchmark_overhead.py
```

### 3. Start the FastAPI Control Plane Server
```bash
# Optional: Set Gemini API key
# set GEMINI_API_KEY=your_api_key_here  (Windows CMD)
# $env:GEMINI_API_KEY="your_api_key_here" (PowerShell)

python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload
```
Open `http://127.0.0.1:8000/docs` to test the interactive Swagger UI.
