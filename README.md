# VectorForge

## What is VectorForge?

**VectorForge** is an in-memory, high-performance C++20 vector search engine engineered for fast, scalable nearest-neighbor retrieval with SIMD hardware acceleration, graph-based HNSW indexing, Product Quantization, Python pybind11 bindings, and real-time statistical telemetry and embedding drift detection.

---

## Current Status: Phase 6 — Telemetry & Drift Detection

VectorForge features a production-grade, bounded telemetry collector and statistical drift detection layer around its search and insertion workflows.

> [!IMPORTANT]
> **Detection-Only Architecture**: Phase 6 detects drift and reports real-time metrics but does **NOT** automatically tune the index or modify parameters. Autonomous tuning is strictly out of scope for Phase 6.

### Architecture

```text
Client / Web Requests
          │
          ▼
FastAPI Control Plane (server/app.py)
   ├── /vectors/insert, /vectors/query
   ├── /telemetry, /drift, /drift/reset
   └── /system/telemetry, /index/tune
          │
          ├──────────────────────────┐
          ▼                          ▼
VectorForge C++20 Core      Telemetry & Drift Subsystem
 (pybind11 zero-copy)        (Thread-safe, bounded memory)
   ├── HNSW Index Graph       ├── TelemetryCollector (latency stats & counts)
   ├── Brute-force Index      ├── Query DriftDetector (query distribution)
   ├── SIMD AVX2 / AVX-512    └── Dataset DriftDetector (index distribution)
   └── Product Quantizer
```

---

## Phase 6 — Telemetry & Drift Detection Details

### 1. Telemetry Collector
* **What it measures**:
  - `total_vectors`: Active count of vectors stored in the index.
  - `dimension`: Vector dimensionality $D$.
  - `total_insert_operations` & `total_query_operations`: Total invoked API operations.
  - `query_count` / `successful_queries` & `failed_queries`: Total completed queries vs errors.
  - `successful_inserts` & `failed_inserts`: Insertion health counters.
  - `latency`: Exact query latency percentiles and distribution metrics in milliseconds (`avg_ms`, `p50_ms`, `p95_ms`, `p99_ms`, `min_ms`, `max_ms`).
  - `backend`: Current search backend (`hnsw`, `bruteforce`).
  - `simd`: Active hardware SIMD kernel (`AVX-512`, `AVX2 + FMA`, or `Scalar Fallback`).
  - `hnsw_parameters`: Active runtime graph parameters (`ef_search`, `ef_construction`, `m`, `max_level`, `entry_point`).
* **Bounded In-Memory History**: Query latencies are recorded into a fixed circular buffer (`MAX_LATENCY_SAMPLES = 10,000`) preventing unbounded memory growth under continuous traffic. Oldest entries are automatically evicted in $O(1)$ time.
* **Calculation of Latency Percentiles**: Percentiles ($p50, p95, p99$) and average latency are computed using exact numpy percentile routines over the active sample window. Failed queries record error state without polluting valid query latency statistics.

### 2. Embedding Drift Detection
Embedding drift occurs when the statistical properties of incoming query embeddings or indexed vectors shift relative to the reference baseline distribution (e.g. shifts in topics, vocabulary domain, or representation scaling).

* **Baseline Statistics**:
  The reference baseline computes distribution parameters over vectors:
  - Per-dimension mean vector: $\mu_{\text{base}} \in \mathbb{R}^D$
  - Per-dimension standard deviation vector: $\sigma_{\text{base}} \in \mathbb{R}^D$
  - Scalar $L_2$ vector norm mean and standard deviation: $\mu_{\text{norm, base}}, \sigma_{\text{norm, base}}$
* **Mathematical Drift Scoring**:
  Incoming vectors populate a bounded sliding buffer (`MAX_SAMPLE_SIZE = 2,000`). When evaluated, the detector calculates three standardized shift components:
  1. **Standardized Mean Shift ($S_{\mu}$)**:
     $$d_{\mu} = \frac{\|\mu_{\text{curr}} - \mu_{\text{base}}\|_2}{\sqrt{D} \cdot (\bar{\sigma}_{\text{base}} + \epsilon)}$$
     $$S_{\mu} = 1.0 - \exp(-1.5 \cdot d_{\mu}) \in [0, 1]$$
  2. **Per-Dimension Dispersion Shift ($S_{\sigma}$)**:
     $$d_{\sigma} = \frac{1}{D} \sum_{i=1}^D \frac{|\sigma_{\text{curr}, i} - \sigma_{\text{base}, i}|}{\sigma_{\text{base}, i} + \epsilon}$$
     $$S_{\sigma} = 1.0 - \exp(-1.5 \cdot d_{\sigma}) \in [0, 1]$$
  3. **Vector Norm Distribution Shift ($S_{\text{norm}}$)**:
     $$d_{\text{norm}} = \frac{|\mu_{\text{norm, curr}} - \mu_{\text{norm, base}}|}{\sigma_{\text{norm, base}} + \epsilon}$$
     $$S_{\text{norm}} = 1.0 - \exp(-1.5 \cdot d_{\text{norm}}) \in [0, 1]$$
  4. **Composite Drift Score ($S$)**:
     $$S = 0.50 \cdot S_{\mu} + 0.30 \cdot S_{\sigma} + 0.20 \cdot S_{\text{norm}}, \quad S \in [0.0, 1.0]$$

* **Configurable Drift Thresholds**:
  - `score < 0.30`: `"normal"`
  - `0.30 <= score < 0.60`: `"warning"`
  - `score >= 0.60`: `"drift_detected"`

* **Query Drift vs Dataset Drift**:
  - `query_drift`: Tracks search query vectors independently to detect changing user intents or query domain shifts.
  - `dataset_drift`: Tracks inserted vectors to detect evolving corpus content.

---

## FastAPI REST Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/telemetry` | `GET` | Returns full telemetry summary (latency percentiles, counters, SIMD state) |
| `/drift` | `GET` | Returns real-time drift metrics and status for query and dataset workloads |
| `/drift/reset` | `POST` | Manually re-initializes the baseline reference distribution (`{"channel": "all"}`) |
| `/vectors/insert` | `POST` | Inserts batch of float32 vectors (`{"vectors": [[...], ...]}`) |
| `/vectors/query` | `POST` | Executes k-NN nearest neighbor query (`{"vector": [...], "k": 10}`) |
| `/system/telemetry` | `GET` | Returns Phase 5 compatible system and hardware telemetry |
| `/index/tune` | `POST` | Dynamically updates runtime search parameters (e.g. `ef_search`) |
| `/docs` | `GET` | Interactive OpenAPI Swagger UI documentation |

---

## Benchmark: Telemetry & Search Overhead

Evaluated on **10,000 vectors $\times$ 128 dimensions, 100 queries ($k=10$)**:

| Layer / Configuration | Avg Query Latency | Throughput (QPS) | Overhead |
| :--- | :---: | :---: | :---: |
| **Python Direct (No Telemetry)** | **234.16 $\mu$s** | **4,251.65 QPS** | **Baseline (0.0%)** |
| **Python Direct (+ Telemetry & Drift)** | **244.16 $\mu$s** | **4,083.17 QPS** | **+4.27% (+10.00 $\mu$s)** |
| **FastAPI HTTP (+ Full Telemetry)** | **5,028.33 $\mu$s** (~5.0 ms) | **198.73 QPS** | **21.47x** |

> **Overhead Result**: Real-time telemetry collection and bounded drift sampling adds only **10.00 $\mu$s (4.27%)** overhead to pure vector search.

---

## Roadmap

* [x] **Phase 0 — Project Foundation**
* [x] **Phase 1 — Brute-Force Vector Search (k-NN Baseline)**
* [x] **Phase 2 — HNSW Indexing (Hierarchical Navigable Small World)**
* [x] **Phase 3 — SIMD Acceleration (AVX-512 / AVX2)**
* [x] **Phase 4 — Product Quantization (PQ) & ADC**
* [x] **Phase 5 — Python Bindings (pybind11) + FastAPI Control Plane**
* [x] **Phase 6 — Telemetry + Drift Detection** *(Completed)*
* [ ] **Phase 7 — Gemini AI Autonomous Tuner**
* [ ] **Phase 8 — Next.js Telemetry Dashboard**
* [ ] **Phase 9 — Integration, Benchmarking & End-to-End Demo**

---

## Building and Running

### Prerequisites
* CMake 3.20+
* C++20 compiler (GCC 11+, Clang 13+, MSVC 2019+)
* Python 3.10+ with `fastapi`, `uvicorn`, `numpy`, `pybind11`, `pydantic`, `pytest`

### Build Instructions
```bash
# 1. Install Python requirements
python -m pip install -r server/requirements.txt

# 2. Configure and build C++ core and pybind11 extension
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### 1. Run All Tests
```bash
# Python & FastAPI Unit, Telemetry, and Drift Tests (24/24 passing)
pytest server/ -v

# C++ Core Tests (127/127 passing)
ctest --test-dir build --output-on-failure
```

### 2. Run Overhead Benchmark
```bash
python server/benchmark_overhead.py
```

### 3. Start the FastAPI Control Plane Server
```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload
```
Open `http://127.0.0.1:8000/docs` in your browser for interactive API testing.
