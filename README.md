# VectorForge

## What is VectorForge?

**VectorForge** is an in-memory, high-performance C++20 vector search engine engineered for fast, scalable nearest-neighbor retrieval.

---

## Current Status: Phase 5 — pybind11 Python Binding & FastAPI Control Plane

VectorForge exposes its high-performance C++20 engine directly to Python through **pybind11** zero-copy NumPy bindings and a lightweight **FastAPI** REST control plane.

### Architecture
```text
Python Client / Web Requests
            │
            ▼
    FastAPI (server/app.py)
   (Pydantic Validation & REST Endpoints)
            │
            ▼
   pybind11 Binding (vectorforge.pyd)
   (Zero-copy float32 NumPy buffer views, GIL release)
            │
            ▼
   VectorForge C++20 Core
   ├── Brute-Force VectorIndex (Exact Ground Truth)
   ├── HNSWIndex (Approximate Graph Navigation)
   ├── SIMD Math (AVX-512 / AVX2 + FMA Dispatched)
   └── ProductQuantizer (32x-64x ADC Compression)
```

### Why This Architecture?
* **C++ Core as Performance Truth**: Vector distance calculations and graph traversals execute entirely in native C++ with SIMD instructions. Python never performs mathematical vector loops.
* **pybind11 Zero-Copy Buffer Protocol**: Passes 1D and 2D `np.ndarray` float32 contiguous arrays directly into C++ without redundant serialization or copying at the Python/C++ boundary.
* **GIL Release**: CPU-intensive operations (batch insertion and graph candidate traversal) release Python's Global Interpreter Lock (`py::gil_scoped_release`), allowing concurrent Python async worker execution.
* **FastAPI Orchestration**: Provides clean REST APIs for inserting vectors, executing k-NN queries, monitoring hardware telemetry, and tuning search parameters.

---

## FastAPI REST Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/vectors/insert` | `POST` | Inserts batch of float32 vectors (`{"vectors": [[...], ...]}`) |
| `/vectors/query` | `POST` | Executes k-NN nearest neighbor query (`{"vector": [...], "k": 10}`) |
| `/system/telemetry` | `GET` | Returns vector counts, dimension, active backend, and hardware SIMD capability |
| `/index/tune` | `POST` | Dynamically updates runtime search parameters (e.g. `ef_search`) |
| `/docs` | `GET` | Interactive Swagger OpenAPI documentation and testing playground |

---

## Search Latency & Layer Overhead Comparison

Evaluated on **10,000 vectors $\times$ 128 dimensions, 50 queries ($k=10$)**:

| Layer / Interface | Avg Query Latency | Throughput (QPS) | Overhead vs Native C++ |
| :--- | :---: | :---: | :---: |
| **C++ Core HNSW Direct (SIMD)** | **310.63 $\mu$s** | **3,216.61 QPS** | **1.00x (Baseline)** |
| **Python pybind11 Direct Search** | **521.60 $\mu$s** | **1,906.18 QPS** | **1.68x** |
| **FastAPI HTTP Request + Search** | **4,951.61 $\mu$s** (~4.9 ms) | **201.81 QPS** | **15.94x** |

> **Overhead Insight**: Python binding adds negligible overhead (~210 $\mu$s) for Python dictionary instantiation. The HTTP server overhead is driven by JSON serialization, HTTP header parsing, and Starlette request stack handling, while the underlying search execution remains 100% native C++.

---

## Roadmap

* [x] **Phase 0 — Project Foundation**
* [x] **Phase 1 — Brute-Force Vector Search (k-NN Baseline)**
* [x] **Phase 2 — HNSW Indexing (Hierarchical Navigable Small World)**
* [x] **Phase 3 — SIMD Acceleration (AVX-512 / AVX2)**
* [x] **Phase 4 — Product Quantization (PQ) & ADC**
* [x] **Phase 5 — Python Bindings (pybind11) + FastAPI Control Plane** *(Completed)*
* [ ] **Phase 6 — Telemetry + Drift Detection**
* [ ] **Phase 7 — Gemini AI Autonomous Tuner**
* [ ] **Phase 8 — Next.js Telemetry Dashboard**
* [ ] **Phase 9 — Integration, Benchmarking & End-to-End Demo**

---

## Building and Running

### Prerequisites
* CMake 3.20+
* C++20 compiler (GCC 11+, Clang 13+, MSVC 2019+)
* Python 3.10+ with `fastapi`, `uvicorn`, `numpy`, `pybind11`, `pydantic`

### Build Instructions
```bash
# 1. Install Python requirements
python -m pip install -r server/requirements.txt

# 2. Configure and build C++ core and pybind11 extension
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### 1. Run Python Binding Smoke Test
```bash
python server/test_binding.py
```

### 2. Run Python & FastAPI Test Suite
```bash
pytest server/test_api.py -v
```

### 3. Run C++ Test Suite (127/127 Passed)
```bash
.\build\core\vectorforge_test.exe
```

### 4. Start the FastAPI Control Plane Server
```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at `http://127.0.0.1:8000/docs` to test the interactive Swagger UI.
