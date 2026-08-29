# VectorForge

## What is VectorForge?

**VectorForge** is an in-memory, high-performance C++20 vector search engine engineered for fast, scalable nearest-neighbor retrieval. It is designed to combine:

* **HNSW** (Hierarchical Navigable Small World) approximate nearest-neighbor indexing
* **SIMD acceleration** (AVX-512 / AVX2) for low-latency distance computations
* **Product Quantization (PQ)** for high-ratio memory compression
* **Runtime Telemetry** for tracking query latency, throughput, and index health
* **Embedding & Query Drift Detection**
* **Gemini-Powered Autonomous Index Tuning** for zero-downtime optimization
* **Next.js Telemetry Dashboard** & FastAPI control plane

---

## Current Phase: Phase 0 — Project Foundation

Phase 0 establishes the core foundation of VectorForge:
* Modern **C++20** standard compliance
* Modular **CMake** build configuration with compiler warnings enabled
* Clean directory layout separating headers, source, tests, and future modules
* Core `Vector` class representing floating-point embedding arrays
* Minimal runnable executable demonstrating vector construction and inspection

---

## Roadmap / Future Phases

* **Phase 0 — Project Foundation** *(Current)*
* **Phase 1 — Brute-Force Vector Search**
* **Phase 2 — HNSW Indexing**
* **Phase 3 — SIMD Acceleration (AVX-512 / AVX2)**
* **Phase 4 — Product Quantization (PQ)**
* **Phase 5 — Python Bindings (pybind11) + FastAPI Control Plane**
* **Phase 6 — Telemetry + Drift Detection**
* **Phase 7 — Gemini AI Autonomous Tuner**
* **Phase 8 — Next.js Telemetry Dashboard**
* **Phase 9 — Integration, Benchmarking & End-to-End Demo**

---

## Building and Running (Phase 0)

### Prerequisites
* CMake 3.20+
* C++20 compatible compiler (GCC 11+, Clang 13+, or MSVC 2019+)

### Build Instructions
```bash
# Configure the build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build targets
cmake --build build --config Release
```

### Run Executable
```bash
# Linux/macOS
./build/core/vectorforge_main

# Windows
.\build\core\vectorforge_main.exe
# or (Multi-config MSVC)
.\build\core\Release\vectorforge_main.exe
```
