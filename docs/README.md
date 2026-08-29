# VectorForge Documentation

Welcome to the VectorForge developer and architectural documentation.

## System Architecture

```text
Client Applications / External Services
                 │
                 ▼
     FastAPI Control Plane (server/app.py)
   ├── REST Endpoints: /vectors/insert, /vectors/query
   ├── Telemetry Endpoints: /telemetry, /system/telemetry
   ├── Drift Detection Endpoints: /drift, /drift/reset
   └── Index Configuration: /index/tune
                 │
                 ├──────────────────────────┐
                 ▼                          ▼
      VectorForge C++20 Core      Telemetry & Drift Subsystem
      (pybind11 zero-copy)        (Thread-safe, bounded memory)
        ├── HNSW Index Graph       ├── TelemetryCollector
        ├── Flat VectorIndex       ├── Query DriftDetector
        ├── SIMD Math Kernels      └── Dataset DriftDetector
        └── Product Quantizer
```

## Phase 6 Telemetry & Drift Detection Specifications

### 1. Telemetry Collector (`server/telemetry.py`)
- **Bounded Buffer**: Stores up to `MAX_LATENCY_SAMPLES = 10,000` query latency samples in memory in a circular queue.
- **Latency Percentiles**: Calculates exact `avg_ms`, `p50_ms`, `p95_ms`, `p99_ms`, `min_ms`, `max_ms`.
- **Counters**: Atomic tracking of queries, inserts, and operational success/failure counts.

### 2. Embedding Drift Detection (`server/drift.py`)
- **Baseline**: Distribution statistics ($\mu_{\text{base}}, \sigma_{\text{base}}, \mu_{\text{norm}}, \sigma_{\text{norm}}$) capturing reference embeddings.
- **Drift Score**:
  $$S = 0.50 \cdot S_{\mu} + 0.30 \cdot S_{\sigma} + 0.20 \cdot S_{\text{norm}}$$
- **Thresholds**:
  - `score < 0.30`: `normal`
  - `0.30 <= score < 0.60`: `warning`
  - `score >= 0.60`: `drift_detected`
- **Scope Boundary**: Detection only. Automatic tuning or parameter modifications are strictly disabled.
