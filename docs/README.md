# VectorForge Documentation

Welcome to the VectorForge developer and architectural documentation.

## System Architecture (Phase 7)

```text
Client Applications / External Services
                 │
                 ▼
     FastAPI Control Plane (server/app.py)
   ├── REST Endpoints: /vectors/insert, /vectors/query
   ├── Telemetry Endpoints: /telemetry, /system/telemetry
   ├── Drift Detection Endpoints: /drift, /drift/reset
   └── AI Tuning Endpoints: /tune/recommend, /tune/apply, /tune/status, /tune/rollback
                 │
                 ├──────────────────────────┬──────────────────────────┐
                 ▼                          ▼                          ▼
      VectorForge C++20 Core      Telemetry & Drift Subsystem      Gemini AI Tuner
      (pybind11 zero-copy)        (Thread-safe, bounded memory)   (Structured Pydantic)
        ├── HNSW Index Graph       ├── TelemetryCollector          ├── Safety bounds validator
        ├── Flat VectorIndex       ├── Query DriftDetector         ├── Heuristic fallback
        ├── SIMD Math Kernels      └── Dataset DriftDetector       └── Rollback stack
        └── Product Quantizer
```

## Phase 7 AI-Assisted Adaptive Index Tuning Specifications

### 1. AI Tuner Subsystem (`server/tuner.py`)
- **Gemini AI Integration**: Uses `google-genai` SDK with `GEMINI_API_KEY` environment variable.
- **Safety Bounds**:
  - `ef_search`: `[1, 10000]` *(Runtime tunable)*
  - `M`: `[2, 128]` *(Requires index rebuild)*
  - `ef_construction`: `[4, 2000]` *(Requires index rebuild)*
- **Rebuild-Required Protection**: Refuses live modification of structural graph parameters, returning `rebuild_required: true`.
- **Rollback Stack**: In-memory stack of prior configurations enabling safe rollbacks.
