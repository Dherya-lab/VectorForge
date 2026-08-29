import threading
import time
from collections import deque
from typing import Dict, Any, Optional, List
import numpy as np

# Configurable bounded buffer size to prevent unbounded memory growth
MAX_LATENCY_SAMPLES = 10000
MAX_RECENT_QUERIES = 100

class TelemetryCollector:
    """
    Thread-safe, bounded in-memory telemetry collector for VectorForge.
    Measures latency statistics and operational metrics without storing raw vectors.
    """
    def __init__(self, max_latency_samples: int = MAX_LATENCY_SAMPLES, max_recent_queries: int = MAX_RECENT_QUERIES):
        self._lock = threading.Lock()
        self.max_latency_samples = max_latency_samples
        self.max_recent_queries = max_recent_queries

        # Counters
        self.total_insert_operations: int = 0
        self.total_query_operations: int = 0
        self.successful_queries: int = 0
        self.failed_queries: int = 0
        self.successful_inserts: int = 0
        self.failed_inserts: int = 0
        self.total_vectors_inserted: int = 0

        # Bounded buffers
        self._latency_history_ms: deque = deque(maxlen=self.max_latency_samples)
        self._recent_queries: deque = deque(maxlen=self.max_recent_queries)

    def record_query(
        self,
        latency_ms: float,
        k: int,
        backend: str,
        dimension: int,
        result_count: int,
        success: bool = True
    ) -> None:
        """
        Records a query execution with latency and operational metadata.
        """
        with self._lock:
            self.total_query_operations += 1
            if success:
                self.successful_queries += 1
                self._latency_history_ms.append(float(latency_ms))
            else:
                self.failed_queries += 1

            self._recent_queries.append({
                "timestamp": time.time(),
                "latency_ms": round(float(latency_ms), 4),
                "k": k,
                "backend": backend,
                "dimension": dimension,
                "result_count": result_count,
                "success": success
            })

    def record_insert(self, count: int, dimension: int, success: bool = True) -> None:
        """
        Records an insert operation.
        """
        with self._lock:
            self.total_insert_operations += 1
            if success:
                self.successful_inserts += 1
                self.total_vectors_inserted += count
            else:
                self.failed_inserts += 1

    def get_latency_stats(self) -> Dict[str, Optional[float]]:
        """
        Calculates exact latency metrics (average, min, max, p50, p95, p99) in milliseconds.
        """
        with self._lock:
            if not self._latency_history_ms:
                return {
                    "avg_ms": None,
                    "p50_ms": None,
                    "p95_ms": None,
                    "p99_ms": None,
                    "min_ms": None,
                    "max_ms": None,
                    "sample_count": 0,
                    "max_buffer_size": self.max_latency_samples
                }

            samples = np.array(self._latency_history_ms, dtype=np.float64)
            return {
                "avg_ms": round(float(np.mean(samples)), 4),
                "p50_ms": round(float(np.percentile(samples, 50)), 4),
                "p95_ms": round(float(np.percentile(samples, 95)), 4),
                "p99_ms": round(float(np.percentile(samples, 99)), 4),
                "min_ms": round(float(np.min(samples)), 4),
                "max_ms": round(float(np.max(samples)), 4),
                "sample_count": len(samples),
                "max_buffer_size": self.max_latency_samples
            }

    def get_summary(self, index_telemetry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Returns full telemetry summary combining operational counters, latency stats,
        and underlying C++ index hardware/dispatch metadata.
        """
        latency_stats = self.get_latency_stats()
        idx_meta = index_telemetry or {}

        with self._lock:
            total_queries = self.total_query_operations
            successful_queries = self.successful_queries
            failed_queries = self.failed_queries
            total_inserts = self.total_insert_operations
            successful_inserts = self.successful_inserts
            failed_inserts = self.failed_inserts

        return {
            "vectors": idx_meta.get("size", self.total_vectors_inserted),
            "dimension": idx_meta.get("dimension", 0),
            "total_insert_operations": total_inserts,
            "total_query_operations": total_queries,
            "query_count": successful_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "successful_inserts": successful_inserts,
            "failed_inserts": failed_inserts,
            "latency": latency_stats,
            "backend": idx_meta.get("backend", "unavailable"),
            "simd": idx_meta.get("simd", {}).get("active_dispatch", "unavailable"),
            "simd_details": idx_meta.get("simd", {}),
            "hnsw_parameters": {
                "ef_search": idx_meta.get("ef_search"),
                "ef_construction": idx_meta.get("ef_construction"),
                "m": idx_meta.get("m"),
                "max_level": idx_meta.get("max_level"),
                "entry_point": idx_meta.get("entry_point")
            } if idx_meta.get("backend") == "hnsw" else None,
            "pq_parameters": None # Product Quantization standalone index not active in single-index mode
        }

    def reset(self) -> None:
        """
        Resets in-memory telemetry counters and buffers.
        """
        with self._lock:
            self.total_insert_operations = 0
            self.total_query_operations = 0
            self.successful_queries = 0
            self.failed_queries = 0
            self.successful_inserts = 0
            self.failed_inserts = 0
            self.total_vectors_inserted = 0
            self._latency_history_ms.clear()
            self._recent_queries.clear()
