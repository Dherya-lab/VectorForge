import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import numpy as np

# Configurable defaults
DEFAULT_MAX_SAMPLES = 2000
DEFAULT_DRIFT_THRESHOLD = 0.30
DEFAULT_SEVERE_THRESHOLD = 0.60
EPSILON = 1e-6

class DistributionStatistics:
    """
    Holds summary statistics of a vector distribution:
    - mean vector (D,)
    - stddev vector (D,)
    - scalar vector norm mean and stddev
    - sample count
    """
    def __init__(self, vectors: np.ndarray):
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array of vectors (N, D), got shape {vectors.shape}")
        
        self.sample_count = int(vectors.shape[0])
        self.dimension = int(vectors.shape[1])
        
        if self.sample_count == 0:
            self.mean = np.zeros(self.dimension, dtype=np.float64)
            self.std = np.zeros(self.dimension, dtype=np.float64)
            self.norm_mean = 0.0
            self.norm_std = 0.0
        else:
            self.mean = np.mean(vectors, axis=0, dtype=np.float64)
            self.std = np.std(vectors, axis=0, dtype=np.float64)
            norms = np.linalg.norm(vectors, axis=1)
            self.norm_mean = float(np.mean(norms))
            self.norm_std = float(np.std(norms))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "dimension": self.dimension,
            "mean_norm": round(self.norm_mean, 5),
            "std_norm": round(self.norm_std, 5),
            "mean_of_dimension_means": round(float(np.mean(self.mean)), 5) if self.dimension > 0 else 0.0,
            "mean_of_dimension_stds": round(float(np.mean(self.std)), 5) if self.dimension > 0 else 0.0
        }

class DriftDetector:
    """
    Statistical embedding drift detector for VectorForge.
    
    Compares the distribution of incoming vector workloads against a calibrated baseline.
    Computes a normalized drift score S in [0.0, 1.0] derived from:
    1. Normalized Mean Shift (Euclidean distance between mean vectors standardized by baseline variance)
    2. Dispersion / Variance Shift (mean relative variation in per-coordinate standard deviations)
    3. Norm Distribution Shift (standardized shift in L2 vector lengths)

    Status Categorization:
    - score < threshold: "normal"
    - threshold <= score < severe_threshold: "warning"
    - score >= severe_threshold: "drift_detected"
    """
    def __init__(
        self,
        name: str = "drift_detector",
        threshold: float = DEFAULT_DRIFT_THRESHOLD,
        severe_threshold: float = DEFAULT_SEVERE_THRESHOLD,
        max_samples: int = DEFAULT_MAX_SAMPLES
    ):
        self.name = name
        self.threshold = threshold
        self.severe_threshold = severe_threshold
        self.max_samples = max_samples
        self._lock = threading.Lock()

        self._baseline_stats: Optional[DistributionStatistics] = None
        self._sample_buffer: deque = deque(maxlen=self.max_samples)
        self._last_score: float = 0.0
        self._last_status: str = "normal"
        self._last_evaluated_at: Optional[str] = None

    def set_baseline(self, vectors: np.ndarray) -> None:
        """
        Initializes or replaces the baseline reference distribution.
        """
        with self._lock:
            if vectors.size == 0:
                self._baseline_stats = None
                return
            
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            
            self._baseline_stats = DistributionStatistics(arr)
            self._sample_buffer.clear()
            self._last_score = 0.0
            self._last_status = "normal"
            self._last_evaluated_at = datetime.now(timezone.utc).isoformat()

    def add_samples(self, vectors: np.ndarray) -> None:
        """
        Adds incoming vectors to the bounded sliding sample buffer.
        """
        with self._lock:
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            
            for v in arr:
                self._sample_buffer.append(v)
            
            # Automatically establish initial baseline if none exists yet
            if self._baseline_stats is None and len(self._sample_buffer) >= 10:
                buf_array = np.array(self._sample_buffer, dtype=np.float32)
                self._baseline_stats = DistributionStatistics(buf_array)

    def compute_drift(self) -> Dict[str, Any]:
        """
        Evaluates the current vector distribution in the buffer against the baseline.
        Returns a complete drift status report.
        """
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()

            if self._baseline_stats is None or len(self._sample_buffer) == 0:
                return {
                    "status": "normal",
                    "score": 0.0,
                    "threshold": self.threshold,
                    "severe_threshold": self.severe_threshold,
                    "sample_count": len(self._sample_buffer),
                    "timestamp": now_iso,
                    "baseline_statistics": self._baseline_stats.to_dict() if self._baseline_stats else None,
                    "current_statistics": None
                }

            curr_arr = np.array(self._sample_buffer, dtype=np.float32)
            curr_stats = DistributionStatistics(curr_arr)
            base_stats = self._baseline_stats

            # Dimensionality check
            if curr_stats.dimension != base_stats.dimension:
                # Dimension mismatch between baseline and current workload is severe drift
                self._last_score = 1.0
                self._last_status = "drift_detected"
                self._last_evaluated_at = now_iso
                return {
                    "status": "drift_detected",
                    "score": 1.0,
                    "threshold": self.threshold,
                    "severe_threshold": self.severe_threshold,
                    "sample_count": curr_stats.sample_count,
                    "timestamp": now_iso,
                    "baseline_statistics": base_stats.to_dict(),
                    "current_statistics": curr_stats.to_dict()
                }

            dim = float(base_stats.dimension)
            avg_base_std = float(np.mean(base_stats.std)) + EPSILON

            # 1. Standardized Mean Shift: ||mu_curr - mu_base||_2 / (sqrt(D) * avg_base_std)
            mean_diff = curr_stats.mean - base_stats.mean
            mean_dist = float(np.linalg.norm(mean_diff))
            norm_mean_shift = mean_dist / (np.sqrt(dim) * avg_base_std)
            score_mean = 1.0 - np.exp(-1.5 * norm_mean_shift)

            # 2. Per-Dimension Variance / Stddev Shift: (1/D) sum(|sigma_curr_i - sigma_base_i| / (sigma_base_i + eps))
            std_diff = np.abs(curr_stats.std - base_stats.std)
            rel_std_diff = float(np.mean(std_diff / (base_stats.std + EPSILON)))
            score_std = 1.0 - np.exp(-1.5 * rel_std_diff)

            # 3. Vector Norm Distribution Shift: |norm_mean_curr - norm_mean_base| / (norm_std_base + eps)
            norm_diff = abs(curr_stats.norm_mean - base_stats.norm_mean)
            norm_std_base = base_stats.norm_std if base_stats.norm_std > EPSILON else avg_base_std
            norm_shift = norm_diff / (norm_std_base + EPSILON)
            score_norm = 1.0 - np.exp(-1.5 * norm_shift)

            # Weighted combination
            composite_score = float(0.50 * score_mean + 0.30 * score_std + 0.20 * score_norm)
            composite_score = max(0.0, min(1.0, composite_score))
            score_rounded = round(composite_score, 4)

            if score_rounded >= self.severe_threshold:
                status = "drift_detected"
            elif score_rounded >= self.threshold:
                status = "warning"
            else:
                status = "normal"

            self._last_score = score_rounded
            self._last_status = status
            self._last_evaluated_at = now_iso

            return {
                "status": status,
                "score": score_rounded,
                "threshold": self.threshold,
                "severe_threshold": self.severe_threshold,
                "sample_count": curr_stats.sample_count,
                "timestamp": now_iso,
                "baseline_statistics": base_stats.to_dict(),
                "current_statistics": curr_stats.to_dict()
            }

    def reset(self, new_baseline: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Manually resets the baseline and clears the sample buffer.
        """
        with self._lock:
            self._sample_buffer.clear()
            if new_baseline is not None and new_baseline.size > 0:
                arr = np.asarray(new_baseline, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                self._baseline_stats = DistributionStatistics(arr)
            else:
                self._baseline_stats = None

            self._last_score = 0.0
            self._last_status = "normal"
            self._last_evaluated_at = datetime.now(timezone.utc).isoformat()

            return {
                "status": "normal",
                "score": 0.0,
                "threshold": self.threshold,
                "severe_threshold": self.severe_threshold,
                "sample_count": 0,
                "timestamp": self._last_evaluated_at,
                "baseline_statistics": self._baseline_stats.to_dict() if self._baseline_stats else None,
                "message": "Drift baseline reset successfully."
            }
