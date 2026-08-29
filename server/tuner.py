import os
import json
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Literal, Tuple, Union
from pydantic import BaseModel, Field, AliasChoices, model_validator, field_serializer
import numpy as np

# Safe parameter bounds
PARAMETER_BOUNDS = {
    "ef_search": {"min": 1, "max": 10000, "runtime_tunable": True},
    "m": {"min": 2, "max": 128, "runtime_tunable": False},
    "ef_construction": {"min": 4, "max": 2000, "runtime_tunable": False},
    "pq_subvectors": {"min": 1, "max": 512, "runtime_tunable": False},
    "pq_nbits": {"min": 1, "max": 16, "runtime_tunable": False}
}

ALLOWED_PARAMETERS = set(PARAMETER_BOUNDS.keys())

# Pydantic models for AI Tuner
class TuningTarget(BaseModel):
    target_recall: Optional[float] = Field(0.95, ge=0.0, le=1.0, description="Target Recall@K objective (0.0 to 1.0)")
    max_p95_latency_ms: Optional[float] = Field(20.0, gt=0.0, description="Maximum permissible P95 query latency in ms")
    priority: Literal["recall", "latency", "balanced"] = Field("balanced", description="Optimization trade-off priority")

class ExpectedEffect(BaseModel):
    recall: Optional[str] = Field("neutral", description="Predicted effect on recall ('increase', 'decrease', 'neutral')")
    latency: Optional[str] = Field("neutral", description="Predicted effect on latency ('increase', 'decrease', 'neutral')")
    memory: Optional[str] = Field("neutral", description="Predicted effect on memory usage")

class TuningChanges(BaseModel):
    ef_search: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("ef_search", "efSearch"),
        ge=1,
        le=10000,
        description="HNSW runtime candidate search beam width"
    )
    m: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("m", "M"),
        ge=2,
        le=128,
        description="HNSW max outgoing connections per node (rebuild required)"
    )
    ef_construction: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("ef_construction", "efConstruction"),
        ge=4,
        le=2000,
        description="HNSW construction beam width (rebuild required)"
    )
    pq_subvectors: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("pq_subvectors", "pqSubvectors"),
        ge=1,
        le=512,
        description="PQ subvector count (rebuild required)"
    )
    pq_nbits: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("pq_nbits", "pqNbits"),
        ge=1,
        le=16,
        description="PQ bit quantization depth (rebuild required)"
    )

    model_config = {
        "populate_by_name": True,
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "ef_search": 120
            }
        }
    }

    def to_dict(self) -> Dict[str, int]:
        return {k: v for k, v in self.model_dump(exclude_none=True).items() if v is not None}

    def __getitem__(self, item: str) -> int:
        norm_key = item.lower().replace("efsearch", "ef_search").replace("efconstruction", "ef_construction")
        val = getattr(self, norm_key, None)
        if val is None:
            raise KeyError(item)
        return val

    def __contains__(self, item: str) -> bool:
        norm_key = item.lower().replace("efsearch", "ef_search").replace("efconstruction", "ef_construction")
        return getattr(self, norm_key, None) is not None

    def get(self, item: str, default: Any = None) -> Any:
        norm_key = item.lower().replace("efsearch", "ef_search").replace("efconstruction", "ef_construction")
        val = getattr(self, norm_key, None)
        return val if val is not None else default

    def items(self):
        return self.to_dict().items()

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

class TuningRecommendation(BaseModel):
    action: Literal["no_change", "tune", "rebuild_required"] = Field(..., description="Action recommendation")
    reason: str = Field(..., description="Technical explanation of the tuning recommendation")
    changes: TuningChanges = Field(default_factory=TuningChanges, description="Explicit parameter updates")
    expected_effect: ExpectedEffect = Field(default_factory=ExpectedEffect, description="Predicted performance trade-offs")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of recommendation (0.0 to 1.0)")
    rebuild_required: bool = Field(False, description="True if recommended changes require index reconstruction")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
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

    @field_serializer("changes")
    def serialize_changes(self, changes: TuningChanges, _info):
        return changes.to_dict()

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if isinstance(data, dict):
            raw_changes = data.get("changes")
            if isinstance(raw_changes, dict):
                norm_changes = {}
                for k, v in raw_changes.items():
                    norm_k = k.lower().replace("efsearch", "ef_search").replace("efconstruction", "ef_construction")
                    norm_changes[norm_k] = int(v) if v is not None else None
                data["changes"] = norm_changes
        return data

class ValidationResult(BaseModel):
    status: Literal["improved", "acceptable", "regressed", "failed", "rolled_back", "pending"] = Field("pending", description="Empirical validation status")
    decision: Literal["ACCEPT", "REJECT", "ROLLED_BACK", "PENDING"] = Field("PENDING", description="Validation gate decision")
    reason: str = Field(..., description="Explanation of measured validation outcome vs prediction")
    target_objective: str = Field("balanced", description="Target objective evaluated")
    before_metrics: Dict[str, Any] = Field(default_factory=dict, description="Pre-apply baseline or previous benchmark metrics")
    after_metrics: Dict[str, Any] = Field(default_factory=dict, description="Post-apply measured benchmark metrics")
    deltas: Dict[str, float] = Field(default_factory=dict, description="Percentage metric changes observed")
    validated_at: Optional[str] = Field(None, description="ISO timestamp of validation execution")

class TuningHistoryEntry(BaseModel):
    timestamp: str
    target: TuningTarget
    old_configuration: Dict[str, Any]
    recommendation: TuningRecommendation
    new_configuration: Optional[Dict[str, Any]] = None
    applied: bool = False
    telemetry_before: Dict[str, Any]
    telemetry_after: Optional[Dict[str, Any]] = None
    validation: Optional[ValidationResult] = None

class VectorForgeTuner:
    """
    AI-assisted index tuning orchestrator powered by Gemini with closed-loop empirical validation (Phase 8).
    Analyzes telemetry, drift signals, and search objectives to recommend, safely apply, and validate configuration updates against real benchmarks.
    Singleton pattern ensures atomic state sharing across threads and endpoints.
    """
    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super(VectorForgeTuner, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        if getattr(self, "_initialized", False):
            return
        with self._singleton_lock:
            if getattr(self, "_initialized", False):
                return
            self.model_name = model_name
            self._lock = threading.Lock()
            self.history: List[TuningHistoryEntry] = []
            self._rollback_stack: List[Dict[str, Any]] = []
            self.last_recommendation: Optional[TuningRecommendation] = None
            self.last_applied_change: Optional[Dict[str, Any]] = None
            self.recommendation_status: Literal["idle", "proposed", "applied", "validated", "regressed", "rolled_back"] = "idle"
            self.last_validation: Optional[Dict[str, Any]] = None
            self.active_snapshot: Optional[Dict[str, Any]] = None
            self._initialized = True

    def _get_gemini_client(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        try:
            from google import genai
            return genai.Client(api_key=api_key)
        except Exception:
            return None

    def validate_recommendation(self, rec: TuningRecommendation) -> Tuple[bool, str, bool]:
        """
        Validates safety bounds and determines whether any change requires index rebuilding.
        Returns: (is_valid, error_message, requires_rebuild)
        """
        changes_dict = rec.changes.to_dict() if isinstance(rec.changes, TuningChanges) else rec.changes
        for param, val in changes_dict.items():
            norm_key = param.lower().replace("efsearch", "ef_search").replace("efconstruction", "ef_construction")
            if norm_key not in ALLOWED_PARAMETERS:
                return False, f"Unsupported parameter: {param}", False
            bounds = PARAMETER_BOUNDS[norm_key]
            if not isinstance(val, int) or val < bounds["min"] or val > bounds["max"]:
                return False, f"Parameter {param}={val} violates safety bounds [{bounds['min']}, {bounds['max']}]", False
            if not bounds["runtime_tunable"]:
                return True, "Parameter requires index rebuild", True

        return True, "Valid", False

    def generate_prompt(
        self,
        telemetry: Dict[str, Any],
        drift: Dict[str, Any],
        current_config: Dict[str, Any],
        target: TuningTarget
    ) -> str:
        """
        Constructs the structured prompt for Gemini AI analysis without sending raw vectors.
        """
        system_snapshot = {
            "telemetry": telemetry,
            "drift_detection": drift,
            "current_configuration": current_config,
            "user_tuning_targets": target.model_dump(),
            "safety_bounds": PARAMETER_BOUNDS
        }

        return f"""You are the AI Performance Optimization Agent for VectorForge, a high-performance C++20 vector search engine.
Analyze the following runtime telemetry, statistical drift status, and current index configuration to optimize vector search performance.

Current System Snapshot:
{json.dumps(system_snapshot, indent=2)}

Tuning Directives & Rules:
1. Trade-Off Principles:
   - 'ef_search': Higher values increase graph search beam width, improving recall at the cost of higher latency. Lower values decrease latency at the cost of recall. (Runtime changeable).
   - 'm' & 'ef_construction': Structural HNSW build parameters. Modifying them requires rebuilding the index ('action': 'rebuild_required').
   - If drift is detected, evaluate whether current ef_search is adequate for the shifting distribution.
2. Safety Bounds:
   - ef_search: [1, 10000]
   - m: [2, 128]
   - ef_construction: [4, 2000]
3. Required Output Format:
Return ONLY a valid JSON object with the following schema:
{{
  "action": "tune" | "no_change" | "rebuild_required",
  "reason": "<Concise technical justification>",
  "changes": {{
    "ef_search": <integer value if tuning>
  }},
  "expected_effect": {{
    "recall": "increase" | "decrease" | "neutral",
    "latency": "increase" | "decrease" | "neutral",
    "memory": "increase" | "decrease" | "neutral"
  }},
  "confidence": <float between 0.0 and 1.0>,
  "rebuild_required": <true if structural parameter changed, false otherwise>
}}
Do not include markdown codeblocks or extraneous text."""

    def recommend(
        self,
        telemetry: Dict[str, Any],
        drift: Dict[str, Any],
        current_config: Dict[str, Any],
        target: TuningTarget
    ) -> TuningRecommendation:
        """
        Queries Gemini to produce a performance optimization recommendation.
        Falls back to deterministic rule-based analysis if GEMINI_API_KEY is not configured.
        """
        client = self._get_gemini_client()

        if client is not None:
            try:
                prompt = self.generate_prompt(telemetry, drift, current_config, target)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                text_resp = response.text.strip()
                if text_resp.startswith("```json"):
                    text_resp = text_resp[7:]
                if text_resp.startswith("```"):
                    text_resp = text_resp[3:]
                if text_resp.endswith("```"):
                    text_resp = text_resp[:-3]
                text_resp = text_resp.strip()

                parsed_json = json.loads(text_resp)
                recommendation = TuningRecommendation(**parsed_json)
                valid, msg, req_rebuild = self.validate_recommendation(recommendation)
                if not valid:
                    raise ValueError(f"AI recommendation failed safety bounds: {msg}")
                if req_rebuild:
                    recommendation.rebuild_required = True
                    recommendation.action = "rebuild_required"

                with self._lock:
                    self.last_recommendation = recommendation
                    self.recommendation_status = "proposed"
                    self.last_validation = None
                return recommendation
            except Exception:
                pass

        # Deterministic Heuristic Fallback Analysis
        rec = self._heuristic_recommend(telemetry, drift, current_config, target)
        with self._lock:
            self.last_recommendation = rec
            self.recommendation_status = "proposed"
            self.last_validation = None
        return rec

    def _heuristic_recommend(
        self,
        telemetry: Dict[str, Any],
        drift: Dict[str, Any],
        current_config: Dict[str, Any],
        target: TuningTarget
    ) -> TuningRecommendation:
        """
        Robust rule-based reasoning engine ensuring safe recommendations when AI API is unavailable.
        """
        curr_ef_search = current_config.get("ef_search", 50)
        p95_ms = telemetry.get("latency", {}).get("p95_ms")
        max_p95 = target.max_p95_latency_ms or 20.0
        q_drift = drift.get("query_drift", {}).get("status", "normal")

        # Case 1: Drift detected or recall prioritized with latency headroom
        if (q_drift == "drift_detected" or target.priority == "recall") and (p95_ms is None or p95_ms < max_p95 * 0.7):
            new_ef = 96 if curr_ef_search == 50 else min(200, max(curr_ef_search + 40, int(curr_ef_search * 1.5)))
            if new_ef != curr_ef_search:
                return TuningRecommendation(
                    action="tune",
                    reason=f"Workload exhibits {'query drift' if q_drift == 'drift_detected' else 'recall priority'} while P95 latency ({p95_ms or 0:.2f}ms) has headroom below limit ({max_p95:.2f}ms). Increasing ef_search from {curr_ef_search} to {new_ef} to expand candidate graph coverage and maximize recall.",
                    changes=TuningChanges(ef_search=new_ef),
                    expected_effect=ExpectedEffect(recall="increase", latency="increase", memory="neutral"),
                    confidence=0.85,
                    rebuild_required=False
                )

        # Case 2: Latency exceeding budget -> throttle ef_search
        if p95_ms is not None and p95_ms > max_p95 and curr_ef_search > 20:
            new_ef = max(20, int(curr_ef_search * 0.75))
            return TuningRecommendation(
                action="tune",
                reason=f"Current P95 query latency ({p95_ms:.2f}ms) exceeds the target threshold ({max_p95:.2f}ms). Decreasing ef_search from {curr_ef_search} to {new_ef} to accelerate candidate traversal.",
                changes=TuningChanges(ef_search=new_ef),
                expected_effect=ExpectedEffect(recall="decrease", latency="decrease", memory="neutral"),
                confidence=0.88,
                rebuild_required=False
            )

        # Case 3: Balanced default recommendation
        if curr_ef_search < 80 and (p95_ms is None or p95_ms < max_p95 * 0.5):
            new_ef = 96 if curr_ef_search == 50 else min(120, curr_ef_search + 30)
            return TuningRecommendation(
                action="tune",
                reason=f"P95 latency is well within budget ({p95_ms or 0:.2f}ms < {max_p95:.2f}ms). Increasing ef_search from {curr_ef_search} to {new_ef} for higher graph exploration accuracy.",
                changes=TuningChanges(ef_search=new_ef),
                expected_effect=ExpectedEffect(recall="increase", latency="increase", memory="neutral"),
                confidence=0.80,
                rebuild_required=False
            )

        return TuningRecommendation(
            action="no_change",
            reason="Current index parameters operate within target recall and latency bounds.",
            changes=TuningChanges(),
            expected_effect=ExpectedEffect(recall="neutral", latency="neutral", memory="neutral"),
            confidence=0.90,
            rebuild_required=False
        )

    def apply(
        self,
        recommendation: TuningRecommendation,
        current_config: Dict[str, Any],
        telemetry_before: Dict[str, Any],
        apply_fn,
        benchmark_metrics_before: Optional[Dict[str, Any]] = None,
        target: Optional[TuningTarget] = None
    ) -> Dict[str, Any]:
        """
        Safely applies the validated recommendation using the provided callback.
        Saves the complete previous configuration to the rollback stack before modifying state.
        Creates a Phase 8 snapshot of baseline metrics and proposed targets for post-apply validation.
        """
        valid, msg, req_rebuild = self.validate_recommendation(recommendation)
        if not valid:
            return {
                "status": "error",
                "applied": False,
                "rebuild_required": False,
                "message": f"Recommendation rejected: {msg}",
                "configuration": current_config
            }

        if req_rebuild or recommendation.rebuild_required or recommendation.action == "rebuild_required":
            return {
                "status": "rejected",
                "applied": False,
                "rebuild_required": True,
                "message": "Proposed parameters modify structural graph topology (M / efConstruction) which require full index reconstruction. Live update skipped.",
                "configuration": current_config
            }

        changes_dict = recommendation.changes.to_dict() if isinstance(recommendation.changes, TuningChanges) else recommendation.changes

        if recommendation.action == "no_change" or not changes_dict:
            return {
                "status": "success",
                "applied": False,
                "rebuild_required": False,
                "message": "No configuration changes required.",
                "configuration": current_config
            }

        with self._lock:
            # Save complete snapshot of previous configuration to rollback stack
            complete_previous_config = {
                "backend": current_config.get("backend", "hnsw"),
                "ef_search": int(current_config.get("ef_search", 50)),
                "m": int(current_config.get("m", 16)),
                "ef_construction": int(current_config.get("ef_construction", 100)),
                "dimension": int(current_config.get("dimension", 128))
            }
            self._rollback_stack.append(complete_previous_config)

            # Apply runtime changes
            new_config = apply_fn(changes_dict)
            self.last_applied_change = dict(changes_dict)
            self.recommendation_status = "applied"
            self.last_validation = None

            # Create Phase 8 Validation Snapshot
            self.active_snapshot = {
                "previous_configuration": complete_previous_config,
                "benchmark_metrics_before": benchmark_metrics_before or {},
                "target": (target or TuningTarget()).model_dump(),
                "recommendation": recommendation.model_dump(),
                "applied_at": datetime.now(timezone.utc).isoformat()
            }

            # Record history
            entry = TuningHistoryEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                target=target or TuningTarget(),
                old_configuration=complete_previous_config,
                recommendation=recommendation,
                new_configuration=new_config,
                applied=True,
                telemetry_before=telemetry_before,
                telemetry_after=None,
                validation=None
            )
            self.history.append(entry)

            return {
                "status": "success",
                "applied": True,
                "rebuild_required": False,
                "message": "Configuration parameters successfully applied.",
                "previous_configuration": complete_previous_config,
                "configuration": new_config
            }

    def validate_benchmark(self, after_metrics: Dict[str, Any], current_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Closed-loop Phase 8 validation: Compares post-apply benchmark measurements against baseline/pre-apply metrics.
        Determines whether the recommendation IMPROVED, is ACCEPTABLE, or REGRESSED.
        """
        with self._lock:
            if self.recommendation_status != "applied" or not self.active_snapshot:
                return None

            before_metrics = self.active_snapshot.get("benchmark_metrics_before", {})
            target_info = self.active_snapshot.get("target", {})
            rec_target_recall = target_info.get("target_recall", 0.95) or 0.95
            max_p95 = target_info.get("max_p95_latency_ms", 20.0) or 20.0
            priority = target_info.get("priority", "balanced")

            # Calculate precise percentage deltas
            before_rec = before_metrics.get("recall", after_metrics.get("recall", 0.916))
            after_rec = after_metrics.get("recall", 0.916)
            rec_delta_pct = round(((after_rec - before_rec) / before_rec) * 100, 2) if before_rec > 0 else 0.0

            before_p95 = before_metrics.get("p95_latency_ms", after_metrics.get("p95_latency_ms", 0.16))
            after_p95 = after_metrics.get("p95_latency_ms", 0.16)
            lat_delta_pct = round(((after_p95 - before_p95) / before_p95) * 100, 2) if before_p95 > 0 else 0.0

            before_qps = before_metrics.get("qps", after_metrics.get("qps", 5000.0))
            after_qps = after_metrics.get("qps", 5000.0)
            qps_delta_pct = round(((after_qps - before_qps) / before_qps) * 100, 2) if before_qps > 0 else 0.0

            before_mem = before_metrics.get("memory_mb", after_metrics.get("memory_mb", 1.28))
            after_mem = after_metrics.get("memory_mb", 1.28)
            mem_delta_pct = round(((after_mem - before_mem) / before_mem) * 100, 2) if before_mem > 0 else 0.0

            deltas = {
                "recall_pct": rec_delta_pct,
                "latency_pct": lat_delta_pct,
                "qps_pct": qps_delta_pct,
                "memory_pct": mem_delta_pct
            }

            # Decision Logic
            if priority == "recall":
                if after_rec >= before_rec and after_p95 <= max_p95:
                    status = "improved" if rec_delta_pct > 0.5 else "acceptable"
                    decision = "ACCEPT"
                    reason = f"Measured Recall@10 increased by {rec_delta_pct:+.1f}% ({before_rec*100:.1f}% -> {after_rec*100:.1f}%) with acceptable latency ({after_p95:.2f}ms <= {max_p95:.1f}ms limit)."
                elif after_p95 > max_p95:
                    status = "regressed"
                    decision = "REJECT"
                    reason = f"P95 query latency ({after_p95:.2f}ms) exceeded permissible target ({max_p95:.1f}ms)."
                else:
                    status = "regressed"
                    decision = "REJECT"
                    reason = f"Recall@10 dropped by {abs(rec_delta_pct):.1f}% below baseline."
            elif priority == "latency":
                if after_p95 <= before_p95 and after_rec >= (rec_target_recall - 0.05):
                    status = "improved" if lat_delta_pct < -0.5 else "acceptable"
                    decision = "ACCEPT"
                    reason = f"P95 query latency reduced by {abs(lat_delta_pct):.1f}% ({before_p95:.2f}ms -> {after_p95:.2f}ms) while meeting recall threshold."
                elif after_rec < (rec_target_recall - 0.08):
                    status = "regressed"
                    decision = "REJECT"
                    reason = f"Recall@10 dropped to {after_rec*100:.1f}%, violating target floor ({rec_target_recall*100:.1f}%)."
                else:
                    status = "acceptable"
                    decision = "ACCEPT"
                    reason = "Latency and recall operate within acceptable target bounds."
            else: # balanced
                if after_rec >= before_rec and after_p95 <= max_p95:
                    status = "improved" if (rec_delta_pct > 0.5 or lat_delta_pct < -0.5) else "acceptable"
                    decision = "ACCEPT"
                    reason = f"Balanced trade-off validated: Recall {before_rec*100:.1f}% -> {after_rec*100:.1f}%, P95 {before_p95:.2f}ms -> {after_p95:.2f}ms."
                elif after_p95 > max_p95:
                    status = "regressed"
                    decision = "REJECT"
                    reason = f"P95 latency ({after_p95:.2f}ms) exceeded allowable limit ({max_p95:.1f}ms)."
                else:
                    status = "regressed"
                    decision = "REJECT"
                    reason = f"Recall dropped by {abs(rec_delta_pct):.1f}%."

            val_res = {
                "status": status,
                "decision": decision,
                "reason": reason,
                "target_objective": priority,
                "before_metrics": before_metrics,
                "after_metrics": after_metrics,
                "deltas": deltas,
                "validated_at": datetime.now(timezone.utc).isoformat()
            }

            self.last_validation = val_res
            self.recommendation_status = status if status in ["improved", "acceptable", "regressed"] else "validated"
            
            # Update latest history entry
            if self.history:
                self.history[-1].validation = ValidationResult(**val_res)
                self.history[-1].telemetry_after = after_metrics

            return val_res

    def rollback(self, apply_fn) -> Dict[str, Any]:
        """
        Restores the previous configuration from the rollback stack.
        """
        with self._lock:
            if not self._rollback_stack:
                return {
                    "status": "error",
                    "applied": False,
                    "message": "No previous configuration available for rollback.",
                    "configuration": None
                }

            prev_config = self._rollback_stack.pop()

            # Restore using complete configuration dictionary
            restored_config = apply_fn(prev_config)
            
            # Update last_applied_change to reflect restored parameters
            self.last_applied_change = {
                "ef_search": restored_config.get("ef_search", 50)
            }
            self.recommendation_status = "rolled_back"
            if self.last_validation:
                self.last_validation["status"] = "rolled_back"
                self.last_validation["decision"] = "ROLLED_BACK"
                self.last_validation["reason"] += " (Rollback executed)"

            return {
                "status": "success",
                "applied": True,
                "message": "Configuration successfully rolled back to previous state.",
                "configuration": restored_config
            }

    def get_status(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns full tuner status, active configuration, last recommendation, and history length.
        """
        with self._lock:
            has_api_key = bool(os.environ.get("GEMINI_API_KEY"))
            return {
                "tuner_status": "online",
                "ai_provider": "Gemini" if has_api_key else "Heuristic Rule-Engine (GEMINI_API_KEY unset)",
                "gemini_api_key_configured": has_api_key,
                "model": self.model_name,
                "current_configuration": current_config,
                "last_recommendation": self.last_recommendation.model_dump() if self.last_recommendation else None,
                "recommendation_status": self.recommendation_status,
                "last_applied_change": self.last_applied_change,
                "last_validation": self.last_validation,
                "active_snapshot": self.active_snapshot,
                "tuning_history_count": len(self.history),
                "rollback_available": len(self._rollback_stack) > 0,
                "parameter_bounds": PARAMETER_BOUNDS
            }
