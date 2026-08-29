import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import numpy as np

from server import VectorForgeIndex, simd_info

app = FastAPI(
    title="VectorForge Control Plane",
    description="FastAPI control plane and REST API for high-performance C++20 VectorForge vector search engine.",
    version="0.5.0"
)

# Global in-memory index state
class GlobalState:
    index: Optional[VectorForgeIndex] = None
    dimension: int = 128
    backend: str = "hnsw"
    m: int = 16
    ef_construction: int = 100
    ef_search: int = 50

state = GlobalState()

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

class TelemetryResponse(BaseModel):
    status: str
    total_vectors: int
    dimension: int
    backend: str
    simd: Dict[str, Any]
    hnsw_parameters: Dict[str, Any]

# =========================================================================
# REST Endpoints
# =========================================================================

@app.get("/", tags=["Health"])
def root():
    return {
        "engine": "VectorForge",
        "version": "0.5.0",
        "status": "online",
        "docs_url": "/docs"
    }

@app.post("/vectors/insert", response_model=InsertResponse, status_code=status.HTTP_201_CREATED, tags=["Vectors"])
def insert_vectors(payload: InsertRequest):
    """
    Inserts a batch of floating-point vectors into the active index.
    """
    if not payload.vectors or len(payload.vectors) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'vectors' list cannot be empty."
        )

    first_len = len(payload.vectors[0])
    if first_len == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vector dimensionality cannot be 0."
        )

    for idx, v in enumerate(payload.vectors):
        if len(v) != first_len:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Inconsistent vector dimension at index {idx}: expected {first_len}, got {len(v)}."
            )

    # Initialize or match index dimension
    idx = get_or_create_index(first_len)

    try:
        np_arr = np.array(payload.vectors, dtype=np.float32)
        count = idx.insert(np_arr)
        return InsertResponse(
            inserted=count,
            total_vectors=idx.size(),
            dimension=idx.dimension()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insertion failed: {str(e)}"
        )

@app.post("/vectors/query", response_model=QueryResponse, tags=["Vectors"])
def query_vectors(payload: QueryRequest):
    """
    Executes approximate k-NN nearest neighbor search against the index.
    """
    idx = get_or_create_index()

    if idx.size() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Index is empty. Please insert vectors before querying."
        )

    if len(payload.vector) != idx.dimension():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query dimension mismatch: expected {idx.dimension()}, got {len(payload.vector)}."
        )

    try:
        np_query = np.array(payload.vector, dtype=np.float32)
        t0 = time.perf_counter_ns()
        raw_results = idx.search(np_query, payload.k)
        t1 = time.perf_counter_ns()

        latency_us = (t1 - t0) / 1000.0

        formatted_results = [
            SearchResultItem(id=r["id"], distance=float(r["distance"]))
            for r in raw_results
        ]

        return QueryResponse(
            results=formatted_results,
            query_latency_us=latency_us,
            k=payload.k
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query failed: {str(e)}"
        )

@app.get("/system/telemetry", response_model=TelemetryResponse, tags=["System"])
def get_system_telemetry():
    """
    Returns current index state, vector counts, active backend, and hardware SIMD capabilities.
    """
    idx = get_or_create_index()
    tele = idx.get_telemetry()

    return TelemetryResponse(
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
