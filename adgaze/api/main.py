"""FastAPI analytics API for AdGaze.

Exposes endpoints for querying engagement events, tracks, and session summaries.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from ..storage.db import Storage

_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage("data/adgaze.db")
    return _storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_storage()
    yield


app = FastAPI(
    title="AdGaze Analytics API",
    description="Real-time advertisement engagement analytics",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/events/recent")
def recent_events(limit: int = Query(100, ge=1, le=1000)):
    """Get most recent frame events."""
    return get_storage().get_recent_events(limit)


@app.get("/api/stats/stage-distribution")
def stage_distribution(since: float = Query(0, ge=0)):
    """Get distribution of how far frames progress through the cascade."""
    return get_storage().get_stage_distribution(since)


@app.get("/api/stats/person-count")
def person_count_series(
    since: float = Query(0, ge=0),
    bucket_sec: int = Query(60, ge=1, le=3600),
):
    """Get person count time series (bucketed)."""
    return get_storage().get_person_count_series(since, bucket_sec)


@app.get("/api/stats/latencies")
def avg_latencies(since: float = Query(0, ge=0)):
    """Get average per-stage latencies."""
    return get_storage().get_avg_latencies(since)


@app.get("/api/tracks")
def tracks(limit: int = Query(50, ge=1, le=500)):
    """Get recent person tracks."""
    return get_storage().get_tracks(limit)


@app.get("/api/sessions")
def sessions(limit: int = Query(20, ge=1, le=100)):
    """Get session summaries."""
    return get_storage().get_sessions(limit)
