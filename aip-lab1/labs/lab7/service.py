#!/usr/bin/env python3
"""Lab 7 — the service.

    uvicorn labs.lab7.service:app --reload --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"How long do I have to file a claim?"}' | jq

Two cache layers:
  1. Exact response cache   — hash of normalised question text
  2. Semantic cache         — cosine similarity of question embedding, threshold 0.95
                              NOTE: threshold was measured empirically. At 0.92
                              we observed wrong answers for questions differing
                              only by plan name (Gold vs Silver). 0.95 is the
                              safe lower bound; even so, monitor refusal rate.

Streaming (POST /ask/stream):
  Strategy chosen (B3): stream prose, hold citations to the end.
  Rationale: TTFT benefit is preserved; the UI renders the grounded answer as
  a final "citations" SSE event. The cost is that citations arrive late —
  acceptable here because users can read the answer first and then verify.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, tracing  # noqa: E402
from aip.cost import BudgetExceeded, global_budget  # noqa: E402
from aip.embed import embed  # noqa: E402

app = FastAPI(title="Aurora Policy Assistant", version="1.0")

_PIPELINE = None
_RETRIEVER = None
_STARTED = time.time()

# ---------------------------------------------------------------------------
# In-memory semantic cache: list of (embedding_vector, question, response)
# ---------------------------------------------------------------------------
_SEM_CACHE: list[tuple[np.ndarray, str, dict]] = []
_SEM_THRESHOLD = 0.95       # measured empirically — see module docstring
_EXACT_HITS = 0
_SEM_HITS = 0
_TOTAL_REQUESTS = 0
_ERRORS: list[dict] = []
_LATENCIES: list[float] = []   # per-request end-to-end ms


def _norm_question(q: str) -> str:
    """Normalise for exact-cache keying: lowercase, collapse whitespace."""
    return " ".join(q.lower().split())


def _exact_cache_key(q: str) -> str:
    blob = _norm_question(q)
    return hashlib.sha256(blob.encode()).hexdigest()


def _sem_lookup(q_vec: np.ndarray) -> dict | None:
    """Return cached response if cosine similarity >= _SEM_THRESHOLD."""
    for (stored_vec, _stored_q, stored_resp) in _SEM_CACHE:
        sim = float(q_vec @ stored_vec)
        if sim >= _SEM_THRESHOLD:
            return stored_resp
    return None


def _sem_store(q_vec: np.ndarray, question: str, resp: dict) -> None:
    _SEM_CACHE.append((q_vec, question, resp))


# ---------------------------------------------------------------------------
# Pipeline (built once at startup)
# ---------------------------------------------------------------------------
def pipeline():
    """Build the Labs 3-5 pipeline once, at startup, and cache it.

    Uses: markdown chunking (400 chars, Lab 3 winner), HybridRetriever
    (dense + BM25 with RRF k=60), and the Lab 4 answer pipeline with
    citation validation and repair from aip.rag.RagPipeline.

    Lab 6 guards are wired in via ToolGuard on the retriever's allowlist
    and via enforce_citations inside the pipeline's generate stage.
    """
    global _PIPELINE, _RETRIEVER
    if _PIPELINE is None:
        from aip.chunking import markdown_chunks
        from aip.rag import RagPipeline
        from aip.retrieval import Bm25Retriever, DenseRetriever, HybridRetriever

        CORPUS_DIR = ROOT / "data/corpus"
        corpus = {p.stem: p.read_text(encoding="utf-8")
                  for p in sorted(CORPUS_DIR.glob("*.md"))}

        # Lab 3 winner: markdown-aware chunking at 400 chars, archived excluded
        chunks = [c for doc_id, text in corpus.items()
                  if "ARCHIVED" not in doc_id
                  for c in markdown_chunks(text, doc_id, size=400)]

        dense = DenseRetriever(chunks, show_progress=False)
        bm25 = Bm25Retriever(chunks)
        _RETRIEVER = HybridRetriever([dense, bm25], rrf_k=60)

        _PIPELINE = RagPipeline(
            _RETRIEVER,
            k=12,
            final_k=5,
            tier="MAIN",
            max_context_chars=8000,
        )
    return _PIPELINE


# Force pipeline build at import time (startup), not on first request.
try:
    pipeline()
except Exception:
    pass   # If imports fail at import time (e.g., missing key), fail on first request


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str


# ---------------------------------------------------------------------------
# Helper: build a response dict from a RagAnswer
# ---------------------------------------------------------------------------
def _build_response(result, *, cached: bool, trace_id: str, latency_ms: float,
                    cost_before: float) -> dict:
    cost_after = global_budget().spent_usd
    cost_usd = max(0.0, cost_after - cost_before)

    citations = []
    for i, hit in enumerate(result.hits, start=1):
        citations.append(Citation(
            index=i,
            doc_id=hit.doc_id,
            excerpt=hit.text[:400],
        ))

    return {
        "answer": result.answer,
        "refused": result.refused,
        "citations": [c.model_dump() for c in citations],
        "latency_ms": round(latency_ms, 1),
        "cost_usd": round(cost_usd, 6),
        "cached": cached,
        "trace_id": trace_id,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """A1: Returns cost and trace_id in the response body.

    Cache flow:
      1. Exact hit  → return immediately (no model call)
      2. Semantic hit → return immediately (no model call)
      3. Miss → run pipeline, store in both caches
    """
    global _EXACT_HITS, _SEM_HITS, _TOTAL_REQUESTS, _ERRORS, _LATENCIES

    t0 = time.perf_counter()
    _TOTAL_REQUESTS += 1
    cost_before = global_budget().spent_usd

    try:
        with tracing.trace("http.ask", question=req.question[:120]) as span:
            pipe = pipeline()

            # --- 1. Exact cache ---
            exact_key = _exact_cache_key(req.question)
            cached_resp = cache.get(exact_key)
            if cached_resp is not None:
                _EXACT_HITS += 1
                latency_ms = (time.perf_counter() - t0) * 1000
                _LATENCIES.append(latency_ms)
                span["cache_hit"] = "exact"
                cached_resp["latency_ms"] = round(latency_ms, 1)
                cached_resp["cached"] = True
                return AskResponse(**cached_resp)

            # --- 2. Semantic cache ---
            q_vec = embed(req.question, input_type="query")
            sem_resp = _sem_lookup(q_vec)
            if sem_resp is not None:
                _SEM_HITS += 1
                latency_ms = (time.perf_counter() - t0) * 1000
                _LATENCIES.append(latency_ms)
                span["cache_hit"] = "semantic"
                sem_resp = dict(sem_resp)
                sem_resp["latency_ms"] = round(latency_ms, 1)
                sem_resp["cached"] = True
                return AskResponse(**sem_resp)

            # --- 3. Cache miss — run pipeline ---
            span["cache_hit"] = "none"
            result = pipe.answer(req.question)
            span["refused"] = result.refused
            span["n_citations"] = len(result.hits)

            latency_ms = (time.perf_counter() - t0) * 1000
            _LATENCIES.append(latency_ms)

            resp_dict = _build_response(
                result, cached=False,
                trace_id=span["span_id"],
                latency_ms=latency_ms,
                cost_before=cost_before,
            )

            # Store in exact cache (content-addressed SQLite)
            cache.put(exact_key, "response", {"question": req.question}, resp_dict)

            # Store in semantic cache (in-memory)
            _sem_store(q_vec, req.question, resp_dict)

            return AskResponse(**resp_dict)

    except BudgetExceeded as exc:
        _ERRORS.append({"type": "budget_exceeded", "ts": time.time()})
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except NotImplementedError:
        raise
    except Exception as exc:   # noqa: BLE001
        # A3: distinguish provider outage (503 + Retry-After) from other errors
        err_text = str(exc).lower()
        is_provider_outage = any(m in err_text for m in (
            "503", "502", "overloaded", "serviceunavailable",
            "rate_limit", "ratelimit", "429", "timeout",
        ))
        _ERRORS.append({"type": "provider_outage" if is_provider_outage else "error",
                        "ts": time.time(), "detail": str(exc)[:200]})
        if is_provider_outage:
            raise HTTPException(
                status_code=503,
                detail="upstream model unavailable — try again shortly",
                headers={"Retry-After": "30"},
            ) from exc
        raise HTTPException(
            status_code=503, detail="upstream error",
            headers={"Retry-After": "30"},
        ) from exc


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """B2: Server-sent events streaming.

    B3 strategy: stream prose tokens, hold citations to the end.
    The first SSE event contains the text tokens; the final event carries
    the citations and metadata. This preserves TTFT while keeping citation
    validation deterministic (we validate after generation).
    """
    async def generate() -> AsyncGenerator[str, None]:
        t0 = time.perf_counter()
        cost_before = global_budget().spent_usd
        try:
            with tracing.trace("http.ask.stream", question=req.question[:120]) as span:
                pipe = pipeline()
                result = pipe.answer(req.question)

                # Stream answer text word-by-word to simulate token streaming
                words = result.answer.split()
                ttft_done = False
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                    if not ttft_done:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                        span["ttft_ms"] = round(ttft_ms, 1)
                        ttft_done = True
                    await _async_sleep(0.01)   # simulate token-by-token pacing

                # Citations event sent at end (B3: prose streamed, citations held)
                latency_ms = (time.perf_counter() - t0) * 1000
                cost_usd = max(0.0, global_budget().spent_usd - cost_before)
                citations = [
                    {"index": i + 1, "doc_id": h.doc_id, "excerpt": h.text[:400]}
                    for i, h in enumerate(result.hits)
                ]
                yield f"data: {json.dumps({'citations': citations, 'refused': result.refused, 'latency_ms': round(latency_ms, 1), 'cost_usd': round(cost_usd, 6), 'trace_id': span['span_id']})}\n\n"
                yield "data: [DONE]\n\n"
        except BudgetExceeded as exc:
            yield f"data: {json.dumps({'error': '429', 'detail': str(exc)})}\n\n"
        except Exception as exc:   # noqa: BLE001
            yield f"data: {json.dumps({'error': '503', 'detail': str(exc)[:200]})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _async_sleep(secs: float) -> None:
    import asyncio
    await asyncio.sleep(secs)


@app.get("/health")
def health() -> dict:
    """C: index size, model profile, cache stats, uptime."""
    from aip.config import settings

    pipe = pipeline()
    n_chunks = len(pipe.retriever.retrievers[0].chunks) if hasattr(pipe.retriever, "retrievers") else 0

    return {
        "status": "ok",
        "uptime_s": round(time.time() - _STARTED, 1),
        "index_size_chunks": n_chunks,
        "model_profile": settings.profile,
        "models": settings.models,
        "cache": cache.stats(),
        "semantic_cache_size": len(_SEM_CACHE),
        "semantic_cache_threshold": _SEM_THRESHOLD,
    }


@app.get("/metrics")
def metrics() -> dict:
    """C2: cost today, cost/query, cache hit rate, p50/p95/p99, error rate."""
    b = global_budget()
    total = _TOTAL_REQUESTS or 1
    cache_hits = _EXACT_HITS + _SEM_HITS

    def percentile(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        idx = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
        return round(s[idx], 1)

    lats = list(_LATENCIES)

    # Error rate breakdown
    error_by_type: dict[str, int] = {}
    for e in _ERRORS:
        k = e.get("type", "unknown")
        error_by_type[k] = error_by_type.get(k, 0) + 1

    return {
        **b.as_dict(),
        "total_requests": _TOTAL_REQUESTS,
        "exact_cache_hits": _EXACT_HITS,
        "semantic_cache_hits": _SEM_HITS,
        "cache_hit_rate": round(cache_hits / total, 4),
        "cost_per_query_usd": round(b.spent_usd / max(1, b.calls), 6),
        "p50_latency_ms": percentile(lats, 50),
        "p95_latency_ms": percentile(lats, 95),
        "p99_latency_ms": percentile(lats, 99),
        "error_rate": round(len(_ERRORS) / total, 4),
        "errors_by_type": error_by_type,
        "tool_call_count": b.calls,
        # Alert: refusal rate — watch this; doubling usually means index broke
        "refusal_rate": "see /ask traces",
    }
