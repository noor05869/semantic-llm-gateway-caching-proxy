"""
⚡ Production Semantic LLM Gateway (main.py)

OpenAI-compatible FastAPI microservice proxy featuring:
1. POST /v1/chat/completions (Full OpenAI-compatible payload & response).
2. Semantic caching integration (Tier 0 exact + Tier 1 vector match).
3. Complexity routing & real-time token cost calculation.
4. Automated 429/503 failover protection.
5. Server-Sent Events (SSE) streaming support (stream=True).
6. Performance & tokenomics response headers.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from cache_engine import SemanticCacheEngine
from cache_store import SemanticCacheStore
from router import ComplexityRouter, MODEL_CATALOG, DEFAULT_TIER_1_MODEL, DEFAULT_TIER_2_MODEL
from providers import GeminiProvider, LLMResponse

# =====================================================================
# Pydantic Schemas (OpenAI v1 Chat Completions Spec)
# =====================================================================

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


# =====================================================================
# Gateway Application State & Lifespan
# =====================================================================

cache_store: Optional[SemanticCacheStore] = None
router: Optional[ComplexityRouter] = None
provider: Optional[GeminiProvider] = None


def init_gateway_state(threshold: Optional[float] = None) -> None:
    """Initialize or reset gateway components."""
    global cache_store, router, provider
    if threshold is None:
        threshold = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.88"))
    ttl = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", None)

    engine = SemanticCacheEngine(threshold=threshold)
    cache_store = SemanticCacheStore(
        engine=engine,
        threshold=threshold,
        default_ttl=ttl,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
    )
    router = ComplexityRouter()
    provider = GeminiProvider()


# Initialize state at module load
init_gateway_state()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown."""
    init_gateway_state()

    print("\n" + "=" * 65)
    print("⚡ Semantic LLM Gateway Initialized")
    print(f"• Active Storage Backend: [{cache_store.active_backend}]")
    print(f"• Semantic Threshold:     [{cache_store.threshold}]")
    print(f"• Provider Configured:    [{provider.is_configured}]")
    print("=" * 65 + "\n")

    yield

    print("⚡ Gateway shutting down...")


app = FastAPI(
    title="⚡ Semantic LLM Gateway & Caching Proxy",
    description="Enterprise-grade caching, complexity routing, and failover proxy.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Core Endpoints
# =====================================================================

@app.get("/health")
def health_check():
    """Service health and diagnostic status."""
    return {
        "status": "healthy",
        "storage_backend": cache_store.active_backend if cache_store is not None else "uninitialized",
        "cached_entries_count": len(cache_store) if cache_store is not None else 0,
        "provider_configured": provider.is_configured if provider is not None else False,
    }


@app.get("/v1/models")
def list_models():
    """List supported models for OpenAI SDK compatibility."""
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "owned_by": "gateway"}
            for model_id in MODEL_CATALOG.keys()
        ],
    }


async def generate_sse_stream(
    full_text: str,
    model: str,
    completion_id: str,
) -> AsyncGenerator[str, None]:
    """Stream response words as OpenAI-compatible SSE chunks."""
    words = full_text.split(" ")
    for idx, word in enumerate(words):
        chunk_text = word if idx == len(words) - 1 else word + " "
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk_text},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Send final stop chunk
    stop_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, response: Response):
    """
    OpenAI-compatible chat completion endpoint.
    Orchestrates: Cache -> Complexity Router -> Failover Provider -> Cache Update.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    t_start = time.perf_counter()

    # Extract user prompt (last user message)
    user_prompt = ""
    for m in reversed(req.messages):
        if m.role == "user":
            user_prompt = m.content
            break
    if not user_prompt:
        user_prompt = req.messages[-1].content

    # -------------------------------------------------------------
    # Step 1: Semantic Cache Lookup
    # -------------------------------------------------------------
    cache_result = cache_store.lookup(user_prompt)

    if cache_result.hit and cache_result.entry:
        entry = cache_result.entry
        total_latency = (time.perf_counter() - t_start) * 1000
        completion_id = f"chatcmpl-cache-{uuid.uuid4().hex[:8]}"

        # Set transparency headers
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Lookup-Type"] = cache_result.lookup_type
        response.headers["X-Similarity-Score"] = f"{cache_result.similarity_score:.4f}"
        response.headers["X-Model-Used"] = entry.model_used
        response.headers["X-Latency-MS"] = f"{total_latency:.2f}"
        response.headers["X-Cost-USD"] = "0.000000"
        response.headers["X-Savings-USD"] = f"{entry.cost_usd:.6f}"
        response.headers["X-Failover"] = "false"

        if req.stream:
            return StreamingResponse(
                generate_sse_stream(entry.response_text, entry.model_used, completion_id),
                media_type="text/event-stream",
                headers=dict(response.headers),
            )

        return ChatCompletionResponse(
            id=completion_id,
            model=entry.model_used,
            choices=[
                ChatCompletionResponseChoice(
                    message=ChatMessage(role="assistant", content=entry.response_text),
                )
            ],
            usage=UsageInfo(
                prompt_tokens=entry.prompt_tokens,
                completion_tokens=entry.completion_tokens,
                total_tokens=entry.prompt_tokens + entry.completion_tokens,
            ),
        )

    # -------------------------------------------------------------
    # Step 2: Cache Miss -> Complexity Routing
    # -------------------------------------------------------------
    routing_decision = router.route(user_prompt, explicit_model=req.model)
    target_model = routing_decision.selected_model

    # -------------------------------------------------------------
    # Step 3: Provider Execution with Automated Failover
    # -------------------------------------------------------------
    llm_resp: LLMResponse = provider.generate(
        prompt=user_prompt,
        primary_model=target_model,
        allow_failover=True,
    )

    # Calculate tokenomics and savings
    cost_data = router.calculate_savings(
        actual_model=llm_resp.model_used,
        prompt_tokens=llm_resp.prompt_tokens,
        completion_tokens=llm_resp.completion_tokens,
    )

    # -------------------------------------------------------------
    # Step 4: Write to Semantic Cache for Future Requests
    # -------------------------------------------------------------
    cache_store.set(
        prompt=user_prompt,
        response_text=llm_resp.text,
        model_used=llm_resp.model_used,
        prompt_tokens=llm_resp.prompt_tokens,
        completion_tokens=llm_resp.completion_tokens,
        cost_usd=cost_data["actual_cost_usd"],
    )

    total_latency = (time.perf_counter() - t_start) * 1000
    completion_id = f"chatcmpl-live-{uuid.uuid4().hex[:8]}"

    # Set response headers
    response.headers["X-Cache"] = "MISS"
    response.headers["X-Lookup-Type"] = "MISS"
    response.headers["X-Similarity-Score"] = f"{cache_result.similarity_score:.4f}"
    response.headers["X-Model-Used"] = llm_resp.model_used
    response.headers["X-Latency-MS"] = f"{total_latency:.2f}"
    response.headers["X-Cost-USD"] = f"{cost_data['actual_cost_usd']:.6f}"
    response.headers["X-Savings-USD"] = f"{cost_data['dollars_saved']:.6f}"
    response.headers["X-Failover"] = "true" if llm_resp.is_failover else "false"

    if req.stream:
        return StreamingResponse(
            generate_sse_stream(llm_resp.text, llm_resp.model_used, completion_id),
            media_type="text/event-stream",
            headers=dict(response.headers),
        )

    return ChatCompletionResponse(
        id=completion_id,
        model=llm_resp.model_used,
        choices=[
            ChatCompletionResponseChoice(
                message=ChatMessage(role="assistant", content=llm_resp.text),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=llm_resp.prompt_tokens,
            completion_tokens=llm_resp.completion_tokens,
            total_tokens=llm_resp.total_tokens,
        ),
    )


# =====================================================================
# Standalone Verification Suite
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    from fastapi.testclient import TestClient

    print("=" * 65)
    print("FastAPI Semantic Gateway: Verification Suite")
    print("=" * 65)

    with TestClient(app) as client:
        # Clear cache for a clean verification run
        if cache_store is not None:
            cache_store.clear()

        # Test 1: Health Check
        print("\n--- Test 1: Health Check Endpoint ---")
        health = client.get("/health").json()
        print(f"Health Response: {health}")
        assert health["status"] == "healthy"
        assert health["storage_backend"] in ("Redis", "In-Memory (Local Vector Fallback)")
        print(f"✅ Health Check Passed! Active backend: [{health['storage_backend']}]")

        # Test 2: First Request (Must be a Cache MISS)
        print("\n--- Test 2: First Query (Expected: Cache MISS) ---")
        payload = {
            "messages": [{"role": "user", "content": "What is the capital of Canada?"}]
        }
        t0 = time.perf_counter()
        r1 = client.post("/v1/chat/completions", json=payload)
        ms1 = (time.perf_counter() - t0) * 1000

        print(f"Status Code:      {r1.status_code}")
        print(f"X-Cache Header:   {r1.headers.get('x-cache')}")
        print(f"X-Model-Used:     {r1.headers.get('x-model-used')}")
        print(f"X-Cost-USD:       ${r1.headers.get('x-cost-usd')}")
        print(f"Total Latency:    {ms1:.1f}ms")
        print(f"Answer:           \"{r1.json()['choices'][0]['message']['content'].strip()[:60]}...\"")
        assert r1.headers.get("x-cache") == "MISS"
        print("✅ Cache MISS Verified!")

        # Test 3: Second Request (Semantically Equivalent -> Must be a Cache HIT)
        print("\n--- Test 3: Semantic Variation (Expected: Cache HIT in < 15ms) ---")
        paraphrased_payload = {
            "messages": [{"role": "user", "content": "Tell me the capital city of Canada."}]
        }
        t0 = time.perf_counter()
        r2 = client.post("/v1/chat/completions", json=paraphrased_payload)
        ms2 = (time.perf_counter() - t0) * 1000

        print(f"Status Code:      {r2.status_code}")
        print(f"X-Cache Header:   {r2.headers.get('x-cache')}")
        print(f"X-Lookup-Type:    {r2.headers.get('x-lookup-type')}")
        print(f"X-Similarity:     {r2.headers.get('x-similarity-score')}")
        print(f"X-Cost-USD:       ${r2.headers.get('x-cost-usd')}")
        print(f"Total Latency:    {ms2:.1f}ms")
        print(f"Cached Answer:    \"{r2.json()['choices'][0]['message']['content'].strip()[:60]}...\"")
        assert r2.headers.get("x-cache") == "HIT"
        print(f"⚡ SPEEDUP: {ms1:.0f}ms -> {ms2:.1f}ms ({ms1 / max(1, ms2):.1f}x faster!)")
        print("✅ Semantic Cache HIT Verified!")

        # Test 4: Streaming Test (SSE stream=True)
        print("\n--- Test 4: SSE Streaming (stream=True) ---")
        stream_payload = {
            "messages": [{"role": "user", "content": "What is the capital of Canada?"}],
            "stream": True,
        }
        r3 = client.post("/v1/chat/completions", json=stream_payload)
        lines = [line for line in r3.text.split("\n") if line.startswith("data: ")]
        print(f"Received {len(lines)} streaming chunks.")
        print(f"First chunk: {lines[0]}")
        print(f"Last chunk:  {lines[-1]}")
        assert lines[-1] == "data: [DONE]"
        print("✅ Streaming Response Verified!")

    print("\n" + "=" * 65)
    print("🎯 Step 5 Verification Complete: Full OpenAI-Compatible Proxy Operational!")
    print("=" * 65)
