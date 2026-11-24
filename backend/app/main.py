"""FastAPI application: /health, /chat, /sessions/{id}/messages."""
from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import load_settings
from .db import Database
from .llm import AllProvidersUnavailable, LLMGateway, parse_failure_header
from .schemas import ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("chatbot")


class _SessionLocks:
    """One lock per session so concurrent requests can't interleave history."""

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def get(self, session_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(session_id, threading.Lock())


def create_app() -> FastAPI:
    settings = load_settings()
    db = Database(settings.db_path)
    gateway: LLMGateway | None = None
    locks = _SessionLocks()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal gateway
        db.init()  # schema is created automatically on startup
        gateway = LLMGateway(settings)
        log.info(
            "started: keys=%d model=%s cache=%s simulation=%s db=%s",
            gateway.keys_configured, settings.model, gateway.cache_type,
            settings.enable_failure_simulation, settings.db_path,
        )
        if gateway.keys_configured < 2:
            log.warning("fewer than 2 keys in LLM_API_KEYS; failover is not possible")
        yield

    app = FastAPI(title="LLM Chatbot with Key Failover and Caching", lifespan=lifespan)

    @app.middleware("http")
    async def never_leak_errors(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception:  # last-resort guard: JSON, never a stack trace
            log.exception("unhandled error")
            return JSONResponse({"detail": "internal server error"}, status_code=500)

    origins = settings.allowed_origins
    app.add_middleware(  # added last => outermost, so CORS headers cover errors too
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    @app.get("/health")
    def health():
        ok_db = db.ping()
        keys = gateway.keys_configured if gateway else 0
        body = {
            "status": "ok" if ok_db and keys > 0 else "degraded",
            "keys_configured": keys,
            "cache_enabled": gateway is not None,
        }
        return JSONResponse(body, status_code=200 if ok_db and keys > 0 else 503)

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        req: ChatRequest,
        x_simulate_failure: str | None = Header(default=None),
    ):
        started = time.perf_counter()
        simulate = (
            parse_failure_header(x_simulate_failure)
            if settings.enable_failure_simulation
            else {}
        )
        assert gateway is not None

        with locks.get(req.session_id):
            history = db.history(req.session_id, limit=settings.max_history_messages)
            system_prompt = (
                req.system_prompt
                or db.get_system_prompt(req.session_id)
                or settings.default_system_prompt
            )
            messages = (
                [{"role": "system", "content": system_prompt}]
                + [{"role": m["role"], "content": m["content"]} for m in history]
                + [{"role": "user", "content": req.message}]
            )

            try:
                result = gateway.complete(messages, simulate)
            except AllProvidersUnavailable:
                raise HTTPException(status_code=503, detail="all providers unavailable")

            latency_ms = max(1, int(round((time.perf_counter() - started) * 1000)))
            metadata = {
                "resolution_source": result.resolution_source,
                "model": result.model,
                "latency_ms": latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "key_id": result.key_id,
                "fallback_attempts": result.fallback_attempts,
            }
            db.save_exchange(
                req.session_id, req.system_prompt, req.message, result.reply, metadata
            )

        return {"session_id": req.session_id, "reply": result.reply, "metadata": metadata}

    @app.get("/sessions/{session_id}/messages")
    def session_messages(session_id: str):
        if not db.session_exists(session_id):
            raise HTTPException(status_code=404, detail="session not found")
        return {"session_id": session_id, "messages": db.history(session_id)}

    return app


app = create_app()
