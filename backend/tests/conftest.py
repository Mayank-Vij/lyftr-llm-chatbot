"""Test harness: a fake OpenAI-compatible provider so the *real* LiteLLM path
(keys, retries, cache, usage) is exercised without network access or real keys.

API keys named like ``bad429`` / ``bad500`` make the fake provider answer with
that HTTP status; any other key succeeds.
"""
from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.app = FastAPI()

        @self.app.post("/v1/chat/completions")
        async def completions(request: Request):
            key = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            body = await request.json()
            self.calls.append({"key": key, "messages": body["messages"]})
            if key.startswith("bad"):
                status = int(key.removeprefix("bad").split("_")[0] or 500)
                return JSONResponse({"error": {"message": f"boom {status}"}}, status_code=status)
            msgs = body["messages"]
            users = [m["content"] for m in msgs if m["role"] == "user"]
            # reply reveals what the model "saw", so tests can check memory
            reply = f"seen={len(users)}; last={users[-1]}; first={users[0]}"
            if any(m["role"] == "system" for m in msgs):
                reply += f"; sys={msgs[0]['content']}"
            prompt_tokens = sum(len(m["content"].split()) for m in msgs)
            return {
                "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": len(reply.split()),
                          "total_tokens": prompt_tokens + len(reply.split())},
            }


@pytest.fixture(scope="session")
def provider():
    fake = FakeProvider()
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(fake.app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    fake.base_url = f"http://127.0.0.1:{port}/v1"
    yield fake
    server.should_exit = True


@pytest.fixture
def make_client(tmp_path, monkeypatch, provider):
    clients: list[TestClient] = []

    def _make(keys="good_a,good_b", simulation=True, **env) -> TestClient:
        monkeypatch.setenv("LLM_API_KEYS", keys)
        monkeypatch.setenv("LLM_MODEL", "openai/fake-model")
        monkeypatch.setenv("LLM_API_BASE", provider.base_url)
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/app.db")
        monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "true" if simulation else "false")
        monkeypatch.setenv("LITELLM_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv("KEY_COOLDOWN_SECONDS", "0")
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
        from app.main import create_app

        client = TestClient(create_app(), raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        provider.calls.clear()
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()
