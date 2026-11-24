import sqlite3
import uuid


def sid() -> str:
    return "s-" + uuid.uuid4().hex[:8]


def chat(client, message, session_id=None, headers=None, **extra):
    return client.post(
        "/chat",
        json={"session_id": session_id or sid(), "message": message, **extra},
        headers=headers or {},
    )


# ---- health ---------------------------------------------------------------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "keys_configured": 2, "cache_enabled": True}


# ---- basic contract --------------------------------------------------------
def test_chat_contract_shape(client):
    s = sid()
    r = chat(client, "What is the capital of France?", s)
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == s
    assert body["reply"]
    m = body["metadata"]
    assert m["resolution_source"] == "LIVE_INFERENCE"
    assert m["model"] == "openai/fake-model"
    assert isinstance(m["latency_ms"], int) and m["latency_ms"] >= 0
    assert m["input_tokens"] > 0 and m["output_tokens"] > 0
    assert m["key_id"] == "key_1"
    assert m["fallback_attempts"] == 0


# ---- validation ------------------------------------------------------------
def test_validation_422(client):
    assert client.post("/chat", json={"session_id": "a"}).status_code == 422
    assert client.post("/chat", json={"message": "hi"}).status_code == 422
    assert chat(client, "   ").status_code == 422
    assert chat(client, "x" * 4001).status_code == 422
    assert chat(client, "x" * 4000).status_code == 200
    assert client.post("/chat", json={"session_id": "", "message": "hi"}).status_code == 422
    assert client.post("/chat", json={"session_id": "a" * 129, "message": "hi"}).status_code == 422
    bad = client.post("/chat", content="{not json", headers={"content-type": "application/json"})
    assert bad.status_code == 422 and "detail" in bad.json()


def test_failed_requests_not_in_history(client):
    s = sid()
    assert chat(client, "   ", s).status_code == 422
    assert client.get(f"/sessions/{s}/messages").status_code == 404


# ---- multi-turn ------------------------------------------------------------
def test_multi_turn_memory_and_isolation(client, provider):
    a, b = sid(), sid()
    chat(client, "My name is Ada", a)
    r = chat(client, "What is my name?", a)
    assert "seen=2" in r.json()["reply"] and "first=My name is Ada" in r.json()["reply"]
    # the client only sends the new message; the model still saw both turns
    last = provider.calls[-1]["messages"]
    assert [m["role"] for m in last] == ["system", "user", "assistant", "user"]
    # other session is separate
    r = chat(client, "What is my name?", b)
    assert "seen=1" in r.json()["reply"]


def test_history_endpoint(client):
    s = sid()
    chat(client, "hello there", s)
    chat(client, "second one", s)
    r = client.get(f"/sessions/{s}/messages")
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "hello there"
    assert "metadata" not in msgs[0]
    md = msgs[1]["metadata"]
    assert {"resolution_source", "input_tokens", "output_tokens"} <= set(md)
    assert not any(m["role"] == "system" for m in msgs)


def test_unknown_session_404(client):
    r = client.get("/sessions/nope/messages")
    assert r.status_code == 404
    assert r.json() == {"detail": "session not found"}


def test_history_persists_in_sqlite(client, tmp_path):
    s = sid()
    chat(client, "persist me", s)
    con = sqlite3.connect(tmp_path / "app.db")
    rows = con.execute("SELECT role, content, metadata, created_at FROM messages WHERE session_id=?", (s,)).fetchall()
    assert [r[0] for r in rows] == ["user", "assistant"]
    assert rows[0][2] is None and rows[1][2] and rows[1][3]


def test_system_prompt_override_is_sticky(client):
    s = sid()
    r = chat(client, "hi", s, system_prompt="You are a pirate")
    assert "sys=You are a pirate" in r.json()["reply"]
    r = chat(client, "again", s)  # no prompt sent -> stored one still applies
    assert "sys=You are a pirate" in r.json()["reply"]
    r = chat(client, "new session", sid())
    assert "pirate" not in r.json()["reply"]


# ---- caching ---------------------------------------------------------------
def test_cache_hit_across_sessions(client, provider):
    first = chat(client, "Cache me please")
    assert first.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    n = len(provider.calls)
    second = chat(client, "Cache me please")  # new session id
    m1, m2 = first.json()["metadata"], second.json()["metadata"]
    assert m2["resolution_source"] == "LLM_CACHE"
    assert second.json()["reply"] == first.json()["reply"]
    assert m2["key_id"] is None and m2["fallback_attempts"] == 0
    assert (m2["input_tokens"], m2["output_tokens"]) == (m1["input_tokens"], m1["output_tokens"])
    assert len(provider.calls) == n  # provider not contacted
    assert m2["latency_ms"] <= m1["latency_ms"]


def test_different_message_never_cached(client):
    chat(client, "alpha question")
    r = chat(client, "beta question")
    assert r.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_depends_on_history(client):
    a, b = sid(), sid()
    chat(client, "one", a)
    chat(client, "other", b)
    r1 = chat(client, "same follow up", a)
    r2 = chat(client, "same follow up", b)
    assert r2.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert r1.json()["reply"] != r2.json()["reply"]


def test_tokens_are_per_request_not_cumulative(client):
    s = sid()
    first = chat(client, "aaa bbb ccc", s).json()["metadata"]["input_tokens"]
    for i in range(4):
        chat(client, f"filler {i}", s)
    other = chat(client, "ddd eee fff", sid()).json()["metadata"]["input_tokens"]
    assert other == first  # equal-length first messages => equal input tokens


# ---- failover --------------------------------------------------------------
def test_real_failover_on_429_and_500(make_client):
    for bad in ("bad429", "bad500", "bad503"):
        c = make_client(keys=f"{bad},good_b")
        r = chat(c, f"hello {bad}")
        assert r.status_code == 200
        m = r.json()["metadata"]
        assert m["key_id"] == "key_2" and m["fallback_attempts"] == 1
        assert m["resolution_source"] == "LIVE_INFERENCE"


def test_all_real_keys_fail_503(make_client):
    c = make_client(keys="bad429,bad500")
    s = sid()
    r = chat(c, "hello", s)
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}
    assert c.get(f"/sessions/{s}/messages").status_code == 404


def test_simulated_failover(client):
    for header in ("key_1=429", "key_1=500", "key_1=502", "key_1=503"):
        r = chat(client, f"sim {header}", headers={"X-Simulate-Failure": header})
        assert r.status_code == 200, header
        m = r.json()["metadata"]
        assert m["key_id"] == "key_2" and m["fallback_attempts"] == 1
        assert m["resolution_source"] == "LIVE_INFERENCE"
        assert "boom" not in r.json()["reply"] and r.json()["reply"]


def test_simulated_all_fail_503_and_not_stored(client):
    s = sid()
    for header in ("*=429", "key_1=429,key_2=500"):
        r = chat(client, "doomed", s, headers={"X-Simulate-Failure": header})
        assert r.status_code == 503
        assert r.json() == {"detail": "all providers unavailable"}
    assert client.get(f"/sessions/{s}/messages").status_code == 404


def test_simulation_only_affects_that_request(client):
    chat(client, "x1", headers={"X-Simulate-Failure": "key_1=429"})
    r = chat(client, "x2")
    assert r.json()["metadata"]["key_id"] == "key_1"


def test_simulation_bypasses_cache_so_failover_is_visible(client):
    chat(client, "cached msg")
    r = chat(client, "cached msg", headers={"X-Simulate-Failure": "key_1=429"})
    m = r.json()["metadata"]
    assert m["key_id"] == "key_2" and m["resolution_source"] == "LIVE_INFERENCE"


def test_simulation_disabled_is_ignored(make_client):
    c = make_client(simulation=False)
    r = chat(c, "hi", headers={"X-Simulate-Failure": "*=429"})
    assert r.status_code == 200 and r.json()["metadata"]["key_id"] == "key_1"


def test_invalid_simulation_header_is_ignored(client):
    r = chat(client, "hi", headers={"X-Simulate-Failure": "garbage,key_1=418,=500"})
    assert r.status_code == 200 and r.json()["metadata"]["key_id"] == "key_1"


# ---- misc ------------------------------------------------------------------
def test_cors_allows_configured_origin(make_client):
    c = make_client(ALLOWED_ORIGINS="https://app.example.com")
    r = c.options(
        "/chat",
        headers={"Origin": "https://app.example.com", "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type,x-simulate-failure"},
    )
    assert r.headers["access-control-allow-origin"] == "https://app.example.com"
    assert "x-simulate-failure" in r.headers["access-control-allow-headers"].lower()


def test_key_cooldown_skips_recently_failed_key(make_client, provider):
    c = make_client(keys="bad429,good_b", KEY_COOLDOWN_SECONDS=60)
    chat(c, "one")
    provider.calls.clear()
    r = chat(c, "two")
    assert r.json()["metadata"]["key_id"] == "key_2"
    assert [call["key"] for call in provider.calls] == ["good_b"]  # bad key not retried
