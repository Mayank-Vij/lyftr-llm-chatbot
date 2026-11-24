# LLM Chatbot with Key Failover and Response Caching

React + FastAPI + LiteLLM + SQLite. Multi-turn chat that survives provider key failures
(429 / quota / 5xx) by failing over across a key pool, and caches responses with LiteLLM's
built-in cache. Every reply shows its own metadata: resolution source, model, latency and
per-request token usage.

## 1. Live URLs

| | URL |
|---|---|
| Frontend | _add after deploy_ |
| Backend  | _add after deploy_ (`/health`, `/chat`, `/sessions/{id}/messages`) |
| Demo video | _add link_ |

Deploy notes: backend via `render.yaml` / `backend/Dockerfile` (set `LLM_API_KEYS`,
`ALLOWED_ORIGINS=<frontend url>`, `ENABLE_FAILURE_SIMULATION=true`); frontend on Vercel/Netlify
with root `frontend/` and env `VITE_API_URL=<backend url>`. On Render's free tier there is no
persistent disk, so history resets on restart; attach a disk at `/data` (see `render.yaml`)
to keep it.

## 2. Run locally

### With Docker
```bash
cp .env.example .env      # put at least two real keys in LLM_API_KEYS
make up                   # backend :8000, frontend :3000
make down
```
`make up` creates `.env` from `.env.example` if missing. The SQLite file and the LiteLLM
cache live in the named volume `app-data` mounted at `/data`, so history survives restarts.

### Without Docker
```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export LLM_API_KEYS=key-one,key-two LLM_MODEL=gemini/gemini-2.0-flash \
       DATABASE_URL=sqlite:///./data/app.db ALLOWED_ORIGINS=http://localhost:3000 \
       ENABLE_FAILURE_SIMULATION=true
uvicorn app.main:app --port 8000
python -m pytest -q       # tests

# frontend (new terminal)
cd frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev    # http://localhost:3000
```

### Environment variables
See `.env.example`. Required: `LLM_API_KEYS`, `DATABASE_URL`, `ALLOWED_ORIGINS`,
`ENABLE_FAILURE_SIMULATION`. Optional: `LLM_MODEL`, `LLM_API_BASE`, `LLM_TIMEOUT_SECONDS`,
`MAX_HISTORY_MESSAGES`, `KEY_COOLDOWN_SECONDS`, `DEFAULT_SYSTEM_PROMPT`, `LITELLM_CACHE_DIR`.
Any LiteLLM model name works (`gemini/...`, `groq/...`, `gpt-4o-mini`, ...); keys in the pool
must belong to that provider.

## 3. Design decisions

**Key fallback.** `LLM_API_KEYS` becomes an ordered pool with ids `key_1, key_2, ...`
(secrets never leave the process or appear in responses). For each request the gateway calls
`litellm.completion(api_key=...)` with the first healthy key; on any failure (429, quota, 5xx,
timeout, connection error, bad key, empty completion) it moves to the next key and counts
`fallback_attempts`. If every key fails the API returns `503 {"detail":"all providers unavailable"}`
and nothing is written to history. A key that just failed is put on a short cooldown
(`KEY_COOLDOWN_SECONDS`, default 30) so a rate-limited key is not retried on every request;
if all keys are cooling they are all tried anyway. I drive failover with a small explicit loop
rather than LiteLLM's Router so that the per-request `key_id`/`fallback_attempts` metadata and
the `X-Simulate-Failure` hook are exact and testable.

**Caching.** LiteLLM's built-in cache (`litellm.cache = Cache(type="disk")`) stored under the
`/data` volume. LiteLLM's key is derived from the model and the full message list
(system prompt + history + new message), not from the API key or session id, so the same first
message in a new session is a hit and any different message or history is a miss. A hit is
detected from LiteLLM's `cache_hit` flag and reported as `LLM_CACHE` with `key_id: null`,
`fallback_attempts: 0` and the token usage that was stored with the response. Hits also survive
a backend restart because the cache is on disk.

**SQLite session storage.** The backend is the source of truth for conversation state: the
client sends only the new message; history for `session_id` is loaded from SQLite, sent to the
model (newest `MAX_HISTORY_MESSAGES`, default 50), and the user+assistant pair is written in one
transaction **only after success**, so 422/503 never pollute a session. Schema is created on
startup (`messages` exactly as specified plus a `sessions` table holding the optional system
prompt, and an index). `DATABASE_URL` points at the file in the Docker volume; WAL mode is on.
A per-session lock keeps concurrent requests from interleaving history.

**Other.** Pydantic validation gives the 422s (message is trimmed before the 1-4000 check); a
catch-all middleware turns unexpected errors into JSON, never a stack trace; CORS is limited to
`ALLOWED_ORIGINS`. Docker images are multi-stage (backend runs as non-root).

## 4. Verification

Replace `$API` with `http://localhost:8000` or the deployed backend.

```bash
curl $API/health
# {"status":"ok","keys_configured":2,"cache_enabled":true}

# Live inference, then the same first message in a NEW session -> cache hit
curl -s $API/chat -H 'content-type: application/json' \
  -d '{"session_id":"s-a","message":"What is the capital of France?"}'
curl -s $API/chat -H 'content-type: application/json' \
  -d '{"session_id":"s-b","message":"What is the capital of France?"}'
# 2nd: resolution_source LLM_CACHE, identical reply, key_id null, lower latency_ms

# Failover (requires ENABLE_FAILURE_SIMULATION=true)
curl -s $API/chat -H 'content-type: application/json' -H 'X-Simulate-Failure: key_1=429' \
  -d '{"session_id":"s-c","message":"Say hi"}'
# 200, key_id "key_2", fallback_attempts 1
curl -s -i $API/chat -H 'content-type: application/json' -H 'X-Simulate-Failure: *=429' \
  -d '{"session_id":"s-d","message":"Say hi"}'
# 503 {"detail":"all providers unavailable"}; GET /sessions/s-d/messages -> 404

# History from SQLite
curl -s $API/sessions/s-a/messages
```
While a failure is simulated the cache read is bypassed, so the failover path is always served
by a real provider call (a cached answer would otherwise hide which key answered).

In the UI each assistant reply shows source, model, latency and that request's input/output
tokens. Open **Settings** to set a system prompt or type an `X-Simulate-Failure` value
(e.g. `key_1=429`) to demo failover from the browser. Reloading the page starts a new session.

Tests: `make test` (23 tests). They run the real LiteLLM code path against a local fake
OpenAI-compatible provider, covering failover on 429/500/503, 503 when all keys fail, the
simulation header, cache hits/misses, per-request tokens, multi-turn memory, session isolation,
SQLite persistence, 422/404 and CORS.

## 5. Setup Used

- Editor / agent: Claude (Anthropic) via Claude Code-style agentic workflow; Vite for the frontend build.
- Libraries: FastAPI, LiteLLM 1.74.9 (pinned), diskcache, pytest, React 18.

## 6. Bonus features

- Optional per-session system prompt (sticky for the session) in the UI and API.
- Key cooldown so a rate-limited key is skipped for a short time.
- Failover demo field in the UI for `X-Simulate-Failure`.
- Disk-backed cache that survives restarts; history window cap (`MAX_HISTORY_MESSAGES`).
- `render.yaml` blueprint and `vercel.json` for deployment.
