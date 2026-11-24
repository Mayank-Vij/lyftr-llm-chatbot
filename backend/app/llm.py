"""LiteLLM gateway: key pool, automatic failover and response caching."""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import litellm
from litellm.caching.caching import Cache

from .config import Settings

log = logging.getLogger("chatbot.llm")

litellm.drop_params = True
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

SIMULATED_STATUSES = {429, 500, 502, 503}


class AllProvidersUnavailable(Exception):
    """Every key in the pool failed for this request."""


@dataclass
class LLMResult:
    reply: str
    resolution_source: str  # LIVE_INFERENCE | LLM_CACHE
    model: str
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


def parse_failure_header(value: str | None) -> dict[str, int]:
    """Parse `key_1=429,key_2=500` / `*=429` into {key_id: status}.

    Malformed pairs and unsupported statuses are ignored.
    """
    out: dict[str, int] = {}
    if not value:
        return out
    for part in value.split(","):
        if "=" not in part:
            continue
        key, _, status = part.partition("=")
        key, status = key.strip(), status.strip()
        if not key or not status.isdigit() or int(status) not in SIMULATED_STATUSES:
            continue
        out[key] = int(status)
    return out


def init_cache(settings: Settings) -> str:
    """Enable LiteLLM's built-in response cache. Returns the cache type used."""
    cache_dir = settings.cache_dir
    if not cache_dir:
        base = settings.db_path
        parent = os.path.dirname(os.path.abspath(base)) if base != ":memory:" else "/tmp"
        cache_dir = os.path.join(parent, "litellm_cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        litellm.cache = Cache(type="disk", disk_cache_dir=cache_dir)
        return "disk"
    except Exception as exc:  # fall back to memory if the disk is unusable
        log.warning("disk cache unavailable (%s); using in-memory cache", exc)
        litellm.cache = Cache(type="local")
        return "local"


class KeyPool:
    """Ordered pool of API keys with a short cooldown for keys that just failed.

    Key ids are positional (key_1, key_2, ...). The secret itself never leaves
    this object.
    """

    def __init__(self, keys: list[str], cooldown_seconds: int = 30):
        self._keys = list(keys)
        self._cooldown = cooldown_seconds
        self._cool_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._keys)

    def ordered(self) -> list[tuple[str, str]]:
        """(key_id, secret) pairs: healthy keys first, cooling keys last."""
        now = time.monotonic()
        pairs = [(f"key_{i + 1}", k) for i, k in enumerate(self._keys)]
        with self._lock:
            healthy = [p for p in pairs if self._cool_until.get(p[0], 0) <= now]
            cooling = [p for p in pairs if self._cool_until.get(p[0], 0) > now]
        return healthy + cooling

    def mark_failed(self, key_id: str) -> None:
        if self._cooldown <= 0:
            return
        with self._lock:
            self._cool_until[key_id] = time.monotonic() + self._cooldown

    def mark_ok(self, key_id: str) -> None:
        with self._lock:
            self._cool_until.pop(key_id, None)


class LLMGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pool = KeyPool(settings.api_keys, settings.key_cooldown_seconds)
        self.cache_type = init_cache(settings)

    @property
    def keys_configured(self) -> int:
        return len(self.pool)

    def complete(
        self,
        messages: list[dict[str, str]],
        simulate: dict[str, int] | None = None,
    ) -> LLMResult:
        """Return a reply, trying keys in order until one succeeds.

        `simulate` maps key_id (or "*") to an HTTP status; matching keys are
        treated as failed for this call only, without contacting the provider.
        When simulating, the cache read is bypassed so the failover path is
        always exercised and answered by a real provider call.
        """
        simulate = simulate or {}
        attempts = 0

        for key_id, secret in self.pool.ordered():
            status = simulate.get(key_id) or simulate.get("*")
            if status:
                log.info("simulated %s for %s", status, key_id)
                attempts += 1
                continue

            kwargs: dict[str, Any] = dict(
                model=self.settings.model,
                messages=messages,
                api_key=secret,
                timeout=self.settings.request_timeout,
                num_retries=0,
                caching=True,
            )
            if self.settings.api_base:
                kwargs["api_base"] = self.settings.api_base
            if simulate:
                kwargs["cache"] = {"no-cache": True}  # skip read, still store

            try:
                resp = litellm.completion(**kwargs)
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    raise ValueError("empty completion")
            except Exception as exc:  # 429 / quota / 5xx / network / bad key
                code = getattr(exc, "status_code", None)
                log.warning("%s failed (%s %s): %s", key_id, code, type(exc).__name__, str(exc)[:300])
                self.pool.mark_failed(key_id)
                attempts += 1
                continue

            hidden = getattr(resp, "_hidden_params", None) or {}
            cache_hit = bool(hidden.get("cache_hit"))
            usage = getattr(resp, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            if not cache_hit:
                self.pool.mark_ok(key_id)
            return LLMResult(
                reply=resp.choices[0].message.content,
                resolution_source="LLM_CACHE" if cache_hit else "LIVE_INFERENCE",
                model=self.settings.model,
                input_tokens=in_tok,
                output_tokens=out_tok,
                key_id=None if cache_hit else key_id,
                fallback_attempts=0 if cache_hit else attempts,
            )

        raise AllProvidersUnavailable()
