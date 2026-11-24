"""Runtime configuration, read from environment variables.

Settings are read on demand (not at import time) so tests can change the
environment between app instances.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer clearly and accurately."
)


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def _csv(value: str | None) -> list[str]:
    return [p.strip() for p in (value or "").split(",") if p.strip()]


@dataclass(frozen=True)
class Settings:
    api_keys: list[str] = field(default_factory=list)
    model: str = "gemini/gemini-2.0-flash"
    api_base: str | None = None
    database_url: str = "sqlite:///./data/app.db"
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    enable_failure_simulation: bool = False
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    cache_dir: str | None = None
    request_timeout: int = 30
    max_history_messages: int = 50
    key_cooldown_seconds: int = 30

    @property
    def db_path(self) -> str:
        """Filesystem path of the SQLite file from DATABASE_URL."""
        url = self.database_url
        prefix = "sqlite:///"
        if url.startswith(prefix):
            # sqlite:////data/app.db -> /data/app.db ; sqlite:///./x.db -> ./x.db
            return url[len(prefix):]
        if url.startswith("sqlite://"):
            return url[len("sqlite://"):] or ":memory:"
        return url  # bare path


def load_settings() -> Settings:
    env = os.environ
    database_url = env.get("DATABASE_URL") or "sqlite:///./data/app.db"
    s = Settings(
        api_keys=_csv(env.get("LLM_API_KEYS")),
        model=env.get("LLM_MODEL") or "gemini/gemini-2.0-flash",
        api_base=env.get("LLM_API_BASE") or None,
        database_url=database_url,
        allowed_origins=_csv(env.get("ALLOWED_ORIGINS")) or ["*"],
        enable_failure_simulation=_bool(env.get("ENABLE_FAILURE_SIMULATION")),
        default_system_prompt=env.get("DEFAULT_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT,
        cache_dir=env.get("LITELLM_CACHE_DIR") or None,
        request_timeout=_int(env.get("LLM_TIMEOUT_SECONDS"), 30),
        max_history_messages=_int(env.get("MAX_HISTORY_MESSAGES"), 50),
        key_cooldown_seconds=_int(env.get("KEY_COOLDOWN_SECONDS"), 30),
    )
    return s
