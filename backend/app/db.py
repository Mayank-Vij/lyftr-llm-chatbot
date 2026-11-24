"""SQLite storage for chat sessions (no external services)."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);

CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    system_prompt TEXT,
    created_at    TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            with conn:  # commit / rollback
                yield conn
        finally:
            conn.close()

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def ping(self) -> bool:
        try:
            with self.connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    # ---- sessions -------------------------------------------------------
    def get_system_prompt(self, session_id: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row["system_prompt"] if row else None

    def session_exists(self, session_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ? "
                "UNION SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                (session_id, session_id),
            ).fetchone()
        return row is not None

    def history(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Stored messages oldest first. With `limit`, only the newest N."""
        with self.connect() as conn:
            if limit:
                rows = conn.execute(
                    "SELECT role, content, metadata FROM ("
                    " SELECT id, role, content, metadata FROM messages"
                    " WHERE session_id = ? ORDER BY id DESC LIMIT ?"
                    ") ORDER BY id ASC",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT role, content, metadata FROM messages "
                    "WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                ).fetchall()
        out = []
        for r in rows:
            item: dict[str, Any] = {"role": r["role"], "content": r["content"]}
            if r["metadata"]:
                try:
                    item["metadata"] = json.loads(r["metadata"])
                except json.JSONDecodeError:
                    pass
            out.append(item)
        return out

    def save_exchange(
        self,
        session_id: str,
        system_prompt: str | None,
        user_content: str,
        assistant_content: str,
        metadata: dict[str, Any],
    ) -> None:
        """Atomically persist a successful user/assistant exchange.

        The session row is only created here, so a failed request (422/503)
        leaves no trace in the history.
        """
        ts = now_iso()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, system_prompt, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "system_prompt = COALESCE(excluded.system_prompt, sessions.system_prompt)",
                (session_id, system_prompt, ts),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) "
                "VALUES (?, 'user', ?, NULL, ?)",
                (session_id, user_content, ts),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?)",
                (session_id, assistant_content, json.dumps(metadata), now_iso()),
            )
