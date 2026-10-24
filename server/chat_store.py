"""SQLite persistence for multi-turn legal conversations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "output" / "evaluation" / "chat.sqlite3"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class ChatStore:
    """Durable session/message storage; unlike the old demo, it survives restarts."""

    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        self.path = path

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '新的法律咨询',
                    case_memory_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated
                    ON chat_sessions(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                    ON chat_messages(session_id, id);
                """
            )

    def create_session(self, title: str = "新的法律咨询") -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO chat_sessions(id, title, case_memory_json, created_at, updated_at) VALUES (?, ?, '{}', ?, ?)",
                (session_id, (title or "新的法律咨询")[:120], now, now),
            )
        return {"id": session_id, "title": (title or "新的法律咨询")[:120], "case_memory": {}, "messages": [], "created_at": now, "updated_at": now}

    @staticmethod
    def _message_view(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        raw = item.pop("result_json", None)
        item["result"] = json.loads(raw) if raw else None
        return item

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            session = connection.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
            if session is None:
                return None
            rows = connection.execute(
                "SELECT id, role, content, result_json, created_at FROM chat_messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        item = dict(session)
        item["case_memory"] = json.loads(item.pop("case_memory_json") or "{}")
        item["messages"] = [self._message_view(row) for row in rows]
        return item

    def list_sessions(self, limit: int = 30) -> list[dict[str, Any]]:
        safe_limit = max(1, min(100, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.id, s.title, s.created_at, s.updated_at,
                       COUNT(m.id) AS message_count,
                       (SELECT content FROM chat_messages WHERE session_id = s.id ORDER BY id DESC LIMIT 1) AS last_message
                FROM chat_sessions s LEFT JOIN chat_messages m ON m.session_id = s.id
                GROUP BY s.id ORDER BY s.updated_at DESC LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        result: dict[str, Any] | None = None,
        title: str | None = None,
        case_memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("不支持的消息角色")
        now = utc_now()
        with self._connect() as connection:
            session = connection.execute("SELECT id, title FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
            if session is None:
                raise KeyError(session_id)
            connection.execute(
                "INSERT INTO chat_messages(session_id, role, content, result_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, str(content), json_text(result) if result is not None else None, now),
            )
            if title or case_memory is not None:
                connection.execute(
                    "UPDATE chat_sessions SET title = COALESCE(?, title), case_memory_json = COALESCE(?, case_memory_json), updated_at = ? WHERE id = ?",
                    ((title or "")[:120] or None, json_text(case_memory) if case_memory is not None else None, now, session_id),
                )
            else:
                connection.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        return {"role": role, "content": str(content), "result": result, "created_at": now}

    def update_memory(self, session_id: str, case_memory: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE chat_sessions SET case_memory_json = ?, updated_at = ? WHERE id = ?",
                (json_text(case_memory), utc_now(), session_id),
            )

    def append_turn(
        self, session_id: str, user_text: str, assistant_text: str,
        result: dict[str, Any], case_memory: dict[str, Any], title: str | None = None,
    ) -> dict[str, Any]:
        """Commit both messages and memory together only after generation succeeds."""
        now = utc_now()
        with self._connect() as connection:
            if connection.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,)).fetchone() is None:
                raise KeyError(session_id)
            connection.executemany(
                "INSERT INTO chat_messages(session_id, role, content, result_json, created_at) VALUES (?, ?, ?, ?, ?)",
                [
                    (session_id, "user", user_text, None, now),
                    (session_id, "assistant", assistant_text, json_text(result), now),
                ],
            )
            connection.execute(
                "UPDATE chat_sessions SET title = COALESCE(?, title), case_memory_json = ?, updated_at = ? WHERE id = ?",
                ((title or "")[:120] or None, json_text(case_memory), now, session_id),
            )
        return {"role": "assistant", "content": assistant_text, "result": result, "created_at": now}

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as connection:
            # Also clean orphaned rows created by older versions that did not
            # enable SQLite's per-connection foreign-key enforcement.
            connection.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            cursor = connection.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0
