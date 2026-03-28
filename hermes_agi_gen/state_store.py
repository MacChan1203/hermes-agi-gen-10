from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "state2.db"
SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    title TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    metadata_json TEXT,
    message_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    finish_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content=messages,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


class SessionDB:
    """SQLite + FTS5 の軽量セッション保存。"""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(SCHEMA_SQL)
        cur.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        if row is None:
            cur.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        cur.executescript(FTS_SQL)
        self._conn.commit()

    def create_session(self, session_id: str, *, source: str = "cli", model: str | None = None, title: str | None = None, user_id: str | None = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions(id, source, user_id, model, title, started_at, metadata_json, message_count) VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT message_count FROM sessions WHERE id = ?), 0))",
                (session_id, source, user_id, model, title, time.time(), json.dumps(metadata or {}, ensure_ascii=False), session_id),
            )
            self._conn.commit()

    def append_message(self, session_id: str, role: str, content: str, *, tool_name: str | None = None, finish_reason: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages(session_id, role, content, tool_name, timestamp, finish_reason) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, tool_name, time.time(), finish_reason),
            )
            self._conn.execute("UPDATE sessions SET message_count = message_count + 1 WHERE id = ?", (session_id,))
            self._conn.commit()

    def end_session(self, session_id: str, reason: str = "completed") -> None:
        with self._lock:
            self._conn.execute("UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?", (time.time(), reason, session_id))
            self._conn.commit()

    def search_messages(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT m.session_id, m.role, m.content, m.timestamp FROM messages_fts f JOIN messages m ON m.id = f.rowid WHERE messages_fts MATCH ? ORDER BY m.timestamp DESC LIMIT ?",
            (query, limit),
        )
        return [dict(row) for row in cur.fetchall()]



def _agent2_db_path(repo_root: str | Path) -> Path:
    repo_root = Path(repo_root)
    return repo_root / ".hermes_agi_gen_state.db"


def _ensure_agent2_summary_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            goal TEXT,
            summary TEXT,
            priority_upgrades_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def save_run_summary(
    repo_root: str | Path,
    *,
    session_id: str,
    goal: str,
    summary: str,
    priority_upgrades: list[str] | None = None,
) -> None:
    db_path = _agent2_db_path(repo_root)
    conn = sqlite3.connect(db_path)
    try:
        _ensure_agent2_summary_table(conn)
        conn.execute(
            """
            INSERT INTO run_summaries (
                session_id,
                goal,
                summary,
                priority_upgrades_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                goal,
                summary,
                json.dumps(priority_upgrades or [], ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_latest_run_summary(repo_root: str | Path) -> dict[str, Any] | None:
    db_path = _agent2_db_path(repo_root)
    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_agent2_summary_table(conn)
        row = conn.execute(
            """
            SELECT session_id, goal, summary, priority_upgrades_json, created_at
            FROM run_summaries
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

        if row is None:
            return None

        return {
            "session_id": row["session_id"],
            "goal": row["goal"],
            "summary": row["summary"],
            "priority_upgrades": json.loads(row["priority_upgrades_json"] or "[]"),
            "created_at": row["created_at"],
        }
    finally:
        conn.close()
