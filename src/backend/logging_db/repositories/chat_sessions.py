from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List, Optional


class ChatSessionsRepository:
    def create(self, con: sqlite3.Connection, *, title: Optional[str] = None, is_default: bool = False) -> int:
        now = time.time()
        resolved_title = (title or "New Chat").strip() or "New Chat"
        if is_default:
            con.execute("UPDATE chat_sessions SET is_default=0 WHERE is_default=1")
        cur = con.execute(
            """
            INSERT INTO chat_sessions(title, summary, is_default, created_at, updated_at)
            VALUES(?, '', ?, ?, ?)
            """,
            (resolved_title, 1 if is_default else 0, now, now),
        )
        con.commit()
        return int(cur.lastrowid)

    def list(self, con: sqlite3.Connection) -> List[Dict[str, Any]]:
        rows = con.execute(
            """
            SELECT s.id, s.title, s.summary, s.is_default, s.created_at, s.updated_at,
                   COUNT(l.id) AS message_count,
                   MAX(l.created_at) AS last_message_at
            FROM chat_sessions s
            LEFT JOIN logs l ON l.session_id = s.id AND l.origin IN ('chat', 'llm')
            GROUP BY s.id
            ORDER BY s.updated_at DESC, s.id DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def get(self, con: sqlite3.Connection, session_id: int) -> Optional[Dict[str, Any]]:
        row = con.execute(
            "SELECT id, title, summary, is_default, created_at, updated_at FROM chat_sessions WHERE id = ?",
            (int(session_id),),
        ).fetchone()
        return dict(row) if row else None

    def get_default(self, con: sqlite3.Connection) -> Optional[Dict[str, Any]]:
        row = con.execute(
            "SELECT id, title, summary, is_default, created_at, updated_at FROM chat_sessions WHERE is_default=1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def touch(self, con: sqlite3.Connection, session_id: int) -> None:
        con.execute("UPDATE chat_sessions SET updated_at=? WHERE id=?", (time.time(), int(session_id)))
        con.commit()

    def rename(self, con: sqlite3.Connection, session_id: int, title: str) -> None:
        con.execute("UPDATE chat_sessions SET title=?, updated_at=? WHERE id=?", (title.strip() or "New Chat", time.time(), int(session_id)))
        con.commit()

    def set_summary(self, con: sqlite3.Connection, session_id: int, summary: str) -> None:
        con.execute("UPDATE chat_sessions SET summary=?, updated_at=? WHERE id=?", (summary or "", time.time(), int(session_id)))
        con.commit()

    def delete(self, con: sqlite3.Connection, session_id: int) -> None:
        con.execute("DELETE FROM logs WHERE session_id = ?", (int(session_id),))
        con.execute("DELETE FROM chat_sessions WHERE id = ?", (int(session_id),))
        con.commit()
