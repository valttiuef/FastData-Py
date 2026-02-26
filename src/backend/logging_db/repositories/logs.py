
from __future__ import annotations
import sqlite3
from typing import Any, Dict, Iterable, List, Optional


class LogsRepository:
    """CRUD helpers for the ``logs`` and ``logs_fts`` tables."""

    def insert(
        self,
        con: sqlite3.Connection,
        *,
        created_at: float,
        level: int,
        logger_name: str,
        origin: str,
        message: str,
        formatted: str,
    ) -> int:
        cur = con.execute(
            """
            INSERT INTO logs (created_at, level, logger_name, origin, message, formatted)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (created_at, level, logger_name, origin, message, formatted),
        )
        log_id = int(cur.lastrowid)
        con.execute(
            """
            INSERT INTO logs_fts(rowid, message, formatted, logger_name, origin)
            VALUES (?, ?, ?, ?, ?);
            """,
            (log_id, message, formatted, logger_name, origin),
        )
        con.commit()
        return log_id

    def fetch_all(
        self,
        con: sqlite3.Connection,
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        sql = (
            "SELECT id, created_at, level, logger_name, origin, message, formatted "
            "FROM logs ORDER BY created_at ASC"
        )
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(limit),)
        rows = con.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def search(
        self,
        con: sqlite3.Connection,
        *,
        keyword: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        base = (
            "SELECT l.id, l.created_at, l.level, l.logger_name, l.origin, l.message, l.formatted "
            "FROM logs l"
        )
        clauses = []
        params: list[Any] = []
        if keyword:
            base += " JOIN logs_fts f ON l.id = f.rowid"
            clauses.append("logs_fts MATCH ?")
            params.append(keyword)
        if start is not None:
            clauses.append("l.created_at >= ?")
            params.append(float(start))
        if end is not None:
            clauses.append("l.created_at <= ?")
            params.append(float(end))
        if clauses:
            base += " WHERE " + " AND ".join(clauses)
        base += " ORDER BY l.created_at ASC"
        rows = con.execute(base, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def delete_all(self, con: sqlite3.Connection) -> None:
        con.execute("DELETE FROM logs;")
        self.rebuild_fts(con)

    def delete_by_origin(self, con: sqlite3.Connection, origins: Iterable[str]) -> None:
        origins_list = [origin for origin in origins if origin]
        if not origins_list:
            return
        placeholders = ", ".join(["?"] * len(origins_list))
        con.execute(
            f"DELETE FROM logs WHERE origin IN ({placeholders})",
            tuple(origins_list),
        )
        self.rebuild_fts(con)

    def rebuild_fts(self, con: sqlite3.Connection) -> None:
        con.execute("INSERT INTO logs_fts(logs_fts) VALUES('delete-all');")
        con.execute(
            """
            INSERT INTO logs_fts(rowid, message, formatted, logger_name, origin)
            SELECT id, message, formatted, logger_name, origin FROM logs;
            """
        )
        con.commit()


__all__ = ["LogsRepository"]
