
from __future__ import annotations
import sqlite3
import threading
from pathlib import Path

from .repositories.chat_sessions import ChatSessionsRepository
from .repositories.logs import LogsRepository
import logging

logger = logging.getLogger(__name__)


class LoggingDatabase:
    """Lightweight SQLite database dedicated to log/chat history."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        self._con.row_factory = sqlite3.Row
        with self._lock:
            self._con.execute("PRAGMA journal_mode=WAL;")
            self._con.execute("PRAGMA foreign_keys=ON;")
        self.logs_repo = LogsRepository()
        self.chat_sessions_repo = ChatSessionsRepository()
        self._load_schema()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._con

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def close(self) -> None:
        try:
            with self._lock:
                self._con.close()
        except Exception:
            logger.warning("Exception in close", exc_info=True)

    def _load_schema(self) -> None:
        sql_root = Path(__file__).with_name("sql")
        sql_files = sorted(sql_root.glob("*.sql"))
        for file in sql_files:
            script = file.read_text(encoding="utf-8").strip()
            if not script:
                continue
            with self._lock:
                self._con.executescript(script)
        with self._lock:
            self._con.commit()
            has_logs = (
                self._con.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='logs' LIMIT 1;"
                ).fetchone()
                is not None
            )
            if not has_logs:
                self._con.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at REAL NOT NULL,
                        level INTEGER NOT NULL,
                        logger_name TEXT NOT NULL,
                        origin TEXT NOT NULL,
                        message TEXT NOT NULL,
                        formatted TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_logs_created_at ON logs(created_at);
                    CREATE INDEX IF NOT EXISTS idx_logs_logger ON logs(logger_name);
                    CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(
                        message,
                        formatted,
                        logger_name,
                        origin,
                        content=''
                    );
                    """
                )
                self._con.commit()
            self._ensure_logs_chat_columns()

    def _ensure_logs_chat_columns(self) -> None:
        columns = {row[1] for row in self._con.execute("PRAGMA table_info(logs)").fetchall()}
        if "session_id" not in columns:
            self._con.execute("ALTER TABLE logs ADD COLUMN session_id INTEGER")
        if "turn_id" not in columns:
            self._con.execute("ALTER TABLE logs ADD COLUMN turn_id TEXT")
        self._con.execute("CREATE INDEX IF NOT EXISTS idx_logs_session_id ON logs(session_id)")
        self._con.execute("CREATE INDEX IF NOT EXISTS idx_logs_session_turn ON logs(session_id, turn_id)")
        self._con.commit()


__all__ = ["LoggingDatabase"]
