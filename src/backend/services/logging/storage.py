
from __future__ import annotations
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ...logging_db import LoggingDatabase
from core.paths import get_default_log_database_path

_db_lock = threading.RLock()
_shared_db: Optional[LoggingDatabase] = None
_text_log_lock = threading.RLock()


def default_log_directory() -> Path:
    """Return the default log folder used for DB and text logs."""
    folder = get_default_log_database_path().parent
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def default_log_db_path() -> Path:
    """Return the default Dataset for the log history database."""
    return default_log_directory() / "history.db"


def warning_log_path() -> Path:
    return default_log_directory() / "warnings.log"


def error_log_path() -> Path:
    return default_log_directory() / "errors.log"


def crash_log_path() -> Path:
    return default_log_directory() / "crash.log"


def append_text_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _text_log_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip("\n") + "\n")


def _set_database(path: Path) -> LoggingDatabase:
    global _shared_db
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _db_lock:
        if _shared_db is not None:
            # If the current database matches the requested path but the
            # underlying file was removed (for example, during a reset), drop
            # the existing connection so we can recreate it cleanly.
            if _shared_db.path == target and target.exists():
                return _shared_db
            _shared_db.close()
        _shared_db = LoggingDatabase(target)
        return _shared_db


def get_log_database() -> LoggingDatabase:
    """Return the shared SQLite database instance, creating it if needed."""
    global _shared_db
    with _db_lock:
        if _shared_db is None:
            _shared_db = LoggingDatabase(default_log_db_path())
        return _shared_db


def load_log_database(path: Path) -> LoggingDatabase:
    """Load an existing database from ``path`` and make it the shared instance."""
    return _set_database(Path(path))


def create_log_database(path: Optional[Path] = None) -> LoggingDatabase:
    """Create a new empty database at ``path`` (or default Dataset)."""
    target = Path(path) if path else default_log_db_path()
    with _db_lock:
        global _shared_db
        if _shared_db is not None and _shared_db.path == target:
            _shared_db.close()
            _shared_db = None

        if target.exists():
            target.unlink()

        _shared_db = LoggingDatabase(target)
        return _shared_db


def delete_log_database(path: Optional[Path] = None) -> None:
    """Delete the database file and drop the shared instance if it matches."""
    target = Path(path) if path else default_log_db_path()
    global _shared_db
    with _db_lock:
        if _shared_db is not None and _shared_db.path == target:
            _shared_db.close()
            _shared_db = None
    if target.exists():
        target.unlink()


def save_log_record(
    *,
    created_at: float,
    level: int,
    logger_name: str,
    origin: str,
    message: str,
    formatted: str,
    session_id: Optional[int] = None,
    turn_id: Optional[str] = None,
) -> int:
    """Persist a log entry in SQLite and return its row id."""
    db = get_log_database()
    with db.lock:
        return db.logs_repo.insert(
            db.connection,
            created_at=created_at,
            level=level,
            logger_name=logger_name,
            origin=origin,
            message=message,
            formatted=formatted,
            session_id=session_id,
            turn_id=turn_id,
        )


def fetch_log_records(
    *,
    keyword: Optional[str] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return stored log rows filtered by keyword or timestamps."""
    db = get_log_database()
    with db.lock:
        return db.logs_repo.search(db.connection, keyword=keyword, start=start, end=end)


def fetch_all_log_records(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return the most recent log rows."""
    db = get_log_database()
    with db.lock:
        return db.logs_repo.fetch_all(db.connection, limit=limit)


def delete_log_records_by_origin(origins: Iterable[str]) -> None:
    """Delete stored log rows that match any of the provided origins."""
    db = get_log_database()
    with db.lock:
        db.logs_repo.delete_by_origin(db.connection, origins)


def ensure_default_chat_session() -> int:
    db = get_log_database()
    with db.lock:
        existing = db.chat_sessions_repo.get_default(db.connection)
        if existing is not None:
            return int(existing["id"])
        return db.chat_sessions_repo.create(db.connection, title="Chat", is_default=True)


def create_chat_session(title: Optional[str] = None) -> int:
    db = get_log_database()
    with db.lock:
        return db.chat_sessions_repo.create(db.connection, title=title, is_default=False)


def list_chat_sessions() -> List[Dict[str, Any]]:
    db = get_log_database()
    with db.lock:
        return db.chat_sessions_repo.list(db.connection)


def get_chat_session(session_id: int) -> Optional[Dict[str, Any]]:
    db = get_log_database()
    with db.lock:
        return db.chat_sessions_repo.get(db.connection, session_id)


def rename_chat_session(session_id: int, title: str) -> None:
    db = get_log_database()
    with db.lock:
        db.chat_sessions_repo.rename(db.connection, session_id, title)


def delete_chat_session(session_id: int) -> None:
    db = get_log_database()
    with db.lock:
        db.chat_sessions_repo.delete(db.connection, session_id)


def fetch_chat_session_messages(session_id: int) -> List[Dict[str, Any]]:
    db = get_log_database()
    with db.lock:
        return db.logs_repo.fetch_session_messages(db.connection, session_id)


def touch_chat_session(session_id: int) -> None:
    db = get_log_database()
    with db.lock:
        db.chat_sessions_repo.touch(db.connection, session_id)


def set_chat_session_summary(session_id: int, summary: str) -> None:
    db = get_log_database()
    with db.lock:
        db.chat_sessions_repo.set_summary(db.connection, session_id, summary)


__all__ = [
    "append_text_log",
    "crash_log_path",
    "create_chat_session",
    "create_log_database",
    "default_log_directory",
    "default_log_db_path",
    "delete_chat_session",
    "delete_log_database",
    "delete_log_records_by_origin",
    "ensure_default_chat_session",
    "error_log_path",
    "fetch_all_log_records",
    "fetch_chat_session_messages",
    "fetch_log_records",
    "get_chat_session",
    "get_log_database",
    "list_chat_sessions",
    "load_log_database",
    "rename_chat_session",
    "save_log_record",
    "set_chat_session_summary",
    "touch_chat_session",
    "warning_log_path",
]

