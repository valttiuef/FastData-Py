
from __future__ import annotations
import sqlite3
import threading
from pathlib import Path

from .repositories.selection_settings import SelectionSettingsRepository
import logging

logger = logging.getLogger(__name__)


class SelectionSettingsDatabase:
    """SQLite database dedicated to persisted feature selection presets."""

    _EMBEDDED_SCHEMA = (
        """
        CREATE TABLE IF NOT EXISTS selection_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            auto_load INTEGER NOT NULL DEFAULT 0,
            is_active INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_selection_settings_name
        ON selection_settings(LOWER(name));
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_selection_settings_active
        ON selection_settings(is_active);
        """,
    )

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
        self.selection_settings_repo = SelectionSettingsRepository()
        self._load_schema()
        self.ensure_schema()
        try:
            with self._lock:
                self.selection_settings_repo.ensure_active_from_auto_load(self._con)
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)

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

    def ensure_schema(self) -> None:
        """Ensure core selection settings tables/indexes exist."""
        with self._lock:
            if not self._has_table("selection_settings"):
                self._apply_embedded_schema()
            else:
                # Ensure indexes exist even if created DB is missing them.
                for script in self._EMBEDDED_SCHEMA[1:]:
                    self._con.executescript(script)
            self._con.commit()

    def _load_schema(self) -> None:
        sql_root = Path(__file__).with_name("sql")
        sql_files = sorted(sql_root.glob("*.sql"))
        loaded_any = False
        for file in sql_files:
            script = file.read_text(encoding="utf-8").strip()
            if not script:
                continue
            with self._lock:
                self._con.executescript(script)
            loaded_any = True
        with self._lock:
            self._con.commit()
        if not loaded_any:
            self._apply_embedded_schema()

    def _apply_embedded_schema(self) -> None:
        with self._lock:
            for script in self._EMBEDDED_SCHEMA:
                self._con.executescript(script)

    def _has_table(self, name: str) -> bool:
        row = self._con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?;",
            (str(name),),
        ).fetchone()
        return row is not None


__all__ = ["SelectionSettingsDatabase"]
