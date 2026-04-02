# --- @ai START ---
# model: gpt-5
# tool: codex-cli
# role: architectural-refactor
# reviewed: no
# date: 2026-04-02
# --- @ai END ---
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from backend.services.logging.storage import (
    create_log_database,
    load_log_database,
)
from core.paths import get_default_log_database_path

from .base import SettingsGroupBase


class LogSettings(SettingsGroupBase):
    def get_log_db_path(self) -> Path:
        path_str = self._value("log_db_path", "")
        if path_str:
            return Path(str(path_str))
        default_path = self.default_log_db_path()
        self.set_log_db_path(default_path)
        return default_path

    def set_log_db_path(self, path: Path | str) -> None:
        self._set_value("log_db_path", str(Path(path)))

    def default_log_db_path(self) -> Path:
        return get_default_log_database_path()

    def ensure_log_database(self) -> Path:
        path = self.get_log_db_path()
        try:
            if not path.exists():
                create_log_database(path)
            else:
                load_log_database(path)
        except Exception:
            path = self.default_log_db_path()
            try:
                create_log_database(path)
            except Exception:
                load_log_database(path)
            self.set_log_db_path(path)
        return path

    def get_log_sources(self) -> list[str]:
        value = self._value("log_sources", [])
        if isinstance(value, str):
            return [v for v in value.split("|") if v]
        return [str(v) for v in value] if isinstance(value, (list, tuple)) else []

    def set_log_sources(self, sources: list[str]) -> None:
        self._set_value("log_sources", [str(v) for v in (sources or []) if str(v).strip()])

    def get_log_visible(self) -> bool:
        return bool(self._value("log_visible", False, value_type=bool))

    def set_log_visible(self, visible: bool) -> None:
        self._set_value("log_visible", bool(visible))

    def defaults(self) -> dict[str, Any]:
        return {
            "log_db_path": str(self.default_log_db_path()),
            "log_sources": [],
            "log_visible": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "log_db_path": str(self.get_log_db_path()),
            "log_sources": self.get_log_sources(),
            "log_visible": self.get_log_visible(),
        }

    def update_from_dict(self, payload: Mapping[str, Any]) -> None:
        log_db_path = payload.get("log_db_path")
        if log_db_path:
            self.set_log_db_path(str(log_db_path))
        if "log_sources" in payload:
            value = payload.get("log_sources")
            self.set_log_sources([str(v) for v in value] if isinstance(value, (list, tuple)) else [])
        if "log_visible" in payload:
            self.set_log_visible(bool(payload.get("log_visible")))

    def reset(self) -> None:
        self.set_log_db_path(self.default_log_db_path())
        self.set_log_sources([])
        self.set_log_visible(False)
