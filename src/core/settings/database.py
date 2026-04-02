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

from core.paths import (
    get_default_database_path,
    get_default_selection_db_path,
)

from .base import SettingsGroupBase


class DatabaseSettings(SettingsGroupBase):
    def get_db_path(self) -> Path:
        path_str = self._value("db_path", "")
        if path_str:
            return Path(str(path_str))
        default_path = self.default_db_path()
        self.set_db_path(default_path)
        return default_path

    def set_db_path(self, path: Path | str) -> None:
        self._set_value("db_path", str(Path(path)))

    def default_db_path(self) -> Path:
        return get_default_database_path()

    def get_selection_db_path(self) -> Path:
        path_str = self._value("selection_db_path", "")
        if path_str:
            return Path(str(path_str))
        default_path = self.default_selection_db_path()
        self.set_selection_db_path(default_path)
        return default_path

    def set_selection_db_path(self, path: Path | str) -> None:
        self._set_value("selection_db_path", str(Path(path)))

    def default_selection_db_path(self) -> Path:
        return get_default_selection_db_path()

    def defaults(self) -> dict[str, Any]:
        return {
            "db_path": str(self.default_db_path()),
            "selection_db_path": str(self.default_selection_db_path()),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "db_path": str(self.get_db_path()),
            "selection_db_path": str(self.get_selection_db_path()),
        }

    def update_from_dict(self, payload: Mapping[str, Any]) -> None:
        db_path = payload.get("db_path")
        if db_path:
            self.set_db_path(str(db_path))
        selection_db_path = payload.get("selection_db_path")
        if selection_db_path:
            self.set_selection_db_path(str(selection_db_path))

    def reset(self) -> None:
        self.set_db_path(self.default_db_path())
        self.set_selection_db_path(self.default_selection_db_path())
