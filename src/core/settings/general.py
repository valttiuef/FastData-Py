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

from core.paths import get_default_exports_directory

from .base import SettingsGroupBase


class GeneralSettings(SettingsGroupBase):
    DEFAULT_LANGUAGE = "en"
    DEFAULT_THEME = ""
    _ALLOWED_LANGUAGES = {"en", "fi"}
    _ALLOWED_THEMES = {"", "dark", "light"}

    def get_language(self) -> str:
        value = self._value("language", self.DEFAULT_LANGUAGE)
        normalized = str(value or self.DEFAULT_LANGUAGE).strip().lower()
        return normalized if normalized in self._ALLOWED_LANGUAGES else self.DEFAULT_LANGUAGE

    def set_language(self, language: str) -> None:
        normalized = str(language or self.DEFAULT_LANGUAGE).strip().lower()
        self._set_value(
            "language",
            normalized if normalized in self._ALLOWED_LANGUAGES else self.DEFAULT_LANGUAGE,
        )

    def get_theme(self) -> str:
        value = self._value("theme", self.DEFAULT_THEME)
        normalized = str(value or self.DEFAULT_THEME).strip().lower()
        return normalized if normalized in self._ALLOWED_THEMES else self.DEFAULT_THEME

    def set_theme(self, theme: str) -> None:
        normalized = str(theme or self.DEFAULT_THEME).strip().lower()
        if normalized not in self._ALLOWED_THEMES:
            normalized = self.DEFAULT_THEME
        self._set_value("theme", normalized)

    # @ai(gpt-5, codex-cli, feature, 2026-04-02)
    def get_file_dialog_directory(self, scope: str, fallback: Path | None = None) -> Path:
        normalized_scope = str(scope or "default").strip().lower() or "default"
        key = f"file_dialog_dir/{normalized_scope}"
        value = self._value(key, "")
        if value:
            return Path(str(value))
        if normalized_scope == "database":
            return self._manager.database.default_db_path().parent
        if normalized_scope == "export":
            return get_default_exports_directory()
        return Path(fallback) if fallback is not None else Path.cwd()

    # @ai(gpt-5, codex-cli, feature, 2026-04-02)
    def set_file_dialog_directory(self, scope: str, path: Path | str) -> None:
        candidate = Path(path)
        directory = candidate if candidate.is_dir() else candidate.parent
        key = f"file_dialog_dir/{str(scope or 'default').strip().lower() or 'default'}"
        self._set_value(key, str(directory))

    def file_dialog_directories(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key in self._settings.allKeys():
            if not str(key).startswith("file_dialog_dir/"):
                continue
            scope = str(key).split("/", 1)[1]
            value = self._value(key, "")
            if value:
                out[scope] = str(value)
        return out

    def reset_file_dialog_directories(self) -> None:
        for key in list(self._settings.allKeys()):
            if str(key).startswith("file_dialog_dir/"):
                self._remove(str(key))

    def defaults(self) -> dict[str, Any]:
        return {
            "language": self.DEFAULT_LANGUAGE,
            "theme": self.DEFAULT_THEME,
            "file_dialog_directories": {},
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.get_language(),
            "theme": self.get_theme(),
            "file_dialog_directories": self.file_dialog_directories(),
        }

    def update_from_dict(self, payload: Mapping[str, Any]) -> None:
        if "language" in payload:
            self.set_language(str(payload.get("language", self.DEFAULT_LANGUAGE)))
        if "theme" in payload:
            self.set_theme(str(payload.get("theme", self.DEFAULT_THEME)))
        if "file_dialog_directories" in payload:
            self.reset_file_dialog_directories()
            directories = payload.get("file_dialog_directories")
            if isinstance(directories, Mapping):
                for scope, path in directories.items():
                    if path:
                        self.set_file_dialog_directory(str(scope), str(path))

    def reset(self) -> None:
        self.set_language(self.DEFAULT_LANGUAGE)
        self.set_theme(self.DEFAULT_THEME)
        self.reset_file_dialog_directories()
