from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QCoreApplication, QSettings
from PySide6.QtWidgets import QWidget

from core.paths import get_default_database_path, get_default_exports_directory


def _settings_key(scope: str) -> str:
    normalized = str(scope or "default").strip().lower() or "default"
    return f"file_dialog_dir/{normalized}"


def _fallback_settings() -> QSettings:
    org = QCoreApplication.organizationName() or "Visima"
    app = QCoreApplication.applicationName() or "FastData"
    return QSettings(org, app)


def _resolve_settings_model(parent: QWidget | None) -> Any | None:
    if parent is None:
        return None

    window = parent.window()
    model = getattr(window, "settings_model", None) if window is not None else None
    if model is not None:
        return model

    current = parent
    while current is not None:
        model = getattr(current, "settings_model", None)
        if model is not None:
            return model
        current = current.parentWidget()
    return None


# @ai(gpt-5, codex-cli, feature, 2026-03-11)
def get_dialog_directory(parent: QWidget | None, scope: str, fallback: Path | str | None = None) -> Path:
    normalized_scope = str(scope or "default").strip().lower() or "default"
    fallback_path = Path(fallback) if fallback is not None else Path.cwd()
    model = _resolve_settings_model(parent)
    if model is not None and hasattr(model, "get_file_dialog_directory"):
        try:
            return Path(model.get_file_dialog_directory(scope, fallback_path))
        except Exception:
            pass

    settings = _fallback_settings()
    value = settings.value(_settings_key(scope), "")
    if value:
        return Path(str(value))
    if normalized_scope == "database":
        return get_default_database_path().parent
    if normalized_scope == "export":
        return get_default_exports_directory()
    return fallback_path


# @ai(gpt-5, codex-cli, feature, 2026-03-11)
def remember_dialog_path(parent: QWidget | None, scope: str, selected_path: Path | str) -> None:
    candidate = Path(selected_path)
    directory = candidate if candidate.is_dir() else candidate.parent
    model = _resolve_settings_model(parent)
    if model is not None and hasattr(model, "set_file_dialog_directory"):
        try:
            model.set_file_dialog_directory(scope, directory)
            return
        except Exception:
            pass

    settings = _fallback_settings()
    settings.setValue(_settings_key(scope), str(directory))
