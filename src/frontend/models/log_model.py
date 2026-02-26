
from __future__ import annotations
import logging
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PySide6.QtCore import QObject, Signal

from backend.services.logging import LogEvent as _LogEvent, get_log_service
from backend.services.logging.storage import (
    create_log_database,
    default_log_db_path,
    delete_log_records_by_origin,
    get_log_database,
    load_log_database,
)

logger = logging.getLogger(__name__)

__all__ = ["LogModel", "LogEvent", "get_log_model"]

LogEvent = _LogEvent


class LogModel(QObject):
    """Qt model that exposes log messages emitted through LogService."""

    entry_added = Signal(object)
    cleared = Signal()
    loggers_changed = Signal(list)
    filter_changed = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._service = get_log_service()
        self._service.ensure_installed()
        self._entries: List[LogEvent] = []
        self._listener = self._on_service_event
        self._logger_listener = self._on_logger_names
        self._known_loggers: set[str] = set()
        self._enabled_loggers: set[str] = set()
        self._restore_from_storage()
        self._service.add_listener(self._listener)
        self._service.add_logger_listener(self._logger_listener)
        self.destroyed.connect(self._detach_listener)

    # ------------------------------------------------------------------
    def entries(self) -> List[LogEvent]:
        return list(self._entries)

    def filtered_entries(self) -> List[LogEvent]:
        if not self._enabled_loggers:
            return []
        enabled = self._enabled_loggers
        return [entry for entry in self._entries if entry.logger_name in enabled]

    def available_loggers(self) -> List[str]:
        return sorted(self._known_loggers)

    def enabled_logger_names(self) -> List[str]:
        return sorted(self._enabled_loggers)

    def set_enabled_loggers(self, names: Iterable[str]) -> None:
        new_enabled = {name for name in names if name in self._known_loggers}
        if new_enabled == self._enabled_loggers:
            return
        self._enabled_loggers = new_enabled
        self.filter_changed.emit()

    # ------------------------------------------------------------------
    def clear(self) -> None:
        if not self._entries:
            return
        self._entries.clear()
        self.cleared.emit()

    def clear_storage(self) -> Path:
        """Reset the persisted log database while keeping the current path."""

        path = get_log_database().path
        create_log_database(path)
        self.reload_from_storage()
        return path

    def clear_chat_history(self) -> None:
        """Remove chat/LLM entries from the current log database."""
        preferred = list(self._enabled_loggers)
        delete_log_records_by_origin(["chat", "llm"])
        self.reload_from_storage(preferred_loggers=preferred)

    # ------------------------------------------------------------------
    def log_text(
        self,
        message: str,
        *,
        level: int = logging.INFO,
        origin: str = "ui",
    ) -> None:
        """Forward a log message to the backend service."""

        try:
            self._service.log_text(message, level=level, origin=origin)
        except Exception:
            # The service should always be available, but logging must never crash
            logger.warning("Exception in log_text", exc_info=True)

    # ------------------------------------------------------------------
    def _on_service_event(self, event: LogEvent) -> None:
        logger_added = self._register_logger(event.logger_name)
        self._entries.append(event)
        if not self._enabled_loggers:
            return
        if event.logger_name in self._enabled_loggers:
            self.entry_added.emit(event)
        if logger_added:
            self.loggers_changed.emit(self.available_loggers())

    def _on_logger_names(self, names: Sequence[str]) -> None:
        added = False
        for name in names:
            added = self._register_logger(name) or added
        if added:
            self.loggers_changed.emit(self.available_loggers())
            self.filter_changed.emit()

    def _detach_listener(self, _obj: Optional[QObject] = None) -> None:
        try:
            self._service.remove_listener(self._listener)
        except Exception:
            logger.warning("Exception in _detach_listener", exc_info=True)
        try:
            self._service.remove_logger_listener(self._logger_listener)
        except Exception:
            logger.warning("Exception in _detach_listener", exc_info=True)

    def _register_logger(self, name: str) -> bool:
        if not name or name in self._known_loggers:
            return False
        self._known_loggers.add(name)
        self._enabled_loggers.add(name)
        return True

    def _restore_from_storage(self) -> None:
        try:
            stored = self._service.load_persisted_events()
        except Exception:
            stored = []
        for event in stored:
            self._register_logger(event.logger_name)
            self._entries.append(event)

    # ------------------------------------------------------------------
    def reload_from_storage(self, preferred_loggers: Optional[Iterable[str]] = None) -> None:
        """Reload entries from the active log database."""

        self._entries.clear()
        self._known_loggers.clear()
        self._enabled_loggers.clear()
        self._restore_from_storage()

        if preferred_loggers is not None:
            preferred = {name for name in preferred_loggers if name in self._known_loggers}
            if preferred:
                self._enabled_loggers = preferred

        self.cleared.emit()
        self.loggers_changed.emit(self.available_loggers())
        self.filter_changed.emit()

    # ------------------------------------------------------------------
    def set_database(self, path: Path) -> None:
        load_log_database(path)
        self.reload_from_storage(preferred_loggers=self._enabled_loggers)

    def reset_database(self) -> Path:
        path = default_log_db_path()
        create_log_database(path)
        load_log_database(path)
        self.reload_from_storage()
        return path

    def save_database_as(self, target: Path) -> Path:
        source = get_log_database().path
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_path)
        load_log_database(target_path)
        self.reload_from_storage(preferred_loggers=self._enabled_loggers)
        return target_path

    def current_database_path(self) -> Path:
        return get_log_database().path


_shared_log_model: Optional[LogModel] = None


def get_log_model(parent: Optional[QObject] = None) -> LogModel:
    """Return a shared :class:`LogModel` instance."""

    global _shared_log_model
    if _shared_log_model is None:
        _shared_log_model = LogModel(parent=parent)
    elif parent is not None and _shared_log_model.parent() is None:
        _shared_log_model.setParent(parent)
    return _shared_log_model
