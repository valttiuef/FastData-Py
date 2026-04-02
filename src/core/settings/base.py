# --- @ai START ---
# model: gpt-5
# tool: codex-cli
# role: architectural-refactor
# reviewed: no
# date: 2026-04-02
# --- @ai END ---
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class SettingsGroupBase(ABC):
    """Base helper for grouped settings access."""

    def __init__(self, manager) -> None:
        self._manager = manager
        self._settings = manager.settings

    def _value(self, key: str, default: Any = None, *, value_type=None) -> Any:
        if value_type is not None:
            return self._settings.value(key, default, type=value_type)
        return self._settings.value(key, default)

    def _set_value(self, key: str, value: Any) -> None:
        self._settings.setValue(key, value)

    def _remove(self, key: str) -> None:
        self._settings.remove(key)

    @abstractmethod
    def defaults(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def update_from_dict(self, payload: Mapping[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
