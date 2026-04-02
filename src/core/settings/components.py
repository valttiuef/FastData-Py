# --- @ai START ---
# model: gpt-5
# tool: codex-cli
# role: architectural-refactor
# reviewed: no
# date: 2026-04-02
# --- @ai END ---
from __future__ import annotations

import json
from typing import Any, Mapping

from .base import SettingsGroupBase


class ComponentSettings(SettingsGroupBase):
    """Generic per-component persisted payloads (for sidebars/tabs/widgets)."""

    _PREFIX = "components/"

    def _key(self, component_key: str) -> str:
        safe = str(component_key or "").strip().lower()
        return f"{self._PREFIX}{safe}"

    def _normalize_payload(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return {}
        out: dict[str, Any] = {}
        for key, value in payload.items():
            if not key:
                continue
            out[str(key)] = value
        return out

    def get_component_settings(
        self, component_key: str, defaults: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        fallback = self._normalize_payload(defaults)
        key = self._key(component_key)
        raw = self._value(key, "")
        if not raw:
            return dict(fallback)
        try:
            parsed = json.loads(str(raw))
        except Exception:
            return dict(fallback)
        if not isinstance(parsed, dict):
            return dict(fallback)
        merged = dict(fallback)
        merged.update(parsed)
        return merged

    def set_component_settings(
        self,
        component_key: str,
        payload: Mapping[str, Any],
        *,
        merge: bool = True,
    ) -> dict[str, Any]:
        normalized = self._normalize_payload(payload)
        if merge:
            current = self.get_component_settings(component_key)
            current.update(normalized)
            normalized = current
        key = self._key(component_key)
        self._set_value(key, json.dumps(normalized, ensure_ascii=True, sort_keys=True))
        return normalized

    def reset_component_settings(self, component_key: str) -> None:
        self._remove(self._key(component_key))

    def reset_all_components(self) -> None:
        for key in list(self._settings.allKeys()):
            if str(key).startswith(self._PREFIX):
                self._remove(str(key))

    def list_components(self) -> list[str]:
        out: list[str] = []
        for key in self._settings.allKeys():
            text = str(key)
            if text.startswith(self._PREFIX):
                out.append(text[len(self._PREFIX) :])
        return sorted(set(out))

    def defaults(self) -> dict[str, Any]:
        return {}

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for component_key in self.list_components():
            out[component_key] = self.get_component_settings(component_key)
        return out

    def update_from_dict(self, payload: Mapping[str, Any]) -> None:
        for key, value in payload.items():
            if isinstance(value, Mapping):
                self.set_component_settings(str(key), value, merge=False)

    def reset(self) -> None:
        self.reset_all_components()
