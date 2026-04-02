from __future__ import annotations

from qt_compat import ensure_qt

ensure_qt()

from typing import Any, Mapping, Optional

from PySide6.QtCore import QSettings

from core.paths import get_default_exports_directory
from core.settings import (
    AISettings,
    ChartSettings,
    ComponentSettings,
    DatabaseSettings,
    GeneralSettings,
    LogSettings,
    TrainingSettings,
)

_shared_settings_manager: Optional["SettingsManager"] = None


# --- @ai START ---
# model: gpt-5
# tool: codex-cli
# role: architectural-refactor
# reviewed: no
# date: 2026-04-02
# --- @ai END ---
class SettingsManager:
    def __init__(self, organization="MyCompany", application="MyApp"):
        self.organization = str(organization or "MyCompany")
        self.application = str(application or "MyApp")
        self.settings = QSettings(organization, application)
        self._secret_service = f"{organization}/{application}"

        self.general = GeneralSettings(self)
        self.database = DatabaseSettings(self)
        self.logs = LogSettings(self)
        self.ai = AISettings(self, secret_service=self._secret_service)
        self.training = TrainingSettings(self)
        self.charts = ChartSettings(self)
        self.components = ComponentSettings(self)

        get_default_exports_directory()

    # ------------------------------------------------------------------
    # Aggregate helpers
    def export_all(self) -> dict[str, Any]:
        return {
            "general": self.general.as_dict(),
            "database": self.database.as_dict(),
            "logs": self.logs.as_dict(),
            "ai": self.ai.as_dict(),
            "training": self.training.as_dict(),
            "charts": self.charts.as_dict(),
            "components": self.components.as_dict(),
        }

    def defaults_all(self) -> dict[str, Any]:
        return {
            "general": self.general.defaults(),
            "database": self.database.defaults(),
            "logs": self.logs.defaults(),
            "ai": self.ai.defaults(),
            "training": self.training.defaults(),
            "charts": self.charts.defaults(),
            "components": self.components.defaults(),
        }

    def import_all(self, payload: Mapping[str, Any], *, reset_missing: bool = False) -> None:
        groups = {
            "general": self.general,
            "database": self.database,
            "logs": self.logs,
            "ai": self.ai,
            "training": self.training,
            "charts": self.charts,
            "components": self.components,
        }
        for name, group in groups.items():
            value = payload.get(name)
            if isinstance(value, Mapping):
                group.update_from_dict(value)
            elif reset_missing:
                group.reset()

    def reset_all(self) -> None:
        self.general.reset()
        self.database.reset()
        self.logs.reset()
        self.ai.reset()
        self.training.reset()
        self.charts.reset()
        self.components.reset()

    # ------------------------------------------------------------------
    # Component settings API
    def get_component_settings(
        self, component_key: str, defaults: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.components.get_component_settings(component_key, defaults=defaults)

    def set_component_settings(
        self, component_key: str, payload: Mapping[str, Any], *, merge: bool = True
    ) -> dict[str, Any]:
        return self.components.set_component_settings(component_key, payload, merge=merge)

    def reset_component_settings(self, component_key: str) -> None:
        self.components.reset_component_settings(component_key)

def configure_settings_manager(organization: str, application: str) -> SettingsManager:
    global _shared_settings_manager
    _shared_settings_manager = SettingsManager(organization, application)
    return _shared_settings_manager


def get_settings_manager(
    organization: str = "MyCompany", application: str = "MyApp", *, create: bool = True
) -> Optional[SettingsManager]:
    global _shared_settings_manager
    if _shared_settings_manager is None and create:
        _shared_settings_manager = SettingsManager(organization, application)
    return _shared_settings_manager


def get_configured_settings_manager() -> Optional[SettingsManager]:
    return _shared_settings_manager
