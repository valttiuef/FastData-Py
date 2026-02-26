
from __future__ import annotations
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QSettings, Signal, QCoreApplication, QThread

from ..threading.utils import run_in_main_thread

from core.settings_manager import SettingsManager


class SettingsModel(QObject):
    """Central wrapper around application settings.

    Provides a Qt-friendly interface to the persistent settings used by the
    application. The model exposes the current theme selection and the active
    database path via signals so UI components can keep themselves in sync.
    """

    theme_changed = Signal(str)
    database_path_changed = Signal(Path)
    selection_db_path_changed = Signal(Path)
    log_db_path_changed = Signal(Path)
    log_sources_changed = Signal(list)
    log_visibility_changed = Signal(bool)
    language_changed = Signal(str)

    def __init__(
        self,
        *,
        organization: str = "Visima",
        application: str = "FastData",
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._organization = organization
        self._application = application
        self._settings = QSettings(organization, application)
        self._settings_manager = SettingsManager(organization, application)

        self._theme = self._load_theme()
        self._db_path = self._settings_manager.get_db_path()
        self._selection_db_path = self._settings_manager.get_selection_db_path()
        self._log_db_path = self._settings_manager.ensure_log_database()
        self._log_sources = self._settings_manager.get_log_sources()
        self._log_visible = self._settings_manager.get_log_visible()
        self._llm_provider = self._settings_manager.get_llm_provider()
        self._llm_models: dict[str, str] = {}
        for provider in ("openai", "ollama"):
            model_value = self._settings_manager.get_llm_model(provider)
            if model_value:
                self._llm_models[provider] = model_value
        self._openai_api_key = self._settings_manager.get_openai_api_key()
        self._language = self._settings_manager.get_language()

    # ------------------------------------------------------------------
    def _emit_in_main_thread(self, signal, *args) -> None:
        app = QCoreApplication.instance()
        if app is None:
            signal.emit(*args)
            return
        try:
            if QThread.currentThread() == app.thread():
                signal.emit(*args)
            else:
                run_in_main_thread(signal.emit, *args)
        except Exception:
            signal.emit(*args)

    # ------------------------------------------------------------------
    def _load_theme(self) -> str:
        value = self._settings.value("theme", "", type=str) or ""
        return value if value in {"dark", "light"} else ""

    # ------------------------------------------------------------------
    @property
    def theme(self) -> str:
        return self._theme or ""

    def set_theme(self, theme: str) -> None:
        normalized = "dark" if theme not in {"dark", "light"} else theme
        if normalized == self._theme:
            return
        self._theme = normalized
        self._settings.setValue("theme", normalized)
        self.theme_changed.emit(normalized)

    # ------------------------------------------------------------------
    @property
    def database_path(self) -> Path:
        return self._db_path

    def set_database_path(self, path: Path | str) -> None:
        new_path = Path(path)
        if new_path == self._db_path:
            return
        self._db_path = new_path
        self._settings_manager.set_db_path(new_path)
        self._emit_in_main_thread(self.database_path_changed, new_path)

    def refresh_database_path(self) -> Path:
        """Ensure the stored database path exists and return it."""
        self._db_path = self._settings_manager.get_db_path()
        return self._db_path

    def default_database_path(self) -> Path:
        return self._settings_manager.default_db_path()

    @property
    def selection_db_path(self) -> Path:
        return self._selection_db_path

    def set_selection_db_path(self, path: Path | str) -> None:
        new_path = Path(path)
        if new_path == self._selection_db_path:
            return
        self._selection_db_path = new_path
        self._settings_manager.set_selection_db_path(new_path)
        self._emit_in_main_thread(self.selection_db_path_changed, new_path)

    def refresh_selection_db_path(self) -> Path:
        self._selection_db_path = self._settings_manager.get_selection_db_path()
        return self._selection_db_path

    def default_selection_db_path(self) -> Path:
        return self._settings_manager.default_selection_db_path()

    # ------------------------------------------------------------------
    @property
    def log_database_path(self) -> Path:
        return self._log_db_path

    def set_log_database_path(self, path: Path | str) -> None:
        new_path = Path(path)
        if new_path == self._log_db_path:
            return
        self._log_db_path = new_path
        self._settings_manager.set_log_db_path(new_path)
        self.log_db_path_changed.emit(new_path)

    def default_log_database_path(self) -> Path:
        return self._settings_manager.default_log_db_path()

    # ------------------------------------------------------------------
    @property
    def log_sources(self) -> list[str]:
        return list(self._log_sources)

    def set_log_sources(self, sources: list[str]) -> None:
        if sources == self._log_sources:
            return
        self._log_sources = list(sources)
        self._settings_manager.set_log_sources(self._log_sources)
        self.log_sources_changed.emit(self._log_sources)

    # ------------------------------------------------------------------
    @property
    def log_visible(self) -> bool:
        return bool(self._log_visible)

    def set_log_visible(self, visible: bool) -> None:
        normalized = bool(visible)
        if normalized == self._log_visible:
            return
        self._log_visible = normalized
        self._settings_manager.set_log_visible(normalized)
        self.log_visibility_changed.emit(normalized)

    # ------------------------------------------------------------------
    @property
    def llm_provider(self) -> str:
        return self._llm_provider or "openai"

    def set_llm_provider(self, provider: str) -> None:
        normalized = provider or "openai"
        if normalized == self._llm_provider:
            return
        self._llm_provider = normalized
        self._settings_manager.set_llm_provider(normalized)

    def llm_model(self, provider: str | None = None) -> str:
        chosen = provider or self.llm_provider
        return self._llm_models.get(chosen, self._settings_manager.default_llm_model(chosen))

    def set_llm_model(self, model: str, provider: str | None = None) -> None:
        chosen = provider or self.llm_provider
        if self._llm_models.get(chosen) == model:
            return
        self._llm_models[chosen] = model
        self._settings_manager.set_llm_model(model, chosen)

    def default_llm_model(self, provider: str) -> str:
        return self._settings_manager.default_llm_model(provider)


    @property
    def language(self) -> str:
        return self._language

    def set_language(self, language: str) -> None:
        normalized = (language or "en").lower()
        if normalized not in {"en", "fi"}:
            normalized = "en"
        if normalized == self._language:
            return
        self._language = normalized
        self._settings_manager.set_language(normalized)
        self.language_changed.emit(normalized)

    @property
    def openai_api_key(self) -> str:
        return self._openai_api_key

    def set_openai_api_key(self, value: str | None) -> None:
        cleaned = value or ""
        if cleaned == self._openai_api_key:
            return
        self._openai_api_key = cleaned
        self._settings_manager.set_openai_api_key(cleaned)
