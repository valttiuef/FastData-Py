from qt_compat import ensure_qt

ensure_qt()

from pathlib import Path

from PySide6.QtCore import QSettings

from backend.services.logging.storage import (
    create_log_database,
    default_log_db_path,
    load_log_database,
)
from core.secure_storage import load_secret, save_secret
from core.paths import (
    get_default_database_path,
    get_default_selection_db_path,
    get_default_log_database_path,
)

class SettingsManager:
    def __init__(self, organization="MyCompany", application="MyApp"):
        self.settings = QSettings(organization, application)
        self._secret_service = f"{organization}/{application}"

    def get_db_path(self) -> Path:
        path_str = self.settings.value("db_path", "")
        if path_str:
            return Path(str(path_str))

        default_path = self.default_db_path()
        self.set_db_path(default_path)
        return default_path

    def set_db_path(self, path: Path):
        self.settings.setValue("db_path", str(path))

    def default_db_path(self) -> Path:
        return get_default_database_path()

    def get_selection_db_path(self) -> Path:
        path_str = self.settings.value("selection_db_path", "")
        if path_str:
            return Path(str(path_str))

        default_path = self.default_selection_db_path()
        self.set_selection_db_path(default_path)
        return default_path

    def set_selection_db_path(self, path: Path):
        self.settings.setValue("selection_db_path", str(path))

    def default_selection_db_path(self) -> Path:
        return get_default_selection_db_path()

    # --- Logging -----------------------------------------------------------
    def get_log_db_path(self) -> Path:
        path_str = self.settings.value("log_db_path", "")
        if path_str:
            return Path(str(path_str))

        default_path = self.default_log_db_path()
        self.set_log_db_path(default_path)
        return default_path

    def set_log_db_path(self, path: Path):
        self.settings.setValue("log_db_path", str(path))

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
            # Fall back to default path if stored path is invalid
            path = self.default_log_db_path()
            try:
                create_log_database(path)
            except Exception:
                load_log_database(path)
            self.set_log_db_path(path)
        return path

    def get_log_sources(self) -> list[str]:
        value = self.settings.value("log_sources", [])
        if isinstance(value, str):
            return [v for v in value.split("|") if v]
        return [str(v) for v in value] if isinstance(value, (list, tuple)) else []

    def set_log_sources(self, sources: list[str]) -> None:
        self.settings.setValue("log_sources", sources)

    def get_log_visible(self) -> bool:
        return bool(self.settings.value("log_visible", False, type=bool))

    def set_log_visible(self, visible: bool) -> None:
        self.settings.setValue("log_visible", bool(visible))

    # --- LLM preferences ----------------------------------------------------
    def get_llm_provider(self) -> str:
        value = self.settings.value("llm_provider", "openai")
        return str(value) if value else "openai"

    def set_llm_provider(self, provider: str) -> None:
        self.settings.setValue("llm_provider", provider or "openai")

    def _model_key(self, provider: str) -> str:
        return f"llm_model_{provider or 'openai'}"

    def get_llm_model(self, provider: str) -> str:
        stored = self.settings.value(self._model_key(provider), "")
        if stored:
            return str(stored)
        return self.default_llm_model(provider)

    def set_llm_model(self, model: str, provider: str) -> None:
        self.settings.setValue(self._model_key(provider), model)

    @staticmethod
    def default_llm_model(provider: str) -> str:
        return "gpt-4o-mini" if provider == "openai" else "llama3.2"


    def get_language(self) -> str:
        value = self.settings.value("language", "en")
        code = str(value or "en").lower()
        return code if code in {"en", "fi"} else "en"

    def set_language(self, language: str) -> None:
        code = str(language or "en").lower()
        self.settings.setValue("language", code if code in {"en", "fi"} else "en")
    def get_openai_api_key(self) -> str:
        return load_secret(self._secret_service, "openai_api_key") or ""

    def set_openai_api_key(self, api_key: str | None) -> None:
        save_secret(self._secret_service, "openai_api_key", api_key)
