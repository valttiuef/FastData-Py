
from __future__ import annotations
import json
import logging
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, QTranslator

from core.paths import get_resource_path

LOGGER = logging.getLogger(__name__)


class JsonTranslator(QTranslator):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._messages: dict[str, str] = {}

    def load_json(self, path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            LOGGER.warning("Failed to load translation file %s: %s", path, exc)
            self._messages = {}
            return False

        if not isinstance(payload, dict):
            LOGGER.warning("Translation file %s must be a JSON object", path)
            self._messages = {}
            return False

        self._messages = {str(k): str(v) for k, v in payload.items()}
        return True

    def translate(self, context: str, source_text: str, disambiguation: str | None = None, n: int = -1) -> str:
        return self._messages.get(source_text, source_text)


class LocalizationManager(QObject):
    def __init__(self, app: QCoreApplication) -> None:
        super().__init__(app)
        self._app = app
        self._translator: JsonTranslator | None = None

    def available_languages(self) -> dict[str, str]:
        return {"en": "English", "fi": "Suomi"}

    def apply_language(self, language_code: str) -> bool:
        code = (language_code or "en").lower()
        self.clear_language()
        if code == "en":
            return True

        lang_path = get_resource_path(f"languages/{code}.json")
        translator = JsonTranslator(self._app)
        if not translator.load_json(lang_path):
            return False
        self._app.installTranslator(translator)
        self._translator = translator
        return True

    def clear_language(self) -> None:
        if self._translator is not None:
            self._app.removeTranslator(self._translator)
            self._translator = None


_localization_manager: LocalizationManager | None = None


def init_localization_manager(app: QCoreApplication) -> LocalizationManager:
    global _localization_manager
    if _localization_manager is None:
        _localization_manager = LocalizationManager(app)
    return _localization_manager


def get_localization_manager() -> LocalizationManager | None:
    return _localization_manager


def tr(text: str) -> str:
    return QCoreApplication.translate("app", text)
