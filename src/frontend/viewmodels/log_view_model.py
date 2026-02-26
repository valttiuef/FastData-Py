
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

from ..models.llm_model import LlmModel
from ..models.log_model import LogModel


class LogViewModel(QObject):
    """Coordinates log persistence and LLM chat streaming for the UI."""

    llm_response_started = Signal()
    llm_token_received = Signal(str)
    llm_response_finished = Signal(str)
    llm_error = Signal(str)

    def __init__(self, log_model: LogModel, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.log_model = log_model
        self.llm_model = LlmModel(parent=self)
        self._api_key: Optional[str] = None
        self._default_context: Optional[str] = None
        self._model_name: Optional[str] = None
        self._provider: str = "openai"

        self.llm_model.response_started.connect(self.llm_response_started)
        self.llm_model.token_received.connect(self.llm_token_received)
        self.llm_model.response_finished.connect(self._on_llm_finished)
        self.llm_model.error.connect(self._on_llm_error)

    # ------------------------------------------------------------------
    def set_api_key(self, value: Optional[str]) -> None:
        self._api_key = value or None

    def set_context(self, value: Optional[str]) -> None:
        self._default_context = value or None

    def set_model_name(self, value: Optional[str]) -> None:
        self._model_name = value or None

    def set_provider(self, value: str) -> None:
        self._provider = value or "openai"

    # ------------------------------------------------------------------
    def log_message(self, text: str, *, level: int = logging.INFO, origin: str = "chat") -> None:
        self.log_model.log_text(text, level=level, origin=origin)

    def ask_llm(
        self,
        prompt: str,
        *,
        context: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        cleaned = prompt.strip()
        if not cleaned:
            return

        self.log_model.log_text(cleaned, level=logging.INFO, origin="chat")
        self.llm_model.ask(
            cleaned,
            context=context or self._default_context,
            api_key=api_key or self._api_key,
            model=model or self._model_name,
            provider=self._provider,
        )

    def cancel_llm(self) -> None:
        self.llm_model.cancel()

    # ------------------------------------------------------------------
    def set_log_database(self, path: Path) -> None:
        self.log_model.set_database(path)

    def clear_log_database(self) -> Path:
        return self.log_model.clear_storage()

    def clear_chat_history(self) -> None:
        self.log_model.clear_chat_history()

    def reset_log_database(self) -> Path:
        return self.log_model.reset_database()

    def save_log_database_as(self, path: Path) -> Path:
        return self.log_model.save_database_as(path)

    def current_log_database_path(self) -> Path:
        return self.log_model.current_database_path()

    # ------------------------------------------------------------------
    def _on_llm_finished(self, text: str) -> None:
        if text.strip():
            self.log_model.log_text(text, level=logging.INFO, origin="llm")
        self.llm_response_finished.emit(text)

    def _on_llm_error(self, message: str) -> None:
        if message:
            self.log_model.log_text(message, level=logging.ERROR, origin="llm")
        self.llm_error.emit(message)
