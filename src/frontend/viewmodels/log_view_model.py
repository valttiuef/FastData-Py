from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

from backend.services.logging.storage import (
    create_chat_session,
    delete_chat_session,
    ensure_default_chat_session,
    fetch_chat_session_messages,
    list_chat_sessions,
    touch_chat_session,
)

from ..models.llm_model import LlmModel
from ..models.log_model import LogModel, get_log_model
from ..threading.runner import run_in_thread


class LogViewModel(QObject):
    """Coordinates log persistence and LLM chat streaming for the UI."""

    llm_response_started = Signal()
    llm_thinking_started = Signal()
    llm_thinking_token_received = Signal(str)
    llm_thinking_finished = Signal(str)
    llm_token_received = Signal(str)
    llm_response_finished = Signal(str)
    llm_error = Signal(str)

    sessions_refreshed = Signal(list)
    session_selected = Signal(int)
    session_messages_loaded = Signal(list)
    llm_models_refreshed = Signal(str, list)
    llm_models_refresh_failed = Signal(str, str)

    def __init__(self, log_model: LogModel, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.log_model = log_model
        self.llm_model = LlmModel(parent=self)
        self._api_key: Optional[str] = None
        self._default_context: Optional[str] = None
        self._model_name: Optional[str] = None
        self._provider: str = "openai"
        self._thinking_mode: str = "standard"
        self.current_session_id: Optional[int] = ensure_default_chat_session()
        self._pending_turn_id: Optional[str] = None
        self._pending_thinking_text: str = ""

        self.llm_model.response_started.connect(self.llm_response_started)
        self.llm_model.thinking_started.connect(self._on_llm_thinking_started)
        self.llm_model.thinking_token_received.connect(self._on_llm_thinking_token)
        self.llm_model.thinking_finished.connect(self._on_llm_thinking_finished)
        self.llm_model.token_received.connect(self.llm_token_received)
        self.llm_model.response_finished.connect(self._on_llm_finished)
        self.llm_model.error.connect(self._on_llm_error)

    def set_api_key(self, value: Optional[str]) -> None:
        self._api_key = value or None

    def set_context(self, value: Optional[str]) -> None:
        self._default_context = value or None

    def set_model_name(self, value: Optional[str]) -> None:
        self._model_name = value or None

    def set_provider(self, value: str) -> None:
        self._provider = value or "openai"

    # @ai(gpt-5.2-codex, codex-cli, feature, 2026-03-05)
    def set_thinking_mode(self, value: str) -> None:
        normalized = (value or "standard").strip().lower()
        if normalized not in {"off", "standard", "high"}:
            normalized = "standard"
        self._thinking_mode = normalized

    def log_message(
        self,
        text: str,
        *,
        level: int = logging.INFO,
        origin: str = "chat",
        session_id: Optional[int] = None,
        turn_id: Optional[str] = None,
    ) -> None:
        self.log_model.log_text(
            text,
            level=level,
            origin=origin,
            session_id=session_id,
            turn_id=turn_id,
        )

    def ask_llm(
        self,
        prompt: str,
        *,
        context: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[int] = None,
    ) -> None:
        cleaned = prompt.strip()
        if not cleaned:
            return

        resolved_session_id = int(session_id or self.current_session_id or ensure_default_chat_session())
        self.current_session_id = resolved_session_id
        self._pending_turn_id = f"turn-{time.time_ns()}"

        self.log_model.log_text(
            cleaned,
            level=logging.INFO,
            origin="chat",
            session_id=resolved_session_id,
            turn_id=self._pending_turn_id,
        )
        touch_chat_session(resolved_session_id)

        self.llm_model.ask(
            cleaned,
            context=context or self._default_context,
            api_key=api_key or self._api_key,
            model=model or self._model_name,
            provider=self._provider,
            session_id=resolved_session_id,
            thinking_mode=self._thinking_mode,
        )

    def cancel_llm(self) -> None:
        self.llm_model.cancel()

    def refresh_sessions_list(self) -> list[dict]:
        sessions = list_chat_sessions()
        if not sessions:
            ensure_default_chat_session()
            sessions = list_chat_sessions()
        normalized = [
            {
                "id": int(row.get("id")),
                "title": str(row.get("title") or "Chat"),
                "updated_at": float(row.get("updated_at") or 0.0),
                "message_count": int(row.get("message_count") or 0),
            }
            for row in sessions
        ]
        self.sessions_refreshed.emit(normalized)
        if self.current_session_id is None and normalized:
            self.select_session(int(normalized[0]["id"]))
        return normalized

    def select_session(self, session_id: int) -> list[dict]:
        self.current_session_id = int(session_id)
        rows = fetch_chat_session_messages(self.current_session_id)
        thinking_by_turn: dict[str, str] = {}
        for row in rows:
            if row.get("origin") != "llm_thinking":
                continue
            turn_id = str(row.get("turn_id") or "")
            existing = thinking_by_turn.get(turn_id, "")
            thinking_by_turn[turn_id] = f"{existing}\n{row.get('message') or ''}".strip()

        messages: list[dict] = []
        for row in rows:
            origin = str(row.get("origin") or "")
            if origin == "llm_thinking":
                continue
            turn_id = str(row.get("turn_id") or "")
            role = "error" if int(row.get("level", 0)) >= logging.ERROR else ("user" if origin == "chat" else "assistant")
            messages.append(
                {
                    "role": role,
                    "content": str(row.get("message") or ""),
                    "thinking": thinking_by_turn.get(turn_id, "") if role in {"assistant", "error"} else "",
                }
            )
        self.session_selected.emit(self.current_session_id)
        self.session_messages_loaded.emit(messages)
        return messages

    def new_session(self) -> int:
        session_id = create_chat_session("New Chat")
        self.current_session_id = session_id
        self.refresh_sessions_list()
        self.select_session(session_id)
        return session_id

    def delete_current_session(self) -> int:
        target = self.current_session_id
        if target is None:
            return ensure_default_chat_session()
        delete_chat_session(target)
        sessions = self.refresh_sessions_list()
        if sessions:
            fallback = int(sessions[0]["id"])
            self.select_session(fallback)
            return fallback
        fallback = ensure_default_chat_session()
        self.refresh_sessions_list()
        self.select_session(fallback)
        return fallback

    def set_log_database(self, path: Path) -> None:
        self.log_model.set_database(path)

    def clear_log_database(self) -> Path:
        return self.log_model.clear_storage()

    def clear_chat_history(self) -> None:
        self.log_model.clear_chat_history()
        self.refresh_sessions_list()
        if self.current_session_id is not None:
            self.select_session(self.current_session_id)

    def reset_log_database(self) -> Path:
        return self.log_model.reset_database()

    def save_log_database_as(self, path: Path) -> Path:
        return self.log_model.save_database_as(path)

    def current_log_database_path(self) -> Path:
        return self.log_model.current_database_path()

    def enabled_logger_names(self) -> list[str]:
        return self.log_model.enabled_logger_names()

    def reload_from_storage(self, *, preferred_loggers: Optional[list[str]] = None) -> None:
        self.log_model.reload_from_storage(preferred_loggers=preferred_loggers)

    # @ai(gpt-5.2-codex, codex-cli, feature, 2026-03-05)
    def refresh_provider_models(self, provider: str, *, api_key: Optional[str] = None) -> None:
        chosen = (provider or self._provider or "openai").strip() or "openai"
        resolved_key = (api_key or self._api_key or "").strip() if chosen == "openai" else ""
        if chosen == "openai" and not resolved_key:
            self.llm_models_refresh_failed.emit(chosen, "OpenAI API key is required to fetch models.")
            return

        def _run_fetch() -> list[str]:
            return self.llm_model.fetch_available_models(provider=chosen, api_key=resolved_key or None)

        def _on_result(models: list[str]) -> None:
            self.llm_models_refreshed.emit(chosen, models)

        def _on_error(message: str) -> None:
            self.llm_models_refresh_failed.emit(chosen, message)

        run_in_thread(
            _run_fetch,
            on_result=_on_result,
            on_error=_on_error,
            owner=self,
            key=f"llm_models_refresh_{chosen}",
            cancel_previous=True,
        )

    def _on_llm_finished(self, text: str) -> None:
        if text.strip():
            self.log_model.log_text(
                text,
                level=logging.INFO,
                origin="llm",
                session_id=self.current_session_id,
                turn_id=self._pending_turn_id,
            )
            if self.current_session_id is not None:
                touch_chat_session(self.current_session_id)
        self.llm_response_finished.emit(text)
        self._pending_turn_id = None
        self._pending_thinking_text = ""

    def _on_llm_error(self, message: str) -> None:
        if message:
            self.log_model.log_text(
                message,
                level=logging.ERROR,
                origin="llm",
                session_id=self.current_session_id,
                turn_id=self._pending_turn_id,
            )
        self.llm_error.emit(message)
        self._pending_turn_id = None
        self._pending_thinking_text = ""

    def _on_llm_thinking_started(self) -> None:
        self._pending_thinking_text = ""
        self.llm_thinking_started.emit()

    def _on_llm_thinking_token(self, token: str) -> None:
        if not token:
            return
        self._pending_thinking_text += token
        self.llm_thinking_token_received.emit(token)

    def _on_llm_thinking_finished(self, text: str) -> None:
        combined = (text or self._pending_thinking_text).strip()
        if combined:
            self.log_model.log_text(
                combined,
                level=logging.INFO,
                origin="llm_thinking",
                session_id=self.current_session_id,
                turn_id=self._pending_turn_id,
            )
        self.llm_thinking_finished.emit(combined)


_shared_log_view_model: Optional[LogViewModel] = None


def get_log_view_model(log_model: Optional[LogModel] = None, parent: Optional[QObject] = None) -> LogViewModel:
    """Get or create the shared LogViewModel instance."""
    global _shared_log_view_model

    if _shared_log_view_model is None:
        _shared_log_view_model = LogViewModel(log_model or get_log_model(), parent=parent)
    elif parent is not None and _shared_log_view_model.parent() is None:
        _shared_log_view_model.setParent(parent)

    return _shared_log_view_model
