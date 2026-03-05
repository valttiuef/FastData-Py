
from __future__ import annotations
from typing import Optional

from PySide6.QtCore import QObject, Signal

from backend.services.llm import get_llm_service

from ..threading.runner import run_in_thread, stop_owner_threads


class LlmModel(QObject):
    """Qt-facing wrapper around the backend LLM service with streaming support."""

    response_started = Signal()
    thinking_started = Signal()
    thinking_token_received = Signal(str)
    thinking_finished = Signal(str)
    token_received = Signal(str)
    response_finished = Signal(str)
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

    # ------------------------------------------------------------------
    # @ai(gpt-5.2-codex, codex-cli, refactor, 2026-03-05)
    def ask(
        self,
        prompt: str,
        *,
        context: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        session_id: Optional[int] = None,
        thinking_mode: str = "standard",
    ) -> None:
        prompt = prompt.strip()
        if not prompt:
            return

        mode = (thinking_mode or "standard").strip().lower()
        self.cancel()
        self.response_started.emit()
        if mode != "off":
            self.thinking_started.emit()

        def _run_stream(result_callback, stop_event):
            service = get_llm_service()
            # Set the provider before streaming
            if provider:
                service.set_provider(provider)
            collected: list[str] = []
            thinking_collected: list[str] = []
            thinking_finished_emitted = False

            def _emit_thinking_finished() -> None:
                nonlocal thinking_finished_emitted
                if mode == "off" or thinking_finished_emitted:
                    return
                self.thinking_finished.emit("".join(thinking_collected))
                thinking_finished_emitted = True

            def _on_thinking(token: str) -> None:
                if mode == "off":
                    return
                if thinking_finished_emitted:
                    return
                if not token:
                    return
                thinking_collected.append(token)
                self.thinking_token_received.emit(token)

            for token in service.stream_chat_for_session(
                prompt,
                session_id=session_id,
                context=context,
                api_key=api_key,
                model=model,
                stop_event=stop_event,
                on_thinking_token=_on_thinking if mode != "off" else None,
                thinking_mode=mode,
            ):
                if token and mode != "off" and not thinking_finished_emitted:
                    # Consider reasoning phase done once answer tokens start streaming.
                    _emit_thinking_finished()
                collected.append(token)
                result_callback(token)
                if stop_event.is_set():
                    break
            _emit_thinking_finished()
            return "".join(collected)

        run_in_thread(
            _run_stream,
            on_result=self._on_finished,
            on_error=self._on_error,
            on_intermediate_result=self.token_received.emit,
            owner=self,
            key="llm_request",
            cancel_previous=True,
        )

    def cancel(self) -> None:
        stop_owner_threads(self, key="llm_request", wait=False)

    # @ai(gpt-5.2-codex, codex-cli, feature, 2026-03-05)
    def fetch_available_models(
        self,
        *,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> list[str]:
        service = get_llm_service()
        chosen = provider or service.current_provider()
        service.set_provider(chosen)
        kwargs = {"api_key": api_key} if chosen == "openai" else {}
        return service.list_models(provider=chosen, **kwargs)

    # ------------------------------------------------------------------
    def _on_finished(self, text: str) -> None:
        self.response_finished.emit(text)

    def _on_error(self, message: str) -> None:
        self.error.emit(message)
