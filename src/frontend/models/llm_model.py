
from __future__ import annotations
from typing import Optional

from PySide6.QtCore import QObject, Signal

from backend.services.llm import ChatMessage, get_llm_service

from ..threading.runner import run_in_thread, stop_owner_threads


class LlmModel(QObject):
    """Qt-facing wrapper around the backend LLM service with streaming support."""

    response_started = Signal()
    token_received = Signal(str)
    response_finished = Signal(str)
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

    # ------------------------------------------------------------------
    def ask(
        self,
        prompt: str,
        *,
        context: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        prompt = prompt.strip()
        if not prompt:
            return

        messages: list[ChatMessage] = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})

        self.cancel()
        self.response_started.emit()

        def _run_stream(result_callback, stop_event):
            service = get_llm_service()
            # Set the provider before streaming
            if provider:
                service.set_provider(provider)
            collected: list[str] = []
            for token in service.stream_chat(
                messages, api_key=api_key, model=model, stop_event=stop_event
            ):
                collected.append(token)
                result_callback(token)
                if stop_event.is_set():
                    break
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

    # ------------------------------------------------------------------
    def _on_finished(self, text: str) -> None:
        self.response_finished.emit(text)

    def _on_error(self, message: str) -> None:
        self.error.emit(message)
