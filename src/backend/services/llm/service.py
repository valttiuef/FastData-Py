from __future__ import annotations
import threading
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Protocol, Sequence

from backend.services.logging.storage import (
    ensure_default_chat_session,
    fetch_chat_session_messages,
    get_chat_session,
)

from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .types import ChatMessage


class LLMProvider(Protocol):
    name: str

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        stop_event: Optional[threading.Event] = None,
        **kwargs,
    ) -> Iterator[str]:
        ...

    def list_models(self, **kwargs) -> Sequence[str]:
        ...


@dataclass
class SessionTrimConfig:
    enabled: bool = True
    keep_last_turns: int = 8
    max_estimated_tokens: Optional[int] = 3000
    include_summary: bool = False


class LLMService:
    """Facade that dispatches chat requests to the configured provider."""

    def __init__(self) -> None:
        self._providers: Dict[str, LLMProvider] = {}
        self._current_provider: str = "openai"
        self._trim_config = SessionTrimConfig()
        self.register_provider(OpenAIProvider())
        self.register_provider(OllamaProvider())

    def register_provider(self, provider: LLMProvider) -> None:
        self._providers[provider.name] = provider

    def available_providers(self) -> Sequence[str]:
        return tuple(sorted(self._providers.keys()))

    def current_provider(self) -> str:
        return self._current_provider

    def set_provider(self, name: str) -> None:
        if name not in self._providers:
            raise ValueError(f"Unknown LLM provider: {name}")
        self._current_provider = name

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        stop_event: Optional[threading.Event] = None,
        **kwargs,
    ) -> Iterator[str]:
        provider = self._providers.get(self._current_provider)
        if provider is None:
            raise RuntimeError("No LLM provider has been registered")
        return provider.stream_chat(messages, stop_event=stop_event, **kwargs)

    # @ai(gpt-5.2-codex, codex-cli, feature, 2026-03-05)
    def list_models(self, *, provider: Optional[str] = None, **kwargs) -> list[str]:
        chosen_provider = provider or self._current_provider
        impl = self._providers.get(chosen_provider)
        if impl is None:
            raise ValueError(f"Unknown LLM provider: {chosen_provider}")
        raw_models = impl.list_models(**kwargs)
        cleaned = [str(item).strip() for item in raw_models if str(item).strip()]
        return sorted(set(cleaned), key=str.casefold)

    def stream_chat_for_session(
        self,
        prompt: str,
        *,
        session_id: Optional[int] = None,
        context: Optional[str] = None,
        stop_event: Optional[threading.Event] = None,
        **kwargs,
    ) -> Iterator[str]:
        resolved_session = int(session_id or ensure_default_chat_session())
        messages = self.build_session_payload(prompt=prompt, session_id=resolved_session, context=context)
        return self.stream_chat(messages, stop_event=stop_event, **kwargs)

    def build_session_payload(self, *, prompt: str, session_id: int, context: Optional[str]) -> list[ChatMessage]:
        system_messages: list[ChatMessage] = []
        if context:
            system_messages.append({"role": "system", "content": context})

        session_meta = get_chat_session(session_id)
        if self._trim_config.include_summary and session_meta and session_meta.get("summary"):
            system_messages.append({"role": "system", "content": str(session_meta.get("summary") or "")})

        if not self._trim_config.enabled:
            history = self._history_as_messages(session_id)
            return [*system_messages, *history, {"role": "user", "content": prompt.strip()}]

        turns = self._load_history_turns(session_id)
        turns.append([{"role": "user", "content": prompt.strip()}])
        trimmed_turns = self._trim_turns(turns, system_messages)

        payload = [*system_messages]
        for turn in trimmed_turns:
            payload.extend(turn)
        return payload

    def _history_as_messages(self, session_id: int) -> list[ChatMessage]:
        rows = fetch_chat_session_messages(session_id)
        messages: list[ChatMessage] = []
        for row in rows:
            if row.get("level", 0) >= 40:
                continue
            origin = row.get("origin")
            if origin not in {"chat", "llm"}:
                continue
            role = "user" if origin == "chat" else "assistant"
            messages.append({"role": role, "content": row.get("message", "")})
        return messages

    def _load_history_turns(self, session_id: int) -> list[list[ChatMessage]]:
        rows = fetch_chat_session_messages(session_id)
        turns: list[list[ChatMessage]] = []
        current_turn_id: Optional[str] = None
        for row in rows:
            if row.get("level", 0) >= 40:
                continue
            if row.get("origin") not in {"chat", "llm"}:
                continue
            turn_id = str(row.get("turn_id") or "")
            message: ChatMessage = {
                "role": "user" if row.get("origin") == "chat" else "assistant",
                "content": str(row.get("message") or ""),
            }
            if not turns or turn_id != current_turn_id:
                turns.append([message])
                current_turn_id = turn_id
            else:
                turns[-1].append(message)
        return turns

    def _trim_turns(self, turns: list[list[ChatMessage]], system_messages: list[ChatMessage]) -> list[list[ChatMessage]]:
        keep = max(int(self._trim_config.keep_last_turns), 1)
        trimmed = turns[-keep:]

        max_tokens = self._trim_config.max_estimated_tokens
        if not max_tokens:
            return trimmed

        budget_chars = int(max_tokens) * 4

        def _chars(payload_turns: list[list[ChatMessage]]) -> int:
            msgs = [*system_messages]
            for t in payload_turns:
                msgs.extend(t)
            return sum(len(msg["content"]) for msg in msgs)

        while len(trimmed) > 1 and _chars(trimmed) > budget_chars:
            trimmed.pop(0)
        return trimmed


_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _service
    if _service is None:
        _service = LLMService()
    return _service


__all__ = ["ChatMessage", "LLMService", "get_llm_service"]
