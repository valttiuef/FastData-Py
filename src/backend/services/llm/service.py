
from __future__ import annotations
import threading
from typing import Dict, Iterator, Optional, Protocol, Sequence

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


class LLMService:
    """Facade that dispatches chat requests to the configured provider."""

    def __init__(self) -> None:
        self._providers: Dict[str, LLMProvider] = {}
        self._current_provider: str = "openai"
        self.register_provider(OpenAIProvider())
        self.register_provider(OllamaProvider())

    # ------------------------------------------------------------------
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


_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _service
    if _service is None:
        _service = LLMService()
    return _service


__all__ = ["ChatMessage", "LLMService", "get_llm_service"]
