
from __future__ import annotations
import os
import threading
from typing import Callable, Iterator, Optional, Sequence

from openai import OpenAI

from .types import ChatMessage


class OpenAIProvider:
    name = "openai"

    def __init__(self, *, default_model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY") -> None:
        self._default_model = default_model
        self._api_key_env = api_key_env

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stop_event: Optional[threading.Event] = None,
        temperature: float = 0.6,
        on_thinking_token: Optional[Callable[[str], None]] = None,
        thinking_mode: str = "standard",
    ) -> Iterator[str]:
        key = api_key or os.getenv(self._api_key_env)
        if not key:
            raise RuntimeError("OpenAI API key is required to stream chat completions.")

        client = OpenAI(api_key=key)
        chosen_model = model or self._default_model

        request_payload = {
            "model": chosen_model,
            "messages": [{"role": item["role"], "content": item["content"]} for item in messages],
            "temperature": temperature,
            "stream": True,
        }
        mode = (thinking_mode or "standard").strip().lower()
        if mode == "high":
            request_payload["reasoning_effort"] = "high"
        elif mode == "off":
            request_payload["reasoning_effort"] = "low"

        try:
            response = client.chat.completions.create(**request_payload)
        except TypeError:
            request_payload.pop("reasoning_effort", None)
            response = client.chat.completions.create(**request_payload)

        for chunk in response:
            if stop_event is not None and stop_event.is_set():
                break
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
            if mode != "off" and reasoning and on_thinking_token is not None:
                text = str(reasoning)
                if text:
                    on_thinking_token(text)
            content = delta.content or ""
            if not content:
                continue
            for char in content:
                if stop_event is not None and stop_event.is_set():
                    break
                yield char

    # @ai(gpt-5.2-codex, codex-cli, feature, 2026-03-05)
    def list_models(self, *, api_key: Optional[str] = None) -> Sequence[str]:
        key = api_key or os.getenv(self._api_key_env)
        if not key:
            raise RuntimeError("OpenAI API key is required to fetch models.")

        client = OpenAI(api_key=key)
        try:
            response = client.models.list()
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch OpenAI models: {exc}") from exc

        return [str(item.id) for item in getattr(response, "data", []) if getattr(item, "id", None)]
