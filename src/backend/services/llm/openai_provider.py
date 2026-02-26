
from __future__ import annotations
import os
import threading
from typing import Iterator, Optional, Sequence

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
    ) -> Iterator[str]:
        key = api_key or os.getenv(self._api_key_env)
        if not key:
            raise RuntimeError("OpenAI API key is required to stream chat completions.")

        client = OpenAI(api_key=key)
        chosen_model = model or self._default_model

        response = client.chat.completions.create(
            model=chosen_model,
            messages=[{"role": item["role"], "content": item["content"]} for item in messages],
            temperature=temperature,
            stream=True,
        )

        for chunk in response:
            if stop_event is not None and stop_event.is_set():
                break
            delta = chunk.choices[0].delta
            content = delta.content or ""
            if not content:
                continue
            for char in content:
                if stop_event is not None and stop_event.is_set():
                    break
                yield char
