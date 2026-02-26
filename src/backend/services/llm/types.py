from typing import TypedDict


class ChatMessage(TypedDict):
    """Simple chat message envelope used by LLM providers."""

    role: str
    content: str


__all__ = ["ChatMessage"]
