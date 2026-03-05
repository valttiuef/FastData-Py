from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from typing import Callable, Iterator, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .types import ChatMessage

try:
    import ollama
    from ollama import ResponseError

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None  # type: ignore[assignment]
    ResponseError = Exception  # type: ignore[misc, assignment]


def _http_ok(url: str, timeout: float = 0.25) -> bool:
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            _ = resp.status
        return True
    except Exception:
        return False


def _ensure_ollama_running(host: str, wait_s: float = 3.0) -> None:
    parsed = urlparse(host)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError(f"Invalid Ollama host URL: {host!r}")

    health_url = host.rstrip("/") + "/api/tags"
    if _http_ok(health_url):
        return

    ollama_exe = shutil.which("ollama")
    if not ollama_exe:
        raise RuntimeError(
            "Ollama does not appear to be running, and the 'ollama' executable was not found on PATH. "
            "Install Ollama and/or add it to PATH, or configure a service to start it automatically."
        )

    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True

    subprocess.Popen([ollama_exe, "serve"], **kwargs)

    deadline = time.time() + wait_s
    while time.time() < deadline:
        if _http_ok(health_url, timeout=0.3):
            return
        time.sleep(0.1)

    raise RuntimeError(
        f"Tried to start Ollama with `ollama serve` but could not reach it at {host} "
        f"within {wait_s:.1f}s."
    )


class OllamaProvider:
    """LLM provider that connects to a local Ollama instance using the official library."""

    name = "ollama"

    def __init__(
        self,
        *,
        default_model: str = "llama3.2",
        host: str = "http://localhost:11434",
    ) -> None:
        self._default_model = default_model
        self._host = host

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
        if not OLLAMA_AVAILABLE:
            raise RuntimeError(
                "The 'ollama' Python package is not installed. "
                "Please install it with: pip install ollama"
            )

        chosen_model = model or self._default_model
        in_think_block = False

        try:
            _ensure_ollama_running(self._host)
            client = ollama.Client(host=self._host)
            mode = (thinking_mode or "standard").strip().lower()
            options = {"temperature": temperature}
            if mode == "off":
                options["think"] = False
            elif mode in {"standard", "high"}:
                options["think"] = True
            stream = client.chat(
                model=chosen_model,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                stream=True,
                options=options,
            )

            for chunk in stream:
                if stop_event is not None and stop_event.is_set():
                    break

                message = chunk.get("message", {})
                thinking = message.get("thinking")
                if mode != "off" and thinking and on_thinking_token is not None:
                    on_thinking_token(str(thinking))

                content = str(message.get("content", "") or "")
                if not content:
                    continue

                # Some reasoning models stream <think>...</think> in content.
                while content:
                    if in_think_block:
                        end_idx = content.find("</think>")
                        if end_idx >= 0:
                            part = content[:end_idx]
                            if mode != "off" and part and on_thinking_token is not None:
                                on_thinking_token(part)
                            content = content[end_idx + len("</think>"):]
                            in_think_block = False
                        else:
                            if mode != "off" and on_thinking_token is not None:
                                on_thinking_token(content)
                            content = ""
                    else:
                        start_idx = content.find("<think>")
                        if start_idx < 0:
                            yield content
                            content = ""
                        else:
                            before = content[:start_idx]
                            if before:
                                yield before
                            content = content[start_idx + len("<think>"):]
                            in_think_block = True

        except ResponseError as e:
            raise RuntimeError(f"Ollama error: {e}") from e
        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg or "connect" in error_msg.lower():
                raise RuntimeError(
                    f"Could not connect to Ollama at {self._host}. "
                    f"Please ensure Ollama is running. Error: {error_msg}"
                ) from e
            raise RuntimeError(f"Ollama error: {error_msg}") from e

    # @ai(gpt-5.2-codex, codex-cli, feature, 2026-03-05)
    def list_models(self, **_kwargs) -> Sequence[str]:
        if not OLLAMA_AVAILABLE:
            raise RuntimeError(
                "The 'ollama' Python package is not installed. "
                "Please install it with: pip install ollama"
            )

        try:
            _ensure_ollama_running(self._host)
            client = ollama.Client(host=self._host)
            data = client.list()
            if isinstance(data, dict):
                models = data.get("models", [])
            else:
                models = getattr(data, "models", []) or []

            names: list[str] = []
            for item in models:
                if isinstance(item, dict):
                    value = item.get("name") or item.get("model")
                else:
                    value = getattr(item, "name", None) or getattr(item, "model", None)
                text = str(value or "").strip()
                if text:
                    names.append(text)
            return names
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch Ollama models: {exc}") from exc
