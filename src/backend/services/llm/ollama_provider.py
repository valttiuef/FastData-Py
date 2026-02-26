
from __future__ import annotations
import threading
from typing import Iterator, Optional, Sequence

from .types import ChatMessage

# Try to import the official ollama library
try:
    import ollama
    from ollama import ResponseError

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None  # type: ignore[assignment]
    ResponseError = Exception  # type: ignore[misc, assignment]

import shutil
import subprocess
import sys
import time
import threading
from typing import Iterator, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError

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
    """Return True if an HTTP endpoint responds (any 2xx/3xx/4xx is 'up' for our purpose)."""
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            _ = resp.status
        return True
    except Exception:
        return False


def _ensure_ollama_running(host: str, wait_s: float = 3.0) -> None:
    """
    Ensure Ollama server is running for the given host.
    If not reachable, try to start it via `ollama serve`.
    """
    parsed = urlparse(host)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError(f"Invalid Ollama host URL: {host!r}")

    # A simple endpoint that exists on the Ollama server:
    # /api/tags returns known models; good for "is it alive?"
    health_url = host.rstrip("/") + "/api/tags"

    if _http_ok(health_url):
        return  # already running

    ollama_exe = shutil.which("ollama")
    if not ollama_exe:
        raise RuntimeError(
            "Ollama does not appear to be running, and the 'ollama' executable was not found on PATH. "
            "Install Ollama and/or add it to PATH, or configure a service to start it automatically."
        )

    # Start detached so your app doesn't hang waiting on it.
    # On Windows, use creation flags; on Unix, start a new session.
    kwargs: dict = {}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
        kwargs["stdin"] = subprocess.DEVNULL
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    else:
        kwargs["start_new_session"] = True
        kwargs["stdin"] = subprocess.DEVNULL
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL

    # Note: `ollama serve` listens on the default address.
    # If you need a non-default host/port, you typically configure env vars (OLLAMA_HOST)
    # before launching, or run via system service configuration.
    subprocess.Popen([ollama_exe, "serve"], **kwargs)

    # Wait briefly for it to come up
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
    ) -> Iterator[str]:
        """Stream chat completions from a local Ollama instance.

        Uses the official ollama Python library for efficient streaming.

        Raises:
            RuntimeError: If Ollama library is not installed, not running, or returns an error.
        """
        if not OLLAMA_AVAILABLE:
            raise RuntimeError(
                "The 'ollama' Python package is not installed. "
                "Please install it with: pip install ollama"
            )

        chosen_model = model or self._default_model

        try:
            # NEW: auto-start if not running
            _ensure_ollama_running(self._host)

            # Create a client with the specified host
            client = ollama.Client(host=self._host)

            # Use streaming for immediate token-by-token response
            stream = client.chat(
                model=chosen_model,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                stream=True,
                options={"temperature": temperature},
            )

            for chunk in stream:
                if stop_event is not None and stop_event.is_set():
                    break
                content = chunk.get("message", {}).get("content", "")
                if not content:
                    continue
                # Yield the entire chunk content at once for faster streaming
                yield content

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
