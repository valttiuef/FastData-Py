# --- @ai START ---
# model: gpt-5
# tool: codex
# role: observability
# reviewed: no
# date: 2026-03-23
# --- @ai END ---
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Optional

from .storage import append_text_log, performance_debug_log_path
import logging

logger = logging.getLogger(__name__)

_EVENT_LOCK = threading.RLock()
_EVENT_COUNTER = 0
_ENABLED_VALUES = {"1", "true", "yes", "on", "y"}


def perf_debug_enabled(env_var: str = "FASTDATA_PERF_DEBUG") -> bool:
    raw = str(os.getenv(env_var, "")).strip().lower()
    return raw in _ENABLED_VALUES


def _next_event_id() -> int:
    global _EVENT_COUNTER
    with _EVENT_LOCK:
        _EVENT_COUNTER += 1
        return int(_EVENT_COUNTER)


def _safe_text(value: object, *, max_len: int = 220) -> str:
    text = str(value).replace("\n", " ").strip()
    if not text:
        return ""
    if len(text) > max_len:
        return f"{text[: max_len - 3]}..."
    return text


def perf_debug_log(
    scope: str,
    step: str,
    *,
    elapsed_ms: Optional[float] = None,
    env_var: str = "FASTDATA_PERF_DEBUG",
    event_key: str = "id",
    **fields: object,
) -> None:
    if not perf_debug_enabled(env_var):
        return
    event_id = _next_event_id()
    parts: list[str] = [
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}",
        f"{event_key}={event_id}",
        f"scope={_safe_text(scope, max_len=120)}",
        f"step={_safe_text(step, max_len=120)}",
    ]
    if elapsed_ms is not None:
        try:
            parts.append(f"ms={float(elapsed_ms):.2f}")
        except Exception:
            pass
    for key, value in fields.items():
        if value is None:
            continue
        text = _safe_text(value)
        if not text:
            continue
        parts.append(f"{key}={text}")
    try:
        append_text_log(performance_debug_log_path(), " | ".join(parts))
    except Exception:
        logger.warning("Failed to write performance debug line.", exc_info=True)


class PerfDebugStopwatch:
    def __init__(
        self,
        scope: str,
        *,
        env_var: str = "FASTDATA_PERF_DEBUG",
        event_key: str = "id",
        **base_fields: object,
    ) -> None:
        self._scope = str(scope)
        self._env_var = str(env_var or "FASTDATA_PERF_DEBUG")
        self._event_key = str(event_key or "id")
        self._base_fields = dict(base_fields)
        self._started = time.perf_counter()
        self._last = self._started

    def log(self, step: str, *, elapsed_ms: Optional[float] = None, **fields: object) -> None:
        merged = dict(self._base_fields)
        merged.update(fields)
        perf_debug_log(
            self._scope,
            step,
            elapsed_ms=elapsed_ms,
            env_var=self._env_var,
            event_key=self._event_key,
            **merged,
        )

    def mark(self, step: str, **fields: object) -> float:
        now = time.perf_counter()
        elapsed_ms = (now - self._last) * 1000.0
        self._last = now
        self.log(step, elapsed_ms=elapsed_ms, **fields)
        return float(elapsed_ms)

    def total(self, step: str = "total", **fields: object) -> float:
        now = time.perf_counter()
        elapsed_ms = (now - self._started) * 1000.0
        self.log(step, elapsed_ms=elapsed_ms, **fields)
        return float(elapsed_ms)
