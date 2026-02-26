
from __future__ import annotations
import io
import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .storage import (
    append_text_log,
    error_log_path,
    fetch_log_records,
    save_log_record,
    warning_log_path,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LogEvent:
    """Container describing a single log entry destined for the UI."""

    message: str
    level: int
    logger_name: str
    created: float
    origin: str
    formatted: str


Listener = Callable[[LogEvent], None]
LoggerListener = Callable[[Sequence[str]], None]


class LogService(logging.Handler):
    """Central logging handler that relays log messages to registered listeners."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self._listeners: List[Listener] = []
        self._lock = threading.RLock()
        self._formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        self._stdout_proxy: Optional[_StreamProxy] = None
        self._stderr_proxy: Optional[_StreamProxy] = None
        self._installed = False
        self._logger_names: set[str] = set()
        self._logger_listeners: List[LoggerListener] = []
        self._shutdown = False  # Flag to prevent notifications after shutdown

    # ------------------------------------------------------------------
    def emit(self, record: logging.LogRecord) -> None:
        try:
            formatted = self._formatter.format(record)
        except Exception:  # pragma: no cover - defensive, mirrors logging.Handler
            self.handleError(record)
            return
        self._register_logger_name(record.name)
        event = LogEvent(
            message=record.getMessage(),
            level=record.levelno,
            logger_name=record.name,
            created=record.created,
            origin="logging",
            formatted=formatted,
        )
        self._persist_event(event)
        self._notify(event)

    # ------------------------------------------------------------------
    def log_text(self, message: str, *, level: int = logging.INFO, origin: str = "app") -> None:
        """Record a free-form message (for chat/stdout mirroring)."""

        created = time.time()
        clean_message = message.rstrip("\n")
        if not clean_message:
            return
        self._register_logger_name(origin)
        record = logging.LogRecord(
            name=origin,
            level=level,
            pathname="",
            lineno=0,
            msg=clean_message,
            args=(),
            exc_info=None,
        )
        record.created = created
        event = LogEvent(
            message=clean_message,
            level=level,
            logger_name=origin,
            created=created,
            origin=origin,
            formatted=self._formatter.format(record),
        )
        self._persist_event(event)
        self._notify(event)

    # ------------------------------------------------------------------
    def add_listener(self, listener: Listener) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    # ------------------------------------------------------------------
    def ensure_installed(self) -> None:
        """Attach the handler to the root logger and mirror stdout/stderr once."""

        with self._lock:
            if self._installed:
                return
            root = logging.getLogger()
            if self not in root.handlers:
                root.addHandler(self)
            root.setLevel(min(root.level, logging.INFO))
            self._mirror_streams()
            self._installed = True
            self._register_logger_name(root.name)

    def install_on_logger(self, logger: logging.Logger) -> None:
        """Attach the handler to the provided logger if not already present."""

        with self._lock:
            if self not in logger.handlers:
                logger.addHandler(self)
            logger.propagate = False
        self._register_logger_name(logger.name)
        self.ensure_installed()

    # ------------------------------------------------------------------
    def load_persisted_events(
        self,
        *,
        keyword: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> List[LogEvent]:
        try:
            rows = fetch_log_records(keyword=keyword, start=start, end=end)
        except Exception:
            return []
        events: List[LogEvent] = []
        for row in rows:
            event = LogEvent(
                message=row.get("message", ""),
                level=int(row.get("level", logging.INFO)),
                logger_name=row.get("logger_name", ""),
                created=float(row.get("created_at", 0.0)),
                origin=row.get("origin", "logging"),
                formatted=row.get("formatted", row.get("message", "")),
            )
            events.append(event)
        return events

    # ------------------------------------------------------------------
    def _notify(self, event: LogEvent) -> None:
        # Skip notifications if service is shutting down to avoid "signal source deleted" errors
        if self._shutdown:
            return
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                _safe_print_exception()

    def add_logger_listener(self, listener: LoggerListener) -> None:
        with self._lock:
            if listener not in self._logger_listeners:
                self._logger_listeners.append(listener)
            names = sorted(self._logger_names)
        try:
            listener(names)
        except Exception:
            _safe_print_exception()

    def remove_logger_listener(self, listener: LoggerListener) -> None:
        with self._lock:
            if listener in self._logger_listeners:
                self._logger_listeners.remove(listener)

    def shutdown(self) -> None:
        """
        Cleanly shutdown the log service.
        Prevents further Qt signal notifications to avoid "signal source deleted" errors
        when the service runs longer than the UI objects.
        """
        with self._lock:
            self._shutdown = True
            self._listeners.clear()
            self._logger_listeners.clear()

    def get_logger_names(self) -> List[str]:
        with self._lock:
            return sorted(self._logger_names)

    def _mirror_streams(self) -> None:
        if self._stdout_proxy is None:
            self._stdout_proxy = _StreamProxy(sys.stdout, self, logging.INFO, "stdout")
            sys.stdout = self._stdout_proxy  # type: ignore[assignment]
        if self._stderr_proxy is None:
            self._stderr_proxy = _StreamProxy(sys.stderr, self, logging.ERROR, "stderr")
            sys.stderr = self._stderr_proxy  # type: ignore[assignment]
        self._register_logger_name("stdout")
        self._register_logger_name("stderr")

    def _register_logger_name(self, name: str) -> None:
        if not name:
            return
        with self._lock:
            if name in self._logger_names:
                return
            self._logger_names.add(name)
            listeners = list(self._logger_listeners)
            names = sorted(self._logger_names)
        for listener in listeners:
            try:
                listener(names)
            except Exception:
                _safe_print_exception()

    def _persist_event(self, event: LogEvent) -> None:
        try:
            save_log_record(
                created_at=event.created,
                level=event.level,
                logger_name=event.logger_name,
                origin=event.origin,
                message=event.message,
                formatted=event.formatted,
            )
            if int(event.level) >= int(logging.WARNING):
                append_text_log(warning_log_path(), event.formatted)
            if int(event.level) >= int(logging.ERROR):
                append_text_log(error_log_path(), event.formatted)
        except Exception:
            _safe_print_exception()


class _StreamProxy(io.TextIOBase):
    """Tee writes to the original stream and to the log service."""

    def __init__(self, stream: io.TextIOBase, service: LogService, level: int, origin: str) -> None:
        super().__init__()
        self._stream = stream
        self._service = service
        self._level = level
        self._origin = origin
        self._buffer: List[str] = []

    def writable(self) -> bool:  # pragma: no cover - passthrough
        return True

    def write(self, s: str) -> int:
        # if stream is missing, don't crash – still keep buffer behavior
        if self._stream is None:
            if s:
                self._buffer.append(s)
                self._flush_buffer(False)
            return len(s or "")

        written = self._stream.write(s)
        try:
            self._stream.flush()
        except Exception:
            logger.warning("Exception in write", exc_info=True)

        if not s:
            return written

        self._buffer.append(s)
        self._flush_buffer(False)
        return written

    def flush(self) -> None:
        self._stream.flush()
        self._flush_buffer(True)

    def isatty(self) -> bool:  # pragma: no cover - passthrough
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def fileno(self):  # pragma: no cover - passthrough
        try:
            return self._stream.fileno()
        except Exception:
            raise io.UnsupportedOperation("fileno")

    @property
    def encoding(self):  # pragma: no cover - passthrough
        return getattr(self._stream, "encoding", None)

    @property
    def errors(self):  # pragma: no cover - passthrough
        return getattr(self._stream, "errors", None)

    def __getattr__(self, item):  # pragma: no cover - passthrough fallback
        return getattr(self._stream, item)

    def _flush_buffer(self, final: bool) -> None:
        # Skip flushing if service is shutting down to prevent cascading errors
        if self._service._shutdown:
            self._buffer.clear()
            return
        if not self._buffer:
            return
        data = "".join(self._buffer)
        if final:
            self._buffer.clear()
        lines = data.splitlines(keepends=not final)
        remainder = ""
        for line in lines:
            if line.endswith("\n") or final:
                text = line.rstrip("\n")
                if text:
                    try:
                        self._service.log_text(text, level=self._level, origin=self._origin)
                    except Exception:
                        # Prevent errors during logging from cascading further
                        logger.warning("Exception in _flush_buffer", exc_info=True)
            else:
                remainder += line
        if final and remainder:
            try:
                self._service.log_text(remainder, level=self._level, origin=self._origin)
            except Exception:
                # Prevent errors during logging from cascading further
                logger.warning("Exception in _flush_buffer", exc_info=True)
        elif not final:
            self._buffer = [remainder] if remainder else []
        else:
            self._buffer = []


def _safe_print_exception() -> None:  # pragma: no cover - best effort
    try:
        import traceback

        stream = getattr(sys, "__stderr__", None)
        if stream is None:
            stream = getattr(sys, "__stdout__", None)
        traceback.print_exc(file=stream)
    except Exception:
        logger.warning("Exception in _safe_print_exception", exc_info=True)


_service = LogService()


def get_log_service() -> LogService:
    return _service
