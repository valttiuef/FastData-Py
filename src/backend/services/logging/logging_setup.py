import logging
import sys
import threading
import traceback

from .log_service import get_log_service
from .storage import append_text_log, crash_log_path


def configure_logging(level: int = logging.INFO, *, name: str = "FastData.db") -> logging.Logger:
    """Create or fetch a sane default logger used by the backend.
    
    The LogService handler (installed below) already handles formatting and emitting,
    so we don't add a separate StreamHandler to avoid duplication.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Install the LogService handler; it will handle all log output and persistence
    service = get_log_service()
    service.install_on_logger(logger)
    return logger


def install_global_exception_hooks() -> None:
    """Install process-wide hooks so uncaught exceptions are always persisted."""

    def _handle_exception(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            return
        try:
            logging.getLogger("FastData.unhandled").critical(
                "Unhandled exception",
                exc_info=(exc_type, exc_value, exc_tb),
            )
        except Exception:
            pass
        try:
            text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb)).rstrip("\n")
            append_text_log(crash_log_path(), text)
        except Exception:
            pass

    def _handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        _handle_exception(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = _handle_exception
    threading.excepthook = _handle_thread_exception


__all__ = ["configure_logging", "install_global_exception_hooks"]
