from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
"""Frontend utilities: expose global helpers for status text and progress

This module provides simple global functions other windows/widgets can call
without holding a reference to the main window. The main window should call
`register_main_window(window)` during initialization so the helpers forward to
real widgets. If no main window is registered the calls are remembered and
applied when a window is registered.
"""

import weakref
import time
from pathlib import Path
from typing import Optional
from PySide6.QtCore import QCoreApplication, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from ..threading.utils import run_in_main_thread
from ..viewmodels.log_view_model import get_log_view_model

# Weak reference to the main window
_main_window_ref: Optional[weakref.ref] = None

# Pending values applied once a window is registered
_pending_status: Optional[str] = ""
_pending_progress: Optional[int] = None
_pending_toasts: list[tuple[str, str, str, Optional[int], Optional[str], Optional[object], Optional[str]]] = []
_pending_toast_flush_scheduled: bool = False
_recent_toast_times: dict[tuple[str, str, str, Optional[str]], float] = {}

_DATABASE_IN_USE_MARKERS: tuple[str, ...] = (
    "database file is in use",
    "database is locked",
    "database is busy",
    "could not set lock on file",
    "conflicting lock is held",
    "being used by another process",
    "the process cannot access the file",
    "resource temporarily unavailable",
    "access is denied",
    "permission denied",
)


def _is_database_in_use_message(message: str) -> bool:
    text = str(message or "").strip().casefold()
    if not text:
        return False
    return any(marker in text for marker in _DATABASE_IN_USE_MARKERS)


def _window_ready_for_toasts(window) -> bool:
    if window is None:
        return False
    if getattr(window, "toast_manager", None) is None:
        return False
    try:
        return bool(window.isVisible())
    except Exception:
        return False


def _normalize_toast_key(
    kind: str,
    title: str,
    message: str,
    tab_key: Optional[str],
) -> tuple[str, str, str, Optional[str]]:
    return (
        str(kind or "").strip().casefold(),
        str(title or "").strip().casefold(),
        " ".join(str(message or "").split()).strip().casefold(),
        tab_key,
    )


def _should_emit_toast(
    kind: str,
    title: str,
    message: str,
    tab_key: Optional[str],
) -> bool:
    key = _normalize_toast_key(kind, title, message, tab_key)
    now = time.monotonic()
    cooldown = 20.0 if _is_database_in_use_message(message) else 2.0
    last = _recent_toast_times.get(key)
    if last is not None and (now - last) < cooldown:
        return False
    _recent_toast_times[key] = now
    stale_before = now - 180.0
    stale_keys = [k for k, ts in _recent_toast_times.items() if ts < stale_before]
    for stale_key in stale_keys:
        _recent_toast_times.pop(stale_key, None)
    return True


def _enqueue_toast(
    kind: str,
    title: str,
    message: str,
    msec: Optional[int],
    tab_key: Optional[str],
    on_click,
    icon_key: Optional[str],
) -> None:
    item = (kind, title, str(message), msec, tab_key, on_click, icon_key)
    if _pending_toasts and _pending_toasts[-1] == item:
        return
    _pending_toasts.append(item)
    del _pending_toasts[:-10]
    _schedule_pending_toast_flush()


def _flush_pending_toasts() -> None:
    global _pending_toast_flush_scheduled
    _pending_toast_flush_scheduled = False
    if not _pending_toasts:
        return
    w = _get_window()
    if not _window_ready_for_toasts(w):
        _schedule_pending_toast_flush(delay_ms=250)
        return
    pending = list(_pending_toasts)
    _pending_toasts.clear()
    for kind, title, message, msec, tab_key, on_click, icon_key in pending:
        _dispatch_toast(
            kind,
            message,
            title=title,
            msec=msec,
            tab_key=tab_key,
            on_click=on_click,
            icon_key=icon_key,
        )


def _schedule_pending_toast_flush(*, delay_ms: int = 120) -> None:
    global _pending_toast_flush_scheduled
    if _pending_toast_flush_scheduled:
        return
    if QCoreApplication.instance() is None:
        return
    _pending_toast_flush_scheduled = True
    QTimer.singleShot(max(0, int(delay_ms)), _flush_pending_toasts)


def _log_toast(kind: str, message: str, *, title: str = "", tab_key: Optional[str] = None) -> None:
    text = str(message or "").strip()
    if not text:
        return
    parts = []
    title_text = str(title or "").strip()
    if title_text:
        parts.append(title_text)
    if tab_key:
        parts.append(f"tab={tab_key}")
    prefix = " | ".join(parts)
    payload = f"{prefix}: {text}" if prefix else text

    level = logging.INFO
    if kind == "warn":
        level = logging.WARNING
    elif kind == "error":
        level = logging.ERROR

    try:
        get_log_view_model().log_message(payload, level=level, origin="ui.toast")
    except Exception:
        logger.warning("Exception in _log_toast", exc_info=True)


def _get_window():
    global _main_window_ref
    if _main_window_ref is None:
        return None
    w = _main_window_ref()
    if w is None:
        _main_window_ref = None
        return None
    return w


def register_main_window(win) -> None:
    # @ai(gpt-5, codex-cli, bugfix, 2026-03-23)
    """Register the application main window so helpers forward to it.

    The function will also apply any pending status/progress values.
    """
    global _main_window_ref, _pending_status, _pending_progress
    _main_window_ref = weakref.ref(win)
    if _pending_status:
        try:
            win.set_status_text(_pending_status)
        except Exception:
            logger.warning("Exception in register_main_window", exc_info=True)
        _pending_status = ""
    if _pending_progress is not None:
        try:
            win.set_progress(_pending_progress)
        except Exception:
            logger.warning("Exception in register_main_window", exc_info=True)
        _pending_progress = None

    if _pending_toasts:
        _schedule_pending_toast_flush(delay_ms=120)


def unregister_main_window() -> None:
    """Unregister the main window reference."""
    global _main_window_ref
    _main_window_ref = None


def set_status_text(text: Optional[str]) -> None:
    """Set right-aligned status text (best-effort).

    If no main window is registered the value is kept pending and applied
    when `register_main_window` is called.
    """
    global _pending_status
    w = _get_window()
    if w is not None:
        try:
            run_in_main_thread(w.set_status_text, text)
            return
        except Exception:
            logger.warning("Exception in set_status_text", exc_info=True)
    # store pending
    _pending_status = str(text) if text is not None else ""


def clear_status_text() -> None:
    """Clear status text (best-effort)."""
    w = _get_window()
    if w is not None:
        try:
            run_in_main_thread(w.clear_status_text)
            return
        except Exception:
            logger.warning("Exception in clear_status_text", exc_info=True)
    global _pending_status
    _pending_status = ""


def set_progress(percent: Optional[int]) -> None:
    """Set progress (0-100). Passing None hides the progress bar.

    If main window isn't available the value is stored pending.
    """
    global _pending_progress
    w = _get_window()
    if w is not None:
        try:
            run_in_main_thread(w.set_progress, percent)
            return
        except Exception:
            logger.warning("Exception in set_progress", exc_info=True)
    # pending store (normalize)
    if percent is None:
        _pending_progress = None
    else:
        try:
            _pending_progress = max(0, min(100, int(percent)))
        except Exception:
            _pending_progress = None


def clear_progress() -> None:
    """Hide and reset progress (best-effort)."""
    w = _get_window()
    if w is not None:
        try:
            run_in_main_thread(w.clear_progress)
            return
        except Exception:
            logger.warning("Exception in clear_progress", exc_info=True)
    global _pending_progress
    _pending_progress = None


def increment_progress(delta: int = 1) -> None:
    """Increment progress by `delta` (best-effort). If no window is present,
    increments the pending progress value.
    """
    w = _get_window()
    global _pending_progress
    if w is not None:
        try:
            def _increment() -> None:
                cur = 0
                try:
                    cur = int(w._progress.value())
                except Exception:
                    cur = 0
                w.set_progress(cur + int(delta))

            run_in_main_thread(_increment)
            return
        except Exception:
            logger.warning("Exception in increment_progress", exc_info=True)
    # no window: update pending
    if _pending_progress is None:
        _pending_progress = 0
    try:
        _pending_progress = max(0, min(100, int(_pending_progress) + int(delta)))
    except Exception:
        _pending_progress = None


def _dispatch_toast(
    kind: str,
    message: str,
    *,
    title: str = "",
    msec: Optional[int] = None,
    tab_key: Optional[str] = None,
    on_click=None,
    icon_key: Optional[str] = None,
) -> None:
    w = _get_window()
    if w is None:
        return
    manager = getattr(w, "toast_manager", None)
    if manager is None:
        return
    def _emit() -> None:
        try:
            if kind == "success":
                manager.success(
                    message,
                    title=title or "Done",
                    msec=msec,
                    tab_key=tab_key,
                    on_click=on_click,
                    icon_key=icon_key,
                )
            elif kind == "warn":
                manager.warn(
                    message,
                    title=title or "Warning",
                    msec=msec,
                    tab_key=tab_key,
                    on_click=on_click,
                    icon_key=icon_key,
                )
            elif kind == "error":
                manager.error(
                    message,
                    title=title or "Error",
                    msec=msec,
                    tab_key=tab_key,
                    on_click=on_click,
                    icon_key=icon_key,
                )
            else:
                manager.info(
                    message,
                    title=title or "Info",
                    msec=msec,
                    tab_key=tab_key,
                    on_click=on_click,
                    icon_key=icon_key,
                )
        except Exception:
            logger.warning("Exception in _dispatch_toast", exc_info=True)

    run_in_main_thread(_emit)


def toast_info(
    message: str,
    *,
    title: str = "Info",
    msec: Optional[int] = None,
    tab_key: Optional[str] = None,
    on_click=None,
    icon_key: Optional[str] = None,
) -> None:
    """Show a non-blocking toast (best-effort). Falls back to pending queue."""
    if not _should_emit_toast("info", title, message, tab_key):
        return
    _log_toast("info", message, title=title, tab_key=tab_key)
    w = _get_window()
    if _window_ready_for_toasts(w):
        _dispatch_toast("info", message, title=title, msec=msec, tab_key=tab_key, on_click=on_click, icon_key=icon_key)
        return
    _enqueue_toast("info", title, str(message), msec, tab_key, on_click, icon_key)


def toast_success(
    message: str,
    *,
    title: str = "Done",
    msec: Optional[int] = None,
    tab_key: Optional[str] = None,
    on_click=None,
    icon_key: Optional[str] = None,
) -> None:
    if not _should_emit_toast("success", title, message, tab_key):
        return
    _log_toast("success", message, title=title, tab_key=tab_key)
    w = _get_window()
    if _window_ready_for_toasts(w):
        _dispatch_toast("success", message, title=title, msec=msec, tab_key=tab_key, on_click=on_click, icon_key=icon_key)
        return
    _enqueue_toast("success", title, str(message), msec, tab_key, on_click, icon_key)


def toast_warn(
    message: str,
    *,
    title: str = "Warning",
    msec: Optional[int] = None,
    tab_key: Optional[str] = None,
    on_click=None,
    icon_key: Optional[str] = None,
) -> None:
    if not _should_emit_toast("warn", title, message, tab_key):
        return
    _log_toast("warn", message, title=title, tab_key=tab_key)
    w = _get_window()
    if _window_ready_for_toasts(w):
        _dispatch_toast("warn", message, title=title, msec=msec, tab_key=tab_key, on_click=on_click, icon_key=icon_key)
        return
    _enqueue_toast("warn", title, str(message), msec, tab_key, on_click, icon_key)


def toast_error(
    # @ai(gpt-5, codex-cli, bugfix, 2026-03-23)
    message: str,
    *,
    title: str = "Error",
    msec: Optional[int] = None,
    tab_key: Optional[str] = None,
    on_click=None,
    icon_key: Optional[str] = None,
) -> None:
    if _is_database_in_use_message(message):
        set_status_text(
            "Database in use. Close other apps using the database file, then use Refresh Databases or create a new database."
        )
        return
    if not _should_emit_toast("error", title, message, tab_key):
        return
    _log_toast("error", message, title=title, tab_key=tab_key)
    w = _get_window()
    if _window_ready_for_toasts(w):
        _dispatch_toast("error", message, title=title, msec=msec, tab_key=tab_key, on_click=on_click, icon_key=icon_key)
        return
    _enqueue_toast("error", title, str(message), msec, tab_key, on_click, icon_key)


# @ai(gpt-5, codex, refactor, 2026-03-11)
def toast_file_saved(
    path: str | Path,
    *,
    title: str = "File saved",
    tab_key: Optional[str] = None,
    msec: Optional[int] = 6500,
) -> None:
    saved_path = Path(path)
    message = f"{saved_path.name} saved. Click to open."
    if saved_path.suffix == "":
        message = f"Export location ready: {saved_path.name}. Click to open."

    def _open_saved_path() -> None:
        target = saved_path
        if not target.exists():
            toast_warn(f"Saved path not found: {target}", title="Open file", tab_key=tab_key)
            return
        url = QUrl.fromLocalFile(str(target))
        opened = bool(QDesktopServices.openUrl(url))
        if not opened:
            toast_warn(f"Could not open: {target}", title="Open file", tab_key=tab_key)

    toast_success(
        message,
        title=title,
        tab_key=tab_key,
        on_click=_open_saved_path,
        msec=msec,
        icon_key="open_file",
    )
