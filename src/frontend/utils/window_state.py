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
from typing import Optional
from ..viewmodels.log_view_model import get_log_view_model

# Weak reference to the main window
_main_window_ref: Optional[weakref.ref] = None

# Pending values applied once a window is registered
_pending_status: Optional[str] = ""
_pending_progress: Optional[int] = None
_pending_toasts: list[tuple[str, str, str, Optional[int], Optional[str], Optional[object]]] = []


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
        pending = list(_pending_toasts)
        _pending_toasts.clear()
        for kind, title, message, msec, tab_key, on_click in pending:
            _dispatch_toast(kind, message, title=title, msec=msec, tab_key=tab_key, on_click=on_click)


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
            w.set_status_text(text)
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
            w.clear_status_text()
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
            w.set_progress(percent)
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
            w.clear_progress()
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
            # read current value in a defensive way then set new value
            cur = 0
            try:
                cur = int(w._progress.value())
            except Exception:
                cur = 0
            w.set_progress(cur + int(delta))
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
) -> None:
    w = _get_window()
    if w is None:
        return
    manager = getattr(w, "toast_manager", None)
    if manager is None:
        return
    try:
        if kind == "success":
            manager.success(message, title=title or "Done", msec=msec, tab_key=tab_key, on_click=on_click)
        elif kind == "warn":
            manager.warn(message, title=title or "Warning", msec=msec, tab_key=tab_key, on_click=on_click)
        elif kind == "error":
            manager.error(message, title=title or "Error", msec=msec, tab_key=tab_key, on_click=on_click)
        else:
            manager.info(message, title=title or "Info", msec=msec, tab_key=tab_key, on_click=on_click)
    except Exception:
        logger.warning("Exception in _dispatch_toast", exc_info=True)


def toast_info(
    message: str,
    *,
    title: str = "Info",
    msec: Optional[int] = None,
    tab_key: Optional[str] = None,
    on_click=None,
) -> None:
    """Show a non-blocking toast (best-effort). Falls back to pending queue."""
    _log_toast("info", message, title=title, tab_key=tab_key)
    w = _get_window()
    if w is not None and getattr(w, "toast_manager", None) is not None:
        _dispatch_toast("info", message, title=title, msec=msec, tab_key=tab_key, on_click=on_click)
        return
    _pending_toasts.append(("info", title, str(message), msec, tab_key, on_click))
    del _pending_toasts[:-10]


def toast_success(
    message: str,
    *,
    title: str = "Done",
    msec: Optional[int] = None,
    tab_key: Optional[str] = None,
    on_click=None,
) -> None:
    _log_toast("success", message, title=title, tab_key=tab_key)
    w = _get_window()
    if w is not None and getattr(w, "toast_manager", None) is not None:
        _dispatch_toast("success", message, title=title, msec=msec, tab_key=tab_key, on_click=on_click)
        return
    _pending_toasts.append(("success", title, str(message), msec, tab_key, on_click))
    del _pending_toasts[:-10]


def toast_warn(
    message: str,
    *,
    title: str = "Warning",
    msec: Optional[int] = None,
    tab_key: Optional[str] = None,
    on_click=None,
) -> None:
    _log_toast("warn", message, title=title, tab_key=tab_key)
    w = _get_window()
    if w is not None and getattr(w, "toast_manager", None) is not None:
        _dispatch_toast("warn", message, title=title, msec=msec, tab_key=tab_key, on_click=on_click)
        return
    _pending_toasts.append(("warn", title, str(message), msec, tab_key, on_click))
    del _pending_toasts[:-10]


def toast_error(
    message: str,
    *,
    title: str = "Error",
    msec: Optional[int] = None,
    tab_key: Optional[str] = None,
    on_click=None,
) -> None:
    _log_toast("error", message, title=title, tab_key=tab_key)
    w = _get_window()
    if w is not None and getattr(w, "toast_manager", None) is not None:
        _dispatch_toast("error", message, title=title, msec=msec, tab_key=tab_key, on_click=on_click)
        return
    _pending_toasts.append(("error", title, str(message), msec, tab_key, on_click))
    del _pending_toasts[:-10]
