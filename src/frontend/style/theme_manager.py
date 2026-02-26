# frontend/style/theme_manager.py
from __future__ import annotations
from contextlib import contextmanager
import logging
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QWidget
from typing import Optional
from .styles import apply_theme

logger = logging.getLogger(__name__)

class ThemeManager(QObject):
    theme_changed = Signal(str)  # emits "dark" | "light"

    def __init__(self, app: QApplication):
        super().__init__()
        self._app = app
        self._current: Optional[str] = None
        self._updating = False

    @contextmanager
    def _suspend_updates(self):
        widgets = [w for w in self._app.topLevelWidgets() if isinstance(w, QWidget)]
        previous_states = []
        safe_widgets: list[QWidget] = []
        for widget in widgets:
            try:
                previous_states.append(widget.updatesEnabled())
                safe_widgets.append(widget)
            except Exception:
                previous_states.append(True)
                logger.warning("Failed to read widget update state during theme switch", exc_info=True)
        widgets = safe_widgets
        for widget in widgets:
            try:
                widget.setUpdatesEnabled(False)
            except Exception:
                logger.warning("Failed to suspend widget updates during theme switch", exc_info=True)
        try:
            yield
        finally:
            for widget, enabled in zip(widgets, previous_states):
                try:
                    widget.setUpdatesEnabled(enabled)
                    if enabled:
                        widget.update()
                except Exception:
                    logger.warning("Failed to restore widget updates during theme switch", exc_info=True)

    def set_theme(self, theme: str) -> None:
        if theme == self._current or self._updating:
            return  # no-op, prevents duplicate signals
        self._updating = True
        try:
            try:
                with self._suspend_updates():
                    apply_theme(self._app, theme)
            except Exception:
                # Fallback: apply directly without update suppression.
                logger.exception("Theme apply with suspended updates failed, retrying without suspension")
                apply_theme(self._app, theme)
            self._current = theme
            self.theme_changed.emit(theme)
        finally:
            self._updating = False

    @property
    def current_theme(self) -> Optional[str]:
        """Return the last theme that was applied (``None`` until the first call)."""
        return self._current

# simple module-level accessor
_theme_manager: Optional[ThemeManager] = None

def init_theme_manager(app: QApplication) -> ThemeManager:
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager(app)
    return _theme_manager

def theme_manager() -> ThemeManager:
    assert _theme_manager is not None, "Call init_theme_manager(qApp) at startup."
    return _theme_manager
