from __future__ import annotations

import logging
from typing import Callable, List, Optional

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ..localization import tr

logger = logging.getLogger(__name__)


def _is_dark(color: QColor) -> bool:
    l = 0.2126 * color.redF() + 0.7152 * color.greenF() + 0.0722 * color.blueF()
    return l < 0.5


class _Toast(QWidget):
    # Outer transparent margins reserved so the drop shadow is not clipped.
    SHADOW_MARGIN_LEFT = 10
    SHADOW_MARGIN_TOP = 8
    SHADOW_MARGIN_RIGHT = 10
    SHADOW_MARGIN_BOTTOM = 0

    def __init__(
        self,
        title: str,
        message: str,
        msec: int,
        kind: str,
        *,
        parent: Optional[QWidget] = None,
        max_width: Optional[int] = None,
        tab_key: Optional[str] = None,
        on_click: Optional[Callable[[], None]] = None,
        icon_key: Optional[str] = None,
    ):
        super().__init__(parent)
        self._on_click = on_click
        self.tab_key = tab_key

        # Frameless, non-blocking, stays on top of your app
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAutoFillBackground(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        if self._on_click is not None:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

        bg = QWidget(self)
        bg.setObjectName("toastCard")

        accent = QWidget(bg)
        accent.setObjectName("toastAccent")
        accent.setProperty("toastKind", str(kind or "info"))
        accent.setFixedWidth(6)

        content = QWidget(bg)
        content.setObjectName("toastContent")
        content_lay = QHBoxLayout(content)
        content_lay.setContentsMargins(18, 16, 18, 16)
        content_lay.setSpacing(10)

        icon_lbl: Optional[QLabel] = None
        if icon_key:
            icon_lbl = QLabel(content)
            icon_lbl.setObjectName("toastIcon")
            icon_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
            icon_lbl.setFixedWidth(20)
            icon = self._resolve_icon(icon_key)
            if icon is not None:
                icon_lbl.setPixmap(icon.pixmap(16, 16))
            else:
                icon_lbl.setText(">")
            content_lay.addWidget(icon_lbl, 0)

        text_wrap = QWidget(content)
        text_wrap.setObjectName("toastTextWrap")
        text_lay = QVBoxLayout(text_wrap)
        text_lay.setContentsMargins(0, 0, 0, 0)
        text_lay.setSpacing(4)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("toastTitle")
        title_font = QFont(title_lbl.font())
        title_font.setWeight(QFont.Weight.DemiBold)
        title_lbl.setFont(title_font)

        msg_lbl = QLabel(message)
        msg_lbl.setObjectName("toastMsg")
        msg_lbl.setWordWrap(True)

        text_lay.addWidget(title_lbl)
        text_lay.addWidget(msg_lbl)
        content_lay.addWidget(text_wrap, 1)

        lay = QHBoxLayout(bg)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(accent)
        lay.addWidget(content)

        root = QHBoxLayout(self)
        root.setContentsMargins(
            self.SHADOW_MARGIN_LEFT,
            self.SHADOW_MARGIN_TOP,
            self.SHADOW_MARGIN_RIGHT,
            self.SHADOW_MARGIN_BOTTOM,
        )
        root.addWidget(bg)

        pal = self.palette()
        bg_color = pal.color(QPalette.Window)
        dark = _is_dark(bg_color)

        shadow = QGraphicsDropShadowEffect(bg)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 6)
        shadow_color = QColor(0, 0, 0, 145 if dark else 92)
        shadow.setColor(shadow_color)
        bg.setGraphicsEffect(shadow)

        if max_width is not None:
            safe_width = max(220, int(max_width))
            bg.setMaximumWidth(safe_width)
            msg_lbl.setMaximumWidth(max(160, safe_width - 60))

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(180)
        self._fade.setEasingCurve(QEasingCurve.OutCubic)
        self._closing = False
        self._fade.finished.connect(self._on_fade_finished)

        self.setWindowOpacity(0.0)
        self.adjustSize()

        # Auto close
        QTimer.singleShot(msec, self.close_animated)

    def visual_top_margin(self) -> int:
        return self.SHADOW_MARGIN_TOP

    def visual_bottom_margin(self) -> int:
        return self.SHADOW_MARGIN_BOTTOM

    def visual_height(self) -> int:
        return max(0, self.height() - self.visual_top_margin() - self.visual_bottom_margin())

    def _resolve_icon(self, icon_key: str):
        style = self.style()
        if style is None:
            return None

        key = str(icon_key or "").strip().lower()
        icon_map = {
            "open_file": QStyle.StandardPixmap.SP_DialogOpenButton,
        }
        pixmap_key = icon_map.get(key)
        if pixmap_key is None:
            return None

        try:
            return style.standardIcon(pixmap_key)
        except Exception:
            logger.warning("Exception in _resolve_icon", exc_info=True)
            return None

    def mousePressEvent(self, event):  # type: ignore[override]
        if self._on_click is not None:
            try:
                self._on_click()
            except Exception:
                logger.warning("Exception in mousePressEvent", exc_info=True)

        self.close_animated()
        event.accept()

    def show_animated(self):
        self._fade.stop()
        self._closing = False
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self.show()
        self._fade.start()

    def close_animated(self):
        self._fade.stop()
        self._closing = True
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(0.0)
        self._fade.start()

    def _on_fade_finished(self) -> None:
        if self._closing and self.windowOpacity() <= 0.01:
            self.close()


class ToastManager:
    """VSCode-like bottom-right notifications inside your app."""

    def __init__(self, anchor: Optional[QWidget] = None):
        # anchor: main window (recommended). If None, uses primary screen geometry.
        self.anchor = anchor
        self._toasts: List[_Toast] = []
        self.max_toasts = 5

        # layout constants
        self.margin_right = 26
        self.margin_bottom = 22

        # Visible gap between cards. Set to 0 so cards sit directly on top of each other.
        self.gap = 6

        self.default_msec_by_kind = {
            "info": 3600,
            "success": 3600,
            "warn": 4200,
            "error": 5200,
        }

    def info(
        self,
        message: str,
        title: Optional[str] = None,
        msec: Optional[int] = None,
        *,
        tab_key: Optional[str] = None,
        on_click: Optional[Callable[[], None]] = None,
        icon_key: Optional[str] = None,
    ):
        title = title or tr("Info")
        self._show(
            title,
            message,
            msec or self.default_msec_by_kind["info"],
            "info",
            tab_key=tab_key,
            on_click=on_click,
            icon_key=icon_key,
        )

    def success(
        self,
        message: str,
        title: Optional[str] = None,
        msec: Optional[int] = None,
        *,
        tab_key: Optional[str] = None,
        on_click: Optional[Callable[[], None]] = None,
        icon_key: Optional[str] = None,
    ):
        title = title or tr("Done")
        self._show(
            title,
            message,
            msec or self.default_msec_by_kind["success"],
            "success",
            tab_key=tab_key,
            on_click=on_click,
            icon_key=icon_key,
        )

    def warn(
        self,
        message: str,
        title: Optional[str] = None,
        msec: Optional[int] = None,
        *,
        tab_key: Optional[str] = None,
        on_click: Optional[Callable[[], None]] = None,
        icon_key: Optional[str] = None,
    ):
        title = title or tr("Warning")
        self._show(
            title,
            message,
            msec or self.default_msec_by_kind["warn"],
            "warn",
            tab_key=tab_key,
            on_click=on_click,
            icon_key=icon_key,
        )

    def error(
        self,
        message: str,
        title: Optional[str] = None,
        msec: Optional[int] = None,
        *,
        tab_key: Optional[str] = None,
        on_click: Optional[Callable[[], None]] = None,
        icon_key: Optional[str] = None,
    ):
        title = title or tr("Error")
        self._show(
            title,
            message,
            msec or self.default_msec_by_kind["error"],
            "error",
            tab_key=tab_key,
            on_click=on_click,
            icon_key=icon_key,
        )

    def _show(
        self,
        title: str,
        message: str,
        msec: int,
        kind: str,
        *,
        tab_key: Optional[str] = None,
        on_click: Optional[Callable[[], None]] = None,
        icon_key: Optional[str] = None,
    ):
        if tab_key:
            self._close_tab_toasts(tab_key)

        self._trim_toasts(max(1, int(self.max_toasts)) - 1)

        geo = self._anchor_geometry()
        max_width = min(460, int(geo.width() * 0.38))

        if tab_key:
            activate = lambda: self._activate_tab(tab_key)
            if on_click is None:
                on_click = activate
            else:
                base_click = on_click

                def combined_click():
                    activate()
                    base_click()

                on_click = combined_click

        toast = _Toast(
            title,
            message,
            msec,
            kind,
            parent=self.anchor,
            max_width=max_width,
            tab_key=tab_key,
            on_click=on_click,
            icon_key=icon_key,
        )
        toast.destroyed.connect(lambda *_: self._remove_toast(toast))

        self._toasts.append(toast)
        self._reposition()
        toast.show_animated()

    def _close_tab_toasts(self, tab_key: str) -> None:
        remaining: List[_Toast] = []
        for toast in self._toasts:
            try:
                if getattr(toast, "tab_key", None) == tab_key:
                    toast.close_animated()
                else:
                    remaining.append(toast)
            except RuntimeError:
                continue

        self._toasts = remaining
        self._reposition()

    def _trim_toasts(self, keep: int) -> None:
        if keep < 0:
            keep = 0
        if len(self._toasts) <= keep:
            return

        excess = len(self._toasts) - keep
        for toast in list(self._toasts[:excess]):
            try:
                toast.close_animated()
            except Exception:
                logger.warning("Exception in _trim_toasts", exc_info=True)

        self._toasts = self._toasts[excess:]

    def _activate_tab(self, tab_key: str) -> None:
        if not tab_key:
            return

        anchor = self.anchor
        if anchor is None:
            return

        handler = getattr(anchor, "activate_tab", None)
        if handler is None:
            return

        try:
            handler(tab_key)
        except Exception:
            logger.warning("Exception in _activate_tab", exc_info=True)

    def _remove_toast(self, toast: _Toast) -> None:
        try:
            self._toasts = [t for t in self._toasts if t is not toast]
        finally:
            self._cleanup()

    def _cleanup(self):
        alive: List[_Toast] = []
        for toast in self._toasts:
            try:
                if toast is not None and toast.isVisible():
                    alive.append(toast)
            except RuntimeError:
                # Underlying C++ object already deleted (best-effort cleanup)
                continue

        self._toasts = alive
        self._reposition()

    def _anchor_geometry(self):
        if self.anchor is not None and self.anchor.windowHandle() is not None:
            return self.anchor.frameGeometry()
        return QApplication.primaryScreen().availableGeometry()

    def _reposition(self):
        # Determine bottom-right corner based on anchor window or screen
        geo = self._anchor_geometry()

        x_right = geo.x() + geo.width() - self.margin_right
        y_bottom = geo.y() + geo.height() - self.margin_bottom

        y = y_bottom

        # Stack upwards using the visible card height, not the full widget height,
        # because the widget includes transparent shadow margins.
        for toast in reversed(self._toasts):
            try:
                toast.adjustSize()

                w, h = toast.width(), toast.height()
                top_margin = toast.visual_top_margin()
                visual_height = toast.visual_height()

                top_y = y - visual_height - top_margin
                toast.move(QPoint(x_right - w, top_y))

                y -= visual_height + self.gap
            except RuntimeError:
                continue