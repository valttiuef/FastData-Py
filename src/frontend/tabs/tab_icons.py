
from __future__ import annotations
from functools import lru_cache

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap

from core.paths import get_resource_path

_TAB_ICON_GLYPHS: dict[str, str] = {
    "data": "📁",
    "selections": "🎯",
    "statistics": "📒",
    "charts": "📊",
    "som": "🧠",
    "regression": "📉",
    "forecasting": "📈",
}

_TAB_ICON_NAME_CANDIDATES: dict[str, tuple[str, ...]] = {
    "data": ("data",),
    "selections": ("selections",),
    "statistics": ("statistics",),
    "charts": ("charts",),
    "som": ("som",),
    "regression": ("regression",),
    "forecasting": ("forecasting",),
}


def resolve_tab_icon(*, tab_key: str | None = None, help_key: str | None = None) -> QIcon:
    key = _normalize_tab_key(tab_key=tab_key, help_key=help_key)
    return _resolve_tab_icon_cached(key)


def _normalize_tab_key(*, tab_key: str | None, help_key: str | None) -> str:
    if tab_key:
        return str(tab_key).strip().lower()
    if not help_key:
        return ""
    key = str(help_key).strip().lower()
    if key.startswith("tab."):
        return key[4:]
    return key


@lru_cache(maxsize=32)
def _resolve_tab_icon_cached(tab_key: str) -> QIcon:
    for name in _TAB_ICON_NAME_CANDIDATES.get(tab_key, (tab_key,)):
        path = get_resource_path(f"images/tabs/{name}.svg")
        if path.exists():
            return QIcon(str(path))
    return _glyph_icon(_TAB_ICON_GLYPHS.get(tab_key, "◻"))


@lru_cache(maxsize=32)
def _glyph_icon(glyph: str) -> QIcon:
    pixmap = QPixmap(18, 18)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.setPen(QColor("#6a6a6a"))
    font = QFont("Segoe UI Emoji", 12)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), int(Qt.AlignmentFlag.AlignCenter), glyph)
    painter.end()

    return QIcon(pixmap)
