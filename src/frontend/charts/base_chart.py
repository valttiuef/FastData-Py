
from __future__ import annotations
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QFrame, QVBoxLayout
from PySide6.QtCharts import QChart, QChartView

from frontend.style.chart_theme import (
    apply_chart_background,
    make_colors_from_palette,
)
from frontend.style.theme_manager import theme_manager


class BaseChart(QFrame):
    """Common chart frame that listens for theme changes."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.chart = QChart()
        self.chart.setTitle(title)
        self.view = QChartView(self.chart, self)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout = QVBoxLayout(self)
        layout.addWidget(self.view)

        self._theme_ready = False
        try:
            tm = theme_manager()
            self._current_theme = tm.current_theme
            tm.theme_changed.connect(self._apply_theme)
        except AssertionError:
            self._current_theme = None

    def apply_current_theme(self) -> None:
        """Apply the stored theme once the subclass has finished init."""
        self._theme_ready = True
        self._apply_theme(self._current_theme)

    def _apply_theme(self, theme_name: Optional[str]) -> None:
        self._current_theme = theme_name
        if not self._theme_ready:
            return
        colors = make_colors_from_palette(self, theme_name)
        apply_chart_background(self.chart, colors)
        self._apply_colors(colors)

    def _apply_colors(self, colors):
        raise NotImplementedError


__all__ = ["BaseChart"]
