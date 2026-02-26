
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QBrush
from PySide6.QtCharts import QLineSeries, QScatterSeries, QValueAxis
from ..localization import tr

from frontend.style.chart_theme import (

    is_dark_color,
    style_axis,
    style_line_series,
)

import logging
logger = logging.getLogger(__name__)

from .base_chart import BaseChart



class ScatterChart(BaseChart):
    """Scatter chart comparing two columns, optionally grouped by dataset."""

    _BASE_RGB = [
        (66, 133, 244),   # blue
        (219, 68, 55),    # red
        (244, 180, 0),    # orange
        (15, 157, 88),    # green
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(tr("Scatter"), parent=parent)
        self.chart.legend().setVisible(True)

        self.identity = QLineSeries()
        self.identity.setUseOpenGL(False)
        self.chart.addSeries(self.identity)
        self._hide_identity_from_legend()

        self.axis_x = QValueAxis()
        self.axis_x.setLabelFormat("%.2f")
        self.axis_x.setTitleText(tr("X"))
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.identity.attachAxis(self.axis_x)

        self.axis_y = QValueAxis()
        self.axis_y.setLabelFormat("%.2f")
        self.axis_y.setTitleText(tr("Y"))
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.identity.attachAxis(self.axis_y)

        self._series_map: dict[str, QScatterSeries] = {}
        self._dataset_order: list[str] = []
        self._current_colors = None
        self._dark_theme = False

        self.apply_current_theme()

    def set_axis_labels(self, x_label: str, y_label: str) -> None:
        self.axis_x.setTitleText(x_label or tr("X"))
        self.axis_y.setTitleText(y_label or tr("Y"))

    def _apply_colors(self, colors):
        style_axis(self.axis_x, colors)
        style_axis(self.axis_y, colors)
        style_line_series(self.identity, colors)
        try:
            self.chart.setTitleBrush(QBrush(colors.text))
        except Exception:
            logger.warning("Exception in _apply_colors", exc_info=True)
        self._current_colors = colors
        self._dark_theme = is_dark_color(colors.plot_bg)
        self._update_dataset_series_colors()

    def _update_dataset_series_colors(self) -> None:
        if self._current_colors is None:
            return
        for idx, label in enumerate(self._dataset_order):
            series = self._series_map.get(label)
            if series is None:
                continue
            color = self._color_for_index(idx)
            pen = series.pen()
            try:
                pen.setColor(color)
                pen.setWidthF(1.2)
                pen.setCosmetic(True)
                series.setPen(pen)
            except Exception:
                logger.warning("Exception in _update_dataset_series_colors", exc_info=True)
            brush = series.brush()
            try:
                brush.setColor(color)
                series.setBrush(brush)
            except Exception:
                logger.warning("Exception in _update_dataset_series_colors", exc_info=True)

    def _color_for_index(self, index: int) -> QColor:
        rgb = self._BASE_RGB[index % len(self._BASE_RGB)]
        color = self._build_color(rgb)
        if self._dark_theme:
            try:
                return color.lighter(140)
            except Exception:
                return color
        return color

    @staticmethod
    def _build_color(rgb: tuple[int, int, int]) -> QColor:
        try:
            return QColor(*rgb)
        except Exception:
            return QColor()

    def _reset_series(self) -> None:
        for series in self._series_map.values():
            try:
                self.chart.removeSeries(series)
            except Exception:
                logger.warning("Exception in _reset_series", exc_info=True)
        self._series_map.clear()
        self._dataset_order.clear()

    def clear(self) -> None:
        self._reset_series()
        self.identity.clear()

    def set_points(
        self,
        frame: Optional[pd.DataFrame],
        *,
        x_column: str = "actual",
        y_column: str = "predicted",
        group_column: Optional[str] = "dataset",
        show_identity_line: bool = True,
        force_equal_axes: bool = False,
        show_legend: bool = True,
    ) -> None:
        self._reset_series()
        self.identity.clear()
        try:
            self.chart.legend().setVisible(bool(show_legend))
        except Exception:
            logger.warning("Exception in set_points", exc_info=True)
        try:
            self.identity.setVisible(bool(show_identity_line))
        except Exception:
            logger.warning("Exception in set_points", exc_info=True)
        if frame is None or frame.empty:
            return

        df = frame.copy()
        if x_column not in df.columns or y_column not in df.columns:
            return

        df[x_column] = pd.to_numeric(df[x_column], errors="coerce")
        df[y_column] = pd.to_numeric(df[y_column], errors="coerce")
        df = df.dropna(subset=[x_column, y_column])
        if df.empty:
            return

        if group_column and group_column in df.columns:
            df[group_column] = df[group_column].fillna("Data").astype(str)
        else:
            group_column = "__dataset__"
            df[group_column] = "Data"

        dataset_labels = list(pd.unique(df[group_column]))
        self._dataset_order = []

        max_points = 5000
        xs_all: list[float] = []
        ys_all: list[float] = []

        for label in dataset_labels:
            subset = df[df[group_column] == label]
            xs = subset[x_column].to_numpy(dtype=float, copy=False)
            ys = subset[y_column].to_numpy(dtype=float, copy=False)

            finite_mask = np.isfinite(xs) & np.isfinite(ys)
            xs = xs[finite_mask]
            ys = ys[finite_mask]
            if xs.size == 0:
                continue

            if xs.size > max_points:
                indices = np.linspace(0, xs.size - 1, max_points, dtype=int)
                xs = xs[indices]
                ys = ys[indices]

            xs_all.extend(xs.tolist())
            ys_all.extend(ys.tolist())

            series = QScatterSeries()
            series.setMarkerSize(8.0)
            series.setUseOpenGL(False)
            series.setName(label)
            self.chart.addSeries(series)
            series.attachAxis(self.axis_x)
            series.attachAxis(self.axis_y)
            points = [QPointF(float(x), float(y)) for x, y in zip(xs, ys)]
            series.replace(points)
            self._series_map[label] = series
            self._dataset_order.append(label)

        self._update_dataset_series_colors()

        if not xs_all or not ys_all:
            return

        xmin = float(np.nanmin(xs_all))
        xmax = float(np.nanmax(xs_all))
        ymin = float(np.nanmin(ys_all))
        ymax = float(np.nanmax(ys_all))
        lo = min(xmin, ymin)
        hi = max(xmax, ymax)
        if lo == hi:
            hi = lo + 1.0

        if show_identity_line:
            self.identity.clear()
            self.identity.append(lo, lo)
            self.identity.append(hi, hi)
        else:
            self.identity.clear()

        if force_equal_axes:
            self.axis_x.setRange(lo, hi)
            self.axis_y.setRange(lo, hi)
            return

        x_lo = xmin
        x_hi = xmax
        y_lo = ymin
        y_hi = ymax
        if x_lo == x_hi:
            x_hi = x_lo + 1.0
        if y_lo == y_hi:
            y_hi = y_lo + 1.0
        self.axis_x.setRange(x_lo, x_hi)
        self.axis_y.setRange(y_lo, y_hi)

    def _hide_identity_from_legend(self) -> None:
        legend = self.chart.legend()
        if legend is None:
            return
        markers = legend.markers(self.identity)
        for marker in markers:
            marker.setVisible(False)


__all__ = ["ScatterChart"]
