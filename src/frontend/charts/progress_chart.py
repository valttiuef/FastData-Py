
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QPointF
from PySide6.QtCharts import QLineSeries, QValueAxis

from frontend.style.chart_theme import style_axis, style_line_series

from .base_chart import BaseChart


class ProgressChart(BaseChart):
    """Simple line chart showing R² progress over runs."""

    def __init__(self, parent=None) -> None:
        super().__init__("Training progress", parent=parent)
        self.series = QLineSeries()
        self.chart.addSeries(self.series)
        self.axis_x = QValueAxis()
        self.axis_x.setLabelFormat("%d")
        self.axis_x.setTitleText("Run")
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.series.attachAxis(self.axis_x)

        self.axis_y = QValueAxis()
        self.axis_y.setLabelFormat("%.2f")
        self.axis_y.setTitleText("R² score")
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        self.series.attachAxis(self.axis_y)

        self.apply_current_theme()

    def _apply_colors(self, colors):
        style_axis(self.axis_x, colors)
        style_axis(self.axis_y, colors)
        style_line_series(self.series, colors)

    def clear(self) -> None:
        self.series.clear()

    def set_dataframe(self, frame: Optional[pd.DataFrame]) -> None:
        self.series.clear()
        if frame is None or frame.empty:
            return

        df = frame.copy()
        if "r2" not in df.columns:
            return

        r2 = pd.to_numeric(df["r2"], errors="coerce")
        if "step" in df.columns:
            steps = pd.to_numeric(df["step"], errors="coerce")
        else:
            steps = pd.Series(np.arange(1, len(df) + 1), index=df.index)

        mask = r2.notna() & steps.notna()
        if not mask.any():
            return

        xs = steps[mask].to_numpy(dtype=float, copy=False)
        ys = r2[mask].to_numpy(dtype=float, copy=False)

        finite_mask = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[finite_mask]
        ys = ys[finite_mask]
        if xs.size == 0:
            return

        points = [QPointF(float(x), float(y)) for x, y in zip(xs, ys)]
        self.series.replace(points)

        xmin = float(np.nanmin(xs))
        xmax = float(np.nanmax(xs))
        ymin = float(np.nanmin(ys))
        ymax = float(np.nanmax(ys))

        if xmin == xmax:
            xmax += 1.0
        self.axis_x.setRange(xmin, xmax)

        if ymin == ymax:
            ymax += 0.01
        self.axis_y.setRange(min(ymin, 0.0), max(ymax, 1.0))


__all__ = ["ProgressChart"]
