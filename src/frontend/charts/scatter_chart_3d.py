
from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from PySide6.QtGui import QColor, QVector3D
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QVBoxLayout, QWidget
from PySide6.QtDataVisualization import (
    Q3DScatter,
    Q3DTheme,
    QScatter3DSeries,
    QScatterDataItem,
    QScatterDataProxy,
    QValue3DAxis,
)

from frontend.style.chart_theme import (
    make_colors_from_palette,
    is_dark_color,
    window_color_from_theme,
)
from frontend.style.theme_manager import theme_manager


class Scatter3DChart(QWidget):
    """Simple 3D scatter chart powered by Qt Data Visualization."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scatter = Q3DScatter()
        self._scatter.setShadowQuality(Q3DScatter.ShadowQuality.ShadowQualityNone)
        self._scatter.setReflection(False)
        self._scatter.setReflectivity(0.0)
        self._container = QWidget.createWindowContainer(self._scatter, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.addWidget(self._container, 1)

        self._proxy = QScatterDataProxy()
        self._series = QScatter3DSeries(self._proxy)
        self._series.setColorStyle(Q3DTheme.ColorStyle.ColorStyleUniform)
        self._series.setMesh(QScatter3DSeries.Mesh.MeshSphere)
        self._series.setMeshSmooth(True)
        self._series.setItemSize(0.08)
        self._scatter.addSeries(self._series)

        self._axis_x = QValue3DAxis()
        self._axis_y = QValue3DAxis()
        self._axis_z = QValue3DAxis()
        self._scatter.setAxisX(self._axis_x)
        self._scatter.setAxisY(self._axis_y)
        self._scatter.setAxisZ(self._axis_z)
        for axis in (self._axis_x, self._axis_y, self._axis_z):
            axis.setLabelFormat("%.2f")

        try:
            tm = theme_manager()
            tm.theme_changed.connect(self._on_theme_changed)
            self._apply_theme(tm.current_theme)
        except AssertionError:
            self._apply_theme(None)

    def set_axis_labels(self, labels: Sequence[str]) -> None:
        labels = list(labels)
        while len(labels) < 3:
            labels.append("Axis")
        self._axis_x.setTitle(labels[0])
        self._axis_y.setTitle(labels[1])
        self._axis_z.setTitle(labels[2])

    def set_points(
        self,
        frame: Optional[pd.DataFrame],
        columns: Sequence[str],
    ) -> None:
        self._proxy.resetArray([])
        if frame is None or frame.empty:
            return
        cols = [c for c in columns if c in frame.columns][:3]
        if len(cols) != 3:
            return
        data = frame.loc[:, cols].copy()
        for col in cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=cols)
        if data.empty:
            return

        xs = data[cols[0]].to_numpy(dtype=float, copy=False)
        ys = data[cols[1]].to_numpy(dtype=float, copy=False)
        zs = data[cols[2]].to_numpy(dtype=float, copy=False)

        finite_mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
        xs = xs[finite_mask]
        ys = ys[finite_mask]
        zs = zs[finite_mask]
        if xs.size == 0:
            return

        items = [QScatterDataItem(QVector3D(float(x), float(y), float(z))) for x, y, z in zip(xs, ys, zs)]
        self._proxy.resetArray(items)

        self._axis_x.setRange(float(np.nanmin(xs)), float(np.nanmax(xs)))
        self._axis_y.setRange(float(np.nanmin(ys)), float(np.nanmax(ys)))
        self._axis_z.setRange(float(np.nanmin(zs)), float(np.nanmax(zs)))

    def clear(self) -> None:
        self._proxy.resetArray([])

    def _on_theme_changed(self, theme_name: Optional[str]) -> None:
        self._apply_theme(theme_name)

    # @ai(gpt-5, codex, ui-theming, 2026-03-03)
    def _apply_theme(self, theme_name: Optional[str]) -> None:
        colors = make_colors_from_palette(self, theme_name)
        base_color = QColor(colors.line_primary)
        window_bg = window_color_from_theme(self, theme_name)
        dark_theme = is_dark_color(window_bg)
        theme = self._scatter.activeTheme()

        theme.setType(Q3DTheme.Theme.ThemeUserDefined)
        theme.setBackgroundEnabled(False)
        theme.setGridEnabled(True)
        theme.setColorStyle(Q3DTheme.ColorStyle.ColorStyleUniform)
        theme.setLabelBackgroundEnabled(False)
        theme.setLabelBorderEnabled(False)
        theme.setWindowColor(QColor(window_bg))
        theme.setBackgroundColor(QColor(colors.plot_bg))
        theme.setGridLineColor(QColor(colors.grid))
        theme.setLabelTextColor(QColor(colors.text))
        theme.setBaseColors([QColor(base_color)])
        theme.setLightColor(QColor(245, 245, 245))
        theme.setAmbientLightStrength(0.45 if dark_theme else 0.40)
        theme.setLightStrength(1.6 if dark_theme else 1.4)
        theme.setHighlightLightStrength(1.1 if dark_theme else 1.0)

        self._scatter.setReflection(False)
        self._scatter.setReflectivity(0.0)
        self._series.setColorStyle(Q3DTheme.ColorStyle.ColorStyleUniform)
        self._series.setBaseColor(base_color)
        self._series.setSingleHighlightColor(base_color.lighter(135 if dark_theme else 120))
        self._series.setMultiHighlightColor(base_color.lighter(160 if dark_theme else 140))
        self._axis_x.setTitleVisible(True)
        self._axis_y.setTitleVisible(True)
        self._axis_z.setTitleVisible(True)

        frame_palette = self.palette()
        frame_palette.setColor(QPalette.ColorRole.Window, QColor(window_bg))
        frame_palette.setColor(QPalette.ColorRole.Base, QColor(window_bg))
        self.setPalette(frame_palette)
        self.setAutoFillBackground(True)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        except Exception:
            pass

        container_palette = self._container.palette()
        container_palette.setColor(QPalette.ColorRole.Window, QColor(window_bg))
        container_palette.setColor(QPalette.ColorRole.Base, QColor(window_bg))
        self._container.setPalette(container_palette)
        self._container.setAutoFillBackground(True)


__all__ = ["Scatter3DChart"]
