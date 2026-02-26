from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
# frontend/style/chart_theme.py
from dataclasses import dataclass
from typing import Optional

from PySide6.QtGui import QColor, QPen, QBrush, QPalette
from PySide6.QtCharts import (
    QChart,
    QValueAxis,
    QDateTimeAxis,
    QBarCategoryAxis,
    QLineSeries,
    QBarSeries,
    QLegend,
)
from PySide6.QtCore import Qt

from frontend.style.styles import THEMES


def _is_dark(c: QColor) -> bool:
    # perceptual luminance
    l = 0.2126 * c.redF() + 0.7152 * c.greenF() + 0.0722 * c.blueF()
    return l < 0.5


def is_dark_color(color: QColor) -> bool:
    """Public helper so widgets can branch on dark/light chart backgrounds."""
    return _is_dark(color)

@dataclass
class ChartColors:
    plot_bg: QColor
    text: QColor
    grid: QColor
    axis_line: QColor
    line_primary: QColor
    bar_fill: QColor
    bar_border: QColor


def _from_token(value: str) -> QColor:
    return QColor(value)


def _choose_line_primary(theme_name: Optional[str], dark_background: bool) -> QColor:
    if theme_name == "dark":
        return QColor(91, 155, 255)
    if theme_name == "light":
        return QColor(66, 133, 244)
    return QColor(66, 133, 244) if not dark_background else QColor(91, 155, 255)


def make_colors_from_palette(widget, theme_name: Optional[str] = None) -> ChartColors:
    pal: QPalette = widget.palette()

    if theme_name and theme_name in THEMES:
        tokens = THEMES[theme_name]
        base = _from_token(tokens["IN_BG"])
        txt = _from_token(tokens["TXT_1"])
        axis = _from_token(tokens["BRD_1"])
        grid_base = _from_token(tokens["BRD_2"])
    else:
        base = pal.base().color()
        txt = pal.text().color()
        axis = pal.mid().color()
        grid_base = pal.mid().color()

    grid = QColor(grid_base)
    grid.setAlpha(110)

    dark_background = _is_dark(base)
    line_primary = _choose_line_primary(theme_name, dark_background)
    bar_fill = QColor(line_primary)
    bar_border = QColor(bar_fill).darker(120 if not dark_background else 130)

    return ChartColors(
        plot_bg=base,
        text=txt,
        grid=grid,
        axis_line=axis,
        line_primary=line_primary,
        bar_fill=bar_fill,
        bar_border=bar_border,
    )


def window_color_from_theme(widget, theme_name: Optional[str] = None) -> QColor:
    """Return the palette window color for *theme_name* or fall back to the widget."""

    if theme_name and theme_name in THEMES:
        tokens = THEMES[theme_name]
        return _from_token(tokens["BG_1"])

    return widget.palette().color(QPalette.Window)

def apply_chart_background(chart: QChart, colors: ChartColors):
    """Apply palette-derived background styling with rounded panels."""

    panel_color = QColor(colors.plot_bg)
    if _is_dark(panel_color):
        panel_color = panel_color.lighter(115)
    else:
        panel_color = panel_color.darker(103)

    chart.setBackgroundVisible(True)
    chart.setBackgroundBrush(QBrush(panel_color))

    border_color = QColor(colors.axis_line)
    border_color.setAlpha(160)
    border_pen = QPen(border_color, 1.0)
    border_pen.setCosmetic(True)
    chart.setBackgroundPen(border_pen)
    chart.setBackgroundRoundness(12.0)

    chart.setPlotAreaBackgroundVisible(True)
    chart.setPlotAreaBackgroundBrush(QBrush(colors.plot_bg))
    plot_pen = QPen(Qt.PenStyle.NoPen)
    chart.setPlotAreaBackgroundPen(plot_pen)

def style_axis(ax, colors: ChartColors):
    # Supports QValueAxis / QDateTimeAxis / QBarCategoryAxis
    ax.setLabelsBrush(QBrush(colors.text))
    try:
        ax.setTitleBrush(QBrush(colors.text))
    except Exception:
        logger.warning("Exception in style_axis", exc_info=True)
    line_pen = QPen(colors.axis_line, 1.0); line_pen.setCosmetic(True)
    try:
        ax.setLinePen(line_pen)
    except Exception:
        logger.warning("Exception in style_axis", exc_info=True)
    try:
        ax.setGridLineColor(colors.grid)
        ax.setMinorGridLineColor(colors.grid.darker(110))
    except Exception:
        logger.warning("Exception in style_axis", exc_info=True)

def style_line_series(series: QLineSeries, colors: ChartColors, width: float = 2.0):
    pen = QPen(colors.line_primary, width); pen.setCosmetic(True)
    series.setPen(pen)

def style_bar_series(series: QBarSeries, colors: ChartColors):
    for s in series.barSets():
        s.setBrush(QBrush(colors.bar_fill))
        s.setPen(QPen(colors.bar_border))


def style_legend(legend: QLegend | None, colors: ChartColors) -> None:
    """Keep legend label colours in sync with the surrounding palette."""

    if legend is None:
        return

    try:
        legend.setLabelColor(colors.text)
    except Exception:
        logger.warning("Exception in style_legend", exc_info=True)

    try:
        label_brush = QBrush(colors.text)
        for marker in list(legend.markers()):
            try:
                marker.setLabelBrush(label_brush)
            except Exception:
                continue
    except Exception:
        logger.warning("Exception in style_legend", exc_info=True)
