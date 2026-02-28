from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
# frontend/charts/group_chart.py
"""Group bar chart for displaying aggregated statistics by category/group."""

from typing import List, Optional

import pandas as pd

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QCursor, QPalette
from PySide6.QtCharts import (
    QChart, QChartView, QBarSeries, QBarSet, QBarCategoryAxis, QValueAxis
)
from PySide6.QtWidgets import (

    QFrame,
    QVBoxLayout,
    QToolTip,
)
from ..localization import tr

from ..style.chart_theme import (
    make_colors_from_palette,
    apply_chart_background,
    style_axis,
    style_legend,
    is_dark_color,
    window_color_from_theme,
)
from ..style.theme_manager import theme_manager
from ..style.group_colors import group_color_cycle



def _short_label(name: str, max_len: int = 24) -> str:
    """Truncate a label to max_len characters."""
    if not name:
        return ""
    try:
        primary = str(name).split(" · ")[0]
    except Exception:
        primary = str(name)
    if len(primary) <= max_len:
        return primary
    return primary[: max_len - 1] + "…"


class _ChartView(QChartView):
    """Chart view with disabled zooming."""
    
    def __init__(self, chart: QChart, parent=None):
        super().__init__(chart, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRubberBand(QChartView.NoRubberBand)

    def wheelEvent(self, ev):
        ev.accept()


class GroupBarChart(QFrame):
    """Simple bar chart for displaying values by group/category.
    
    This chart displays bars for categorical data where each bar
    represents a group (e.g., Datasets, custom categories).
    
    Unlike MonthlyBarChart, this does not support time-based drill-down
    or aggregation - it simply displays the provided categories and values.
    
    Signals:
        bar_clicked(int): Emitted when a bar is clicked with the bar index
        category_selected(str, float): Emitted when a bar is clicked with
            the category name and value
    """
    
    bar_clicked = Signal(int)
    category_selected = Signal(str, object)  # (category_name, value)

    def __init__(
        self,
        title: str = "Group Chart",
        parent=None,
        y_label: str = "Value",
    ):
        super().__init__(parent)
        if title == "Group Chart":
            title = tr("Group Chart")
        if y_label == "Value":
            y_label = tr("Value")
        lay = QVBoxLayout(self)
        self._title = title
        self._y_label = y_label

        # Chart + bar series
        self.chart = QChart()
        self.chart.legend().hide()
        self.chart.setTitle(title)
        self.series = QBarSeries()
        self.series.setBarWidth(0.6)
        self.chart.addSeries(self.series)

        # Axes
        self.axis_x = QBarCategoryAxis()
        self.axis_y = QValueAxis()
        self.axis_y.setLabelFormat("%.3f")
        self.axis_x.setTitleText("")
        self.axis_y.setTitleText(y_label)
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        self.series.attachAxis(self.axis_x)
        self.series.attachAxis(self.axis_y)

        # View
        self.view = _ChartView(self.chart)
        lay.addWidget(self.view, 1)
        self._native_mouse_press_event = self.view.mousePressEvent
        self._native_mouse_double_click_event = self.view.mouseDoubleClickEvent
        try:
            self.view.setMouseTracking(True)
            self.view.setAttribute(Qt.WA_Hover, True)
        except Exception:
            logger.warning("Failed to enable hover tracking for group chart view.", exc_info=True)

        # State
        self._categories: List[str] = []
        self._values: List[float] = []
        self._hover_idx: Optional[int] = None
        self._connected_barsets = set()
        self._single_series_name: str = tr("Value")
        self._tooltip_overrides: dict[tuple[str, str], str] = {}

        # Theme
        self._current_theme: Optional[str] = None
        self._dark_theme: bool = False
        self._chart_colors = None

        try:
            self._current_theme = theme_manager().current_theme
        except AssertionError:
            self._current_theme = None

        self._apply_theme(self._current_theme)

        try:
            theme_manager().theme_changed.connect(self._on_theme_changed)
        except AssertionError:
            logger.warning("Failed to connect theme change signal for group chart.", exc_info=True)

        # Connect hover/click signals
        self._connect_signals()

    def _on_theme_changed(self, theme_name: str):
        self._current_theme = theme_name
        self._apply_theme(theme_name)

    def _apply_theme(self, theme_name: Optional[str] = None):
        if theme_name is None:
            theme_name = self._current_theme
        else:
            self._current_theme = theme_name

        colors = make_colors_from_palette(self, theme_name)
        self._chart_colors = colors
        self._dark_theme = is_dark_color(colors.plot_bg)
        apply_chart_background(self.chart, colors)
        style_axis(self.axis_x, colors)
        style_axis(self.axis_y, colors)
        style_legend(self.chart.legend(), colors)

        container_color = window_color_from_theme(self, theme_name)

        try:
            self.chart.setTitleBrush(QBrush(colors.text))
        except Exception:
            logger.warning("Failed to update title brush while applying group chart theme.", exc_info=True)

        frame_palette = self.palette()
        frame_palette.setColor(QPalette.Window, container_color)
        frame_palette.setColor(QPalette.Base, container_color)
        self.setPalette(frame_palette)
        self.setAutoFillBackground(True)
        try:
            self.setAttribute(Qt.WA_StyledBackground, True)
        except Exception:
            logger.warning("Failed to set styled-background attribute while applying group chart theme.", exc_info=True)

        view_palette = self.view.palette()
        view_palette.setColor(QPalette.Window, container_color)
        view_palette.setColor(QPalette.Base, container_color)
        self.view.setPalette(view_palette)
        self.view.setAutoFillBackground(True)
        try:
            self.view.setBackgroundBrush(QBrush(container_color))
        except Exception:
            logger.warning("Failed to set view background brush while applying group chart theme.", exc_info=True)

        viewport_palette = self.view.viewport().palette()
        viewport_palette.setColor(QPalette.Window, container_color)
        viewport_palette.setColor(QPalette.Base, container_color)
        self.view.viewport().setPalette(viewport_palette)
        self.view.viewport().setAutoFillBackground(True)

        # Keep grouped-series colors stable across theme changes.
        existing_sets = list(self.series.barSets())
        fallback = group_color_cycle(len(existing_sets), dark_theme=self._dark_theme)
        for idx, bar_set in enumerate(existing_sets):
            brush_color = bar_set.brush().color() if bar_set.brush().color().isValid() else None
            color = QColor(brush_color) if brush_color is not None else QColor(fallback[idx])
            if not brush_color:
                bar_set.setBrush(QBrush(color))
            bar_set.setPen(QPen(QColor(color).darker(110 if not self._dark_theme else 140)))

        self.axis_x.setLabelsAngle(-45)

        try:
            scene = self.chart.scene()
            if scene is not None:
                scene.update()
        except Exception:
            self.chart.update()
        self.view.viewport().update()

    # ---------- Public API ----------
    def set_title(self, title: str):
        """Set the chart title."""
        self._title = title
        self.chart.setTitle(title)

    def set_y_label(self, label: str):
        """Set the Y-axis label."""
        self._y_label = label
        self.axis_y.setTitleText(label)

    # @ai(gpt-5, codex, refactor, 2026-02-27)
    def clear(self, *, request_repaint: bool = True):
        """Clear all data from the chart."""
        for s in list(self.series.barSets()):
            try:
                if s in self._connected_barsets:
                    try:
                        s.hovered.disconnect()
                    except Exception:
                        logger.warning("Failed to disconnect hovered signal while clearing group bars.", exc_info=True)
                    try:
                        s.clicked.disconnect()
                    except Exception:
                        logger.warning("Failed to disconnect clicked signal while clearing group bars.", exc_info=True)
                    self._connected_barsets.discard(s)
            except Exception:
                logger.warning("Failed to clean up connected bar-set signals while clearing group chart.", exc_info=True)
            self.series.remove(s)
        self.axis_x.clear()
        self.axis_x.setLabelsVisible(False)
        self.axis_x.setGridLineVisible(False)
        self.axis_x.setMinorGridLineVisible(False)
        self.axis_y.setRange(0.0, 1.0)
        self._categories.clear()
        self._values.clear()
        self._tooltip_overrides.clear()
        if request_repaint:
            try:
                self.chart.update()
            except Exception:
                logger.warning("Failed to refresh chart scene after clearing group chart.", exc_info=True)
            try:
                self.view.viewport().update()
            except Exception:
                logger.warning("Failed to refresh chart viewport after clearing group chart.", exc_info=True)

    def set_tooltip_overrides(self, overrides: Optional[dict[tuple[str, str], str]] = None):
        """Set per-bar tooltip detail lines.

        Keys are tuples of (series_name, category_name).
        """
        self._tooltip_overrides = dict(overrides or {})

    def set_data(
        self,
        categories: List[str],
        values: List[float],
        series_name: str = "Value",
    ):
        """Set the chart data.
        
        Args:
            categories: List of category/group names for X-axis
            values: List of values corresponding to each category
            series_name: Name for the data series (used in legend)
        """
        self.clear(request_repaint=False)
        
        if not categories or not values:
            return
        
        # Ensure same length
        n = min(len(categories), len(values))
        self._categories = [str(c) for c in categories[:n]]
        self._values = [float(v) if pd.notna(v) else 0.0 for v in values[:n]]

        # Create bar set
        if series_name == "Value":
            series_name = tr("Value")
        self._single_series_name = str(series_name)
        bar = QBarSet(series_name)
        bar.append(self._values)
        primary = group_color_cycle(1, dark_theme=self._dark_theme)[0]
        bar.setBrush(QBrush(primary))
        bar.setPen(QPen(QColor(primary).darker(110 if not self._dark_theme else 140)))
        self.series.append(bar)

        # Connect signals
        try:
            bar.hovered.connect(self._on_set_hovered)
            bar.clicked.connect(self._on_bar_clicked)
            self._connected_barsets.add(bar)
        except Exception:
            logger.warning("Failed to connect hover/click handlers while setting group chart data.", exc_info=True)

        # Set categories
        self.axis_x.append([_short_label(c, 20) for c in self._categories])
        try:
            self.axis_x.setLabelsVisible(True)
            self.axis_x.setGridLineVisible(True)
            self.axis_x.setMinorGridLineVisible(True)
        except Exception:
            logger.warning("Failed to restore X-axis visibility while rendering group chart data.", exc_info=True)

        # Set Y range
        if self._values:
            vmin = min(self._values)
            vmax = max(self._values)
        else:
            vmin, vmax = 0.0, 1.0
        
        y_min = min(vmin, 0.0)
        y_max = max(vmax, 0.0)
        if y_max == y_min:
            y_max = y_min + 1.0
        
        # Add padding
        padding = (y_max - y_min) * 0.1
        if vmin >= 0:
            y_min = 0.0
        self.axis_y.setRange(y_min, y_max + padding)

        # Keep explicit group palette color for single-series group bars.

    def set_dataframe(
        self,
        df: pd.DataFrame,
        category_col: str,
        value_col: str,
        series_name: Optional[str] = None,
    ):
        """Set chart data from a DataFrame.
        
        Args:
            df: DataFrame with category and value columns
            category_col: Name of the column containing categories
            value_col: Name of the column containing values
            series_name: Optional name for the series (defaults to value_col)
        """
        if df is None or df.empty:
            self.clear()
            return
        
        if category_col not in df.columns or value_col not in df.columns:
            self.clear()
            return
        
        categories = df[category_col].tolist()
        values = df[value_col].tolist()
        name = series_name or value_col
        
        self.set_data(categories, values, series_name=name)

    def set_multi_series(
        self,
        df: pd.DataFrame,
        category_col: str = "group",
        value_cols: Optional[List[str]] = None,
        series_colors: Optional[dict[str, QColor]] = None,
    ):
        """Set multiple series (stacked or grouped bars) from a DataFrame.
        
        Args:
            df: DataFrame with category column and multiple value columns
            category_col: Name of the column containing categories
            value_cols: List of column names to use as series (default: all numeric columns)
        """
        self.clear(request_repaint=False)
        
        if df is None or df.empty or category_col not in df.columns:
            return
        
        # Determine value columns
        if value_cols is None:
            value_cols = [
                c for c in df.columns
                if c != category_col and pd.api.types.is_numeric_dtype(df[c])
            ]
        
        if not value_cols:
            return
        
        # Get unique categories
        categories = df[category_col].unique().tolist()
        self._categories = [str(c) for c in categories]
        
        # Create bar sets for each value column
        colors = self._get_series_colors(len(value_cols))

        for idx, col in enumerate(value_cols):
            series_label = _short_label(str(col), 20)
            bar = QBarSet(series_label)
            values = []
            for cat in categories:
                subset = df[df[category_col] == cat]
                if not subset.empty:
                    val = subset[col].mean()  # Use mean if multiple rows per category
                else:
                    val = 0.0
                values.append(float(val) if pd.notna(val) else 0.0)
            bar.append(values)
            
            # Apply color
            color = None
            if isinstance(series_colors, dict):
                color = series_colors.get(str(col))
            if color is None and idx < len(colors):
                color = colors[idx]
            if color is not None:
                qcolor = QColor(color)
                bar.setBrush(QBrush(qcolor))
                bar.setPen(QPen(QColor(qcolor).darker(110 if not self._dark_theme else 140)))
            
            self.series.append(bar)
            
            # Connect signals
            try:
                bar.clicked.connect(self._make_click_handler(col))
                bar.hovered.connect(self._make_hover_handler(str(col), series_label, categories, values))
                self._connected_barsets.add(bar)
            except Exception:
                logger.warning("Failed to connect hover/click handlers for multi-series group bars.", exc_info=True)
        
        # Set categories
        self.axis_x.append([_short_label(c, 20) for c in self._categories])
        try:
            self.axis_x.setLabelsVisible(True)
            self.axis_x.setGridLineVisible(True)
            self.axis_x.setMinorGridLineVisible(True)
        except Exception:
            logger.warning("Failed to restore X-axis visibility while rendering multi-series group chart.", exc_info=True)
        
        # Update Y range
        all_values = []
        for col in value_cols:
            all_values.extend(df[col].dropna().tolist())
        
        if all_values:
            vmin = min(all_values)
            vmax = max(all_values)
        else:
            vmin, vmax = 0.0, 1.0
        
        y_min = min(vmin, 0.0)
        y_max = max(vmax, 0.0)
        if y_max == y_min:
            y_max = y_min + 1.0
        
        padding = (y_max - y_min) * 0.1
        if vmin >= 0:
            y_min = 0.0
        self.axis_y.setRange(y_min, y_max + padding)
        
        # Show legend if multiple series
        if len(value_cols) > 1:
            self.chart.legend().setVisible(True)
            if self._chart_colors is not None:
                style_legend(self.chart.legend(), self._chart_colors)
        else:
            self.chart.legend().hide()

    def _get_series_colors(self, count: int) -> List[QColor]:
        """Get a list of colors for multiple series."""
        return group_color_cycle(count, dark_theme=self._dark_theme)

    def _make_click_handler(self, series_name: str):
        """Create a click handler for a specific series."""
        def handler(index: int):
            if index < 0 or index >= len(self._categories):
                return
            category = self._categories[index]
            try:
                self.bar_clicked.emit(index)
                self.category_selected.emit(category, series_name)
            except Exception:
                logger.warning("Failed to update cursor while handling group-bar hover leave.", exc_info=True)
        return handler

    def _make_hover_handler(
        self,
        series_key: str,
        series_display_name: str,
        categories: list[str],
        values: list[float],
    ):
        """Create a hover handler that shows category + value tooltips."""

        def handler(status: bool, index: int):
            if not status:
                QToolTip.hideText()
                try:
                    self.view.setCursor(Qt.ArrowCursor)
                except Exception:
                    logger.warning("Failed to display tooltip while handling group-bar hover.", exc_info=True)
                return

            if index < 0 or index >= len(categories):
                return

            category = categories[index]
            value = values[index] if index < len(values) else 0.0
            try:
                self.view.setCursor(Qt.PointingHandCursor)
            except Exception:
                logger.warning("Failed to emit bar-click signal for group chart.", exc_info=True)
            extra = self._tooltip_overrides.get((str(series_key), str(category)), "")
            try:
                QToolTip.showText(
                    QCursor.pos(),
                    f"{category}\n{series_display_name}: {value:.4f}" + (f"\n{extra}" if extra else ""),
                    self,
                )
            except Exception:
                logger.warning("Failed to emit category-selected signal for group chart.", exc_info=True)

        return handler

    # ---------- Signal Handlers ----------
    def _connect_signals(self):
        """Connect chart interaction signals."""
        self.view.mousePressEvent = self._mouse_press_wrapper(self._native_mouse_press_event)

    def _on_set_hovered(self, status: bool, index: int):
        """Handle bar hover events."""
        self._hover_idx = index if status else None
        self.view.setCursor(Qt.PointingHandCursor if status else Qt.ArrowCursor)
        
        if not status:
            QToolTip.hideText()
            return
        
        try:
            if 0 <= index < len(self._categories):
                category = self._categories[index]
                value = self._values[index] if index < len(self._values) else 0.0
                extra = self._tooltip_overrides.get((str(self._single_series_name), str(category)), "")
                QToolTip.showText(
                    QCursor.pos(),
                    f"{category}\n{self._single_series_name}: {value:.4f}" + (f"\n{extra}" if extra else ""),
                    self
                )
        except Exception:
            logger.warning("Failed to update tooltip/cursor on group chart hover.", exc_info=True)

    def _on_bar_clicked(self, index: int):
        """Handle bar click events."""
        if index < 0 or index >= len(self._categories):
            return
        
        category = self._categories[index]
        value = self._values[index] if index < len(self._values) else 0.0
        
        try:
            self.bar_clicked.emit(index)
            self.category_selected.emit(category, value)
        except Exception:
            logger.warning("Failed to emit click-selection signals for group chart bar.", exc_info=True)

    def _mouse_press_wrapper(self, base_handler):
        """Wrap mouse press to handle clicks on bars."""
        def handler(ev):
            if ev.button() == Qt.LeftButton and self._hover_idx is not None:
                self._on_bar_clicked(self._hover_idx)
            return base_handler(ev)
        return handler


__all__ = ["GroupBarChart"]

