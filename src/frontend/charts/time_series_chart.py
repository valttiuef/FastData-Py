
from __future__ import annotations
from bisect import bisect_left
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QDate, QDateTime, QTime, QPointF, QRectF, QTimer, Signal
from PySide6.QtWidgets import QFrame, QVBoxLayout, QGraphicsRectItem, QGraphicsSimpleTextItem
from PySide6.QtCharts import QChart, QLineSeries, QDateTimeAxis, QValueAxis
from PySide6.QtGui import QColor, QPen, QPalette, QBrush, QFont

from .interactive_chart_view import InteractiveChartView
from . import MAX_FEATURES_SHOWN_LEGEND
from ..style.cluster_colors import cluster_color_for_label
from ..style.chart_theme import (

    make_colors_from_palette,
    apply_chart_background,
    style_axis,
    style_legend,
    is_dark_color,
    window_color_from_theme,
)

import logging
logger = logging.getLogger(__name__)
from ..style.theme_manager import theme_manager


def _ts_to_qdatetime_local(ts: pd.Timestamp) -> QDateTime:
    """Build a QDateTime by components (no tz math)."""
    ts = pd.Timestamp(ts)
    return QDateTime(
        QDate(ts.year, ts.month, ts.day),
        QTime(ts.hour, ts.minute, ts.second, int(ts.microsecond / 1000)),
    )


def _qdatetime_to_wall_timestamp(qdt: QDateTime) -> pd.Timestamp | None:
    """Convert QDateTime to timezone-naive wall-clock pandas Timestamp."""
    if not qdt or not qdt.isValid():
        return None
    d = qdt.date()
    t = qdt.time()
    try:
        return pd.Timestamp(
            year=d.year(),
            month=d.month(),
            day=d.day(),
            hour=t.hour(),
            minute=t.minute(),
            second=t.second(),
            microsecond=t.msec() * 1000,
        )
    except Exception:
        return None


def _to_ms_epoch_array(t_arr) -> np.ndarray:
    """
    Convert timestamps to ms since epoch such that 00:00 stays 00:00 on the axis.
    We build a local QDateTime by components, then call toMSecsSinceEpoch().
    """
    ts = pd.to_datetime(list(t_arr), errors="coerce")
    out: list[int] = []
    for t in ts:
        if pd.isna(t):
            continue
        t = pd.Timestamp(t)
        qdt = QDateTime(
            QDate(t.year, t.month, t.day),
            QTime(t.hour, t.minute, t.second, int(t.microsecond / 1000)),
        )
        out.append(int(qdt.toMSecsSinceEpoch()))
    return np.asarray(out, dtype=np.int64)


def _short_label(name: str, max_len: int = 24) -> str:
    """Return a compact label using only the primary feature name and truncate if too long."""
    if not name:
        return ""
    # Feature display names are joined with ' · '; take the first part (label)
    try:
        primary = str(name).split(" · ")[0]
    except Exception:
        primary = str(name)
    if len(primary) <= max_len:
        return primary
    return primary[: max_len - 1] + "…"


class TimeSeriesChart(QFrame):
    """
    Presents one or more line series with interactive zoom/pan.
    Emits pandas Timestamps when the user changes the visible window.
    """
    range_changed = Signal(object, object)   # (pd.Timestamp start, end)
    reset_requested = Signal()               # ask controller to reset to true bounds

    def __init__(self, title: str = "Timeseries", parent=None):
        super().__init__(parent)

        self.chart = QChart()
        self.chart.legend().setVisible(True)

        self.axis_x = QDateTimeAxis()
        self.axis_x.setFormat("yyyy-MM-dd HH:mm")
        self.axis_x.setTitleText("Time")
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)

        self.axis_y = QValueAxis()
        self.axis_y.setLabelFormat("%.3f")
        self.axis_y.setTitleText("Value")
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)

        self.view = InteractiveChartView(self.chart)
        self.view._owner = self  # best-effort
        self.view.user_reset_requested.connect(self._bubble_reset)
        self.view.user_range_selected.connect(self._on_user_range_selected)

        lay = QVBoxLayout(self)
        lay.addWidget(self.view)

        self.chart.setTitle(title)

        self._series_segments: dict[str, list[QLineSeries]] = {}
        self._series_colors: dict[str, QColor] = {}
        self._feature_order: list[str] = []
        self._max_series: int = MAX_FEATURES_SHOWN_LEGEND
        self._current_frame: pd.DataFrame = pd.DataFrame()
        self._current_theme: str | None = None
        self._dark_theme: bool = False
        self._chart_colors = None
        self._data_x_range: tuple[QDateTime, QDateTime] | None = None
        self._data_y_range: tuple[float, float] | None = None
        self._preserve_series_colors: bool = False
        self._group_box_specs: list[tuple[float, float, float, float, str, str]] = []
        self._group_hover_specs: list[tuple[float, float, float, str]] = []
        self._hover_group_points: list[tuple[int, str]] = []
        self._hover_group_ms: list[int] = []
        self._group_box_items: list[QGraphicsRectItem] = []
        self._group_box_text_items: list[QGraphicsSimpleTextItem] = []
        self._deferred_refresh_pending: bool = False
        self.view.set_hover_tooltip_callback(self._group_timeline_hover_tooltip)

        try:
            self._current_theme = theme_manager().current_theme
        except AssertionError:
            self._current_theme = None

        self._apply_theme(self._current_theme)
        self._refresh_legend_markers()

        try:
            theme_manager().theme_changed.connect(self._on_theme_changed)
        except AssertionError:
            logger.warning("Failed to connect theme change signal for time-series chart.", exc_info=True)
        try:
            self.axis_x.rangeChanged.connect(lambda *_args: self._refresh_group_timeline_overlays())
        except Exception:
            logger.warning("Failed to connect X-axis range change handler for timeline overlays.", exc_info=True)
        try:
            self.chart.plotAreaChanged.connect(lambda *_args: self._refresh_group_timeline_overlays())
        except Exception:
            logger.warning("Failed to connect plot-area change handler for timeline overlays.", exc_info=True)

    # @ai(gpt-5, codex, refactor, 2026-02-27)
    def clear(
        self,
        *,
        reset_axes: bool = True,
        request_repaint: bool = True,
    ):
        self._clear_group_timeline_overlays()
        self._group_box_specs = []
        self._group_hover_specs = []
        self._hover_group_points = []
        self._hover_group_ms = []
        self._preserve_series_colors = False
        self._remove_all_series()
        self._current_frame = pd.DataFrame()
        self._feature_order = []
        self._data_x_range = None
        self._data_y_range = None
        if reset_axes:
            try:
                self.axis_x.setLabelsVisible(False)
            except Exception:
                logger.warning("Failed to hide X-axis labels while clearing time-series chart.", exc_info=True)
            try:
                self.axis_x.setFormat("yyyy-MM-dd HH:mm")
            except Exception:
                logger.warning("Failed to reset X-axis format while clearing time-series chart.", exc_info=True)
            try:
                self.axis_y.setRange(0.0, 1.0)
            except Exception:
                logger.warning("Failed to reset Y-axis range while clearing time-series chart.", exc_info=True)
            try:
                self.axis_y.setTickType(QValueAxis.TickType.TicksFixed)
                self.axis_y.setTickCount(6)
                self.axis_y.setLabelFormat("%.3f")
            except Exception:
                logger.warning("Failed to reset Y-axis tick settings while clearing time-series chart.", exc_info=True)
        self._refresh_legend_markers()
        if request_repaint:
            try:
                self.chart.update()
            except Exception:
                logger.warning("Failed to refresh chart scene after clearing time-series chart.", exc_info=True)
            try:
                self.view.viewport().update()
            except Exception:
                logger.warning("Failed to refresh chart viewport after clearing time-series chart.", exc_info=True)

    def set_title(self, title: str):
        self.chart.setTitle(title)

    def set_max_series(self, count: int | None) -> None:
        if count is None:
            self._max_series = MAX_FEATURES_SHOWN_LEGEND
            return
        try:
            value = int(count)
        except Exception:
            value = MAX_FEATURES_SHOWN_LEGEND
        self._max_series = max(1, value)

    def set_dataframe(self, frame: pd.DataFrame | None):
        # Keep current axis state during regular data refreshes to avoid
        # intermediate 0..1 axis flicker before new ranges are applied.
        self.clear(reset_axes=False, request_repaint=False)
        self._preserve_series_colors = False
        self._group_hover_specs = []
        if frame is None:
            return

        df = frame.copy()
        if df.empty or "t" not in df.columns:
            return

        # Keep whatever dtype/timezone `t` already has; just parse if needed.
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df = df.dropna(subset=["t"]).sort_values("t")
        if df.empty:
            return

        feature_cols = [c for c in df.columns if c != "t"][: self._max_series]
        if not feature_cols:
            return

        self._current_frame = df
        self._feature_order = feature_cols

        colors = self._generate_colors(len(feature_cols))
        self._series_segments = {}
        self._series_colors = {name: color for name, color in zip(feature_cols, colors)}

        # raw times
        times_ms_all = _to_ms_epoch_array(df["t"])

        all_values: list[float] = []
        for name, color in zip(feature_cols, colors):
            try:
                values_arr = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            except Exception:
                values_arr = df[name].to_numpy()

            segments = self._build_segments_for_feature(name, times_ms_all, values_arr, color)
            if segments:
                self._series_segments[name] = segments
                for seg in segments:
                    all_values.extend([point.y() for point in seg.points()])

        # Use the SAME ms numbers to set the axis range (no tz logic here).
        if times_ms_all.size:
            ms_min = int(times_ms_all.min())
            ms_max = int(times_ms_all.max())
            if ms_min == ms_max:
                ms_max += 1  # avoid zero-span

            # If you want local-looking labels, just use default LocalTime here.
            qmin = QDateTime.fromMSecsSinceEpoch(ms_min)       # LocalTime by default
            qmax = QDateTime.fromMSecsSinceEpoch(ms_max)
            self.axis_x.setRange(qmin, qmax)
            self._data_x_range = (QDateTime(qmin), QDateTime(qmax))
            try:
                self.axis_x.setLabelsVisible(True)
            except Exception:
                logger.warning("Failed to show X-axis labels after setting time-series data.", exc_info=True)

            try:
                self._set_x_axis_format_for_span_ms(ms_max - ms_min)
            except Exception:
                logger.warning("Failed to format X-axis labels for time-series span.", exc_info=True)

        # Y-range as before
        try:
            self.axis_y.setLabelsVisible(True)
        except Exception:
            logger.warning("Failed to show Y-axis labels after setting time-series data.", exc_info=True)
        try:
            # Reset any categorical tick setup left by group timelines.
            self.axis_y.setTickType(QValueAxis.TickType.TicksFixed)
            self.axis_y.setTickCount(6)
            self.axis_y.setLabelFormat("%.3f")
        except Exception:
            logger.warning("Failed to restore numeric Y-axis tick settings for time-series data.", exc_info=True)
        self.axis_y.setTitleText("Value")
        if all_values:
            ymin = float(min(all_values)); ymax = float(max(all_values))
        else:
            ymin, ymax = 0.0, 1.0
        if ymin == ymax:
            ymin -= 0.5; ymax += 0.5
        self.axis_y.setRange(ymin, ymax)
        self._data_y_range = (float(ymin), float(ymax))

        # Refresh legend markers once after all series are built
        self._refresh_legend_markers()
        self._apply_theme(self._current_theme)
        self._request_deferred_refresh()

    def set_group_timeline(
        self,
        frame: pd.DataFrame | None,
        *,
        time_col: str = "t",
        group_col: str = "group",
        group_names: dict[object, str] | None = None,
        overlay_cols: list[str] | None = None,
        max_groups: int = 160,
        max_rows: int = 120_000,
    ) -> None:
        """Render categorical timeline as colored horizontal group lanes."""
        self.clear(reset_axes=False, request_repaint=False)
        self._preserve_series_colors = True
        if frame is None:
            return

        df = frame.copy()
        if df.empty or time_col not in df.columns or group_col not in df.columns:
            return

        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df[group_col] = pd.to_numeric(df[group_col], errors="coerce")
        df = df.dropna(subset=[time_col, group_col]).sort_values(time_col).reset_index(drop=True)
        if df.empty:
            return

        if len(df) > max_rows:
            stride = int(np.ceil(len(df) / max(1, int(max_rows))))
            sampled = df.iloc[::max(1, stride)].copy()
            if sampled.index[-1] != df.index[-1]:
                sampled = pd.concat([sampled, df.iloc[[-1]]], ignore_index=True)
            df = sampled.reset_index(drop=True)

        group_counts = df[group_col].value_counts(dropna=True)
        if max_groups > 0 and len(group_counts) > max_groups:
            keep_groups = set(group_counts.nlargest(max_groups).index.tolist())
            df = df[df[group_col].isin(keep_groups)].copy()
            if df.empty:
                return
            group_counts = df[group_col].value_counts(dropna=True)

        ordered_groups = [float(g) for g in sorted(group_counts.index.tolist())]
        if not ordered_groups:
            return

        times_ms = _to_ms_epoch_array(df[time_col])
        if times_ms.size == 0:
            return

        class _Run:
            __slots__ = ("group", "start_ms", "end_ms")

            def __init__(self, group: float, start_ms: int, end_ms: int) -> None:
                self.group = group
                self.start_ms = int(start_ms)
                self.end_ms = int(end_ms)

        runs: list[_Run] = []
        # Build per-row intervals from each timestamp to the next timestamp so
        # timeline ranges align with saved group timeframes.
        row_starts: list[int] = [0] * len(times_ms)
        row_ends: list[int] = [0] * len(times_ms)
        if len(times_ms) == 1:
            start = int(times_ms[0])
            row_starts[0] = start
            row_ends[0] = start + 1000
        else:
            positive_steps: list[int] = []
            for idx in range(len(times_ms) - 1):
                left = int(times_ms[idx])
                right = int(times_ms[idx + 1])
                step = right - left
                if step > 0:
                    positive_steps.append(step)
            inferred_step = min(positive_steps) if positive_steps else 1000
            for idx in range(len(times_ms) - 1):
                start = int(times_ms[idx])
                next_start = int(times_ms[idx + 1])
                row_starts[idx] = start
                row_ends[idx] = next_start if next_start > start else start + 1
            last_start = int(times_ms[-1])
            row_starts[-1] = last_start
            row_ends[-1] = last_start + max(1, int(inferred_step))

        current_group = float(df[group_col].iat[0])
        run_start_ms = int(row_starts[0])
        run_end_ms = int(row_ends[0])
        for idx in range(1, len(df)):
            group_value = float(df[group_col].iat[idx])
            start_ms = int(row_starts[idx])
            end_ms = int(row_ends[idx])
            if group_value == current_group:
                run_end_ms = max(run_end_ms, end_ms)
                continue
            runs.append(_Run(current_group, run_start_ms, run_end_ms))
            current_group = group_value
            run_start_ms = start_ms
            run_end_ms = end_ms
        runs.append(_Run(current_group, run_start_ms, run_end_ms))

        max_runs = 30_000
        if len(runs) > max_runs:
            stride = int(np.ceil(len(runs) / max_runs))
            runs = runs[::max(1, stride)]

        self._series_segments = {}
        self._series_colors = {}
        self._feature_order = []
        self._current_frame = df[[time_col, group_col]].copy()

        overlay_columns = [
            col for col in (overlay_cols or []) if col in frame.columns and col not in {time_col, group_col}
        ]

        # Create one minimal series per group to preserve legend entries/colors.
        legend_limit = 24
        group_meta: dict[float, tuple[str, str, QColor]] = {}
        ms_min = int(min(row_starts)) if row_starts else int(times_ms.min())
        ms_max = int(max(row_ends)) if row_ends else int(times_ms.max())
        if ms_min == ms_max:
            ms_max += 1
        for idx, group in enumerate(ordered_groups):
            series = QLineSeries()
            series.setUseOpenGL(False)
            label = (
                (group_names or {}).get(group)
                or (group_names or {}).get(int(group))
                or f"Cluster {int(group)}"
            )
            series.setName(str(label) if idx < legend_limit else "")
            # One off-range point keeps legend marker, but renders nothing in plot area.
            series.append(float(ms_min), -10_000.0)
            self.chart.addSeries(series)
            series.attachAxis(self.axis_x)
            series.attachAxis(self.axis_y)
            color = cluster_color_for_label(group, border=False)
            pen = QPen(color)
            pen.setWidthF(2.0)
            pen.setCosmetic(True)
            series.setPen(pen)
            key = f"group:{int(group)}:{label}"
            self._series_segments[key] = [series]
            self._series_colors[key] = QColor(color)
            self._feature_order.append(key)
            group_meta[group] = (str(label), key, QColor(color))

        # Optional numeric overlays: draw as thin traces on the chart's current
        # Y-scale. Callers can pre-scale values (for example to cluster bounds).
        if overlay_columns:
            overlay_frame = df[[time_col, *overlay_columns]].copy()
            overlay_times_ms = _to_ms_epoch_array(overlay_frame[time_col])
            for col in overlay_columns:
                vals = pd.to_numeric(overlay_frame[col], errors="coerce")
                segments = self._build_segments_for_feature(
                    str(col),
                    overlay_times_ms,
                    vals.to_numpy(dtype=float),
                    QColor(140, 140, 140),
                )
                if segments:
                    self._series_segments[str(col)] = segments
                    self._series_colors[str(col)] = QColor(140, 140, 140)
                    self._feature_order.append(str(col))

        qmin = QDateTime.fromMSecsSinceEpoch(ms_min)
        qmax = QDateTime.fromMSecsSinceEpoch(ms_max)
        self.axis_x.setRange(qmin, qmax)
        self._data_x_range = (QDateTime(qmin), QDateTime(qmax))
        self.axis_x.setLabelsVisible(True)
        self._set_x_axis_format_for_span_ms(ms_max - ms_min)

        y_min = float(min(ordered_groups) - 0.5)
        y_max = float(max(ordered_groups) + 0.5)
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5
        self.axis_y.setRange(y_min, y_max)
        try:
            # Use integer cluster-id ticks (0, 1, 2, ...) instead of rounded half-step ticks.
            self.axis_y.setTickType(QValueAxis.TickType.TicksDynamic)
            self.axis_y.setTickAnchor(float(min(ordered_groups)))
            self.axis_y.setTickInterval(1.0)
        except Exception:
            logger.warning("Failed to configure dynamic Y-axis ticks for group timeline rendering.", exc_info=True)
        self.axis_y.setLabelFormat("%.0f")
        try:
            self.axis_y.setLabelsVisible(True)
        except Exception:
            logger.warning("Failed to show Y-axis labels for group timeline rendering.", exc_info=True)
        self.axis_y.setTitleText("Cluster")
        self._data_y_range = (float(y_min), float(y_max))

        self._group_box_specs = []
        self._group_hover_specs = []
        for run in runs:
            label, key, _color = group_meta.get(
                float(run.group),
                (f"Cluster {int(run.group)}", f"group:{int(run.group)}", QColor(100, 100, 100)),
            )
            start = float(run.start_ms)
            end = float(run.end_ms if run.end_ms > run.start_ms else run.start_ms + 1)
            center = float(run.group)
            self._group_box_specs.append((start, end, center - 0.45, center + 0.45, label, key))
            self._group_hover_specs.append((start, end, center, label))

        legend_visible = len(ordered_groups) <= legend_limit
        self.chart.legend().setVisible(legend_visible)
        self._refresh_legend_markers()
        self._apply_theme(self._current_theme)
        self._refresh_group_timeline_overlays()
        self._request_deferred_refresh()

    def _group_timeline_hover_tooltip(self, pos: QPointF) -> str:
        chart_series = self.chart.series()
        if not chart_series:
            return ""
        try:
            data_pt = self.chart.mapToValue(pos, chart_series[0])
            x_value = float(data_pt.x())
        except Exception:
            return ""

        if self._group_hover_specs:
            matched_label = ""
            for start, end, _center, label in self._group_hover_specs:
                if start <= x_value <= end:
                    matched_label = label
                    break
            if matched_label:
                dt = QDateTime.fromMSecsSinceEpoch(int(round(x_value)))
                return f"Date: {dt.toString('yyyy-MM-dd HH:mm:ss')}\nGroup: {matched_label}"

        if not self._hover_group_points:
            return ""
        x_ms = int(round(x_value))
        idx = bisect_left(self._hover_group_ms, x_ms)
        candidates: list[tuple[int, str]] = []
        if 0 <= idx < len(self._hover_group_points):
            candidates.append(self._hover_group_points[idx])
        if idx - 1 >= 0:
            candidates.append(self._hover_group_points[idx - 1])
        if not candidates:
            return ""
        nearest_label = min(candidates, key=lambda item: abs(item[0] - x_ms))[1]
        return f"Group: {nearest_label}"

    def set_hover_group_series(
        self,
        frame: pd.DataFrame | None,
        *,
        time_col: str = "t",
        group_col: str = "group",
    ) -> None:
        self._hover_group_points = []
        self._hover_group_ms = []
        if frame is None or frame.empty:
            return
        if time_col not in frame.columns or group_col not in frame.columns:
            return

        work = frame[[time_col, group_col]].copy()
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.dropna(subset=[time_col])
        if work.empty:
            return

        points: list[tuple[int, str]] = []
        for ts, raw_group in zip(work[time_col], work[group_col]):
            if pd.isna(raw_group):
                continue
            label = str(raw_group).strip()
            if not label:
                continue
            qdt = _ts_to_qdatetime_local(pd.Timestamp(ts))
            points.append((int(qdt.toMSecsSinceEpoch()), label))
        if not points:
            return
        points.sort(key=lambda item: item[0])
        self._hover_group_points = points
        self._hover_group_ms = [item[0] for item in points]

    def set_x_range(self, start: pd.Timestamp, end: pd.Timestamp):
        qmin = _ts_to_qdatetime_local(pd.Timestamp(start))
        qmax = _ts_to_qdatetime_local(pd.Timestamp(end))
        self.axis_x.setRange(qmin, qmax)
        try:
            self.axis_x.setLabelsVisible(True)
        except Exception:
            logger.warning("Failed to show X-axis labels after applying explicit chart range.", exc_info=True)
        try:
            span_ms = int(qmax.toMSecsSinceEpoch() - qmin.toMSecsSinceEpoch())
            self._set_x_axis_format_for_span_ms(span_ms)
        except Exception:
            logger.warning("Failed to update X-axis format after applying explicit chart range.", exc_info=True)
        self._refresh_group_timeline_overlays()
        self._request_deferred_refresh()

    def _bubble_reset(self, reset_x: bool = True, reset_y: bool = True):
        self._reset_to_data_bounds(reset_x=reset_x, reset_y=reset_y)
        # Only emit when X was reset because connected tabs use this signal to
        # recompute date-range filters and reload data windows.
        if reset_x:
            try:
                self.reset_requested.emit()
            except Exception:
                logger.warning("Failed to emit reset request from time-series chart reset action.", exc_info=True)

    # @ai(gpt-5, codex, fix, 2026-02-27)
    def _on_user_range_selected(self, qmin: QDateTime, qmax: QDateTime):
        # Update axis label format based on selected span
        try:
            span_ms = int(qmax.toMSecsSinceEpoch() - qmin.toMSecsSinceEpoch())
            self._set_x_axis_format_for_span_ms(span_ms)
        except Exception:
            logger.warning("Failed to update X-axis label format after user-selected range.", exc_info=True)
        self._refresh_group_timeline_overlays()

        # Keep wall-clock semantics symmetric with _ts_to_qdatetime_local()
        # to avoid timezone-offset jumps when interactions emit QDateTime.
        start = _qdatetime_to_wall_timestamp(qmin)
        end = _qdatetime_to_wall_timestamp(qmax)
        if start is None or end is None:
            return
        self.range_changed.emit(start, end)

    def _reset_to_data_bounds(self, *, reset_x: bool = True, reset_y: bool = True) -> None:
        if reset_x and self._data_x_range is not None:
            qmin, qmax = self._data_x_range
            try:
                self.axis_x.setRange(qmin, qmax)
                span_ms = int(qmax.toMSecsSinceEpoch() - qmin.toMSecsSinceEpoch())
                self._set_x_axis_format_for_span_ms(span_ms)
            except Exception:
                logger.warning("Failed to reset X-axis to data bounds in time-series chart.", exc_info=True)

        if reset_y and self._data_y_range is not None:
            try:
                self.axis_y.setRange(*self._data_y_range)
            except Exception:
                logger.warning("Failed to reset Y-axis to data bounds in time-series chart.", exc_info=True)
        self._refresh_group_timeline_overlays()

    def _remove_all_series(self):
        for segments in self._series_segments.values():
            for series in segments:
                try:
                    self.chart.removeSeries(series)
                except Exception:
                    logger.warning("Failed to remove a line-series segment while clearing time-series data.", exc_info=True)
        self._series_segments.clear()
        self._series_colors.clear()

    def _build_segments_for_feature(
        self,
        name: str,
        times_ms: np.ndarray,
        values: np.ndarray,
        color: QColor,
    ) -> list[QLineSeries]:
        """
        Build QLineSeries segments for runs of finite values. `times_ms` and
        `values` are numpy arrays aligned by index (same length). Missing
        timestamps should have been removed already; this function assumes
        times_ms contains valid int64 ms values.
        """
        segments: list[QLineSeries] = []

        if times_ms is None or len(times_ms) == 0:
            # nothing to plot
            series = QLineSeries()
            series.setName(name)
            self.chart.addSeries(series)
            series.attachAxis(self.axis_x)
            series.attachAxis(self.axis_y)
            self._set_series_pen(series, color)
            return [series]

        # Ensure values is a mutable float numpy array; some upstream arrays
        # can be read-only views (e.g., from pandas), which breaks gap fill.
        vals = np.array(values, dtype=float, copy=True)
        finite = np.isfinite(vals)

        # If there are no finite values, return an empty series (legend preserved)
        if not finite.any():
            series = QLineSeries()
            series.setName(name)
            self.chart.addSeries(series)
            series.attachAxis(self.axis_x)
            series.attachAxis(self.axis_y)
            self._set_series_pen(series, color)
            return [series]

        # Keep NaN/inf gaps as hard breaks. Gap filling/interpolation must come
        # from preprocessing options, not chart rendering.
        # find runs of consecutive finite values
        edges = np.flatnonzero(np.diff(np.concatenate(([0], finite.view(np.int8), [0]))))
        runs = list(zip(edges[0::2], edges[1::2]))

        # Create QLineSeries for each run
        for seg_idx, (start_idx, end_idx) in enumerate(runs):
            series = QLineSeries()
            series.setUseOpenGL(False)
            series.setName(name if seg_idx == 0 else "")
            # Build QPointF list for the segment
            pts = [QPointF(float(times_ms[i]), float(vals[i])) for i in range(start_idx, end_idx)]
            series.replace(pts)
            self.chart.addSeries(series)
            series.attachAxis(self.axis_x)
            series.attachAxis(self.axis_y)
            self._set_series_pen(series, color)
            segments.append(series)

        return segments

    def _set_series_pen(self, series: QLineSeries, color: QColor):
        pen = QPen(color)
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        series.setPen(pen)

    def _generate_colors(self, count: int) -> list[QColor]:
        base_palette = [
            QColor(66, 133, 244),   # blue
            QColor(219, 68, 55),    # red
            QColor(244, 180, 0),    # orange
            QColor(15, 157, 88),    # green
            QColor(171, 71, 188),   # purple
            QColor(0, 172, 193),    # teal
        ]
        colors: list[QColor] = []
        for idx in range(max(1, count)):
            base = QColor(base_palette[idx % len(base_palette)])
            if self._dark_theme:
                base = base.lighter(130)
            if idx >= len(base_palette):
                factor = 110 + 15 * (idx // len(base_palette))
                if base.lightness() < 128:
                    base = base.lighter(factor)
                else:
                    base = base.darker(factor)
            colors.append(base)
        return colors

    def _on_theme_changed(self, theme_name: str):
        self._current_theme = theme_name
        self._apply_theme(theme_name)

    def _apply_theme(self, theme_name: str | None = None):
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
            logger.warning("Failed to set title brush while applying time-series theme.", exc_info=True)

        frame_palette = self.palette()
        frame_palette.setColor(QPalette.Window, container_color)
        frame_palette.setColor(QPalette.Base, container_color)
        self.setPalette(frame_palette)
        self.setAutoFillBackground(True)
        try:
            self.setAttribute(Qt.WA_StyledBackground, True)
        except Exception:
            logger.warning("Failed to set styled-background attribute while applying time-series theme.", exc_info=True)

        view_palette = self.view.palette()
        view_palette.setColor(QPalette.Window, container_color)
        view_palette.setColor(QPalette.Base, container_color)
        self.view.setPalette(view_palette)
        self.view.setAutoFillBackground(True)
        try:
            self.view.setBackgroundBrush(QBrush(container_color))
        except Exception:
            logger.warning("Failed to set chart-view background brush while applying time-series theme.", exc_info=True)

        viewport_palette = self.view.viewport().palette()
        viewport_palette.setColor(QPalette.Window, container_color)
        viewport_palette.setColor(QPalette.Base, container_color)
        self.view.viewport().setPalette(viewport_palette)
        self.view.viewport().setAutoFillBackground(True)
        try:
            self.view.viewport().setBackgroundRole(QPalette.Window)
        except Exception:
            logger.warning("Failed to set viewport background role while applying time-series theme.", exc_info=True)

        try:
            scene = self.chart.scene()
            if scene is not None:
                scene.setBackgroundBrush(QBrush(container_color))
        except Exception:
            logger.warning("Failed to update scene background while applying time-series theme.", exc_info=True)

        if self._feature_order and not self._preserve_series_colors:
            regenerated = self._generate_colors(len(self._feature_order))
            self._series_colors = {
                name: QColor(color)
                for name, color in zip(self._feature_order, regenerated)
            }

        if not self._series_segments:
            return

        for name, segments in self._series_segments.items():
            color = self._series_colors.get(name, colors.line_primary)
            for series in segments:
                self._set_series_pen(series, color)

        self._refresh_legend_markers()

        try:
            scene = self.chart.scene()
            if scene is not None:
                scene.update()
        except Exception:
            self.chart.update()
        self.view.viewport().update()

    def _refresh_legend_markers(self) -> None:
        legend = self.chart.legend()
        try:
            markers = list(legend.markers())
        except Exception:
            markers = []

        seen: set[str] = set()
        for marker in markers:
            series = marker.series()
            if series is None:
                marker.setVisible(False)
                continue
            name = series.name() or ""
            if not name:
                marker.setVisible(False)
                continue
            if name in seen:
                marker.setVisible(False)
                continue
            seen.add(name)
            marker.setLabel(_short_label(name))
            marker.setVisible(True)

        if self._chart_colors is not None:
            style_legend(legend, self._chart_colors)
        self._refresh_group_timeline_overlays()

    def _clear_group_timeline_overlays(self) -> None:
        scene = self.chart.scene()
        for item in self._group_box_items:
            try:
                if scene is not None:
                    scene.removeItem(item)
            except Exception:
                logger.warning("Failed to remove a group timeline rectangle overlay item.", exc_info=True)
        for item in self._group_box_text_items:
            try:
                if scene is not None:
                    scene.removeItem(item)
            except Exception:
                logger.warning("Failed to remove a group timeline text overlay item.", exc_info=True)
        self._group_box_items = []
        self._group_box_text_items = []

    def _refresh_group_timeline_overlays(self) -> None:
        self._clear_group_timeline_overlays()
        if not self._group_box_specs:
            return
        if not self.chart.series():
            return
        scene = self.chart.scene()
        if scene is None:
            return
        plot_area = self.chart.plotArea()
        if plot_area.width() <= 0 or plot_area.height() <= 0:
            return
        ref_series = self.chart.series()[0]
        visible_start = float(self.axis_x.min().toMSecsSinceEpoch())
        visible_end = float(self.axis_x.max().toMSecsSinceEpoch())
        if visible_end <= visible_start:
            return
        visible_specs = [
            spec for spec in self._group_box_specs if not (spec[1] < visible_start or spec[0] > visible_end)
        ]
        max_visible_boxes = 5_000
        if len(visible_specs) > max_visible_boxes:
            stride = int(np.ceil(len(visible_specs) / max_visible_boxes))
            visible_specs = visible_specs[::max(1, stride)]

        font = QFont()
        font.setPointSize(8)
        text_budget = 1_000
        min_box_width_px = 2.0
        min_box_height_px = 1.5
        for start_ms, end_ms, y0, y1, label, key in visible_specs:
            x1 = self.chart.mapToPosition(QPointF(float(start_ms), float(y0)), ref_series).x()
            x2 = self.chart.mapToPosition(QPointF(float(end_ms), float(y0)), ref_series).x()
            left = float(min(x1, x2))
            right = float(max(x1, x2))
            if right < plot_area.left() or left > plot_area.right():
                continue
            left = max(float(plot_area.left()), left)
            right = min(float(plot_area.right()), right)
            width = right - left
            if width < min_box_width_px:
                center_x = (left + right) / 2.0
                left = center_x - min_box_width_px / 2.0
                right = center_x + min_box_width_px / 2.0
                if left < float(plot_area.left()):
                    right += float(plot_area.left()) - left
                    left = float(plot_area.left())
                if right > float(plot_area.right()):
                    left -= right - float(plot_area.right())
                    right = float(plot_area.right())
                left = max(float(plot_area.left()), left)
                right = min(float(plot_area.right()), right)
                width = max(0.0, right - left)
                if width <= 0.0:
                    continue
            py_top = self.chart.mapToPosition(QPointF(float(start_ms), float(y1)), ref_series).y()
            py_bottom = self.chart.mapToPosition(QPointF(float(start_ms), float(y0)), ref_series).y()
            top = float(min(py_top, py_bottom))
            bottom = float(max(py_top, py_bottom))
            if bottom < plot_area.top() or top > plot_area.bottom():
                continue
            top = max(float(plot_area.top()), top)
            bottom = min(float(plot_area.bottom()), bottom)
            height = bottom - top
            if height < min_box_height_px:
                center_y = (top + bottom) / 2.0
                top = center_y - min_box_height_px / 2.0
                bottom = center_y + min_box_height_px / 2.0
                if top < float(plot_area.top()):
                    bottom += float(plot_area.top()) - top
                    top = float(plot_area.top())
                if bottom > float(plot_area.bottom()):
                    top -= bottom - float(plot_area.bottom())
                    bottom = float(plot_area.bottom())
                top = max(float(plot_area.top()), top)
                bottom = min(float(plot_area.bottom()), bottom)
                height = max(0.0, bottom - top)
                if height <= 0.0:
                    continue
            rect = QRectF(left, top, width, height)
            fill = QColor(self._series_colors.get(key, QColor(120, 120, 120)))
            fill.setAlpha(140)
            border = QColor(fill).darker(130)
            border.setAlpha(195)
            rect_item = QGraphicsRectItem(rect)
            rect_item.setPen(QPen(border, 1.0))
            rect_item.setBrush(QBrush(fill))
            scene.addItem(rect_item)
            self._group_box_items.append(rect_item)

            if text_budget <= 0:
                continue
            text_item = QGraphicsSimpleTextItem(label)
            text_item.setFont(font)
            text_bounds = text_item.boundingRect()
            if text_bounds.width() + 8.0 > width or text_bounds.height() + 2.0 > height:
                continue
            text_color = QColor(15, 15, 15) if fill.lightness() > 150 else QColor(250, 250, 250)
            text_item.setBrush(QBrush(text_color))
            tx = left + (width - text_bounds.width()) / 2.0
            ty = top + (height - text_bounds.height()) / 2.0
            text_item.setPos(tx, ty)
            scene.addItem(text_item)
            self._group_box_text_items.append(text_item)
            text_budget -= 1

    def _request_deferred_refresh(self) -> None:
        if self._deferred_refresh_pending:
            return
        self._deferred_refresh_pending = True

        def _do_refresh() -> None:
            self._deferred_refresh_pending = False
            try:
                self._refresh_group_timeline_overlays()
            except Exception:
                logger.warning("Failed to schedule deferred chart scene refresh for time-series chart.", exc_info=True)
            try:
                self.chart.update()
            except Exception:
                logger.warning("Failed to execute deferred chart scene refresh for time-series chart.", exc_info=True)
            try:
                self.view.viewport().update()
            except Exception:
                logger.warning("Failed to execute deferred viewport refresh for time-series chart.", exc_info=True)

        QTimer.singleShot(0, _do_refresh)

    def _set_x_axis_format_for_span_ms(self, span_ms: int):
        """Choose a QDateTimeAxis format string based on the visible span in milliseconds.

        The goal is to keep labels readable at different zoom levels (show seconds when
        zoomed in, dates when zoomed out).
        """
        try:
            # thresholds
            sec = 1000
            minute = 60 * sec
            hour = 60 * minute
            day = 24 * hour
            year = 365 * day

            if span_ms <= 5 * sec:
                fmt = "HH:mm:ss.zzz"
            elif span_ms <= 60 * sec:
                fmt = "HH:mm:ss"
            elif span_ms <= 60 * minute:
                fmt = "HH:mm"
            elif span_ms <= day:
                fmt = "yyyy-MM-dd HH:mm"
            # Keep daily labels for full-month windows (31 days) and nearby ranges.
            elif span_ms <= 62 * day:
                fmt = "yyyy-MM-dd"
            elif span_ms <= year*5: # 5 years to show months
                fmt = "yyyy-MM"
            else:
                fmt = "yyyy"

            self.axis_x.setFormat(fmt)
        except Exception:
            logger.warning("Failed to apply X-axis datetime label format for current timespan.", exc_info=True)
