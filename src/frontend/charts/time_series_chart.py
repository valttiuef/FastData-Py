
from __future__ import annotations
from bisect import bisect_left
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QDate, QDateTime, QTime, QPointF, QRectF, QTimer, Signal, QSize
from PySide6.QtWidgets import QFrame, QVBoxLayout, QGraphicsRectItem, QGraphicsSimpleTextItem, QToolButton, QStyle
from PySide6.QtCharts import QChart, QLineSeries, QDateTimeAxis, QValueAxis
from PySide6.QtGui import QColor, QPen, QPalette, QBrush, QFont
import qtawesome as qta

from .interactive_chart_view import InteractiveChartView
from . import (
    MAX_FEATURES_SHOWN_LEGEND,
    TIMESERIES_GAP_DETECTION_ENABLED,
    TIMESERIES_GAP_IRREGULAR_MULTIPLIER,
    TIMESERIES_GAP_REGULAR_MULTIPLIER,
)
from ..style.group_colors import group_color_for_label, group_color_cycle
from ..style.chart_theme import (

    make_colors_from_palette,
    apply_chart_background,
    style_axis,
    style_legend,
    is_dark_color,
)
from ..style.styles import muted_icon_color

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
        self._x_axis_title_base = "Time"
        self.axis_x.setTitleText(self._x_axis_title_base)
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)

        self.axis_y = QValueAxis()
        self.axis_y.setLabelFormat("%.3f")
        self._y_axis_title_base = "Value"
        self.axis_y.setTitleText(self._y_axis_title_base)
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)

        self.view = InteractiveChartView(self.chart)
        self.view.setFrameShape(QFrame.Shape.NoFrame)
        self.view._owner = self  # best-effort
        self.view.set_live_pan_emit_enabled(True)
        self.view.user_reset_requested.connect(self._bubble_reset)
        self.view.user_range_selected.connect(self._on_user_range_selected)
        self.view.axis_lock_hint_changed.connect(self._on_axis_lock_hint_changed)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        lay.addWidget(self.view)
        self._reset_button = QToolButton(self)
        self._reset_button.setObjectName("chartResetButton")
        self._set_reset_button_icon()
        self._reset_button.setIconSize(QSize(12, 12))
        self._reset_button.setAutoRaise(True)
        self._reset_button.setToolTip("Reset chart view")
        self._reset_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_button.clicked.connect(lambda: self._bubble_reset(True, True))
        self._position_reset_button()

        self.chart.setTitle(title)

        self._series_segments: dict[str, list[QLineSeries]] = {}
        self._legend_series: dict[str, QLineSeries] = {}
        self._legend_order: list[str] = []
        self._series_colors: dict[str, QColor] = {}
        self._explicit_series_colors: dict[str, QColor] = {}
        self._feature_order: list[str] = []
        self._max_series: int = MAX_FEATURES_SHOWN_LEGEND
        self._gap_threshold_ms: int | None = None
        self._render_points_per_pixel: float = 1.0
        self._render_points_min: int = 350
        self._ms_epoch_cache: dict[int, int] = {}
        self._ms_epoch_cache_limit: int = 200_000
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
        self._group_timeline_max_runs: int = 12_000
        self._group_timeline_min_box_width_px: float = 2.0
        self._group_timeline_min_box_height_px: float = 1.5
        self._axis_lock_hint_bg_item: QGraphicsRectItem | None = None
        self._axis_lock_hint_text_item: QGraphicsSimpleTextItem | None = None
        self._deferred_refresh_pending: bool = False
        self._axis_x_locked: bool = False
        self._axis_y_locked: bool = False
        self._delegate_x_reset_to_controller: bool = False
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
    # @ai(gpt-5, codex, ui-enhancement, 2026-03-03)
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_reset_button()

    # @ai(gpt-5, codex, ui-enhancement, 2026-03-03)
    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._position_reset_button()
        try:
            self._reset_button.setVisible(True)
            self._reset_button.raise_()
        except Exception:
            logger.warning("Failed to refresh reset button visibility when showing time-series chart.", exc_info=True)

    def _position_reset_button(self) -> None:
        button = getattr(self, "_reset_button", None)
        if button is None:
            return
        margin = 24
        x = self.width() - button.width() - margin
        y = margin
        x = max(margin, x)
        y = max(margin, y)
        button.move(x, y)
        button.raise_()

    def _set_reset_button_icon(self) -> None:
        button = getattr(self, "_reset_button", None)
        if button is None:
            return
        try:
            color = muted_icon_color(self.palette())
            button.setIcon(qta.icon("fa5s.home", color=color))
        except Exception:
            button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirHomeIcon))

    # @ai(gpt-5, codex, refactor, 2026-02-27)
    def clear(
        self,
        *,
        reset_axes: bool = True,
        request_repaint: bool = True,
        refresh_legend: bool = True,
    ):
        self._clear_group_timeline_overlays()
        self._clear_axis_lock_hint_overlay()
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
        if refresh_legend:
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

    def _estimate_gap_threshold_ms(self, times_ms: np.ndarray) -> int | None:
        """
        Estimate a stable, robust gap threshold from timestamp deltas.

        The threshold is intentionally smoothed across updates to avoid visible
        gap flicker while panning/zooming through nearby ranges.
        """
        if times_ms is None:
            return self._gap_threshold_ms
        if not TIMESERIES_GAP_DETECTION_ENABLED:
            return None
        # Internal stability/sensitivity constants. Keep these local so only
        # the primary multipliers remain public in chart globals.
        regularity_ratio = 3.8
        regular_mad_multiplier = 16.0
        irregular_mad_multiplier = 24.0
        reset_low_ratio = 0.08
        reset_high_ratio = 12.0
        bound_low_ratio = 0.65
        bound_high_ratio = 1.75
        smooth_prev_weight = 0.95
        smooth_curr_weight = 0.05
        try:
            ts = np.asarray(times_ms, dtype=np.int64)
        except Exception:
            return self._gap_threshold_ms
        if ts.size < 3:
            return self._gap_threshold_ms

        deltas = np.diff(ts)
        deltas = deltas[deltas > 0]
        if deltas.size == 0:
            return self._gap_threshold_ms

        median_step = float(np.median(deltas))
        if not np.isfinite(median_step) or median_step <= 0:
            return self._gap_threshold_ms

        p80 = float(np.quantile(deltas, 0.80))
        core = deltas[deltas <= p80]
        if core.size == 0:
            core = deltas

        core_median = float(np.median(core))
        if not np.isfinite(core_median) or core_median <= 0:
            core_median = median_step

        mad = float(np.median(np.abs(core - core_median))) if core.size else 0.0
        p90 = float(np.quantile(deltas, 0.90))
        p95 = float(np.quantile(deltas, 0.95))
        regular = p90 <= (core_median * float(regularity_ratio))

        if regular:
            candidate = max(
                core_median * float(TIMESERIES_GAP_REGULAR_MULTIPLIER),
                core_median + max(1.0, float(regular_mad_multiplier) * mad),
            )
        else:
            candidate = max(
                core_median * float(TIMESERIES_GAP_IRREGULAR_MULTIPLIER),
                core_median + max(1.0, float(irregular_mad_multiplier) * mad),
            )
        # Forgiving guard for mixed/coarse cadences:
        # keep threshold near upper-typical spacing so same-day fragments do not
        # become separate segments in month-like timelines.
        candidate = max(candidate, p95 * 0.80)

        if self._gap_threshold_ms is None:
            self._gap_threshold_ms = int(max(1.0, round(candidate)))
            return self._gap_threshold_ms

        prev = float(self._gap_threshold_ms)
        if (
            candidate >= prev * float(reset_high_ratio)
            or candidate <= prev * float(reset_low_ratio)
        ):
            self._gap_threshold_ms = int(max(1.0, round(candidate)))
            return self._gap_threshold_ms
        bounded = min(
            max(candidate, prev * float(bound_low_ratio)),
            prev * float(bound_high_ratio),
        )
        smoothed = (
            float(smooth_prev_weight) * prev
            + float(smooth_curr_weight) * bounded
        )
        self._gap_threshold_ms = int(max(1.0, round(smoothed)))
        return self._gap_threshold_ms

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

    def _target_render_points_per_series(self) -> int:
        width_px = 0
        try:
            width_px = int(self.chart.plotArea().width())
        except Exception:
            width_px = 0
        if width_px <= 0:
            try:
                width_px = int(self.view.viewport().width())
            except Exception:
                width_px = 0
        if width_px <= 0:
            width_px = 1200
        target = int(max(int(self._render_points_min), width_px * float(self._render_points_per_pixel)))
        return max(200, target)

    def _to_ms_epoch_array_cached(self, t_arr) -> np.ndarray:
        ts = pd.to_datetime(list(t_arr), errors="coerce")
        out: list[int] = []
        cache = self._ms_epoch_cache
        cache_limit = int(max(10_000, self._ms_epoch_cache_limit))
        for t in ts:
            if pd.isna(t):
                continue
            tt = pd.Timestamp(t)
            key = int(tt.value)
            ms = cache.get(key)
            if ms is None:
                qdt = QDateTime(
                    QDate(tt.year, tt.month, tt.day),
                    QTime(tt.hour, tt.minute, tt.second, int(tt.microsecond / 1000)),
                )
                ms = int(qdt.toMSecsSinceEpoch())
                if len(cache) >= cache_limit:
                    cache.clear()
                cache[key] = ms
            out.append(ms)
        return np.asarray(out, dtype=np.int64)

    @staticmethod
    def _sample_run_indices(start_idx: int, end_idx: int, target_points: int) -> np.ndarray:
        run_len = max(0, int(end_idx) - int(start_idx))
        if run_len <= 0:
            return np.array([], dtype=np.int64)
        if run_len <= max(2, int(target_points)):
            return np.arange(int(start_idx), int(end_idx), dtype=np.int64)
        sampled = np.linspace(
            int(start_idx),
            int(end_idx) - 1,
            num=max(2, int(target_points)),
            dtype=np.int64,
        )
        sampled = np.unique(sampled)
        if sampled[0] != int(start_idx):
            sampled = np.insert(sampled, 0, int(start_idx))
        last_idx = int(end_idx) - 1
        if sampled[-1] != last_idx:
            sampled = np.append(sampled, last_idx)
        return sampled

    # @ai(gpt-5, codex, fix, 2026-03-20)
    def set_dataframe(
        self,
        frame: pd.DataFrame | None,
        *,
        series_colors: dict[str, QColor] | None = None,
        refresh_legend: bool = True,
    ):
        updates_enabled = self.view.updatesEnabled()
        self.view.setUpdatesEnabled(False)
        try:
            self._preserve_series_colors = False
            self._group_hover_specs = []
            self._explicit_series_colors = {
                str(name): QColor(color)
                for name, color in (series_colors or {}).items()
                if color is not None
            }
            if frame is None:
                self.clear(reset_axes=False, request_repaint=False, refresh_legend=False)
                return

            df = frame.copy()
            if "t" not in df.columns:
                self.clear(reset_axes=False, request_repaint=False, refresh_legend=False)
                return

            feature_cols = [c for c in df.columns if c != "t"][: self._max_series]
            if not feature_cols:
                self.clear(reset_axes=False, request_repaint=False, refresh_legend=False)
                return

            # Keep whatever dtype/timezone `t` already has; just parse if needed.
            if df.empty:
                # Preserve selected-feature legend rows even when the current
                # pan/zoom window has no data rows.
                df = pd.DataFrame(columns=["t"] + feature_cols)
            else:
                t_series = df["t"]
                if not pd.api.types.is_datetime64_any_dtype(t_series):
                    df["t"] = pd.to_datetime(t_series, errors="coerce")
                if bool(df["t"].isna().any()):
                    df = df.dropna(subset=["t"])
                if not df.empty and not bool(df["t"].is_monotonic_increasing):
                    df = df.sort_values("t")
            if df.empty:
                # If all timestamps were invalid/filtered out, keep stable
                # feature legend markers using empty per-feature series.
                df = pd.DataFrame(columns=["t"] + feature_cols)
            else:
                df = df[["t"] + feature_cols]

            can_reuse_segments = (
                bool(self._series_segments)
                and not self._group_box_specs
                and list(self._feature_order) == list(feature_cols)
            )
            if not can_reuse_segments:
                # Keep current axis state during regular data refreshes to avoid
                # intermediate 0..1 axis flicker before new ranges are applied.
                self.clear(reset_axes=False, request_repaint=False, refresh_legend=False)
            else:
                self._clear_group_timeline_overlays()
                self._clear_axis_lock_hint_overlay()
                self._group_box_specs = []
                self._group_hover_specs = []
                self._hover_group_points = []
                self._hover_group_ms = []

            self._current_frame = df
            self._feature_order = feature_cols

            colors = self._generate_colors(len(feature_cols))
            self._series_colors = {
                name: QColor(self._explicit_series_colors.get(name, color))
                for name, color in zip(feature_cols, colors)
            }
            self._preserve_series_colors = bool(self._explicit_series_colors)
            self._rebuild_feature_legend_series(feature_cols)

            # raw times
            times_ms_all = self._to_ms_epoch_array_cached(df["t"])
            gap_threshold_ms = self._estimate_gap_threshold_ms(times_ms_all)

            ymin: float | None = None
            ymax: float | None = None
            new_segments: dict[str, list[QLineSeries]] = {}
            for name, color in zip(feature_cols, colors):
                color = QColor(self._series_colors.get(name, color))
                try:
                    values_arr = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
                except Exception:
                    values_arr = df[name].to_numpy()
                try:
                    finite_values = np.asarray(values_arr, dtype=float)
                    finite_values = finite_values[np.isfinite(finite_values)]
                except Exception:
                    finite_values = np.array([], dtype=float)
                if finite_values.size:
                    local_min = float(np.nanmin(finite_values))
                    local_max = float(np.nanmax(finite_values))
                    ymin = local_min if ymin is None else min(ymin, local_min)
                    ymax = local_max if ymax is None else max(ymax, local_max)

                existing_segments = self._series_segments.get(name) if can_reuse_segments else None
                segments = self._build_segments_for_feature(
                    name,
                    times_ms_all,
                    values_arr,
                    color,
                    existing_segments=existing_segments,
                    gap_threshold_ms=gap_threshold_ms,
                    touch_markers=False,
                    show_name_on_primary=False,
                )
                if segments:
                    new_segments[name] = segments

            if can_reuse_segments:
                for old_name, segments in list(self._series_segments.items()):
                    if old_name in new_segments:
                        continue
                    for series in segments:
                        try:
                            self.chart.removeSeries(series)
                        except Exception:
                            logger.warning("Failed to remove a stale time-series segment during in-place update.", exc_info=True)
            self._series_segments = new_segments

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
            self._y_axis_title_base = "Value"
            self.axis_y.setTitleText(self._y_axis_title_base)
            if ymin is None or ymax is None:
                ymin, ymax = 0.0, 1.0
            if ymin == ymax:
                ymin -= 0.5; ymax += 0.5
            self.axis_y.setRange(ymin, ymax)
            self._data_y_range = (float(ymin), float(ymax))

            # Refresh legend markers once after all series are built.
            if refresh_legend:
                self._refresh_legend_markers()
            self._request_deferred_refresh()
        finally:
            self.view.setUpdatesEnabled(updates_enabled)
            try:
                self.chart.update()
            except Exception:
                logger.warning("Failed to refresh chart scene after time-series dataframe update.", exc_info=True)
            try:
                self.view.viewport().update()
            except Exception:
                logger.warning("Failed to refresh chart viewport after time-series dataframe update.", exc_info=True)

    # @ai(gpt-5, codex, fix, 2026-03-20)
    def update_window_dataframe(
        self,
        frame: pd.DataFrame | None,
        *,
        series_colors: dict[str, QColor] | None = None,
        preserve_x_range: bool = False,
        preserve_y_range: bool = False,
    ) -> bool:
        """Update only time-windowed series data while keeping legend structure intact."""
        if frame is None:
            self.set_dataframe(frame, series_colors=series_colors, refresh_legend=True)
            return False
        try:
            probe = frame
            if "t" not in probe.columns:
                self.set_dataframe(frame, series_colors=series_colors, refresh_legend=True)
                return False
            incoming = [c for c in probe.columns if c != "t"][: self._max_series]
            current = list(self._feature_order)
        except Exception:
            self.set_dataframe(frame, series_colors=series_colors, refresh_legend=True)
            return False
        if not incoming or not current:
            self.set_dataframe(frame, series_colors=series_colors, refresh_legend=True)
            return False
        if set(incoming) != set(current):
            self.set_dataframe(frame, series_colors=series_colors, refresh_legend=True)
            return False
        if not self._series_segments:
            self.set_dataframe(frame, series_colors=series_colors, refresh_legend=True)
            return False
        try:
            df = frame.copy()
            t_series = df["t"]
            if not pd.api.types.is_datetime64_any_dtype(t_series):
                df["t"] = pd.to_datetime(t_series, errors="coerce")
            if bool(df["t"].isna().any()):
                df = df.dropna(subset=["t"])
            if not df.empty and not bool(df["t"].is_monotonic_increasing):
                df = df.sort_values("t")
        except Exception:
            self.set_dataframe(frame, series_colors=series_colors, refresh_legend=True)
            return False
        if df.empty:
            self.set_dataframe(frame, series_colors=series_colors, refresh_legend=True)
            return False

        feature_cols = [name for name in current if name in df.columns]
        if len(feature_cols) != len(current):
            self.set_dataframe(frame, series_colors=series_colors, refresh_legend=True)
            return False
        df = df[["t"] + feature_cols]

        updates_enabled = self.view.updatesEnabled()
        self.view.setUpdatesEnabled(False)
        try:
            self._preserve_series_colors = False
            self._group_hover_specs = []
            self._clear_group_timeline_overlays()
            self._clear_axis_lock_hint_overlay()
            self._group_box_specs = []
            self._hover_group_points = []
            self._hover_group_ms = []
            self._explicit_series_colors = {
                str(name): QColor(color)
                for name, color in (series_colors or {}).items()
                if color is not None
            }

            self._current_frame = df
            colors = self._generate_colors(len(feature_cols))
            self._series_colors = {
                name: QColor(self._explicit_series_colors.get(name, color))
                for name, color in zip(feature_cols, colors)
            }
            self._preserve_series_colors = bool(self._explicit_series_colors)
            self._rebuild_feature_legend_series(feature_cols)

            times_ms_all = self._to_ms_epoch_array_cached(df["t"])
            if times_ms_all.size == 0:
                self.set_dataframe(frame, series_colors=series_colors, refresh_legend=True)
                return False
            # Keep a stable gap threshold while panning/zooming window slices so
            # gap rendering does not shift near sparse edges.
            gap_threshold_ms = self._gap_threshold_ms
            if gap_threshold_ms is None:
                gap_threshold_ms = self._estimate_gap_threshold_ms(times_ms_all)

            ymin: float | None = None
            ymax: float | None = None
            new_segments: dict[str, list[QLineSeries]] = {}
            for name, color in zip(feature_cols, colors):
                segments = list(self._series_segments.get(name) or [])
                if not segments:
                    self.set_dataframe(frame, series_colors=series_colors, refresh_legend=True)
                    return False
                color = QColor(self._series_colors.get(name, color))
                try:
                    values_arr = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
                except Exception:
                    values_arr = np.asarray(df[name].to_numpy(), dtype=float)
                finite_values = values_arr[np.isfinite(values_arr)]
                if finite_values.size:
                    local_min = float(np.nanmin(finite_values))
                    local_max = float(np.nanmax(finite_values))
                    ymin = local_min if ymin is None else min(ymin, local_min)
                    ymax = local_max if ymax is None else max(ymax, local_max)
                built_segments = self._build_segments_for_feature(
                    name,
                    times_ms_all,
                    values_arr,
                    color,
                    existing_segments=segments,
                    gap_threshold_ms=gap_threshold_ms,
                    touch_markers=False,
                    show_name_on_primary=False,
                )
                if not built_segments:
                    self.set_dataframe(frame, series_colors=series_colors, refresh_legend=True)
                    return False
                new_segments[name] = built_segments
            self._series_segments = new_segments

            ms_min = int(times_ms_all.min())
            ms_max = int(times_ms_all.max())
            if ms_min == ms_max:
                ms_max += 1
            qmin = QDateTime.fromMSecsSinceEpoch(ms_min)
            qmax = QDateTime.fromMSecsSinceEpoch(ms_max)
            self._data_x_range = (QDateTime(qmin), QDateTime(qmax))
            if not preserve_x_range:
                self.axis_x.setRange(qmin, qmax)
                try:
                    self.axis_x.setLabelsVisible(True)
                except Exception:
                    logger.warning("Failed to show X-axis labels after windowed time-series data update.", exc_info=True)
                try:
                    self._set_x_axis_format_for_span_ms(ms_max - ms_min)
                except Exception:
                    logger.warning("Failed to format X-axis labels for windowed time-series span.", exc_info=True)

            if not preserve_y_range:
                try:
                    self.axis_y.setLabelsVisible(True)
                except Exception:
                    logger.warning("Failed to show Y-axis labels after windowed time-series data update.", exc_info=True)
                try:
                    self.axis_y.setTickType(QValueAxis.TickType.TicksFixed)
                    self.axis_y.setTickCount(6)
                    self.axis_y.setLabelFormat("%.3f")
                except Exception:
                    logger.warning("Failed to restore numeric Y-axis tick settings for windowed time-series data.", exc_info=True)
                self._y_axis_title_base = "Value"
                self.axis_y.setTitleText(self._y_axis_title_base)
            if ymin is None or ymax is None:
                ymin, ymax = 0.0, 1.0
            if ymin == ymax:
                ymin -= 0.5
                ymax += 0.5
            self._data_y_range = (float(ymin), float(ymax))
            if not preserve_y_range:
                self.axis_y.setRange(ymin, ymax)
            self._refresh_legend_markers()
            self._request_deferred_refresh()
        finally:
            self.view.setUpdatesEnabled(updates_enabled)
            try:
                self.chart.update()
            except Exception:
                logger.warning("Failed to refresh chart scene after windowed time-series update.", exc_info=True)
            try:
                self.view.viewport().update()
            except Exception:
                logger.warning("Failed to refresh chart viewport after windowed time-series update.", exc_info=True)
        return True

    # @ai(gpt-5, codex, fix, 2026-03-20)
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
        updates_enabled = self.view.updatesEnabled()
        self.view.setUpdatesEnabled(False)
        try:
            self.clear(reset_axes=False, request_repaint=False, refresh_legend=False)
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

            times_ms = self._to_ms_epoch_array_cached(df[time_col])
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

            max_runs = int(max(100, self._group_timeline_max_runs))
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
                color = group_color_for_label(group, dark_theme=self._dark_theme)
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
                overlay_times_ms = self._to_ms_epoch_array_cached(overlay_frame[time_col])
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
            self._y_axis_title_base = "Cluster"
            self.axis_y.setTitleText(self._y_axis_title_base)
            self._data_y_range = (float(y_min), float(y_max))

            self._group_box_specs = []
            self._group_hover_specs = []
            for run in runs:
                label, key, _color = group_meta.get(
                    float(run.group),
                    (
                        f"Cluster {int(run.group)}",
                        f"group:{int(run.group)}",
                        group_color_for_label(run.group, dark_theme=self._dark_theme),
                    ),
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
        finally:
            self.view.setUpdatesEnabled(updates_enabled)
            try:
                self.chart.update()
            except Exception:
                logger.warning("Failed to refresh chart scene after group timeline update.", exc_info=True)
            try:
                self.view.viewport().update()
            except Exception:
                logger.warning("Failed to refresh chart viewport after group timeline update.", exc_info=True)

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

    # @ai(gpt-5, codex, fix, 2026-03-20)
    def _bubble_reset(self, reset_x: bool = True, reset_y: bool = True):
        delegate_to_controller = bool(reset_x and self._delegate_x_reset_to_controller)
        if not delegate_to_controller:
            self._reset_to_data_bounds(reset_x=reset_x, reset_y=reset_y)
        # Only emit when X was reset because connected tabs use this signal to
        # recompute date-range filters and reload data windows.
        if reset_x:
            try:
                self.reset_requested.emit()
            except Exception:
                logger.warning("Failed to emit reset request from time-series chart reset action.", exc_info=True)

    # @ai(gpt-5, codex, fix, 2026-03-20)
    def set_delegate_x_reset_to_controller(self, enabled: bool) -> None:
        """Let parent controller handle X-axis reset/reload without local pre-reset."""
        self._delegate_x_reset_to_controller = bool(enabled)

    # @ai(gpt-5, codex, fix, 2026-03-20)
    def set_live_pan_emit_enabled(self, enabled: bool) -> None:
        """Control whether drag-pan emits range updates continuously or on release only."""
        self.view.set_live_pan_emit_enabled(enabled)

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
        for series in self._legend_series.values():
            try:
                self.chart.removeSeries(series)
            except Exception:
                logger.warning("Failed to remove a legend anchor series while clearing time-series data.", exc_info=True)
        self._series_segments.clear()
        self._legend_series.clear()
        self._legend_order = []
        self._series_colors.clear()

    def _rebuild_feature_legend_series(self, feature_names: list[str]) -> None:
        desired = [str(name) for name in (feature_names or [])]
        if desired == self._legend_order and all(name in self._legend_series for name in desired):
            for name in desired:
                series = self._legend_series.get(name)
                if series is None:
                    continue
                self._set_series_pen(series, self._series_colors.get(name, QColor(120, 120, 120)))
            return

        for series in self._legend_series.values():
            try:
                self.chart.removeSeries(series)
            except Exception:
                logger.warning("Failed to remove stale time-series legend anchor series.", exc_info=True)
        self._legend_series = {}
        self._legend_order = []

        for name in desired:
            series = QLineSeries()
            series.setUseOpenGL(False)
            series.setName(name)
            series.replace([])
            self.chart.addSeries(series)
            series.attachAxis(self.axis_x)
            series.attachAxis(self.axis_y)
            self._set_series_pen(series, self._series_colors.get(name, QColor(120, 120, 120)))
            self._legend_series[name] = series
            self._legend_order.append(name)

    # @ai(gpt-5, codex, fix, 2026-03-03)
    def _build_segments_for_feature(
        self,
        name: str,
        times_ms: np.ndarray,
        values: np.ndarray,
        color: QColor,
        *,
        existing_segments: list[QLineSeries] | None = None,
        gap_threshold_ms: int | None = None,
        touch_markers: bool = True,
        show_name_on_primary: bool = True,
    ) -> list[QLineSeries]:
        """
        Build QLineSeries segments for runs of finite values. `times_ms` and
        `values` are numpy arrays aligned by index (same length). Missing
        timestamps should have been removed already; this function assumes
        times_ms contains valid int64 ms values.
        """
        existing = list(existing_segments or [])
        segments: list[QLineSeries] = []

        if times_ms is None or len(times_ms) == 0:
            target_count = 1
            for seg_idx in range(target_count):
                if seg_idx < len(existing):
                    series = existing[seg_idx]
                else:
                    series = QLineSeries()
                    series.setUseOpenGL(False)
                    self.chart.addSeries(series)
                    series.attachAxis(self.axis_x)
                    series.attachAxis(self.axis_y)
                series.setName(name if (show_name_on_primary and seg_idx == 0) else "")
                series.replace([])
                self._set_series_pen(series, color)
                if touch_markers:
                    self._set_series_marker_visibility(
                        series,
                        visible=(seg_idx == 0),
                        label=_short_label(name),
                    )
                segments.append(series)
            for series in existing[target_count:]:
                try:
                    self.chart.removeSeries(series)
                except Exception:
                    logger.warning("Failed to remove excess empty line-series segment during in-place update.", exc_info=True)
            return segments

        # Ensure values is a mutable float numpy array; some upstream arrays
        # can be read-only views (e.g., from pandas), which breaks gap fill.
        vals = np.array(values, dtype=float, copy=True)
        finite = np.isfinite(vals)

        # If there are no finite values, return an empty series (legend preserved)
        if not finite.any():
            target_count = 1
            for seg_idx in range(target_count):
                if seg_idx < len(existing):
                    series = existing[seg_idx]
                else:
                    series = QLineSeries()
                    series.setUseOpenGL(False)
                    self.chart.addSeries(series)
                    series.attachAxis(self.axis_x)
                    series.attachAxis(self.axis_y)
                series.setName(name if (show_name_on_primary and seg_idx == 0) else "")
                series.replace([])
                self._set_series_pen(series, color)
                if touch_markers:
                    self._set_series_marker_visibility(
                        series,
                        visible=(seg_idx == 0),
                        label=_short_label(name),
                    )
                segments.append(series)
            for series in existing[target_count:]:
                try:
                    self.chart.removeSeries(series)
                except Exception:
                    logger.warning("Failed to remove excess NaN-only line-series segment during in-place update.", exc_info=True)
            return segments

        # Keep NaN/inf gaps as hard breaks. Also detect large timestamp jumps
        # and split lines there to make missing periods visually explicit.
        resolved_gap_threshold_ms = (
            int(gap_threshold_ms)
            if gap_threshold_ms is not None
            else self._estimate_gap_threshold_ms(times_ms)
        )
        time_gap_breaks = np.zeros(max(0, len(times_ms) - 1), dtype=bool)
        if TIMESERIES_GAP_DETECTION_ENABLED and resolved_gap_threshold_ms is not None and len(times_ms) > 1:
            deltas = np.diff(np.asarray(times_ms, dtype=np.int64))
            time_gap_breaks = deltas > int(resolved_gap_threshold_ms)
        runs: list[tuple[int, int]] = []
        start_idx: int | None = None
        last_idx = len(vals) - 1
        for idx in range(len(vals)):
            if not finite[idx]:
                if start_idx is not None:
                    runs.append((start_idx, idx))
                    start_idx = None
                continue
            if start_idx is None:
                start_idx = idx
            is_end = idx >= last_idx
            next_non_finite = (not is_end) and (not finite[idx + 1])
            has_large_time_gap = (not is_end) and bool(time_gap_breaks[idx])
            if is_end or next_non_finite or has_large_time_gap:
                runs.append((start_idx, idx + 1))
                start_idx = None

        # Create/update QLineSeries for each run
        target_count = max(1, len(runs))
        total_run_points = int(sum(max(0, end - start) for start, end in runs))
        target_points_per_series = self._target_render_points_per_series()
        for seg_idx in range(target_count):
            if seg_idx < len(existing):
                series = existing[seg_idx]
            else:
                series = QLineSeries()
                series.setUseOpenGL(False)
                self.chart.addSeries(series)
                series.attachAxis(self.axis_x)
                series.attachAxis(self.axis_y)
            series.setName(name if (show_name_on_primary and seg_idx == 0) else "")
            if seg_idx < len(runs):
                start_idx, end_idx = runs[seg_idx]
                run_len = max(0, int(end_idx) - int(start_idx))
                if total_run_points > target_points_per_series and run_len > 2:
                    run_target = int(np.ceil((run_len / max(1, total_run_points)) * target_points_per_series))
                    run_target = max(2, min(run_len, run_target))
                    run_indices = self._sample_run_indices(start_idx, end_idx, run_target)
                else:
                    run_indices = np.arange(int(start_idx), int(end_idx), dtype=np.int64)
                x_vals = times_ms[run_indices]
                y_vals = vals[run_indices]
                pts = [QPointF(float(x), float(y)) for x, y in zip(x_vals, y_vals)]
            else:
                pts = []
            series.replace(pts)
            self._set_series_pen(series, color)
            if touch_markers:
                self._set_series_marker_visibility(
                    series,
                    visible=(seg_idx == 0),
                    label=_short_label(name),
                )
            segments.append(series)

        for series in existing[target_count:]:
            try:
                self.chart.removeSeries(series)
            except Exception:
                logger.warning("Failed to remove excess line-series segment during in-place update.", exc_info=True)

        return segments

    def _set_series_pen(self, series: QLineSeries, color: QColor):
        pen = QPen(color)
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        series.setPen(pen)

    def _set_series_marker_visibility(
        self,
        series: QLineSeries,
        *,
        visible: bool,
        label: str | None = None,
    ) -> None:
        legend = self.chart.legend()
        try:
            markers = list(legend.markers(series))
        except Exception:
            markers = []
        for marker in markers:
            if visible and label is not None:
                try:
                    if marker.label() != label:
                        marker.setLabel(label)
                except Exception:
                    logger.warning("Failed to set time-series legend marker label.", exc_info=True)
            try:
                current = bool(marker.isVisible())
            except Exception:
                current = (not visible)
            if current == bool(visible):
                continue
            try:
                marker.setVisible(bool(visible))
            except Exception:
                logger.warning("Failed to set time-series legend marker visibility.", exc_info=True)

    def _generate_colors(self, count: int) -> list[QColor]:
        return group_color_cycle(count, dark_theme=self._dark_theme)

    def _on_theme_changed(self, theme_name: str):
        self._current_theme = theme_name
        self._apply_theme(theme_name)

    # @ai(gpt-5, codex, ui-enhancement, 2026-03-03)
    def _on_axis_lock_hint_changed(self, x_locked: bool, y_locked: bool) -> None:
        self._axis_x_locked = bool(x_locked)
        self._axis_y_locked = bool(y_locked)
        self._apply_axis_lock_visuals()
        self._refresh_axis_lock_hint_overlay()
        try:
            self.view.viewport().update()
        except Exception:
            logger.warning("Failed to refresh viewport after axis-lock visual update.", exc_info=True)

    def _apply_axis_lock_visuals(self) -> None:
        if self._chart_colors is None:
            return

        def _apply(axis, *, locked: bool, base_title: str) -> None:
            text = QColor(self._chart_colors.text)
            axis_line = QColor(self._chart_colors.axis_line)
            grid = QColor(self._chart_colors.grid)
            if locked:
                text.setAlpha(max(85, int(text.alpha() * 0.60)))
                axis_line.setAlpha(max(70, int(axis_line.alpha() * 0.60)))
                grid.setAlpha(max(35, int(grid.alpha() * 0.45)))

            title = f"{base_title} (Locked)" if locked else base_title
            try:
                axis.setTitleText(title)
            except Exception:
                logger.warning("Failed to set axis title for lock visual state.", exc_info=True)
            try:
                axis.setLabelsBrush(QBrush(text))
            except Exception:
                logger.warning("Failed to set axis labels brush for lock visual state.", exc_info=True)
            try:
                axis.setTitleBrush(QBrush(text))
            except Exception:
                logger.warning("Failed to set axis title brush for lock visual state.", exc_info=True)
            try:
                line_pen = QPen(axis_line, 1.0)
                line_pen.setCosmetic(True)
                axis.setLinePen(line_pen)
            except Exception:
                logger.warning("Failed to set axis line pen for lock visual state.", exc_info=True)
            try:
                axis.setGridLineColor(grid)
                axis.setMinorGridLineColor(grid.darker(110))
            except Exception:
                logger.warning("Failed to set axis grid colors for lock visual state.", exc_info=True)

        _apply(self.axis_x, locked=self._axis_x_locked, base_title=self._x_axis_title_base)
        _apply(self.axis_y, locked=self._axis_y_locked, base_title=self._y_axis_title_base)

    def _clear_axis_lock_hint_overlay(self) -> None:
        scene = self.chart.scene()
        if self._axis_lock_hint_bg_item is not None:
            try:
                if scene is not None:
                    scene.removeItem(self._axis_lock_hint_bg_item)
            except Exception:
                logger.warning("Failed to remove axis-lock hint background item.", exc_info=True)
            self._axis_lock_hint_bg_item = None
        if self._axis_lock_hint_text_item is not None:
            try:
                if scene is not None:
                    scene.removeItem(self._axis_lock_hint_text_item)
            except Exception:
                logger.warning("Failed to remove axis-lock hint text item.", exc_info=True)
            self._axis_lock_hint_text_item = None

    def _refresh_axis_lock_hint_overlay(self) -> None:
        self._clear_axis_lock_hint_overlay()
        if self._chart_colors is None:
            return
        if not (self._axis_x_locked or self._axis_y_locked):
            return

        if self._axis_x_locked and self._axis_y_locked:
            label = "X and Y locked"
        elif self._axis_x_locked:
            label = "X axis locked"
        else:
            label = "Y axis locked"

        scene = self.chart.scene()
        if scene is None:
            return
        plot_area = self.chart.plotArea()
        if plot_area.width() <= 0 or plot_area.height() <= 0:
            return

        text_item = QGraphicsSimpleTextItem(label)
        font = QFont()
        font.setPointSize(8)
        text_item.setFont(font)
        txt = QColor(self._chart_colors.text)
        txt.setAlpha(235)
        text_item.setBrush(QBrush(txt))

        text_rect = text_item.boundingRect()
        x = float(plot_area.left() + 12.0)
        y = float(plot_area.top() + 8.0)
        pad_x = 6.0
        pad_y = 3.0
        bg_rect = QRectF(
            x - pad_x,
            y - pad_y,
            text_rect.width() + pad_x * 2.0,
            text_rect.height() + pad_y * 2.0,
        )

        bg = QColor(self._chart_colors.plot_bg)
        bg.setAlpha(185)
        border = QColor(self._chart_colors.axis_line)
        border.setAlpha(160)

        bg_item = QGraphicsRectItem(bg_rect)
        bg_item.setBrush(QBrush(bg))
        bg_item.setPen(QPen(border, 1.0))
        text_item.setPos(x, y)

        scene.addItem(bg_item)
        scene.addItem(text_item)
        self._axis_lock_hint_bg_item = bg_item
        self._axis_lock_hint_text_item = text_item

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
        self._apply_axis_lock_visuals()
        self._refresh_axis_lock_hint_overlay()

        parent = self.parentWidget()
        if parent is not None:
            container_color = parent.palette().color(QPalette.ColorRole.Window)
        else:
            container_color = self.palette().color(QPalette.ColorRole.Window)

        try:
            self.chart.setTitleBrush(QBrush(colors.text))
        except Exception:
            logger.warning("Failed to set title brush while applying time-series theme.", exc_info=True)
        self._set_reset_button_icon()

        frame_palette = self.palette()
        frame_palette.setColor(QPalette.ColorRole.Window, container_color)
        frame_palette.setColor(QPalette.ColorRole.Base, container_color)
        self.setPalette(frame_palette)
        self.setAutoFillBackground(True)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        except Exception:
            logger.warning("Failed to set styled-background attribute while applying time-series theme.", exc_info=True)

        view_palette = self.view.palette()
        view_palette.setColor(QPalette.ColorRole.Window, container_color)
        view_palette.setColor(QPalette.ColorRole.Base, container_color)
        self.view.setPalette(view_palette)
        self.view.setAutoFillBackground(True)
        try:
            self.view.setBackgroundBrush(QBrush(container_color))
        except Exception:
            logger.warning("Failed to set chart-view background brush while applying time-series theme.", exc_info=True)

        viewport_palette = self.view.viewport().palette()
        viewport_palette.setColor(QPalette.ColorRole.Window, container_color)
        viewport_palette.setColor(QPalette.ColorRole.Base, container_color)
        self.view.viewport().setPalette(viewport_palette)
        self.view.viewport().setAutoFillBackground(True)
        try:
            self.view.viewport().setBackgroundRole(QPalette.ColorRole.Window)
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
        elif self._feature_order and self._explicit_series_colors:
            regenerated = self._generate_colors(len(self._feature_order))
            self._series_colors = {
                name: QColor(self._explicit_series_colors.get(name, color))
                for name, color in zip(self._feature_order, regenerated)
            }

        if not self._series_segments and not self._legend_series:
            return

        for name, segments in self._series_segments.items():
            color = self._series_colors.get(name, colors.line_primary)
            for series in segments:
                self._set_series_pen(series, color)
        for name, series in self._legend_series.items():
            self._set_series_pen(series, self._series_colors.get(name, colors.line_primary))

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

        if self._legend_series:
            allowed_series_ids = {
                id(series)
                for name, series in self._legend_series.items()
                if name in self._legend_order
            }
            for marker in markers:
                series = marker.series()
                if series is None or id(series) not in allowed_series_ids:
                    marker.setVisible(False)
                    continue
                name = series.name() or ""
                if not name:
                    marker.setVisible(False)
                    continue
                marker.setLabel(_short_label(name))
                marker.setVisible(True)
            if self._chart_colors is not None:
                style_legend(legend, self._chart_colors)
            self._refresh_group_timeline_overlays()
            return

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
            short = _short_label(name)
            marker.setLabel(short)
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

    def _visible_group_box_cap(
        self,
        *,
        plot_width: float,
        plot_height: float,
        lane_count: int,
    ) -> int:
        try:
            width = float(plot_width)
        except Exception:
            width = 0.0
        try:
            height = float(plot_height)
        except Exception:
            height = 0.0
        try:
            lanes = max(1, int(lane_count))
        except Exception:
            lanes = 1

        min_width = max(2.0, float(self._group_timeline_min_box_width_px))
        min_height = max(2.0, float(self._group_timeline_min_box_height_px))

        # Display-aware cap:
        # - horizontally, each distinguishable box needs minimum width
        # - vertically, only lanes with sufficient pixel height can be distinguished
        horizontal_slots = max(1, int(np.floor(width / min_width)))
        visible_lane_capacity = max(1, int(np.floor(height / min_height)))
        effective_lanes = max(1, min(lanes, visible_lane_capacity))
        return max(50, horizontal_slots * effective_lanes)

    def _refresh_group_timeline_overlays(self) -> None:
        self._clear_group_timeline_overlays()
        if not self._group_box_specs:
            self._refresh_axis_lock_hint_overlay()
            return
        if not self.chart.series():
            self._refresh_axis_lock_hint_overlay()
            return
        scene = self.chart.scene()
        if scene is None:
            self._refresh_axis_lock_hint_overlay()
            return
        plot_area = self.chart.plotArea()
        if plot_area.width() <= 0 or plot_area.height() <= 0:
            self._refresh_axis_lock_hint_overlay()
            return
        ref_series = self.chart.series()[0]
        visible_start = float(self.axis_x.min().toMSecsSinceEpoch())
        visible_end = float(self.axis_x.max().toMSecsSinceEpoch())
        if visible_end <= visible_start:
            self._refresh_axis_lock_hint_overlay()
            return
        visible_specs = [
            spec for spec in self._group_box_specs if not (spec[1] < visible_start or spec[0] > visible_end)
        ]
        visible_lane_count = len({str(spec[5]) for spec in visible_specs}) or 1
        max_visible_boxes = self._visible_group_box_cap(
            plot_width=float(plot_area.width()),
            plot_height=float(plot_area.height()),
            lane_count=visible_lane_count,
        )
        if len(visible_specs) > max_visible_boxes:
            stride = int(np.ceil(len(visible_specs) / max_visible_boxes))
            visible_specs = visible_specs[::max(1, stride)]

        font = QFont()
        font.setPointSize(8)
        text_budget = 1_000
        min_box_width_px = max(1.0, float(self._group_timeline_min_box_width_px))
        min_box_height_px = max(1.0, float(self._group_timeline_min_box_height_px))
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
        self._refresh_axis_lock_hint_overlay()

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

