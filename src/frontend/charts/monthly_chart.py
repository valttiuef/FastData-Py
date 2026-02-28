from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
# frontend/data_tab/monthly_chart.py
from typing import Iterable, List
import pandas as pd

from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QTimer
from PySide6.QtGui import QPen, QBrush, QColor, QCursor
from PySide6.QtCharts import QBarSet
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsRectItem,
    QGraphicsLineItem,
    QToolTip,
)

from ..style.chart_theme import style_legend
from ..style.group_colors import group_color_cycle
from core.datetime_utils import drop_timezone_preserving_wall
from . import MAX_FEATURES_SHOWN_LEGEND
from .group_chart import GroupBarChart


def _short_label(name: str, max_len: int = 24) -> str:
    if not name:
        return ""
    try:
        primary = str(name).split(" · ")[0]
    except Exception:
        primary = str(name)
    if len(primary) <= max_len:
        return primary
    return primary[: max_len - 1] + "…"


class MonthlyBarChart(GroupBarChart):
    """Flexible time-aggregate bar chart.

    This chart chooses a sensible base aggregation (prefer monthly, limited to
    max_bars) and allows drilling down to finer granularities by clicking a
    bar (year -> month -> week -> day -> hour). Right-click steps back one level,
    double-click resets to base.

    Signals:
    - bucket_selected(start: pd.Timestamp, end: pd.Timestamp, level: str)
      emitted whenever the user clicks a bucket (also when drilling down).
    """
    bucket_selected = Signal(object, object, str)
    reset_requested = Signal()

    # Supported aggregation levels (code, name, display fmt)
    _LEVELS = [
        ("A", "year", "%Y"),
        ("M", "month", "%Y-%m"),
        ("W", "week", "%Y-W%V"),
        ("D", "day", "%Y-%m-%d"),
        ("h", "hour", "%Y-%m-%d %H:00"),
    ]

    def __init__(self, title: str = "Bar Chart", parent=None, max_bars: int = 31):
        # These are accessed by _apply_theme, which can be invoked from
        # GroupBarChart.__init__ before MonthlyBarChart init finishes.
        self._hover_item = None
        self._zero_line_item = None
        self._multi_mode = False
        super().__init__(title=title, parent=parent, y_label="Value")
        # keep a base title so we can append timeframe info when aggregation changes
        self._base_title = title
        self._append_timeframe_to_title = True

        # Raw data cache
        self._raw_t = []
        self._raw_v = []

        # simple history stack for drill state: list of tuples
        # (level_code, t_series, v_series, parent_start, parent_end)
        self._history = []

        # Current rendered buckets
        self._cats: List[str] = []
        self._vals: List[float] = []
        self._raw_vals: List[float | None] = []
        self._bucket_periods: List[pd.Period] = []
        self._hover_idx = None

        # Multi-feature state
        self._multi_mode: bool = False
        self._raw_multi_df: pd.DataFrame | None = None
        self._multi_base_level: str | None = None
        self._multi_categories: list[str] = []
        self._multi_category_bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        self._multi_current_agg: pd.DataFrame | None = None

        # Visual overlays
        self._hover_item = QGraphicsRectItem(self.chart)
        self._hover_item.setZValue(10_000); self._hover_item.setVisible(False)
        self._zero_line_item = QGraphicsLineItem(self.chart)
        self._zero_line_item.setZValue(5_000); self._zero_line_item.setVisible(False)
        try:
            self.chart.plotAreaChanged.connect(self._update_zero_line)
        except Exception:
            logger.warning("Failed to connect plot-area change handler for monthly zero line.", exc_info=True)

        # Interaction state
        self._max_bars = int(max_bars)
        self._base_level = None  # code like 'M'
        self._current_level = None
        self._single_series_name = "Value"

        # Signals
        self._series_hover_supported = False
        self._pending_click_index: int | None = None
        self._single_click_timer = QTimer(self)
        self._single_click_timer.setSingleShot(True)
        self._single_click_timer.timeout.connect(self._commit_pending_single_click)
        self._connect_hover_signals()
        self.view.mousePressEvent = self._mouse_press_wrapper(self._native_mouse_press_event)
        self.view.mouseDoubleClickEvent = self._mouse_double_click_wrapper(self._native_mouse_double_click_event)
        self._apply_theme(self._current_theme)

    def _apply_theme(self, theme_name: str | None = None):
        super()._apply_theme(theme_name)
        colors = self._chart_colors
        self._set_zero_line_pen(colors)

        if getattr(self, "_multi_mode", False) and self.series.barSets():
            bar_colors = self._multi_bar_colors(len(self.series.barSets()))
            for bar_set, color in zip(self.series.barSets(), bar_colors):
                bar_set.setBrush(QBrush(color))
                bar_set.setPen(QPen(QColor(color).darker(110 if not self._dark_theme else 140)))
            if self._chart_colors is not None:
                style_legend(self.chart.legend(), self._chart_colors)

        if self._hover_item is None or colors is None:
            return

        hover_fill = QColor(colors.bar_fill).lighter(120)
        hover_fill.setAlpha(160)
        hover_border = QColor(colors.bar_border).darker(110)
        self._hover_item.setBrush(QBrush(hover_fill)); self._hover_item.setPen(QPen(hover_border, 1.0))

        # baseline rendering removed — keep hover only

        self.axis_x.setLabelsAngle(-60)

        try:
            scene = self.chart.scene()
            if scene is not None:
                scene.update()
        except Exception:
            self.chart.update()
        self.view.viewport().update()
        self._update_zero_line()

    def _set_zero_line_pen(self, colors):
        if getattr(self, "_zero_line_item", None) is None or colors is None:
            return
        pen = QPen(QColor(colors.grid))
        pen.setCosmetic(True)
        self._zero_line_item.setPen(pen)

    def _update_zero_line(self):
        if self._zero_line_item is None:
            return
        try:
            y_min = float(self.axis_y.min())
            y_max = float(self.axis_y.max())
        except Exception:
            self._zero_line_item.setVisible(False)
            return
        if not (y_min < 0 < y_max):
            self._zero_line_item.setVisible(False)
            return
        plot_area = self.chart.plotArea()
        if not plot_area.isValid():
            self._zero_line_item.setVisible(False)
            return
        span = y_max - y_min
        if span == 0:
            self._zero_line_item.setVisible(False)
            return
        ratio = (0.0 - y_min) / span
        y = plot_area.bottom() - (plot_area.height() * ratio)
        self._zero_line_item.setLine(plot_area.left(), y, plot_area.right(), y)
        self._zero_line_item.setVisible(True)

    # ---------- Public API ----------
    def set_title(self, title: str):
        # remember base title and show it (timeframe will be appended dynamically)
        self._base_title = title
        self.chart.setTitle(title)

    def set_append_timeframe_to_title(self, enabled: bool) -> None:
        self._append_timeframe_to_title = bool(enabled)

    # @ai(gpt-5, codex, refactor, 2026-02-27)
    def clear(
        self,
        *,
        reset_navigation: bool = True,
        reset_axes: bool = True,
        request_repaint: bool = True,
    ):
        for s in list(self.series.barSets()):
            try:
                if s in self._connected_barsets:
                    s.hovered.disconnect(self._on_set_hovered)
                    self._connected_barsets.discard(s)
            except Exception:
                # Some PySide bindings raise a RuntimeWarning when disconnecting
                # slots that aren't connected; we defensively track connections
                # above to avoid calling disconnect unnecessarily.
                logger.warning("Failed to disconnect hover handler while clearing monthly chart bars.", exc_info=True)
            self.series.remove(s)
        self.axis_x.clear()
        if reset_axes:
            self.axis_x.setLabelsVisible(False)
            self.axis_x.setGridLineVisible(False)
            self.axis_x.setMinorGridLineVisible(False)
            self.axis_y.setRange(0.0, 1.0)
            self.chart.setTitle(self._base_title)
        if reset_navigation:
            self._raw_t = []
            self._raw_v = []
            self._history.clear()
            self._base_level = None
            self._current_level = None
        self._cats.clear(); self._vals.clear(); self._raw_vals.clear(); self._bucket_periods.clear()
        self._pending_click_index = None
        if getattr(self, "_single_click_timer", None) is not None and self._single_click_timer.isActive():
            self._single_click_timer.stop()
        self._hover_item.setVisible(False)
        if self._zero_line_item is not None:
            self._zero_line_item.setVisible(False)
        self._multi_categories = []
        self._multi_category_bounds = {}
        self._multi_current_agg = None
        try:
            self.view.setCursor(Qt.ArrowCursor)
        except Exception:
            logger.warning("Failed to reset cursor while clearing monthly chart.", exc_info=True)
        try:
            QToolTip.hideText()
        except Exception:
            logger.warning("Failed to hide tooltip while clearing monthly chart.", exc_info=True)
        if request_repaint:
            try:
                self.chart.update()
            except Exception:
                logger.warning("Failed to refresh chart scene after clearing monthly chart.", exc_info=True)
            try:
                self.view.viewport().update()
            except Exception:
                logger.warning("Failed to refresh chart viewport after clearing monthly chart.", exc_info=True)

    def set_data(self, t_like: Iterable, v_like: Iterable, series_name: str = "Value"):
        """Load raw (already postprocessed) data and choose a sensible base level."""
        self._multi_mode = False
        self._raw_multi_df = None
        self._single_series_name = _short_label(str(series_name or "Value"))
        self.chart.legend().hide()
        t = pd.Series(pd.to_datetime(list(t_like), errors="coerce").astype("datetime64[ns]"))
        v = pd.Series(pd.to_numeric(list(v_like), errors="coerce"))
        self._raw_t = t
        self._raw_v = v
        if t.empty or t.notna().sum() == 0:
            self.clear()
            return

        # Decide base level by span
        level = self._determine_base_level(t)

        self._base_level = level
        self._current_level = level
        self._history.clear()
        # Cap multi-feature rendering at chart-level; here set_data is single-series
        self._history.append((self._base_level, self._raw_t, self._raw_v, None, None))
        self._render_level(self._raw_t, self._raw_v, self._current_level)

    def set_frame(self, frame: pd.DataFrame):
        """Load a dataframe with columns ['t', <feature...>] and render grouped bars."""
        self._multi_mode = True
        self._raw_multi_df = None
        self._multi_current_agg = None
        try:
            df = frame.copy()
        except Exception:
            self.clear()
            return
        if df is None or df.empty or "t" not in df.columns:
            self.clear()
            return
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df = df.dropna(subset=["t"]).sort_values("t")
        feature_cols = [c for c in df.columns if c != "t"]
        # Cap how many features we render in multi-feature mode
        feature_cols = feature_cols[:MAX_FEATURES_SHOWN_LEGEND]
        if not feature_cols:
            self.clear()
            return
        self._raw_multi_df = df[["t"] + feature_cols]
        self._multi_base_level = self._determine_base_level(df["t"])
        self._multi_categories = []
        self._multi_category_bounds = {}
        self._render_multi_level(self._multi_base_level)

    def _determine_base_level(self, t_series: pd.Series) -> str:
        t_valid = t_series[t_series.notna()]
        level = "M"
        if t_valid.empty:
            return level
        nunique_months = t_valid.dt.to_period("M").nunique()
        if nunique_months <= 1:
            nunique_days = t_valid.dt.to_period("D").nunique()
            if nunique_days <= 1:
                if t_valid.dt.to_period("h").nunique() >= 1:
                    return "h"
                return "D"
            return "D"
        return "M"

    def _aggregate(self, level_code: str, t: pd.Series, v: pd.Series):
        """Aggregate timestamps t and values v into buckets for level_code.

        Returns (cats, vals, periods) where periods is a list of pd.Period for bucket starts.
        """
        if t is None or len(t) == 0:
            return [], [], []
        df = pd.DataFrame({"t": t, "v": v})
        df = df[df["t"].notna()]
        if df.empty:
            return [], [], []

        # Convert to period bucket
        try:
            df["period"] = df["t"].dt.to_period(level_code)
        except Exception:
            df["period"] = df["t"].dt.to_period("M")

        agg = df.groupby("period", as_index=False)["v"].mean().sort_values("period")

        start_per = agg["period"].min()
        end_per = agg["period"].max()
        full = pd.period_range(start=start_per, end=end_per, freq=level_code)

        # ---- decide label compactness based on overall span ----
        # We'll compute booleans telling whether the entire range is within one year/month/day.
        try:
            start_ts = start_per.to_timestamp()
            end_ts = end_per.to_timestamp()
        except Exception:
            start_ts = pd.Timestamp(start_per.start_time)
            end_ts = pd.Timestamp(end_per.start_time)

        single_year_span  = (start_ts.year == end_ts.year)
        single_month_span = (single_year_span and start_ts.month == end_ts.month)
        single_day_span   = (single_month_span and start_ts.day == end_ts.day)

        # Build full frame to include empty buckets
        full_df = agg.set_index("period").reindex(full).rename_axis("period").reset_index()

        cats, vals, periods = [], [], []
        for _, row in full_df.iterrows():
            per = row["period"]
            periods.append(per)

            # Convert period to a representative timestamp for formatting
            try:
                ts = per.to_timestamp()
            except Exception:
                ts = pd.Timestamp(getattr(per, "start_time", pd.NaT))

            # ---- label selection rules ----
            try:
                if level_code == "A":
                    label = ts.strftime("%Y")

                elif level_code == "M":
                    # One calendar year? show just month name; otherwise show year+month
                    label = ts.strftime("%b") if single_year_span else ts.strftime("%Y-%m")

                elif level_code == "W":
                    # Keep ISO year-week; you can also compact when single year if desired
                    iso = ts.isocalendar()
                    label = f"{iso.year}-W{int(iso.week):02d}"

                elif level_code == "D":
                    # One calendar month? show just day of month; otherwise full date
                    label = ts.strftime("%d") if single_month_span else ts.strftime("%Y-%m-%d")

                else:  # "h"
                    # One calendar day? show just hour; otherwise full datetime hour
                    label = ts.strftime("%H:00") if single_day_span else ts.strftime("%Y-%m-%d %H:00")
            except Exception:
                label = str(ts)

            cats.append(label)

            val = row.get("v", None)
            vals.append(float(val) if pd.notna(val) else float(0.0))

        return cats, vals, list(periods)

    def _to_naive(self, ts, keep_hour: bool = False) -> pd.Timestamp:
        """Return a timezone-naive Timestamp. If keep_hour is False, normalize to 00:00."""
        raw = pd.Timestamp(ts)
        cleaned = drop_timezone_preserving_wall(raw)
        t = pd.Timestamp(cleaned)
        if not keep_hour:
            t = t.replace(hour=0, minute=0, second=0, microsecond=0)
        return t

    # @ai(gpt-5, codex, refactor, 2026-02-27)
    def _render_level(self, t: pd.Series, v: pd.Series, level_code: str):
        # Aggregate
        cats, vals, periods = self._aggregate(level_code, t, v)
        if not periods:
            self.clear(reset_navigation=False, reset_axes=True, request_repaint=True)
            return
        updates_enabled = self.view.updatesEnabled()
        self.view.setUpdatesEnabled(False)
        try:
            # Keep current axis state while rebuilding bars to avoid a visible
            # intermediate 0..1 axis jump during normal data refreshes.
            self.clear(reset_navigation=False, reset_axes=False, request_repaint=False)
            self._cats = cats
            self._vals = vals
            # raw vals keep None for missing
            self._raw_vals = [None if x == 0.0 and not pd.notna(x) else (float(x) if x is not None else None) for x in vals]
            self._bucket_periods = periods

            # Update title with optional timeframe suffix derived from bucket periods.
            try:
                base_title = str(self._base_title or "").strip()
                if self._append_timeframe_to_title and periods and base_title:
                    timeframe = self._timeframe_for_periods(level_code, periods)
                    self.chart.setTitle(f"{base_title} — {timeframe}")
                else:
                    self.chart.setTitle(base_title)
            except Exception:
                self.chart.setTitle(self._base_title)

            # Build bar set
            bar = QBarSet(self._single_series_name)
            bar.append(self._vals)
            primary = group_color_cycle(1, dark_theme=self._dark_theme)[0]
            bar.setBrush(QBrush(primary))
            bar.setPen(QPen(QColor(primary).darker(110 if not self._dark_theme else 140)))
            self.series.append(bar)
            bar.hovered.connect(self._on_set_hovered)

            self.axis_x.append(self._cats)
            try:
                self.axis_x.setLabelsVisible(True)
                self.axis_x.setGridLineVisible(True)
                self.axis_x.setMinorGridLineVisible(True)
            except Exception:
                logger.warning("Failed to show X-axis labels after rendering monthly bars.", exc_info=True)

            # Y range
            if self._vals:
                vmin = min(self._vals)
                vmax = max(self._vals)
            else:
                vmin, vmax = 0.0, 1.0
            y_min = min(vmin, 0.0); y_max = max(vmax, 0.0)
            if y_max == y_min: y_max = y_min + 1.0
            # small padding
            pad = 0.0
            if vmin >= 0:
                y_min = 0.0
            self.axis_y.setRange(y_min, y_max + pad)
            self._update_zero_line()

            self._connect_hover_signals()
            self._update_highlight()
        finally:
            self.view.setUpdatesEnabled(updates_enabled)
            self.chart.update()
            self.view.viewport().update()

    def set_series(self, t_like: Iterable, v_like: Iterable, series_name: str = "Value"):
        self.set_data(t_like, v_like, series_name=series_name)

    def _render_multi_level(self, level_code: str):
        if self._raw_multi_df is None or self._raw_multi_df.empty:
            self.clear()
            return
        agg = self._aggregate_multi(self._raw_multi_df, level_code)
        self._multi_current_agg = agg
        self._current_level = level_code
        if agg is None or agg.empty:
            self.clear()
            return

        bounds = (
            agg.groupby("label")[["start_ts", "end_ts"]]
            .agg({"start_ts": "min", "end_ts": "max"})
        )
        self._multi_category_bounds = {
            label: (pd.Timestamp(row["start_ts"]), pd.Timestamp(row["end_ts"]))
            for label, row in bounds.iterrows()
        }

        self._render_multi_bars(agg)

    def _aggregate_multi(self, df: pd.DataFrame, level_code: str) -> pd.DataFrame:
        freq_map = {"A": "A", "M": "M", "W": "W", "D": "D", "h": "h"}
        freq = freq_map.get(level_code, "M")
        long_df = df.melt(id_vars="t", var_name="feature", value_name="value")
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
        long_df = long_df.dropna(subset=["value"])
        if long_df.empty:
            return pd.DataFrame(columns=["feature", "label", "value", "start_ts", "end_ts"])

        long_df["period"] = long_df["t"].dt.to_period(freq)
        grouped = long_df.groupby(["period", "feature"], as_index=False)["value"].mean()
        grouped["start_ts"] = grouped["period"].dt.start_time
        grouped["end_ts"] = (grouped["period"] + 1).dt.start_time
        grouped["label"] = grouped["period"].apply(lambda per: self._format_period_label(level_code, per))
        grouped = grouped.sort_values(["start_ts", "feature"])
        return grouped

    # @ai(gpt-5, codex, refactor, 2026-02-27)
    def _render_multi_bars(self, agg: pd.DataFrame):
        updates_enabled = self.view.updatesEnabled()
        self.view.setUpdatesEnabled(False)
        try:
            self.clear(reset_navigation=False, reset_axes=False, request_repaint=False)
            if agg is None or agg.empty:
                self._multi_categories = []
                return

            categories_df = agg[["label", "start_ts"]].drop_duplicates().sort_values("start_ts")
            categories = categories_df["label"].tolist()
            self._multi_categories = categories

            pivot = (
                agg.pivot(index="label", columns="feature", values="value")
                .reindex(categories)
                .fillna(0.0)
            )

            self.axis_x.append(categories)
            try:
                self.axis_x.setLabelsVisible(True)
                self.axis_x.setGridLineVisible(True)
                self.axis_x.setMinorGridLineVisible(True)
            except Exception:
                logger.warning("Failed to show X-axis labels after rendering multi-series monthly bars.", exc_info=True)

            features = list(pivot.columns)
            colors = self._multi_bar_colors(len(features))
            self.chart.legend().setVisible(True)

            for idx, (feature, color) in enumerate(zip(features, colors)):
                bar_set = QBarSet(_short_label(str(feature)))
                values = pivot[feature].tolist()
                bar_set.append(values)
                bar_set.setBrush(QBrush(color))
                bar_set.setPen(QPen(QColor(color).darker(110)))
                # New QBarSet instances are fresh; directly connect the click handler.
                bar_set.clicked.connect(self._on_multi_bar_clicked_factory(feature))
                bar_set.hovered.connect(self._on_multi_bar_hovered_factory(feature, values))
                self.series.append(bar_set)

            if self._chart_colors is not None:
                style_legend(self.chart.legend(), self._chart_colors)

            # Update Y-axis range
            if not pivot.empty:
                vmin = float(pivot.min().min())
                vmax = float(pivot.max().max())
            else:
                vmin = vmax = 0.0
            if vmin == vmax:
                vmax = vmin + 1.0
            self.axis_y.setRange(min(0.0, vmin), max(0.0, vmax))
            self._update_zero_line()

            # disable hover overlay in multi-mode
            self._series_hover_supported = False
            self._hover_item.setVisible(False)
        finally:
            self.view.setUpdatesEnabled(updates_enabled)
            self.chart.update()
            self.view.viewport().update()

    def _on_multi_bar_hovered_factory(self, feature: str, values: list[float]):
        def handler(status: bool, index: int):
            if not status:
                try:
                    self.view.setCursor(Qt.ArrowCursor)
                except Exception:
                    logger.warning("Failed to reset cursor while handling multi-bar hover leave.", exc_info=True)
                QToolTip.hideText()
                return
            if index < 0 or index >= len(self._multi_categories):
                return
            category = self._multi_categories[index]
            value = values[index] if index < len(values) else 0.0
            try:
                self.view.setCursor(Qt.PointingHandCursor)
            except Exception:
                logger.warning("Failed to display tooltip while handling multi-bar hover.", exc_info=True)
            try:
                QToolTip.showText(
                    QCursor.pos(),
                    f"{category}\n{_short_label(str(feature))}: {float(value):.3f}",
                    self,
                )
            except Exception:
                logger.warning("Failed to update cursor for multi-bar hover interaction.", exc_info=True)

        return handler

    def _on_multi_bar_clicked_factory(self, feature: str):
        def handler(index: int):
            if index < 0 or index >= len(self._multi_categories):
                return
            label = self._multi_categories[index]
            bounds = self._multi_category_bounds.get(label)
            if not bounds:
                return
            start_ts, end_ts = bounds
            try:
                level = self._current_level or self._multi_base_level or "M"
                self.bucket_selected.emit(start_ts, end_ts, level)
            except Exception:
                logger.warning("Failed to emit bucket-selected signal for multi-bar click.", exc_info=True)
        return handler

    def _format_period_label(self, level_code: str, period: pd.Period) -> str:
        try:
            start = period.start_time
        except Exception:
            start = pd.Timestamp(period.to_timestamp())
        if level_code == "A":
            return start.strftime("%Y")
        if level_code == "M":
            return start.strftime("%Y-%m")
        if level_code == "W":
            iso = start.isocalendar()
            return f"{iso.year}-W{int(iso.week):02d}"
        if level_code == "D":
            return start.strftime("%Y-%m-%d")
        return start.strftime("%Y-%m-%d %H:00")

    def _multi_bar_colors(self, count: int) -> list[QColor]:
        return group_color_cycle(count, dark_theme=self._dark_theme)

    def _noop_hover(self, *args, **kwargs):
        """Placeholder to disable hover handling in multi mode."""
        return

    # Hover & Click
    def _connect_hover_signals(self):
        if self._multi_mode:
            return
        if not self._series_hover_supported:
            self.series.hovered.connect(self._on_series_hovered)
            self._series_hover_supported = True
        # Connect hovered for every bar set, but avoid disconnecting unless
        # we've previously connected the handler (tracked in _connected_barsets).
        for s in self.series.barSets():
            if s in self._connected_barsets:
                # already connected; ensure we don't duplicate
                continue
            s.hovered.connect(self._on_set_hovered)
            self._connected_barsets.add(s)

    def _on_series_hovered(self, status: bool, index: int, _barset: QBarSet):
        if self._multi_mode:
            return
        self._set_hover_index(index if status else None)

    def _on_set_hovered(self, status: bool, index: int):
        if self._multi_mode:
            return
        self._set_hover_index(index if status else None)

    def _set_hover_index(self, idx: int | None):
        if self._multi_mode:
            return
        self._hover_idx = idx
        self.view.setCursor(Qt.PointingHandCursor if idx is not None else Qt.ArrowCursor)
        self._update_highlight()
        if idx is None:
            QToolTip.hideText(); return
        try:
            label = self._cats[idx] if 0 <= idx < len(self._cats) else ""
            raw = None
            if 0 <= idx < len(self._raw_vals): raw = self._raw_vals[idx]
            val_text = "No data" if raw is None else f"{raw:.3f}"
            QToolTip.showText(QCursor.pos(), f"{label}\n{val_text}", self)
        except Exception:
            logger.warning("Failed to show hover tooltip for monthly chart bar.", exc_info=True)

    def _mouse_press_wrapper(self, base_handler):
        def handler(ev):
            if self._multi_mode:
                return base_handler(ev)
            # Right click steps back a single drill level
            if ev.button() == Qt.RightButton:
                try:
                    self._drill_up()
                except Exception:
                    logger.warning("Failed to drill up one level from monthly chart right-click.", exc_info=True)
                return base_handler(ev)

            # Left click -> drill down if hovered
            if ev.button() == Qt.LeftButton:
                try:
                    if self._hover_idx is not None:
                        self._pending_click_index = int(self._hover_idx)
                        delay_ms = 250
                        app = QApplication.instance()
                        if app is not None:
                            try:
                                delay_ms = max(120, int(app.doubleClickInterval()))
                            except Exception:
                                delay_ms = 250
                        self._single_click_timer.start(delay_ms)
                except Exception:
                    logger.warning("Failed while handling monthly chart left-click interaction.", exc_info=True)
            return base_handler(ev)
        return handler

    def _mouse_double_click_wrapper(self, base_handler):
        def handler(ev):
            if ev.button() == Qt.LeftButton:
                self._pending_click_index = None
                if self._single_click_timer.isActive():
                    self._single_click_timer.stop()
                try:
                    self.reset_to_base()
                except Exception:
                    logger.warning("Failed to reset monthly chart to base level on double-click.", exc_info=True)
            return base_handler(ev)
        return handler

    def _commit_pending_single_click(self) -> None:
        if self._multi_mode:
            self._pending_click_index = None
            return
        idx = self._pending_click_index
        self._pending_click_index = None
        if idx is None:
            return
        try:
            self._on_bar_clicked(int(idx))
        except Exception:
            logger.warning("Failed to handle delayed monthly chart bar click.", exc_info=True)

    def _on_bar_clicked(self, idx: int):
        if self._multi_mode:
            return
        # if no bucket periods available, nothing to do
        if not self._bucket_periods or idx < 0 or idx >= len(self._bucket_periods):
            return

        cur_code = self._current_level if self._current_level is not None else (self._base_level or self._LEVELS[0][0])
        # compute start and next_start timestamps for the clicked bucket
        per = self._bucket_periods[idx]
        try:
            # for non-hour buckets, normalize to midnight naive timestamps
            if cur_code == "h":
                start = self._to_naive(per.to_timestamp(), keep_hour=True)
                next_start = self._to_naive((per + 1).to_timestamp(), keep_hour=True)
            else:
                start = self._to_naive(per.to_timestamp(), keep_hour=False)
                next_start = self._to_naive((per + 1).to_timestamp(), keep_hour=False)
        except Exception:
            start = self._to_naive(per.to_timestamp(), keep_hour=(cur_code == "h"))
            next_start = self._to_naive((per + 1).to_timestamp(), keep_hour=(cur_code == "h"))

        # emit selection so sidebar can filter
        try:
            self.bucket_selected.emit(start, next_start, cur_code)
        except Exception:
            logger.warning("Failed to emit bucket-selected signal from monthly chart bar click.", exc_info=True)

        # Do not locally render the next timeframe. Controller/viewmodel handles
        # filter + preprocessing + reload, then pushes new data back to this chart.

    def reset_to_base(self):
        self._pending_click_index = None
        if self._single_click_timer.isActive():
            self._single_click_timer.stop()
        try:
            self.reset_requested.emit()
        except Exception:
            logger.warning("Failed to restore base drill level while resetting monthly chart.", exc_info=True)

    def _drill_up(self):
        """Step back one level in history. If at base, do nothing."""
        if self._multi_mode:
            return
        if not self._history:
            # nothing to do
            return
        if len(self._history) <= 1:
            # already at base
            return
        # pop current
        popped = self._history.pop()
        # render previous
        level, t_s, v_s, _, _ = self._history[-1]
        self._current_level = level
        self._render_level(t_s, v_s, level)
        # emit bucket_selected for the parent bucket that created the popped view
        parent_start, parent_end = popped[3], popped[4]
        if parent_start is None:
            try: 
                self.reset_requested.emit()
            except Exception: 
                logger.warning("Failed to emit reset request while drilling up from monthly chart.", exc_info=True)
        else:
            # normalize parent times according to the parent level (level)
            try:
                ps = self._to_naive(parent_start, keep_hour=(level == "h"))
                pe = self._to_naive(parent_end, keep_hour=(level == "h"))
            except Exception:
                try:
                    ps = pd.Timestamp(parent_start)
                    pe = pd.Timestamp(parent_end)
                except Exception:
                    ps = parent_start; pe = parent_end
            try:
                self.bucket_selected.emit(ps, pe, level)
            except Exception:
                logger.warning("Failed to emit bucket-selected signal while drilling up monthly chart.", exc_info=True)

    def _timeframe_for_periods(self, level_code: str, periods: list[pd.Period]) -> str:
        """Return a human-friendly timeframe string for the given aggregation level and periods."""
        if not periods:
            return ""
        try:
            start_per = periods[0]
            end_per = periods[-1]
            start_ts = start_per.to_timestamp()
            end_ts = end_per.to_timestamp()
        except Exception:
            try:
                start_ts = pd.Timestamp(getattr(periods[0], "start_time", pd.NaT))
                end_ts = pd.Timestamp(getattr(periods[-1], "start_time", pd.NaT))
            except Exception:
                return ""

        # Format depending on level
        try:
            if level_code == "A":
                if start_ts.year == end_ts.year:
                    return f"{start_ts.year}"
                return f"{start_ts.year} — {end_ts.year}"

            if level_code == "M":
                if start_ts.year == end_ts.year:
                    return f"{start_ts.strftime('%b')} — {end_ts.strftime('%b %Y')}"
                return f"{start_ts.strftime('%Y-%m')} — {end_ts.strftime('%Y-%m')}"

            if level_code == "W":
                siso = start_ts.isocalendar(); eiso = end_ts.isocalendar()
                s_lbl = f"{siso.year}-W{int(siso.week):02d}"
                e_lbl = f"{eiso.year}-W{int(eiso.week):02d}"
                return f"{s_lbl} — {e_lbl}"

            if level_code == "D":
                if start_ts.year == end_ts.year and start_ts.month == end_ts.month:
                    return f"{start_ts.day} — {end_ts.strftime('%d %b %Y')}"
                return f"{start_ts.strftime('%Y-%m-%d')} — {end_ts.strftime('%Y-%m-%d')}"

            # H
            return f"{start_ts.strftime('%Y-%m-%d %H:%M')} — {end_ts.strftime('%Y-%m-%d %H:%M')}"
        except Exception:
            return f"{start_ts} — {end_ts}"

    # Overlay geometry
    def _update_highlight(self):
        if self._multi_mode:
            self._hover_item.setVisible(False)
            return
        idx = self._hover_idx
        if idx is None or idx < 0 or idx >= len(self._vals):
            self._hover_item.setVisible(False); return
        v = float(self._vals[idx])
        plot = self.chart.plotArea()
        n = max(1, len(self._cats))
        cell_w_px = plot.width() / n
        bar_w_px = (self.series.barWidth() or 0.6) * cell_w_px
        left_x = plot.left() + idx * cell_w_px + (cell_w_px - bar_w_px) / 2.0
        right_x = left_x + bar_w_px
        p0 = self.chart.mapToPosition(QPointF(idx + 0.5, 0.0), self.series)
        pv = self.chart.mapToPosition(QPointF(idx + 0.5, v), self.series)
        y0 = p0.y() - 1.0; yv = pv.y()
        top = min(y0, yv); bottom = max(y0, yv)
        if right_x - left_x < 1 or bottom - top < 1:
            self._hover_item.setVisible(False); return
        self._hover_item.setRect(QRectF(left_x, top, right_x - left_x, bottom - top))
        self._hover_item.setVisible(True)

    # baseline removed; hover overlay only
