
from __future__ import annotations
from bisect import bisect_left
from typing import Callable

from PySide6.QtCore import Qt, QDateTime, QPoint, QPointF, QRectF, QEvent, Signal, QElapsedTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QColor
from PySide6.QtWidgets import QApplication, QGraphicsView, QToolTip
from PySide6.QtCharts import QChart, QChartView, QXYSeries
import logging

logger = logging.getLogger(__name__)


class InteractiveChartView(QChartView):
    """
    Left-drag  = rubber-band zoom (emit on release if moved enough)
    Wheel      = zoom anchored at cursor (cursor x stays fixed in time)
    Right-drag = pan (X/Y; emit continuously + on release when X changes)
    Double L-click = reset (Shift=Y only, Ctrl=X only, none/both=both; bubble up)
    Hover (idle)= crosshair + tooltip snapped to nearest data point (series[0])
    """
    user_range_selected = Signal(QDateTime, QDateTime)
    user_reset_requested = Signal(bool, bool)  # (reset_x, reset_y)

    def __init__(self, chart: QChart, parent=None):
        super().__init__(chart, parent)

        # Rendering and interaction baseline
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)  # get move events with no buttons
        try:
            self.setRubberBand(QChartView.NoRubberBand)
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)
        try:
            self.setDragMode(QGraphicsView.NoDrag)
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)

        # ---- Zoom state (left button)
        self._rubber_active = False
        self._press_pos: QPointF | None = None
        self._drag_moved = False
        self._min_drag_px = 6  # pixels required to count as a drag
        self._zoom_start_min_ms: int | None = None
        self._zoom_start_max_ms: int | None = None

        # ---- Pan state (right button)
        self._panning = False
        self._pan_start_pos: QPointF | None = None
        self._pan_start_min_ms: int | None = None
        self._pan_start_max_ms: int | None = None
        self._pan_start_y_min: float | None = None
        self._pan_start_y_max: float | None = None
        self._pan_moved = False
        self._pan_x_changed = False
        self._pan_emit_live = True  # keep, but now throttled

        # Live-emit throttling during pan
        self._pan_emit_min_px = 6           # require ~6 px movement from last emit
        self._pan_emit_min_ms = 25          # or ~25 ms range shift from last emit
        self._pan_emit_interval_ms = 40     # OR at least every 40 ms (debounce)
        self._pan_last_emit_min_ms: int | None = None
        self._pan_last_emit_max_ms: int | None = None
        self._pan_last_emit_pos_x: float | None = None
        self._pan_emit_timer = QElapsedTimer()

        # ---- Hover highlight (snaps to nearest data point)
        self._show_crosshair = False
        self._last_mouse_pos: QPointF | None = None
        self._snap_screen_pt: QPointF | None = None  # where to draw the dot
        self._snap_data_x_ms: int | None = None      # snapped x (ms)
        self._snap_data_y: float | None = None       # snapped y

        # Light, subtle crosshair styling
        self._cross_pen = QPen(QColor(200, 200, 200, 180))  # light gray, slightly transparent
        self._cross_pen.setWidthF(1.0)
        self._dot_brush = QBrush(QColor(200, 200, 200, 180))
        self._hover_tooltip_callback: Callable[[QPointF], str] | None = None

    def set_hover_tooltip_callback(self, callback: Callable[[QPointF], str] | None) -> None:
        self._hover_tooltip_callback = callback

    # =========================
    # Helpers
    # =========================
    def _emit_axis_range(self):
        axes = self.chart().axes(Qt.Orientation.Horizontal)
        if not axes:
            return
        ax = axes[0]
        mn, mx = getattr(ax, "min", None), getattr(ax, "max", None)
        if callable(mn) and callable(mx):
            qmin, qmax = mn(), mx()
            if isinstance(qmin, QDateTime) and isinstance(qmax, QDateTime):
                self.user_range_selected.emit(qmin, qmax)

    def _get_x_axis_ms_range(self) -> tuple[int | None, int | None]:
        axes = self.chart().axes(Qt.Orientation.Horizontal)
        if not axes:
            return None, None
        ax = axes[0]
        mn, mx = getattr(ax, 'min', None), getattr(ax, 'max', None)
        if not (callable(mn) and callable(mx)):
            return None, None
        qmin, qmax = mn(), mx()
        try:
            return qmin.toMSecsSinceEpoch(), qmax.toMSecsSinceEpoch()
        except Exception:
            return None, None

    def _distance_sq_from_press(self, pos: QPointF | None) -> float:
        if self._press_pos is None or pos is None:
            return 0.0
        dx = pos.x() - self._press_pos.x()
        dy = pos.y() - self._press_pos.y()
        return dx*dx + dy*dy

    def _chart_series0(self):
        s = self.chart().series()
        return s[0] if s else None

    def _pos_to_data_x_ms(self, pos: QPointF) -> float | None:
        # Any QXYSeries is enough because axes are shared.
        for s in self.chart().series():
            if isinstance(s, QXYSeries):
                data_pt = self.chart().mapToValue(pos, s)
                return float(data_pt.x())
        return None

    def _pos_to_data_y(self, pos: QPointF) -> float | None:
        # Any QXYSeries is enough because axes are shared.
        for s in self.chart().series():
            if isinstance(s, QXYSeries):
                data_pt = self.chart().mapToValue(pos, s)
                return float(data_pt.y())
        return None

    def _snap_to_nearest_point(self, pos: QPointF) -> tuple[int | None, float | None, QPointF | None]:
        x_ms_float = self._pos_to_data_x_ms(pos)
        if x_ms_float is None:
            return None, None, None

        best_dx = float("inf")
        best_point = None
        best_series = None

        for s in self.chart().series():
            if not isinstance(s, QXYSeries):
                continue

            pts = s.points()
            if not pts:
                continue

            xs = [p.x() for p in pts]
            i = bisect_left(xs, x_ms_float)

            for idx in (i - 1, i):
                if 0 <= idx < len(pts):
                    p = pts[idx]
                    dx = abs(p.x() - x_ms_float)
                    if dx < best_dx:
                        best_dx = dx
                        best_point = p
                        best_series = s

        if best_point is None or best_series is None:
            return None, None, None

        x_ms = int(round(best_point.x()))
        y_val = float(best_point.y())
        screen_pt = self.chart().mapToPosition(best_point, best_series)
        return x_ms, y_val, screen_pt


    # =========================
    # Events
    # =========================
    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            # Begin pan
            min_ms, max_ms = self._get_x_axis_ms_range()
            y_min, y_max = None, None
            try:
                v_axes = self.chart().axes(Qt.Orientation.Vertical)
                if v_axes:
                    v_ax = v_axes[0]
                    v_min_get = getattr(v_ax, "min", None)
                    v_max_get = getattr(v_ax, "max", None)
                    if callable(v_min_get) and callable(v_max_get):
                        y_min = float(v_min_get())
                        y_max = float(v_max_get())
            except Exception:
                logger.warning("Exception in mousePressEvent", exc_info=True)
            try:
                pos = ev.position()
                self._pan_start_pos = QPointF(pos.x(), pos.y())
            except Exception:
                self._pan_start_pos = None
            self._pan_start_min_ms = min_ms
            self._pan_start_max_ms = max_ms
            self._pan_start_y_min = y_min
            self._pan_start_y_max = y_max
            self._panning = (self._pan_start_pos is not None
                             and min_ms is not None and max_ms is not None)
            self._pan_moved = False
            self._pan_x_changed = False
            if self._panning:
                self._pan_emit_timer.start()
                self._pan_last_emit_min_ms = self._pan_start_min_ms
                self._pan_last_emit_max_ms = self._pan_start_max_ms
                self._pan_last_emit_pos_x = self._pan_start_pos.x()
                self.setCursor(Qt.ClosedHandCursor)
                QToolTip.hideText()
                self._show_crosshair = False
                self.viewport().update()
            ev.accept()
            return

        if ev.button() == Qt.LeftButton:
            # Start rubber band immediately so QChartView tracks it
            try:
                self.setRubberBand(QChartView.RectangleRubberBand)
            except Exception:
                logger.warning("Exception in mousePressEvent", exc_info=True)
            self._rubber_active = True
            self._drag_moved = False

            try:
                pos = ev.position()
                self._press_pos = QPointF(pos.x(), pos.y())
            except Exception:
                self._press_pos = None

            # capture axis at press
            self._zoom_start_min_ms, self._zoom_start_max_ms = self._get_x_axis_ms_range()
            QToolTip.hideText()
            self._show_crosshair = False
            self.viewport().update()

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        pos = ev.position()

        # ----- PANNING (right button)
        if ev.buttons() & Qt.RightButton and self._panning and self._pan_start_pos is not None:
            try:
                dx_px = pos.x() - self._pan_start_pos.x()
                dy_px = pos.y() - self._pan_start_pos.y()
                plot_rect: QRectF = self.chart().plotArea()
                plot_w = max(1.0, float(plot_rect.width()))
                plot_h = max(1.0, float(plot_rect.height()))
                mods = ev.modifiers()
                try:
                    # Mirror wheel behavior: fallback to global modifiers when needed.
                    mods = mods | QApplication.keyboardModifiers()
                except Exception:
                    logger.warning("Exception in mouseMoveEvent", exc_info=True)
                shift_down = bool(mods & Qt.KeyboardModifier.ShiftModifier)
                ctrl_down = bool(mods & Qt.KeyboardModifier.ControlModifier)
                lock_y_only = shift_down and not ctrl_down
                lock_x_only = ctrl_down and not shift_down
                pan_x = not lock_y_only
                pan_y = not lock_x_only

                span_ms = float(self._pan_start_max_ms - self._pan_start_min_ms)
                shift_ms = -dx_px * (span_ms / plot_w)
                new_min = int(self._pan_start_min_ms + shift_ms)
                new_max = int(self._pan_start_max_ms + shift_ms)

                # Update axis continuously (smooth visual pan)
                try:
                    if pan_x:
                        self.chart().axes(Qt.Orientation.Horizontal)[0].setRange(
                            QDateTime.fromMSecsSinceEpoch(new_min),
                            QDateTime.fromMSecsSinceEpoch(new_max)
                        )
                        self._pan_x_changed = True
                except Exception:
                    logger.warning("Exception in mouseMoveEvent", exc_info=True)

                # Vertical pan
                try:
                    if (
                        pan_y
                        and
                        self._pan_start_y_min is not None
                        and self._pan_start_y_max is not None
                        and self._pan_start_y_max > self._pan_start_y_min
                    ):
                        y_span = float(self._pan_start_y_max - self._pan_start_y_min)
                        y_shift = dy_px * (y_span / plot_h)
                        new_y_min = float(self._pan_start_y_min + y_shift)
                        new_y_max = float(self._pan_start_y_max + y_shift)
                        v_axes = self.chart().axes(Qt.Orientation.Vertical)
                        if v_axes and new_y_max > new_y_min:
                            v_axes[0].setRange(new_y_min, new_y_max)
                except Exception:
                    logger.warning("Exception in mouseMoveEvent", exc_info=True)

                # mark that we moved
                if not self._pan_moved and ((dx_px*dx_px + dy_px*dy_px) >= (self._min_drag_px*self._min_drag_px)):
                    self._pan_moved = True

                # ---- Throttled live emit
                if self._pan_emit_live and self._pan_moved and pan_x:
                    should_emit = False

                    # 1) pixel threshold from last emit
                    if self._pan_last_emit_pos_x is None or abs(pos.x() - self._pan_last_emit_pos_x) >= self._pan_emit_min_px:
                        should_emit = True

                    # 2) ms threshold from last emit
                    if not should_emit and self._pan_last_emit_min_ms is not None and self._pan_last_emit_max_ms is not None:
                        if (abs(new_min - self._pan_last_emit_min_ms) >= self._pan_emit_min_ms or
                            abs(new_max - self._pan_last_emit_max_ms) >= self._pan_emit_min_ms):
                            should_emit = True

                    # 3) time-based debounce (emit at least every N ms)
                    if not should_emit:
                        if not self._pan_emit_timer.isValid():
                            self._pan_emit_timer.start()
                        if self._pan_emit_timer.elapsed() >= self._pan_emit_interval_ms:
                            should_emit = True

                    if should_emit:
                        self._emit_axis_range()
                        self._pan_last_emit_min_ms = new_min
                        self._pan_last_emit_max_ms = new_max
                        self._pan_last_emit_pos_x = float(pos.x())
                        self._pan_emit_timer.restart()

                ev.accept()
                return
            except Exception:
                logger.warning("Exception in mouseMoveEvent", exc_info=True)

        # ----- RUBBER-BAND (left button)
        if ev.buttons() & Qt.LeftButton and self._rubber_active:
            if self._distance_sq_from_press(pos) >= (self._min_drag_px * self._min_drag_px):
                self._drag_moved = True
            self.setCursor(Qt.CrossCursor)
            QToolTip.hideText()
            super().mouseMoveEvent(ev)
            return

        # ----- HOVER (no buttons): show crosshair + snapped tooltip
        if not (ev.buttons() & (Qt.LeftButton | Qt.RightButton)):
            self._last_mouse_pos = QPointF(pos.x(), pos.y())
            if self.chart().plotArea().contains(self._last_mouse_pos):
                x_ms, y_val, screen_pt = self._snap_to_nearest_point(self._last_mouse_pos)
                snapped_in_plot = (
                    x_ms is not None
                    and y_val is not None
                    and screen_pt is not None
                    and self.chart().plotArea().contains(screen_pt)
                )
                if snapped_in_plot:
                    self._snap_data_x_ms = x_ms
                    self._snap_data_y = y_val
                    self._snap_screen_pt = screen_pt
                    self._show_crosshair = True
                    self._update_hover_tooltip_snapped()
                else:
                    custom_tooltip = ""
                    if self._hover_tooltip_callback is not None:
                        try:
                            custom_tooltip = self._hover_tooltip_callback(self._last_mouse_pos) or ""
                        except Exception:
                            logger.warning("Exception in mouseMoveEvent", exc_info=True)
                            custom_tooltip = ""
                    # Keep the vertical hover line visible even when there is no snapped point.
                    self._show_crosshair = True
                    self._snap_screen_pt = None
                    if custom_tooltip:
                        global_pos = self.mapToGlobal(self.mapFromScene(self._last_mouse_pos.toPoint()))
                        QToolTip.showText(global_pos + QPoint(12, 16), custom_tooltip, self)
                    else:
                        QToolTip.hideText()
            else:
                self._show_crosshair = False
                QToolTip.hideText()

            self.viewport().update()
            super().mouseMoveEvent(ev)
            return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        # ----- Finish panning
        if ev.button() == Qt.RightButton and self._panning:
            # Apply one final pan update at release position so the committed
            # range matches where the drag ended (no small backward snap).
            try:
                pos = ev.position()
                if (
                    self._pan_start_pos is not None
                    and self._pan_start_min_ms is not None
                    and self._pan_start_max_ms is not None
                ):
                    dx_px = pos.x() - self._pan_start_pos.x()
                    plot_rect: QRectF = self.chart().plotArea()
                    plot_w = max(1.0, float(plot_rect.width()))
                    mods = ev.modifiers()
                    try:
                        mods = mods | QApplication.keyboardModifiers()
                    except Exception:
                        logger.warning("Exception in mouseReleaseEvent", exc_info=True)
                    shift_down = bool(mods & Qt.KeyboardModifier.ShiftModifier)
                    ctrl_down = bool(mods & Qt.KeyboardModifier.ControlModifier)
                    lock_y_only = shift_down and not ctrl_down
                    pan_x = not lock_y_only
                    if pan_x:
                        span_ms = float(self._pan_start_max_ms - self._pan_start_min_ms)
                        shift_ms = -dx_px * (span_ms / plot_w)
                        final_min = int(self._pan_start_min_ms + shift_ms)
                        final_max = int(self._pan_start_max_ms + shift_ms)
                        self.chart().axes(Qt.Orientation.Horizontal)[0].setRange(
                            QDateTime.fromMSecsSinceEpoch(final_min),
                            QDateTime.fromMSecsSinceEpoch(final_max),
                        )
                        self._pan_x_changed = True
            except Exception:
                logger.warning("Exception in mouseReleaseEvent", exc_info=True)

            self._panning = False
            self._pan_start_pos = None
            self._pan_start_min_ms = None
            self._pan_start_max_ms = None
            self._pan_start_y_min = None
            self._pan_start_y_max = None
            self.setCursor(Qt.ArrowCursor)

            if self._pan_moved and self._pan_x_changed:
                self._emit_axis_range()
            self._pan_moved = False
            self._pan_x_changed = False
            # cleanup
            self._pan_last_emit_min_ms = None
            self._pan_last_emit_max_ms = None
            self._pan_last_emit_pos_x = None
            ev.accept()
            return

        super().mouseReleaseEvent(ev)

        # ----- Finish rubber-band zoom
        if ev.button() == Qt.LeftButton and self._rubber_active:
            try:
                self.setRubberBand(QChartView.NoRubberBand)
            except Exception:
                logger.warning("Exception in mouseReleaseEvent", exc_info=True)

            # Only emit if the user actually dragged enough & rect big enough
            if self._drag_moved:
                self._emit_axis_range()

            # cleanup
            self._rubber_active = False
            self._drag_moved = False
            self._press_pos = None
            self._zoom_start_min_ms = None
            self._zoom_start_max_ms = None

            # resume hover
            self._show_crosshair = True
            self.viewport().update()

    def wheelEvent(self, ev):
        """
        Zoom anchored at cursor:
        - If delta > 0: zoom in
        - If delta < 0: zoom out
        Modifiers:
        - Shift: Y axis only
        - Ctrl: X axis only
        - none / both: both axes
        """
        try:
            ad = ev.angleDelta()
            dy = ad.y() if ad is not None else 0
            x_range_changed = False

            if dy == 0:
                ev.ignore()
                return

            # factor < 1 -> zoom in (tighter); factor > 1 -> zoom out
            base = 0.8
            factor = base if dy > 0 else 1.0 / base
            mods = ev.modifiers()
            try:
                # Some wheel event paths can miss keyboard modifiers on Windows;
                # merge with the global keyboard state as a fallback.
                mods = mods | QApplication.keyboardModifiers()
            except Exception:
                logger.warning("Exception in wheelEvent", exc_info=True)
            shift_down = bool(mods & Qt.KeyboardModifier.ShiftModifier)
            ctrl_down = bool(mods & Qt.KeyboardModifier.ControlModifier)
            lock_y_only = shift_down and not ctrl_down
            lock_x_only = ctrl_down and not shift_down
            zoom_x = not lock_y_only
            zoom_y = not lock_x_only

            min_ms, max_ms = self._get_x_axis_ms_range()
            pos = ev.position()
            plot_rect: QRectF = self.chart().plotArea()

            if zoom_x:
                if min_ms is None or max_ms is None or min_ms >= max_ms:
                    ev.accept()
                    return

                # data-x under cursor (only if inside plot area)
                if plot_rect.contains(pos):
                    center_ms = self._pos_to_data_x_ms(pos)
                else:
                    center_ms = (min_ms + max_ms) / 2.0
                if center_ms is None:
                    center_ms = (min_ms + max_ms) / 2.0

                # scale left/right spans independently to keep cursor anchor stable
                left_span = center_ms - min_ms
                right_span = max_ms - center_ms
                new_left = max(1.0, left_span * factor)
                new_right = max(1.0, right_span * factor)

                new_min = int(center_ms - new_left)
                new_max = int(center_ms + new_right)
                if new_max <= new_min:
                    ev.accept()
                    return

                try:
                    self.chart().axes(Qt.Orientation.Horizontal)[0].setRange(
                        QDateTime.fromMSecsSinceEpoch(new_min),
                        QDateTime.fromMSecsSinceEpoch(new_max)
                    )
                    x_range_changed = (new_min != min_ms) or (new_max != max_ms)
                except Exception:
                    logger.warning("Exception in wheelEvent", exc_info=True)

            # Y-axis zoom around cursor Y (fallback to center if needed)
            if zoom_y:
                try:
                    v_axes = self.chart().axes(Qt.Orientation.Vertical)
                    if v_axes:
                        v_ax = v_axes[0]
                        v_min_get = getattr(v_ax, "min", None)
                        v_max_get = getattr(v_ax, "max", None)
                        if callable(v_min_get) and callable(v_max_get):
                            cur_vmin = float(v_min_get())
                            cur_vmax = float(v_max_get())
                            if cur_vmax > cur_vmin:
                                if plot_rect.contains(pos):
                                    center_y = self._pos_to_data_y(pos)
                                else:
                                    center_y = (cur_vmin + cur_vmax) / 2.0
                                if center_y is None:
                                    center_y = (cur_vmin + cur_vmax) / 2.0

                                bottom_span = center_y - cur_vmin
                                top_span = cur_vmax - center_y
                                new_bottom = max(1e-12, bottom_span * factor)
                                new_top = max(1e-12, top_span * factor)
                                new_vmin = float(center_y - new_bottom)
                                new_vmax = float(center_y + new_top)
                                if new_vmax > new_vmin:
                                    v_ax.setRange(new_vmin, new_vmax)
                except Exception:
                    logger.warning("Exception in wheelEvent", exc_info=True)

            if x_range_changed:
                self._emit_axis_range()
            ev.accept()
        except Exception:
            ev.accept()

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            mods = ev.modifiers()
            try:
                # Some event paths can miss keyboard modifiers on Windows;
                # merge with the global keyboard state as a fallback.
                mods = mods | QApplication.keyboardModifiers()
            except Exception:
                logger.warning("Exception in mouseDoubleClickEvent", exc_info=True)
            shift_down = bool(mods & Qt.KeyboardModifier.ShiftModifier)
            ctrl_down = bool(mods & Qt.KeyboardModifier.ControlModifier)
            reset_x = ctrl_down and not shift_down
            reset_y = shift_down and not ctrl_down
            if not (reset_x or reset_y):
                reset_x = True
                reset_y = True
            self.user_reset_requested.emit(reset_x, reset_y)
        super().mouseDoubleClickEvent(ev)

    def leaveEvent(self, ev: QEvent):
        # Hide crosshair when cursor leaves the view
        self._show_crosshair = False
        QToolTip.hideText()
        self.viewport().update()
        super().leaveEvent(ev)

    # =========================
    # Hover tooltip / crosshair (snapped)
    # =========================
    def _update_hover_tooltip_snapped(self):
        """Show time/value for the snapped data point."""
        if self._snap_data_x_ms is None or self._snap_data_y is None or self._last_mouse_pos is None:
            QToolTip.hideText()
            return
        plot_rect: QRectF = self.chart().plotArea()
        if self._snap_screen_pt is None or not plot_rect.contains(self._snap_screen_pt):
            QToolTip.hideText()
            return

        dt = QDateTime.fromMSecsSinceEpoch(int(self._snap_data_x_ms))
        text = f"{dt.toString('yyyy-MM-dd HH:mm:ss.zzz')}  |  {self._snap_data_y:g}"
        if self._hover_tooltip_callback is not None:
            try:
                extra = self._hover_tooltip_callback(self._last_mouse_pos) or ""
                if extra:
                    lines = [line for line in str(extra).splitlines() if line.strip()]
                    if lines and lines[0].strip().lower().startswith("date:"):
                        lines = lines[1:]
                    extra = "\n".join(lines).strip()
                if extra:
                    text = f"{text}\n{extra}"
            except Exception:
                logger.warning("Exception in _update_hover_tooltip_snapped", exc_info=True)

        # Place tooltip near the cursor, not the data point (more natural while scanning)
        global_pos = self.mapToGlobal(self.mapFromScene(self._last_mouse_pos.toPoint()))
        QToolTip.showText(global_pos + QPoint(12, 16), text, self)

    def drawForeground(self, painter: QPainter, rect: QRectF):
        """Draw vertical crosshair and, when available, a dot at the snapped data point."""
        super().drawForeground(painter, rect)
        if not self._show_crosshair or self._last_mouse_pos is None:
            return

        plot_rect: QRectF = self.chart().plotArea()
        if not plot_rect.contains(self._last_mouse_pos):
            return

        # Draw vertical line across the plot at the mouse x (so it follows the hand)
        x = self._last_mouse_pos.x()

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(self._cross_pen)

        painter.drawLine(int(x), int(plot_rect.top()), int(x), int(plot_rect.bottom()))

        # Draw a small circle at the snapped data point Dataset when available.
        if self._snap_screen_pt is not None and plot_rect.contains(self._snap_screen_pt):
            r = 3.5
            painter.setBrush(self._dot_brush)
            painter.drawEllipse(self._snap_screen_pt, r, r)

        painter.restore()

