from __future__ import annotations
"""Interactive SOM widget with hover highlighting (modern blue→red colormap)."""


import math
from typing import Callable, List, Optional, Iterable, Tuple, Set

import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QAction, QPixmap
from PySide6.QtWidgets import QToolTip, QWidget, QMenu, QInputDialog
from ...localization import tr
from ...style.cluster_colors import build_cluster_palette_from_frame



class SomMapView(QWidget):
    """Custom widget that renders SOM matrices with interactive hovers."""

    # Signal emitted when user renames a cluster via context menu
    # Emits (cluster_id: int, new_name: str)
    cluster_rename_requested = Signal(int, str)
    # Signal emitted when user requests details from context menu
    # Emits (row: int, col: int)
    details_requested = Signal(int, int)
    # Signal emitted on left click to select a cell
    # Emits (row: int, col: int)
    cell_clicked = Signal(int, int)

    # --- Modern, perceptually pleasant color ramps ---------------------
    # Sequential (for strictly ≥0 data): deep blue → teal → yellow → orange → red
    _SEQ_STOPS = ("#1b3a8a", "#2e6bd3", "#2ec4b6", "#f4d35e", "#f26419", "#d72638")
    # Diverging (for symmetric=True): blue → light → red with a neutral center
    _DIV_STOPS = ("#2e6fcd", "#9db7ff", "#f8f9fb", "#ffb0b0", "#d72638")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._matrix: Optional[pd.DataFrame] = None
        self._annotations: Optional[pd.DataFrame] = None
        self._values: Optional[np.ndarray] = None
        self._tooltip_callback: Optional[Callable[[int, int], str]] = None
        self._value_formatter: Optional[Callable[[float], str]] = None
        self._hover_cell: Optional[tuple[int, int]] = None
        self._symmetric: bool = False
        self._hex_layout: Optional[dict] = None
        self._geometry_cache: Optional[dict] = None
        self._geometry_size: tuple[int, int] = (-1, -1)
        self._cluster_overlay: Optional[pd.DataFrame] = None
        self._cluster_palette: dict[object, QColor] = {}
        self._cluster_border_palette: dict[object, QColor] = {}  # Separate palette for borders
        self._cluster_fill_only: bool = False
        self._cluster_border_mode: bool = False  # When True, show borders instead of color overlay
        self._highlight_cells: Set[tuple[int, int]] = set()
        self._cluster_names: dict[int, str] = {}  # Custom names for clusters
        self._show_cluster_centered_labels: bool = False  # Show single label at cluster center
        self._render_cache: Optional[QPixmap] = None
        self._render_cache_key: Optional[tuple] = None
        self._data_version = 0
        self._overlay_version = 0
        self._label_version = 0
        self._boundary_cache: Optional[list[tuple[QPointF, QPointF, object, object]]] = None
        self._boundary_cache_key: Optional[tuple] = None
        self._centroid_cache: Optional[dict[object, QPointF]] = None
        self._centroid_cache_key: Optional[tuple] = None
        self._placeholder_text: str = ""
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    # ------------------------------------------------------------------
    def set_map_data(
        self,
        matrix: Optional[pd.DataFrame],
        tooltip_callback: Optional[Callable[[int, int], str]] = None,
        *,
        annotations: Optional[pd.DataFrame] = None,
        value_formatter: Optional[Callable[[float], str]] = None,
        symmetric: bool = False,
    ) -> None:
        if matrix is None or matrix.empty:
            self.clear_map()
            return

        # Robust numeric conversion: coerce non-numeric to NaN instead of raising
        numeric = matrix.apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float)

        # Replace NaNs with the mean of finite values (or zeros if none)
        if np.isfinite(values).any():
            fill_value = float(np.nanmean(values[np.isfinite(values)]))
        else:
            fill_value = 0.0
        values = np.nan_to_num(values, nan=fill_value, posinf=fill_value, neginf=fill_value)

        self._matrix = matrix.copy()
        self._annotations = annotations.copy() if annotations is not None else None
        self._values = values
        self._tooltip_callback = tooltip_callback
        self._value_formatter = value_formatter
        self._symmetric = symmetric
        self._placeholder_text = ""
        self._hex_layout = self._compute_hex_layout(values.shape[0], values.shape[1])
        self._geometry_cache = None
        self._geometry_size = (-1, -1)
        self._data_version += 1
        self._invalidate_render_cache()
        self._ensure_overlay_alignment()
        self.update()

    def clear_map(self) -> None:
        self._matrix = None
        self._annotations = None
        self._values = None
        self._tooltip_callback = None
        self._value_formatter = None
        self._hover_cell = None
        self._hex_layout = None
        self._geometry_cache = None
        self._geometry_size = (-1, -1)
        self._cluster_overlay = None
        self._cluster_palette = {}
        self._cluster_border_palette = {}
        self._cluster_fill_only = False
        self._cluster_border_mode = False
        self._highlight_cells.clear()
        self._cluster_names = {}
        self._show_cluster_centered_labels = False
        self._render_cache = None
        self._render_cache_key = None
        self._data_version = 0
        self._overlay_version = 0
        self._label_version = 0
        self._boundary_cache = None
        self._boundary_cache_key = None
        self._centroid_cache = None
        self._centroid_cache_key = None
        self._placeholder_text = ""
        QToolTip.hideText()
        self.update()

    def show_placeholder(self, text: str) -> None:
        self._matrix = None
        self._annotations = None
        self._values = None
        self._tooltip_callback = None
        self._value_formatter = None
        self._hover_cell = None
        self._hex_layout = None
        self._geometry_cache = None
        self._geometry_size = (-1, -1)
        self._cluster_overlay = None
        self._cluster_palette = {}
        self._cluster_border_palette = {}
        self._cluster_fill_only = False
        self._cluster_border_mode = False
        self._highlight_cells.clear()
        self._cluster_names = {}
        self._show_cluster_centered_labels = False
        self._render_cache = None
        self._render_cache_key = None
        self._data_version = 0
        self._overlay_version = 0
        self._label_version = 0
        self._boundary_cache = None
        self._boundary_cache_key = None
        self._centroid_cache = None
        self._centroid_cache_key = None
        self._placeholder_text = str(text or "")
        QToolTip.hideText()
        self.update()

    # ------------------------------------------------------------------
    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._matrix is None or self._values is None:
            QToolTip.hideText()
            return super().mouseMoveEvent(event)

        pos: QPointF = event.position()
        row, col = self._cell_from_point(pos)
        if row is None or col is None:
            if self._hover_cell is not None:
                self._hover_cell = None
                self.update()
            QToolTip.hideText()
            return super().mouseMoveEvent(event)

        # Only repaint if the hovered cell actually changed
        if self._hover_cell != (row, col):
            self._hover_cell = (row, col)
            self.update()

        tooltip = self._tooltip_for(row, col)
        if tooltip:
            QToolTip.showText(event.globalPosition().toPoint(), tooltip, self)
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self._matrix is not None:
            pos: QPointF = event.position()
            row, col = self._cell_from_point(pos)
            if row is not None and col is not None:
                self.cell_clicked.emit(row, col)
        super().mousePressEvent(event)

    def leaveEvent(self, event):  # type: ignore[override]
        if self._hover_cell is not None:
            self._hover_cell = None
            self.update()
        QToolTip.hideText()
        super().leaveEvent(event)

    def contextMenuEvent(self, event):  # type: ignore[override]
        """Show context menu for map details and cluster operations like renaming."""
        # Find which cell was right-clicked
        pos = event.pos()
        row, col = self._cell_from_point(QPointF(pos))
        if row is None or col is None:
            return super().contextMenuEvent(event)

        menu = QMenu(self)

        details_action = QAction(tr("Details..."), self)
        details_action.triggered.connect(lambda: self.details_requested.emit(row, col))
        menu.addAction(details_action)

        cluster_label = self._cluster_label_at(row, col)
        if cluster_label is not None:
            try:
                cluster_id = int(cluster_label)
            except (ValueError, TypeError):
                cluster_id = None

            if cluster_id is not None:
                current_name = self._cluster_names.get(cluster_id, "")
                rename_action = QAction(
                    tr("Rename cluster {cluster_id}...").format(cluster_id=cluster_id),
                    self,
                )
                rename_action.triggered.connect(
                    lambda: self._show_rename_dialog(cluster_id, current_name)
                )
                menu.addAction(rename_action)

        menu.exec(event.globalPos())

    def _show_rename_dialog(self, cluster_id: int, current_name: str) -> None:
        """Show dialog to rename a cluster and emit signal if changed."""
        new_name, ok = QInputDialog.getText(
            self,
            tr("Rename Cluster {cluster_id}").format(cluster_id=cluster_id),
            tr("Enter new name for cluster:"),
            text=current_name,
        )
        if ok:
            # Emit signal so the view model can be updated
            self.cluster_rename_requested.emit(cluster_id, new_name.strip())

    # ------------------------------------------------------------------
    def paintEvent(self, event):  # type: ignore[override]
        # NOTE: we intentionally do NOT call super().paintEvent here to avoid
        # double painting on some styles; Qt will clear the background for QWidget.
        if self._matrix is None or self._values is None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), self.palette().window().color())
            if self._placeholder_text:
                painter.setPen(self.palette().text().color())
                painter.drawText(
                    self.rect().adjusted(12, 12, -12, -12),
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.TextWordWrap,
                    self._placeholder_text,
                )
            return

        rows, cols = self._values.shape
        if rows == 0 or cols == 0:
            return

        geometry = self._prepare_geometry()
        if geometry is None:
            return

        cache_key = self._render_cache_key_for_current_state()
        if self._render_cache is None or self._render_cache_key != cache_key:
            self._render_cache = self._render_base_layer(geometry)
            self._render_cache_key = cache_key

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.drawPixmap(0, 0, self._render_cache)

        polygons: dict[tuple[int, int], QPolygonF] = geometry["polygons"]
        radius: float = geometry["radius"]

        if self._highlight_cells:
            outer_pen = QPen(QColor(0, 0, 0), max(3, int(radius * 0.30)))
            inner_pen = QPen(QColor(255, 255, 255), max(2, int(radius * 0.20)))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for cell in self._highlight_cells:
                polygon = polygons.get(cell)
                if polygon is None:
                    continue
                painter.setPen(outer_pen)
                painter.drawPolygon(polygon)
                painter.setPen(inner_pen)
                painter.drawPolygon(polygon)

        # Hover highlight
        if self._hover_cell is not None:
            polygon = polygons.get(self._hover_cell)
            if polygon is not None:
                highlight_pen = QPen(QColor(255, 255, 255), max(2, int(radius * 0.2)))
                painter.setPen(highlight_pen)
                painter.setBrush(QColor(255, 255, 255, 40))
                painter.drawPolygon(polygon)

    def _render_cache_key_for_current_state(self) -> tuple:
        size = (self.width(), self.height())
        return (
            size,
            float(self.devicePixelRatioF()),
            self._data_version,
            self._overlay_version,
            self._label_version,
            bool(self._cluster_overlay is not None),
            bool(self._cluster_fill_only),
            bool(self._cluster_border_mode),
            bool(self._show_cluster_centered_labels),
        )

    def _invalidate_render_cache(self) -> None:
        self._render_cache = None
        self._render_cache_key = None

    def _render_base_layer(self, geometry: dict) -> QPixmap:
        size = self.size()
        dpr = float(self.devicePixelRatioF())
        pixmap = QPixmap(int(size.width() * dpr), int(size.height() * dpr))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(self.palette().window().color())

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        polygons: dict[tuple[int, int], QPolygonF] = geometry["polygons"]
        radius: float = geometry["radius"]
        rows, cols = self._values.shape  # type: ignore[union-attr]

        vmin, vmax = self._color_range()

        draw_base = not (self._cluster_overlay is not None and self._cluster_fill_only)
        backgrounds: dict[tuple[int, int], QColor] = {}
        texts: dict[tuple[int, int], str] = {}

        for r in range(rows):
            for c in range(cols):
                polygon = polygons.get((r, c))
                if polygon is None:
                    continue
                cell_color = QColor(32, 32, 32)
                if draw_base:
                    value = float(self._values[r, c])  # type: ignore[index]
                    cell_color = self._value_to_color(value, vmin, vmax)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(cell_color)
                    painter.drawPolygon(polygon)
                backgrounds[(r, c)] = cell_color

                if not self._show_cluster_centered_labels:
                    text = self._cell_text(r, c)
                    if text:
                        texts[(r, c)] = text

        if self._cluster_overlay is not None and not self._cluster_border_mode:
            overlay_alpha = 255 if self._cluster_fill_only else 120
            for r in range(rows):
                for c in range(cols):
                    polygon = polygons.get((r, c))
                    if polygon is None:
                        continue
                    label = self._cluster_label_at(r, c)
                    if label is None:
                        continue
                    color = self._cluster_color(label)
                    if color is None:
                        continue
                    overlay_color = QColor(color)
                    overlay_color.setAlpha(overlay_alpha)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(overlay_color)
                    painter.drawPolygon(polygon)
                    if self._cluster_fill_only:
                        backgrounds[(r, c)] = QColor(color)
        elif self._cluster_overlay is not None and self._cluster_fill_only:
            for r in range(rows):
                for c in range(cols):
                    polygon = polygons.get((r, c))
                    if polygon is None:
                        continue
                    label = self._cluster_label_at(r, c)
                    if label is None:
                        continue
                    color = self._cluster_color(label)
                    if color is None:
                        continue
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(color)
                    painter.drawPolygon(polygon)
                    backgrounds[(r, c)] = QColor(color)

        if not self._show_cluster_centered_labels:
            for (r, c), text in texts.items():
                polygon = polygons.get((r, c))
                if polygon is None:
                    continue
                bg_color = backgrounds.get((r, c), QColor(32, 32, 32))
                painter.setPen(self._text_pen(bg_color))
                painter.drawText(
                    polygon.boundingRect(),
                    Qt.AlignmentFlag.AlignCenter,
                    text,
                )

        grid_pen = QPen(QColor(255, 255, 255, 110), max(1, int(radius * 0.10)))
        painter.setPen(grid_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for polygon in polygons.values():
            painter.drawPolygon(polygon)

        if self._cluster_overlay is not None and self._cluster_border_mode:
            self._draw_cluster_boundary_edges(painter, radius, polygons=polygons)

        if self._show_cluster_centered_labels and self._cluster_overlay is not None:
            self._draw_cluster_centered_labels(painter, polygons, backgrounds, radius)

        painter.end()
        return pixmap

    # ------------------------------------------------------------------
    def _cell_from_point(self, pos: QPointF) -> tuple[Optional[int], Optional[int]]:
        if self._matrix is None or self._values is None:
            return (None, None)
        rows, cols = self._values.shape
        if rows == 0 or cols == 0:
            return (None, None)
        geometry = self._prepare_geometry()
        if geometry is None:
            return (None, None)
        polygons: dict[tuple[int, int], QPolygonF] = geometry["polygons"]
        for (row, col), polygon in polygons.items():
            if polygon.containsPoint(pos, Qt.FillRule.WindingFill):
                return (row, col)
        return (None, None)

    def _tooltip_for(self, row: int, col: int) -> str:
        tooltip = ""
        if self._tooltip_callback:
            try:
                tooltip = self._tooltip_callback(row, col) or ""
            except Exception:
                tooltip = ""
        if tooltip:
            return tooltip
        if self._matrix is None or row >= self._matrix.shape[0] or col >= self._matrix.shape[1]:
            return ""
        value = self._matrix.iat[row, col]
        parts = [f"Neuron ({col}, {row})"]
        if isinstance(value, (int, float, np.floating)):
            parts.append(f"Value: {float(value):.4f}")
        else:
            parts.append(f"Value: {value}")
        cluster_label = self._cluster_label_at(row, col)
        if cluster_label is not None:
            label_text = cluster_label
            if isinstance(cluster_label, (int, float, np.integer, np.floating)):
                try:
                    label_text = int(cluster_label)
                except Exception:
                    label_text = cluster_label
            parts.append(f"Cluster: {label_text}")
        return "\n".join(parts)

    def _cell_text(self, row: int, col: int) -> str:
        if self._matrix is None:
            return ""
        if self._annotations is not None:
            if row >= self._annotations.shape[0] or col >= self._annotations.shape[1]:
                return ""
            display_value = self._annotations.iat[row, col]
            if display_value is None or (isinstance(display_value, float) and not np.isfinite(display_value)):
                return ""
            if self._value_formatter is not None:
                try:
                    return self._value_formatter(display_value) or ""
                except Exception:
                    return ""
            return str(display_value)
        else:
            # No annotations → keep cells clean unless formatter is provided
            if self._value_formatter is None:
                return ""
            try:
                val = self._matrix.iat[row, col]
                return self._value_formatter(val) or ""
            except Exception:
                return ""

    def _color_range(self) -> tuple[float, float]:
        if self._values is None or not self._values.size:
            return (0.0, 1.0)

        # true data range
        vmin = float(np.nanmin(self._values))
        vmax = float(np.nanmax(self._values))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return (0.0, 1.0)

        # Only symmetrize if the data cross zero
        if self._symmetric and (vmin < 0.0 < vmax):
            bound = max(abs(vmin), abs(vmax)) or 1.0
            return (-bound, bound)

        if vmax == vmin:
            vmax = vmin + 1.0
        return (vmin, vmax)

    # --- New: perceptual colormap with smooth interpolation -------------
    @staticmethod
    def _hex_to_rgb(c: str) -> Tuple[int, int, int]:
        c = c.lstrip("#")
        return tuple(int(c[i:i+2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    @staticmethod
    def _interpolate_rgb(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
        return (
            int(round(a[0] + (b[0] - a[0]) * t)),
            int(round(a[1] + (b[1] - a[1]) * t)),
            int(round(a[2] + (b[2] - a[2]) * t)),
        )

    @classmethod
    def _palette(cls, symmetric: bool) -> Iterable[Tuple[int, int, int]]:
        stops = cls._DIV_STOPS if symmetric else cls._SEQ_STOPS
        return [cls._hex_to_rgb(s) for s in stops]

    def _value_to_color(self, value: float, vmin: float, vmax: float) -> QColor:
        if not np.isfinite(value) or vmax <= vmin:
            return QColor(96, 96, 96)

        # diverging only if range crosses zero (based on vmin/vmax from _color_range)
        #use_diverging = self._symmetric and (vmin < 0.0 < vmax)
        use_diverging = False  # always use sequential blue→red, no white

        if use_diverging:
            bound = max(abs(vmin), abs(vmax)) or 1.0
            t = (value / (2 * bound)) + 0.5
        else:
            t = (value - vmin) / (vmax - vmin)

        t = max(0.0, min(1.0, float(t)))
        pals = list(self._palette(use_diverging))
        n = len(pals) - 1
        idx = min(int(t * n), n - 1)
        local_t = (t * n) - idx
        r, g, b = self._interpolate_rgb(pals[idx], pals[idx + 1], local_t)
        return QColor(r, g, b)

    @staticmethod
    def _text_pen(background: QColor) -> QPen:
        luminance = 0.299 * background.red() + 0.587 * background.green() + 0.114 * background.blue()
        color = QColor(0, 0, 0) if luminance > 145 else QColor(255, 255, 255)
        return QPen(color)

    # ------------------------------------------------------------------
    def set_cluster_overlay(self, cluster_df: Optional[pd.DataFrame], *, fill_only: bool = False) -> None:
        if cluster_df is None or cluster_df.empty:
            self._cluster_overlay = None
            self._cluster_palette = {}
            self._cluster_border_palette = {}
            self._cluster_fill_only = False
        else:
            self._cluster_overlay = cluster_df.copy()
            self._cluster_fill_only = bool(fill_only)
            self._cluster_palette = self._build_cluster_palette(self._cluster_overlay)
            self._cluster_border_palette = self._build_cluster_border_palette(self._cluster_overlay)
        self._ensure_overlay_alignment()
        if self._cluster_overlay is None:
            self._cluster_fill_only = False
        self._overlay_version += 1
        self._invalidate_render_cache()
        self.update()

    def set_cluster_names(self, names: Optional[dict[int, str]]) -> None:
        """Set custom names for clusters to display on the map."""
        self._cluster_names = dict(names) if names else {}
        self._label_version += 1
        self._invalidate_render_cache()
        self.update()

    def set_show_cluster_centered_labels(self, enabled: bool) -> None:
        """Enable or disable showing single labels at cluster centers.
        
        When enabled, instead of showing values in each cell, displays one
        label (custom name or cluster ID) at the center of each cluster.
        """
        if self._show_cluster_centered_labels == enabled:
            return
        self._show_cluster_centered_labels = bool(enabled)
        self._label_version += 1
        self._invalidate_render_cache()
        self.update()

    def set_highlight_cells(self, cells: Optional[Iterable[tuple[int, int]]]) -> None:
        new_cells: Set[tuple[int, int]] = set()
        if cells is not None:
            for coord in cells:
                try:
                    row, col = int(coord[0]), int(coord[1])
                except Exception:
                    continue
                new_cells.add((row, col))
        if new_cells == self._highlight_cells:
            return
        self._highlight_cells = new_cells
        self.update()

    def _cluster_label_at(self, row: int, col: int):
        if self._cluster_overlay is None:
            return None
        if row >= self._cluster_overlay.shape[0] or col >= self._cluster_overlay.shape[1]:
            return None
        value = self._cluster_overlay.iat[row, col]
        if pd.isna(value):
            return None
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def _cluster_color(self, label) -> Optional[QColor]:
        color = self._cluster_palette.get(label)
        if color is not None:
            return QColor(color)
        return None

    def _cluster_border_color(self, label) -> Optional[QColor]:
        """Get the border color for a cluster label (brighter colors)."""
        color = self._cluster_border_palette.get(label)
        if color is not None:
            return QColor(color)
        return None

    def _build_cluster_palette(self, cluster_df: Optional[pd.DataFrame]) -> dict[object, QColor]:
        return build_cluster_palette_from_frame(cluster_df, border=False)

    def _build_cluster_border_palette(self, cluster_df: Optional[pd.DataFrame]) -> dict[object, QColor]:
        """Build palette for cluster borders using shared global cluster colors."""
        return build_cluster_palette_from_frame(cluster_df, border=True)

    def _ensure_overlay_alignment(self) -> None:
        if self._cluster_overlay is None or self._values is None:
            return
        if self._cluster_overlay.shape != self._values.shape:
            self._cluster_overlay = None
            self._cluster_palette = {}
            self._cluster_border_palette = {}
            self._cluster_fill_only = False

    def set_cluster_border_mode(self, enabled: bool) -> None:
        """Enable or disable cluster border mode.
        
        When enabled, cluster boundaries are shown as colored edges instead of
        overlaying colors on the cells. This preserves the original cell colors
        while clearly showing cluster boundaries.
        """
        if self._cluster_border_mode == enabled:
            return
        self._cluster_border_mode = bool(enabled)
        self._overlay_version += 1
        self._invalidate_render_cache()
        self.update()

    def _compute_cluster_centroids(
        self, polygons: dict[tuple[int, int], QPolygonF]
    ) -> dict[object, QPointF]:
        """Compute the visual centroid (average position) for each cluster.
        
        Returns a dictionary mapping cluster label to the center QPointF where
        the cluster label should be drawn.
        """
        if self._cluster_overlay is None:
            return {}
        
        # Group cells by cluster
        cluster_cells: dict[object, List[tuple[int, int]]] = {}
        rows, cols = self._cluster_overlay.shape
        for r in range(rows):
            for c in range(cols):
                label = self._cluster_label_at(r, c)
                if label is None:
                    continue
                if label not in cluster_cells:
                    cluster_cells[label] = []
                cluster_cells[label].append((r, c))
        
        # Compute centroid for each cluster
        cache_key = (self._geometry_size, self._overlay_version)
        if self._centroid_cache is not None and self._centroid_cache_key == cache_key:
            return dict(self._centroid_cache)

        centroids: dict[object, QPointF] = {}
        for label, cells in cluster_cells.items():
            if not cells:
                continue
            total_x, total_y = 0.0, 0.0
            count = 0
            for r, c in cells:
                polygon = polygons.get((r, c))
                if polygon is None:
                    continue
                rect = polygon.boundingRect()
                total_x += rect.center().x()
                total_y += rect.center().y()
                count += 1
            if count > 0:
                centroids[label] = QPointF(total_x / count, total_y / count)
        
        self._centroid_cache = dict(centroids)
        self._centroid_cache_key = cache_key
        return centroids

    def _hex_neighbors(self, row: int, col: int, rows: int, cols: int) -> list[tuple[int, int]]:
        """Get the neighboring cells for a hexagonal grid cell.
        
        For offset hex grids (odd rows shifted right):
        - Even rows: neighbors are at relative positions different from odd rows
        - Each cell has up to 6 neighbors
        """
        neighbors = []
        # For pointy-top hexagons with odd-row offset:
        # Even row neighbors
        even_row_offsets = [
            (-1, -1), (-1, 0),   # top-left, top-right
            (0, -1), (0, 1),     # left, right
            (1, -1), (1, 0),     # bottom-left, bottom-right
        ]
        # Odd row neighbors
        odd_row_offsets = [
            (-1, 0), (-1, 1),    # top-left, top-right
            (0, -1), (0, 1),     # left, right
            (1, 0), (1, 1),      # bottom-left, bottom-right
        ]
        
        offsets = odd_row_offsets if row % 2 else even_row_offsets
        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc))
        return neighbors

    def _get_neighbor_direction(self, row: int, col: int, neighbor_row: int, neighbor_col: int) -> int:
        """Get the direction index (0-5) for the shared edge between this cell and its neighbor.
        
        For pointy-top hexagons, edges are numbered clockwise starting from the upper-right:
        0 = upper-right edge (vertices 0-1)
        1 = right edge (vertices 1-2)
        2 = lower-right edge (vertices 2-3)
        3 = lower-left edge (vertices 3-4)
        4 = left edge (vertices 4-5)
        5 = upper-left edge (vertices 5-0)
        
        The neighbor offset to edge mapping depends on whether we're in an odd or even row
        due to the hex grid's staggered layout.
        """
        dr = neighbor_row - row
        dc = neighbor_col - col
        is_odd_row = row % 2
        
        # Map (dr, dc) to direction based on row parity
        # In odd rows, the row is shifted right, so right neighbors have different offsets
        if is_odd_row:
            direction_map = {
                (-1, 0): 5,   # neighbor above-left -> upper-left edge
                (-1, 1): 0,   # neighbor above-right -> upper-right edge
                (0, -1): 4,   # neighbor left -> left edge
                (0, 1): 1,    # neighbor right -> right edge
                (1, 0): 3,    # neighbor below-left -> lower-left edge
                (1, 1): 2,    # neighbor below-right -> lower-right edge
            }
        else:
            # In even rows, neighbors are at different relative positions
            direction_map = {
                (-1, -1): 5,  # neighbor above-left -> upper-left edge
                (-1, 0): 0,   # neighbor above-right -> upper-right edge
                (0, -1): 4,   # neighbor left -> left edge
                (0, 1): 1,    # neighbor right -> right edge
                (1, -1): 3,   # neighbor below-left -> lower-left edge
                (1, 0): 2,    # neighbor below-right -> lower-right edge
            }
        return direction_map.get((dr, dc), -1)

    def _get_hex_edge_points(self, polygon: QPolygonF, direction: int) -> tuple[QPointF, QPointF]:
        """Get the two vertex points forming the edge in the given direction.
        
        For pointy-top hexagons, vertices are arranged starting from top, going clockwise:
        0 = top, 1 = upper-right, 2 = lower-right, 3 = bottom, 4 = lower-left, 5 = upper-left
        
        Edge direction to vertex indices (edges are sides of the hexagon):
        - Direction 0 (upper-right edge): vertices 0 and 1
        - Direction 1 (right edge): vertices 1 and 2
        - Direction 2 (lower-right edge): vertices 2 and 3
        - Direction 3 (lower-left edge): vertices 3 and 4
        - Direction 4 (left edge): vertices 4 and 5
        - Direction 5 (upper-left edge): vertices 5 and 0
        """
        if polygon.count() < 6:
            return (QPointF(), QPointF())
        
        # Map direction to edge vertex indices
        edge_map = {
            0: (0, 1),  # upper-right edge
            1: (1, 2),  # right edge
            2: (2, 3),  # lower-right edge
            3: (3, 4),  # lower-left edge
            4: (4, 5),  # left edge
            5: (5, 0),  # upper-left edge
        }
        
        v1_idx, v2_idx = edge_map.get(direction, (0, 1))
        return (polygon.at(v1_idx), polygon.at(v2_idx))

    def _compute_cluster_boundary_edges(
        self, polygons: dict[tuple[int, int], QPolygonF]
    ) -> List[tuple[QPointF, QPointF, object, object]]:
        """Compute all boundary edges between different clusters.
        
        Returns a list of (point1, point2, cluster_label1, cluster_label2) tuples
        representing edges that should be drawn. Both cluster labels are included
        so we can draw dual-color thin lines (one for each cluster).
        """
        if self._cluster_overlay is None or self._values is None:
            return []

        cache_key = (self._geometry_size, self._overlay_version)
        if self._boundary_cache is not None and self._boundary_cache_key == cache_key:
            return list(self._boundary_cache)
        
        rows, cols = self._values.shape
        edges: List[tuple[QPointF, QPointF, object, object]] = []
        # Use frozenset for efficient edge deduplication
        processed: Set[frozenset[tuple[int, int]]] = set()
        
        for r in range(rows):
            for c in range(cols):
                cell_label = self._cluster_label_at(r, c)
                if cell_label is None:
                    continue
                
                polygon = polygons.get((r, c))
                if polygon is None:
                    continue
                
                neighbors = self._hex_neighbors(r, c, rows, cols)
                for nr, nc in neighbors:
                    # Avoid processing the same edge twice using frozenset
                    edge_key = frozenset(((r, c), (nr, nc)))
                    if edge_key in processed:
                        continue
                    processed.add(edge_key)
                    
                    neighbor_label = self._cluster_label_at(nr, nc)
                    
                    # Draw edge if:
                    # 1. Neighbor is in a different cluster
                    # 2. Neighbor has no cluster label (boundary with unclustered area)
                    if neighbor_label == cell_label:
                        continue
                    
                    # Get the direction from this cell to neighbor
                    direction = self._get_neighbor_direction(r, c, nr, nc)
                    if direction < 0:
                        continue
                    
                    # Get the edge points
                    p1, p2 = self._get_hex_edge_points(polygon, direction)
                    if p1.isNull() or p2.isNull():
                        continue
                    
                    # Store both cluster labels for dual-color edge drawing
                    edges.append((p1, p2, cell_label, neighbor_label))
        
        self._boundary_cache = list(edges)
        self._boundary_cache_key = cache_key
        return edges

    def _draw_cluster_boundary_edges(
        self,
        painter: QPainter,
        radius: float,
        *,
        polygons: Optional[dict[tuple[int, int], QPolygonF]] = None,
    ) -> None:
        """Draw cluster boundary edges with dual thin lines (one per cluster).
        
        Each edge between two clusters is drawn as two thin parallel lines,
        one in each cluster's color, so both clusters are clearly visible
        without overlapping.
        """
        if polygons is None:
            geometry = self._prepare_geometry()
            if geometry is None:
                return
            polygons = geometry["polygons"]
        edges = self._compute_cluster_boundary_edges(polygons)
        
        # Use thin lines that don't overlap - total width ~2/3 of previous thick line
        base_width = max(2, int(radius * 0.15))
        offset_dist = base_width * 0.6  # Distance to offset each line from center
        
        for p1, p2, label1, label2 in edges:
            # Calculate perpendicular offset direction
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            length = (dx * dx + dy * dy) ** 0.5
            if length <= 0.001:  # Skip degenerate edges
                continue
            
            # Perpendicular unit vector
            perp_x = -dy / length
            perp_y = dx / length
            
            # Offset points for first line (cluster label1)
            p1_offset1 = QPointF(p1.x() + perp_x * offset_dist, p1.y() + perp_y * offset_dist)
            p2_offset1 = QPointF(p2.x() + perp_x * offset_dist, p2.y() + perp_y * offset_dist)
            
            # Offset points for second line (cluster label2)
            p1_offset2 = QPointF(p1.x() - perp_x * offset_dist, p1.y() - perp_y * offset_dist)
            p2_offset2 = QPointF(p2.x() - perp_x * offset_dist, p2.y() - perp_y * offset_dist)
            
            # Draw first cluster's line
            color1 = self._cluster_border_color(label1)
            if color1 is None:
                color1 = QColor(255, 255, 255)
            pen1 = QPen(color1, base_width)
            pen1.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen1)
            painter.drawLine(p1_offset1, p2_offset1)
            
            # Draw second cluster's line (if it has a label)
            if label2 is not None:
                color2 = self._cluster_border_color(label2)
                if color2 is None:
                    color2 = QColor(200, 200, 200)
                pen2 = QPen(color2, base_width)
                pen2.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen2)
                painter.drawLine(p1_offset2, p2_offset2)

    def _draw_cluster_centered_labels(
        self,
        painter: QPainter,
        polygons: dict[tuple[int, int], QPolygonF],
        backgrounds: dict[tuple[int, int], QColor],
        radius: float,
    ) -> None:
        """Draw cluster labels at the center of each cluster.
        
        Shows custom cluster names if set, otherwise shows the cluster ID.
        """
        centroids = self._compute_cluster_centroids(polygons)
        if not centroids:
            return
        
        # Use a slightly larger font for cluster labels
        font = painter.font()
        font.setPointSize(max(8, int(radius * 0.5)))
        font.setBold(True)
        painter.setFont(font)
        
        for label, center in centroids.items():
            # Get display text: custom name or cluster ID
            display_text = ""
            try:
                cluster_id = int(label)
                custom_name = self._cluster_names.get(cluster_id, "")
                if custom_name:
                    display_text = custom_name
                else:
                    display_text = str(cluster_id)
            except (ValueError, TypeError):
                display_text = str(label)
            
            if not display_text:
                continue
            
            # Find background color at centroid (use nearest cell)
            # Find nearest cell to determine text color
            bg_color = QColor(128, 128, 128)  # Default gray
            min_dist = float("inf")
            for (r, c), polygon in polygons.items():
                rect = polygon.boundingRect()
                cell_center = rect.center()
                dist = (cell_center.x() - center.x()) ** 2 + (cell_center.y() - center.y()) ** 2
                if dist < min_dist:
                    min_dist = dist
                    bg_color = backgrounds.get((r, c), QColor(128, 128, 128))
            
            # Draw text with appropriate contrast
            painter.setPen(self._text_pen(bg_color))
            
            # Create a rect around the centroid for text drawing
            text_rect = painter.fontMetrics().boundingRect(display_text)
            text_rect.moveCenter(center.toPoint())
            # Make the rect a bit larger for drawing
            text_rect.adjust(-5, -2, 5, 2)
            
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignCenter,
                display_text,
            )

    # ------------------------------------------------------------------
    def _compute_hex_layout(self, rows: int, cols: int) -> Optional[dict]:
        if rows <= 0 or cols <= 0:
            return None

        radius = 1.0
        dx = math.sqrt(3) * radius
        dy = 1.5 * radius
        x_offset = (math.sqrt(3) / 2) * radius
        horizontal_radius = math.cos(math.pi / 6) * radius

        centers: list[tuple[int, int, float, float]] = []
        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")

        for r in range(rows):
            for c in range(cols):
                cx = c * dx + (x_offset if r % 2 else 0.0)
                cy = r * dy
                centers.append((r, c, cx, cy))
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

        min_x_total = min_x - horizontal_radius
        max_x_total = max_x + horizontal_radius
        min_y_total = min_y - radius
        max_y_total = max_y + radius

        return {
            "centers": centers,
            "radius": radius,
            "bounds": (min_x_total, min_y_total, max_x_total, max_y_total),
        }

    def _prepare_geometry(self) -> Optional[dict]:
        if self._hex_layout is None or self._values is None:
            return None

        size = (self.width(), self.height())
        if size[0] <= 0 or size[1] <= 0:
            return None

        if self._geometry_cache is not None and self._geometry_size == size:
            return self._geometry_cache

        layout = self._hex_layout
        bounds = layout.get("bounds") if layout else None
        centers = layout.get("centers") if layout else None
        radius = layout.get("radius") if layout else 1.0
        if bounds is None or centers is None:
            return None

        min_x, min_y, max_x, max_y = bounds
        width_norm = max(max_x - min_x, 1e-6)
        height_norm = max(max_y - min_y, 1e-6)
        scale = min(size[0] / width_norm, size[1] / height_norm)
        offset_x = (size[0] - width_norm * scale) / 2
        offset_y = (size[1] - height_norm * scale) / 2

        scaled_radius = radius * scale
        polygons: dict[tuple[int, int], QPolygonF] = {}
        centers_map: dict[tuple[int, int], QPointF] = {}

        for row, col, cx, cy in centers:
            sx = (cx - min_x) * scale + offset_x
            sy = (cy - min_y) * scale + offset_y
            polygon = self._create_hex_polygon(QPointF(sx, sy), scaled_radius)
            polygons[(row, col)] = polygon
            centers_map[(row, col)] = QPointF(sx, sy)

        self._geometry_cache = {
            "polygons": polygons,
            "centers": centers_map,
            "radius": scaled_radius,
            "scale": scale,
        }
        self._geometry_size = size
        return self._geometry_cache

    @staticmethod
    def _create_hex_polygon(center: QPointF, radius: float) -> QPolygonF:
        points = []
        for i in range(6):
            angle = math.pi / 3 * i - math.pi / 2
            x = center.x() + radius * math.cos(angle)
            y = center.y() + radius * math.sin(angle)
            points.append(QPointF(x, y))
        return QPolygonF(points)
