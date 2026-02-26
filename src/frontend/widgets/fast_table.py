from __future__ import annotations
# @ai(gpt-5, codex, refactor, 2026-02-26)
import logging

logger = logging.getLogger(__name__)
# frontend/widgets/fast_table.py

from typing import Optional, Callable, Iterable, Sequence

import pandas as pd
from PySide6.QtCore import Qt, QPoint, QRect, QSortFilterProxyModel, Signal, QEvent, QTimer, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QAction, QColor, QPainter
from PySide6.QtWidgets import (

    QTableView, QHeaderView, QAbstractItemView, QMenu, QWidget, QApplication,
    QStyledItemDelegate, QStyleOptionViewItem, QStyle
)
from ..localization import tr


class FastDataFrameModel(QAbstractTableModel):
    """High-throughput pandas-backed model used internally by FastTable."""

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        parent=None,
        *,
        float_format: str = "{:.4f}",
        editable_columns: Optional[Sequence[str]] = None,
        editable_all: bool = False,
        include_index: bool = False,
    ) -> None:
        super().__init__(parent)
        self._float_format = float_format
        self._editable_columns = set(editable_columns or [])
        self._editable_all = bool(editable_all)
        self._include_index = bool(include_index)
        self._df = pd.DataFrame()
        self.set_dataframe(df if df is not None else pd.DataFrame())

    def rowCount(self, parent=QModelIndex()):  # noqa: N802
        if parent.isValid():
            return 0
        return int(self._df.shape[0]) if self._df is not None else 0

    def columnCount(self, parent=QModelIndex()):  # noqa: N802
        if parent.isValid():
            return 0
        return int(self._df.shape[1]) if self._df is not None else 0

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        return self._data_impl(index, role)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        return self._header_impl(section, orientation, role)

    def flags(self, index):
        return self._flags_impl(index)

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):  # noqa: N802
        return self._set_data_impl(index, value, role)

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):  # noqa: N802
        self._sort_impl(column, order)

    def _normalize_dataframe(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        out = pd.DataFrame() if df is None else df.copy()
        if self._include_index:
            if isinstance(out.index, pd.MultiIndex) or out.index.name is not None:
                out = out.reset_index()
        return out

    def set_dataframe(self, df: Optional[pd.DataFrame]) -> None:
        new_df = self._normalize_dataframe(df)
        self.beginResetModel()
        self._df = new_df
        self.endResetModel()

    def dataframe(self) -> pd.DataFrame:
        return self._df.copy() if self._df is not None else pd.DataFrame()

    def set_editable_columns(self, columns: Iterable[str]) -> None:
        self._editable_columns = set(columns or [])

    def set_editable_all(self, enabled: bool) -> None:
        self._editable_all = bool(enabled)

    def set_float_format(self, float_format: str) -> None:
        self._float_format = float_format or "{:.4f}"

    def _data_impl(self, index, role: int):
        if not index.isValid() or self._df is None:
            return None
        try:
            value = self._df.iat[index.row(), index.column()]
        except Exception:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self._format_value(value)
        if role in (Qt.ItemDataRole.EditRole, Qt.ItemDataRole.UserRole):
            return value
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(value, (int, float)) and not pd.isna(value):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        return None

    def _header_impl(self, section: int, orientation: Qt.Orientation, role: int):
        if role != Qt.ItemDataRole.DisplayRole or self._df is None:
            return None
        if orientation == Qt.Orientation.Horizontal:
            try:
                return str(self._df.columns[section])
            except Exception:
                return None
        return str(section + 1)

    def _flags_impl(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if self._editable_all:
            return base | Qt.ItemFlag.ItemIsEditable
        try:
            column_name = self._df.columns[index.column()]
        except Exception:
            return base
        if column_name in self._editable_columns:
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def _set_data_impl(self, index, value, role: int):
        if role != Qt.ItemDataRole.EditRole or not index.isValid() or self._df is None:
            return False
        if not self._editable_all:
            try:
                column_name = self._df.columns[index.column()]
            except Exception:
                return False
            if column_name not in self._editable_columns:
                return False
        try:
            self._df.iat[index.row(), index.column()] = value
        except Exception:
            return False
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
        return True

    def _sort_impl(self, column: int, order: Qt.SortOrder):
        if self._df is None or self._df.empty:
            return
        if column < 0 or column >= self._df.shape[1]:
            return
        ascending = order != Qt.SortOrder.DescendingOrder
        series = self._df.iloc[:, column]
        numeric = pd.to_numeric(series, errors="coerce")
        sort_key = numeric if bool(numeric.notna().any()) else series.astype(str)
        try:
            sorted_index = sort_key.sort_values(
                ascending=ascending,
                kind="mergesort",
                na_position="last",
            ).index
        except Exception:
            return
        self.layoutAboutToBeChanged.emit()
        self._df = self._df.loc[sorted_index].reset_index(drop=True)
        self.layoutChanged.emit()

    def _format_value(self, value) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        if isinstance(value, float):
            try:
                return self._float_format.format(value)
            except Exception:
                return str(value)
        return str(value)


class HoverRowDelegate(QStyledItemDelegate):
    """
    Minimal hover highlight: tints the entire row under the mouse.
    Does NOT follow selection behavior, does NOT do column/cell modes.
    Tries to be as cheap as possible:
      - tracks hovered row
      - only repaints the old+new hovered row rectangles
      - suppresses hover while mouse is pressed (optional)
    """
    def __init__(self, view: QTableView, color: QColor):
        super().__init__(view)
        self._view = view
        self._hover_row = -1
        self._color = color

        view.setMouseTracking(True)
        if view.viewport():
            view.viewport().setMouseTracking(True)
            view.viewport().installEventFilter(self)

    def set_color(self, color: QColor) -> None:
        self._color = color
        vp = self._view.viewport()
        if vp:
            vp.update()

    def eventFilter(self, obj, ev):
        # Guard if view is being destroyed
        try:
            vp = self._view.viewport()
        except Exception:
            return False

        if obj is vp:
            t = ev.type()
            if t == QEvent.Type.MouseMove:
                # Qt6: QMouseEvent.position() is QPointF
                try:
                    pos = ev.position().toPoint()
                except Exception:
                    # fallback for older events
                    try:
                        pos = ev.pos()
                    except Exception:
                        return False

                idx = self._view.indexAt(pos)
                if idx.isValid():
                    self._set_hover_row(idx.row())
                else:
                    self._set_hover_row(-1)

            elif t == QEvent.Type.Leave:
                self._set_hover_row(-1)

        return super().eventFilter(obj, ev)

    def _row_rect(self, row: int) -> QRect | None:
        if row < 0:
            return None
        vp = self._view.viewport()
        if not vp:
            return None
        y = self._view.rowViewportPosition(row)
        if y < 0:
            return None
        h = self._view.rowHeight(row)
        if h <= 0:
            return None
        return QRect(0, y, vp.width(), h)

    def _set_hover_row(self, row: int) -> None:
        if row == self._hover_row:
            return

        old = self._hover_row
        self._hover_row = row

        vp = self._view.viewport()
        if not vp:
            return

        old_rect = self._row_rect(old)
        new_rect = self._row_rect(row)
        if old_rect:
            vp.update(old_rect)
        if new_rect and new_rect != old_rect:
            vp.update(new_rect)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        opt = QStyleOptionViewItem(option)

        # Optional: suppress hover while actively selecting/pressing
        if getattr(self._view, "_suppress_hover", False):
            return super().paint(painter, opt, index)

        if (
            self._color.isValid()
            and index.isValid()
            and index.row() == self._hover_row
            and not (opt.state & QStyle.StateFlag.State_Selected)
        ):
            painter.save()
            painter.fillRect(opt.rect, self._color)
            painter.restore()

        super().paint(painter, opt, index)


class FastTable(QTableView):
    rowActivated = Signal(int)            # row index in the *view*
    selectionChangedInstant = Signal()    # emitted immediately on selection change

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        select: str = "rows",                 # "rows" | "columns" | "items"
        single_selection: bool = True,
        tint_current_selection: bool = True,  # now used: enables hover row tint
        editable: bool = False,
        auto_resize_once: bool = True,
        stretch_column: int = -1,
        alt_row_colors: bool = True,
        fixed_row_height: int = 24,
        min_column_width: int = 80,
        context_menu_builder: Optional[Callable[[QMenu, QPoint, "FastTable"], None]] = None,
        initial_uniform_column_widths: bool = False,
        initial_uniform_column_count: int | None = None,
        sorting_enabled: bool = True,
        uniform_row_heights: bool = True,
        show_grid: bool = True,
    ):
        super().__init__(parent)

        self._auto_resize_once = bool(auto_resize_once)
        self._did_initial_resize = False
        self._stretch_column = stretch_column
        self._min_column_width = max(20, int(min_column_width))
        self._fixed_row_height = int(fixed_row_height)
        self._context_menu_builder = context_menu_builder
        self._uniform_column_widths = bool(initial_uniform_column_widths)
        try:
            parsed_uniform_count = int(initial_uniform_column_count) if initial_uniform_column_count is not None else 0
        except Exception:
            parsed_uniform_count = 0
        self._initial_uniform_column_count = max(0, parsed_uniform_count)
        self._uniform_applied = False
        self._dataframe_model: Optional[FastDataFrameModel] = None

        # hover
        self._hover_delegate: Optional[HoverRowDelegate] = None
        self._hover_enabled = bool(tint_current_selection)
        self._suppress_hover = False  # used by delegate to avoid "flash" during press

        # selection behavior
        sel_beh_map = {
            "rows": QAbstractItemView.SelectionBehavior.SelectRows,
            "columns": QAbstractItemView.SelectionBehavior.SelectColumns,
            "items": QAbstractItemView.SelectionBehavior.SelectItems,
        }
        self.setSelectionBehavior(sel_beh_map.get(select, QAbstractItemView.SelectionBehavior.SelectRows))
        self.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
            if single_selection
            else QAbstractItemView.SelectionMode.ExtendedSelection
        )

        # editing
        self.set_editable(editable)

        # performance-friendly defaults
        self.setWordWrap(False)
        self.setAlternatingRowColors(bool(alt_row_colors))
        self.setSortingEnabled(bool(sorting_enabled))
        self.setShowGrid(bool(show_grid))
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

        # headers
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setStretchLastSection(False)
        hh.setMinimumSectionSize(self._min_column_width)

        vh = self.verticalHeader()
        vh.setVisible(False)
        vh.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        vh.setDefaultSectionSize(self._fixed_row_height)
        vh.setMinimumSectionSize(self._fixed_row_height)

        # activation -> rowActivated
        self.doubleClicked.connect(lambda idx: self.rowActivated.emit(idx.row()))
        self.activated.connect(lambda idx: self.rowActivated.emit(idx.row()))

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # track selection model for safe disconnect/reconnect
        self._last_sel_model = None

        # install hover delegate if enabled
        if self._hover_enabled:
            self._install_hover(alpha=36)

    # Optional suppression to avoid hover repaint thrash while clicking/dragging selection
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._suppress_hover = True
            QTimer.singleShot(0, self._clear_hover_suppression)
        super().mousePressEvent(ev)

    def _clear_hover_suppression(self):
        self._suppress_hover = False
        vp = self.viewport()
        if vp:
            vp.update()

    # ---------- Public API ----------
    def set_editable(self, editable: bool):
        if editable:
            self.setEditTriggers(
                QAbstractItemView.EditTrigger.EditKeyPressed
                | QAbstractItemView.EditTrigger.DoubleClicked
                | QAbstractItemView.EditTrigger.AnyKeyPressed
            )
            m = self.model()
            if isinstance(m, FastDataFrameModel):
                m.set_editable_all(True)
        else:
            self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            m = self.model()
            if isinstance(m, FastDataFrameModel):
                m.set_editable_all(False)

    def set_dataframe(
        self,
        df: Optional[pd.DataFrame],
        *,
        include_index: bool = False,
        editable_columns: Optional[Sequence[str]] = None,
        float_format: str = "{:.4f}",
    ) -> None:
        model = self._ensure_dataframe_model(
            include_index=include_index,
            editable_columns=editable_columns,
            float_format=float_format,
        )
        model.set_dataframe(df)

    def dataframe(self) -> pd.DataFrame:
        model = self.model()
        if isinstance(model, FastDataFrameModel):
            return model.dataframe()
        return pd.DataFrame()

    def set_dataframe_editable_columns(self, columns: Iterable[str]) -> None:
        model = self._ensure_dataframe_model()
        model.set_editable_columns(columns)

    def _ensure_dataframe_model(
        self,
        *,
        include_index: bool = False,
        editable_columns: Optional[Sequence[str]] = None,
        float_format: str = "{:.4f}",
    ) -> FastDataFrameModel:
        existing = self.model()
        if isinstance(existing, FastDataFrameModel):
            if editable_columns is not None:
                existing.set_editable_columns(editable_columns)
            if float_format:
                existing.set_float_format(float_format)
            existing._include_index = bool(include_index)
            return existing
        model = FastDataFrameModel(
            pd.DataFrame(),
            parent=self,
            include_index=include_index,
            editable_columns=editable_columns,
            float_format=float_format,
            editable_all=(self.editTriggers() != QAbstractItemView.EditTrigger.NoEditTriggers),
        )
        self._dataframe_model = model
        self.setModel(model)
        return model

    def enable_hover_tint(self, enabled: bool, alpha: int = 36):
        """
        Enables/disables hover row tint at runtime.
        """
        enabled = bool(enabled)
        if enabled and self._hover_delegate is None:
            self._install_hover(alpha=alpha)
        elif not enabled and self._hover_delegate is not None:
            if self.itemDelegate() is self._hover_delegate:
                self.setItemDelegate(None)
            self._hover_delegate = None
        self._hover_enabled = enabled

    def map_to_source_row(self, row: int) -> int:
        m = self.model()
        if isinstance(m, QSortFilterProxyModel):
            return m.mapToSource(m.index(row, 0)).row()
        return row

    def set_stretch_column(self, col: int | None):
        self._stretch_column = col
        QTimer.singleShot(0, self._apply_stretch_column)

    def reapply_uniform_column_widths(self) -> None:
        """Force one-shot recomputation of initial uniform column widths."""
        if not self._uniform_column_widths:
            return
        self._uniform_applied = False
        QTimer.singleShot(0, self._apply_uniform_column_widths)

    def set_default_context_menu(self):
        def build(menu: QMenu, _pos: QPoint, table: FastTable):
            act_copy = QAction(tr("Copy cell"), menu)
            act_copy.triggered.connect(table._copy_current_cell)
            menu.addAction(act_copy)

            act_copy_row = QAction(tr("Copy row"), menu)
            act_copy_row.triggered.connect(table._copy_current_row)
            menu.addAction(act_copy_row)

        self._context_menu_builder = build

    # ---------- Model / sizing ----------
    def setModel(self, model):  # noqa: N802
        sorting = self.isSortingEnabled()
        if sorting:
            super().setSortingEnabled(False)
        self.setUpdatesEnabled(False)

        try:
            super().setModel(model)

            if sorting:
                super().setSortingEnabled(True)

            # disconnect old selection model
            prev = getattr(self, "_last_sel_model", None)
            if prev is not None:
                try:
                    prev.selectionChanged.disconnect(self._emit_sel_changed_instant)
                except Exception:
                    logger.warning("Exception in setModel", exc_info=True)

            sel = self.selectionModel()
            if sel:
                try:
                    sel.selectionChanged.connect(self._emit_sel_changed_instant)
                    self._last_sel_model = sel
                except Exception:
                    self._last_sel_model = None

            if model is None:
                return

            # initial sizing (ONE shot only)
            if self._auto_resize_once and not self._did_initial_resize:
                if self._uniform_column_widths:
                    self._apply_uniform_column_widths()
                else:
                    # keep it cheap
                    self._apply_min_column_widths()
                self._did_initial_resize = True
            else:
                self._apply_min_column_widths()

            self._apply_stretch_column()

            # refresh hover tint color from palette if installed
            if self._hover_delegate is not None:
                self._hover_delegate.set_color(self._make_mild_tint(36))
        finally:
            self.setUpdatesEnabled(True)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._uniform_column_widths and self.model() and not self._uniform_applied:
            self._apply_uniform_column_widths()
            return
        self._grow_stretch_column_to_viewport()

    def showEvent(self, ev):
        super().showEvent(ev)
        if self._uniform_column_widths and self.model() and not self._uniform_applied:
            self._apply_uniform_column_widths()
        self._grow_stretch_column_to_viewport()

    def _apply_uniform_column_widths(self) -> None:
        model = self.model()
        count = int(self._initial_uniform_column_count)
        if model:
            try:
                count = max(count, int(model.columnCount()))
            except Exception:
                count = max(count, 0)
        if count <= 0:
            return
        vp = self.viewport()
        available = vp.width() if vp else self.width()
        if available <= 0:
            available = count * 120
        width = max(self._min_column_width, int(available / max(1, count)))
        for col in range(count):
            self.setColumnWidth(col, width)
        self._uniform_applied = True

    def _apply_min_column_widths(self) -> None:
        model = self.model()
        if not model:
            return
        for c in range(model.columnCount()):
            if self.columnWidth(c) < self._min_column_width:
                self.setColumnWidth(c, self._min_column_width)

    def _apply_stretch_column(self) -> None:
        model = self.model()
        if not model:
            self._stretch_col_idx = None
            return
        count = model.columnCount()
        if count <= 0:
            self._stretch_col_idx = None
            return

        if self._stretch_column is None:
            self._stretch_col_idx = None
        else:
            self._stretch_col_idx = min(
                self._stretch_column if self._stretch_column >= 0 else (count - 1),
                count - 1
            )

    def _sum_other_columns_width(self, skip_idx: int) -> int:
        model = self.model()
        if not model:
            return 0
        total = 0
        for c in range(model.columnCount()):
            if c != skip_idx:
                total += self.columnWidth(c)
        return total

    def _grow_stretch_column_to_viewport(self) -> None:
        idx = getattr(self, "_stretch_col_idx", None)
        if idx is None or idx < 0 or not self.model():
            return
        vp = self.viewport()
        if not vp:
            return
        available = vp.width() - self._sum_other_columns_width(idx)
        if available <= 0:
            return
        cur = self.columnWidth(idx)
        if available > cur:
            self.setColumnWidth(idx, available)

    # ---------- Context menu ----------
    def contextMenuEvent(self, ev):
        if self._context_menu_builder is None:
            return super().contextMenuEvent(ev)
        menu = QMenu(self)
        self._context_menu_builder(menu, ev.globalPos(), self)
        if not menu.isEmpty():
            menu.exec(ev.globalPos())

    # ---------- Clipboard helpers ----------
    def _copy_current_cell(self):
        idx = self.currentIndex()
        if not idx.isValid():
            return
        QApplication.clipboard().setText(str(idx.data()))

    def _copy_current_row(self):
        idx = self.currentIndex()
        if not idx.isValid():
            return
        model = self.model()
        if not model:
            return
        row = idx.row()
        texts = [str(model.index(row, c).data()) for c in range(model.columnCount())]
        QApplication.clipboard().setText("\t".join(texts))

    # ---------- Signals ----------
    def _emit_sel_changed_instant(self, *_):
        self.selectionChangedInstant.emit()

    # ---------- Hover internals ----------
    def _install_hover(self, alpha: int = 36):
        self._hover_delegate = HoverRowDelegate(self, self._make_mild_tint(alpha))
        self.setItemDelegate(self._hover_delegate)

    def _make_mild_tint(self, alpha: int) -> QColor:
        base = self.palette().highlight().color()
        c = QColor(base)
        c.setAlpha(max(0, min(255, int(alpha))))
        return c
