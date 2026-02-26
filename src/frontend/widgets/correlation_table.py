
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHeaderView, QTableView, QAbstractItemView


class CorrelationTableModel(QAbstractTableModel):
    """Table model that renders a correlation heatmap with colored cells."""

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        parent=None,
        *,
        symmetric: bool = False,
        show_values: bool = True,
        float_format: str = "{:.4f}",
    ) -> None:
        super().__init__(parent)
        self._df = pd.DataFrame() if df is None else df.copy()
        self._symmetric = symmetric
        self._show_values = show_values
        self._float_format = float_format
        self._vmin = 0.0
        self._vmax = 1.0
        self._recalculate_bounds()

    def set_dataframe(self, df: Optional[pd.DataFrame], *, symmetric: Optional[bool] = None, show_values: Optional[bool] = None) -> None:
        if symmetric is not None:
            self._symmetric = symmetric
        if show_values is not None:
            self._show_values = show_values
        new_df = pd.DataFrame() if df is None else df.copy()
        self._apply_dataframe_incremental(new_df)
        self._recalculate_bounds()
        if self.rowCount() > 0 and self.columnCount() > 0:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.rowCount() - 1, self.columnCount() - 1),
                [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.BackgroundRole],
            )

    def _apply_dataframe_incremental(self, new_df: pd.DataFrame) -> None:
        if self._df is None:
            self._df = pd.DataFrame()

        new_columns = list(new_df.columns)
        self._sync_columns(new_columns)
        self._sync_rows(int(new_df.shape[0]))

        self._df = new_df.reindex(columns=self._df.columns).copy()

    def _sync_rows(self, new_rows: int) -> None:
        old_rows = self.rowCount()
        if old_rows > new_rows:
            self.beginRemoveRows(QModelIndex(), new_rows, old_rows - 1)
            self._df = self._df.iloc[:new_rows].copy()
            self.endRemoveRows()
        elif new_rows > old_rows:
            self.beginInsertRows(QModelIndex(), old_rows, new_rows - 1)
            append_rows = pd.DataFrame(np.nan, index=range(new_rows - old_rows), columns=self._df.columns)
            self._df = pd.concat([self._df, append_rows], ignore_index=True)
            self.endInsertRows()

    def _sync_columns(self, new_columns: list[object]) -> None:
        current = list(self._df.columns)
        target_set = set(new_columns)

        for idx in range(len(current) - 1, -1, -1):
            if current[idx] in target_set:
                continue
            self.beginRemoveColumns(QModelIndex(), idx, idx)
            self._df = self._df.drop(columns=[self._df.columns[idx]])
            self.endRemoveColumns()
            current.pop(idx)

        for idx, name in enumerate(new_columns):
            if idx < len(current) and current[idx] == name:
                continue
            if name in current:
                continue
            self.beginInsertColumns(QModelIndex(), idx, idx)
            self._df.insert(idx, name, np.nan)
            self.endInsertColumns()
            current.insert(idx, name)

        if current != new_columns:
            self.layoutAboutToBeChanged.emit()
            self._df = self._df.reindex(columns=new_columns)
            self.layoutChanged.emit()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return int(self._df.shape[0]) if self._df is not None else 0

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return int(self._df.shape[1]) if self._df is not None else 0

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if not index.isValid() or self._df is None:
            return None
        try:
            value = float(self._df.iat[index.row(), index.column()])
        except Exception:
            value = float("nan")

        if role == Qt.ItemDataRole.DisplayRole:
            if not self._show_values or np.isnan(value):
                return ""
            try:
                return self._float_format.format(value)
            except Exception:
                return str(value)
        if role == Qt.ItemDataRole.BackgroundRole:
            return self._value_to_color(value, self._vmin, self._vmax)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return None

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole or self._df is None:
            return None
        if orientation == Qt.Orientation.Horizontal:
            try:
                return str(self._df.columns[section])
            except Exception:
                return None
        try:
            return str(self._df.index[section])
        except Exception:
            return str(section + 1)

    def _recalculate_bounds(self) -> None:
        if self._df is None or self._df.empty:
            self._vmin, self._vmax = 0.0, 1.0
            return
        values = self._df.astype(float).to_numpy(dtype=float)
        if self._symmetric:
            vmax = np.nanmax(np.abs(values))
            vmin = -vmax
        else:
            vmin = np.nanmin(values)
            vmax = np.nanmax(values)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        self._vmin, self._vmax = float(vmin), float(vmax)

    @staticmethod
    def _value_to_color(value: float, vmin: float, vmax: float) -> QColor:
        if np.isnan(value):
            return QColor(64, 64, 64)
        if vmax <= vmin:
            ratio = 0.5
        else:
            ratio = (value - vmin) / (vmax - vmin)
            ratio = max(0.0, min(1.0, ratio))
        start = QColor(54, 92, 180)
        end = QColor(235, 140, 52)
        r = start.red() + (end.red() - start.red()) * ratio
        g = start.green() + (end.green() - start.green()) * ratio
        b = start.blue() + (end.blue() - start.blue()) * ratio
        return QColor(int(r), int(g), int(b))


class CorrelationTable(QTableView):
    """Correlation heatmap table with colorized cells."""

    def __init__(self, parent=None, *, symmetric: bool = False, show_values: bool = True) -> None:
        super().__init__(parent)
        self._model = CorrelationTableModel(
            pd.DataFrame(),
            parent=self,
            symmetric=symmetric,
            show_values=show_values,
        )
        self.setModel(self._model)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)  # type: ignore
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)  # type: ignore
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)  # type: ignore
        self.setWordWrap(False)
        self.setAlternatingRowColors(False)
        self.setSortingEnabled(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)  # type: ignore
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)  # type: ignore

    def set_dataframe(self, df: Optional[pd.DataFrame], *, symmetric: Optional[bool] = None, show_values: Optional[bool] = None) -> None:
        self._model.set_dataframe(df, symmetric=symmetric, show_values=show_values)


__all__ = ["CorrelationTable", "CorrelationTableModel"]
