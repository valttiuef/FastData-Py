
from __future__ import annotations
from typing import Iterable, Optional, Sequence

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


class DataFrameTableModel(QAbstractTableModel):
    """Lightweight table model for displaying pandas DataFrames."""

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        parent=None,
        *,
        float_format: str = "{:.4f}",
        editable_columns: Optional[Sequence[str]] = None,
        include_index: bool = True,
    ) -> None:
        super().__init__(parent)
        self._float_format = float_format
        self._editable_columns = set(editable_columns or [])
        self._include_index = include_index
        self._df = pd.DataFrame() if df is None else df.copy()
        self._normalize_dataframe()

    def set_dataframe(self, df: Optional[pd.DataFrame]) -> None:
        new_df = pd.DataFrame() if df is None else df.copy()
        if self._include_index:
            if isinstance(new_df.index, pd.MultiIndex):
                new_df = new_df.reset_index()
            elif new_df.index.name is not None:
                new_df = new_df.reset_index()
        self._apply_dataframe_incremental(new_df)

    def set_editable_columns(self, columns: Iterable[str]) -> None:
        self._editable_columns = set(columns or [])

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
            value = self._df.iat[index.row(), index.column()]
        except Exception:
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            return self._format_value(value)
        if role == Qt.ItemDataRole.UserRole:
            return value
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(value, (int, float)) and not pd.isna(value):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
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
        return str(section + 1)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:  # type: ignore[override]
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        column = self._df.columns[index.column()] if self._df is not None else None
        if column in self._editable_columns:
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole):  # type: ignore[override]
        if role != Qt.ItemDataRole.EditRole or not index.isValid() or self._df is None:
            return False
        column = self._df.columns[index.column()]
        if column not in self._editable_columns:
            return False
        try:
            self._df.iat[index.row(), index.column()] = value
        except Exception:
            return False
        self.dataChanged.emit(index, index, [role])
        return True

    def dataframe(self) -> pd.DataFrame:
        return self._df.copy() if self._df is not None else pd.DataFrame()

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:  # type: ignore[override]
        if self._df is None or self._df.empty:
            return
        if column < 0 or column >= self._df.shape[1]:
            return

        ascending = order != Qt.SortOrder.DescendingOrder
        series = self._df.iloc[:, column]

        numeric = pd.to_numeric(series, errors="coerce")
        use_numeric = bool(numeric.notna().any())
        key = numeric if use_numeric else series.astype(str)

        try:
            sorted_index = key.sort_values(
                ascending=ascending,
                kind="mergesort",
                na_position="last",
            ).index
        except Exception:
            sorted_index = self._df.index

        self.layoutAboutToBeChanged.emit()
        self._df = self._df.loc[sorted_index].reset_index(drop=True)
        self.layoutChanged.emit()

    def _normalize_dataframe(self) -> None:
        if self._df is None:
            self._df = pd.DataFrame()
            return
        if not self._include_index:
            return
        if isinstance(self._df.index, pd.MultiIndex):
            self._df = self._df.reset_index()
        elif self._df.index.name is not None:
            self._df = self._df.reset_index()

    def _apply_dataframe_incremental(self, new_df: pd.DataFrame) -> None:
        if self._df is None:
            self._df = pd.DataFrame()

        new_columns = list(new_df.columns)
        self._sync_columns(new_columns)

        old_rows = self.rowCount()
        new_rows = int(new_df.shape[0])
        if old_rows > new_rows:
            self.beginRemoveRows(QModelIndex(), new_rows, old_rows - 1)
            self._df = self._df.iloc[:new_rows].copy()
            self.endRemoveRows()
        elif new_rows > old_rows:
            self.beginInsertRows(QModelIndex(), old_rows, new_rows - 1)
            append_rows = pd.DataFrame(index=range(new_rows - old_rows), columns=self._df.columns)
            self._df = pd.concat([self._df, append_rows], ignore_index=True)
            self.endInsertRows()

        self._df = new_df.reindex(columns=self._df.columns).copy()
        if new_rows > 0 and self.columnCount() > 0:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(new_rows - 1, self.columnCount() - 1),
                [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.UserRole],
            )

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
            self._df.insert(idx, name, pd.NA)
            self.endInsertColumns()
            current.insert(idx, name)

        if current != new_columns:
            self.layoutAboutToBeChanged.emit()
            self._df = self._df.reindex(columns=new_columns)
            self.layoutChanged.emit()

    def _format_value(self, value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, float):
            try:
                return self._float_format.format(value)
            except Exception:
                return str(value)
        return str(value)


__all__ = ["DataFrameTableModel"]
