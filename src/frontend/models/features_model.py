
from __future__ import annotations
from typing import Optional

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex, QSortFilterProxyModel
import logging

logger = logging.getLogger(__name__)


class FeaturesTableModel(QAbstractTableModel):
    """
    Simple table model over a features DataFrame:
    columns keep 'notes' as the last field for consistent stretch-last-column behavior.
    """
    TABLE_COLUMNS: tuple[str, ...] = (
        "feature_id",
        "name",
        "source",
        "unit",
        "type",
        "lag_seconds",
        "tags",
        "notes",
    )
    REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
        "feature_id",
        "name",
        "source",
        "unit",
        "type",
        "lag_seconds",
        "notes",
    )

    def __init__(self, df: Optional[pd.DataFrame] = None, parent=None):
        super().__init__(parent)
        if df is None:
            df = pd.DataFrame(columns=list(self.TABLE_COLUMNS))
        self._df = self._normalize_for_table(df)
        self._search_cache: list[str] = []
        self._build_search_cache()

    def set_dataframe(self, df: pd.DataFrame):
        normalized = self._normalize_for_table(df)
        if self._df is not None:
            try:
                if self._df.equals(normalized):
                    return
            except Exception:
                logger.warning("Exception in set_dataframe equality check", exc_info=True)
        old_rows = self.rowCount()
        new_rows = int(normalized.shape[0])

        if old_rows > new_rows:
            self.beginRemoveRows(QModelIndex(), new_rows, old_rows - 1)
            self._df = self._df.iloc[:new_rows].copy()
            self.endRemoveRows()
        elif new_rows > old_rows:
            self.beginInsertRows(QModelIndex(), old_rows, new_rows - 1)
            self._df = pd.concat(
                [self._df, pd.DataFrame(index=range(new_rows - old_rows), columns=self._df.columns)],
                ignore_index=True,
            )
            self.endInsertRows()

        self._df = normalized
        self._build_search_cache()
        if new_rows > 0 and self.columnCount() > 0:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(new_rows - 1, self.columnCount() - 1),
                [Qt.DisplayRole, Qt.EditRole],
            )

    def _normalize_for_table(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("FeaturesTableModel expects a pandas DataFrame input.")

        expected = set(self.REQUIRED_INPUT_COLUMNS)
        actual = set(str(col) for col in df.columns)
        if not expected.issubset(actual):
            missing = sorted(expected - actual)
            raise ValueError(
                "Invalid features table schema: missing required columns "
                f"{missing}. Received columns={sorted(actual)}."
            )

        out = df.copy()
        if "tags" not in out.columns:
            out["tags"] = [[] for _ in range(len(out))]
        return out.loc[:, list(self.TABLE_COLUMNS)].copy()

    def rowCount(self, parent=QModelIndex()) -> int:
        # For tree models; for a flat table always return 0 for children
        if parent.isValid():
            return 0

        if self._df is None:
            return 0

        return self._df.shape[0]

    def columnCount(self, parent=QModelIndex()) -> int:
        if parent.isValid():
            return 0

        if self._df is None:
            return 0

        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self._df is None:
            return None

        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None

        try:
            val = self._df.iat[index.row(), index.column()]
        except Exception:
            # Out-of-range or other DF issue – fail gracefully
            return None

        key = str(self._df.columns[index.column()]) if self._df is not None else ""
        if key in {"feature_id", "lag_seconds"}:
            as_int = self._coerce_int_display(val)
            if as_int is not None:
                return str(as_int)

        if isinstance(val, (list, tuple, set)) and not val:
            return ""
        # Be safe with pd.isna
        try:
            return "" if pd.isna(val) else str(val)
        except Exception:
            return str(val)

    def raw_value(self, row: int, column: int):
        if self._df is None:
            return None
        if row < 0 or row >= len(self._df):
            return None
        if column < 0 or column >= self._df.shape[1]:
            return None
        try:
            return self._df.iat[row, column]
        except Exception:
            return None

    @staticmethod
    def _coerce_int_display(value: object) -> Optional[int]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            if isinstance(value, str):
                txt = value.strip()
                if not txt:
                    return None
                return int(float(txt))
            return int(value)
        except Exception:
            return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                key = str(self._df.columns[section])
                labels = {
                    "feature_id": "Id",
                    "name": "Name",
                    "source": "Source",
                    "unit": "Unit",
                    "type": "Type",
                    "lag_seconds": "Lag",
                    "tags": "Tags",
                    "notes": "Notes",
                }
                return labels.get(key, key)
            else:
                return str(section + 1)
        return None

    def feature_payload_at(self, row: int) -> dict | None:
        if row < 0 or row >= len(self._df):
            return None
        r = self._df.iloc[row]
        def _normalize(value):
            try:
                if pd.isna(value):
                    return None
            except Exception:
                logger.warning("Exception in _normalize", exc_info=True)
            return value
        fid = r.get("feature_id")
        try:
            feature_id = int(fid) if pd.notna(fid) else None
        except Exception:
            feature_id = None
        return dict(
            feature_id=feature_id,
            name      = _normalize(r.get("name")),
            source    = _normalize(r.get("source")),
            unit      = _normalize(r.get("unit")),
            type      = _normalize(r.get("type")),
            notes     = _normalize(r.get("notes")),
            lag_seconds = _normalize(r.get("lag_seconds")),
        )

    def search_text_at(self, row: int) -> str:
        if row < 0 or row >= len(self._search_cache):
            return ""
        return self._search_cache[row]

    def _build_search_cache(self) -> None:
        if self._df is None or self._df.empty:
            self._search_cache = []
            return
        try:
            safe = self._df.fillna("")
            joined = safe.astype(str).agg(" ".join, axis=1).str.lower()
            self._search_cache = joined.tolist()
        except Exception:
            self._search_cache = []
            for _idx in range(len(self._df)):
                self._search_cache.append("")
            for idx in range(len(self._df)):
                try:
                    row = self._df.iloc[idx]
                    text = " ".join(str(v) for v in row if pd.notna(v)).lower()
                except Exception:
                    text = ""
                self._search_cache[idx] = text


class FeaturesFilterProxyModel(QSortFilterProxyModel):
    """
    Filter across ALL columns using case-insensitive contains.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._needle = ""
        self._needles: tuple[str, ...] = ()

    def set_search_text(self, text: str):
        needle = (text or "").strip().lower()
        needles = tuple(part.strip() for part in needle.split(",") if part.strip())
        if needle == self._needle and needles == self._needles:
            return
        self._needle = needle
        self._needles = needles
        self.invalidateFilter()

    def search_text(self) -> str:
        return self._needle

    def filterAcceptsRow(self, source_row, source_parent):
        if not self._needles:
            return True
        src = self.sourceModel()
        if src is None:
            return False
        search_text = None
        try:
            search_text = src.search_text_at(source_row)
        except Exception:
            search_text = None
        if search_text is not None:
            return any(needle in search_text for needle in self._needles)
        cols = src.columnCount()
        for c in range(cols):
            idx = src.index(source_row, c, source_parent)
            val = src.data(idx, Qt.DisplayRole)
            if val and any(needle in str(val).lower() for needle in self._needles):
                return True
        return False

    def lessThan(self, left, right):  # noqa: N802
        try:
            if left.column() == 0 and right.column() == 0:
                src = self.sourceModel()
                if src is not None:
                    left_data = getattr(src, "raw_value", lambda *_: None)(left.row(), left.column())
                    right_data = getattr(src, "raw_value", lambda *_: None)(right.row(), right.column())
                    if left_data in (None, "") and right_data not in (None, ""):
                        return True
                    if right_data in (None, "") and left_data not in (None, ""):
                        return False
                    return int(float(left_data)) < int(float(right_data))
        except Exception:
            pass
        return super().lessThan(left, right)
