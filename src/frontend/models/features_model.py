from __future__ import annotations
from typing import Optional

import logging
import numpy as np
import pandas as pd

from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex

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

        # Hot-path caches (paint/sort/filter call data() a lot)
        self._np = self._df.to_numpy(copy=False) if self._df is not None else None
        self._disp_cache: dict[tuple[int, int], str] = {}

        # Search cache for quick filter fallback / other uses
        self._search_cache: list[str] | None = None

    def set_dataframe(self, df: pd.DataFrame):
        normalized = self._normalize_for_table(df)
        old_df = self._df
        try:
            if old_df is not None and old_df.shape == normalized.shape and tuple(old_df.columns) == tuple(normalized.columns):
                if old_df.equals(normalized):
                    return
        except Exception:
            logger.warning("Exception in set_dataframe equality check", exc_info=True)

        old_columns = tuple(old_df.columns) if old_df is not None else ()
        old_shape = old_df.shape if old_df is not None else (0, 0)
        new_shape = normalized.shape

        # invalidate caches
        self._search_cache = None
        self._disp_cache.clear()

        # Resetting is significantly cheaper than incremental row inserts/removals for full table reloads.
        if old_shape != new_shape or old_columns != tuple(normalized.columns):
            self.beginResetModel()
            self._df = normalized
            self._np = self._df.to_numpy(copy=False) if self._df is not None else None
            self.endResetModel()
            return

        self._df = normalized
        self._np = self._df.to_numpy(copy=False) if self._df is not None else None
        if new_shape[0] > 0 and new_shape[1] > 0:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(new_shape[0] - 1, new_shape[1] - 1),
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
        return out.loc[:, list(self.TABLE_COLUMNS)]

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        if self._df is None:
            return 0
        return int(self._df.shape[0])

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        if self._df is None:
            return 0
        return int(self._df.shape[1])

    def data(self, index, role=Qt.DisplayRole):  # noqa: N802
        if not index.isValid() or self._df is None:
            return None

        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None

        r, c = int(index.row()), int(index.column())

        # Display cache helps a lot during repaints and sort-triggered redraws
        if role == Qt.DisplayRole:
            hit = self._disp_cache.get((r, c))
            if hit is not None:
                return hit

        try:
            val = self._np[r, c] if self._np is not None else self._df.iat[r, c]
        except Exception:
            return None

        key = str(self._df.columns[c]) if self._df is not None else ""
        if key in {"feature_id", "lag_seconds"}:
            as_int = self._coerce_int_display(val)
            if as_int is not None:
                out = str(as_int)
                if role == Qt.DisplayRole and len(self._disp_cache) < 200_000:
                    self._disp_cache[(r, c)] = out
                return out

        if isinstance(val, (list, tuple, set)) and not val:
            out = ""
            if role == Qt.DisplayRole and len(self._disp_cache) < 200_000:
                self._disp_cache[(r, c)] = out
            return out

        try:
            out = "" if pd.isna(val) else str(val)
        except Exception:
            out = str(val)

        if role == Qt.DisplayRole and len(self._disp_cache) < 200_000:
            self._disp_cache[(r, c)] = out
        return out

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

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role == Qt.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                try:
                    key = str(self._df.columns[section])
                except Exception:
                    key = str(section)
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
            return str(section + 1)
        return None

    def feature_payload_at(self, row: int) -> dict | None:
        if self._df is None or row < 0 or row >= len(self._df):
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
            name=_normalize(r.get("name")),
            source=_normalize(r.get("source")),
            unit=_normalize(r.get("unit")),
            type=_normalize(r.get("type")),
            notes=_normalize(r.get("notes")),
            lag_seconds=_normalize(r.get("lag_seconds")),
            tags=_normalize(r.get("tags")),
        )

    def search_text_at(self, row: int) -> str:
        if self._search_cache is None:
            self._build_search_cache()
        if self._search_cache is None or row < 0 or row >= len(self._search_cache):
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
            self._search_cache = [""] * len(self._df)
            for idx in range(len(self._df)):
                try:
                    row = self._df.iloc[idx]
                    text = " ".join(str(v) for v in row if pd.notna(v)).lower()
                except Exception:
                    text = ""
                self._search_cache[idx] = text