from __future__ import annotations
from typing import Optional

import logging
import numpy as np
import pandas as pd

from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex

logger = logging.getLogger(__name__)


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict, np.ndarray)):
        return False
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, (list, tuple, set, dict, np.ndarray)):
        return False
    try:
        return bool(result)
    except Exception:
        return False


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
        self._payload_df = self._normalize_payload_dataframe(df)
        self._df = self._normalize_for_table(self._payload_df)
        self._payload_cache = self._build_payload_cache(self._payload_df)
        self._row_by_feature_id = self._build_row_by_feature_id(self._payload_cache)

        # Hot-path caches (paint/sort/filter call data() a lot)
        self._np = self._df.to_numpy(copy=False) if self._df is not None else None
        self._disp_cache: dict[tuple[int, int], str] = {}

        # Search cache for quick filter fallback / other uses
        self._search_cache: list[str] | None = None

    def set_dataframe(self, df: pd.DataFrame):
        normalized_payload = self._normalize_payload_dataframe(df)
        normalized = self._normalize_for_table(normalized_payload)
        payload_cache = self._build_payload_cache(normalized_payload)
        row_by_feature_id = self._build_row_by_feature_id(payload_cache)
        old_df = self._df
        try:
            if old_df is not None and old_df.shape == normalized.shape and tuple(old_df.columns) == tuple(normalized.columns):
                if old_df.equals(normalized):
                    self._payload_df = normalized_payload
                    self._payload_cache = payload_cache
                    self._row_by_feature_id = row_by_feature_id
                    self._search_cache = None
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
            self._payload_df = normalized_payload
            self._df = normalized
            self._payload_cache = payload_cache
            self._row_by_feature_id = row_by_feature_id
            self._np = self._df.to_numpy(copy=False) if self._df is not None else None
            self.endResetModel()
            return

        self._payload_df = normalized_payload
        self._df = normalized
        self._payload_cache = payload_cache
        self._row_by_feature_id = row_by_feature_id
        self._np = self._df.to_numpy(copy=False) if self._df is not None else None
        if new_shape[0] > 0 and new_shape[1] > 0:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(new_shape[0] - 1, new_shape[1] - 1),
                [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole],
            )

    def _normalize_payload_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
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
        return out

    def _normalize_for_table(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, list(self.TABLE_COLUMNS)]

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

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if not index.isValid() or self._df is None:
            return None

        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return None

        r, c = int(index.row()), int(index.column())

        # Display cache helps a lot during repaints and sort-triggered redraws
        if role == Qt.ItemDataRole.DisplayRole:
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
                if role == Qt.ItemDataRole.DisplayRole and len(self._disp_cache) < 200_000:
                    self._disp_cache[(r, c)] = out
                return out

        if isinstance(val, (list, tuple, set)) and not val:
            out = ""
            if role == Qt.ItemDataRole.DisplayRole and len(self._disp_cache) < 200_000:
                self._disp_cache[(r, c)] = out
            return out

        try:
            out = "" if pd.isna(val) else str(val)
        except Exception:
            out = str(val)

        if role == Qt.ItemDataRole.DisplayRole and len(self._disp_cache) < 200_000:
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
        if _is_missing_scalar(value):
            return None
        try:
            if isinstance(value, str):
                txt = value.strip()
                if not txt:
                    return None
                return int(float(txt))
            return int(value)
        except Exception:
            return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole:
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
        if row < 0 or row >= len(self._payload_cache):
            return None
        payload = self._payload_cache[row]
        return dict(payload) if payload is not None else None

    def _payload_map_by_feature_id(self) -> dict[int, dict]:
        return {
            int(feature_id): dict(self._payload_cache[row_idx])
            for feature_id, row_idx in self._row_by_feature_id.items()
            if 0 <= row_idx < len(self._payload_cache) and self._payload_cache[row_idx] is not None
        }

    def _rows_for_feature_ids(self, feature_ids: set[int]) -> list[int]:
        if not feature_ids:
            return []
        rows = [self._row_by_feature_id[fid] for fid in feature_ids if fid in self._row_by_feature_id]
        rows.sort()
        return rows

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
        rows = self._np if self._np is not None else self._df.to_numpy(copy=False)
        cache: list[str] = []
        try:
            for row in rows:
                parts: list[str] = []
                for value in row:
                    if _is_missing_scalar(value):
                        continue
                    if isinstance(value, (list, tuple, set)):
                        for item in value:
                            if _is_missing_scalar(item):
                                continue
                            text = str(item).strip()
                            if text:
                                parts.append(text)
                        continue
                    text = str(value).strip()
                    if text:
                        parts.append(text)
                cache.append(" ".join(parts).lower())
        except Exception:
            cache = [""] * len(self._df)
            for idx in range(len(self._df)):
                try:
                    row = self._df.iloc[idx]
                    text = " ".join(str(v) for v in row if pd.notna(v)).lower()
                except Exception:
                    text = ""
                cache[idx] = text
        self._search_cache = cache

    def _build_payload_cache(self, df: pd.DataFrame) -> list[dict]:
        if df is None or df.empty:
            return []

        def _normalize(value):
            if _is_missing_scalar(value):
                return None
            return value

        records = df.to_dict(orient="records")
        payloads: list[dict] = []
        for record in records:
            fid = record.get("feature_id")
            try:
                feature_id = int(fid) if pd.notna(fid) else None
            except Exception:
                feature_id = None

            payload = dict(
                feature_id=feature_id,
                name=_normalize(record.get("name")),
                source=_normalize(record.get("source")),
                unit=_normalize(record.get("unit")),
                type=_normalize(record.get("type")),
                notes=_normalize(record.get("notes")),
                lag_seconds=_normalize(record.get("lag_seconds")),
                tags=_normalize(record.get("tags")),
            )
            for key in (
                "system",
                "systems",
                "dataset",
                "datasets",
                "dataset_ids",
                "imports",
                "import_ids",
            ):
                if key in record:
                    payload[key] = _normalize(record.get(key))
            if "locations" not in payload and "datasets" in payload:
                payload["locations"] = payload.get("datasets")
            payloads.append(payload)
        return payloads

    def _build_row_by_feature_id(self, payloads: list[dict]) -> dict[int, int]:
        rows: dict[int, int] = {}
        for row_idx, payload in enumerate(payloads):
            fid = payload.get("feature_id") if isinstance(payload, dict) else None
            if fid is None:
                continue
            try:
                rows[int(fid)] = row_idx
            except Exception:
                continue
        return rows
