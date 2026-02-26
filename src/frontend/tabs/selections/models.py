
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import logging

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from ...models.selection_settings import FeatureLabelFilter, FeatureValueFilter, SelectionSettingsPayload


class FeatureSelectionTableModel(QAbstractTableModel):
    """Editable model exposing feature metadata and per-feature filters."""

    FEATURE_COLUMNS = [
        "name",
        "source",
        "unit",
        "type",
        "lag_seconds",
        "notes",
    ]

    SELECTION_COLUMNS = ["selected", "filter_min", "filter_max", "filter_global"]

    COLUMN_DEFINITIONS: List[Tuple[str, str]] = [
        ("selected", "Use"),
        ("feature_id", "Id"),
        ("name", "Name"),
        ("source", "Source"),
        ("unit", "Unit"),
        ("type", "Type"),
        ("lag_seconds", "Lag (s)"),
        ("tags", "Tags"),
        ("filter_min", "Min value"),
        ("filter_max", "Max value"),
        ("filter_global", "Global filter"),
        ("notes", "Notes"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: List[Dict[str, Any]] = []
        self._last_selection_key: tuple | None = None

    # --- Qt model API ---------------------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.COLUMN_DEFINITIONS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            try:
                return self.COLUMN_DEFINITIONS[section][1]
            except IndexError:
                return None
        return str(section + 1)

    def flags(self, index: QModelIndex):  # noqa: N802
        if not index.isValid():
            return Qt.NoItemFlags
        key = self.COLUMN_DEFINITIONS[index.column()][0]
        base = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if key == "feature_id":
            return base
        if key in {"selected", "filter_global"}:
            return base | Qt.ItemIsUserCheckable | Qt.ItemIsEditable
        return base | Qt.ItemIsEditable

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        key = self.COLUMN_DEFINITIONS[index.column()][0]
        value = row.get(key)
        if key in {"selected", "filter_global"}:
            if role == Qt.CheckStateRole:
                return Qt.CheckState.Checked if bool(value) else Qt.CheckState.Unchecked
            if role in (Qt.DisplayRole, Qt.EditRole):
                return "Enabled" if bool(value) else "Disabled"
            return None
        if key == "feature_id":
            if role in (Qt.DisplayRole, Qt.EditRole):
                return "" if value in (None, "") else str(value)
            return None
        if key == "tags":
            tags = ", ".join(row.get("tags") or [])
            if role in (Qt.DisplayRole, Qt.EditRole):
                return tags
            return None
        if role == Qt.DisplayRole:
            if value in (None, ""):
                return ""
            return str(value)
        if role == Qt.EditRole:
            if value is None:
                return ""
            return str(value) if isinstance(value, (int, float)) else value
        return None

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole):  # noqa: N802
        if not index.isValid():
            return False
        row = self._rows[index.row()]
        key = self.COLUMN_DEFINITIONS[index.column()][0]
        if key in {"selected", "filter_global"}:
            if role == Qt.CheckStateRole:
                try:
                    state = Qt.CheckState(value)
                except Exception:
                    state = Qt.CheckState.Checked if bool(value) else Qt.CheckState.Unchecked
                row[key] = state == Qt.CheckState.Checked
                self.dataChanged.emit(index, index, [Qt.CheckStateRole, Qt.DisplayRole])
                return True
            if role == Qt.EditRole:
                row[key] = bool(value)
                self.dataChanged.emit(index, index, [Qt.CheckStateRole, Qt.DisplayRole])
                return True
            return False
        if role != Qt.EditRole:
            return False
        if key in {"filter_min", "filter_max"}:
            text = "" if value is None else str(value).strip()
            if text == "":
                row[key] = None
            else:
                try:
                    row[key] = float(text)
                except Exception:
                    row[key] = None
        elif key == "lag_seconds":
            try:
                row[key] = int(value) if value not in (None, "") else 0
            except Exception:
                row[key] = 0
        elif key == "tags":
            row["tags"] = self._coerce_tags(value)
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True
        else:
            row[key] = "" if value is None else str(value)
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
        return True

    # --- Data manipulation ----------------------------------------------
    def set_features(self, df: pd.DataFrame) -> None:
        rows: List[Dict[str, Any]] = []
        if df is not None and not df.empty:
            for record in df.fillna("").to_dict("records"):
                payload = dict(
                    feature_id=record.get("feature_id"),
                    name=record.get("name") or "",
                    source=record.get("source") or "",
                    unit=record.get("unit") or "",
                    type=record.get("type") or "",
                    notes=record.get("notes") or "",
                    lag_seconds=int(record.get("lag_seconds") or 0),
                    selected=True,
                    filter_min=None,
                    filter_max=None,
                    filter_global=True,
                    tags=self._coerce_tags(record.get("tags")),
                )
                payload["_original"] = {field: payload.get(field) for field in self.FEATURE_COLUMNS}
                payload["_original_tags"] = list(payload.get("tags") or [])
                rows.append(payload)

        old_count = len(self._rows)
        new_count = len(rows)
        if old_count > new_count:
            self.beginRemoveRows(QModelIndex(), new_count, old_count - 1)
            self._rows = self._rows[:new_count]
            self.endRemoveRows()
        elif new_count > old_count:
            self.beginInsertRows(QModelIndex(), old_count, new_count - 1)
            self._rows.extend({} for _ in range(new_count - old_count))
            self.endInsertRows()

        self._rows = rows
        self._last_selection_key = None
        if new_count > 0:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(new_count - 1, self.columnCount() - 1),
                [Qt.DisplayRole, Qt.EditRole, Qt.CheckStateRole],
            )

    def apply_selection(self, payload: Optional[SelectionSettingsPayload], *, select_all_by_default: bool = False) -> None:
        selection_key = self._selection_key(payload, select_all_by_default=select_all_by_default)
        if selection_key == self._last_selection_key:
            return
        self._last_selection_key = selection_key
        selected_ids = set(int(fid) for fid in (payload.feature_ids if payload else []) if fid is not None)
        selected_labels = set(str(label) for label in (payload.feature_labels if payload else []) if str(label).strip())
        use_labels = bool(selected_labels)
        filters: Dict[int, FeatureValueFilter] = {}
        if payload:
            for flt in payload.feature_filters:
                if flt.feature_id is not None:
                    filters[int(flt.feature_id)] = flt
        label_filters: Dict[str, FeatureLabelFilter] = {}
        if payload:
            for flt in payload.feature_filter_labels:
                label = str(flt.label or "").strip()
                if label:
                    label_filters[label] = flt
        use_label_filters = bool(label_filters)
        if not self._rows:
            return
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self._rows) - 1, self.columnCount() - 1)
        if use_labels:
            label_counts: Dict[str, int] = {}
            for row in self._rows:
                label = str(row.get("notes") or "").strip()
                if not label:
                    continue
                label_counts[label] = label_counts.get(label, 0) + 1
            missing_labels = [lbl for lbl in selected_labels if lbl not in label_counts]
            if missing_labels:
                logger.debug("Selection labels not found in current feature list: %s", missing_labels)
            dup_labels = [lbl for lbl, count in label_counts.items() if count > 1 and lbl in selected_labels]
            if dup_labels:
                logger.warning("Selection labels match multiple features: %s", dup_labels)
        for row in self._rows:
            fid = row.get("feature_id")
            label = row.get("notes") or ""
            if use_labels:
                row["selected"] = bool(str(label) in selected_labels)
            elif selected_ids:
                row["selected"] = bool(fid in selected_ids)
            else:
                row["selected"] = bool(select_all_by_default)
            flt = None
            if use_label_filters and label:
                flt = label_filters.get(str(label))
            if flt is None and fid is not None:
                flt = filters.get(int(fid))
            if flt:
                row["filter_min"] = flt.min_value
                row["filter_max"] = flt.max_value
                row["filter_global"] = bool(flt.apply_globally)
            else:
                row["filter_min"] = None
                row["filter_max"] = None
                row["filter_global"] = True
        self.dataChanged.emit(top_left, bottom_right)

    def _selection_key(self, payload: Optional[SelectionSettingsPayload], *, select_all_by_default: bool) -> tuple:
        if payload is None:
            return ("none", bool(select_all_by_default))
        selected = tuple(sorted(int(fid) for fid in (payload.feature_ids or []) if fid is not None))
        selected_labels = tuple(sorted(str(label) for label in (payload.feature_labels or []) if str(label).strip()))
        filters = tuple(
            sorted(
                (
                    int(flt.feature_id),
                    flt.min_value,
                    flt.max_value,
                    bool(flt.apply_globally),
                )
                for flt in (payload.feature_filters or [])
                if flt.feature_id is not None
            )
        )
        label_filters = tuple(
            sorted(
                (
                    str(flt.label),
                    flt.min_value,
                    flt.max_value,
                    bool(flt.apply_globally),
                )
                for flt in (payload.feature_filter_labels or [])
                if str(flt.label).strip()
            )
        )
        return (selected, selected_labels, filters, label_filters, bool(select_all_by_default))

    def insert_blank_row(self) -> int:
        row = dict(
            feature_id=None,
            name="",
            source="",
            unit="",
            type="",
            notes="",
            lag_seconds=0,
            selected=True,
            filter_min=None,
            filter_max=None,
            filter_global=True,
            tags=[],
            _original={field: "" for field in self.FEATURE_COLUMNS},
        )
        row["_original_tags"] = []
        position = len(self._rows)
        self.beginInsertRows(QModelIndex(), position, position)
        self._rows.append(row)
        self.endInsertRows()
        return position

    def remove_rows(self, rows: List[int]) -> List[Dict[str, Any]]:
        removed: List[Dict[str, Any]] = []
        for row_index in sorted(set(rows), reverse=True):
            if row_index < 0 or row_index >= len(self._rows):
                continue
            self.beginRemoveRows(QModelIndex(), row_index, row_index)
            removed.append(self._rows.pop(row_index))
            self.endRemoveRows()
        return removed

    def feature_changes(self) -> Tuple[List[Dict[str, Any]], List[Tuple[int, Dict[str, Any]]]]:
        new_features: List[Dict[str, Any]] = []
        updated_features: List[Tuple[int, Dict[str, Any]]] = []
        for row in self._rows:
            feature_id = row.get("feature_id")
            if feature_id in (None, "", 0):
                name = (row.get("name") or "").strip()
                if name:
                    payload = {field: row.get(field) for field in self.FEATURE_COLUMNS}
                    payload["name"] = name
                    payload["source"] = row.get("source") or ""
                    payload["type"] = row.get("type") or ""
                    payload["notes"] = row.get("notes") or ""
                    payload["tags"] = list(row.get("tags") or [])
                    new_features.append(payload)
                continue
            original = row.get("_original", {})
            changes: Dict[str, Any] = {}
            for field in self.FEATURE_COLUMNS:
                if self._normalize_value(row.get(field)) != self._normalize_value(original.get(field)):
                    changes[field] = row.get(field)
            if self._tag_signature(row.get("tags")) != self._tag_signature(row.get("_original_tags", [])):
                changes["tags"] = list(row.get("tags") or [])
            if changes:
                updated_features.append((int(feature_id), changes))
        return new_features, updated_features

    def selection_state(self) -> Tuple[List[int], List[FeatureValueFilter]]:
        selected: List[int] = []
        filters: List[FeatureValueFilter] = []
        for row in self._rows:
            fid = row.get("feature_id")
            if fid is None:
                continue
            if row.get("selected"):
                selected.append(int(fid))
            # Only include a filter if min/max values are set, even if apply_globally changed
            if row.get("filter_min") is not None or row.get("filter_max") is not None:
                filters.append(
                    FeatureValueFilter(
                        feature_id=int(fid),
                        min_value=row.get("filter_min"),
                        max_value=row.get("filter_max"),
                        apply_globally=bool(row.get("filter_global")),
                    )
                )
        return selected, filters

    def selection_state_labels(self) -> Tuple[List[str], List[FeatureLabelFilter]]:
        selected: List[str] = []
        filters: List[FeatureLabelFilter] = []
        for row in self._rows:
            label = str(row.get("notes") or "").strip()
            if not label:
                continue
            if row.get("selected"):
                selected.append(label)
            if row.get("filter_min") is not None or row.get("filter_max") is not None:
                filters.append(
                    FeatureLabelFilter(
                        label=label,
                        min_value=row.get("filter_min"),
                        max_value=row.get("filter_max"),
                        apply_globally=bool(row.get("filter_global")),
                    )
                )
        return selected, filters

    def set_rows_selected(self, rows: List[int], enabled: bool) -> None:
        selected_col = next(
            (idx for idx, (key, _label) in enumerate(self.COLUMN_DEFINITIONS) if key == "selected"),
            0,
        )
        for row_index in rows:
            if 0 <= row_index < len(self._rows):
                if self._rows[row_index].get("selected") != bool(enabled):
                    self._rows[row_index]["selected"] = bool(enabled)
                    idx = self.index(row_index, selected_col)
                    self.dataChanged.emit(idx, idx, [Qt.CheckStateRole, Qt.DisplayRole])

    def _normalize_value(self, value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, float):
            try:
                if pd.isna(value):
                    return ""
            except Exception:
                logger.warning("Exception in _normalize_value", exc_info=True)
        return value

    def _coerce_tags(self, value: Any) -> List[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            candidates = [part.strip() for part in value.split(",")]
        elif isinstance(value, (list, tuple, set)):
            candidates = list(value)
        else:
            candidates = [value]
        tags: List[str] = []
        seen: set[str] = set()
        for item in candidates:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            tags.append(text)
        return tags

    def _tag_signature(self, tags: Any) -> tuple[str, ...]:
        normalized: List[str] = []
        seen: set[str] = set()
        for tag in self._coerce_tags(tags):
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(key)
        return tuple(normalized)

    def feature_id_for_row(self, row: int) -> Optional[int]:
        if row < 0 or row >= len(self._rows):
            return None
        fid = self._rows[row].get("feature_id")
        if fid in (None, ""):
            return None
        try:
            return int(fid)
        except Exception:
            return None

    def row_payload(self, row: int) -> Optional[Dict[str, Any]]:
        if row < 0 or row >= len(self._rows):
            return None
        data = dict(self._rows[row])
        data.pop("_original", None)
        data.pop("_original_tags", None)
        return data


__all__ = ["FeatureSelectionTableModel"]
logger = logging.getLogger(__name__)
