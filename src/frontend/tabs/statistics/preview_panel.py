
from __future__ import annotations
# @ai(gpt-5, codex, refactor, 2026-02-26)
import re
from typing import Optional

import pandas as pd
from PySide6.QtGui import QResizeEvent, QShowEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QSplitter, QVBoxLayout, QWidget, QInputDialog
from PySide6.QtCore import Qt, QItemSelectionModel
from ...localization import tr

from ...widgets.fast_table import FastTable
from ...widgets.panel import Panel
from ...charts import GroupBarChart, MAX_FEATURES_SHOWN_LEGEND
import logging

logger = logging.getLogger(__name__)


PREVIEW_TABLE_COLUMNS: tuple[str, ...] = ("Name", "Source", "Unit", "Type", "Notes")


def empty_preview_table_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=list(PREVIEW_TABLE_COLUMNS))


def normalize_preview_table_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_preview_table_dataframe()
    out = df.copy()
    for column in PREVIEW_TABLE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    ordered = list(PREVIEW_TABLE_COLUMNS) + [c for c in out.columns if c not in PREVIEW_TABLE_COLUMNS]
    return out.loc[:, ordered]


def set_preview_table_dataframe(table: FastTable, df: Optional[pd.DataFrame]) -> pd.DataFrame:
    normalized = normalize_preview_table_dataframe(df)
    if table is None:
        return normalized
    table.set_dataframe(normalized, include_index=False)
    return normalized



class StatisticsPreview(Panel):
    """Container for the statistics preview panel."""
    _DEFAULT_NOTES = ""
    _NO_GROUP_LABEL = "No group"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("", parent=parent)
        preview_layout = self.content_layout()

        header_layout = QHBoxLayout()
        self.title = QLabel(tr("Select features and gather statistics."), self)
        self.title.setObjectName("DataInfo")
        header_layout.addWidget(self.title, 1)

        preview_layout.addLayout(header_layout)

        # Create a splitter to show both table and chart
        self._splitter = QSplitter(Qt.Orientation.Vertical, self)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 1)
        
        # Table widget
        table_container = QWidget(self._splitter)
        table_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_table = FastTable(
            select="items",
            single_selection=False,
            tint_current_selection=True,
            editable=True,
            parent=table_container,
            initial_uniform_column_widths=True,
            initial_uniform_column_count=len(PREVIEW_TABLE_COLUMNS),
            context_menu_builder=self._build_table_context_menu,
            sorting_enabled=True,
        )
        set_preview_table_dataframe(self.preview_table, empty_preview_table_dataframe())
        table_layout.addWidget(self.preview_table, 1)
        self._splitter.addWidget(table_container)
        
        # Chart widget
        self.preview_chart = GroupBarChart(
            title=tr("Statistics Preview"),
            parent=self._splitter,
            y_label=tr("Value"),
        )
        self._splitter.addWidget(self.preview_chart)
        
        # Set initial splitter sizes (balanced; refined on show/resize)
        self._splitter.setSizes([400, 400])
        
        preview_layout.addWidget(self._splitter, 1)
        self._preview_model: Optional[object] = None
        self._current_mode: str = "time"
        self._current_group_column: Optional[str] = None
        self._preview_df: Optional[pd.DataFrame] = None
        self._feature_rows: list[dict] = []
        self._last_splitter_sizes: list[int] = []
        self._splitters_initialized = False
        self._initial_table_layout_applied = False

        try:
            self.preview_table.selectionChangedInstant.connect(self._on_table_selection_changed)
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)

    # ------------------------------------------------------------------
    def set_status(self, text: str) -> None:
        self.title.setText(text)

    # ------------------------------------------------------------------
    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if not self._splitters_initialized:
            self._splitters_initialized = self._initialize_splitter_sizes()
        elif event.size() != event.oldSize():
            self._rebalance_splitter()

    def showEvent(self, event: QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._splitters_initialized:
            self._splitters_initialized = self._initialize_splitter_sizes()
        if self._splitters_initialized:
            self._rebalance_splitter()
            self._apply_initial_table_layout_once()

    def _rebalance_splitter(self) -> None:
        height = self._splitter.size().height()
        if height <= 0:
            return
        half_height = max(1, height // 2)
        self._splitter.setSizes([half_height, height - half_height])

    def _initialize_splitter_sizes(self) -> bool:
        height = self._splitter.size().height()
        if height <= 0:
            return False
        half_height = max(1, height // 2)
        self._splitter.setSizes([half_height, height - half_height])
        return True

    def _apply_initial_table_layout_once(self) -> None:
        if self._initial_table_layout_applied:
            return
        self._initial_table_layout_applied = True
        try:
            self.preview_table.reapply_uniform_column_widths()
        except Exception:
            logger.warning("Exception in _apply_initial_table_layout_once", exc_info=True)

    # ------------------------------------------------------------------
    def set_mode(self, mode: str, group_column: Optional[str] = None) -> None:
        """Set the current statistics mode for chart rendering."""
        self._current_mode = mode
        self._current_group_column = group_column

    # @ai(gpt-5, codex, refactor, 2026-03-12)
    def export_frames(
        self,
        *,
        include_summary_table: bool = True,
        include_feature_data: bool = True,
        selected_feature_keys: Optional[set[str]] = None,
        include_raw_statistics: bool = False,
    ) -> dict[str, pd.DataFrame]:
        datasets: dict[str, pd.DataFrame] = {}
        if include_raw_statistics and self._preview_df is not None:
            datasets[tr("Raw statistics")] = self._preview_df.copy()

        if include_summary_table:
            summary_df = self.export_summary_table_frame()
            if not summary_df.empty:
                datasets[tr("Summary table")] = summary_df

        if include_feature_data:
            feature_df = self.export_feature_series_frame(selected_feature_keys=selected_feature_keys)
            if not feature_df.empty:
                datasets[tr("Feature data")] = feature_df

        return datasets

    # @ai(gpt-5, codex, refactor, 2026-03-12)
    def export_summary_table_frame(self) -> pd.DataFrame:
        model = self.preview_table.model()
        if model is None:
            return pd.DataFrame()
        return self._model_to_frame(model)

    # @ai(gpt-5, codex, refactor, 2026-03-12)
    def export_feature_options(self) -> list[tuple[str, str]]:
        rows = self._feature_selections_from_table()
        return [(key, label) for key, label, _selection in rows]

    # @ai(gpt-5, codex, refactor, 2026-03-12)
    def default_selected_feature_export_keys(self) -> list[str]:
        selected_infos = self._selected_row_infos()
        selected_keys: list[str] = []
        seen: set[str] = set()
        for selection in selected_infos:
            key = self._feature_selection_key(selection)
            if key in seen:
                continue
            seen.add(key)
            selected_keys.append(key)
        if selected_keys:
            return selected_keys
        return [key for key, _label, _selection in self._feature_selections_from_table()]

    # @ai(gpt-5, codex, feature, 2026-03-12)
    def export_feature_series_frame(self, *, selected_feature_keys: Optional[set[str]] = None) -> pd.DataFrame:
        rows = self._feature_selections_from_table()
        if not rows:
            return pd.DataFrame()

        if selected_feature_keys is None:
            defaults = self.default_selected_feature_export_keys()
            selected_keys = set(defaults) if defaults else None
        else:
            selected_keys = set(selected_feature_keys)
        selected_frames: list[tuple[pd.DataFrame, str]] = []
        used_names: set[str] = set()
        for key, label, selection in rows:
            if selected_keys is not None and key not in selected_keys:
                continue
            subset = self._filter_preview(selection)
            if subset is None or subset.empty:
                continue
            column_name = self._build_export_value_column_name(selection, label, used_names)
            selected_frames.append((subset.copy(), column_name))

        if not selected_frames:
            return pd.DataFrame()

        if len(selected_frames) == 1:
            single = selected_frames[0][0].copy()
            single = single.drop_duplicates(ignore_index=True)
            sort_columns = [
                column
                for column in ("base_name", "source", "unit", "statistic", "t", "group_value", "value")
                if column in single.columns
            ]
            if sort_columns:
                single = single.sort_values(sort_columns, kind="stable").reset_index(drop=True)
            return single

        frames = [frame for frame, _column_name in selected_frames]
        key_columns = self._export_row_key_columns(frames)
        if not key_columns:
            max_len = max(len(frame.index) for frame in frames)
            export_df = pd.DataFrame({"Row": range(max_len)})
            for frame, column_name in selected_frames:
                values = pd.to_numeric(frame.get("value"), errors="coerce").reset_index(drop=True)
                export_df[column_name] = values.reindex(range(max_len))
            return export_df

        merged: Optional[pd.DataFrame] = None
        value_columns: list[str] = []
        for frame, column_name in selected_frames:
            reduced = frame.copy()
            if "t" in key_columns:
                reduced["t"] = pd.to_datetime(reduced["t"], errors="coerce")
            reduced = reduced.loc[:, [*key_columns, "value"]]
            reduced["value"] = pd.to_numeric(reduced["value"], errors="coerce")
            reduced = (
                reduced.groupby(key_columns, dropna=False, sort=False)["value"]
                .mean()
                .reset_index()
                .rename(columns={"value": column_name})
            )
            value_columns.append(column_name)
            if merged is None:
                merged = reduced
            else:
                merged = merged.merge(reduced, on=key_columns, how="outer")

        if merged is None or merged.empty:
            return pd.DataFrame()

        sort_columns = [col for col in ("t", "group_value", "import_id", "system", "Dataset") if col in key_columns]
        if sort_columns:
            merged = merged.sort_values(sort_columns, kind="stable").reset_index(drop=True)

        ordered_columns = [*key_columns, *value_columns]
        merged = merged.loc[:, [col for col in ordered_columns if col in merged.columns]]
        rename_map = {
            "t": "Timestamp",
            "group_value": "Group",
            "import_id": "Import ID",
            "import_name": "Import",
            "system": "System",
            "Dataset": "Dataset",
        }
        return merged.rename(columns=rename_map)

    def _model_to_frame(self, model) -> pd.DataFrame:
        headers = [
            str(model.headerData(col, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole) or "")
            for col in range(model.columnCount())
        ]
        rows: list[list[object]] = []
        for row in range(model.rowCount()):
            values: list[object] = []
            for col in range(model.columnCount()):
                values.append(model.data(model.index(row, col), Qt.ItemDataRole.DisplayRole))
            rows.append(values)
        return pd.DataFrame(rows, columns=headers)

    def editable_preview_for_save(self) -> pd.DataFrame:
        """Return preview dataframe with user-edited metadata applied."""
        if self._preview_df is None:
            return pd.DataFrame()
        model = self.preview_table.model()
        if self._preview_df.empty or model is None:
            return self._preview_df.copy()

        edited = self._preview_df.copy()
        row_count = model.rowCount()
        if row_count <= 0:
            return edited

        for row_idx in range(row_count):
            raw = {}
            if 0 <= row_idx < len(self._feature_rows):
                raw = dict(self._feature_rows[row_idx] or {})
            if not raw:
                continue
            mask = self._mask_for_raw_row(edited, raw)
            if not mask.any():
                continue

            display_row = self._model_row_payload(model, row_idx)
            name_value = self._normalize_display_value(display_row.get("Name"))
            source_value = self._normalize_display_value(display_row.get("Source"))
            unit_value = self._normalize_display_value(display_row.get("Unit"))
            type_value = self._normalize_display_value(display_row.get("Type"))
            existing_note = self._normalize_display_value(raw.get("notes"))
            notes_value = self._normalize_display_value(display_row.get("Notes")) or existing_note or self._DEFAULT_NOTES

            edited.loc[mask, "base_name"] = name_value
            edited.loc[mask, "source"] = source_value
            edited.loc[mask, "unit"] = unit_value
            edited.loc[mask, "new_qualifier"] = type_value
            edited.loc[mask, "label"] = notes_value
            edited.loc[mask, "notes"] = notes_value

        return edited

    # ------------------------------------------------------------------
    def update_preview(self, df: pd.DataFrame) -> None:
        try:
            self._last_splitter_sizes = self._splitter.sizes()
        except Exception:
            self._last_splitter_sizes = []
        self._preview_df = df.copy() if df is not None else None
        if self._preview_df is not None and "t" in self._preview_df.columns:
            self._preview_df["t"] = pd.to_datetime(self._preview_df["t"], errors="coerce")

        summary, raw_rows = self._build_feature_summary(
            self._preview_df,
        )
        self._feature_rows = raw_rows

        if summary is None or summary.empty:
            set_preview_table_dataframe(self.preview_table, empty_preview_table_dataframe())
            self._preview_model = self.preview_table.model()
            self.preview_chart.clear()
            return

        set_preview_table_dataframe(self.preview_table, summary)
        self._preview_model = self.preview_table.model()
        if self._last_splitter_sizes:
            try:
                self._splitter.setSizes(self._last_splitter_sizes)
            except Exception:
                logger.warning("Exception in update_preview", exc_info=True)

        self._select_default_row()
        self._update_chart_for_selection()

    # @ai(gpt-5, codex, refactor, 2026-03-11)
    def _build_feature_summary(
        self,
        df: Optional[pd.DataFrame],
    ) -> tuple[pd.DataFrame, list[dict]]:
        if df is None or df.empty:
            return empty_preview_table_dataframe(), []

        working = df.copy()
        if "statistic" not in working.columns:
            working["statistic"] = "value"
        if "base_name" not in working.columns:
            working["base_name"] = working.get("name", working.get("label"))
        if "new_qualifier" not in working.columns:
            working["new_qualifier"] = working.get("type")
        if "label" not in working.columns:
            working["label"] = working.get("notes")
        if "notes" not in working.columns:
            working["notes"] = working.get("label")
        if "notes" not in working.columns:
            working["notes"] = self._DEFAULT_NOTES
        else:
            working["notes"] = working["notes"].where(
                working["notes"].notna() & (working["notes"].astype(str).str.strip() != ""),
                self._DEFAULT_NOTES,
            )

        def _base_type_for_display(row: pd.Series) -> str:
            original = self._normalize_display_value(row.get("original_qualifier"))
            return original

        def _merge_type_and_stat(row: pd.Series) -> str:
            base_type = _base_type_for_display(row)
            stat = self._normalize_display_value(row.get("statistic")).lower()
            if base_type and stat:
                return f"{stat} ({base_type})"
            return base_type or stat

        key_columns = ["base_name", "source", "unit", "original_qualifier", "statistic"]
        for col in key_columns:
            if col not in working.columns:
                working[col] = pd.NA
        group_summary = (
            working.groupby(key_columns, dropna=False, sort=False)
            .agg(notes=("notes", "first"), new_qualifier=("new_qualifier", "first"), label=("label", "first"))
            .reset_index()
        )
        raw_rows = group_summary.to_dict("records")

        summary_columns: dict[str, pd.Series] = {
            "Name": group_summary.get("base_name", pd.Series(dtype=object)),
            "Source": group_summary.get("source", pd.Series(dtype=object)),
            "Unit": group_summary.get("unit", pd.Series(dtype=object)),
            "Type": group_summary.apply(
                lambda row: _merge_type_and_stat(
                    pd.Series(
                        {
                            "original_qualifier": row.get("original_qualifier"),
                            "new_qualifier": row.get("new_qualifier"),
                            "statistic": row.get("statistic"),
                        }
                    )
                ),
                axis=1,
            )
            if not group_summary.empty
            else pd.Series(dtype=object),
        }
        summary_columns["Notes"] = group_summary.get("notes", pd.Series(dtype=object))
        summary = pd.DataFrame(summary_columns)
        return summary, raw_rows

    def _select_default_row(self) -> None:
        model = self.preview_table.model()
        if model is None or model.rowCount() == 0:
            return
        sel_model = self.preview_table.selectionModel()
        if sel_model is None:
            return
        first_index = model.index(0, 0)
        flags = QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows
        try:
            sel_model.select(first_index, flags)
            self.preview_table.setCurrentIndex(first_index)
        except Exception:
            logger.warning("Exception in _select_default_row", exc_info=True)

    def _on_table_selection_changed(self) -> None:
        self._update_chart_for_selection()

    def _build_table_context_menu(self, menu, _pos, _table) -> None:
        model = self.preview_table.model()
        if model is None or model.rowCount() <= 0:
            return
        sel_model = self.preview_table.selectionModel()
        if sel_model is None or not sel_model.selectedIndexes():
            return
        set_values = menu.addAction(tr("Set value for selected cells…"))
        set_values.triggered.connect(self._set_value_for_selected_cells)

    def _set_value_for_selected_cells(self) -> None:
        model = self.preview_table.model()
        sel_model = self.preview_table.selectionModel()
        if model is None or sel_model is None:
            return
        cells = [idx for idx in sel_model.selectedIndexes() if idx.isValid()]
        if not cells:
            return
        value, ok = QInputDialog.getText(
            self,
            tr("Set value"),
            tr("Value for selected cells:"),
        )
        if not ok:
            return
        for index in cells:
            model.setData(index, value, Qt.ItemDataRole.EditRole)

    def _selected_row_info(self) -> Optional[dict]:
        selected_rows = self._selected_row_infos()
        if selected_rows:
            return selected_rows[0]
        return None

    def _selected_row_infos(self) -> list[dict]:
        model = self.preview_table.model()
        if model is None:
            return []
        sel_model = self.preview_table.selectionModel()
        if sel_model is None:
            return []
        indexes = sel_model.selectedIndexes()
        if not indexes:
            return []
        selected_rows = sorted({int(index.row()) for index in indexes if index.isValid()})
        selections: list[dict] = []
        seen: set[str] = set()
        for row in selected_rows:
            row_payload = self._model_row_payload(model, row)
            parsed = self._selection_from_table_row(row_payload)
            if parsed is None:
                parsed = self._resolve_selection_from_display(row_payload)
            if parsed is None:
                continue
            key = self._feature_selection_key(parsed)
            if key in seen:
                continue
            seen.add(key)
            selections.append(parsed)
        return selections

    def _feature_selections_from_table(self) -> list[tuple[str, str, dict]]:
        model = self.preview_table.model()
        if model is None:
            return []
        rows: list[tuple[str, str, dict]] = []
        seen: set[str] = set()
        for row in range(model.rowCount()):
            row_payload = self._model_row_payload(model, row)
            parsed = self._selection_from_table_row(row_payload)
            if parsed is None:
                parsed = self._resolve_selection_from_display(row_payload)
            if parsed is None:
                continue
            key = self._feature_selection_key(parsed)
            if key in seen:
                continue
            seen.add(key)
            rows.append((key, self._feature_selection_label(row_payload, parsed), parsed))
        return rows

    def _feature_selection_label(self, row_payload: dict, selection: dict) -> str:
        name = self._normalize_display_value(row_payload.get("Name")) or self._normalize_display_value(
            selection.get("base_name")
        )
        source = self._normalize_display_value(row_payload.get("Source")) or self._normalize_display_value(
            selection.get("source")
        )
        unit = self._normalize_display_value(row_payload.get("Unit")) or self._normalize_display_value(
            selection.get("unit")
        )
        type_text = self._normalize_display_value(row_payload.get("Type"))
        if not type_text:
            statistic = self._normalize_display_value(selection.get("statistic"))
            original = self._normalize_display_value(selection.get("original_qualifier"))
            type_text = statistic
            if original:
                type_text = f"{statistic} ({original})" if statistic else original
        parts = [name]
        if source:
            parts.append(source)
        if unit:
            parts.append(unit)
        if type_text:
            parts.append(type_text)
        return " | ".join(part for part in parts if part) or name or tr("Feature")

    @staticmethod
    def _feature_selection_key(selection: dict) -> str:
        def _norm(value: object) -> str:
            if value is None:
                return ""
            text = str(value).strip()
            return text.lower()

        return "||".join(
            [
                _norm(selection.get("base_name")),
                _norm(selection.get("source")),
                _norm(selection.get("unit")),
                _norm(selection.get("statistic")),
                _norm(selection.get("original_qualifier")),
            ]
        )

    def _selection_from_table_row(self, row_payload: dict) -> Optional[dict]:
        if not isinstance(row_payload, dict) or not row_payload:
            return None
        name = self._normalize_display_value(row_payload.get("Name"))
        if not name:
            return None
        source = self._normalize_display_value(row_payload.get("Source"))
        unit = self._normalize_display_value(row_payload.get("Unit"))
        type_text = self._normalize_display_value(row_payload.get("Type"))
        stat = type_text
        original_qualifier: Optional[str] = None
        match = re.fullmatch(r"\s*([^(]+?)\s*\((.+)\)\s*", type_text)
        if match:
            stat = str(match.group(1) or "").strip()
            original_qualifier = str(match.group(2) or "").strip() or None
        stat = stat.lower().strip()
        if not stat:
            return None
        return {
            "base_name": name,
            "source": source,
            "unit": unit,
            "statistic": stat,
            "original_qualifier": original_qualifier,
        }

    def _resolve_selection_from_display(self, row_payload: dict) -> Optional[dict]:
        if not self._feature_rows:
            return None
        display_to_raw = {
            "Name": "base_name",
            "Feature": "label",
            "Statistic": "statistic",
            "System": "system",
            "Dataset": "Dataset",
            "Source": "source",
            "Unit": "unit",
        }

        def _normalize(value: object) -> object:
            if value is None:
                return None
            if isinstance(value, str) and value.strip() == "":
                return None
            try:
                if pd.isna(value):
                    return None
            except Exception:
                logger.warning("Exception in _normalize", exc_info=True)
            return value

        desired = {raw_key: _normalize(row_payload.get(display_key)) for display_key, raw_key in display_to_raw.items()}
        for raw in self._feature_rows:
            matched = True
            for raw_key, wanted in desired.items():
                if wanted is None:
                    continue
                actual = _normalize(raw.get(raw_key))
                if actual is None:
                    matched = False
                    break
                if isinstance(wanted, str) or isinstance(actual, str):
                    if str(actual) != str(wanted):
                        matched = False
                        break
                elif actual != wanted:
                    matched = False
                    break
            if matched:
                return raw
        return None

    def _filter_preview(self, selection: dict) -> pd.DataFrame:
        if self._preview_df is None or self._preview_df.empty:
            return pd.DataFrame()

        df = self._preview_df
        mask = pd.Series([True] * len(df), index=df.index)

        def _normalize_selected(value: object) -> Optional[object]:
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except Exception:
                logger.warning("Exception in _normalize_selected", exc_info=True)
            if isinstance(value, str):
                text = value.strip()
                return text if text else None
            return value

        base_name = _normalize_selected(selection.get("base_name"))
        if base_name is not None and "base_name" in df.columns:
            mask &= df["base_name"].astype(str) == str(base_name)

        source = _normalize_selected(selection.get("source"))
        if source is not None and "source" in df.columns:
            source_mask = df["source"].astype(str) == str(source)
            mask &= source_mask

        unit = _normalize_selected(selection.get("unit"))
        if unit is not None and "unit" in df.columns:
            unit_mask = df["unit"].astype(str) == str(unit)
            mask &= unit_mask

        stat = _normalize_selected(selection.get("Statistic") or selection.get("statistic"))
        if stat is not None and "statistic" in df.columns:
            mask &= df["statistic"].astype(str) == str(stat)

        original_qualifier = _normalize_selected(selection.get("original_qualifier"))
        if "original_qualifier" in df.columns:
            if original_qualifier is None:
                mask &= df["original_qualifier"].isna() | (df["original_qualifier"].astype(str).str.strip() == "")
            else:
                mask &= df["original_qualifier"].astype(str) == str(original_qualifier)

        return df[mask].copy()

    @staticmethod
    def _normalize_display_value(value: object) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            logger.warning("Exception in _normalize_display_value", exc_info=True)
        return str(value).strip()

    @staticmethod
    def _model_row_payload(model, row: int) -> dict[str, object]:
        payload: dict[str, object] = {}
        for col in range(model.columnCount()):
            header = model.headerData(col, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
            if header is None:
                continue
            payload[str(header)] = model.data(model.index(row, col), Qt.ItemDataRole.DisplayRole)
        return payload

    def _mask_for_raw_row(self, df: pd.DataFrame, raw: dict) -> pd.Series:
        mask = pd.Series([True] * len(df), index=df.index)
        for key in ("base_name", "source", "unit", "statistic", "original_qualifier"):
            if key not in df.columns:
                continue
            wanted = raw.get(key)
            if wanted is None or (isinstance(wanted, str) and wanted.strip() == ""):
                continue
            if pd.isna(wanted):
                mask &= df[key].isna()
            else:
                mask &= df[key].astype(str) == str(wanted)
        return mask

    def _display_group_value(self, value: object) -> str:
        text = self._normalize_display_value(value)
        if not text:
            return self._NO_GROUP_LABEL
        return text

    def _selection_feature_name(self, selection_info: Optional[dict]) -> Optional[str]:
        if not isinstance(selection_info, dict):
            return None
        for key in ("Name", "base_name", "Feature", "feature_label", "label"):
            text = self._normalize_display_value(selection_info.get(key))
            if text:
                return text
        return None

    @staticmethod
    def _selection_stat_name(selection_info: Optional[dict]) -> str:
        if not isinstance(selection_info, dict):
            return ""
        for key in ("Statistic", "statistic"):
            text = str(selection_info.get(key) or "").strip()
            if text:
                return text
        return ""

    def _build_chart_series_name(self, selection_info: dict, used: set[str]) -> str:
        feature_name = self._selection_feature_name(selection_info) or tr("Feature")
        stat_name = self._selection_stat_name(selection_info)
        base_name = f"{feature_name} ({stat_name})" if stat_name else feature_name
        candidate = base_name
        suffix = 2
        while candidate in used:
            candidate = f"{base_name} ({suffix})"
            suffix += 1
        used.add(candidate)
        return candidate

    def _build_export_value_column_name(self, selection_info: dict, fallback_label: str, used: set[str]) -> str:
        feature_name = self._selection_feature_name(selection_info) or str(fallback_label or "").strip() or tr("Feature")
        stat_name = self._selection_stat_name(selection_info)
        original = self._normalize_display_value(selection_info.get("original_qualifier"))
        base_name = feature_name
        if stat_name:
            base_name = f"{base_name} [{stat_name}]"
        if original:
            base_name = f"{base_name} ({original})"
        candidate = base_name
        suffix = 2
        while candidate in used:
            candidate = f"{base_name} ({suffix})"
            suffix += 1
        used.add(candidate)
        return candidate

    @staticmethod
    def _export_row_key_columns(frames: list[pd.DataFrame]) -> list[str]:
        if not frames:
            return []
        common = set(frames[0].columns)
        for frame in frames[1:]:
            common &= set(frame.columns)
        preferred = ["t", "group_value", "import_id", "import_name", "system", "Dataset"]
        return [column for column in preferred if column in common]

    @staticmethod
    def _selection_caption(displayed_count: int, total_selected: int) -> str:
        if total_selected > displayed_count:
            return f" ({displayed_count}/{total_selected} selected)"
        return ""

    def _update_chart_for_selection(self) -> None:
        selections = self._selected_row_infos()
        if not selections:
            self.preview_chart.clear()
            return

        total_selected = len(selections)
        chart_selections = selections[:MAX_FEATURES_SHOWN_LEGEND]
        selected_frames: list[pd.DataFrame] = []
        used_names: set[str] = set()
        for selection in chart_selections:
            subset = self._filter_preview(selection)
            if subset is None or subset.empty:
                continue
            labeled = subset.copy()
            labeled["__chart_series__"] = self._build_chart_series_name(selection, used_names)
            selected_frames.append(labeled)

        if not selected_frames:
            self.preview_chart.clear()
            return

        if len(selected_frames) == 1:
            single = selected_frames[0].drop(columns=["__chart_series__"], errors="ignore")
            self._update_chart(single, selection_info=chart_selections[0], total_selected=total_selected)
            return

        combined = pd.concat(selected_frames, axis=0, ignore_index=True)
        self._update_chart(combined, total_selected=total_selected)

    def _update_chart(
        self,
        df: pd.DataFrame,
        selection_info: Optional[dict] = None,
        *,
        total_selected: int = 1,
    ) -> None:
        """Update the chart based on the current mode and data."""
        if df is None or df.empty:
            self.preview_chart.clear()
            return
        
        # Check which columns we have
        has_group_value = "group_value" in df.columns
        has_value = "value" in df.columns
        has_t = "t" in df.columns
        
        if not has_value:
            self.preview_chart.clear()
            return

        is_multi_selection = "__chart_series__" in df.columns and len(df["__chart_series__"].dropna().unique()) > 1
        # Decide chart rendering based on mode
        if self._current_mode == "column" and has_group_value:
            if is_multi_selection:
                self._render_group_chart_multi(df, total_selected=total_selected)
            else:
                self._render_group_chart(df, selection_info=selection_info, total_selected=total_selected)
        elif has_t:
            if is_multi_selection:
                self._render_time_chart_multi(df, total_selected=total_selected)
            else:
                self._render_time_chart(df, selection_info=selection_info, total_selected=total_selected)
        else:
            self.preview_chart.clear()

    # @ai(gpt-5, codex, feature, 2026-03-12)
    def _render_group_chart(
        self,
        df: pd.DataFrame,
        selection_info: Optional[dict] = None,
        *,
        total_selected: int = 1,
    ) -> None:
        """Render chart for group-by-column mode."""
        if "group_value" not in df.columns or "value" not in df.columns:
            self.preview_chart.clear()
            return
        df = df.copy()
        df["group_value_display"] = df["group_value"].apply(self._display_group_value)
        time_order: list[str] | None = None
        if "t" in df.columns:
            df["t"] = pd.to_datetime(df["t"], errors="coerce")
            order_df = (
                df.dropna(subset=["t"])
                .groupby("group_value_display", dropna=False, sort=False)["t"]
                .min()
                .reset_index(name="t_min")
                .sort_values(["t_min", "group_value_display"], kind="stable")
            )
            if not order_df.empty:
                time_order = [str(v) for v in order_df["group_value_display"].tolist()]
        
        # Check if we have multiple statistics
        statistics = df["statistic"].unique().tolist() if "statistic" in df.columns else ["value"]
        
        if len(statistics) == 1:
            # Single statistic: simple bar chart
            stat = statistics[0]
            stat_df = df[df["statistic"] == stat] if "statistic" in df.columns else df
            
            agg_spec: dict[str, tuple[str, str]] = {"value": ("value", "mean")}
            if "sample_count" in stat_df.columns:
                agg_spec["sample_count"] = ("sample_count", "sum")
            if "outlier_count" in stat_df.columns:
                agg_spec["outlier_count"] = ("outlier_count", "sum")
            agg = stat_df.groupby("group_value_display", dropna=False, sort=False).agg(**agg_spec).reset_index()
            if time_order:
                order_index = {name: idx for idx, name in enumerate(time_order)}
                agg["_order"] = agg["group_value_display"].map(lambda x: order_index.get(str(x), len(order_index)))
                agg = agg.sort_values("_order", kind="stable").drop(columns=["_order"])
            
            feature_name = self._selection_feature_name(selection_info)
            title = f"Statistics by Group ({stat})"
            if feature_name:
                title = f"{feature_name} ({stat})"
            title += self._selection_caption(1, int(total_selected))
            self.preview_chart.set_title(title)
            self.preview_chart.set_dataframe(
                agg,
                category_col="group_value_display",
                value_col="value",
                series_name=stat,
            )
            tooltip_overrides: dict[tuple[str, str], str] = {}
            for row in agg.itertuples(index=False):
                category = str(getattr(row, "group_value_display", ""))
                parts: list[str] = []
                if hasattr(row, "sample_count") and pd.notna(getattr(row, "sample_count")):
                    parts.append(f"Count: {int(getattr(row, 'sample_count'))}")
                if stat == "outliers" and hasattr(row, "outlier_count") and pd.notna(getattr(row, "outlier_count")):
                    parts.append(f"Outliers: {int(getattr(row, 'outlier_count'))}")
                if parts:
                    tooltip_overrides[(str(stat), category)] = " | ".join(parts)
            self.preview_chart.set_tooltip_overrides(tooltip_overrides)
        else:
            # Multiple statistics: pivot to show all
            pivot = df.pivot_table(
                index="group_value_display",
                columns="statistic",
                values="value",
                aggfunc="mean",
            ).reset_index()
            if time_order:
                order_index = {name: idx for idx, name in enumerate(time_order)}
                pivot["_order"] = pivot["group_value_display"].map(lambda x: order_index.get(str(x), len(order_index)))
                pivot = pivot.sort_values("_order", kind="stable").drop(columns=["_order"])

            feature_name = self._selection_feature_name(selection_info)
            title = feature_name or "Statistics by Group"
            title = title or "Statistics by Group"
            title += self._selection_caption(1, int(total_selected))
            self.preview_chart.set_title(title)
            self.preview_chart.set_multi_series(
                pivot,
                category_col="group_value_display",
                value_cols=statistics,
            )
            tooltip_overrides: dict[tuple[str, str], str] = {}
            if "sample_count" in df.columns:
                for stat_name in statistics:
                    stat_rows = df[df["statistic"] == stat_name]
                    if stat_rows.empty:
                        continue
                    agg_spec: dict[str, tuple[str, str]] = {"sample_count": ("sample_count", "sum")}
                    if "outlier_count" in stat_rows.columns:
                        agg_spec["outlier_count"] = ("outlier_count", "sum")
                    meta = (
                        stat_rows.groupby("group_value_display", dropna=False)
                        .agg(**agg_spec)
                        .reset_index()
                    )
                    for row in meta.itertuples(index=False):
                        category = str(getattr(row, "group_value_display", ""))
                        parts = [f"Count: {int(getattr(row, 'sample_count'))}"]
                        if (
                            str(stat_name) == "outliers"
                            and hasattr(row, "outlier_count")
                            and pd.notna(getattr(row, "outlier_count"))
                        ):
                            parts.append(f"Outliers: {int(getattr(row, 'outlier_count'))}")
                        tooltip_overrides[(str(stat_name), category)] = " | ".join(parts)
            self.preview_chart.set_tooltip_overrides(tooltip_overrides)

    # @ai(gpt-5, codex, feature, 2026-03-12)
    def _render_group_chart_multi(self, df: pd.DataFrame, *, total_selected: int) -> None:
        if "group_value" not in df.columns or "value" not in df.columns or "__chart_series__" not in df.columns:
            self.preview_chart.clear()
            return
        chart_df = df.copy()
        chart_df["group_value_display"] = chart_df["group_value"].apply(self._display_group_value)
        if "t" in chart_df.columns:
            chart_df["t"] = pd.to_datetime(chart_df["t"], errors="coerce")
            ordering = (
                chart_df.dropna(subset=["t"])
                .groupby("group_value_display", dropna=False, sort=False)["t"]
                .min()
                .reset_index(name="t_min")
                .sort_values(["t_min", "group_value_display"], kind="stable")
            )
            order_map = {str(value): idx for idx, value in enumerate(ordering["group_value_display"].tolist())}
        else:
            order_map = {}

        series_order = [str(value) for value in chart_df["__chart_series__"].dropna().tolist()]
        series_order = list(dict.fromkeys(series_order))
        pivot = (
            chart_df.pivot_table(
                index="group_value_display",
                columns="__chart_series__",
                values="value",
                aggfunc="mean",
            )
            .reset_index()
        )
        if order_map:
            pivot["_order"] = pivot["group_value_display"].map(lambda value: order_map.get(str(value), len(order_map)))
            pivot = pivot.sort_values("_order", kind="stable").drop(columns=["_order"])
        value_cols = [column for column in series_order if column in pivot.columns]
        if not value_cols:
            self.preview_chart.clear()
            return

        self.preview_chart.set_title(
            "Statistics by Group" + self._selection_caption(len(value_cols), int(total_selected))
        )
        self.preview_chart.set_multi_series(
            pivot,
            category_col="group_value_display",
            value_cols=value_cols,
        )
        tooltip_overrides: dict[tuple[str, str], str] = {}
        for series_name in value_cols:
            series_rows = chart_df[chart_df["__chart_series__"] == series_name]
            if series_rows.empty:
                continue
            agg_spec: dict[str, tuple[str, str]] = {}
            if "sample_count" in series_rows.columns:
                agg_spec["sample_count"] = ("sample_count", "sum")
            if "outlier_count" in series_rows.columns:
                agg_spec["outlier_count"] = ("outlier_count", "sum")
            if not agg_spec:
                continue
            meta = series_rows.groupby("group_value_display", dropna=False).agg(**agg_spec).reset_index()
            for row in meta.itertuples(index=False):
                category = str(getattr(row, "group_value_display", ""))
                parts: list[str] = []
                if hasattr(row, "sample_count") and pd.notna(getattr(row, "sample_count")):
                    parts.append(f"Count: {int(getattr(row, 'sample_count'))}")
                if hasattr(row, "outlier_count") and pd.notna(getattr(row, "outlier_count")):
                    parts.append(f"Outliers: {int(getattr(row, 'outlier_count'))}")
                if parts:
                    tooltip_overrides[(str(series_name), category)] = " | ".join(parts)
        self.preview_chart.set_tooltip_overrides(tooltip_overrides)

    # @ai(gpt-5, codex, feature, 2026-03-12)
    def _render_time_chart(
        self,
        df: pd.DataFrame,
        selection_info: Optional[dict] = None,
        *,
        total_selected: int = 1,
    ) -> None:
        """Render chart for time-based mode."""
        if "t" not in df.columns or "value" not in df.columns:
            self.preview_chart.clear()
            return
        
        # Format time for display
        df = df.copy()
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df = df.dropna(subset=["t"])
        
        if df.empty:
            self.preview_chart.clear()
            return
        
        # Determine appropriate time format based on data span
        time_span = (df["t"].max() - df["t"].min()).days
        if time_span > 365:
            df["time_label"] = df["t"].dt.strftime("%Y-%m")
        elif time_span > 30:
            df["time_label"] = df["t"].dt.strftime("%Y-%m-%d")
        else:
            df["time_label"] = df["t"].dt.strftime("%m-%d %H:%M")
        
        # Check if we have multiple statistics
        statistics = df["statistic"].unique().tolist() if "statistic" in df.columns else ["value"]
        
        if len(statistics) == 1:
            stat = statistics[0]
            stat_df = df[df["statistic"] == stat] if "statistic" in df.columns else df
            
            agg_spec: dict[str, tuple[str, str]] = {"value": ("value", "mean")}
            if "sample_count" in stat_df.columns:
                agg_spec["sample_count"] = ("sample_count", "sum")
            if "outlier_count" in stat_df.columns:
                agg_spec["outlier_count"] = ("outlier_count", "sum")
            agg = stat_df.groupby("time_label", dropna=False).agg(**agg_spec).reset_index()

            feature_name = self._selection_feature_name(selection_info)
            title = f"Statistics over Time ({stat})"
            if feature_name:
                title = f"{feature_name} ({stat})"
            title += self._selection_caption(1, int(total_selected))
            self.preview_chart.set_title(title)
            self.preview_chart.set_dataframe(
                agg,
                category_col="time_label",
                value_col="value",
                series_name=stat,
            )
            tooltip_overrides: dict[tuple[str, str], str] = {}
            for row in agg.itertuples(index=False):
                category = str(getattr(row, "time_label", ""))
                parts: list[str] = []
                if hasattr(row, "sample_count") and pd.notna(getattr(row, "sample_count")):
                    parts.append(f"Count: {int(getattr(row, 'sample_count'))}")
                if stat == "outliers" and hasattr(row, "outlier_count") and pd.notna(getattr(row, "outlier_count")):
                    parts.append(f"Outliers: {int(getattr(row, 'outlier_count'))}")
                if parts:
                    tooltip_overrides[(str(stat), category)] = " | ".join(parts)
            self.preview_chart.set_tooltip_overrides(tooltip_overrides)
        else:
            # Multiple statistics: show grouped bars
            pivot = df.pivot_table(
                index="time_label",
                columns="statistic",
                values="value",
                aggfunc="mean",
            ).reset_index()
            
            self.preview_chart.set_title("Statistics over Time" + self._selection_caption(1, int(total_selected)))
            self.preview_chart.set_multi_series(
                pivot,
                category_col="time_label",
                value_cols=statistics,
            )
            tooltip_overrides: dict[tuple[str, str], str] = {}
            if "sample_count" in df.columns:
                for stat_name in statistics:
                    stat_rows = df[df["statistic"] == stat_name]
                    if stat_rows.empty:
                        continue
                    agg_spec: dict[str, tuple[str, str]] = {"sample_count": ("sample_count", "sum")}
                    if "outlier_count" in stat_rows.columns:
                        agg_spec["outlier_count"] = ("outlier_count", "sum")
                    meta = stat_rows.groupby("time_label", dropna=False).agg(**agg_spec).reset_index()
                    for row in meta.itertuples(index=False):
                        category = str(getattr(row, "time_label", ""))
                        parts = [f"Count: {int(getattr(row, 'sample_count'))}"]
                        if (
                            str(stat_name) == "outliers"
                            and hasattr(row, "outlier_count")
                            and pd.notna(getattr(row, "outlier_count"))
                        ):
                            parts.append(f"Outliers: {int(getattr(row, 'outlier_count'))}")
                        tooltip_overrides[(str(stat_name), category)] = " | ".join(parts)
            self.preview_chart.set_tooltip_overrides(tooltip_overrides)

    # @ai(gpt-5, codex, feature, 2026-03-12)
    def _render_time_chart_multi(self, df: pd.DataFrame, *, total_selected: int) -> None:
        if "t" not in df.columns or "value" not in df.columns or "__chart_series__" not in df.columns:
            self.preview_chart.clear()
            return

        chart_df = df.copy()
        chart_df["t"] = pd.to_datetime(chart_df["t"], errors="coerce")
        chart_df = chart_df.dropna(subset=["t"])
        if chart_df.empty:
            self.preview_chart.clear()
            return

        time_span = (chart_df["t"].max() - chart_df["t"].min()).days
        if time_span > 365:
            chart_df["time_label"] = chart_df["t"].dt.strftime("%Y-%m")
            chart_df["time_order"] = chart_df["t"].dt.to_period("M").dt.start_time
        elif time_span > 30:
            chart_df["time_label"] = chart_df["t"].dt.strftime("%Y-%m-%d")
            chart_df["time_order"] = chart_df["t"].dt.floor("D")
        else:
            chart_df["time_label"] = chart_df["t"].dt.strftime("%m-%d %H:%M")
            chart_df["time_order"] = chart_df["t"]

        ordering = (
            chart_df.groupby("time_label", dropna=False, sort=False)["time_order"]
            .min()
            .reset_index(name="t_min")
            .sort_values("t_min", kind="stable")
        )
        order_map = {str(value): idx for idx, value in enumerate(ordering["time_label"].tolist())}
        series_order = [str(value) for value in chart_df["__chart_series__"].dropna().tolist()]
        series_order = list(dict.fromkeys(series_order))
        pivot = (
            chart_df.pivot_table(
                index="time_label",
                columns="__chart_series__",
                values="value",
                aggfunc="mean",
            )
            .reset_index()
        )
        if not pivot.empty:
            pivot["_order"] = pivot["time_label"].map(lambda value: order_map.get(str(value), len(order_map)))
            pivot = pivot.sort_values("_order", kind="stable").drop(columns=["_order"])
        value_cols = [column for column in series_order if column in pivot.columns]
        if not value_cols:
            self.preview_chart.clear()
            return

        self.preview_chart.set_title(
            "Statistics over Time" + self._selection_caption(len(value_cols), int(total_selected))
        )
        self.preview_chart.set_multi_series(
            pivot,
            category_col="time_label",
            value_cols=value_cols,
        )
        tooltip_overrides: dict[tuple[str, str], str] = {}
        for series_name in value_cols:
            series_rows = chart_df[chart_df["__chart_series__"] == series_name]
            if series_rows.empty:
                continue
            agg_spec: dict[str, tuple[str, str]] = {}
            if "sample_count" in series_rows.columns:
                agg_spec["sample_count"] = ("sample_count", "sum")
            if "outlier_count" in series_rows.columns:
                agg_spec["outlier_count"] = ("outlier_count", "sum")
            if not agg_spec:
                continue
            meta = series_rows.groupby("time_label", dropna=False).agg(**agg_spec).reset_index()
            for row in meta.itertuples(index=False):
                category = str(getattr(row, "time_label", ""))
                parts: list[str] = []
                if hasattr(row, "sample_count") and pd.notna(getattr(row, "sample_count")):
                    parts.append(f"Count: {int(getattr(row, 'sample_count'))}")
                if hasattr(row, "outlier_count") and pd.notna(getattr(row, "outlier_count")):
                    parts.append(f"Outliers: {int(getattr(row, 'outlier_count'))}")
                if parts:
                    tooltip_overrides[(str(series_name), category)] = " | ".join(parts)
        self.preview_chart.set_tooltip_overrides(tooltip_overrides)


__all__ = ["StatisticsPreview"]

