
from __future__ import annotations
# @ai(gpt-5, codex, refactor, 2026-02-26)
from typing import Optional

import pandas as pd
from PySide6.QtGui import QResizeEvent, QShowEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QSplitter, QVBoxLayout, QWidget, QInputDialog
from PySide6.QtCore import Qt, QItemSelectionModel
from ...localization import tr

from ...widgets.fast_table import FastTable
from ...widgets.panel import Panel
from ...charts import GroupBarChart
import logging

logger = logging.getLogger(__name__)



class StatisticsPreview(Panel):
    """Container for the statistics preview panel."""
    _DEFAULT_NOTES = ""
    _NO_GROUP_LABEL = "No group"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("", parent=parent)
        preview_layout = self.content_layout()

        header_layout = QHBoxLayout()
        self.status_label = QLabel(tr("Select features and gather statistics."), self)
        header_layout.addWidget(self.status_label, 1)

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
            context_menu_builder=self._build_table_context_menu,
            sorting_enabled=True,
        )
        self.preview_table.set_stretch_column(-1)
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

        try:
            self.preview_table.selectionChangedInstant.connect(self._on_table_selection_changed)
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)

    # ------------------------------------------------------------------
    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

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

    # ------------------------------------------------------------------
    def set_mode(self, mode: str, group_column: Optional[str] = None) -> None:
        """Set the current statistics mode for chart rendering."""
        self._current_mode = mode
        self._current_group_column = group_column

    def export_frames(self) -> dict[str, pd.DataFrame]:
        datasets: dict[str, pd.DataFrame] = {}
        if self._preview_df is not None:
            datasets[tr("Raw statistics")] = self._preview_df.copy()

        model = self.preview_table.model()
        if model is not None:
            table_df = self._model_to_frame(model)
            if not table_df.empty:
                datasets[tr("Summary table")] = table_df

        selected = self._selected_row_info()
        if selected is not None:
            selected_df = self._filter_preview(selected)
            if not selected_df.empty:
                datasets[tr("Selected feature series")] = selected_df

        return datasets

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
            self.preview_table.setModel(None)
            self._preview_model = None
            self.preview_chart.clear()
            return

        self.preview_table.set_dataframe(summary, include_index=False)
        self._preview_model = self.preview_table.model()
        if self._last_splitter_sizes:
            try:
                self._splitter.setSizes(self._last_splitter_sizes)
            except Exception:
                logger.warning("Exception in update_preview", exc_info=True)

        self._select_default_row()
        self._update_chart_for_selection()

    def _build_feature_summary(
        self,
        df: Optional[pd.DataFrame],
    ) -> tuple[pd.DataFrame, list[dict]]:
        if df is None or df.empty:
            empty_columns = ["Name", "Source", "Unit", "Type", "Notes"]
            empty = pd.DataFrame(columns=empty_columns)
            return empty, []

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
            if original:
                return original

            qualifier = self._normalize_display_value(row.get("new_qualifier"))
            stat = self._normalize_display_value(row.get("statistic"))
            if not qualifier or not stat:
                return qualifier

            q_lower = qualifier.lower()
            s_lower = stat.lower()
            for suffix in (f"_{s_lower}", f" {s_lower}", f"-{s_lower}"):
                if q_lower.endswith(suffix):
                    return qualifier[: -len(suffix)].strip(" _-")
            if q_lower == s_lower:
                return ""
            return qualifier

        def _merge_type_and_stat(row: pd.Series) -> str:
            base_type = _base_type_for_display(row)
            stat = self._normalize_display_value(row.get("statistic")).lower()
            if base_type and stat:
                return f"{base_type} {stat}"
            return base_type or stat

        raw_columns = [
            "label",
            "statistic",
            "base_name",
            "source",
            "unit",
            "new_qualifier",
            "notes",
        ]
        raw_present = [col for col in raw_columns if col in working.columns]
        raw_summary = working[raw_present].drop_duplicates().reset_index(drop=True)
        raw_rows = raw_summary.to_dict("records")

        summary_columns: dict[str, pd.Series] = {
            "Name": raw_summary.get("base_name", pd.Series(dtype=object)),
            "Source": raw_summary.get("source", pd.Series(dtype=object)),
            "Unit": raw_summary.get("unit", pd.Series(dtype=object)),
            "Type": raw_summary.apply(
                lambda row: _merge_type_and_stat(
                    pd.Series(
                        {
                            "new_qualifier": row.get("new_qualifier"),
                            "statistic": row.get("statistic"),
                        }
                    )
                ),
                axis=1,
            )
            if not raw_summary.empty
            else pd.Series(dtype=object),
        }
        summary_columns["Notes"] = raw_summary.get("notes", pd.Series(dtype=object))
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
        model = self.preview_table.model()
        if model is None:
            return None
        sel_model = self.preview_table.selectionModel()
        if sel_model is None:
            return None
        indexes = sel_model.selectedIndexes()
        if not indexes:
            return None
        row = int(indexes[0].row())
        if 0 <= row < len(self._feature_rows):
            return dict(self._feature_rows[row] or {})
        row_payload = {}
        for col in range(model.columnCount()):
            header = model.headerData(col, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
            if header is None:
                continue
            value = model.data(model.index(row, col), Qt.ItemDataRole.DisplayRole)
            row_payload[str(header)] = value
        resolved = self._resolve_selection_from_display(row_payload)
        if resolved is not None:
            return resolved
        if row_payload:
            return row_payload
        return None

    def _resolve_selection_from_display(self, row_payload: dict) -> Optional[dict]:
        if not self._feature_rows:
            return None
        display_to_raw = {
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

        label = selection.get("Feature") or selection.get("label")
        if label is not None and "label" in df.columns:
            mask &= df["label"] == label

        stat = selection.get("Statistic") or selection.get("statistic")
        if stat is not None and "statistic" in df.columns:
            mask &= df["statistic"] == stat

        import_id = selection.get("import_id")
        if import_id is not None and "import_id" in df.columns:
            mask &= df["import_id"] == import_id

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
        for key in ("label", "statistic", "import_id", "system", "Dataset", "source", "unit"):
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

    def _update_chart_for_selection(self) -> None:
        selection = self._selected_row_info()
        if selection is None:
            self.preview_chart.clear()
            return

        subset = self._filter_preview(selection)
        self._update_chart(subset, selection_info=selection)

    def _update_chart(self, df: pd.DataFrame, selection_info: Optional[dict] = None) -> None:
        """Update the chart based on the current mode and data."""
        if df is None or df.empty:
            self.preview_chart.clear()
            return
        
        # Check which columns we have
        has_group_value = "group_value" in df.columns
        has_value = "value" in df.columns
        has_statistic = "statistic" in df.columns
        has_t = "t" in df.columns
        
        if not has_value:
            self.preview_chart.clear()
            return

        # Decide chart rendering based on mode
        if self._current_mode == "column" and has_group_value:
            # Group by column mode: show bars per group
            self._render_group_chart(df, selection_info=selection_info)
        elif has_t:
            # Time-based mode: show bars per time period
            self._render_time_chart(df, selection_info=selection_info)
        else:
            self.preview_chart.clear()

    def _render_group_chart(self, df: pd.DataFrame, selection_info: Optional[dict] = None) -> None:
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
            
            feature_name = None
            if selection_info:
                feature_name = selection_info.get("Feature") or selection_info.get("label")
            title = f"Statistics by Group ({stat})"
            if feature_name:
                title = f"{feature_name} ({stat})"
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

            feature_name = selection_info.get("Feature") if selection_info else None
            if selection_info and not feature_name:
                feature_name = selection_info.get("label")
            title = feature_name or "Statistics by Group"
            self.preview_chart.set_title(title or "Statistics by Group")
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

    def _render_time_chart(self, df: pd.DataFrame, selection_info: Optional[dict] = None) -> None:
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

            feature_name = None
            if selection_info:
                feature_name = selection_info.get("Feature") or selection_info.get("label")
            title = f"Statistics over Time ({stat})"
            if feature_name:
                title = f"{feature_name} ({stat})"
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
            
            self.preview_chart.set_title("Statistics over Time")
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


__all__ = ["StatisticsPreview"]

