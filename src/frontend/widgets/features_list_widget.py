from __future__ import annotations
from typing import Optional, Sequence, Any

import logging
import pandas as pd

from PySide6.QtCore import (
    QObject,
    Qt,
    Signal,
    QTimer,
    QItemSelectionModel,
    QItemSelection,
    QSignalBlocker,
)
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QGroupBox, QLineEdit, QVBoxLayout

from ..localization import tr
from ..models.features_model import FeaturesTableModel
from ..models.hybrid_pandas_model import HybridPandasModel
from .fast_table import FastTable, FastPandasProxyModel
from ..utils import toast_error

logger = logging.getLogger(__name__)

NO_TAG_FILTER_VALUE = "__NO_TAG__"


def _is_selector_visible_feature_type(type_value: object) -> bool:
    """
    Keep DataSelector feature lists focused on numeric/measured features.
    Hide text/group pseudo-features (including wrapped variants like
    "group (...)" and "text (...)"), which are managed in Selections tab.
    """
    text = str(type_value or "").strip()
    if not text:
        return True
    lowered = text.casefold()
    while True:
        if lowered in {"text", "group"}:
            return False
        if lowered.startswith("text (") and lowered.endswith(")"):
            lowered = lowered[6:-1].strip().casefold()
            continue
        if lowered.startswith("group (") and lowered.endswith(")"):
            lowered = lowered[7:-1].strip().casefold()
            continue
        return True


class FeaturesListWidgetViewModel(QObject):
    """Keeps a features list in sync with a HybridPandasModel.

    When enabled, automatically filters features to show only those in the current
    selection (via DatabaseModel.selected_features_changed signal).
    """

    features_loaded = Signal(object)
    load_failed = Signal(str)

    def __init__(
        self,
        *,
        data_model: Optional[HybridPandasModel] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._filters: dict[str, Any] = {
            "systems": None,
            "datasets": None,
            "import_ids": None,
            "tags": None,
        }
        self._data_model: Optional[HybridPandasModel] = data_model
        self._use_selection_filter: bool = False
        self._base_features_df: pd.DataFrame = self._empty_dataframe()
        if self._data_model is not None:
            self._data_model.database_changed.connect(self._on_database_changed)
            self._data_model.selected_features_changed.connect(self._on_selected_features_changed)
            self._data_model.features_list_changed.connect(self._on_features_list_changed)

    def set_use_selection_filter(self, enabled: bool) -> None:
        if self._use_selection_filter == enabled:
            return
        self._use_selection_filter = enabled
        self.reload_features()

    @property
    def use_selection_filter(self) -> bool:
        return bool(self._use_selection_filter)

    def set_filters(
        self,
        *,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
        tags: Optional[Sequence[str]] = None,
        reload: bool = True,
    ) -> None:
        updated = {
            "systems": systems if systems is None else tuple(systems),
            "datasets": datasets if datasets is None else tuple(datasets),
            "import_ids": import_ids if import_ids is None else tuple(int(v) for v in import_ids),
            "tags": tags if tags is None else tuple(tags),
        }
        if updated == self._filters:
            return
        self._filters = updated
        if reload:
            self._emit_filtered_features()

    def reload_features(self) -> None:
        model = self._data_model
        try:
            df = self._load_base_features_dataframe(model)
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to load features: %s", exc)
            self.load_failed.emit(str(exc))
            df = self._empty_dataframe()
        self._base_features_df = df
        self._emit_filtered_features()

    def _on_database_changed(self, *_args) -> None:
        self.reload_features()

    def _on_selected_features_changed(self, selected_features: pd.DataFrame) -> None:
        if not self._use_selection_filter:
            return
        self._base_features_df = self._prepare_features_dataframe(selected_features)
        self._emit_filtered_features()

    def _on_features_list_changed(self) -> None:
        self.reload_features()

    def _load_base_features_dataframe(
        self,
        model: Optional[HybridPandasModel],
    ) -> pd.DataFrame:
        if model is None:
            return self._empty_dataframe()
        df = model.features_df()
        return self._prepare_features_dataframe(df)

    def _emit_filtered_features(self) -> None:
        df = self._filter_dataframe(self._base_features_df, self._filters)
        self.features_loaded.emit(df)

    def _prepare_features_dataframe(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return self._empty_dataframe()
        if "type" not in df.columns:
            return df.copy()
        try:
            mask = df["type"].map(_is_selector_visible_feature_type)
            return df.loc[mask].reset_index(drop=True)
        except Exception:
            logger.warning("Exception while filtering selector-visible feature types", exc_info=True)
            return df.copy()

    def _filter_dataframe(
        self,
        df: pd.DataFrame,
        filters: dict[str, Any],
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return self._empty_dataframe()

        filtered = df
        systems = self._normalize_text_filter_allow_empty(filters.get("systems"))
        datasets = self._normalize_text_filter_allow_empty(filters.get("datasets"))
        import_ids = self._normalize_int_filter_allow_empty(filters.get("import_ids"))
        tags = self._normalize_text_filter(filters.get("tags"))
        include_no_tag = False
        if tags is not None and NO_TAG_FILTER_VALUE in tags:
            include_no_tag = True
            tags = {value for value in tags if value != NO_TAG_FILTER_VALUE}

        if systems is not None:
            if not systems:
                return filtered.iloc[0:0].copy()
            if "systems" in filtered.columns:
                system_sets = filtered["systems"].map(self._normalize_text_cell)
                filtered = filtered.loc[system_sets.map(lambda values: bool(values & systems))]
            elif "system" in filtered.columns:
                filtered = filtered.loc[
                    filtered["system"].astype("string").fillna("").str.strip().isin(list(systems))
                ]

        if datasets is not None:
            if not datasets:
                return filtered.iloc[0:0].copy()
            if "datasets" in filtered.columns:
                dataset_sets = filtered["datasets"].map(self._normalize_text_cell)
                filtered = filtered.loc[dataset_sets.map(lambda values: bool(values & datasets))]
            elif "dataset" in filtered.columns:
                filtered = filtered.loc[
                    filtered["dataset"].astype("string").fillna("").str.strip().isin(list(datasets))
                ]

        if import_ids is not None:
            if not import_ids:
                return filtered.iloc[0:0].copy()
            if "import_ids" in filtered.columns:
                import_sets = filtered["import_ids"].map(self._normalize_int_cell)
                filtered = filtered.loc[import_sets.map(lambda values: bool(values & import_ids))]

        if tags is not None:
            if not tags and not include_no_tag:
                return filtered.iloc[0:0].copy()
            if "tags" in filtered.columns:
                tag_sets = filtered["tags"].map(self._normalize_text_cell)
                filtered = filtered.loc[
                    tag_sets.map(
                        lambda values: bool(values & tags) or (include_no_tag and not values)
                    )
                ]
            elif tags:
                return filtered.iloc[0:0].copy()

        return filtered.reset_index(drop=True)

    @staticmethod
    def _empty_dataframe() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["feature_id", "name", "source", "unit", "type", "lag_seconds", "tags", "notes"],
        )

    @staticmethod
    def _normalize_text_filter(values: Any) -> Optional[set[str]]:
        if values is None:
            return None
        normalized: set[str] = set()
        for value in values or []:
            text = str(value or "").strip()
            if text:
                normalized.add(text)
        return normalized or None

    @staticmethod
    def _normalize_text_filter_allow_empty(values: Any) -> Optional[set[str]]:
        if values is None:
            return None
        normalized: set[str] = set()
        for value in values or []:
            text = str(value or "").strip()
            if text:
                normalized.add(text)
        return normalized

    @staticmethod
    def _normalize_int_filter(values: Any) -> Optional[set[int]]:
        if values is None:
            return None
        normalized: set[int] = set()
        for value in values or []:
            try:
                normalized.add(int(value))
            except Exception:
                continue
        return normalized or None

    @staticmethod
    def _normalize_int_filter_allow_empty(values: Any) -> Optional[set[int]]:
        if values is None:
            return None
        normalized: set[int] = set()
        for value in values or []:
            try:
                normalized.add(int(value))
            except Exception:
                continue
        return normalized

    @staticmethod
    def _normalize_text_cell(value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, (str, bytes, dict)):
            text = str(value).strip()
            return {text} if text else set()
        if hasattr(value, "__iter__"):
            values: set[str] = set()
            for item in value:
                text = str(item or "").strip()
                if text:
                    values.add(text)
            return values
        text = str(value).strip()
        return {text} if text else set()

    @staticmethod
    def _normalize_int_cell(value: Any) -> set[int]:
        if value is None:
            return set()
        candidates = value if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)) else [value]
        values: set[int] = set()
        for item in candidates:
            try:
                values.add(int(item))
            except Exception:
                continue
        return values


class FeaturesListWidget(QGroupBox):
    """Reusable widget that exposes a searchable, filterable features table."""

    selection_changed = Signal(list)
    features_reloaded = Signal(object)
    details_requested = Signal(list)  # Pass selected feature payloads

    def __init__(
        self,
        *,
        title: str = "Features",
        parent=None,
        data_model: Optional[HybridPandasModel] = None,
    ) -> None:
        super().__init__(tr(title), parent)
        self._suppress_autoselect = False
        self._pending_search_text = ""
        self._suppress_selection_emit = False
        self._last_selection_ids: tuple[int, ...] = ()
        self._selection_memory_ids: set[int] = set()
        self._selection_cache_valid = False
        self._selected_payloads_cache: list[dict] = []
        self._visible_selected_ids_cache: list[int] = []
        self._payloads_by_feature_id: dict[int, dict] = {}

        # Domain source model (keeps payload + headers)
        self._table_model = FeaturesTableModel()

        # Fast vectorized proxy (sorting/filtering)
        self._proxy_model = FastPandasProxyModel(self)
        self._proxy_model.setSourceModel(self._table_model)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self._search_edit = QLineEdit(self)
        self._search_edit.setPlaceholderText(tr("Search features…"))
        layout.addWidget(self._search_edit)

        self._table = FastTable(
            select="rows",
            single_selection=False,
            tint_current_selection=True,
            editable=False,
            context_menu_builder=self._build_context_menu,
            parent=self,
        )
        self._table.setModel(self._proxy_model)
        self._table.setSortingEnabled(True)
        self._table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        layout.addWidget(self._table, 1)

        # Search debounce
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(120)
        self._search_timer.timeout.connect(self._apply_search_text)
        self._search_edit.textChanged.connect(self._queue_search_text)

        self._selection_restore_timer = QTimer(self)
        self._selection_restore_timer.setSingleShot(True)
        self._selection_restore_timer.setInterval(0)
        self._selection_restore_timer.timeout.connect(self._restore_selection_after_proxy_change)
        self._proxy_model.modelReset.connect(self._schedule_restore_selection_after_proxy_change)

        # Selection debounce
        self._selection_emit_timer = QTimer(self)
        self._selection_emit_timer.setSingleShot(True)
        # Coalesce bursty row-selection updates (e.g. Ctrl+A on large feature lists).
        self._selection_emit_timer.setInterval(60)
        self._selection_emit_timer.timeout.connect(self._emit_selection_changed)

        selection = self._table.selectionModel()
        if selection is not None:
            selection.selectionChanged.connect(self._queue_selection_changed)

        # View model for loading
        if data_model is not None:
            data_model.database_changed.connect(self._on_database_changed_reset_selection)

        self._view_model = FeaturesListWidgetViewModel(
            data_model=data_model,
            parent=self,
        )
        self._view_model.features_loaded.connect(self._apply_dataframe)
        self._view_model.load_failed.connect(self._log_failure)

    # ------------------------------------------------------------------
    @property
    def table_view(self) -> FastTable:
        return self._table

    @property
    def search_edit(self) -> QLineEdit:
        return self._search_edit

    def set_use_selection_filter(self, enabled: bool) -> None:
        self._view_model.set_use_selection_filter(enabled)

    @property
    def use_selection_filter(self) -> bool:
        return self._view_model.use_selection_filter

    def set_filters(
        self,
        *,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
        tags: Optional[Sequence[str]] = None,
        reload: bool = True,
    ) -> None:
        self._view_model.set_filters(
            systems=systems,
            datasets=datasets,
            import_ids=import_ids,
            tags=tags,
            reload=reload,
        )

    def reload_features(self) -> None:
        self._view_model.reload_features()

    # ------------------------------------------------------------------
    def _queue_search_text(self, text: str) -> None:
        self._pending_search_text = text or ""
        self._search_timer.start()

    def _apply_search_text(self) -> None:
        text = (self._pending_search_text or "").strip()
        # proxy rebuild is fast; our own timer already debounces
        visible_selected_ids = set(self._visible_selected_feature_ids())
        if visible_selected_ids:
            self._selection_memory_ids = set(visible_selected_ids)
        self._suppress_selection_emit = True
        try:
            self._proxy_model.set_filter(text, columns=None, case_sensitive=False, debounce_ms=0)
        finally:
            QTimer.singleShot(0, self._clear_search_suppression)

    def _clear_search_suppression(self) -> None:
        self._suppress_selection_emit = False

    def _invalidate_selection_cache(self) -> None:
        self._selection_cache_valid = False
        self._selected_payloads_cache = []
        self._visible_selected_ids_cache = []

    def _queue_selection_changed(
        self,
        selected: QItemSelection | None = None,
        deselected: QItemSelection | None = None,
    ) -> None:
        if self._suppress_selection_emit:
            return
        self._invalidate_selection_cache()
        changed_rows: set[int] = set()
        for batch in (selected, deselected):
            if batch is None:
                continue
            try:
                for index in batch.indexes():
                    if index.isValid():
                        changed_rows.add(int(index.row()))
            except Exception:
                logger.warning("Exception in _queue_selection_changed", exc_info=True)
                self._selection_emit_timer.start()
                return
        if len(changed_rows) <= 2:
            self._selection_emit_timer.setInterval(0)
            self._selection_emit_timer.stop()
            self._selection_emit_timer.start()
            return
        self._selection_emit_timer.setInterval(60)
        self._selection_emit_timer.start()

    def _schedule_restore_selection_after_proxy_change(self, *_args) -> None:
        self._invalidate_selection_cache()
        self._selection_restore_timer.start()

    def _build_context_menu(self, menu, _pos, _table):
        action = QAction(tr("Details..."), menu)
        payloads = self.selected_payloads()
        action.setEnabled(bool(payloads))
        action.triggered.connect(lambda: self.details_requested.emit(payloads))
        menu.addAction(action)

    # ------------------------------------------------------------------
    def selected_payloads(self) -> list[dict]:
        if self._selection_cache_valid:
            return list(self._selected_payloads_cache)
        selection = self._table.selectionModel()
        if not selection:
            return []
        payloads: list[dict] = []
        ids: list[int] = []
        for index in selection.selectedRows():
            if not index.isValid():
                continue
            source_index = self._proxy_model.mapToSource(index)
            payload = self._table_model.feature_payload_at(source_index.row())
            if payload:
                payloads.append(payload)
                fid = payload.get("feature_id") if isinstance(payload, dict) else None
                if fid is None:
                    continue
                try:
                    ids.append(int(fid))
                except Exception:
                    continue
        self._selected_payloads_cache = list(payloads)
        self._visible_selected_ids_cache = list(ids)
        self._selection_cache_valid = True
        return list(self._selected_payloads_cache)

    def selected_feature_ids(self) -> list[int]:
        if self._selection_memory_ids:
            return sorted(int(fid) for fid in self._selection_memory_ids)
        return self._visible_selected_feature_ids()

    def _visible_selected_feature_ids(self) -> list[int]:
        if not self._selection_cache_valid:
            self.selected_payloads()
        return list(self._visible_selected_ids_cache)

    def all_payloads(self) -> list[dict]:
        payloads: list[dict] = []
        rows = self._table_model.rowCount()
        for row in range(rows):
            payload = self._table_model.feature_payload_at(row)
            if payload:
                payloads.append(payload)
        return payloads

    def clear_selection(self) -> None:
        selection = self._table.selectionModel()
        if selection is None:
            return
        self._invalidate_selection_cache()
        selection.clearSelection()

    def _on_database_changed_reset_selection(self, *_args) -> None:
        had_selection = bool(self._selection_memory_ids) or bool(self._visible_selected_feature_ids())
        self._selection_memory_ids.clear()
        self._invalidate_selection_cache()
        selection = self._table.selectionModel()
        if selection is not None:
            previous_suppression = self._suppress_selection_emit
            self._suppress_selection_emit = True
            try:
                selection.clearSelection()
            finally:
                self._suppress_selection_emit = previous_suppression
        if had_selection:
            self._last_selection_ids = ()
            self.selection_changed.emit([])

    # ------------------------------------------------------------------
    def _apply_dataframe(self, df: pd.DataFrame) -> None:
        # @ai(gpt-5, codex, performance-fix, 2026-03-23)
        previously_selected_ids = set(self._selection_memory_ids)
        if not previously_selected_ids:
            previously_selected_ids = set(self.selected_feature_ids())
        if previously_selected_ids:
            self._selection_memory_ids = set(previously_selected_ids)
        if self._suppress_autoselect:
            previously_selected_ids = set()

        previous_suppression = self._suppress_selection_emit
        self._suppress_selection_emit = True
        self._table.setUpdatesEnabled(False)
        self._invalidate_selection_cache()

        try:
            data_changed = False
            try:
                data_changed = bool(self._table_model.set_dataframe(df))
                self._payloads_by_feature_id = self._table_model._payload_map_by_feature_id()
            except Exception as exc:
                logger.exception(
                    "Features list received invalid dataframe payload: %s (type=%s shape=%s columns=%s)",
                    exc,
                    type(df).__name__,
                    getattr(df, "shape", None),
                    list(df.columns) if isinstance(df, pd.DataFrame) else None,
                )
                toast_error(
                    tr("Features table update failed due to invalid data shape."),
                    title=tr("Features"),
                )
                return

            # Fast path: filtered dataframe unchanged -> preserve current view selection
            # and skip expensive clear/reselect work.
            if data_changed:
                selection = self._table.selectionModel()
                if selection is not None:
                    selection.clearSelection()

                # Re-apply existing sort after data reload (keeps UX stable)
                try:
                    hdr = self._table.horizontalHeader()
                    sort_col = int(hdr.sortIndicatorSection())
                    sort_ord = hdr.sortIndicatorOrder()
                    self._table.sortByColumn(sort_col, sort_ord)
                except Exception:
                    pass

                selection_restored = False
                if previously_selected_ids and self._table_model.rowCount() > 0:
                    selection_restored = self._restore_selection(previously_selected_ids)
                if previously_selected_ids and selection_restored:
                    self._selection_memory_ids = set(previously_selected_ids)

        finally:
            self._table.setUpdatesEnabled(True)
            self._suppress_selection_emit = previous_suppression

        if self._suppress_autoselect:
            self._suppress_autoselect = False

        self.features_reloaded.emit(df)

    def _restore_selection(self, feature_ids: set[int]) -> bool:
        # @ai(gpt-5, codex, performance-fix, 2026-03-23)
        selection = self._table.selectionModel()
        if selection is None:
            return False

        rows_to_select = self._table_model._rows_for_feature_ids(feature_ids)

        if not rows_to_select:
            return False

        try:
            proxy_rows: list[int] = []
            for row in rows_to_select:
                source_index = self._table_model.index(row, 0)
                proxy_index = self._proxy_model.mapFromSource(source_index)
                if proxy_index.isValid():
                    proxy_rows.append(int(proxy_index.row()))

            if not proxy_rows:
                return False

            proxy_rows = sorted(set(proxy_rows))
            selection_ranges = QItemSelection()
            start_row = proxy_rows[0]
            prev_row = proxy_rows[0]
            for proxy_row in proxy_rows[1:]:
                if proxy_row == prev_row + 1:
                    prev_row = proxy_row
                    continue
                selection_ranges.select(
                    self._proxy_model.index(start_row, 0),
                    self._proxy_model.index(prev_row, 0),
                )
                start_row = proxy_row
                prev_row = proxy_row
            selection_ranges.select(
                self._proxy_model.index(start_row, 0),
                self._proxy_model.index(prev_row, 0),
            )

            blocker = QSignalBlocker(selection)
            try:
                selection.select(
                    selection_ranges,
                    QItemSelectionModel.SelectionFlag.ClearAndSelect
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
            finally:
                del blocker
            return True
        except Exception:
            return False

    def _restore_selection_after_proxy_change(self) -> None:
        if self._suppress_selection_emit or not self._selection_memory_ids:
            return
        if self.selected_payloads():
            return
        restored = self._restore_selection(set(self._selection_memory_ids))
        if restored:
            self._queue_selection_changed()

    def _payloads_for_feature_ids(self, feature_ids: Sequence[int]) -> list[dict]:
        return [self._payloads_by_feature_id[fid] for fid in feature_ids if fid in self._payloads_by_feature_id]

    def _log_failure(self, message: str) -> None:  # pragma: no cover
        logger.warning("Features list reload failed: %s", message)
        try:
            toast_error(
                tr("Failed to load features: {message}").format(message=message),
                title=tr("Features"),
            )
        except Exception:
            logger.warning("Exception in _log_failure", exc_info=True)

    def _emit_selection_changed(self) -> None:
        if self._suppress_selection_emit:
            return
        payloads = self.selected_payloads()
        ids = list(self._visible_selected_ids_cache)
        selected_ids = set(ids)
        self._selection_memory_ids = set(selected_ids)
        selection_ids = tuple(sorted(self._selection_memory_ids))
        if selection_ids == self._last_selection_ids:
            return
        self._last_selection_ids = selection_ids
        if selection_ids:
            ordered_payloads = self._payloads_for_feature_ids(selection_ids)
            if len(ordered_payloads) != len(selection_ids):
                return
            self.selection_changed.emit(ordered_payloads)
            return
        self.selection_changed.emit([])


__all__ = ["FeaturesListWidget", "FeaturesListWidgetViewModel"]
