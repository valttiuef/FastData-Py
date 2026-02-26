
from __future__ import annotations
from typing import Optional, Sequence, Any

import logging

import pandas as pd
from PySide6.QtCore import QObject, Qt, Signal, QTimer, QItemSelectionModel
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QGroupBox, QLineEdit, QVBoxLayout
from ..localization import tr

from ..models.features_model import FeaturesFilterProxyModel, FeaturesTableModel
from ..models.hybrid_pandas_model import HybridPandasModel
from ..threading.runner import run_in_thread
from ..threading.utils import run_in_main_thread
from .fast_table import FastTable
from ..utils import toast_error

logger = logging.getLogger(__name__)


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
        self._use_selection_filter: bool = False  # Whether to show only selected features
        if self._data_model is not None:
            self._data_model.database_changed.connect(self._on_database_changed)
            self._data_model.selected_features_changed.connect(self._on_selected_features_changed)
            self._data_model.features_list_changed.connect(self._on_features_list_changed)
        self.reload_features()

    def set_use_selection_filter(self, enabled: bool) -> None:
        """Enable/disable filtering features by current selection."""
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
        if updated == self._filters and not reload:
            return
        self._filters = updated
        if reload:
            self.reload_features()

    def reload_features(self) -> None:
        model = self._data_model
        filters = dict(self._filters)

        def _load() -> pd.DataFrame:
            try:
                return self._load_features_dataframe(model, filters)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to load features: %s", exc)
                raise exc

        def _apply(df: pd.DataFrame) -> None:
            self.features_loaded.emit(df)

        def _handle_error(message: str) -> None:
            self.load_failed.emit(message)
            fallback = pd.DataFrame(
                columns=["feature_id", "name", "source", "unit", "type", "lag_seconds", "tags", "notes"],
            )
            self.features_loaded.emit(fallback)

        run_in_thread(
            _load,
            on_result=lambda df: run_in_main_thread(_apply, df),
            on_error=lambda msg: run_in_main_thread(_handle_error, msg),
            owner=self,
            key="features_list_load",
            cancel_previous=True,
        )

    # ------------------------------------------------------------------
    def _on_database_changed(self, *_args) -> None:
        self.reload_features()

    def _on_selected_features_changed(self, selected_features: pd.DataFrame) -> None:
        """Called when DatabaseModel selection changes for selection-filtered lists."""
        if not self._use_selection_filter:
            return
        self.features_loaded.emit(selected_features)

    def _on_features_list_changed(self) -> None:
        self.reload_features()

    def _load_features_dataframe(
        self,
        model: Optional[HybridPandasModel],
        filters: dict[str, Any],
    ) -> pd.DataFrame:
        systems = filters.get("systems")
        datasets = filters.get("datasets")
        import_ids = filters.get("import_ids")
        tags = filters.get("tags")
        if model is None:
            return pd.DataFrame(
                columns=["feature_id", "name", "source", "unit", "type", "lag_seconds", "tags", "notes"],
            )
        # Load based on system/Dataset filters
        if systems or datasets:
            return model.features_for_systems_datasets(
                systems=systems,
                datasets=datasets,
                import_ids=import_ids,
                tags=tags,
            )
        return model.features_df(import_ids=import_ids, tags=tags)

class FeaturesListWidget(QGroupBox):
    """Reusable widget that exposes a searchable, filterable features table.
    
    Can optionally filter to show only selected features via the data_model's
    active selection state (set via SelectionsViewModel).
    """

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
        self._table_model = FeaturesTableModel()
        self._proxy_model = FeaturesFilterProxyModel(self)
        self._proxy_model.setSourceModel(self._table_model)
        self._proxy_model.setDynamicSortFilter(True)

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
        self._table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        layout.addWidget(self._table, 1)

        self._pending_search_text = ""
        self._suppress_selection_emit = False
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(120)
        self._search_timer.timeout.connect(self._apply_search_text)
        self._search_edit.textChanged.connect(self._queue_search_text)
        self._selection_emit_timer = QTimer(self)
        self._selection_emit_timer.setSingleShot(True)
        self._selection_emit_timer.setInterval(0)
        self._selection_emit_timer.timeout.connect(self._emit_selection_changed)

        selection = self._table.selectionModel()
        if selection is not None:
            selection.selectionChanged.connect(self._queue_selection_changed)

        self._view_model = FeaturesListWidgetViewModel(
            data_model=data_model,
            parent=self,
        )
        self._view_model.features_loaded.connect(self._apply_dataframe)
        self._view_model.load_failed.connect(self._log_failure)
        self._last_selection_ids: tuple[int, ...] = ()

    # ------------------------------------------------------------------
    def _queue_search_text(self, text: str) -> None:
        self._pending_search_text = text or ""
        self._search_timer.start()

    def _apply_search_text(self) -> None:
        if self._pending_search_text.strip().lower() == self._proxy_model.search_text():
            return
        self._suppress_selection_emit = True
        try:
            self._proxy_model.set_search_text(self._pending_search_text)
        finally:
            QTimer.singleShot(0, self._clear_search_suppression)

    def _clear_search_suppression(self) -> None:
        self._suppress_selection_emit = False

    def _queue_selection_changed(self, *_args) -> None:
        self._selection_emit_timer.start()

    def _build_context_menu(self, menu, _pos, _table):
        action = QAction(tr("Details..."), menu)
        payloads = self.selected_payloads()
        action.setEnabled(bool(payloads))
        action.triggered.connect(lambda: self.details_requested.emit(payloads))
        menu.addAction(action)

    # ------------------------------------------------------------------
    @property
    def table_view(self) -> FastTable:
        return self._table

    @property
    def search_edit(self) -> QLineEdit:
        return self._search_edit

    def set_use_selection_filter(self, enabled: bool) -> None:
        """Enable/disable filtering features by current selection."""
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

    def selected_payloads(self) -> list[dict]:
        selection = self._table.selectionModel()
        if not selection:
            return []
        payloads: list[dict] = []
        for index in selection.selectedRows():
            if not index.isValid():
                continue
            source_index = self._proxy_model.mapToSource(index)
            payload = self._table_model.feature_payload_at(source_index.row())
            if payload:
                payloads.append(payload)
        return payloads

    def selected_feature_ids(self) -> list[int]:
        payloads = self.selected_payloads()
        ids: list[int] = []
        for payload in payloads:
            fid = payload.get("feature_id") if isinstance(payload, dict) else None
            if fid is None:
                continue
            try:
                ids.append(int(fid))
            except Exception:
                continue
        return ids

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
        selection.clearSelection()

    def clear_selection_and_suppress_autoselect(self) -> None:
        """Clear current selection and prevent auto-select on the next reload."""
        self._suppress_autoselect = True
        self.clear_selection()

    # ------------------------------------------------------------------
    def _apply_dataframe(self, df: pd.DataFrame) -> None:
        # Remember currently selected feature IDs before updating the model
        previously_selected_ids = set(self.selected_feature_ids())
        if self._suppress_autoselect:
            previously_selected_ids = set()

        previous_suppression = self._suppress_selection_emit
        self._suppress_selection_emit = True
        try:
            selection = self._table.selectionModel()
            if selection is not None:
                selection.clearSelection()
            try:
                self._table_model.set_dataframe(df)
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

            # Try to restore selection if we had previously selected features
            selection_restored = False
            if previously_selected_ids and self._table_model.rowCount() > 0:
                selection_restored = self._restore_selection(previously_selected_ids)

            # If there was a previous explicit selection but none of those features exist anymore,
            # keep the selection empty and notify the user instead of silently selecting row 0.
            if (
                previously_selected_ids
                and not selection_restored
            ):
                try:
                    toast_error(
                        tr("A previously selected feature is no longer available. Please select a feature again."),
                        title=tr("Features"),
                    )
                except Exception:
                    logger.warning("Exception in _apply_dataframe", exc_info=True)
            # For first-time loads (no previous explicit selection), keep legacy auto-select behavior.
            elif (
                not self._suppress_autoselect
                and not self.selected_payloads()
                and self._table_model.rowCount() > 0
            ):
                try:
                    self._table.selectRow(0)
                except Exception:
                    logger.warning("Exception in _apply_dataframe", exc_info=True)
        finally:
            self._suppress_selection_emit = previous_suppression

        if self._suppress_autoselect:
            self._suppress_autoselect = False

        self.features_reloaded.emit(df)
        self._queue_selection_changed()

    def _restore_selection(self, feature_ids: set[int]) -> bool:
        """Try to restore selection to features with the given IDs.
        
        Returns True if at least one feature was selected.
        """
        selection = self._table.selectionModel()
        if selection is None:
            return False
        
        rows_to_select = []
        for row in range(self._table_model.rowCount()):
            payload = self._table_model.feature_payload_at(row)
            if payload and payload.get("feature_id") in feature_ids:
                rows_to_select.append(row)
        
        if not rows_to_select:
            return False
        
        # Select all matching rows through the proxy model without replacing prior selections.
        try:
            selection.clearSelection()
            for row in rows_to_select:
                source_index = self._table_model.index(row, 0)
                proxy_index = self._proxy_model.mapFromSource(source_index)
                if proxy_index.isValid():
                    selection.select(
                        proxy_index,
                        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
                    )
            return len(rows_to_select) > 0
        except Exception:
            return False

    def _log_failure(self, message: str) -> None:  # pragma: no cover - defensive logging
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
        selection_ids = tuple(sorted(self.selected_feature_ids()))
        if selection_ids == self._last_selection_ids:
            return
        self._last_selection_ids = selection_ids
        payloads = self.selected_payloads()
        self.selection_changed.emit(payloads)


__all__ = ["FeaturesListWidget", "FeaturesListWidgetViewModel"]
