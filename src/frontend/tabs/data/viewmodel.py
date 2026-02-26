
from __future__ import annotations
from typing import Any, Optional, Sequence

import pandas as pd
from PySide6.QtCore import QObject, Signal

from ...models.hybrid_pandas_model import HybridPandasModel
from .import_preview_logic import build_import_preview_payload
import logging

logger = logging.getLogger(__name__)


class DataViewModel(QObject):
    """View-model providing a stable interface over :class:`HybridPandasModel`."""

    progress = Signal(str, int, int, str)
    # Emitted with the current db handle (or None) whenever the underlying
    # database connection/path changes.
    database_changed = Signal(object)

    # --- UI coordination signals -------------------------------------------------
    import_requested = Signal()
    new_requested = Signal()
    load_requested = Signal()
    save_requested = Signal()

    refresh_features_requested = Signal()

    systems_changed = Signal(list, list)
    datasets_changed = Signal(list, list)
    months_changed = Signal(list)
    groups_changed = Signal(list)
    date_range_changed = Signal(object, object)
    preprocessing_changed = Signal(str, object)
    tags_changed = Signal(list)

    def __init__(self, model: HybridPandasModel, parent: Optional[QObject] = None) -> None:
        """
        `model` is the shared HybridPandasModel instance that encapsulates
        SettingsModel + database routing. All DB access goes through this model.
        """
        super().__init__(parent)
        self._model: Optional[HybridPandasModel] = None
        self._set_model(model)

    # ------------------------------------------------------------------
    def _set_model(self, model: Optional[HybridPandasModel]) -> None:
        """Swap the underlying HybridPandasModel and wire signals."""
        if getattr(self, "_model", None) is model:
            return

        old_model = getattr(self, "_model", None)
        if old_model is not None:
            # Disconnect old signals
            try:
                old_model.progress.disconnect(self.progress.emit)
            except Exception:
                logger.warning("Exception in _set_model", exc_info=True)
            try:
                old_model.database_changed.disconnect(self._on_model_database_changed)
            except Exception:
                logger.warning("Exception in _set_model", exc_info=True)

        self._model = model

        if self._model is not None:
            # Forward progress
            try:
                self._model.progress.connect(self.progress.emit)
            except Exception:
                logger.warning("Exception in _set_model", exc_info=True)
            # Listen for DB path changes
            try:
                self._model.database_changed.connect(self._on_model_database_changed)
            except Exception:
                logger.warning("Exception in _set_model", exc_info=True)

            # Initial notification with current db handle
            try:
                self.database_changed.emit(self._model.db)
            except Exception:
                self.database_changed.emit(None)
        else:
            self.database_changed.emit(None)

    def _on_model_database_changed(self, _path: object) -> None:
        """
        Called when HybridPandasModel (via DatabaseModel) emits database_changed(Path).
        We convert this into a fresh db handle for listeners.
        """
        if self._model is not None:
            try:
                self.database_changed.emit(self._model.db)
            except Exception:
                self.database_changed.emit(None)
        else:
            self.database_changed.emit(None)

    # ------------------------------------------------------------------
    # UI signal helpers ------------------------------------------------
    def request_import(self) -> None:
        """Request that the importing flow is triggered by the view."""
        self.import_requested.emit()

    def request_new_database(self) -> None:
        """Request that the view opens the create database dialog."""
        self.new_requested.emit()

    def request_load_database(self) -> None:
        """Request that the view opens the load database dialog."""
        self.load_requested.emit()

    def request_save_database(self) -> None:
        """Request that the view opens the save database dialog."""
        self.save_requested.emit()

    def request_refresh_features(self) -> None:
        """Ask views to refresh the available features."""
        self.refresh_features_requested.emit()

    def systems_selection_changed(self, systems: list[str], datasets: list[str]) -> None:
        """Notify that the systems selection has changed."""
        self.systems_changed.emit(list(systems), list(datasets))

    def datasets_selection_changed(self, systems: list[str], datasets: list[str]) -> None:
        """Notify that the datasets selection has changed."""
        self.datasets_changed.emit(list(systems), list(datasets))

    def months_selection_changed(self, months: list[int]) -> None:
        """Notify that the set of selected months has changed."""
        self.months_changed.emit(list(months))

    def groups_selection_changed(self, group_ids: list[int]) -> None:
        """Notify that the set of selected groups has changed."""
        self.groups_changed.emit(list(group_ids))

    def date_range_updated(self, start, end) -> None:
        """Notify that the date range controls changed."""
        self.date_range_changed.emit(start, end)

    def preprocessing_parameter_changed(self, key: str, value: object) -> None:
        """Notify that a preprocessing parameter changed."""
        self.preprocessing_changed.emit(key, value)

    def tags_selection_changed(self, tags: list[str]) -> None:
        """Notify that the set of selected tags has changed."""
        self.tags_changed.emit(list(tags))

    # ------------------------------------------------------------------
    def close_database(self) -> None:
        """
        For the shared-model setup we do NOT close the underlying model here,
        we just tell listeners "treat DB as closed" if they care.
        """
        self.database_changed.emit(None)

    def refresh_database(self) -> None:
        """
        With the shared model, refreshing is handled via SettingsModel/database
        path changes; here we just re-emit the current handle.
        """
        self._on_model_database_changed(None)

    # ------------------------------------------------------------------
    @property
    def model(self) -> HybridPandasModel:
        if self._model is None:
            raise RuntimeError("HybridPandasModel is not initialised")
        return self._model

    def __getattr__(self, item: str) -> Any:
        """
        Delegate unknown attributes to the underlying HybridPandasModel so
        that views can call model methods directly on the view-model.
        """
        model = getattr(self, "_model", None)
        if model is None:
            raise AttributeError(item)
        return getattr(model, item)

    def load_import_preview(
        self,
        *,
        file_path: str,
        csv_delimiter: Optional[str],
        csv_decimal: Optional[str],
        csv_encoding: Optional[str],
        base_header_index: Optional[int],
        guess: bool = True,
        nrows: int = 8,
        ncolumns: int = 32,
    ) -> dict[str, object]:
        return build_import_preview_payload(
            file_path=file_path,
            csv_delimiter=csv_delimiter,
            csv_decimal=csv_decimal,
            csv_encoding=csv_encoding,
            base_header_index=base_header_index,
            guess=guess,
            nrows=nrows,
            ncolumns=ncolumns,
        )


__all__ = ["DataViewModel"]

