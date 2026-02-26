
from __future__ import annotations
from dataclasses import replace
from typing import Optional, Sequence, TYPE_CHECKING

import pandas as pd
from PySide6.QtCore import QObject, Signal

from backend.services.statistics_service import (
    StatisticsResult,
    StatisticsService,
    available_statistics,
)

import logging
logger = logging.getLogger(__name__)

from ...models.hybrid_pandas_model import FeatureSelection, HybridPandasModel
from ...threading.runner import run_in_thread

if TYPE_CHECKING:
    from backend.data_db import Database


class StatisticsViewModel(QObject):
    """Qt-friendly wrapper for statistics operations used by the tab UI."""

    # Emitted with the current Database handle (or None) when DB changes
    database_changed = Signal(object)

    status_changed = Signal(str)
    preview_updated = Signal(object)
    mode_changed = Signal(str, object)  # (mode, group_column)
    group_columns_changed = Signal(object)
    statistics_failed = Signal(str)
    statistics_warning = Signal(str)
    save_finished = Signal(int)
    save_failed = Signal(str)
    run_enabled_changed = Signal(bool)
    save_enabled_changed = Signal(bool)

    def __init__(self, data_model: HybridPandasModel, parent=None):
        """
        `data_model` is a HybridPandasModel that already manages the active
        database (via DatabaseModel + SettingsModel). We just listen to its
        database_changed signal and reuse its `db` handle.
        """
        super().__init__(parent)

        self._data_model: HybridPandasModel = data_model
        self._service: Optional[StatisticsService] = None
        self._current_result: Optional[StatisticsResult] = None
        self._compute_running = False
        self._save_running = False

        # React to DB path changes from the shared HybridPandasModel
        self._data_model.database_changed.connect(self._on_database_changed)
        self._data_model.groups_changed.connect(self._on_groups_changed)

        # Initialise from current state
        self._on_database_changed(self._data_model.path)

    # ------------------------------------------------------------------
    @property
    def data_model(self) -> HybridPandasModel:
        return self._data_model

    # ------------------------------------------------------------------
    def _close_database(self) -> None:
        """Internal reset of DB-related state (but keep the HybridPandasModel)."""
        self._service = None
        self._current_result = None

        self.preview_updated.emit(pd.DataFrame())
        self.status_changed.emit("Database closed. Gather statistics to preview results.")
        self.run_enabled_changed.emit(False)
        self.save_enabled_changed.emit(False)

    def close_database(self) -> None:
        """Release any held database-related resources (service/result)."""
        self._close_database()
        self.database_changed.emit(None)

    def _on_database_changed(self, _path) -> None:
        """
        Called whenever the underlying HybridPandasModel's DatabaseModel
        emits database_changed(Path). We rebuild the StatisticsService
        against the new db handle.
        """
        self._close_database()

        try:
            # HybridPandasModel exposes the shared backend.data_db.Database via .db
            db = self._data_model.db
        except Exception:
            self._service = None
            self._current_result = None
            self.database_changed.emit(None)
            self.preview_updated.emit(pd.DataFrame())
            self.status_changed.emit("Failed to open database.")
            self.run_enabled_changed.emit(False)
            self.save_enabled_changed.emit(False)
            return

        if db is None:
            self.database_changed.emit(None)
            self.preview_updated.emit(pd.DataFrame())
            self.status_changed.emit("Failed to open database.")
            self.run_enabled_changed.emit(False)
            self.save_enabled_changed.emit(False)
            return

        # Fresh service bound to the current DB
        self._service = StatisticsService(db)
        self._current_result = None

        self.database_changed.emit(db)
        self._emit_group_columns_changed()
        self.preview_updated.emit(pd.DataFrame())
        self.status_changed.emit("Database changed. Gather statistics to preview results.")
        self.run_enabled_changed.emit(True)
        self.save_enabled_changed.emit(False)

    def _on_groups_changed(self) -> None:
        self._emit_group_columns_changed()

    # ------------------------------------------------------------------
    def available_statistics(self) -> list[tuple[str, str]]:
        return available_statistics()

    def available_group_columns(self) -> list[tuple[str, str]]:
        """Return available columns for grouping, including database group kinds."""
        base_columns = [
            ("system", "System"),
            ("Dataset", "Dataset"),
        ]
        # Add group kinds from database
        db = self.database()
        if db is not None:
            try:
                existing_keys = {col[0] for col in base_columns}
                kinds = db.list_group_kinds()
                for kind in kinds:
                    if kind and kind not in existing_keys:
                        # Prefix with 'group:' to distinguish from built-in columns
                        base_columns.append((f"group:{kind}", f"Group: {kind}"))
            except Exception:
                logger.warning("Exception in available_group_columns", exc_info=True)
        return base_columns

    def _emit_group_columns_changed(self) -> None:
        try:
            self.group_columns_changed.emit(self.available_group_columns())
        except Exception:
            logger.warning("Exception in _emit_group_columns_changed", exc_info=True)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def compute(
        self,
        *,
        data_frame: pd.DataFrame,
        feature_payloads: Sequence[dict],
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        group_ids: Optional[Sequence[int]] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        months: Optional[Sequence[int]] = None,
        statistics: Sequence[str] = (),
        mode: str = "time",
        group_column: Optional[str] = None,
        preprocessing: Optional[dict] = None,
    ) -> StatisticsResult:
        if not self._service:
            raise RuntimeError("Database is not initialised")
        frame, selections, params = self._build_preprocessed_frame(
            data_frame=data_frame,
            feature_payloads=feature_payloads,
            systems=systems,
            datasets=datasets,
            group_ids=group_ids,
            start=start,
            end=end,
            preprocessing=preprocessing,
        )
        if frame is None:
            raise ValueError("Select at least one feature")

        long_df = self._wide_to_long_statistics_frame(
            frame,
            selections,
            systems=systems,
            datasets=datasets,
        )
        return self._service.compute(
            frame=long_df,
            months=months,
            statistics=statistics,
            mode=mode,
            group_column=group_column,
            preprocessing={
                "timestep": params.get("timestep"),
                "stats_period": params.get("stats_period"),
                "separate_timeframes": (
                    True
                    if preprocessing is None
                    else bool(preprocessing.get("separate_timeframes", True))
                ),
            },
        )

    def save(self, result: StatisticsResult) -> int:
        if not self._service:
            raise RuntimeError("Database is not initialised")
        return self._service.save(result)

    # ------------------------------------------------------------------
    def database(self) -> Optional[Database]:
        """Return the current backend.data_db.Database handle via the data model."""
        try:
            return self._data_model.db
        except Exception:
            return None

    def has_savable_result(self) -> bool:
        return bool(
            self._current_result
            and self._current_result.preview is not None
            and not self._current_result.preview.empty
        )

    # ------------------------------------------------------------------
    def run_statistics(
        self,
        *,
        data_frame: pd.DataFrame,
        feature_payloads: Sequence[dict],
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        group_ids: Optional[Sequence[int]] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        months: Optional[Sequence[int]] = None,
        statistics: Sequence[str] = (),
        mode: str = "time",
        group_column: Optional[str] = None,
        preprocessing: Optional[dict] = None,
    ) -> None:
        if self._compute_running:
            return
        self.run_enabled_changed.emit(False)
        self.save_enabled_changed.emit(False)
        self.status_changed.emit("Gathering statistics…")
        self._compute_running = True
        run_in_thread(
            self.compute,
            on_result=self._on_compute_finished,
            on_error=self._on_compute_error,
            data_frame=data_frame,
            feature_payloads=feature_payloads,
            systems=systems,
            datasets=datasets,
            group_ids=group_ids,
            start=start,
            end=end,
            months=months,
            statistics=statistics,
            mode=mode,
            group_column=group_column,
            preprocessing=preprocessing,
            owner=self,
            key="statistics_compute",
        )

    def _build_preprocessed_frame(
        self,
        *,
        data_frame: pd.DataFrame,
        feature_payloads: Sequence[dict],
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        group_ids: Optional[Sequence[int]] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        preprocessing: Optional[dict] = None,
    ) -> tuple[Optional[pd.DataFrame], list[FeatureSelection], dict]:
        if not feature_payloads:
            return None, [], {}

        selections = [FeatureSelection.from_payload(payload) for payload in feature_payloads]
        if not selections:
            return None, [], {}

        params = dict(preprocessing or {})
        params.setdefault("timestep", "auto")
        params.setdefault("fill", "none")
        params.setdefault("moving_average", None)
        params.setdefault("agg", params.get("agg", "avg"))
        params.pop("target_points", None)
        # Stats period is used only by statistics aggregation
        params.pop("stats_period", None)
        # Timeframe grouping option belongs only to statistics aggregation.
        params.pop("separate_timeframes", None)

        if isinstance(data_frame, pd.DataFrame):
            return data_frame.copy(), selections, params

        raise RuntimeError(
            "StatisticsViewModel requires a pre-fetched DataFrame from DataSelectorWidget."
        )

    @staticmethod
    def _wide_to_long_statistics_frame(
        frame: pd.DataFrame,
        features: Sequence[FeatureSelection],
        *,
        systems: Optional[Sequence[str]],
        datasets: Optional[Sequence[str]],
    ) -> pd.DataFrame:
        if frame is None or frame.empty or "t" not in frame.columns:
            return pd.DataFrame(
                columns=[
                    "t",
                    "v",
                    "feature_id",
                    "feature_label",
                    "system",
                    "Dataset",
                    "base_name",
                    "source",
                    "unit",
                    "type",
                    "import_id",
                ]
            )

        value_cols = [c for c in frame.columns if c != "t"]
        if not value_cols:
            return pd.DataFrame(
                columns=[
                    "t",
                    "v",
                    "feature_id",
                    "feature_label",
                    "system",
                    "Dataset",
                    "base_name",
                    "source",
                    "unit",
                    "type",
                    "import_id",
                ]
            )

        system_value = systems[0] if systems and len(systems) == 1 else None
        dataset_value = datasets[0] if datasets and len(datasets) == 1 else None

        rows: list[pd.DataFrame] = []
        for idx, col in enumerate(value_cols):
            sel = features[idx] if idx < len(features) else FeatureSelection(label=str(col), base_name=str(col))
            part = pd.DataFrame(
                {
                    "t": pd.to_datetime(frame["t"], errors="coerce"),
                    "v": pd.to_numeric(frame[col], errors="coerce"),
                    "feature_id": sel.feature_id,
                    "feature_label": (sel.label or sel.base_name or str(col)),
                    "system": system_value,
                    "Dataset": dataset_value,
                    "base_name": (sel.base_name or str(col)),
                    "source": sel.source,
                    "unit": sel.unit,
                    "type": sel.type,
                    "import_id": pd.NA,
                }
            )
            rows.append(part)
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        out["t"] = pd.to_datetime(out["t"], errors="coerce")
        out = out.dropna(subset=["t"])
        return out

    def _on_compute_finished(self, result: StatisticsResult) -> None:
        self._compute_running = False
        self.run_enabled_changed.emit(True)
        self._current_result = result
        preview = result.preview if result is not None else pd.DataFrame()
        if preview is None:
            preview = pd.DataFrame()
        
        # Emit mode info for chart rendering
        mode = result.mode if result else "time"
        group_column = result.group_column if result else None
        self.mode_changed.emit(mode, group_column)
        
        self.preview_updated.emit(preview)
        for warning in result.warnings:
            text = str(warning or "").strip()
            if text:
                self.statistics_warning.emit(text)
        row_count = len(preview.index) if not preview.empty else 0
        if row_count:
            self.status_changed.emit(f"Preview ready ({row_count} rows).")
            self.save_enabled_changed.emit(True)
        else:
            self.status_changed.emit("No rows produced for the selected inputs.")
            self.save_enabled_changed.emit(False)

    def _on_compute_error(self, message: str) -> None:
        self._compute_running = False
        self.run_enabled_changed.emit(True)
        self.save_enabled_changed.emit(self.has_savable_result())
        self.status_changed.emit("Statistics failed.")
        self.statistics_failed.emit(message)

    # ------------------------------------------------------------------
    def save_current_result(self, *, preview_override: Optional[pd.DataFrame] = None) -> None:
        if self._save_running:
            return
        if not self.has_savable_result():
            self.save_failed.emit("Gather statistics and preview the results before saving.")
            return
        result_to_save = self._current_result
        if (
            result_to_save is not None
            and preview_override is not None
            and isinstance(preview_override, pd.DataFrame)
        ):
            result_to_save = replace(result_to_save, preview=preview_override.copy())
        self.run_enabled_changed.emit(False)
        self.save_enabled_changed.emit(False)
        self.status_changed.emit("Saving statistics…")
        self._save_running = True
        run_in_thread(
            self.save,
            on_result=self._on_save_finished,
            on_error=self._on_save_error,
            result=result_to_save,
            owner=self,
            key="statistics_save",
        )

    def _on_save_finished(self, inserted: int) -> None:
        self._save_running = False
        self.run_enabled_changed.emit(True)
        self.save_enabled_changed.emit(self.has_savable_result())
        self.status_changed.emit(f"Saved {inserted} measurements to the database.")
        try:
            self._data_model.notify_features_changed()
        except Exception:
            logger.warning("Exception in _on_save_finished", exc_info=True)
        self.save_finished.emit(inserted)

    def _on_save_error(self, message: str) -> None:
        self._save_running = False
        self.run_enabled_changed.emit(True)
        self.save_enabled_changed.emit(self.has_savable_result())
        self.status_changed.emit("Saving failed.")
        self.save_failed.emit(message)

    def close(self) -> None:
        self._close_database()
