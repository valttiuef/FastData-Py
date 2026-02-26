# @ai(gpt-5, codex, refactor, 2026-02-26)
from typing import Optional

import math
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (


    QGroupBox,
    QLabel,
    QMenu,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import logging
logger = logging.getLogger(__name__)
from ...localization import tr

from backend.services.forecasting_service import ForecastSummary, ForecastRunResult
from frontend.charts import ProgressChart, TimeSeriesChart, ScatterChart
from ...widgets.panel import Panel
from ...widgets.fast_table import FastTable



class ForecastingResultsPanel(Panel):
    """Display forecasting run summaries, metrics, and charts."""

    details_requested = Signal()
    save_requested = Signal()
    delete_requested = Signal()

    _COLUMN_DEFINITIONS: list[tuple[str, str]] = [
        ("model", "Model"),
        ("feature", "Feature"),
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("mape", "MAPE (%)"),
    ]

    def __init__(self, parent: Optional[QWidget] = None, *, view_model=None) -> None:
        super().__init__("", parent=parent)
        layout = self.content_layout()

        self.status_label = QLabel(tr("Select features and run a forecasting experiment."), self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.main_splitter = QSplitter(Qt.Orientation.Vertical, self)
        layout.addWidget(self.main_splitter, 1)

        self.top_splitter = QSplitter(Qt.Orientation.Horizontal, self.main_splitter)
        self.main_splitter.addWidget(self.top_splitter)

        runs_box = QGroupBox(tr("Forecasting runs"), self)
        runs_layout = QVBoxLayout(runs_box)
        self.runs_table = FastTable(runs_box, select="rows", single_selection=False, editable=False)
        self._runs_display_columns = [tr(label) for _key, label in self._COLUMN_DEFINITIONS]
        self._run_key_column = "__run_key"
        self.runs_table.set_dataframe(
            pd.DataFrame(columns=self._runs_display_columns + [self._run_key_column]),
            include_index=False,
        )
        self.runs_table.hideColumn(len(self._runs_display_columns))
        header = self.runs_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionsClickable(True)
        runs_layout.addWidget(self.runs_table)
        self.top_splitter.addWidget(runs_box)

        self.progress_chart = ProgressChart(self)
        self.top_splitter.addWidget(self.progress_chart)
        
        # Set collapsible state AFTER adding widgets
        self.top_splitter.setCollapsible(0, False)
        self.top_splitter.setCollapsible(1, False)

        self.bottom_splitter = QSplitter(Qt.Orientation.Horizontal, self.main_splitter)
        self.main_splitter.addWidget(self.bottom_splitter)

        self.forecast_chart = TimeSeriesChart(tr("Forecast vs Actual"), parent=self)
        self.bottom_splitter.addWidget(self.forecast_chart)

        self.residual_chart = ScatterChart(parent=self)
        self.residual_chart.set_axis_labels(tr("Actual"), tr("Forecast"))
        self.bottom_splitter.addWidget(self.residual_chart)
        
        # Set collapsible state AFTER adding widgets
        self.bottom_splitter.setCollapsible(0, False)
        self.bottom_splitter.setCollapsible(1, False)

        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        
        # Set stretch factors to ensure equal distribution
        self.top_splitter.setStretchFactor(0, 1)
        self.top_splitter.setStretchFactor(1, 1)
        self.bottom_splitter.setStretchFactor(0, 1)
        self.bottom_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 1)
        
        # Set initial sizes to ensure even 2x2 layout (will be adjusted at runtime)
        # Use reasonable pixel values based on typical window sizes
        self.top_splitter.setSizes([400, 400])
        self.bottom_splitter.setSizes([400, 400])
        self.main_splitter.setSizes([300, 300])

        self._summary: Optional[ForecastSummary] = None
        self._runs: dict[str, ForecastRunResult] = {}
        self._run_order: list[str] = []
        self._expected_runs: Optional[int] = None
        self._current_selection_key: Optional[str] = None
        self._run_contexts: dict[str, dict[str, object]] = {}
        self._active_context: Optional[dict[str, object]] = None
        self.runs_table.selectionChangedInstant.connect(self._on_selection_changed)
        self.runs_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.runs_table.customContextMenuRequested.connect(self._show_context_menu)
        self._view_model = None
        if view_model is not None:
            self.set_view_model(view_model)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._summary = None
        self._runs.clear()
        self._run_order.clear()
        self._expected_runs = None
        self._current_selection_key = None
        self._run_contexts.clear()
        self._active_context = None
        self._refresh_table()
        self.progress_chart.clear()
        self.forecast_chart.clear()
        self.residual_chart.clear()
        self.status_label.setText(tr("Select features and run a forecasting experiment."))

    def prepare_for_run(self, expected_runs: Optional[int] = None) -> None:
        if expected_runs and expected_runs > 0:
            self._expected_runs = expected_runs
            self.status_label.setText(
                tr("Preparing to train {count} forecasting pipelines…").format(count=expected_runs)
            )
        else:
            self.status_label.setText(tr("Preparing forecasting pipelines…"))

    def set_view_model(self, view_model) -> None:
        if self._view_model is view_model:
            return
        if self._view_model is not None:
            try:
                self._view_model.run_started.disconnect(self._on_run_started)
                self._view_model.run_partial.disconnect(self.update_run)
                self._view_model.run_finished.disconnect(self.set_summary)
                self._view_model.run_failed.disconnect(self.show_failure)
            except Exception:
                logger.warning("Exception in set_view_model", exc_info=True)
        self._view_model = view_model
        if self._view_model is None:
            return
        try:
            self._view_model.run_started.connect(self._on_run_started)
            self._view_model.run_partial.connect(self.update_run)
            self._view_model.run_finished.connect(self.set_summary)
            self._view_model.run_failed.connect(self.show_failure)
        except Exception:
            logger.warning("Exception in set_view_model", exc_info=True)

    def show_failure(self, message: str) -> None:
        text = message.strip() if message else tr("Forecasting failed.")
        self.status_label.setText(text)

    def set_summary(self, summary: Optional[ForecastSummary]) -> None:
        if summary is None or not summary.runs:
            self.clear()
            return

        selected_key = self._current_selection_key
        self._summary = summary
        self._expected_runs = len(summary.runs)

        for run in summary.runs:
            self.update_run(run, select_if_new=False)

        if selected_key and selected_key in self._runs:
            self._select_row_for_key(selected_key)
        elif not self._has_selection() and summary.runs:
            self._select_row_for_key(summary.runs[0].key)

        if summary.runs:
            latest_progress = summary.runs[-1].progress_frame
            self.progress_chart.set_dataframe(latest_progress if latest_progress is not None else pd.DataFrame())

        self.status_label.setText(tr("Forecasting runs completed. Select a run to inspect details."))

    # ------------------------------------------------------------------
    def prepare_progress_frame(self, run: ForecastRunResult) -> pd.DataFrame:
        progress = run.progress_frame if run.progress_frame is not None else pd.DataFrame()
        return progress

    def update_run(self, run: Optional[ForecastRunResult], *, select_if_new: bool = True) -> None:
        if run is None:
            return
        self._runs[run.key] = run
        if run.key not in self._run_order:
            self._run_order.append(run.key)
        if run.metadata and run.key not in self._run_contexts:
            self._run_contexts[run.key] = dict(run.metadata)
        if self._active_context is not None and run.key not in self._run_contexts:
            context = dict(self._active_context)
            model_params = context.get("model_params") or {}
            context["model_params"] = dict(model_params.get(run.model_key, {}))
            context["hyperparameters"] = {
                "model_params": context["model_params"],
            }
            context["model_label"] = run.model_label
            context["feature_label"] = run.feature_label
            features = context.get("features") or []
            matched = None
            for payload in features:
                try:
                    if str(payload.get("feature_id")) == str(run.feature_key):
                        matched = payload
                        break
                except Exception:
                    continue
            context["feature"] = dict(matched) if matched else dict(context.get("feature") or {})
            self._run_contexts[run.key] = context
        self._refresh_table()
        if select_if_new and not self._has_selection():
            self._select_row_for_key(run.key)

        progress = self.prepare_progress_frame(run)
        self.progress_chart.set_dataframe(progress)

    # ------------------------------------------------------------------
    def _refresh_table(self) -> None:
        rows: list[dict[str, object]] = []
        for key in self._run_order:
            run = self._runs.get(key)
            if run is None:
                continue
            values = self._row_values_for_run(run)
            out_row: dict[str, object] = {}
            for field_key, label in self._COLUMN_DEFINITIONS:
                value = values.get(field_key)
                col_label = tr(label)
                if field_key in {"rmse", "mae", "mape"}:
                    numeric_value = self._to_float(value)
                    out_row[col_label] = "-" if numeric_value is None else f"{numeric_value:.4f}"
                else:
                    out_row[col_label] = str(value)
            out_row[self._run_key_column] = str(key)
            rows.append(out_row)
        frame = pd.DataFrame(rows, columns=self._runs_display_columns + [self._run_key_column])
        self.runs_table.set_dataframe(frame, include_index=False)
        self.runs_table.hideColumn(len(self._runs_display_columns))

    @staticmethod
    def _to_float(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isfinite(numeric):
            return numeric
        return None

    def _select_row_for_key(self, key: str) -> None:
        row = self._find_row_for_key(key)
        if row is None:
            return
        self.runs_table.selectRow(row)

    def _has_selection(self) -> bool:
        selection = self.runs_table.selectionModel()
        return bool(selection and selection.selectedRows())

    def set_run_context(self, context: Optional[dict[str, object]]) -> None:
        self._active_context = dict(context) if context else None

    def context_for_key(self, key: str) -> dict[str, object]:
        return dict(self._run_contexts.get(key, {}))

    def selected_runs(self) -> list[ForecastRunResult]:
        keys = self._selected_keys()
        return [self._runs[key] for key in keys if key in self._runs]

    def mark_run_saved(self, key: str, model_id: int, metadata: Optional[dict[str, object]] = None) -> None:
        run = self._runs.get(key)
        if run is None:
            return
        try:
            from dataclasses import replace
            updated = replace(run, model_id=model_id, metadata=metadata or run.metadata)
        except Exception:
            return
        self._runs[key] = updated
        if metadata:
            self._run_contexts[key] = dict(metadata)
        if self._summary:
            self._summary = ForecastSummary(
                runs=[self._runs[k] for k in self._run_order if k in self._runs]
            )

    def remove_runs(self, keys: list[str]) -> None:
        if not keys:
            return
        for key in keys:
            self._runs.pop(key, None)
            self._run_contexts.pop(key, None)
            if key in self._run_order:
                self._run_order.remove(key)
        self._refresh_table()
        self._summary = ForecastSummary(
            runs=[self._runs[k] for k in self._run_order if k in self._runs]
        )
        if self._runs:
            self.status_label.setText(tr("Forecasting runs updated. Select a run to inspect details."))
        else:
            self.status_label.setText(tr("Select features and run a forecasting experiment."))
            self.progress_chart.clear()
            self.forecast_chart.clear()
            self.residual_chart.clear()

    def _selected_keys(self) -> list[str]:
        keys: list[str] = []
        selection = self.runs_table.selectionModel()
        if selection is None:
            return keys
        model = self.runs_table.model()
        if model is None:
            return keys
        key_col = len(self._runs_display_columns)
        for index in selection.selectedRows():
            key = model.index(index.row(), key_col).data(Qt.ItemDataRole.UserRole)
            if key and key not in keys:
                keys.append(key)
        return keys

    def _find_row_for_key(self, key: str) -> Optional[int]:
        model = self.runs_table.model()
        if model is None:
            return None
        key_col = len(self._runs_display_columns)
        for row in range(model.rowCount()):
            if model.index(row, key_col).data(Qt.ItemDataRole.UserRole) == key:
                return row
        return None

    def _row_values_for_run(self, run: ForecastRunResult) -> dict[str, object]:
        metrics = run.metrics or {}
        return {
            "model": run.model_label,
            "feature": run.feature_label,
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "mape": metrics.get("mape"),
        }

    def _on_run_started(self, expected_runs) -> None:
        try:
            expected = int(expected_runs) if expected_runs is not None else None
        except Exception:
            expected = None
        self.prepare_for_run(expected)

    # ------------------------------------------------------------------
    def _on_selection_changed(self) -> None:
        keys = self._selected_keys()
        if not keys:
            return
        key = keys[0]
        self._current_selection_key = key
        self._apply_run(self._runs.get(key))

    def _apply_run(self, run: Optional[ForecastRunResult]) -> None:
        if run is None:
            self.forecast_chart.clear()
            self.residual_chart.clear()
            return

        forecast_frame = run.forecast_frame if run.forecast_frame is not None else pd.DataFrame()
        self.forecast_chart.set_dataframe(forecast_frame)

        # Create scatter chart with actual vs forecast
        # Support both "Testing" period (single train/test split) and "Validation" period (cross-validation)
        test_data = pd.DataFrame()
        if not forecast_frame.empty and {"Actual", "Forecast"}.issubset(forecast_frame.columns):
            # First try to find Validation period (from cross-validation with sliding/expanding windows)
            test_data = forecast_frame[
                (forecast_frame["Period"] == "Validation") & 
                (forecast_frame["Actual"].notna()) & 
                (forecast_frame["Forecast"].notna())
            ].copy()
            
            # If no validation data, try Testing period (from single train/test split)
            if test_data.empty:
                test_data = forecast_frame[
                    (forecast_frame["Period"] == "Testing") & 
                    (forecast_frame["Actual"].notna()) & 
                    (forecast_frame["Forecast"].notna())
                ].copy()
        
        self.residual_chart.set_points(
            test_data if not test_data.empty else None,
            x_column="Actual",
            y_column="Forecast",
            group_column=None
        )

    def _show_context_menu(self, pos) -> None:
        model = self.runs_table.model()
        if model is None or model.rowCount() <= 0:
            return
        menu = QMenu(self)
        details_action = menu.addAction(tr("Details..."))
        save_action = menu.addAction(tr("Save selected"))
        delete_action = menu.addAction(tr("Delete selected"))
        action = menu.exec_(self.runs_table.viewport().mapToGlobal(pos))
        if action == details_action:
            self.details_requested.emit()
        elif action == save_action:
            self.save_requested.emit()
        elif action == delete_action:
            self.delete_requested.emit()


__all__ = ["ForecastingResultsPanel"]

