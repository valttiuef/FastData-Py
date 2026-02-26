
from __future__ import annotations
# @ai(gpt-5, codex, refactor, 2026-02-26)
import math
from typing import Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QSignalBlocker, Signal
from PySide6.QtGui import QResizeEvent, QShowEvent
from PySide6.QtWidgets import (


    QGroupBox,
    QLabel,
    QMenu,
    QSplitter,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import logging
logger = logging.getLogger(__name__)
from ...localization import tr

from backend.services.modeling_shared import display_name
from backend.services.regression_service import RegressionSummary, RegressionRunResult
from frontend.charts import ScatterChart, TimeSeriesChart
from ...widgets.panel import Panel
from ...widgets.fast_table import FastTable


class RegressionResultsPanel(Panel):
    """Display regression run summaries, metrics, and charts."""

    details_requested = Signal()
    save_requested = Signal()
    delete_requested = Signal()
    export_available_changed = Signal(bool)

    _COLUMN_DEFINITIONS: list[tuple[str, str]] = [
        ("trained_at", "Trained at"),
        ("selector", "Selection method"),
        ("model", "Model"),
        ("target", "Target"),
        ("rmse_test", "RMSE (test)"),
        ("cv_rmse_mean", "RMSE (cv)"),
        ("rmse_train", "RMSE (train)"),
        ("rows_total", "Rows (total)"),
        ("rows_train", "Rows (train)"),
        ("rows_test", "Rows (test)"),
        ("inputs_selected", "Inputs (after selection)"),
        ("inputs", "Inputs"),
        ("r2_test", "R² (test)"),
        ("r2_train", "R² (train)"),
        ("cv_r2_mean", "CV R² mean"),
        ("cv_r2_std", "CV R² std"),
        ("cv_rmse_std", "CV RMSE std"),
        ("reducer", "Dimensionality reduction"),
    ]
    _TEXT_COLUMNS = {"trained_at", "selector", "model", "reducer", "target"}
    _INT_COLUMNS = {"inputs", "inputs_selected", "rows_total", "rows_train", "rows_test"}

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("", parent=parent)
        self._splitters_initialized = False
        self._initial_table_layout_applied = False
        layout = self.content_layout()

        self.status_label = QLabel(tr("Select features and run a regression experiment."), self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.main_splitter = QSplitter(Qt.Orientation.Vertical, self)
        layout.addWidget(self.main_splitter, 1)

        self.top_splitter = QSplitter(Qt.Orientation.Horizontal, self.main_splitter)
        self.main_splitter.addWidget(self.top_splitter)

        runs_box = QGroupBox(tr("Regression runs"), self)
        runs_layout = QVBoxLayout(runs_box)
        self.runs_table = FastTable(runs_box, select="rows", single_selection=False, editable=False)
        self._runs_display_columns = [tr(label) for _key, label in self._COLUMN_DEFINITIONS]
        self._run_key_column = "__run_key"
        self._trained_sort_column = "__trained_sort"
        self.runs_table.set_dataframe(
            pd.DataFrame(columns=self._runs_display_columns + [self._run_key_column, self._trained_sort_column]),
            include_index=False,
        )
        self.runs_table.hideColumn(len(self._runs_display_columns))
        self.runs_table.hideColumn(len(self._runs_display_columns) + 1)
        self.runs_table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
        header = self.runs_table.horizontalHeader()
        # Allow interactive resizing while keeping the last column stretchable on window resize.
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(30)
        header.setSectionResizeMode(header.ResizeMode.Interactive)
        header.setSectionsClickable(True)
        self.runs_table.set_stretch_column(-1)
        runs_box.setMinimumWidth(0)
        self.runs_table.setMinimumWidth(0)
        runs_layout.addWidget(self.runs_table)
        self.top_splitter.addWidget(runs_box)

        predictions_box = QGroupBox(tr("Predictions"), self)
        predictions_layout = QVBoxLayout(predictions_box)
        self.predictions_table = FastTable(
            predictions_box,
            select="rows",
            single_selection=False,
            editable=False,
            initial_uniform_column_widths=True,
            initial_uniform_column_count=5,
            sorting_enabled=False,
        )
        self.predictions_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._predictions_columns = [tr("Date"), tr("Prediction"), tr("Reference"), tr("Error"), tr("Split")]
        self.predictions_table.set_dataframe(
            pd.DataFrame(columns=self._predictions_columns),
            include_index=False,
            float_format="{:.4f}",
        )
        pred_header = self.predictions_table.horizontalHeader()
        pred_header.setStretchLastSection(False)
        pred_header.setMinimumSectionSize(30)
        pred_header.setSectionResizeMode(pred_header.ResizeMode.Interactive)
        pred_header.setSectionsClickable(True)
        self.predictions_table.set_stretch_column(None)
        predictions_box.setMinimumWidth(0)
        self.predictions_table.setMinimumWidth(0)
        predictions_layout.addWidget(self.predictions_table)
        self.top_splitter.addWidget(predictions_box)
        
        # Set collapsible state AFTER adding widgets
        self.top_splitter.setCollapsible(0, False)
        self.top_splitter.setCollapsible(1, False)

        self.bottom_splitter = QSplitter(Qt.Orientation.Horizontal, self.main_splitter)
        self.main_splitter.addWidget(self.bottom_splitter)

        self.scatter_chart = ScatterChart(self)
        self.scatter_chart.set_axis_labels(tr("Actual"), tr("Predicted"))
        self._set_scatter_title(None)
        self.bottom_splitter.addWidget(self.scatter_chart)

        self.timeline_chart = TimeSeriesChart(tr("Prediction timeline"), parent=self)
        self.timeline_chart.set_max_series(32)
        self.bottom_splitter.addWidget(self.timeline_chart)
        
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

        self._summary: Optional[RegressionSummary] = None
        self._runs: dict[str, RegressionRunResult] = {}
        self._run_order: list[str] = []
        self._expected_runs: Optional[int] = None
        self._current_selection_key: Optional[str] = None
        self._run_contexts: dict[str, dict[str, object]] = {}
        self._active_context: Optional[dict[str, object]] = None
        self._export_available = False
        self._batch_loading = False
        self._timeline_cache: dict[str, pd.DataFrame] = {}
        self._predictions_cache: dict[str, pd.DataFrame] = {}

        self.runs_table.selectionChangedInstant.connect(self._on_selection_changed)
        self.runs_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.runs_table.customContextMenuRequested.connect(self._show_context_menu)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._summary = None
        self._runs.clear()
        self._run_order.clear()
        self._expected_runs = None
        self._current_selection_key = None
        self._run_contexts.clear()
        self._active_context = None
        self._refresh_runs_table()
        self.predictions_table.set_dataframe(pd.DataFrame(columns=self._predictions_columns), include_index=False)
        self.scatter_chart.clear()
        self._set_scatter_title(None)
        self.timeline_chart.clear()
        self.status_label.setText(tr("Select features and run a regression experiment."))
        self._set_export_available(False)
        self._batch_loading = False
        self._timeline_cache.clear()
        self._predictions_cache.clear()

    def begin_batch_update(self) -> None:
        if self._batch_loading:
            return
        self._batch_loading = True
        try:
            self.runs_table.setUpdatesEnabled(False)
        except Exception:
            logger.warning("Exception in begin_batch_update", exc_info=True)

    def end_batch_update(self) -> None:
        if not self._batch_loading:
            return
        try:
            self.runs_table.setUpdatesEnabled(True)
        except Exception:
            logger.warning("Exception in end_batch_update", exc_info=True)
        self._batch_loading = False
        self._refresh_runs_table()
        self.runs_table.resizeColumnsToContents()
        if self._runs:
            self.status_label.setText(tr("Regression runs updated. Select a run to inspect details."))
            self._set_export_available(True)
        else:
            self.status_label.setText(tr("Select features and run a regression experiment."))
            self._set_export_available(False)

    def prepare_for_run(self, expected_runs: Optional[int] = None) -> None:
        if expected_runs and expected_runs > 0:
            self._expected_runs = expected_runs
            self.status_label.setText(
                tr("Preparing to train {count} regression pipelines…").format(count=expected_runs)
            )
        else:
            self.status_label.setText(tr("Preparing regression pipelines…"))

    def show_failure(self, message: str) -> None:
        text = message.strip() if message else tr("Regression failed.")
        self.status_label.setText(text)

    def set_summary(self, summary: Optional[RegressionSummary]) -> None:
        if summary is None or not summary.runs:
            self.clear()
            return

        selected_key = self._current_selection_key
        self._summary = summary
        self._expected_runs = len(summary.runs)

        self.begin_batch_update()
        try:
            for run in summary.runs:
                self.update_run(run, select_if_new=False)
        finally:
            self.end_batch_update()

        if selected_key and selected_key in self._runs:
            self._select_row_for_key(selected_key)
        elif not self._has_selection() and summary.runs:
            self._select_row_for_key(summary.runs[0].key)

        if summary.runs:
            self._apply_run(self._runs.get(self._current_selection_key or summary.runs[0].key))

        self.status_label.setText(tr("Regression runs completed. Select a run to inspect details."))
        self._set_export_available(bool(self._runs))

    def all_runs(self) -> list[RegressionRunResult]:
        return [self._runs[key] for key in self._run_order if key in self._runs]

    def export_summary_frame(self, runs: Optional[list[RegressionRunResult]] = None) -> pd.DataFrame:
        selected_runs = runs if runs is not None else self.all_runs()
        rows: list[dict[str, object]] = []
        columns = [key for key, _ in self._COLUMN_DEFINITIONS]
        for run in selected_runs:
            run_key = self._normalize_key(getattr(run, "key", ""))
            ctx = self._run_contexts.get(run_key, {})
            values = self._row_values(run, ctx)
            _display, sort_value = self._trained_at_display_and_sort(run, ctx)
            row = {key: values.get(key) for key in columns}
            row["__trained_sort"] = sort_value
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows)
        frame = frame.sort_values(by="__trained_sort", ascending=False, kind="stable")
        frame = frame.drop(columns=["__trained_sort"])
        return frame[columns].reset_index(drop=True)

    def export_individual_frame(self, run: RegressionRunResult) -> pd.DataFrame:
        if run is None:
            return pd.DataFrame(columns=["datetime", "actual", "prediction", "split"])
        timeline = run.timeline_frame if run.timeline_frame is not None else pd.DataFrame()
        if timeline.empty:
            return pd.DataFrame(columns=["datetime", "actual", "prediction", "split"])

        actual_col = "Actual" if "Actual" in timeline.columns else None
        train_col = "Prediction (train)" if "Prediction (train)" in timeline.columns else None
        train_split_cols = [
            column
            for column in timeline.columns
            if isinstance(column, str)
            and column.startswith("Prediction (train split ")
            and column.endswith(")")
        ]
        split_col = "Split (train)" if "Split (train)" in timeline.columns else None
        test_col = "Prediction (test)" if "Prediction (test)" in timeline.columns else None
        if "t" not in timeline.columns or actual_col is None:
            return pd.DataFrame(columns=["datetime", "actual", "prediction", "split"])

        train_export_cols: list[str] = []
        # Avoid duplicate rows: either split-specific train traces or the single train trace.
        if train_split_cols:
            train_export_cols.extend(train_split_cols)
        elif train_col:
            train_export_cols.append(train_col)

        working = timeline[
            ["t", actual_col]
            + train_export_cols
            + ([split_col] if split_col else [])
            + ([test_col] if test_col else [])
        ].copy()
        working["t"] = pd.to_datetime(working["t"], errors="coerce")
        working = working.dropna(subset=["t"])
        if working.empty:
            return pd.DataFrame(columns=["datetime", "actual", "prediction", "split"])

        rows: list[dict[str, object]] = []

        for train_name in train_export_cols:
            train_rows = working.dropna(subset=[train_name])
            for _, row in train_rows.iterrows():
                split_text = "train"
                if train_name.startswith("Prediction (train split ") and train_name.endswith(")"):
                    split_text = train_name[len("Prediction (train split ") : -1].strip() or "train"
                else:
                    split_value = row.get(split_col) if split_col else None
                    split_text = str(split_value).strip() if split_value is not None else ""
                    if not split_text or split_text.lower() == "nan":
                        split_text = "train"
                rows.append(
                    {
                        "datetime": row.get("t"),
                        "actual": row.get(actual_col),
                        "prediction": row.get(train_name),
                        "split": split_text,
                    }
                )
        if test_col:
            test_rows = working.dropna(subset=[test_col])
            for _, row in test_rows.iterrows():
                rows.append(
                    {
                        "datetime": row.get("t"),
                        "actual": row.get(actual_col),
                        "prediction": row.get(test_col),
                        "split": "test",
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["datetime", "actual", "prediction", "split"])

        frame = pd.DataFrame(rows)
        frame = frame.sort_values(by=["datetime", "split"], ascending=[True, True]).reset_index(drop=True)
        return frame

    def run_label(self, run: RegressionRunResult) -> str:
        context = self._run_contexts.get(str(getattr(run, "key", "")), {})
        return self._export_run_label(run, context)

    def _export_run_label(self, run: RegressionRunResult, context: dict[str, object]) -> str:
        values = self._row_values(run, context)
        parts = []
        for key in ("model", "selector", "target", "trained_at"):
            value = values.get(key)
            if value:
                parts.append(str(value))
        if parts:
            return tr("Run {summary}").format(summary=" - ".join(parts))
        try:
            return tr("Run {key}").format(key=str(run.key))
        except Exception:
            return tr("Run")

    # ------------------------------------------------------------------
    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        # Initialize splitter sizes once, after all layouts are properly computed
        if not self._splitters_initialized:
            self._splitters_initialized = self._initialize_splitter_sizes()
        # Continue to rebalance on resize to maintain even distribution
        elif event.size().width() != event.oldSize().width() or event.size().height() != event.oldSize().height():
            self._rebalance_splitters()

    def showEvent(self, event: QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        # Ensure the splitters start balanced after the widget becomes visible
        if not self._splitters_initialized:
            self._splitters_initialized = self._initialize_splitter_sizes()
        if self._splitters_initialized:
            self._rebalance_splitters()
            self._apply_initial_table_layout_once()

    def _rebalance_splitters(self) -> None:
        """Rebalance splitter sizes to maintain 50/50 distribution on resize."""
        main_height = self.main_splitter.size().height()
        top_width = self.top_splitter.size().width()
        bottom_width = self.bottom_splitter.size().width()

        if main_height > 0:
            half_height = max(1, main_height // 2)
            self.main_splitter.setSizes([half_height, main_height - half_height])

        if top_width > 0:
            half_top = max(1, top_width // 2)
            self.top_splitter.setSizes([half_top, top_width - half_top])

        if bottom_width > 0:
            half_bottom = max(1, bottom_width // 2)
            self.bottom_splitter.setSizes([half_bottom, bottom_width - half_bottom])

    def _initialize_splitter_sizes(self) -> bool:
        main_height = self.main_splitter.size().height()
        top_width = self.top_splitter.size().width()
        bottom_width = self.bottom_splitter.size().width()

        if main_height <= 0 or top_width <= 0 or bottom_width <= 0:
            return False

        half_height = max(1, main_height // 2)
        self.main_splitter.setSizes([half_height, main_height - half_height])

        half_top = max(1, top_width // 2)
        self.top_splitter.setSizes([half_top, top_width - half_top])

        half_bottom = max(1, bottom_width // 2)
        self.bottom_splitter.setSizes([half_bottom, bottom_width - half_bottom])

        return True

    def _apply_initial_table_layout_once(self) -> None:
        if self._initial_table_layout_applied:
            return
        self._initial_table_layout_applied = True
        try:
            self.predictions_table.reapply_uniform_column_widths()
        except Exception:
            logger.warning("Exception in _apply_initial_table_layout_once", exc_info=True)

    # ------------------------------------------------------------------
    def update_run(self, run: RegressionRunResult, *, select_if_new: bool = True) -> None:
        if run is None:
            return
        try:
            run_key = str(run.key)
        except Exception:
            run_key = ""

        new_row = run_key not in self._run_order
        if new_row:
            self._run_order.append(run_key)

        self._runs[run_key] = run
        self._timeline_cache.pop(run_key, None)
        self._predictions_cache.pop(run_key, None)
        self._summary = RegressionSummary(
            runs=[self._runs[key] for key in self._run_order if key in self._runs]
        )
        metadata = getattr(run, "metadata", None)
        existing_context = self._run_contexts.get(run_key)
        context: Optional[dict[str, object]] = None
        if existing_context:
            context = dict(existing_context)
        elif self._active_context is not None:
            context = dict(self._active_context)

        if isinstance(metadata, dict) and metadata:
            if context is None:
                context = dict(metadata)
            else:
                context.update(dict(metadata))

        if context is not None:
            model_params = context.get("model_params") or {}
            selector_params = context.get("selector_params") or {}
            reducer = context.get("dimensionality_reduction") or context.get("pca") or {}
            selected_reducer_key = getattr(run, "reducer_key", None) or "none"
            selected_reducer_label = getattr(run, "reducer_label", None) or ""
            reducer_params = context.get("reducer_params") or {}
            context["model_params"] = dict(model_params.get(run.model_key, {}))
            context["selector_params"] = dict(selector_params.get(run.selector_key, {}))
            context["reducer_params"] = dict(reducer_params.get(selected_reducer_key, {}))
            context["hyperparameters"] = {
                "model_params": context["model_params"],
                "selector_params": context["selector_params"],
                "reducer_params": context["reducer_params"],
                "dimensionality_reduction": reducer,
                "pca": reducer,
            }
            context["dimensionality_reduction"] = dict(reducer) if isinstance(reducer, dict) else reducer
            context["pca"] = dict(reducer) if isinstance(reducer, dict) else reducer
            if isinstance(context.get("dimensionality_reduction"), dict):
                context["dimensionality_reduction"].setdefault("method", selected_reducer_key)
                context["dimensionality_reduction"].setdefault("label", selected_reducer_label)
            context["model_label"] = run.model_label
            context["selector_label"] = run.selector_label
            context["reducer_label"] = selected_reducer_label
            self._run_contexts[run_key] = context

        self._refresh_runs_table()

        if not self._batch_loading:
            trained = len(self._runs)
            if self._expected_runs:
                self.status_label.setText(
                    tr("Trained {trained} of {total} regression pipelines.").format(
                        trained=trained, total=self._expected_runs
                    )
                )
            else:
                suffix = tr("s") if trained != 1 else ""
                self.status_label.setText(
                    tr("Trained {trained} regression pipeline{suffix}.").format(
                        trained=trained, suffix=suffix
                    )
                )

        # Sorting may change row position; always re-check by key.
        current_row = self._find_row_for_key(run_key)

        if (
            current_row is not None
            and new_row
            and select_if_new
            and not self._has_selection()
        ):
            self._select_row_for_key(run_key)
        elif self._current_selection_key == run_key:
            self._select_row_for_key(run_key)

        if not self._batch_loading:
            self.runs_table.resizeColumnsToContents()
            self._rebalance_splitters()
            self._set_export_available(bool(self._runs))

    # ------------------------------------------------------------------
    def _has_selection(self) -> bool:
        selection = self.runs_table.selectionModel()
        return bool(selection and selection.selectedRows())

    def _set_export_available(self, available: bool) -> None:
        available = bool(available)
        if self._export_available == available:
            return
        self._export_available = available
        self.export_available_changed.emit(available)

    def set_run_context(self, context: Optional[dict[str, object]]) -> None:
        self._active_context = dict(context) if context else None

    @staticmethod
    def _normalize_key(key: object) -> str:
        try:
            return str(key)
        except Exception:
            return ""

    def context_for_key(self, key: str) -> dict[str, object]:
        run_key = self._normalize_key(key)
        return dict(self._run_contexts.get(run_key, {}))

    def selected_runs(self) -> list[RegressionRunResult]:
        keys = self._selected_keys()
        return [self._runs[key] for key in keys if key in self._runs]

    def mark_run_saved(self, key: str, model_id: int, metadata: Optional[dict[str, object]] = None) -> None:
        run_key = self._normalize_key(key)
        run = self._runs.get(run_key)
        if run is None:
            return
        try:
            from dataclasses import replace
            updated = replace(run, model_id=model_id, metadata=metadata or run.metadata)
        except Exception:
            return
        self._runs[run_key] = updated
        if metadata:
            self._run_contexts[run_key] = dict(metadata)
        if self._summary:
            self._summary = RegressionSummary(
                runs=[self._runs[k] for k in self._run_order if k in self._runs]
            )

    def remove_runs(self, keys: list[str]) -> None:
        if not keys:
            return
        for key in keys:
            run_key = self._normalize_key(key)
            self._runs.pop(run_key, None)
            self._run_contexts.pop(run_key, None)
            if run_key in self._run_order:
                self._run_order.remove(run_key)
        self._refresh_runs_table()
        self._summary = RegressionSummary(
            runs=[self._runs[k] for k in self._run_order if k in self._runs]
        )
        if self._runs:
            self.status_label.setText(tr("Regression runs updated. Select a run to inspect details."))
            self._set_export_available(True)
        else:
            self.status_label.setText(tr("Select features and run a regression experiment."))
            self.predictions_table.set_dataframe(pd.DataFrame(columns=self._predictions_columns), include_index=False)
            self.scatter_chart.clear()
            self.timeline_chart.clear()
            self._set_export_available(False)

    def _selected_keys(self) -> list[str]:
        keys: list[str] = []
        selection = self.runs_table.selectionModel()
        if selection is None:
            return keys
        model = self.runs_table.model()
        if model is None:
            return keys
        key_col = self._runs_key_column_index()
        if key_col is None:
            return keys
        for index in selection.selectedRows():
            key = model.index(index.row(), key_col).data(Qt.ItemDataRole.UserRole)
            if key and key not in keys:
                keys.append(key)
        return keys

    def _find_row_for_key(self, key: str) -> Optional[int]:
        model = self.runs_table.model()
        if model is None:
            return None
        key_col = self._runs_key_column_index()
        if key_col is None:
            return None
        for row in range(model.rowCount()):
            if model.index(row, key_col).data(Qt.ItemDataRole.UserRole) == key:
                return row
        return None

    def _select_row_for_key(self, key: str) -> None:
        row = self._find_row_for_key(key)
        if row is None:
            return
        blocker = QSignalBlocker(self.runs_table)
        try:
            self.runs_table.selectRow(row)
        finally:
            del blocker
        self._current_selection_key = key
        self._apply_run(self._runs.get(key))

    def _runs_key_column_index(self) -> Optional[int]:
        model = self.runs_table.model()
        if model is None:
            return None
        key_column_name = self._run_key_column
        for col in range(model.columnCount()):
            if str(model.headerData(col, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole) or "") == key_column_name:
                return col
        return None

    def _refresh_runs_table(self) -> None:
        rows: list[dict[str, object]] = []
        for run_key in self._run_order:
            run = self._runs.get(run_key)
            if run is None:
                continue
            context = self._run_contexts.get(str(run.key), {})
            values = self._row_values(run, context)
            row: dict[str, object] = {}
            _trained_display, sort_value = self._trained_at_display_and_sort(run, context)
            for key, label in self._COLUMN_DEFINITIONS:
                col_label = tr(label)
                if key in self._TEXT_COLUMNS:
                    row[col_label] = str(values.get(key, "-") or "-")
                elif key in self._INT_COLUMNS:
                    numeric_value = self._to_float(values.get(key))
                    row[col_label] = "-" if numeric_value is None else f"{int(round(numeric_value))}"
                else:
                    numeric_value = self._to_float(values.get(key))
                    row[col_label] = "-" if numeric_value is None else f"{numeric_value:.4f}"
            row[self._run_key_column] = str(run_key)
            row[self._trained_sort_column] = float(sort_value)
            rows.append(row)

        frame = pd.DataFrame(rows, columns=self._runs_display_columns + [self._run_key_column, self._trained_sort_column])
        if not frame.empty and self._trained_sort_column in frame.columns:
            frame = frame.sort_values(by=self._trained_sort_column, ascending=False, kind="stable").reset_index(drop=True)
        self.runs_table.set_dataframe(frame, include_index=False)
        self.runs_table.hideColumn(len(self._runs_display_columns))
        self.runs_table.hideColumn(len(self._runs_display_columns) + 1)

    @staticmethod
    def _to_float(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    def _row_values(self, run: RegressionRunResult, context: dict[str, object]) -> dict[str, Optional[float | str]]:
        metrics = run.metrics or {}
        cv_scores = run.cv_scores or {}
        target_payload = context.get("target") if isinstance(context, dict) else None
        inputs_payloads = context.get("inputs") if isinstance(context, dict) else None
        reducer_info = {}
        if isinstance(context, dict):
            reducer_info = context.get("dimensionality_reduction") or context.get("pca") or {}
            if not reducer_info:
                hyper = context.get("hyperparameters") or {}
                reducer_info = (
                    hyper.get("dimensionality_reduction")
                    or hyper.get("pca")
                    or {}
                )

        def _metric(name: str) -> Optional[float]:
            value = metrics.get(name)
            if value is None:
                return None
            try:
                num = float(value)  # type: ignore
            except (TypeError, ValueError):
                return None
            if math.isnan(num) or math.isinf(num):
                return None
            return num

        def _cv_stats(name: str) -> tuple[Optional[float], Optional[float]]:
            values = cv_scores.get(name)
            if not values:
                return None, None
            try:
                arr = np.asarray(values, dtype=float)
            except Exception:
                return None, None
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return None, None
            return float(np.mean(finite)), float(np.std(finite))

        selector_label = run.selector_label.strip() if run.selector_label else ""
        selector_display = selector_label if selector_label else tr("None")
        target_display = self._payload_label(target_payload)
        inputs_count = len(inputs_payloads) if isinstance(inputs_payloads, list) else None
        reducer_display = tr("Disabled")
        run_reducer_label = str(getattr(run, "reducer_label", "") or "").strip()
        run_reducer_key = str(getattr(run, "reducer_key", "") or "").strip()
        if run_reducer_key and run_reducer_key != "none":
            reducer_display = run_reducer_label or run_reducer_key
        elif isinstance(reducer_info, dict) and reducer_info:
            method_label = reducer_info.get("label")
            method_key = reducer_info.get("method")
            methods = reducer_info.get("methods")
            enabled = bool(reducer_info.get("enabled"))
            if isinstance(method_label, str) and method_label.strip():
                reducer_display = method_label.strip()
            elif isinstance(method_key, str) and method_key and method_key != "none":
                reducer_display = method_key
            elif isinstance(methods, list):
                cleaned = [str(m) for m in methods if str(m) and str(m) != "none"]
                if cleaned:
                    reducer_display = ", ".join(cleaned)
                elif not enabled:
                    reducer_display = tr("Disabled")
            elif enabled:
                reducer_display = tr("Enabled")

        cv_r2_mean, cv_r2_std = _cv_stats("r2")
        cv_rmse_mean, cv_rmse_std = _cv_stats("rmse")
        trained_at_display, _sort_value = self._trained_at_display_and_sort(run, context)
        rows_total, rows_train, rows_test = self._row_counts(run, context)
        inputs_selected = self._inputs_selected(run, context)

        return {
            "trained_at": trained_at_display,
            "model": run.model_label,
            "selector": selector_display,
            "reducer": reducer_display,
            "target": target_display,
            "inputs": inputs_count if inputs_count is not None else "-",
            "inputs_selected": inputs_selected if inputs_selected is not None else "-",
            "rows_total": rows_total if rows_total is not None else "-",
            "rows_train": rows_train if rows_train is not None else "-",
            "rows_test": rows_test if rows_test is not None else "-",
            "r2_train": _metric("r2_train"),
            "r2_test": _metric("r2_test"),
            "rmse_train": _metric("rmse_train"),
            "rmse_test": _metric("rmse_test"),
            "cv_r2_mean": cv_r2_mean,
            "cv_r2_std": cv_r2_std,
            "cv_rmse_mean": cv_rmse_mean,
            "cv_rmse_std": cv_rmse_std,
        }

    def _inputs_selected(self, run: RegressionRunResult, context: dict[str, object]) -> Optional[int]:
        for source in (context, getattr(run, "metadata", None)):
            if isinstance(source, dict) and "inputs_selected" in source:
                try:
                    return int(source.get("inputs_selected"))  # type: ignore[arg-type]
                except Exception:
                    return None
        return None

    def _row_counts(
        self, run: RegressionRunResult, context: dict[str, object]
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        row_counts = None
        for source in (context, getattr(run, "metadata", None)):
            if isinstance(source, dict):
                if source.get("row_counts"):
                    row_counts = source.get("row_counts")
                    break
                if any(k in source for k in ("rows_total", "rows_train", "rows_test")):
                    total = self._to_int(source.get("rows_total"))
                    train = self._to_int(source.get("rows_train"))
                    test = self._to_int(source.get("rows_test"))
                    return total, train, test
        if isinstance(row_counts, dict):
            total = self._to_int(row_counts.get("total"))
            train = self._to_int(row_counts.get("train"))
            test = self._to_int(row_counts.get("test"))
            return total, train, test

        timeline = run.timeline_frame if run.timeline_frame is not None else pd.DataFrame()
        if timeline is None or timeline.empty or "t" not in timeline.columns:
            return None, None, None

        total = int(len(timeline))
        train = None
        test = None

        train_col = "Prediction (train)"
        if train_col in timeline.columns:
            train = int(pd.to_numeric(timeline[train_col], errors="coerce").notna().sum())
        else:
            split_train_cols = [
                c for c in timeline.columns
                if isinstance(c, str)
                and c.startswith("Prediction (train split ")
                and c.endswith(")")
            ]
            if split_train_cols:
                split_frame = timeline[split_train_cols].apply(pd.to_numeric, errors="coerce")
                train = int(split_frame.notna().any(axis=1).sum())

        test_col = "Prediction (test)"
        if test_col in timeline.columns:
            test = int(pd.to_numeric(timeline[test_col], errors="coerce").notna().sum())

        if test is None and total is not None and train is not None:
            test = max(0, total - train)

        return total, train, test

    @staticmethod
    def _to_int(value: object) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _trained_at_display_and_sort(
        self,
        run: RegressionRunResult,
        context: dict[str, object],
    ) -> tuple[str, float]:
        raw = ""
        if getattr(run, "trained_at", None):
            raw = str(run.trained_at)
        elif isinstance(context, dict):
            raw = str(context.get("trained_at") or "")

        if not raw:
            metadata = getattr(run, "metadata", None)
            if isinstance(metadata, dict):
                raw = str(metadata.get("trained_at") or "")

        if not raw:
            return "-", float("-inf")

        try:
            ts = pd.to_datetime(raw, errors="coerce", utc=True)
            if pd.isna(ts):
                return raw, float("-inf")
            display = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
            return display, float(ts.timestamp())
        except Exception:
            return raw, float("-inf")

    # ------------------------------------------------------------------
    def _on_selection_changed(self) -> None:
        keys = self._selected_keys()
        if not keys:
            return
        key = keys[0]
        self._current_selection_key = key
        self._apply_run(self._runs.get(key))

    def _apply_run(self, run: Optional[RegressionRunResult]) -> None:
        if run is None:
            self.predictions_table.set_dataframe(pd.DataFrame(columns=self._predictions_columns), include_index=False)
            self.scatter_chart.clear()
            self._set_scatter_title(None)
            self.timeline_chart.clear()
            return

        timeline = run.timeline_frame if run.timeline_frame is not None else pd.DataFrame()
        run_key = str(getattr(run, "key", "")) if run is not None else ""
        cached_timeline = self._timeline_cache.get(run_key)
        if cached_timeline is None:
            cached_timeline = self._timeline_chart_frame(timeline)
            self._timeline_cache[run_key] = cached_timeline
        self.timeline_chart.set_dataframe(cached_timeline)

        cached_predictions = self._predictions_cache.get(run_key)
        if cached_predictions is None:
            cached_predictions = self._prediction_table_frame(timeline)
            self._predictions_cache[run_key] = cached_predictions
        self.predictions_table.set_dataframe(cached_predictions, include_index=False, float_format="{:.4f}")

        scatter = run.scatter_frame if run.scatter_frame is not None else pd.DataFrame()
        self.scatter_chart.set_points(scatter, force_equal_axes=True)
        self._set_scatter_title(scatter)
        self._rebalance_splitters()

    def _set_scatter_title(self, scatter: pd.DataFrame | None) -> None:
        corr_text = tr("Correlation: n/a")
        if scatter is not None and not scatter.empty:
            try:
                if {"actual", "predicted"}.issubset(scatter.columns):
                    corr_frame = scatter[["actual", "predicted"]].copy()
                    corr_frame["actual"] = pd.to_numeric(corr_frame["actual"], errors="coerce")
                    corr_frame["predicted"] = pd.to_numeric(corr_frame["predicted"], errors="coerce")
                    corr_frame = corr_frame.dropna(subset=["actual", "predicted"])
                    if len(corr_frame) >= 2:
                        corr_value = float(corr_frame["actual"].corr(corr_frame["predicted"]))
                        if math.isfinite(corr_value):
                            corr_text = tr("Correlation: {value}").format(value=f"{corr_value:.3f}")
            except Exception:
                logger.warning("Exception in _set_scatter_title", exc_info=True)
        try:
            self.scatter_chart.chart.setTitle(tr("Actual vs Predicted ({corr})").format(corr=corr_text))
        except Exception:
            logger.warning("Exception in _set_scatter_title", exc_info=True)

    def _timeline_chart_frame(self, timeline: pd.DataFrame) -> pd.DataFrame:
        if timeline is None or timeline.empty or "t" not in timeline.columns:
            return pd.DataFrame()
        chart_df = timeline
        out = pd.DataFrame({"t": chart_df["t"]})

        if "Actual" in chart_df.columns:
            actual = self._numeric_series(chart_df["Actual"])
            if actual.notna().any():
                out["Actual"] = actual

        train_col = "Prediction (train)"
        if train_col in chart_df.columns:
            train = self._numeric_series(chart_df[train_col])
        else:
            split_train_cols = [
                c for c in chart_df.columns
                if isinstance(c, str)
                and c.startswith("Prediction (train split ")
                and c.endswith(")")
            ]
            train = None
            if split_train_cols:
                split_frame = self._numeric_frame(chart_df[split_train_cols])
                # CV split traces are mutually exclusive per row; merge into one train series.
                train = split_frame.bfill(axis=1).iloc[:, 0]
        if train is not None and train.notna().any():
            out[train_col] = train

        test_col = "Prediction (test)"
        if test_col in chart_df.columns:
            test = self._numeric_series(chart_df[test_col])
            if test.notna().any():
                out[test_col] = test

        return out if len(out.columns) > 1 else pd.DataFrame(columns=["t"])

    def _prediction_table_frame(self, timeline: pd.DataFrame) -> pd.DataFrame:
        if timeline is None or timeline.empty or "t" not in timeline.columns:
            return pd.DataFrame(columns=self._predictions_columns)

        chart_df = timeline
        actual_col = "Actual" if "Actual" in chart_df.columns else None

        prediction: Optional[pd.Series] = None
        split_series: Optional[pd.Series] = None
        train_col = "Prediction (train)"
        if train_col in chart_df.columns:
            prediction = self._numeric_series(chart_df[train_col])
            if "Split (train)" in chart_df.columns:
                split_series = chart_df["Split (train)"].astype(str)
        else:
            split_train_cols = [
                c for c in chart_df.columns
                if isinstance(c, str)
                and c.startswith("Prediction (train split ")
                and c.endswith(")")
            ]
            if split_train_cols:
                split_frame = self._numeric_frame(chart_df[split_train_cols])
                prediction = split_frame.bfill(axis=1).iloc[:, 0]
                split_labels = []
                for col in split_train_cols:
                    label = col[len("Prediction (train split ") : -1].strip() or "train"
                    split_labels.append(label)
                split_series = pd.Series(index=chart_df.index, dtype=object)
                for col, label in zip(split_train_cols, split_labels):
                    mask = split_frame[col].notna()
                    split_series.loc[mask] = label

        test_col = "Prediction (test)"
        if test_col in chart_df.columns:
            test_pred = self._numeric_series(chart_df[test_col])
            if prediction is None:
                prediction = test_pred
                split_series = pd.Series(["test"] * len(chart_df), index=chart_df.index, dtype=object)
            else:
                missing = prediction.notna()
                prediction = prediction.where(missing, test_pred)
                if split_series is None:
                    split_series = pd.Series(index=chart_df.index, dtype=object)
                split_series = split_series.where(missing, "test")

        if prediction is None:
            return pd.DataFrame(columns=self._predictions_columns)

        ts = pd.to_datetime(chart_df["t"], errors="coerce", utc=True)
        ref = self._numeric_series(chart_df[actual_col]) if actual_col else np.nan

        out = pd.DataFrame(
            {
                "Date": ts,
                "Prediction": prediction,
                "Reference": ref,
                "Split": split_series,
            }
        )
        out = out.dropna(subset=["Date", "Prediction"])
        if out.empty:
            return pd.DataFrame(columns=self._predictions_columns)
        out["Error"] = out["Prediction"] - out["Reference"]
        if "Split" in out.columns:
            out["Split"] = out["Split"].fillna("train")
        out = out.sort_values(by="Date", kind="stable")
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        out = out.reset_index(drop=True)
        return out.reindex(columns=self._predictions_columns)

    @staticmethod
    def _numeric_series(series: pd.Series) -> pd.Series:
        try:
            if pd.api.types.is_numeric_dtype(series):
                return series
        except Exception:
            logger.warning("Exception in _numeric_series", exc_info=True)
        return pd.to_numeric(series, errors="coerce")

    @staticmethod
    def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
        try:
            if all(pd.api.types.is_numeric_dtype(frame[col]) for col in frame.columns):
                return frame
        except Exception:
            logger.warning("Exception in _numeric_frame", exc_info=True)
        return frame.apply(pd.to_numeric, errors="coerce")

    def _payload_label(self, payload: object) -> str:
        if not isinstance(payload, dict):
            return "-"
        label = str(display_name(payload)).strip()
        if label:
            return label
        fid = payload.get("feature_id")
        return tr("Feature {id}").format(id=fid) if fid is not None else "-"

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


__all__ = ["RegressionResultsPanel"]

