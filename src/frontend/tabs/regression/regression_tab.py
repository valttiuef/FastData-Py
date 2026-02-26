
from __future__ import annotations
from typing import Optional
import pandas as pd
from PySide6.QtWidgets import (

    QDialog,
    QMessageBox,
    QWidget,
)

import logging
logger = logging.getLogger(__name__)
from ...localization import tr

from ...models.hybrid_pandas_model import HybridPandasModel
from ...models.log_model import LogModel, get_log_model
from ...threading import run_in_main_thread
from ...utils import clear_status_text, set_status_text, toast_error, toast_info, toast_success
from ...utils.exporting import export_dataframes
from ...utils.model_details import build_model_details_prompt, build_regression_details_text
from ...widgets.export_dialog import ExportOption, ExportSelectionDialog
from ...widgets.model_details_dialog import ModelDetailsDialog
from ...widgets.save_predictions_dialog import SavePredictionsDialog
from .viewmodel import RegressionViewModel

from .results_panel import RegressionResultsPanel
from .sidebar import RegressionSidebar
from ..tab_widget import TabWidget



class RegressionTab(TabWidget):
    """Tab providing regression experiments over selected features."""

    def __init__(
        self,
        database_model: HybridPandasModel,
        parent: Optional[QWidget] = None,
        *,
        log_model: Optional[LogModel] = None,
    ) -> None:
        self._log_model = log_model or get_log_model()
        self._view_model = RegressionViewModel(
            database_model,
            parent=None,
            log_model=self._log_model,
        )
        self._details_dialog: Optional[ModelDetailsDialog] = None

        super().__init__(parent)

        if log_model is None and self._log_model.parent() is None:
            try:
                self._log_model.setParent(self)
            except Exception:
                logger.warning("Exception in __init__", exc_info=True)
        try:
            self._view_model.setParent(self)
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)
        self._connect_signals()

        self._populate_options()

        self._view_model.database_changed.connect(self._on_model_database_changed)
        self._view_model.run_started.connect(self._on_run_started)
        self._view_model.run_partial.connect(self._on_run_partial)
        self._view_model.run_finished.connect(self._on_run_finished)
        self._view_model.run_failed.connect(self._on_run_failed)
        self._view_model.status_changed.connect(self._on_status_changed)
        self._view_model.run_context_changed.connect(self.results_panel.set_run_context)
        self._view_model.save_completed.connect(self._on_save_completed)
        self.results_panel.details_requested.connect(self._show_model_details_dialog)
        self.results_panel.save_requested.connect(self._save_selected_runs)
        self.results_panel.delete_requested.connect(self._delete_selected_runs)
        self.results_panel.export_available_changed.connect(self.sidebar.set_export_enabled)
        self.sidebar.export_requested.connect(self._export_results)
        self.sidebar.save_predictions_requested.connect(self._save_predictions_from_selected_run)
        self.sidebar.save_models_requested.connect(self._save_selected_runs)
        self.sidebar.delete_models_requested.connect(self._delete_selected_runs)
        self._on_model_database_changed(self._view_model.data_model.db)

    # ------------------------------------------------------------------
    def _create_sidebar(self) -> QWidget:
        help_viewmodel = getattr(self.window(), "help_viewmodel", None)
        self.sidebar = RegressionSidebar(
            view_model=self._view_model,
            help_viewmodel=help_viewmodel,
            parent=self,
        )
        return self.sidebar

    def _create_content_widget(self) -> QWidget:
        self.results_panel = RegressionResultsPanel(self)
        return self.results_panel

    def _connect_signals(self) -> None:
        self._view_model.run_requested.connect(self._on_run_clicked)
        self._view_model.features_changed.connect(self._update_stratify_options)

    # ------------------------------------------------------------------
    def _populate_options(self) -> None:
        selectors = self._view_model.available_feature_selectors()
        models = self._view_model.available_models()
        reducers = self._view_model.available_dimensionality_reducers()
        cv_items = self._view_model.available_cv_strategies()
        test_items = self._view_model.available_test_strategies()
        group_kinds = self._view_model.available_group_kinds()

        self.sidebar.set_selectors(selectors)
        self.sidebar.set_models(models)
        self.sidebar.set_reducers(reducers)
        self.sidebar.set_cv_strategies(cv_items)
        self.sidebar.set_test_strategies(test_items)
        self.sidebar.set_group_kinds(group_kinds)
        self._update_stratify_options()

    # ------------------------------------------------------------------
    def _on_model_database_changed(self, _db) -> None:
        self._populate_options()
        self.results_panel.clear()
        self._load_saved_runs()

    def _on_run_clicked(self) -> None:
        if self._view_model.is_running():
            set_status_text("Regression run already in progress.")
            try:
                toast_info("Regression run already in progress.", title="Regression", tab_key="regression")
            except Exception:
                logger.warning("Exception in _on_run_clicked", exc_info=True)
            return

        input_features = self.sidebar.selected_input_payloads()
        target_features = self.sidebar.selected_target_payloads()
        missing: list[str] = []
        if not input_features:
            missing.append("input features")
        if not target_features:
            missing.append("target feature(s)")

        model_keys = self.sidebar.selected_model_keys()
        if not model_keys:
            missing.append("regression model")

        selector_keys = self.sidebar.selected_selector_keys()
        if not selector_keys:
            missing.append("feature selection method")

        if missing:
            missing_text = ", ".join(missing)
            message = f"Cannot run regression: missing {missing_text}."
            set_status_text(message)
            try:
                toast_error(message, title="Regression failed", tab_key="regression")
            except Exception:
                logger.warning("Exception in _on_run_clicked", exc_info=True)
            return

        selector_keys = selector_keys or ["none"]
        selector_params = self.sidebar.selector_parameters()
        model_params = self.sidebar.model_parameters()
        reducer_keys = self.sidebar.selected_reducer_keys() or ["none"]
        reducer_params = self.sidebar.reducer_parameters()

        total_runs = len(target_features) * len(selector_keys) * len(reducer_keys) * len(model_keys)
        self.results_panel.prepare_for_run(total_runs)

        data_filters = self.sidebar.data_selector.build_data_filters()
        systems = data_filters.systems if data_filters else None
        Datasets = data_filters.Datasets if data_filters else None
        start = data_filters.start if data_filters else None
        end = data_filters.end if data_filters else None

        cv_strategy = self.sidebar.cv_combo.currentData() or "none"
        cv_folds = self.sidebar.cv_folds.value()
        cv_group_kind = self.sidebar.selected_group_kind()
        shuffle = self.sidebar.cv_shuffle.isChecked()
        time_gap = self.sidebar.cv_gap.value()

        stratify_payload = self.sidebar.selected_stratify_payload()
        if self.sidebar.enable_test_split.isChecked():
            test_size = float(self.sidebar.test_size.value())
            if test_size >= 1:
                test_size = int(round(test_size))
            test_strategy = self.sidebar.test_strategy.currentData() or "random"
            stratify_bins = self.sidebar.stratify_bins.value()
        else:
            test_size = 0.0
            test_strategy = "random"
            stratify_bins = self.sidebar.stratify_bins.value()

        preprocessing = self.sidebar.data_selector.preprocessing_widget.parameters()
        group_ids = data_filters.group_ids if data_filters else None
        all_payloads: list[dict] = []
        seen_ids: set[int] = set()
        for payload in [*input_features, *target_features, *( [stratify_payload] if stratify_payload else [] )]:
            if not isinstance(payload, dict):
                continue
            payload_dict = dict(payload)
            fid = payload_dict.get("feature_id")
            try:
                key = int(fid) if fid is not None else None
            except Exception:
                key = None
            if key is not None and key in seen_ids:
                continue
            if key is not None:
                seen_ids.add(key)
            all_payloads.append(payload_dict)
        data_frame = self.sidebar.data_selector.fetch_base_dataframe_for_features(
            all_payloads,
        )
        if data_frame is None:
            set_status_text("Failed to load regression data.")
            try:
                toast_error("Failed to load regression data.", title="Regression failed", tab_key="regression")
            except Exception:
                logger.warning("Exception in _on_run_clicked", exc_info=True)
            return
        target_contexts = [
            self._build_run_context(
                input_features,
                target_feature,
                stratify_payload,
                data_filters,
                preprocessing,
                selector_params,
                model_params,
                reducer_keys,
                reducer_params,
                cv_strategy,
                cv_folds,
                cv_group_kind,
                shuffle,
                test_size,
                test_strategy,
                stratify_bins,
                time_gap,
            )
            for target_feature in target_features
        ]

        if not self._confirm_run(
            input_features=input_features,
            targets=target_features,
            model_keys=model_keys,
            selector_keys=selector_keys,
            reducer_keys=reducer_keys,
            total_runs=total_runs,
            cv_strategy=cv_strategy,
            cv_folds=cv_folds,
            cv_group_kind=cv_group_kind,
            shuffle=shuffle,
            time_gap=time_gap,
            test_size=test_size,
            test_strategy=test_strategy,
            stratify_bins=stratify_bins,
            stratify_payload=stratify_payload,
        ):
            return

        try:
            self._view_model.start_regressions(
                data_frame=data_frame,
                input_features=input_features,
                target_features=target_features,
                target_contexts=target_contexts,
                selectors=selector_keys,
                models=model_keys,
                systems=systems,
                Datasets=Datasets,
                groups=group_ids,
                start=start,
                end=end,
                selector_params=selector_params,
                model_params=model_params,
                reducers=reducer_keys,
                reducer_params=reducer_params,
                cv_strategy=cv_strategy,
                cv_folds=cv_folds,
                cv_group_kind=cv_group_kind,
                shuffle=shuffle,
                random_state=0,
                test_size=test_size,
                test_strategy=test_strategy,
                stratify_bins=stratify_bins,
                time_series_gap=time_gap,
                preprocessing=preprocessing,
                stratify_feature=stratify_payload,
            )
        except Exception:
            set_status_text("Failed to start regression run.")
            self.results_panel.show_failure("Failed to start regression run.")
            try:
                toast_error("Failed to start regression run.", title="Regression failed", tab_key="regression")
            except Exception:
                logger.warning("Exception in _on_run_clicked", exc_info=True)

    def _on_run_started(self) -> None:
        def _update() -> None:
            self.sidebar.run_button.setEnabled(False)

        run_in_main_thread(_update)

    def _on_run_partial(self, run) -> None:
        def _update() -> None:
            try:
                self.results_panel.update_run(run)
            except Exception as error:
                message = "Failed to update regression results."
                try:
                    message = f"{message} {type(error).__name__}: {error}"
                except Exception:
                    logger.warning("Exception in _update", exc_info=True)
                set_status_text(message)
                try:
                    toast_error(message, title="Regression failed", tab_key="regression")
                except Exception:
                    logger.warning("Exception in _update", exc_info=True)
                return
            if self.sidebar.auto_save_enabled():
                try:
                    self._auto_save_run(run)
                except Exception:
                    logger.warning("Exception in _update", exc_info=True)

        run_in_main_thread(_update)

    def _on_run_finished(self, summary) -> None:
        def _update() -> None:
            self.sidebar.run_button.setEnabled(True)
            try:
                self.results_panel.set_summary(summary)
            except Exception as error:
                message = "Failed to update regression results."
                try:
                    message = f"{message} {type(error).__name__}: {error}"
                except Exception:
                    logger.warning("Exception in _update", exc_info=True)
                set_status_text(message)
                try:
                    toast_error(message, title="Regression failed", tab_key="regression")
                except Exception:
                    logger.warning("Exception in _update", exc_info=True)

        run_in_main_thread(_update)

    def _on_run_failed(self, message: str) -> None:
        def _update() -> None:
            self.sidebar.run_button.setEnabled(True)
            text = message or "Regression failed."
            set_status_text(text)
            self.results_panel.show_failure(text)

        run_in_main_thread(_update)

    def _on_status_changed(self, message: str) -> None:
        text = message or ""

        def _apply() -> None:
            if text:
                set_status_text(text)
            else:
                clear_status_text()

        run_in_main_thread(_apply)

    def close_database(self) -> None:
        """Release resources held by the regression view model."""
        self._view_model.close_database()

    def _build_run_context(
        self,
        input_features,
        target_feature,
        stratify_feature,
        data_filters,
        preprocessing,
        selector_params,
        model_params,
        reducer_keys,
        reducer_params,
        cv_strategy,
        cv_folds,
        cv_group_kind,
        shuffle,
        test_size,
        test_strategy,
        stratify_bins,
        time_gap,
    ) -> dict[str, object]:
        filters = {}
        if data_filters is not None:
            filters = {
                "systems": list(data_filters.systems or []),
                "Datasets": list(data_filters.Datasets or []),
                "group_ids": list(data_filters.group_ids or []),
                "start": data_filters.start,
                "end": data_filters.end,
            }
        return {
            "inputs": [dict(p) for p in input_features],
            "target": dict(target_feature) if target_feature else {},
            "stratify": dict(stratify_feature) if stratify_feature else {},
            "filters": filters,
            "preprocessing": dict(preprocessing or {}),
            "model_params": dict(model_params or {}),
            "selector_params": dict(selector_params or {}),
            "dimensionality_reduction": {
                "enabled": any(key != "none" for key in (reducer_keys or [])),
                "methods": list(reducer_keys or []),
                "params": dict(reducer_params or {}),
            },
            "split": {
                "cv_strategy": cv_strategy,
                "cv_folds": cv_folds,
                "cv_group_kind": cv_group_kind,
                "shuffle": shuffle,
                "test_size": test_size,
                "test_strategy": test_strategy,
                "stratify_bins": stratify_bins,
                "time_series_gap": time_gap,
            },
        }

    def _confirm_run(
        self,
        *,
        input_features: list[dict],
        targets: list[dict],
        model_keys: list[str],
        selector_keys: list[str],
        reducer_keys: list[str],
        total_runs: int,
        cv_strategy: str,
        cv_folds: int,
        cv_group_kind: Optional[str],
        shuffle: bool,
        time_gap: int,
        test_size: float,
        test_strategy: str,
        stratify_bins: int,
        stratify_payload: Optional[dict],
    ) -> bool:
        target_labels = [self.sidebar.payload_label(t) for t in targets]
        input_labels = [self.sidebar.payload_label(p) for p in input_features]
        model_labels = [self.sidebar.model_label_for_key(k) for k in model_keys]
        selector_labels = [self.sidebar.selector_label_for_key(k) for k in selector_keys]
        selector_text = ", ".join(selector_labels) if selector_labels else "None"
        cv_label = self.sidebar.cv_combo.currentText()
        test_label = self.sidebar.test_strategy.currentText()
        stratify_label = self.sidebar.payload_label(stratify_payload) if stratify_payload else "None"
        reducer_text = "Disabled"
        if reducer_keys and any(k != "none" for k in reducer_keys):
            labels = [self.sidebar.reducer_label_for_key(k) for k in reducer_keys if k != "none"]
            reducer_text = ", ".join(labels) if labels else "Disabled"

        def _format_list(labels: list[str], *, max_items: int = 10) -> str:
            if not labels:
                return "None"
            if len(labels) <= max_items:
                return ", ".join(labels)
            return ", ".join(labels[:max_items]) + ", ..."

        if cv_strategy == "none":
            cv_text = "Disabled"
        else:
            cv_text = f"{cv_label} ({cv_folds} folds)"
            if cv_strategy == "time_series":
                cv_text += f", gap {time_gap}"
            if cv_strategy in {"kfold", "stratified_kfold"}:
                cv_text += f", shuffle {'on' if shuffle else 'off'}"
            if cv_strategy == "stratified_kfold":
                cv_text += f", bins {stratify_bins}, by {stratify_label}"
            if cv_strategy == "group_kfold":
                cv_text += f", group {cv_group_kind or 'None'}"

        if test_size > 0:
            if test_size >= 1:
                test_text = f"{test_label} ({int(round(test_size))} rows)"
            else:
                test_text = f"{test_label} ({int(round(test_size * 100))}%)"
            if test_strategy == "stratified":
                test_text += f", bins {stratify_bins}, by {stratify_label}"
        else:
            test_text = tr("Disabled")
        message = tr(
            "Confirm regression run:\n\n"
            "Targets ({targets_count}): {targets}\n"
            "Inputs ({inputs_count}): {inputs}\n"
            "Models: {models}\n"
            "Selectors: {selectors}\n\n"
            "Dimensionality reduction: {reducers}\n"
            "Cross-validation: {cv}\n"
            "Test split: {test}\n\n"
            "Total models to train: {total}"
        ).format(
            targets_count=len(target_labels),
            targets=_format_list(target_labels),
            inputs_count=len(input_labels),
            inputs=_format_list(input_labels),
            models=", ".join(model_labels),
            selectors=selector_text,
            reducers=reducer_text,
            cv=cv_text,
            test=test_text,
            total=total_runs,
        )
        confirm = QMessageBox.question(
            self,
            tr("Run regression?"),
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return confirm == QMessageBox.StandardButton.Yes

    def _load_saved_runs(self) -> None:
        try:
            saved_runs = self._view_model.load_saved_runs()
        except Exception:
            saved_runs = []
        if not saved_runs:
            return
        self.results_panel.begin_batch_update()
        try:
            for run in saved_runs:
                self.results_panel.update_run(run, select_if_new=False)
        finally:
            self.results_panel.end_batch_update()
        if not self.results_panel.selected_runs() and saved_runs:
            self.results_panel._select_row_for_key(saved_runs[0].key)

    def _resolve_log_view_model(self):
        try:
            win = self.window() or self.parent()
        except Exception:
            win = None
        if win is not None:
            return getattr(win, "log_view_model", None)
        return None

    def _show_log_window(self) -> None:
        try:
            win = self.window() or self.parent()
            if win is None:
                return
            show_chat = getattr(win, "show_chat_window", None)
            if callable(show_chat):
                show_chat()
                return
            set_log_visible = getattr(win, "set_log_visible", None)
            if callable(set_log_visible):
                set_log_visible(True)
        except Exception:
            logger.warning("Exception in _show_log_window", exc_info=True)

    def _ask_model_details_from_ai(self, summary_text: str) -> None:
        prompt = build_model_details_prompt(summary_text)
        if not prompt:
            return
        log_view_model = self._resolve_log_view_model()
        if log_view_model is None:
            return
        self._show_log_window()
        try:
            log_view_model.ask_llm(prompt)
        except Exception:
            logger.warning("Exception in _ask_model_details_from_ai", exc_info=True)

    def _show_model_details_dialog(self) -> None:
        runs = self.results_panel.selected_runs()
        if not runs:
            return
        contexts = {run.key: self.results_panel.context_for_key(run.key) for run in runs}
        summary_text = build_regression_details_text(runs, contexts)
        if self._details_dialog is None:
            self._details_dialog = ModelDetailsDialog(
                summary_text=summary_text,
                on_ask_ai=self._ask_model_details_from_ai,
                parent=self,
            )
            try:
                self._details_dialog.finished.connect(
                    lambda _res: setattr(self, "_details_dialog", None)
                )
            except Exception:
                logger.warning("Exception in _show_model_details_dialog", exc_info=True)
        else:
            try:
                self._details_dialog.set_summary_text(summary_text)
            except Exception:
                logger.warning("Exception in _show_model_details_dialog", exc_info=True)
        try:
            self._details_dialog.show()
            self._details_dialog.raise_()
            self._details_dialog.activateWindow()
        except Exception:
            logger.warning("Exception in _show_model_details_dialog", exc_info=True)

    def _export_results(self) -> None:
        all_runs = self.results_panel.all_runs()
        if not all_runs:
            toast_info(tr("Run at least one regression before exporting."), title=tr("Regression"), tab_key="regression")
            return

        selected_runs = self.results_panel.selected_runs()
        default_model_keys = [str(run.key) for run in (selected_runs or all_runs)]
        options = [
            ExportOption(
                key="summary",
                label=tr("Regression summary"),
                description=tr("One row per model run"),
            )
        ]
        default_keys = ["summary"] + [f"model::{key}" for key in default_model_keys]
        dialog = ExportSelectionDialog(
            title=tr("Export regression results"),
            heading=tr("Choose summary and model outputs to export."),
            options=options,
            default_selected_keys=default_keys,
            parent=self,
        )
        if dialog.exec() != ExportSelectionDialog.DialogCode.Accepted:
            return

        selected_keys = dialog.selected_keys()
        if not selected_keys:
            toast_info(tr("Select at least one export item."), title=tr("Export"), tab_key="regression")
            return

        include_summary = "summary" in selected_keys
        selected_model_keys = {
            key.split("model::", 1)[1]
            for key in selected_keys
            if key.startswith("model::")
        }
        chosen_runs = [run for run in all_runs if str(run.key) in selected_model_keys]

        datasets: dict[str, pd.DataFrame] = {}
        export_mode = dialog.selected_format()
        if include_summary:
            summary_runs = chosen_runs if chosen_runs else all_runs
            summary_df = self.results_panel.export_summary_frame(summary_runs)
            if not summary_df.empty:
                datasets[tr("Regression summary")] = summary_df
        if chosen_runs:
            if export_mode == "excel":
                for run in chosen_runs:
                    run_df = self.results_panel.export_individual_frame(run)
                    if run_df.empty:
                        continue
                    datasets[self.results_panel.run_label(run)] = run_df
            else:
                combined_rows: list[pd.DataFrame] = []
                for run in chosen_runs:
                    run_df = self.results_panel.export_individual_frame(run)
                    if run_df.empty:
                        continue
                    run_df = run_df.copy()
                    run_df.insert(0, "model", self.results_panel.run_label(run))
                    combined_rows.append(run_df)
                if combined_rows:
                    datasets[tr("Regression points")] = pd.concat(combined_rows, axis=0, ignore_index=True)

        if not datasets:
            toast_info(tr("No data available for selected export options."), title=tr("Export"), tab_key="regression")
            return

        ok, message = export_dataframes(
            parent=self,
            title=tr("Export regression"),
            selected_format=export_mode,
            datasets=datasets,
        )
        if not message:
            return
        if ok:
            toast_success(message, title=tr("Export complete"), tab_key="regression")
        else:
            toast_error(message, title=tr("Export failed"), tab_key="regression")

    def _save_selected_runs(self) -> None:
        runs = self.results_panel.selected_runs()
        if not runs:
            return
        pending = [run for run in runs if not getattr(run, "model_id", None)]
        if not pending:
            self._view_model.start_save_runs(runs, {}, auto=False)
            return
        count = len(pending)
        label = tr("model") if count == 1 else tr("models")
        confirm = QMessageBox.question(
            self,
            tr("Save regression models?"),
            tr("Save {count} regression {label} to the database?").format(count=count, label=label),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        contexts = {run.key: self.results_panel.context_for_key(run.key) for run in pending}
        self._view_model.start_save_runs(pending, contexts, auto=False)

    def _auto_save_run(self, run) -> None:
        if run is None or getattr(run, "model_id", None):
            return
        context = self.results_panel.context_for_key(run.key)
        if not context:
            return
        self._view_model.queue_auto_save(run, context)

    def _delete_selected_runs(self) -> None:
        runs = self.results_panel.selected_runs()
        if not runs:
            return
        count = len(runs)
        label = tr("model") if count == 1 else tr("models")
        confirm = QMessageBox.question(
            self,
            tr("Delete regression models?"),
            tr("Delete {count} regression {label}? This removes saved models from the database.").format(
                count=count,
                label=label,
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            toast_info(
                tr("Deleting regression models..."),
                title=tr("Regression"),
                tab_key="regression",
            )
        except Exception:
            logger.warning("Exception in _delete_selected_runs", exc_info=True)
        failures = 0
        for run in runs:
            model_id = getattr(run, "model_id", None)
            if model_id:
                try:
                    self._view_model.delete_saved_model(model_id)
                except Exception:
                    failures += 1
        self.results_panel.remove_runs([run.key for run in runs])
        if failures:
            try:
                toast_error(
                    tr("Some regression models could not be deleted."),
                    title=tr("Regression delete failed"),
                    tab_key="regression",
                )
            except Exception:
                logger.warning("Exception in _delete_selected_runs", exc_info=True)
            return
        try:
            toast_success(
                tr("Deleted {count} regression {label}.").format(count=count, label=label),
                title=tr("Regression deleted"),
                tab_key="regression",
            )
        except Exception:
            logger.warning("Exception in _delete_selected_runs", exc_info=True)

    def _on_save_completed(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        saved = payload.get("saved") or []
        if not isinstance(saved, list):
            return
        for entry in saved:
            if not isinstance(entry, dict):
                continue
            key = entry.get("key")
            model_id = entry.get("model_id")
            context = entry.get("context") or self.results_panel.context_for_key(key)
            if key is None or model_id is None:
                continue
            try:
                self.results_panel.mark_run_saved(key, int(model_id), context)
            except Exception:
                logger.warning("Exception in _on_save_completed", exc_info=True)

    def _save_predictions_from_selected_run(self) -> None:
        runs = self.results_panel.selected_runs()
        if not runs:
            try:
                toast_info(
                    tr("Select one regression run first."),
                    title=tr("Regression"),
                    tab_key="regression",
                )
            except Exception:
                logger.warning("Exception in _save_predictions_from_selected_run", exc_info=True)
            return

        run = runs[0]
        context = self.results_panel.context_for_key(run.key)
        target = context.get("target") if isinstance(context, dict) else {}
        if not isinstance(target, dict):
            target = {}

        frame = self.results_panel.export_individual_frame(run)
        if frame is None or frame.empty:
            try:
                toast_info(
                    tr("No predictions available in the selected run."),
                    title=tr("Regression"),
                    tab_key="regression",
                )
            except Exception:
                logger.warning("Exception in _save_predictions_from_selected_run", exc_info=True)
            return

        save_frame = frame.copy()
        if "datetime" not in save_frame.columns or "prediction" not in save_frame.columns:
            try:
                toast_error(
                    tr("Selected run does not contain exportable predictions."),
                    title=tr("Regression save failed"),
                    tab_key="regression",
                )
            except Exception:
                logger.warning("Exception in _save_predictions_from_selected_run", exc_info=True)
            return

        save_frame = save_frame.rename(columns={"datetime": "ts", "prediction": "value"})
        save_frame["ts"] = pd.to_datetime(save_frame["ts"], errors="coerce")
        save_frame["value"] = pd.to_numeric(save_frame["value"], errors="coerce")
        save_frame = save_frame.dropna(subset=["ts", "value"])
        if save_frame.empty:
            try:
                toast_error(
                    tr("No valid prediction rows could be saved."),
                    title=tr("Regression save failed"),
                    tab_key="regression",
                )
            except Exception:
                logger.warning("Exception in _save_predictions_from_selected_run", exc_info=True)
            return

        defaults = self._prediction_save_defaults(target, context)
        model_label = ""
        try:
            model_label = str(self.results_panel.run_label(run)).strip()
        except Exception:
            model_label = ""
        dialog = SavePredictionsDialog(
            model_label=model_label,
            defaults=defaults,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        values = dialog.values()
        feature_name = str(values.get("name") or "").strip()
        if not feature_name:
            try:
                toast_error(
                    tr("Name is required."),
                    title=tr("Regression save failed"),
                    tab_key="regression",
                )
            except Exception:
                logger.warning("Exception in _save_predictions_from_selected_run", exc_info=True)
            return

        feature_type = str(values.get("type") or "").strip()
        feature_payload = {
            "name": feature_name,
            "notes": str(values.get("notes") or "").strip(),
            "source": str(values.get("source") or "").strip(),
            "unit": str(values.get("unit") or "").strip(),
            "type": feature_type,
            "lag_seconds": 0,
            "tags": list(values.get("tags") or []),
        }

        source_feature_id = None
        try:
            target_id = target.get("feature_id")
            source_feature_id = int(target_id) if target_id is not None else None
        except Exception:
            source_feature_id = None

        save_frame_copy = save_frame[["ts", "value"]].copy()

        def _on_success(_result: dict[str, object]) -> None:
            try:
                self.sidebar.refresh_feature_lists()
            except Exception:
                logger.warning("Exception in _on_success", exc_info=True)

        self._view_model.start_save_predictions(
            feature=feature_payload,
            measurements=save_frame_copy,
            source_feature_id=source_feature_id,
            on_finished=_on_success,
        )

    def _prediction_save_defaults(
        self,
        target: dict[str, object],
        context: dict[str, object],
    ) -> dict[str, object]:
        target_name = str(target.get("notes") or target.get("name") or "target").strip() or "target"
        default_name = f"{target_name}_predictions"
        return {
            "name": default_name,
            "notes": default_name,
            "source": str(target.get("source") or "").strip(),
            "unit": str(target.get("unit") or "").strip(),
            "type": str(target.get("type") or "").strip(),
            "tags": ["Regression_predictions"],
        }

    def _update_stratify_options(self) -> None:
        targets = self.sidebar.selected_target_payloads()
        inputs = self.sidebar.selected_input_payloads()
        group_kinds = self._view_model.available_group_kinds()

        options: list[tuple[str, Optional[dict]]] = []
        seen: set[int] = set()

        for target in targets:
            fid = target.get("feature_id")
            key = None
            try:
                key = int(fid) if fid is not None else None
            except Exception:
                key = None
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
            options.append((self.sidebar.payload_label(target), target))

        for payload in inputs:
            fid = payload.get("feature_id")
            key = None
            try:
                key = int(fid) if fid is not None else None
            except Exception:
                key = None
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
            options.append((self.sidebar.payload_label(payload), payload))

        for kind, label in group_kinds:
            group_label = f"{tr('Group')}: {label}"
            payload = {"group_kind": str(kind), "label": group_label}
            options.append((group_label, payload))

        self.sidebar.set_stratify_options(options)


__all__ = ["RegressionTab"]
