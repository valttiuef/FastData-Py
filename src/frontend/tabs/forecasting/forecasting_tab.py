
from __future__ import annotations
from typing import Optional
from PySide6.QtWidgets import QWidget

from ...models.hybrid_pandas_model import HybridPandasModel
from ...models.log_model import LogModel, get_log_model
from ...utils.model_details import build_forecasting_details_text, build_model_details_prompt
from ...widgets.model_details_dialog import ModelDetailsDialog
from .viewmodel import ForecastingViewModel

from .results_panel import ForecastingResultsPanel
from .sidebar import ForecastingSidebar
from ..tab_widget import TabWidget
import logging

logger = logging.getLogger(__name__)


class ForecastingTab(TabWidget):
    """Tab providing forecasting experiments over selected features."""

    def __init__(
        self,
        database_model: HybridPandasModel,
        parent: Optional[QWidget] = None,
        *,
        log_model: Optional[LogModel] = None,
    ) -> None:
        self._log_model = log_model or get_log_model()
        self._view_model = ForecastingViewModel(
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
        self.sidebar.set_models(self._view_model.available_models())
        self._view_model.database_changed.connect(self._on_model_database_changed)
        self._view_model.run_started.connect(self._on_run_started)
        self._view_model.run_partial.connect(self._on_run_partial)
        self.results_panel.details_requested.connect(self._show_model_details_dialog)
        self.results_panel.save_requested.connect(self._save_selected_runs)
        self.results_panel.delete_requested.connect(self._delete_selected_runs)
        self._on_model_database_changed(self._view_model.data_model.db)

    # ------------------------------------------------------------------
    def _create_sidebar(self) -> QWidget:
        help_viewmodel = getattr(self.window(), "help_viewmodel", None)
        self.sidebar = ForecastingSidebar(
            view_model=self._view_model,
            help_viewmodel=help_viewmodel,
            parent=self,
        )
        return self.sidebar

    def _create_content_widget(self) -> QWidget:
        self.results_panel = ForecastingResultsPanel(self, view_model=self._view_model)
        return self.results_panel

    def _on_model_database_changed(self, _db) -> None:
        self.sidebar.set_models(self._view_model.available_models())
        self.results_panel.clear()
        self._load_saved_runs()

    def _on_run_started(self, _expected_runs) -> None:
        context = self._build_run_context()
        self.results_panel.set_run_context(context)

    def _on_run_partial(self, run) -> None:
        if self.sidebar.auto_save_enabled():
            self._auto_save_run(run)

    def _build_run_context(self) -> dict[str, object]:
        features = self.sidebar.selected_payloads()
        target = self.sidebar.target_payload()
        preprocessing = self.sidebar.preprocessing_parameters()
        data_filters = self.sidebar.data_selector.build_data_filters()
        model_params = self.sidebar.model_parameters()
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
            "features": [dict(p) for p in features],
            "feature": dict(features[0]) if features else {},
            "target": dict(target) if target else {},
            "filters": filters,
            "preprocessing": dict(preprocessing or {}),
            "model_params": dict(model_params or {}),
            "forecast": {
                "forecast_horizon": self.sidebar.forecast_horizon(),
                "window_strategy": self.sidebar.window_strategy(),
                "initial_window": self.sidebar.initial_window_size(),
            },
        }

    def _load_saved_runs(self) -> None:
        try:
            saved_runs = self._view_model.load_saved_runs()
        except Exception:
            saved_runs = []
        if not saved_runs:
            return
        for run in saved_runs:
            self.results_panel.update_run(run, select_if_new=False)
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
        summary_text = build_forecasting_details_text(runs, contexts)
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

    def _save_selected_runs(self) -> None:
        runs = self.results_panel.selected_runs()
        if not runs:
            return
        for run in runs:
            if getattr(run, "model_id", None):
                continue
            context = self.results_panel.context_for_key(run.key)
            if not context:
                continue
            context = self._ensure_context_labels(run, context)
            try:
                model_id = self._view_model.save_run(run, context)
            except Exception:
                continue
            if model_id is not None:
                self.results_panel.mark_run_saved(run.key, model_id, context)

    def _auto_save_run(self, run) -> None:
        if run is None or getattr(run, "model_id", None):
            return
        context = self.results_panel.context_for_key(run.key)
        if not context:
            context = self._build_run_context()
        context = self._ensure_context_labels(run, context)
        try:
            model_id = self._view_model.save_run(run, context)
        except Exception:
            return
        if model_id is not None:
            self.results_panel.mark_run_saved(run.key, model_id, context)

    def _delete_selected_runs(self) -> None:
        runs = self.results_panel.selected_runs()
        if not runs:
            return
        for run in runs:
            model_id = getattr(run, "model_id", None)
            if model_id:
                try:
                    self._view_model.delete_saved_model(model_id)
                except Exception:
                    logger.warning("Exception in _delete_selected_runs", exc_info=True)
        self.results_panel.remove_runs([run.key for run in runs])

    def _ensure_context_labels(self, run, context: dict[str, object]) -> dict[str, object]:
        context = dict(context)
        context.setdefault("model_label", run.model_label)
        context.setdefault("feature_label", run.feature_label)
        if not context.get("feature") and context.get("features"):
            for payload in context.get("features") or []:
                try:
                    if str(payload.get("feature_id")) == str(run.feature_key):
                        context["feature"] = dict(payload)
                        break
                except Exception:
                    continue
        return context


__all__ = ["ForecastingTab"]
