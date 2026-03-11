
from __future__ import annotations
from typing import Optional

from PySide6.QtWidgets import QWidget

from ...models.database_model import DatabaseModel
from ...localization import tr
from ...threading import run_in_main_thread, run_in_thread
from ...utils import set_status_text, toast_error, toast_file_saved, toast_info, toast_success, toast_warn
from ...utils.exporting import execute_export_plan, prepare_dataframes_export_plan
from ...widgets.export_dialog import ExportOption, ExportSelectionDialog
from ...viewmodels.help_viewmodel import get_help_viewmodel
from .viewmodel import StatisticsViewModel
from .preview_panel import StatisticsPreview
from .sidebar import StatisticsSidebar
import logging

logger = logging.getLogger(__name__)

from ..tab_widget import TabWidget

class StatisticsTab(TabWidget):
    """Interactive tab to preview and save statistics results."""

    def __init__(
        self,
        database_model: DatabaseModel,
        parent: Optional[QWidget] = None,
    ):
        self._view_model = StatisticsViewModel(database_model, parent=None)
        self._toast_compute_active = False
        self._toast_save_active = False

        super().__init__(parent)

        try:
            self._view_model.setParent(self)
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)
        self._wire_signals()
        self._view_model.database_changed.connect(self._on_model_database_changed)
        self._sidebar.export_requested.connect(self._on_export_requested)
        self._sidebar.save_requested.connect(self._on_save_requested)

    # ------------------------------------------------------------------
    def _create_sidebar(self) -> QWidget:
        try:
            help_viewmodel = get_help_viewmodel()
        except Exception:
            help_viewmodel = None
        self._sidebar = StatisticsSidebar(
            self._view_model,
            parent=self,
            help_viewmodel=help_viewmodel,
        )
        return self._sidebar

    def _create_content_widget(self) -> QWidget:
        self._preview_panel = StatisticsPreview(parent=self)
        return self._preview_panel

    # ------------------------------------------------------------------
    def _wire_signals(self) -> None:
        self._view_model.status_changed.connect(self._on_status_changed)
        self._view_model.mode_changed.connect(self._preview_panel.set_mode)
        self._view_model.preview_updated.connect(self._preview_panel.update_preview)
        self._view_model.statistics_failed.connect(self._on_statistics_failed)
        self._view_model.statistics_warning.connect(self._on_statistics_warning)
        self._view_model.save_finished.connect(self._on_save_finished)
        self._view_model.save_failed.connect(self._on_save_failed)

    # ------------------------------------------------------------------
    def _on_model_database_changed(self, _database) -> None:
        # Sidebar reacts to this signal for filters; the view model signals
        # already reset the preview, so nothing additional is required here.
        pass

    def close_database(self) -> None:
        """Release resources held by the statistics view model."""
        self._view_model.close_database()

    # ------------------------------------------------------------------
    def _on_status_changed(self, text: str) -> None:
        status = (text or "").strip()
        if not status:
            return

        if status.lower().startswith("calculating statistics"):
            self._toast_compute_active = True
            self._toast_save_active = False
            set_status_text(tr("Calculating statistics..."))
            try:
                toast_info("Gathering statistics…", title="Statistics", tab_key="statistics")
            except Exception:
                logger.warning("Exception in _on_status_changed", exc_info=True)
            return

        if status.lower().startswith("saving statistics"):
            self._toast_compute_active = False
            self._toast_save_active = True
            set_status_text(tr("Saving statistics..."))
            try:
                toast_info("Saving statistics…", title="Statistics", tab_key="statistics")
            except Exception:
                logger.warning("Exception in _on_status_changed", exc_info=True)
            return

        if self._toast_compute_active and status.lower().startswith("statistics ready"):
            self._toast_compute_active = False
            self._sidebar.set_export_enabled(True)
            set_status_text(tr("Statistics ready."))
            try:
                self._preview_panel.set_status(tr("Statistics ready. Review results and save or export."))
            except Exception:
                logger.warning("Exception in _on_status_changed", exc_info=True)
            try:
                toast_success("Statistics preview is ready.", title="Statistics", tab_key="statistics")
            except Exception:
                logger.warning("Exception in _on_status_changed", exc_info=True)
            return

        if self._toast_compute_active and status.lower().startswith("statistics produced no results"):
            self._toast_compute_active = False
            self._sidebar.set_export_enabled(False)
            set_status_text(tr("Statistics produced no results."))
            try:
                self._preview_panel.set_status(tr("No statistics results for current selection."))
            except Exception:
                logger.warning("Exception in _on_status_changed", exc_info=True)
            try:
                toast_warn("No statistics results were produced.", title="Statistics", tab_key="statistics")
            except Exception:
                logger.warning("Exception in _on_status_changed", exc_info=True)
            return

        if self._toast_save_active and status.lower().startswith("statistics saved"):
            self._toast_save_active = False
            set_status_text(tr("Statistics saved."))
            try:
                self._preview_panel.set_status(tr("Statistics saved. You can continue editing or export."))
            except Exception:
                logger.warning("Exception in _on_status_changed", exc_info=True)
            return

    def _on_statistics_failed(self, message: str) -> None:
        self._toast_compute_active = False
        self._toast_save_active = False
        self._sidebar.set_export_enabled(False)
        try:
            toast_error(message, title="Statistics failed", tab_key="statistics")
        except Exception:
            logger.warning("Exception in _on_statistics_failed", exc_info=True)

    def _on_statistics_warning(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        try:
            toast_warn(text, title="Statistics", tab_key="statistics")
        except Exception:
            logger.warning("Exception in _on_statistics_warning", exc_info=True)

    def _on_export_requested(self) -> None:
        datasets = self._preview_panel.export_frames()
        if not datasets:
            toast_info(tr("Gather statistics before exporting."), title=tr("Statistics"), tab_key="statistics")
            return
        options = [ExportOption(key=name, label=name) for name in datasets.keys()]
        dialog = ExportSelectionDialog(
            title=tr("Export statistics"),
            heading=tr("Choose which statistics outputs to export and in which format."),
            options=options,
            parent=self,
        )
        if dialog.exec() != ExportSelectionDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_keys()
        if not selected:
            toast_info(tr("Select at least one export item."), title=tr("Export"), tab_key="statistics")
            return

        chosen = {name: datasets[name] for name in selected if name in datasets}
        plan = prepare_dataframes_export_plan(
            parent=self,
            title=tr("Export statistics"),
            selected_format=dialog.selected_format(),
            datasets=chosen,
        )
        if plan is None:
            return

        self._sidebar.set_export_enabled(False)
        set_status_text(tr("Exporting statistics..."))
        toast_info(tr("Exporting statistics..."), title=tr("Statistics"), tab_key="statistics")

        def _on_export_finished(outcome) -> None:
            self._sidebar.set_export_enabled(True)
            if not getattr(outcome, "message", ""):
                return
            if bool(getattr(outcome, "ok", False)):
                set_status_text(tr("Statistics export finished."))
                toast_success(tr("Statistics export finished."), title=tr("Statistics"), tab_key="statistics")
                open_path = getattr(outcome, "open_path", None)
                if open_path:
                    toast_file_saved(open_path, title=tr("Export saved"), tab_key="statistics")
                return
            set_status_text(tr("Statistics export warning."))
            toast_warn(str(outcome.message), title=tr("Export"), tab_key="statistics")

        def _on_export_error(message: str) -> None:
            self._sidebar.set_export_enabled(True)
            set_status_text(tr("Statistics export failed."))
            text = tr("Statistics export failed: {error}").format(error=str(message or tr("Unknown error")))
            toast_error(text, title=tr("Export failed"), tab_key="statistics")

        run_in_thread(
            execute_export_plan,
            on_result=lambda outcome: run_in_main_thread(_on_export_finished, outcome),
            on_error=lambda message: run_in_main_thread(_on_export_error, message),
            plan=plan,
            owner=self,
            key="statistics_export",
            cancel_previous=True,
        )

    def _on_save_requested(self) -> None:
        edited_preview = self._preview_panel.editable_preview_for_save()
        self._view_model.save_current_result(preview_override=edited_preview)

    def _on_save_finished(self, inserted: int) -> None:
        try:
            toast_success(f"Saved {inserted} measurements.", title="Statistics", tab_key="statistics")
        except Exception:
            logger.warning("Exception in _on_save_finished", exc_info=True)

    def _on_save_failed(self, message: str) -> None:
        self._toast_save_active = False
        text = message.strip()
        if text.lower().startswith("gather statistics"):
            try:
                toast_info(text, title="Statistics", tab_key="statistics")
            except Exception:
                logger.warning("Exception in _on_save_failed", exc_info=True)
        else:
            try:
                toast_error(message, title="Save failed", tab_key="statistics")
            except Exception:
                logger.warning("Exception in _on_save_failed", exc_info=True)
