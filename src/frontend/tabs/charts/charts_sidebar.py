
from __future__ import annotations
from PySide6.QtWidgets import QAbstractItemView, QGroupBox, QPushButton, QSizePolicy, QVBoxLayout
from ...localization import tr

from ...models.hybrid_pandas_model import FeatureSelection
from ...utils import (
    clear_progress,
    set_progress,
    set_status_text,
    toast_error,
    toast_info,
    toast_warn,
)
from ...widgets.data_selector_widget import DataSelectorWidget
from ...widgets.sidebar_widget import SidebarWidget
from .viewmodel import CorrelationEntry, CorrelationSearchResult, ChartsViewModel
import logging

logger = logging.getLogger(__name__)


class ChartsSidebar(SidebarWidget):
    """Sidebar container exposing chart actions and shared data selector controls."""

    def __init__(self, view_model: ChartsViewModel, parent=None):
        super().__init__(title=tr("Charts"), parent=parent)
        self._view_model = view_model
        self._correlation_total = 0
        self._correlation_target_name = ""

        layout = self.content_layout()

        actions = QGroupBox(tr("Actions"))
        actions_layout = QVBoxLayout(actions)

        self.btn_add_chart = QPushButton(tr("Add chart"))
        self.btn_remove_chart = QPushButton(tr("Remove chart"))
        self.btn_export = QPushButton(tr("Export results..."))
        self.btn_find_correlations = QPushButton(tr("Find top feature correlations"))
        self.btn_add_chart.clicked.connect(lambda _checked=False: self._view_model.request_add_chart())
        self.btn_remove_chart.clicked.connect(
            lambda _checked=False: self._view_model.request_remove_chart()
        )
        self.btn_find_correlations.clicked.connect(self._on_find_correlations_clicked)

        actions_layout.addWidget(self.btn_add_chart)
        actions_layout.addWidget(self.btn_remove_chart)
        actions_layout.addWidget(self.btn_export)
        actions_layout.addWidget(self.btn_find_correlations)
        actions.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.set_sticky_actions(actions)

        self.data_selector = DataSelectorWidget(
            title=tr("Data selection"),
            parent=self,
            data_model=self._view_model.hybrid_model,
            show_features_list=True,
        )
        self.data_selector.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.features_widget = self.data_selector.features_widget
        self.features_widget.table_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._view_model.correlation_search_progress.connect(self._on_correlation_search_progress)
        self._view_model.correlation_search_finished.connect(self._on_correlation_search_done)
        self._view_model.correlation_search_failed.connect(self._on_correlation_search_done)

        layout.addWidget(self.data_selector, 1)

    # ------------------------------------------------------------------
    def available_feature_items(self) -> list[tuple[str, FeatureSelection]]:
        items: list[tuple[str, FeatureSelection]] = []
        for payload in self.features_widget.all_payloads():
            if not isinstance(payload, dict):
                continue
            try:
                selection = FeatureSelection.from_payload(payload)
            except Exception:
                continue
            items.append((selection.display_name(), selection))
        return items

    def selected_correlation_feature(self) -> FeatureSelection | None:
        payloads = self.features_widget.selected_payloads()
        if not payloads:
            return None
        payload = payloads[0] if isinstance(payloads[0], dict) else None
        if not payload:
            return None
        try:
            return FeatureSelection.from_payload(payload)
        except Exception:
            logger.warning("Exception in selected_correlation_feature", exc_info=True)
        return None

    def _on_find_correlations_clicked(self) -> None:
        feature = self.selected_correlation_feature()
        if feature is None:
            toast_info(tr("Select a target feature first."), title=tr("Charts"), tab_key="charts")
            return

        candidates = [selection for _label, selection in self.available_feature_items()]
        if len(candidates) < 2:
            toast_info(tr("At least two features are required."), title=tr("Charts"), tab_key="charts")
            return
        ordered: list[FeatureSelection] = []
        seen: set[tuple] = set()
        for item in [feature] + candidates:
            key = item.identity_key()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(item)
        started = self._view_model.begin_correlation_search()
        if not started:
            toast_warn(
                tr("Correlation search is already running."),
                title=tr("Charts"),
                tab_key="charts",
            )
            return

        self._correlation_total = max(0, len(ordered) - 1)
        self._correlation_target_name = feature.display_name()
        self.btn_find_correlations.setEnabled(False)
        set_progress(0)
        set_status_text(tr("Finding correlations..."))

        def _on_progress(percent: int) -> None:
            checked = 0
            bounded_percent = max(0, min(100, int(percent)))
            if self._correlation_total > 0:
                checked = int(round((bounded_percent / 100.0) * self._correlation_total))
            self._view_model.notify_correlation_search_progress(
                checked=checked,
                total=self._correlation_total,
                message=tr("Checking feature correlations"),
            )

        def _on_result(payload: dict) -> None:
            entries_payload = list(payload.get("entries") or [])
            result_entries: list[CorrelationEntry] = []
            for item in entries_payload:
                if not isinstance(item, dict):
                    continue
                selection = item.get("feature")
                if not isinstance(selection, FeatureSelection):
                    continue
                try:
                    corr = float(item.get("correlation"))
                except Exception:
                    continue
                result_entries.append(
                    CorrelationEntry(
                        feature=selection,
                        correlation=corr,
                    )
                )
            self._view_model.complete_correlation_search(
                CorrelationSearchResult(
                    target_feature=feature,
                    top10=result_entries[:10],
                )
            )

        started_search = self.data_selector.find_top_correlations(
            target_feature=feature,
            available_features=ordered,
            limit=10,
            on_result=_on_result,
            on_error=lambda message: self._view_model.fail_correlation_search(str(message)),
            on_progress=_on_progress,
            owner=self,
            key="charts_correlation_search",
            cancel_previous=True,
        )
        if not started_search:
            clear_progress()
            set_status_text(tr("Correlation analysis failed: unavailable."))
            toast_error(
                tr("Failed to start correlation analysis."),
                title=tr("Correlation failed"),
                tab_key="charts",
            )
            self._view_model.fail_correlation_search(tr("Correlation analysis is unavailable."))
            return

        toast_info(
            tr("Finding top feature correlations for {feature}...").format(feature=feature.display_name()),
            title=tr("Charts"),
            tab_key="charts",
        )

    def _on_correlation_search_progress(self, checked: int, total: int, _message: str) -> None:
        pct = 0
        total_int = int(total)
        checked_int = max(0, int(checked))
        if total_int > 0:
            pct = int(round((checked_int * 100) / total_int))
        pct = max(0, min(100, pct))
        set_progress(pct)

    def _on_correlation_search_done(self, _payload) -> None:
        self._correlation_total = 0
        self._correlation_target_name = ""
        self.btn_find_correlations.setEnabled(True)
        clear_progress()

