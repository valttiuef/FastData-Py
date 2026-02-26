
from __future__ import annotations
from PySide6.QtWidgets import QAbstractItemView, QGroupBox, QPushButton, QSizePolicy, QVBoxLayout
from ...localization import tr

from ...models.hybrid_pandas_model import FeatureSelection
from ...utils import toast_info
from ...widgets.data_selector_widget import DataSelectorWidget
from ...widgets.sidebar_widget import SidebarWidget
from .viewmodel import ChartsViewModel
import logging

logger = logging.getLogger(__name__)


class ChartsSidebar(SidebarWidget):
    """Sidebar container exposing chart actions and shared data selector controls."""

    def __init__(self, view_model: ChartsViewModel, parent=None):
        super().__init__(title=tr("Charts"), parent=parent)
        self._view_model = view_model

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
        payloads = [self._feature_payload(item) for item in ordered]
        data_frame = self.data_selector.fetch_base_dataframe_for_features(payloads)
        if data_frame is None or data_frame.empty:
            toast_info(tr("No data for the selected filters."), title=tr("Charts"), tab_key="charts")
            return

        started = self._view_model.start_correlation_search(
            target_feature=feature,
            available_features=candidates,
            data_frame=data_frame,
        )
        if not started:
            toast_info(
                tr("Correlation search is already running."),
                title=tr("Charts"),
                tab_key="charts",
            )
            return

        toast_info(
            tr("Finding top feature correlations for {feature}...").format(feature=feature.display_name()),
            title=tr("Charts"),
            tab_key="charts",
        )

    @staticmethod
    def _feature_payload(selection: FeatureSelection) -> dict:
        return {
            "feature_id": selection.feature_id,
            "notes": selection.label,
            "name": selection.base_name,
            "source": selection.source,
            "unit": selection.unit,
            "type": selection.type,
            "lag_seconds": selection.lag_seconds,
        }
