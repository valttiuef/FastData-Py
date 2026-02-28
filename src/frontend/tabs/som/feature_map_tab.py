
from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import pandas as pd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QResizeEvent, QShowEvent
from PySide6.QtWidgets import (

    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QSplitter,
)
from ...localization import tr

from ...widgets.panel import Panel
from ...widgets.fast_table import FastTable
from ...charts.group_chart import GroupBarChart

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from .viewmodel import SomViewModel


FEATURE_TABLE_COLUMNS: tuple[str, ...] = (
    "feature_id",
    "feature",
    "x",
    "y",
    "min",
    "mean",
    "max",
    "cluster",
)

FEATURE_TABLE_COLUMN_LABELS: dict[str, str] = {
    "feature_id": "Id",
    "feature": "Feature",
    "x": "X",
    "y": "Y",
    "min": "Min",
    "mean": "Mean",
    "max": "Max",
    "cluster": "Feature Cluster",
}


def _empty_feature_table_dataframe() -> pd.DataFrame:
    display_columns = [FEATURE_TABLE_COLUMN_LABELS.get(column, column) for column in FEATURE_TABLE_COLUMNS]
    return pd.DataFrame(columns=display_columns)


class FeatureMapTabWidget(QWidget):
    def __init__(
        self,
        view_model: Optional["SomViewModel"] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._splitter: Optional[QSplitter] = None
        self._splitter_initialized = False
        self._initial_table_layout_applied = False
        feat_layout = QVBoxLayout(self)

        # Primary content is a splitter so the user can resize the tables freely.
        splitter = QSplitter(Qt.Orientation.Vertical, self)

        feature_panel = Panel(tr("Feature placement on map"), parent=splitter)
        feature_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        feature_panel_layout = feature_panel.content_layout()
        self.feature_table = FastTable(
            parent=feature_panel,
            select="rows",
            single_selection=False,
            tint_current_selection=True,
            initial_uniform_column_widths=True,
            initial_uniform_column_count=len(FEATURE_TABLE_COLUMNS),
            sorting_enabled=True,
        )
        self.feature_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.feature_table.set_dataframe(_empty_feature_table_dataframe(), include_index=False)
        feature_panel_layout.addWidget(self.feature_table)
        splitter.addWidget(feature_panel)

        chart_panel = Panel(tr("Feature means by neuron cluster"), parent=splitter)
        chart_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        chart_panel_layout = chart_panel.content_layout()
        self.feature_group_chart = GroupBarChart(parent=chart_panel, y_label=tr("Mean value"))
        self.feature_group_chart.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        chart_panel_layout.addWidget(self.feature_group_chart)
        splitter.addWidget(chart_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        feat_layout.addWidget(splitter)
        self._splitter = splitter
        QTimer.singleShot(0, self._initialize_splitter_sizes)

        if view_model is not None:
            self.view_model = view_model

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if not self._splitter_initialized:
            self._splitter_initialized = self._initialize_splitter_sizes()

    def showEvent(self, event: QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._splitter_initialized:
            self._splitter_initialized = self._initialize_splitter_sizes()
        if self._splitter_initialized:
            self._apply_initial_table_layout_once()

    def _initialize_splitter_sizes(self) -> bool:
        splitter = self._splitter
        if splitter is None:
            return False
        height = splitter.size().height()
        if height <= 0:
            return False
        half = max(1, height // 2)
        try:
            splitter.setSizes([half, height - half])
        except Exception:
            return False
        return True

    def _apply_initial_table_layout_once(self) -> None:
        if self._initial_table_layout_applied:
            return
        self._initial_table_layout_applied = True
        try:
            self.feature_table.reapply_uniform_column_widths()
        except Exception:
            pass
