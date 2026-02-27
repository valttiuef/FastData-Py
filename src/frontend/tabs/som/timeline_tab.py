
from __future__ import annotations
# @ai(gpt-5, codex, refactor, 2026-02-26)
from typing import Any, Optional, TYPE_CHECKING

import pandas as pd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QResizeEvent, QShowEvent
from PySide6.QtWidgets import (

    QLabel,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QSplitter,
    QStackedWidget,
    QSizePolicy,
)
from ...localization import tr

from frontend.charts import TimeSeriesChart
from ...widgets.multi_check_combo import MultiCheckCombo
from .map_view import SomMapView
from ...widgets.fast_table import FastTable
from ...widgets.help_widgets import InfoButton
from ...viewmodels.help_viewmodel import get_help_viewmodel

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from .viewmodel import SomViewModel


TIMELINE_TABLE_COLUMNS: tuple[str, ...] = ("index", "bmu_x", "bmu_y", "bmu", "cluster")


def empty_timeline_table_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=list(TIMELINE_TABLE_COLUMNS))


def normalize_timeline_table_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_timeline_table_dataframe()
    out = df.copy()
    for column in TIMELINE_TABLE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    ordered = list(TIMELINE_TABLE_COLUMNS) + [c for c in out.columns if c not in TIMELINE_TABLE_COLUMNS]
    return out.loc[:, ordered]


def set_timeline_table_dataframe(table: Any, df: Optional[pd.DataFrame]) -> pd.DataFrame:
    normalized = normalize_timeline_table_dataframe(df)
    if table is None:
        return normalized
    model = table.model()
    if model is None:
        return normalized

    previous_widths = [int(table.columnWidth(col)) for col in range(model.columnCount())]
    table.set_dataframe(normalized, include_index=False)

    if previous_widths and len(previous_widths) == model.columnCount():
        for col, width in enumerate(previous_widths):
            if width > 0:
                table.setColumnWidth(col, width)
    return normalized


class TimelineTabWidget(QWidget):
    def __init__(
        self,
        view_model: Optional["SomViewModel"] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._splitter: Optional[QSplitter] = None
        self._rows_splitter: Optional[QSplitter] = None
        self._splitters_initialized = False
        self._initial_table_layout_applied = False
        self._view_model = view_model

        help_viewmodel = get_help_viewmodel()
        time_layout = QVBoxLayout(self)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setSpacing(8)

        def _title_row(text: str, help_key: str) -> QWidget:
            container = QWidget(self)
            layout = QHBoxLayout(container)
            layout.setContentsMargins(4, 0, 4, 0)
            layout.setSpacing(6)
            label = QLabel(text, container)
            layout.addWidget(label)
            if help_viewmodel is not None:
                layout.addWidget(InfoButton(help_key, help_viewmodel, parent=container))
            layout.addStretch(1)
            container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return container

        time_layout.addWidget(_title_row(tr("Cluster timeline"), "controls.som.timeline.chart"))
        time_layout.addSpacing(4)

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel(tr("Display:")))
        self.timeline_display_combo = MultiCheckCombo(
            parent=self,
            placeholder=tr("Select timeline layers"),
            summary_max=3,
        )
        self.timeline_display_combo.setMinimumWidth(240)
        self.timeline_display_combo.set_items(
            [
                (tr("BMU"), "bmu"),
                (tr("Neuron clusters"), "cluster"),
                (tr("Selected features"), "selected_features"),
            ],
            check_all=False,
        )
        self.timeline_display_combo.set_selected_values(["bmu"])
        self.timeline_display_combo.set_summary_formatter(
            lambda labels, checked, _total: (
                tr("None")
                if checked == 0
                else ", ".join(labels)
            )
        )
        controls_row.addWidget(self.timeline_display_combo)
        if help_viewmodel is not None:
            controls_row.addWidget(InfoButton("controls.som.timeline.display", help_viewmodel, parent=self))
        controls_row.addStretch(1)
        time_layout.addLayout(controls_row)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setChildrenCollapsible(False)

        self.timeline_chart = TimeSeriesChart(title="")
        self.timeline_chart.chart.legend().setVisible(True)
        splitter.addWidget(self.timeline_chart)

        rows_panel = QWidget()
        rows_layout = QVBoxLayout(rows_panel)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(6)

        self.timeline_rows_splitter = QSplitter(Qt.Orientation.Horizontal, rows_panel)
        self.timeline_rows_splitter.setChildrenCollapsible(False)

        table_panel = QWidget(rows_panel)
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(6)
        table_layout.addWidget(_title_row(tr("Data table"), "controls.som.timeline.data"))
        self.timeline_table = FastTable(
            parent=table_panel,
            select="rows",
            single_selection=False,
            tint_current_selection=True,
            initial_uniform_column_widths=True,
            initial_uniform_column_count=len(TIMELINE_TABLE_COLUMNS),
            sorting_enabled=True,
        )
        self.timeline_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.timeline_table.set_dataframe(empty_timeline_table_dataframe(), include_index=False)
        table_layout.addWidget(self.timeline_table, 1)
        if self._view_model is not None:
            try:
                self._view_model.timeline_table_dataframe_changed.connect(
                    lambda df: set_timeline_table_dataframe(self.timeline_table, df)
                )
                set_timeline_table_dataframe(self.timeline_table, self._view_model.timeline_table_dataframe())
            except Exception:
                pass
        self.timeline_rows_splitter.addWidget(table_panel)

        cluster_panel = QWidget(rows_panel)
        cluster_layout = QVBoxLayout(cluster_panel)
        cluster_layout.setContentsMargins(0, 0, 0, 0)
        cluster_layout.setSpacing(6)
        cluster_layout.addWidget(_title_row(tr("Cluster map"), "controls.som.timeline.cluster_map"))

        self.timeline_cluster_panel_stack = QStackedWidget(cluster_panel)

        self.timeline_cluster_placeholder = QLabel(
            tr("No neuron clusters yet.\nUse \"Cluster neurons\" in the sidebar to build a cluster map."),
            cluster_panel,
        )
        self.timeline_cluster_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timeline_cluster_panel_stack.addWidget(self.timeline_cluster_placeholder)

        self.timeline_cluster_content = QWidget(cluster_panel)
        cluster_content_layout = QVBoxLayout(self.timeline_cluster_content)
        self.timeline_cluster_map = SomMapView()
        cluster_content_layout.addWidget(self.timeline_cluster_map, 1)

        self.timeline_cluster_panel_stack.addWidget(self.timeline_cluster_content)
        cluster_layout.addWidget(self.timeline_cluster_panel_stack, 1)
        self.timeline_rows_splitter.addWidget(cluster_panel)

        self.timeline_rows_splitter.setStretchFactor(0, 1)
        self.timeline_rows_splitter.setStretchFactor(1, 1)
        rows_layout.addWidget(self.timeline_rows_splitter)
        splitter.addWidget(rows_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        time_layout.addWidget(splitter)
        self.set_splitters(splitter, self.timeline_rows_splitter)

    def set_splitters(self, splitter: QSplitter, rows_splitter: QSplitter) -> None:
        self._splitter = splitter
        self._rows_splitter = rows_splitter

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if not self._splitters_initialized:
            self._splitters_initialized = self._initialize_splitter_sizes()
        elif event.size() != event.oldSize():
            self._rebalance_splitters()

    def showEvent(self, event: QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._splitters_initialized:
            self._splitters_initialized = self._initialize_splitter_sizes()
        if self._splitters_initialized:
            self._rebalance_splitters()
            self._apply_initial_table_layout_once()

    def _rebalance_splitters(self) -> None:
        splitter = self._splitter
        rows_splitter = self._rows_splitter
        if splitter is None or rows_splitter is None:
            return

        main_height = splitter.size().height()
        rows_width = rows_splitter.size().width()

        if main_height > 0:
            half_height = max(1, main_height // 2)
            splitter.setSizes([half_height, main_height - half_height])

        if rows_width > 0:
            half_width = max(1, rows_width // 2)
            rows_splitter.setSizes([half_width, rows_width - half_width])

    def _initialize_splitter_sizes(self) -> bool:
        splitter = self._splitter
        rows_splitter = self._rows_splitter
        if splitter is None or rows_splitter is None:
            return False

        main_height = splitter.size().height()
        rows_width = rows_splitter.size().width()

        if main_height <= 0 or rows_width <= 0:
            return False

        half_height = max(1, main_height // 2)
        splitter.setSizes([half_height, main_height - half_height])

        half_width = max(1, rows_width // 2)
        rows_splitter.setSizes([half_width, rows_width - half_width])

        return True

    def _apply_initial_table_layout_once(self) -> None:
        if self._initial_table_layout_applied:
            return
        self._initial_table_layout_applied = True
        try:
            self.timeline_table.reapply_uniform_column_widths()
        except Exception:
            pass


