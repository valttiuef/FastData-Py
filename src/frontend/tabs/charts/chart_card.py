
from __future__ import annotations
from typing import List, Sequence, TYPE_CHECKING
import math

import pandas as pd

from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

import logging
logger = logging.getLogger(__name__)
from ...localization import tr

from ...widgets.multi_check_combo import MultiCheckCombo
from ...widgets.panel import Panel

from frontend.charts import GroupBarChart, MonthlyBarChart, TimeSeriesChart, ScatterChart, Scatter3DChart

if TYPE_CHECKING:
    from .viewmodel import ChartsViewModel



class ChartCard(Panel):
    """Encapsulates controls and rendering surface for a single chart."""

    correlation_feature_clicked = Signal(str)
    selection_requested = Signal(object)

    def __init__(self, parent=None, *, view_model: "ChartsViewModel | None" = None):
        super().__init__(title="", parent=parent)
        self._view_model: "ChartsViewModel | None" = None
        self.setObjectName("chartCard")
        self.setProperty("selectedTarget", False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = self.content_layout()

        controls = QHBoxLayout()
        controls.setSpacing(4)

        self.chart_type_combo = QComboBox(self)
        self.chart_type_combo.addItem(tr("Monthly rollup"), "monthly")
        self.chart_type_combo.addItem(tr("Time series"), "time_series")
        self.chart_type_combo.addItem(tr("Scatter"), "scatter")
        self.chart_type_combo.addItem(tr("Correlation ranking"), "correlation_bar")
        self.chart_type_combo.setCurrentIndex(2)

        self.features_combo = MultiCheckCombo(placeholder=tr("Select features"), summary_max=3)

        controls.addWidget(QLabel(tr("Chart:")))
        controls.addWidget(self.chart_type_combo, 0)
        controls.addWidget(QLabel(tr("Features:")))
        controls.addWidget(self.features_combo, 1)
        layout.addLayout(controls)

        self._stack = QStackedLayout()
        layout.addLayout(self._stack, 1)

        self._message_label = QLabel(tr("Select at least one feature"))
        self._message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message_label.setWordWrap(True)

        self._chart_container = QWidget()
        self._chart_layout = QVBoxLayout(self._chart_container)
        self._chart_layout.setSpacing(0)

        self._stack.addWidget(self._message_label)
        self._stack.addWidget(self._chart_container)
        self._stack.setCurrentWidget(self._message_label)

        self._chart_widget: QWidget | None = None
        self._chart_type: str = "monthly"
        self._feature_lookup: dict[tuple, object] = {}

        self.chart_type_combo.currentIndexChanged.connect(lambda _idx: self._emit_configuration_changed())
        self.features_combo.selection_changed.connect(self._emit_configuration_changed)
        self.installEventFilter(self)
        self.chart_type_combo.installEventFilter(self)
        self.features_combo.installEventFilter(self)
        self._message_label.installEventFilter(self)
        self._chart_container.installEventFilter(self)

        if view_model is not None:
            self.set_view_model(view_model)

    # ------------------------------------------------------------------
    def _emit_configuration_changed(self):
        if self._view_model is not None:
            self._view_model.notify_chart_configuration_changed(self)

    def set_view_model(self, view_model: "ChartsViewModel | None") -> None:
        self._view_model = view_model

    def set_selected_for_correlation_target(self, selected: bool) -> None:
        is_selected = bool(selected)
        if bool(self.property("selectedTarget")) == is_selected:
            return
        self.setProperty("selectedTarget", is_selected)
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)
        self.update()

    # ------------------------------------------------------------------
    def chart_type(self) -> str:
        return str(self.chart_type_combo.currentData())

    def selected_features(self) -> List:
        return list(self.features_combo.selected_values())

    def set_chart_type(self, chart_type: str, *, emit_changed: bool = True) -> None:
        idx = self.chart_type_combo.findData(chart_type)
        if idx < 0:
            return
        self.chart_type_combo.blockSignals(True)
        try:
            self.chart_type_combo.setCurrentIndex(idx)
        finally:
            self.chart_type_combo.blockSignals(False)
        if emit_changed:
            self._emit_configuration_changed()

    def set_selected_features(self, features: Sequence[object], *, emit_changed: bool = True) -> None:
        self.features_combo.blockSignals(True)
        try:
            self.features_combo.set_selected_values(list(features))
        finally:
            self.features_combo.blockSignals(False)
        if emit_changed:
            self._emit_configuration_changed()

    def set_available_features(self, items: Sequence[tuple[str, object]]) -> None:
        current_keys = {self._feature_identity(f) for f in self.selected_features()}
        self._feature_lookup = {self._feature_identity(val): val for _label, val in items}
        self.features_combo.set_items(items, check_all=False)
        if current_keys:
            restored = [self._feature_lookup[k] for k in current_keys if k in self._feature_lookup]
            if restored:
                self.features_combo.set_selected_values(restored)

    def ensure_default_selection(self) -> None:
        if not self.selected_features() and self._feature_lookup:
            values = list(self._feature_lookup.values())
            self.features_combo.set_selected_values(values[:2])

    # ------------------------------------------------------------------
    def show_message(self, text: str) -> None:
        self._message_label.setText(text)
        self._stack.setCurrentWidget(self._message_label)

    def clear_chart(self, message: str | None = None) -> None:
        if message is None:
            message = tr("Select at least one feature")
        self.show_message(message)
        widget = self._chart_widget
        if widget is None:
            return
        if isinstance(widget, (GroupBarChart, MonthlyBarChart, TimeSeriesChart, ScatterChart, Scatter3DChart)):
            try:
                widget.clear()
            except Exception:
                logger.warning("Exception in clear_chart", exc_info=True)

    def _ensure_chart_widget(self, chart_type: str) -> QWidget:
        if self._chart_widget is not None and self._chart_type == chart_type:
            return self._chart_widget

        # Remove existing widget
        if self._chart_widget is not None:
            self._chart_layout.removeWidget(self._chart_widget)
            self._chart_widget.deleteLater()
            self._chart_widget = None

        if chart_type == "time_series":
            self._chart_widget = TimeSeriesChart(title="", parent=self)
        elif chart_type == "monthly":
            self._chart_widget = MonthlyBarChart(title="", parent=self)
        elif chart_type == "correlation_bar":
            self._chart_widget = GroupBarChart(title=tr("Feature correlations"), parent=self, y_label=tr("Correlation"))
            if isinstance(self._chart_widget, GroupBarChart):
                self._chart_widget.category_selected.connect(self._on_group_bar_category_selected)
        elif chart_type == "scatter2d":
            self._chart_widget = ScatterChart(parent=self)
        elif chart_type == "scatter3d":
            self._chart_widget = Scatter3DChart(parent=self)
        else:
            placeholder = QLabel(tr("Scatter chart support coming soon"))
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._chart_widget = placeholder

        self._chart_layout.addWidget(self._chart_widget)
        self._chart_widget.installEventFilter(self)
        self._chart_type = chart_type
        return self._chart_widget

    def mousePressEvent(self, event) -> None:
        self.selection_requested.emit(self)
        super().mousePressEvent(event)

    def eventFilter(self, watched, event):
        event_type = event.type()
        if event_type in (QEvent.Type.MouseButtonPress, QEvent.Type.FocusIn):
            self.selection_requested.emit(self)
        return super().eventFilter(watched, event)

    def _on_group_bar_category_selected(self, category: str, _value: object) -> None:
        text = str(category or "").strip()
        if text:
            self.correlation_feature_clicked.emit(text)

    def set_monthly_frame(self, frame: pd.DataFrame, title: str) -> None:
        widget = self._ensure_chart_widget("monthly")
        if isinstance(widget, MonthlyBarChart):
            if frame is None or frame.empty:
                widget.clear()
                self.show_message(tr("No data for the selected filters"))
                return
            widget.set_frame(frame)
            self._stack.setCurrentWidget(self._chart_container)
        else:
            self.show_message(tr("No renderer available"))

    def set_time_series_frame(self, frame: pd.DataFrame, title: str) -> None:
        widget = self._ensure_chart_widget("time_series")
        if isinstance(widget, TimeSeriesChart):
            if frame is None or frame.empty:
                widget.clear()
                self.show_message(tr("No data for the selected filters"))
                return
            widget.set_dataframe(frame)
            self._stack.setCurrentWidget(self._chart_container)
        else:
            self.show_message(tr("No renderer available"))

    def set_group_bar_data(
        self,
        categories: Sequence[str],
        values: Sequence[float],
        *,
        title: str,
    ) -> None:
        widget = self._ensure_chart_widget("correlation_bar")
        if isinstance(widget, GroupBarChart):
            if not categories or not values:
                widget.clear()
                self.show_message(tr("No correlation data available"))
                return
            widget.set_title(title)
            widget.set_y_label(tr("Correlation"))
            widget.set_data([str(item) for item in categories], [float(item) for item in values], series_name=tr("Correlation"))
            self._stack.setCurrentWidget(self._chart_container)
        else:
            self.show_message(tr("No renderer available"))

    def set_scatter_frame(
        self,
        frame: pd.DataFrame | None,
        *,
        columns: Sequence[str],
        labels: Sequence[str],
    ) -> None:
        if frame is None or frame.empty:
            self.show_message(tr("No data for the selected filters"))
            return

        clean_columns = [c for c in columns if c in frame.columns]
        axis_labels = list(labels)[: len(clean_columns)]

        if len(clean_columns) not in (2, 3):
            self.show_message(tr("Scatter charts require 2 or 3 features"))
            return

        data = frame.loc[:, clean_columns].copy()
        for col in clean_columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=clean_columns)
        if data.empty:
            self.show_message(tr("No scatter data available"))
            return

        corr_text = tr("Correlation: n/a")
        if len(clean_columns) >= 2:
            try:
                corr = float(data[clean_columns[:2]].corr().iat[0, 1])
                if math.isfinite(corr):
                    corr_text = tr("Correlation: {value}").format(value=f"{corr:.3f}")
            except Exception:
                logger.warning("Exception in set_scatter_frame", exc_info=True)

        if len(clean_columns) == 2:
            widget = self._ensure_chart_widget("scatter2d")
            if isinstance(widget, ScatterChart):
                try:
                    widget.chart.setTitle(tr("Scatter ({corr})").format(corr=corr_text))
                except Exception:
                    logger.warning("Exception in set_scatter_frame", exc_info=True)
                while len(axis_labels) < 2:
                    axis_labels.append(tr("Feature"))
                widget.set_axis_labels(axis_labels[0], axis_labels[1])
                widget.set_points(
                    data,
                    x_column=clean_columns[0],
                    y_column=clean_columns[1],
                    group_column=None,
                    show_identity_line=False,
                    show_legend=False,
                )
                self._stack.setCurrentWidget(self._chart_container)
                return
        elif len(clean_columns) == 3:
            widget = self._ensure_chart_widget("scatter3d")
            if isinstance(widget, Scatter3DChart):
                while len(axis_labels) < 3:
                    axis_labels.append(tr("Feature"))
                widget.set_axis_labels(axis_labels)
                widget.set_points(data, clean_columns)
                self._stack.setCurrentWidget(self._chart_container)
                return

        self.show_message(tr("No renderer available"))

    def show_scatter_placeholder(self) -> None:
        widget = self._ensure_chart_widget("scatter_placeholder")
        if isinstance(widget, QLabel):
            widget.setText(tr("Scatter chart support coming soon"))
        self._stack.setCurrentWidget(self._chart_container)

    # ------------------------------------------------------------------
    @staticmethod
    def _feature_identity(feature) -> tuple:
        try:
            return feature.identity_key()
        except AttributeError:
            return tuple()
