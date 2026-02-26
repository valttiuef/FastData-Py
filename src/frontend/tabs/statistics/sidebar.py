
from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (


    QComboBox,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
    QPushButton,
)

import logging
logger = logging.getLogger(__name__)
from ...localization import tr


from ...widgets.help_widgets import InfoButton
from ...widgets.data_selector_widget import DataSelectorWidget
from ...widgets.sidebar_widget import SidebarWidget
from ...widgets.multi_check_combo import MultiCheckCombo
from ...models.log_model import LogModel, get_log_model
from ...utils import toast_error, toast_info, toast_warn
from ...viewmodels.help_viewmodel import HelpViewModel

if TYPE_CHECKING:
    from .viewmodel import StatisticsViewModel



class StatisticsSidebar(SidebarWidget):
    """Container for the statistics tab sidebar controls."""

    export_requested = Signal()
    save_requested = Signal()

    def __init__(
        self,
        view_model: "StatisticsViewModel",
        *,
        parent: Optional[QWidget] = None,
        log_model: Optional[LogModel] = None,
        help_viewmodel: Optional[HelpViewModel] = None,
    ) -> None:
        super().__init__(title=tr("Statistics"), parent=parent)

        self._view_model = view_model
        self._log_model = log_model or get_log_model(self)
        self._help_viewmodel = help_viewmodel
        if self._view_model is None:
            logger.warning("StatisticsSidebar initialised without view_model.")
        if self._log_model is None:
            logger.warning("StatisticsSidebar initialised without log_model.")
        if self._help_viewmodel is None:
            logger.warning("StatisticsSidebar initialised without help_viewmodel.")

        controls_layout = self.content_layout()

        actions_group = QGroupBox(tr("Actions"), self)
        actions_layout = QVBoxLayout(actions_group)
        self._run_button = QPushButton(tr("Gather statistics"), actions_group)
        self._save_button = QPushButton(tr("Save statistics"), actions_group)
        self._export_button = QPushButton(tr("Export results..."), actions_group)
        self._save_button.setEnabled(False)
        self._export_button.setEnabled(False)

        actions_header = QHBoxLayout()
        actions_header.addWidget(self._run_button, 1)
        actions_layout.addLayout(actions_header)
        actions_layout.addWidget(self._save_button)
        actions_layout.addWidget(self._export_button)
        self.set_sticky_actions(actions_group)

        self.data_selector = DataSelectorWidget(
            title=tr("Data selection"),
            parent=self,
            data_model=view_model.data_model,
            help_viewmodel=self._help_viewmodel,
        )
        try:
            self.data_selector.features_widget.set_use_selection_filter(True)
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)
        controls_layout.addWidget(self.data_selector, 1)

        stats_group = QGroupBox(tr("Statistics"), self)
        stats_layout = QVBoxLayout(stats_group)
        self._stats_combo = MultiCheckCombo(stats_group, placeholder=tr("Select statistics"), summary_max=3)
        stat_items = view_model.available_statistics()
        combo_items = [(tr(label), key) for key, label in stat_items]
        self._stats_combo.set_items(combo_items, check_all=False)
        defaults = {"avg"}
        default_keys = [key for key, _ in stat_items if key in defaults]
        if default_keys:
            self._stats_combo.set_selected_values(default_keys)
        self._stats_combo.set_summary_formatter(
            lambda labels, checked, total: labels[0]
            if checked == 1 and labels
            else ", ".join(labels) + ("…" if checked > len(labels) else "")
        )
        stats_row = QHBoxLayout()
        stats_row.setContentsMargins(0, 0, 0, 0)
        stats_row.setSpacing(6)
        stats_row.addWidget(self._stats_combo, 1)
        stats_info = self._make_info("controls.statistics.metrics")
        if stats_info:
            stats_row.addWidget(stats_info, 0, Qt.AlignmentFlag.AlignRight)
        stats_layout.addLayout(stats_row)
        controls_layout.addWidget(stats_group)

        mode_group = QGroupBox(tr("Mode"), self)
        self._mode_form = QFormLayout(mode_group)
        self._mode_combo = QComboBox(mode_group)
        self._mode_combo.addItem(tr("Time based"), "time")
        self._mode_combo.addItem(tr("Group by column"), "column")
        self._mode_form.addRow(tr("Aggregation:"), self._wrap_with_help(self._mode_combo, "controls.statistics.mode"))

        self._column_combo = QComboBox(mode_group)
        for key, label in view_model.available_group_columns():
            self._column_combo.addItem(label, key)
        self._column_row = self._wrap_with_help(self._column_combo, "controls.statistics.group_column")
        self._mode_form.addRow(tr("Group column:"), self._column_row)

        self._separate_timeframes_checkbox = QCheckBox(tr("Group by separate timeframes"), mode_group)
        self._separate_timeframes_row = self._wrap_with_help(
            self._separate_timeframes_checkbox, "controls.statistics.separate_timeframes"
        )
        self._mode_form.addRow(self._separate_timeframes_row)


        # Statistics period controls for time-based mode
        self._stats_period_combo = QComboBox(mode_group)
        self._stats_period_combo.addItem(tr("hourly"), "hourly")
        self._stats_period_combo.addItem(tr("daily"), "daily")
        self._stats_period_combo.addItem(tr("weekly"), "weekly")
        self._stats_period_combo.addItem(tr("monthly"), "monthly")
        self._stats_period_combo.addItem(tr("custom"), "custom")
        self._stats_period_combo.setCurrentIndex(1)
        self._stats_period_row = self._wrap_with_help(self._stats_period_combo, "controls.statistics.period")
        self._mode_form.addRow(
            tr("Statistics period:"),
            self._stats_period_row,
        )

        self._stats_period_edit = QLineEdit(mode_group)
        self._stats_period_edit.setPlaceholderText(tr("e.g. 3600 (seconds)"))
        self._mode_form.addRow(tr("Custom period (s):"), self._stats_period_edit)

        controls_layout.addWidget(mode_group)

        self._connect_signals()

    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self._view_model.run_enabled_changed.connect(self._run_button.setEnabled)
        self._view_model.save_enabled_changed.connect(self._save_button.setEnabled)

        self._view_model.group_columns_changed.connect(self._on_group_columns_changed)
        self._run_button.clicked.connect(self._on_run_clicked)
        self._save_button.clicked.connect(self.save_requested.emit)
        self._export_button.clicked.connect(self.export_requested.emit)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._column_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._stats_period_combo.currentIndexChanged.connect(self._on_stats_period_changed)

        self._run_button.setEnabled(self._view_model.database() is not None)
        self._save_button.setEnabled(self._view_model.has_savable_result())
        self._refresh_column_combo()
        self._on_mode_changed()
        self._on_stats_period_changed()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _selected_feature_payloads(self) -> list[dict]:
        return self.data_selector.features_widget.selected_payloads()

    def _selected_statistics(self) -> list[str]:
        values = [str(v) for v in self._stats_combo.selected_values() if v]
        return values

    # ------------------------------------------------------------------
    def _on_run_clicked(self) -> None:
        payloads = self._selected_feature_payloads()
        if not payloads:
            self._inform_user(tr("Select one or more features to gather statistics."))
            return
        stats = self._selected_statistics()
        if not stats:
            self._inform_user(tr("Select at least one statistic to compute."))
            return
        mode = str(self.mode_combo.currentData() or "time")
        group_column = None
        if mode == "column":
            group_column = self.column_combo.currentData()
            if not group_column:
                self._inform_user(tr("Choose a column to group by when using column mode."))
                return

        filters = self.data_selector.build_data_filters()
        if filters is None:
            self._inform_user(tr("Failed to build filters for selected inputs."), level=logging.ERROR)
            return
        systems = filters.systems or None
        Datasets = filters.datasets or None
        group_ids = filters.group_ids or None
        months = filters.months or None
        start = filters.start
        end = filters.end
        preprocessing = self.data_selector.preprocessing_widget.parameters()

        # Get statistics period settings
        stats_period = self._get_stats_period()
        preprocessing["stats_period"] = stats_period
        preprocessing["separate_timeframes"] = bool(self._separate_timeframes_checkbox.isChecked())
        stats_preprocessing = dict(preprocessing)
        stats_preprocessing.pop("stats_period", None)
        stats_preprocessing.pop("separate_timeframes", None)

        data_frame = self.data_selector.fetch_base_dataframe_for_features(
            payloads,
            preprocessing_override=stats_preprocessing,
        )
        if data_frame is None:
            self._inform_user(tr("Failed to load data for selected inputs."), level=logging.ERROR)
            return

        self._view_model.run_statistics(
            data_frame=data_frame,
            feature_payloads=payloads,
            systems=systems,
            datasets=Datasets,
            group_ids=group_ids,
            start=start,
            end=end,
            months=months,
            statistics=stats,
            mode=mode,
            group_column=group_column,
            preprocessing=preprocessing,
        )

    def _get_stats_period(self) -> str | int | None:
        """Get the statistics period from the UI controls."""
        period_key = self._stats_period_combo.currentData()
        if period_key == "custom":
            custom_text = self._stats_period_edit.text().strip()
            if custom_text:
                try:
                    return int(custom_text)
                except ValueError:
                    return custom_text
            return None
        # Map period names to seconds
        period_map = {
            "hourly": 3600,
            "daily": 86400,
            "weekly": 604800,
            "monthly": 2592000,  # Approximation: 30 days (months vary 28-31 days)
        }
        return period_map.get(str(period_key))

    # ------------------------------------------------------------------
    def _on_group_columns_changed(self, columns: object) -> None:
        items = columns if isinstance(columns, list) else None
        self._refresh_column_combo(items)

    def _refresh_column_combo(self, items: Optional[list[tuple[str, str]]] = None) -> None:
        """Refresh the column combo with available group columns."""
        current_selection = self._column_combo.currentData()
        was_blocked = self._column_combo.blockSignals(True)
        self._column_combo.clear()
        for key, label in (items if items is not None else self._view_model.available_group_columns()):
            self._column_combo.addItem(label, key)
        # Try to restore previous selection
        for i in range(self._column_combo.count()):
            if self._column_combo.itemData(i) == current_selection:
                self._column_combo.setCurrentIndex(i)
                break
        self._column_combo.blockSignals(was_blocked)

    def _on_mode_changed(self) -> None:
        mode = self._mode_combo.currentData()
        is_column_mode = mode == "column"
        self._set_form_row_visible(self._column_row, is_column_mode)

        selected_group_column = self._column_combo.currentData()
        group_kind_selected = (
            isinstance(selected_group_column, str) and selected_group_column.startswith("group:")
        )
        self._set_form_row_visible(self._separate_timeframes_row, is_column_mode and group_kind_selected)

        # Show/hide statistics period controls based on mode
        is_time_mode = mode == "time"
        self._set_form_row_visible(self._stats_period_row, is_time_mode)
        self._set_form_row_visible(
            self._stats_period_edit,
            is_time_mode and self._stats_period_combo.currentData() == "custom",
        )

    def _on_stats_period_changed(self) -> None:
        """Show/hide custom period input based on combo selection."""
        is_custom = self._stats_period_combo.currentData() == "custom"
        mode = self._mode_combo.currentData()
        self._set_form_row_visible(self._stats_period_edit, is_custom and mode == "time")

    def _set_form_row_visible(self, field_widget: QWidget, visible: bool) -> None:
        field_widget.setVisible(visible)
        label = self._mode_form.labelForField(field_widget)
        if label is not None:
            label.setVisible(visible)

    def _inform_user(self, message: str, *, level: int = logging.INFO) -> None:
        try:
            if level >= logging.ERROR:
                toast_error(message, title=tr("Statistics"), tab_key="statistics")
            elif level >= logging.WARNING:
                toast_warn(message, title=tr("Statistics"), tab_key="statistics")
            else:
                toast_info(message, title=tr("Statistics"), tab_key="statistics")
        except Exception:
            logger.warning("Exception in _inform_user", exc_info=True)
        try:
            self._view_model.status_changed.emit(message)
        except Exception:
            logger.warning("Exception in _inform_user", exc_info=True)

    def _make_info(self, help_key: str | None) -> QWidget | None:
        if help_key and self._help_viewmodel is not None:
            return InfoButton(help_key, self._help_viewmodel)
        return None

    def _wrap_with_help(self, widget: QWidget, help_key: str | None) -> QWidget:
        info = self._make_info(help_key)
        if info is None:
            return widget

        container = QWidget(widget.parent())
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(widget, 1)
        layout.addWidget(info, 0, Qt.AlignmentFlag.AlignRight)
        return container

    # ------------------------------------------------------------------
    @property
    def mode_combo(self) -> QComboBox:
        return self._mode_combo

    def set_export_enabled(self, enabled: bool) -> None:
        self._export_button.setEnabled(bool(enabled))

    @property
    def column_combo(self) -> QComboBox:
        return self._column_combo

    @property
    def stats_combo(self) -> MultiCheckCombo:
        return self._stats_combo

    @property
    def run_button(self) -> QPushButton:
        return self._run_button

    @property
    def save_button(self) -> QPushButton:
        return self._save_button


__all__ = ["StatisticsSidebar"]
