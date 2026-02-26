
from __future__ import annotations
from html import escape
import re
from typing import Callable, Optional, TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (


    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)

import logging
logger = logging.getLogger(__name__)
from ...localization import tr

if TYPE_CHECKING:
    from .viewmodel import RegressionViewModel

from ...widgets.collapsible_section import CollapsibleSection
from ...widgets.data_selector_widget import DataSelectorWidget
from ...widgets.help_widgets import InfoButton
from ...widgets.multi_check_combo import MultiCheckCombo
from ...widgets.sidebar_widget import SidebarWidget
from ...viewmodels.help_viewmodel import HelpViewModel, get_help_viewmodel
from ...param_specs.regression import (
    REGRESSION_MODEL_PARAM_SPECS,
    REGRESSION_SELECTOR_PARAM_SPECS,
    REGRESSION_REDUCER_PARAM_SPECS,
    REGRESSION_REDUCER_DEFAULTS,
)
from backend.services.modeling_shared import display_name



class RegressionSidebar(SidebarWidget):
    """Sidebar housing feature selection and regression options."""

    export_requested = Signal()
    save_predictions_requested = Signal()
    save_models_requested = Signal()
    delete_models_requested = Signal()

    def __init__(
        self,
        *,
        view_model: "RegressionViewModel",
        help_viewmodel: Optional[HelpViewModel] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(title=tr("Regression"), parent=parent)

        self._view_model = view_model
        resolved_help = help_viewmodel
        if resolved_help is None:
            try:
                resolved_help = get_help_viewmodel()
            except Exception:
                resolved_help = None
        self._help_viewmodel = resolved_help
        if self._view_model is None:
            logger.warning("RegressionSidebar initialised without view_model.")
        self._register_help_context()

        self._selector_items: list[tuple[str, str, dict[str, object]]] = []
        self._model_items: list[tuple[str, str, dict[str, object]]] = []
        self._selector_param_getters: dict[str, dict[str, Callable[[], object]]] = {}
        self._model_param_getters: dict[str, dict[str, Callable[[], object]]] = {}
        self._selector_param_groups: dict[str, QGroupBox] = {}
        self._model_param_groups: dict[str, QGroupBox] = {}
        self._reducer_items: list[tuple[str, str, dict[str, object]]] = []
        self._reducer_param_getters: dict[str, dict[str, Callable[[], object]]] = {}
        self._reducer_param_groups: dict[str, QGroupBox] = {}
        self._feature_payload_lookup: dict[int, dict] = {}
        self._feature_items_signature: Optional[tuple[tuple[str, int], ...]] = None
        self._selected_target_id: Optional[int] = None
        self._selected_target_ids: list[int] = []
        self._current_stratify_payload: Optional[dict] = None
        self._user_selected_stratify = False
        self._syncing_stratify = False

        controls_layout = self.content_layout()

        actions_group = QGroupBox(tr("Actions"), self)
        actions_layout = QVBoxLayout(actions_group)
        self.run_button = QPushButton(tr("Run regression"), actions_group)
        self.save_models_button = QPushButton(tr("Save selected models"), actions_group)
        self.delete_models_button = QPushButton(tr("Delete selected models"), actions_group)
        self.save_predictions_button = QPushButton(tr("Save predictions"), actions_group)
        self.export_button = QPushButton(tr("Export results..."), actions_group)
        self.export_button.setEnabled(False)
        self.auto_save_checkbox = QCheckBox(tr("Auto-save models to database"), actions_group)
        actions_layout.addWidget(self.run_button)
        actions_layout.addWidget(self.save_models_button)
        actions_layout.addWidget(self.delete_models_button)
        actions_layout.addWidget(self.save_predictions_button)
        actions_layout.addWidget(self.export_button)
        actions_layout.addWidget(self.auto_save_checkbox)
        self.set_sticky_actions(actions_group)

        self.export_button.clicked.connect(self.export_requested.emit)
        self.save_predictions_button.clicked.connect(self.save_predictions_requested.emit)
        self.save_models_button.clicked.connect(self.save_models_requested.emit)
        self.delete_models_button.clicked.connect(self.delete_models_requested.emit)

        self.data_selector = DataSelectorWidget(
            parent=self,
            data_model=view_model.data_model,
            help_viewmodel=self._help_viewmodel,
        )
        controls_layout.addWidget(self.data_selector, 1)

        self.features_widget = self.data_selector.features_widget

        target_group = QGroupBox(tr("Target feature"), self)
        target_layout = QVBoxLayout(target_group)
        target_placeholder = tr("Select target feature")
        self.target_combo = MultiCheckCombo(target_group, placeholder=target_placeholder, summary_max=2)
        self.target_combo.set_max_checked(None)
        self.target_combo.set_summary_formatter(
            lambda labels, checked, total, placeholder=target_placeholder: (
                placeholder if checked == 0
                else labels[0] if checked == 1
                else tr("{count} targets").format(count=checked)
            )
        )
        target_layout.addWidget(self._with_info(self.target_combo, "controls.regression.target"))
        controls_layout.addWidget(target_group)

        selectors_group = QGroupBox(tr("Feature selection"), self)
        selectors_layout = QVBoxLayout(selectors_group)

        self.selector_combo = MultiCheckCombo(
            selectors_group, placeholder=tr("Select feature selection"), summary_max=3
        )
        selectors_layout.addWidget(
            self._with_info(self.selector_combo, "controls.regression.feature_selection")
        )

        controls_layout.addWidget(selectors_group)

        reducer_group = QGroupBox(tr("Dimensionality reduction"), self)
        reducer_layout = QVBoxLayout(reducer_group)
        self.reducer_combo = MultiCheckCombo(
            reducer_group,
            placeholder=tr("Disabled"),
            summary_max=2,
        )
        self.reducer_combo.set_summary_formatter(
            lambda labels, checked, _total: (
                tr("Disabled")
                if checked == 0
                else labels[0] if checked == 1
                else tr("{count} methods").format(count=checked)
            )
        )
        reducer_layout.addWidget(
            self._with_info(self.reducer_combo, "controls.regression.dimensionality_reduction")
        )
        controls_layout.addWidget(reducer_group)

        models_group = QGroupBox(tr("Models"), self)
        models_layout = QVBoxLayout(models_group)

        self.model_combo = MultiCheckCombo(models_group, placeholder=tr("Select models"), summary_max=3)
        models_layout.addWidget(self._with_info(self.model_combo, "controls.regression.models"))

        controls_layout.addWidget(models_group)

        self.cv_section = CollapsibleSection(tr("Cross-validation"), collapsed=True, parent=self)
        cv_form = QFormLayout()
        self.cv_combo = QComboBox(self.cv_section)
        self.cv_strategy_label = QLabel(tr("Strategy:"), self.cv_section)
        self._cv_strategy_row = self._with_info(self.cv_combo, "controls.regression.cv.strategy")
        cv_form.addRow(self.cv_strategy_label, self._cv_strategy_row)
        self.cv_folds = QSpinBox(self.cv_section)
        self.cv_folds.setRange(2, 20)
        self.cv_folds.setValue(5)
        self.cv_folds_label = QLabel(tr("Folds:"), self.cv_section)
        self._cv_folds_row = self._with_info(self.cv_folds, "controls.regression.cv.folds")
        cv_form.addRow(self.cv_folds_label, self._cv_folds_row)
        self.cv_group_combo = QComboBox(self.cv_section)
        self.cv_group_label = QLabel(tr("Group:"), self.cv_section)
        self._cv_group_row = self._with_info(self.cv_group_combo, "controls.regression.cv.group")
        cv_form.addRow(self.cv_group_label, self._cv_group_row)
        self.cv_shuffle = QCheckBox(tr("Shuffle folds"), self.cv_section)
        self.cv_shuffle.setChecked(True)
        self._cv_shuffle_row = self._with_info(self.cv_shuffle, "controls.regression.cv.shuffle")
        cv_form.addRow("", self._cv_shuffle_row)
        self.cv_gap = QSpinBox(self.cv_section)
        self.cv_gap.setRange(0, 5000)
        self.cv_gap.setValue(0)
        self.cv_gap.setEnabled(False)
        self.cv_gap_label = QLabel(tr("Time gap:"), self.cv_section)
        self._cv_gap_row = self._with_info(self.cv_gap, "controls.regression.cv.time_gap")
        cv_form.addRow(self.cv_gap_label, self._cv_gap_row)
        self.cv_stratify_combo = QComboBox(self.cv_section)
        self.cv_stratify_label = QLabel(tr("Stratify by:"), self.cv_section)
        self._cv_stratify_row = self._with_info(self.cv_stratify_combo, "controls.regression.cv.stratify")
        cv_form.addRow(self.cv_stratify_label, self._cv_stratify_row)
        self.cv_section.body_layout().addLayout(cv_form)
        controls_layout.addWidget(self.cv_section)

        self.test_section = CollapsibleSection(tr("Test split"), collapsed=True, parent=self)
        test_form = QFormLayout()
        self.enable_test_split = QCheckBox(tr("Create test set"), self.test_section)
        self.enable_test_split.setChecked(True)
        self._test_enable_row = self._with_info(self.enable_test_split, "controls.regression.test.enable")
        test_form.addRow("", self._test_enable_row)
        self.test_size = QDoubleSpinBox(self.test_section)
        self.test_size.setRange(0.05, 100000)
        self.test_size.setSingleStep(0.05)
        self.test_size.setValue(0.2)
        self.test_size_label = QLabel(tr("Test size:"), self.test_section)
        self._test_size_row = self._with_info(self.test_size, "controls.regression.test.size")
        test_form.addRow(self.test_size_label, self._test_size_row)
        self.test_strategy = QComboBox(self.test_section)
        self.test_strategy_label = QLabel(tr("Strategy:"), self.test_section)
        self._test_strategy_row = self._with_info(self.test_strategy, "controls.regression.test.strategy")
        test_form.addRow(self.test_strategy_label, self._test_strategy_row)
        self.test_stratify_combo = QComboBox(self.test_section)
        self.test_stratify_label = QLabel(tr("Stratify by:"), self.test_section)
        self._test_stratify_row = self._with_info(self.test_stratify_combo, "controls.regression.test.stratify")
        test_form.addRow(self.test_stratify_label, self._test_stratify_row)
        self.stratify_feature_combo = self.test_stratify_combo
        self.stratify_bins = QSpinBox(self.test_section)
        self.stratify_bins.setRange(2, 20)
        self.stratify_bins.setValue(5)
        self.stratify_bins_label = QLabel(tr("Stratify bins:"), self.test_section)
        self._test_bins_row = self._with_info(self.stratify_bins, "controls.regression.test.bins")
        test_form.addRow(self.stratify_bins_label, self._test_bins_row)
        self.test_section.body_layout().addLayout(test_form)
        controls_layout.addWidget(self.test_section)

        self.hyperparams_section = CollapsibleSection(tr("Hyperparameters"), collapsed=True, parent=self)
        hyperparams_layout = self.hyperparams_section.body_layout()
        self.selector_params_group = QGroupBox(tr("Feature selection"), self.hyperparams_section)
        self.selector_params_layout = QVBoxLayout(self.selector_params_group)
        hyperparams_layout.addWidget(self.selector_params_group)
        self.model_params_group = QGroupBox(tr("Models"), self.hyperparams_section)
        self.model_params_layout = QVBoxLayout(self.model_params_group)
        hyperparams_layout.addWidget(self.model_params_group)
        self.reducer_params_group = QGroupBox(tr("Dimensionality reduction"), self.hyperparams_section)
        self.reducer_params_layout = QVBoxLayout(self.reducer_params_group)
        hyperparams_layout.addWidget(self.reducer_params_group)
        self.reducer_params_group.setVisible(False)
        controls_layout.addWidget(self.hyperparams_section)

        self.features_widget.selection_changed.connect(self._on_features_selection_changed)
        self.target_combo.selection_changed.connect(self._on_target_selection_changed)
        self.selector_combo.selection_changed.connect(self._on_selector_selection_changed)
        self.model_combo.selection_changed.connect(self._on_model_selection_changed)
        self.reducer_combo.selection_changed.connect(self._on_reducer_selection_changed)
        self.run_button.clicked.connect(self._on_run_clicked)
        self.enable_test_split.toggled.connect(self._on_test_toggle)
        self.cv_combo.currentIndexChanged.connect(self._on_cv_changed)
        self.cv_stratify_combo.currentIndexChanged.connect(self._on_cv_stratify_changed)
        self.test_stratify_combo.currentIndexChanged.connect(self._on_test_stratify_changed)
        self.test_strategy.currentIndexChanged.connect(lambda _idx: self._update_test_controls())
        self._on_test_toggle(self.enable_test_split.isChecked())

        try:
            self.features_widget._view_model.features_loaded.connect(
                lambda *_args: self.refresh_feature_lists()
            )
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)
        self.refresh_feature_lists()

    def _with_info(self, widget: QWidget, help_key: Optional[str]) -> QWidget:
        if self._help_viewmodel is None or not help_key:
            return widget

        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        size_policy = widget.sizePolicy()
        size_policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        widget.setSizePolicy(size_policy)
        layout.addWidget(widget, 1)
        layout.addWidget(InfoButton(help_key, self._help_viewmodel, parent=container))
        return container

    def _register_help_context(self) -> None:
        if self._help_viewmodel is None:
            return
        self._help_viewmodel.set_body_extension(
            "controls.regression.feature_selection",
            self._feature_selection_help_extra,
        )
        self._help_viewmodel.set_body_extension(
            "controls.regression.models",
            self._models_help_extra,
        )
        self._help_viewmodel.set_body_extension(
            "controls.regression.dimensionality_reduction",
            self._reducers_help_extra,
        )

    def _feature_selection_help_extra(self) -> str:
        selectors = self._format_available_labels(self._selector_items)
        return self._build_help_extra_block(tr("Available selectors"), selectors)

    def _models_help_extra(self) -> str:
        models = self._format_available_labels(self._model_items)
        return self._build_help_extra_block(tr("Available models"), models)

    def _reducers_help_extra(self) -> str:
        reducers = self._format_available_labels(self._reducer_items)
        return self._build_help_extra_block(tr("Available methods"), reducers)

    def _build_help_extra_block(self, title: str, labels: str) -> str:
        dataset_html = self._dataset_help_html()
        title_text = escape(title)
        return f"<p><b>{title_text}:</b> {labels}</p>{dataset_html}"

    def _format_available_labels(self, items: list[tuple[str, str, dict[str, object]]]) -> str:
        labels = [escape(label) for _key, label, _defaults in items if label]
        return ", ".join(labels) if labels else tr("None")

    def _dataset_help_html(self) -> str:
        frame = self._build_help_dataset_frame()
        feature_count = len(self.selected_input_payloads())
        if frame is None:
            return tr(
                "<p><b>Dataset:</b> No data loaded yet. Selected features: {count}.</p>"
            ).format(count=feature_count)
        rows, _cols = frame.shape
        return tr("<p><b>Dataset:</b> {rows} rows, {count} selected features.</p>").format(
            rows=rows, count=feature_count
        )

    def _build_help_dataset_frame(self):
        payloads: list[dict] = []
        payloads.extend(self.selected_input_payloads())
        payloads.extend(self.selected_target_payloads())
        stratify = self.selected_stratify_payload()
        if stratify and isinstance(stratify, dict) and stratify.get("feature_id") is not None:
            payloads.append(stratify)

        if not payloads:
            return None

        deduped: list[dict] = []
        seen_ids: set[int] = set()
        for payload in payloads:
            fid = self._payload_id(payload)
            if fid is not None and fid in seen_ids:
                continue
            if fid is not None:
                seen_ids.add(fid)
            deduped.append(payload)

        if not deduped:
            return None

        try:
            return self.data_selector.fetch_base_dataframe_for_features(deduped)
        except Exception:
            return None

    def _on_run_clicked(self) -> None:
        self._view_model.request_run()

    def auto_save_enabled(self) -> bool:
        return bool(self.auto_save_checkbox.isChecked())

    def set_export_enabled(self, enabled: bool) -> None:
        self.export_button.setEnabled(bool(enabled))

    def _notify_features_changed(self) -> None:
        self._view_model.notify_features_changed()

    def _on_features_selection_changed(self, *_args) -> None:
        self._notify_features_changed()

    def _on_target_selection_changed(self) -> None:
        previous_ids = list(self._selected_target_ids)
        values = [v for v in self.target_combo.selected_values() if isinstance(v, int)]
        self._selected_target_ids = values
        self._selected_target_id = values[0] if values else None
        if not self._user_selected_stratify:
            self._ensure_stratify_default()
        if self._selected_target_ids != previous_ids:
            self._notify_features_changed()

    def _on_model_selection_changed(self) -> None:
        self._sync_model_param_visibility()

    def _on_selector_selection_changed(self) -> None:
        self._sync_selector_param_visibility()

    def _on_reducer_selection_changed(self) -> None:
        self._sync_reducer_param_visibility()

    def _on_test_toggle(self, checked: bool) -> None:
        self.test_size.setEnabled(checked)
        self.test_strategy.setEnabled(checked)
        self._update_test_controls()

    def _update_test_controls(self) -> None:
        strategy = str(self.test_strategy.currentData() or "random").lower()
        show_strategy = self.enable_test_split.isChecked()
        show_stratify = show_strategy and strategy == "stratified"
        self.test_size.setVisible(show_strategy)
        self.test_size_label.setVisible(show_strategy)
        self._test_size_row.setVisible(show_strategy)
        self.test_strategy.setVisible(show_strategy)
        self.test_strategy_label.setVisible(show_strategy)
        self._test_strategy_row.setVisible(show_strategy)
        self.stratify_feature_combo.setVisible(show_stratify)
        self.test_stratify_label.setVisible(show_stratify)
        self._test_stratify_row.setVisible(show_stratify)
        self.stratify_bins.setVisible(show_stratify)
        self.stratify_bins_label.setVisible(show_stratify)
        self._test_bins_row.setVisible(show_stratify)

        self.stratify_feature_combo.setEnabled(show_stratify)
        self.stratify_bins.setEnabled(show_stratify)

    def _on_cv_changed(self) -> None:
        key = self.cv_combo.currentData()
        self._update_cv_controls(str(key) if key is not None else "none")

    def _update_cv_controls(self, strategy: str) -> None:
        strategy = (strategy or "none").lower()
        show_folds = strategy in {"kfold", "stratified_kfold", "time_series", "group_kfold"}
        show_shuffle = strategy in {"kfold", "stratified_kfold"}
        show_gap = strategy == "time_series"
        show_stratify = strategy == "stratified_kfold"
        show_group = strategy == "group_kfold"

        self.cv_folds.setVisible(show_folds)
        self.cv_folds_label.setVisible(show_folds)
        self._cv_folds_row.setVisible(show_folds)
        self.cv_shuffle.setVisible(show_shuffle)
        self._cv_shuffle_row.setVisible(show_shuffle)
        self.cv_gap.setVisible(show_gap)
        self.cv_gap_label.setVisible(show_gap)
        self._cv_gap_row.setVisible(show_gap)
        self.cv_stratify_combo.setVisible(show_stratify)
        self.cv_stratify_label.setVisible(show_stratify)
        self._cv_stratify_row.setVisible(show_stratify)
        self.cv_group_combo.setVisible(show_group)
        self.cv_group_label.setVisible(show_group)
        self._cv_group_row.setVisible(show_group)

        self.cv_folds.setEnabled(show_folds)
        self.cv_shuffle.setEnabled(show_shuffle)
        self.cv_gap.setEnabled(show_gap)
        self.cv_stratify_combo.setEnabled(show_stratify)
        self.cv_group_combo.setEnabled(show_group)

    def _selected_feature_payloads(self) -> list[dict]:
        return self.features_widget.selected_payloads()

    def _payload_label(self, payload: dict) -> str:
        if not isinstance(payload, dict):
            return tr("Feature")
        return display_name(payload)

    # ------------------------------------------------------------------
    def payload_label(self, payload: dict) -> str:
        return self._payload_label(payload)

    def selected_input_payloads(self) -> list[dict]:
        return self._selected_feature_payloads()

    def selected_target_payloads(self) -> list[dict]:
        payloads: list[dict] = []
        for fid in self._selected_target_ids:
            payload = self._feature_payload_lookup.get(fid)
            if payload is not None:
                payloads.append(payload)
        return payloads

    def target_payload(self) -> Optional[dict]:
        if self._selected_target_id is None:
            return None
        return self._feature_payload_lookup.get(self._selected_target_id)

    def selected_selector_keys(self) -> list[str]:
        return [str(v) for v in self.selector_combo.selected_values()]

    def selected_model_keys(self) -> list[str]:
        return [str(v) for v in self.model_combo.selected_values()]

    def selected_stratify_payload(self) -> Optional[dict]:
        return self._current_stratify_payload

    def model_label_for_key(self, key: str) -> str:
        for model_key, label, _defaults in self._model_items:
            if str(model_key) == str(key):
                return str(label)
        return str(key)

    def selector_label_for_key(self, key: str) -> str:
        for selector_key, label, _defaults in self._selector_items:
            if str(selector_key) == str(key):
                return str(label)
        return str(key)

    def reducer_label_for_key(self, key: str) -> str:
        if str(key) == "none":
            return tr("Disabled")
        for reducer_key, label, _defaults in self._reducer_items:
            if str(reducer_key) == str(key):
                return str(label)
        return str(key)

    def refresh_feature_lists(self) -> None:
        previous_ids = list(self._selected_target_ids)
        lookup: dict[int, dict] = {}
        items: list[tuple[str, int]] = []
        for payload in self.features_widget.all_payloads():
            if not isinstance(payload, dict):
                continue
            fid = self._payload_id(payload)
            if fid is None:
                continue
            lookup[fid] = payload
            items.append((self._payload_label(payload), fid))

        self._feature_payload_lookup = lookup
        signature = tuple(items)
        desired_ids = [fid for fid in previous_ids if fid in lookup]
        needs_items_update = signature != self._feature_items_signature
        needs_selection_update = desired_ids != previous_ids
        if not needs_items_update and not needs_selection_update:
            return

        block = self.target_combo.blockSignals(True)
        try:
            if needs_items_update:
                self.target_combo.set_items(items, check_all=False)
                self._feature_items_signature = signature
            self.target_combo.set_selected_values(desired_ids)
        finally:
            self.target_combo.blockSignals(block)
        if needs_items_update or needs_selection_update:
            self._on_target_selection_changed()

    def _payload_id(self, payload: Optional[dict]) -> Optional[int]:
        if not isinstance(payload, dict):
            return None
        fid = payload.get("feature_id")
        try:
            return int(fid) if fid is not None else None
        except Exception:
            return None

    def _ensure_stratify_default(self) -> None:
        if self._user_selected_stratify:
            return
        target = self.target_payload()
        self._apply_stratify_selection(target)

    def _apply_stratify_selection(self, payload: Optional[dict]) -> None:
        self._syncing_stratify = True
        try:
            self._set_combo_payload(self.cv_stratify_combo, payload)
            self._set_combo_payload(self.test_stratify_combo, payload)
        finally:
            self._syncing_stratify = False
        self._current_stratify_payload = payload

    def _set_combo_payload(self, combo: QComboBox, payload: Optional[dict]) -> None:
        block = combo.blockSignals(True)
        try:
            if payload is None:
                combo.setCurrentIndex(-1)
                if combo.currentIndex() == -1 and combo.count():
                    combo.setCurrentIndex(0)
                return
            index = self._find_payload_index(combo, payload)
            if index >= 0:
                combo.setCurrentIndex(index)
            elif combo.count():
                combo.setCurrentIndex(0)
        finally:
            combo.blockSignals(block)

    def _find_payload_index(self, combo: QComboBox, payload: dict) -> int:
        if isinstance(payload, dict) and payload.get("group_kind"):
            desired = str(payload.get("group_kind"))
            for idx in range(combo.count()):
                data = combo.itemData(idx)
                if isinstance(data, dict) and str(data.get("group_kind")) == desired:
                    return idx
        key = self._payload_id(payload)
        if key is None:
            return -1
        for idx in range(combo.count()):
            data = combo.itemData(idx)
            if isinstance(data, dict) and self._payload_id(data) == key:
                return idx
        return -1

    def _on_cv_stratify_changed(self, index: int) -> None:
        if self._syncing_stratify:
            return
        data = self.cv_stratify_combo.itemData(index)
        payload = data if isinstance(data, dict) else None
        self._user_selected_stratify = True
        self._apply_stratify_selection(payload)
        self._notify_features_changed()

    def _on_test_stratify_changed(self, index: int) -> None:
        if self._syncing_stratify:
            return
        data = self.test_stratify_combo.itemData(index)
        payload = data if isinstance(data, dict) else None
        self._user_selected_stratify = True
        self._apply_stratify_selection(payload)
        self._notify_features_changed()

    def preprocessing_parameters(self) -> dict[str, object]:
        return self.data_selector.preprocessing_widget.parameters()

    def selected_reducer_keys(self) -> list[str]:
        return [str(v) for v in self.reducer_combo.selected_values()]

    def reducer_parameters(self) -> dict[str, dict[str, object]]:
        params: dict[str, dict[str, object]] = {}
        selected = set(self.selected_reducer_keys())
        if not selected:
            return params
        for key, getters in self._reducer_param_getters.items():
            if key not in selected:
                continue
            values: dict[str, object] = {}
            for name, getter in getters.items():
                val = getter()
                if val is None or val == "":
                    continue
                values[name] = val
            if values:
                params[key] = values
        return params

    def selector_parameters(self) -> dict[str, dict[str, object]]:
        params: dict[str, dict[str, object]] = {}
        selected = set(self.selected_selector_keys())
        if not selected:
            return params
        for key, getters in self._selector_param_getters.items():
            if key not in selected:
                continue
            values: dict[str, object] = {}
            for name, getter in getters.items():
                val = getter()
                if val is None or val == "":
                    continue
                values[name] = val
            if values:
                params[key] = values
        return params

    def model_parameters(self) -> dict[str, dict[str, object]]:
        params: dict[str, dict[str, object]] = {}
        selected = set(self.selected_model_keys())
        if not selected:
            return params
        for key, getters in self._model_param_getters.items():
            if selected and key not in selected:
                continue
            values: dict[str, object] = {}
            for name, getter in getters.items():
                val = getter()
                if val is None or val == "":
                    continue
                values[name] = val
            if values:
                params[key] = values
        return params

    # ------------------------------------------------------------------
    def set_selectors(self, items: list[tuple[str, str, dict[str, object]]]):
        self._selector_items = list(items)
        combo_items = [(label, key) for key, label, _defaults in items]
        self.selector_combo.set_items(combo_items, check_all=False)
        self.selector_combo.set_selected_values(["none"] if any(key == "none" for key, _, _ in items) else [])
        self._build_selector_param_forms()
        self._sync_selector_param_visibility()

    def set_models(self, items: list[tuple[str, str, dict[str, object]]]):
        self._model_items = list(items)
        combo_items = [(label, key) for key, label, _defaults in items]
        self.model_combo.set_items(combo_items, check_all=False)
        self.model_combo.set_selected_values(["linear_regression"] if any(key == "linear_regression" for key, _, _ in items) else [])
        self._build_model_param_forms()
        self._sync_model_param_visibility()

    def set_reducers(self, items: list[tuple[str, str, dict[str, object]]]):
        self._reducer_items = [(key, label, defaults) for key, label, defaults in items if key != "none"]
        combo_items = [(label, key) for key, label, _defaults in self._reducer_items]
        self.reducer_combo.set_items(combo_items, check_all=False)
        self._build_reducer_param_forms()
        self._sync_reducer_param_visibility()

    def set_cv_strategies(self, items: list[tuple[str, str]]):
        self.cv_combo.clear()
        for key, label in items:
            self.cv_combo.addItem(tr(label), key)
        if items:
            self.cv_combo.setCurrentIndex(0)
        self._on_cv_changed()

    def set_group_kinds(self, items: list[tuple[str, str]]) -> None:
        self.cv_group_combo.clear()
        self.cv_group_combo.addItem(tr("None"), None)
        for key, label in items:
            self.cv_group_combo.addItem(tr(label), key)
        self.cv_group_combo.setCurrentIndex(0)

    def selected_group_kind(self) -> Optional[str]:
        value = self.cv_group_combo.currentData()
        return str(value) if value else None

    def set_test_strategies(self, items: list[tuple[str, str]]):
        self.test_strategy.clear()
        for key, label in items:
            self.test_strategy.addItem(tr(label), key)
        if items:
            preferred_index = next(
                (idx for idx, (key, _label) in enumerate(items) if key == "time"),
                0,
            )
            self.test_strategy.setCurrentIndex(preferred_index)
        self._on_test_toggle(self.enable_test_split.isChecked())

    def set_stratify_options(self, options: list[tuple[str, Optional[dict]]]):
        combos = (self.cv_stratify_combo, self.test_stratify_combo)
        for combo in combos:
            block = combo.blockSignals(True)
            try:
                combo.clear()
                for label, payload in options:
                    combo.addItem(tr(label), payload)
            finally:
                combo.blockSignals(block)

        desired = self._current_stratify_payload
        if not self._user_selected_stratify:
            target_payload = self.target_payload()
            if target_payload is not None:
                desired = target_payload
        self._apply_stratify_selection(desired)

    # ------------------------------------------------------------------
    def _build_selector_param_forms(self) -> None:
        self._clear_layout(self.selector_params_layout)
        self._selector_param_getters.clear()
        self._selector_param_groups.clear()
        for key, label, defaults in self._selector_items:
            specs = REGRESSION_SELECTOR_PARAM_SPECS.get(key)
            if not specs:
                continue
            box = QGroupBox(tr(label), self.selector_params_group)
            form = QFormLayout(box)
            getters: dict[str, Callable[[], object]] = {}
            for spec in specs:
                name = str(spec.get("name"))
                widget, getter = self._create_param_widget(spec, defaults.get(name), box)
                getters[name] = getter
                help_key = spec.get("help_key")
                label_text = tr(str(spec.get("label", name.replace("_", " ").title())))
                form.addRow(
                    label_text,
                    self._with_info(widget, str(help_key) if help_key else None),
                )
            self.selector_params_layout.addWidget(box)
            self._selector_param_groups[key] = box
            self._selector_param_getters[key] = getters
        self.selector_params_layout.addStretch(1)
        self._sync_selector_param_visibility()

    def _build_model_param_forms(self) -> None:
        self._clear_layout(self.model_params_layout)
        self._model_param_getters.clear()
        self._model_param_groups.clear()
        for key, label, defaults in self._model_items:
            specs = REGRESSION_MODEL_PARAM_SPECS.get(key)
            if not specs:
                continue
            box = QGroupBox(tr(label), self.model_params_group)
            form = QFormLayout(box)
            getters: dict[str, Callable[[], object]] = {}
            for spec in specs:
                name = str(spec.get("name"))
                widget, getter = self._create_param_widget(spec, defaults.get(name), box)
                getters[name] = getter
                help_key = spec.get("help_key")
                label_text = tr(str(spec.get("label", name.replace("_", " ").title())))
                form.addRow(
                    label_text,
                    self._with_info(widget, str(help_key) if help_key else None),
                )
            self.model_params_layout.addWidget(box)
            self._model_param_groups[key] = box
            self._model_param_getters[key] = getters
        self.model_params_layout.addStretch(1)
        self._sync_model_param_visibility()

    def _sync_model_param_visibility(self) -> None:
        selected = set(self.selected_model_keys())
        for key, group in self._model_param_groups.items():
            if group is None:
                continue
            group.setVisible(key in selected)

    def _sync_selector_param_visibility(self) -> None:
        selected = set(self.selected_selector_keys())
        for key, group in self._selector_param_groups.items():
            if group is None:
                continue
            group.setVisible(key in selected)

    def _build_reducer_param_forms(self) -> None:
        self._clear_layout(self.reducer_params_layout)
        self._reducer_param_getters.clear()
        self._reducer_param_groups.clear()
        for key, label, defaults in self._reducer_items:
            specs = REGRESSION_REDUCER_PARAM_SPECS.get(key)
            if not specs:
                continue
            method_defaults = REGRESSION_REDUCER_DEFAULTS.get(key, {})
            merged_defaults = dict(method_defaults)
            merged_defaults.update(defaults or {})
            box = QGroupBox(tr(label), self.reducer_params_group)
            form = QFormLayout(box)
            getters: dict[str, Callable[[], object]] = {}
            method_help_key = f"controls.regression.reducers.{key}"
            form.addRow(
                tr("About"),
                self._with_info(QLabel(tr("Method description"), box), method_help_key),
            )
            for spec in specs:
                name = str(spec.get("name"))
                widget, getter = self._create_param_widget(spec, merged_defaults.get(name), box)
                getters[name] = getter
                help_key = spec.get("help_key")
                label_text = tr(str(spec.get("label", name.replace("_", " ").title())))
                form.addRow(
                    label_text,
                    self._with_info(widget, str(help_key) if help_key else None),
                )
            self.reducer_params_layout.addWidget(box)
            self._reducer_param_groups[key] = box
            self._reducer_param_getters[key] = getters
        self.reducer_params_layout.addStretch(1)

    def _sync_reducer_param_visibility(self) -> None:
        selected = set(self.selected_reducer_keys())
        any_visible = False
        for key, group in self._reducer_param_groups.items():
            if group is None:
                continue
            visible = key in selected
            group.setVisible(visible)
            any_visible = any_visible or visible
        if self.reducer_params_group is not None:
            self.reducer_params_group.setVisible(any_visible)

    def _create_param_widget(self, spec: dict[str, object], default: object, parent: QWidget) -> tuple[QWidget, Callable[[], object]]:
        """Create the widget specified in *spec* and return it with a getter."""

        ptype = str(spec.get("type", "text"))

        if ptype == "bool":
            widget = QCheckBox(parent)
            widget.setChecked(bool(default))
            getter = lambda w=widget: bool(w.isChecked())
            return widget, getter

        if ptype in {"int", "int_optional", "int_all"}:
            widget = QSpinBox(parent)
            minimum = int(spec.get("min", -1 if ptype != "int" else 0))
            maximum = int(spec.get("max", 1000))
            widget.setRange(minimum, maximum)
            widget.setSingleStep(int(spec.get("step", 1)))
            if ptype == "int_optional":
                widget.setSpecialValueText(tr("None"))
                if default is None:
                    widget.setValue(minimum)
                else:
                    widget.setValue(int(default))
                getter = lambda w=widget, min_val=minimum: None if w.value() == min_val else int(w.value())
            elif ptype == "int_all":
                widget.setSpecialValueText(tr("All"))
                if default in (None, "all"):
                    widget.setValue(minimum)
                else:
                    widget.setValue(int(default))
                getter = lambda w=widget, min_val=minimum: "all" if w.value() == min_val else int(w.value())
            else:
                widget.setValue(int(default) if default is not None else minimum)
                getter = lambda w=widget: int(w.value())
            return widget, getter

        if ptype in {"float", "float_optional"}:
            widget = QDoubleSpinBox(parent)
            widget.setRange(float(spec.get("min", 0.0)), float(spec.get("max", 10_000.0)))
            widget.setSingleStep(float(spec.get("step", 0.1)))
            widget.setDecimals(int(spec.get("decimals", 3)))
            if ptype == "float_optional":
                widget.setSpecialValueText(tr("None"))
                if default is None:
                    widget.setValue(widget.minimum())
                else:
                    widget.setValue(float(default))
                getter = lambda w=widget: None if w.value() == w.minimum() else float(w.value())
            else:
                widget.setValue(float(default) if default is not None else widget.minimum())
                getter = lambda w=widget: float(w.value())
            return widget, getter

        if ptype == "choice":
            widget = QComboBox(parent)
            for choice in spec.get("choices", []):
                widget.addItem(tr(str(choice)), choice)
            if default is not None:
                idx = widget.findData(default)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            getter = lambda w=widget: w.currentData()
            return widget, getter

        if ptype == "int_list":
            line = QLineEdit(parent)
            placeholder = spec.get("placeholder")
            if placeholder:
                line.setPlaceholderText(tr(str(placeholder)))
            if isinstance(default, (list, tuple)):
                line.setText(", ".join(str(v) for v in default))
            elif default is None:
                line.setText("")
            else:
                line.setText(str(default))

            def _parse(value: str) -> Optional[tuple[int, ...]]:
                parts = [p for p in re.split(r"[,\\s]+", value.strip()) if p]
                if not parts:
                    return None
                numbers: list[int] = []
                for part in parts:
                    try:
                        numbers.append(int(part))
                    except ValueError:
                        return None
                return tuple(numbers)

            getter = lambda w=line: _parse(w.text())
            return line, getter

        # text fallback
        line = QLineEdit(parent)
        line.setText("" if default is None else str(default))
        getter = lambda w=line: (w.text().strip() or None)
        return line, getter

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)  # type: ignore[arg-type]


__all__ = ["RegressionSidebar"]
