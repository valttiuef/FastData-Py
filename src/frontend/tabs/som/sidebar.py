from __future__ import annotations
"""
Refactored Sidebar widget for SOM tab.

This sidebar is self-contained and receives view models. It calls their methods
directly and reacts to their signals. All inter-widget communication goes through
the view models, not direct signals between widgets.
"""


from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import Qt, Signal

from PySide6.QtWidgets import (

    QWidget,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QSizePolicy,
)
from ...localization import tr

import logging

from ...widgets.collapsible_section import CollapsibleSection
from ...widgets.data_selector_widget import DataSelectorWidget
from ...widgets.help_widgets import InfoButton
from ...widgets.sidebar_widget import SidebarWidget
from ...models.hybrid_pandas_model import HybridPandasModel

from .clustering_viewmodel import ClusteringRequest, ClusteringViewModel
from ...models.log_model import LogModel, get_log_model
from ...utils import toast_error, toast_info, toast_warn
from ...viewmodels.help_viewmodel import HelpViewModel

if TYPE_CHECKING:
    from .viewmodel import SomViewModel



class SomSidebar(SidebarWidget):
    """Self-contained sidebar widget that manages all SOM controls.
    
    Principles:
    - Receives view models (SomViewModel, ClusteringViewModel)
    - Calls methods on view models in response to user actions
    - Reacts to view model signals to update UI state
    - Does NOT send signals directly to other UI components
    - Uses callbacks to coordinate with parent SomTab only when necessary
    """
    save_timeline_clusters_requested = Signal()

    def __init__(
        self,
        view_model: Optional[SomViewModel] = None,
        clustering_view_model: Optional[ClusteringViewModel] = None,
        *,
        log_model: Optional[LogModel] = None,
        data_model: Optional[HybridPandasModel] = None,
        help_viewmodel: Optional[HelpViewModel] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(tr("Settings"), parent=parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._logger = logging.getLogger(__name__)

        self._view_model = view_model
        self._clustering_view_model = clustering_view_model
        self._log_model = log_model or get_log_model(parent)
        self._data_model = data_model
        self._help_viewmodel = help_viewmodel
        if self._view_model is None:
            self._logger.warning("SomSidebar initialised without view_model.")
        if self._clustering_view_model is None:
            self._logger.warning("SomSidebar initialised without clustering_view_model.")
        if self._data_model is None:
            self._logger.warning("SomSidebar initialised without data_model.")
        if self._help_viewmodel is None:
            self._logger.warning("SomSidebar initialised without help_viewmodel.")
        
        self._build_ui()
        self._wire_signals()

    def _log(self, message: str, *, level: int = logging.INFO) -> None:
        """Show a toast for user-visible SOM messages."""
        try:
            if level >= logging.ERROR:
                toast_error(message, title=tr("SOM"), tab_key="som")
            elif level >= logging.WARNING:
                toast_warn(message, title=tr("SOM"), tab_key="som")
            else:
                toast_info(message, title=tr("SOM"), tab_key="som")
        except Exception:
            self._logger.log(level, message)

    def _make_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return label

    def _make_info(self, help_key: str | None) -> Optional[InfoButton]:
        if help_key and self._help_viewmodel is not None:
            return InfoButton(help_key, self._help_viewmodel, parent=self)
        return None

    def _control_with_info(self, control: QWidget, help_key: str | None) -> QWidget:
        """Wrap a control with an info button after it."""
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(control, 1)

        info = self._make_info(help_key)
        if info is not None:
            layout.addWidget(info, alignment=Qt.AlignmentFlag.AlignLeft)

        return container

    def _build_ui(self) -> None:
        """Build all UI elements for the sidebar."""
        layout = self.content_layout()

        # ---- Actions group
        actions_group = QGroupBox(tr("Actions"), self)
        actions_layout = QVBoxLayout(actions_group)
        self.train_button = QPushButton(tr("Train SOM"), actions_group)
        actions_layout.addWidget(self.train_button)
        self.load_button = QPushButton(tr("Load SOM"), actions_group)
        self.save_button = QPushButton(tr("Save SOM"), actions_group)
        self.save_timeline_clusters_button = QPushButton(tr("Save timeline clusters"), actions_group)
        self.export_button = QPushButton(tr("Export results..."), actions_group)
        self.save_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.save_timeline_clusters_button.setEnabled(False)
        actions_layout.addWidget(self.load_button)
        actions_layout.addWidget(self.save_button)
        actions_layout.addWidget(self.save_timeline_clusters_button)
        actions_layout.addWidget(self.export_button)
        self.set_sticky_actions(actions_group)

        self.data_selector = DataSelectorWidget(
            parent=self,
            data_model=self._data_model,
            help_viewmodel=self._help_viewmodel,
        )
        layout.addWidget(self.data_selector, 1)

        self.features_widget = self.data_selector.features_widget
        self.features_view = self.features_widget.table_view

        # ---- SOM hyperparameters
        self.hyperparams_section = CollapsibleSection(
            tr("SOM hyperparameters"), collapsed=True, parent=self
        )
        params_form = QFormLayout()
        
        self.map_width_edit = QLineEdit()
        self.map_width_edit.setPlaceholderText(tr("auto"))
        self.map_height_edit = QLineEdit()
        self.map_height_edit.setPlaceholderText(tr("auto"))
        
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setDecimals(2)
        self.sigma_spin.setRange(0.1, 10.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(6.0)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(2)
        self.lr_spin.setRange(0.01, 1.0)
        self.lr_spin.setSingleStep(0.05)
        self.lr_spin.setValue(0.5)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(0, 10000)
        self.epochs_spin.setSpecialValueText("100")
        self.epochs_spin.setValue(100)
        
        self.norm_combo = QComboBox()
        self.norm_combo.addItem(tr("Z-score"), "zscore")
        self.norm_combo.addItem(tr("Min-max"), "minmax")
        self.norm_combo.addItem(tr("None"), "none")
        
        self.training_mode_combo = QComboBox()
        self.training_mode_combo.addItem(tr("Batch"), "batch")
        self.training_mode_combo.addItem(tr("Random"), "random")

        params_form.addRow(
            self._make_label(tr("Map width")),
            self._control_with_info(self.map_width_edit, "controls.som.hyperparams.map_width"),
        )
        params_form.addRow(
            self._make_label(tr("Map height")),
            self._control_with_info(self.map_height_edit, "controls.som.hyperparams.map_height"),
        )
        params_form.addRow(
            self._make_label(tr("Sigma")),
            self._control_with_info(self.sigma_spin, "controls.som.hyperparams.sigma"),
        )
        params_form.addRow(
            self._make_label(tr("Learning rate")),
            self._control_with_info(self.lr_spin, "controls.som.hyperparams.learning_rate"),
        )
        params_form.addRow(
            self._make_label(tr("Epochs")),
            self._control_with_info(self.epochs_spin, "controls.som.hyperparams.epochs"),
        )
        params_form.addRow(
            self._make_label(tr("Normalisation")),
            self._control_with_info(self.norm_combo, "controls.som.hyperparams.normalisation"),
        )
        params_form.addRow(
            self._make_label(tr("Training mode")),
            self._control_with_info(self.training_mode_combo, "controls.som.hyperparams.training_mode"),
        )
        
        self.hyperparams_section.bodyLayout().addLayout(params_form)
        layout.addWidget(self.hyperparams_section)

        # ---- Feature clustering
        self.feature_cluster_section = CollapsibleSection(
            tr("Feature clustering"), collapsed=True, parent=self
        )
        feature_body = self.feature_cluster_section.bodyLayout()
        auto_feature_row = QHBoxLayout()
        self.auto_cluster_features_checkbox = QCheckBox(tr("Auto cluster features"))
        self.auto_cluster_features_checkbox.setChecked(True)
        auto_feature_row.addWidget(self.auto_cluster_features_checkbox)
        auto_feature_info = self._make_info("controls.som.features.auto_cluster")
        if auto_feature_info is not None:
            auto_feature_row.addWidget(auto_feature_info)
        auto_feature_row.addStretch(1)
        feature_body.addLayout(auto_feature_row)

        feature_grid = QGridLayout()
        feature_grid.setHorizontalSpacing(8)
        feature_grid.setVerticalSpacing(6)
        feature_grid.setColumnStretch(1, 1)
        feature_grid.setColumnStretch(3, 1)
        
        self.feature_cluster_model_combo = QComboBox()
        feature_grid.addWidget(
            self._make_label(tr("Model")),
            0,
            0,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        feature_grid.addWidget(
            self.feature_cluster_model_combo,
            0,
            1,
            1,
            3,
        )
        model_info = self._make_info("controls.som.features.model")
        if model_info is not None:
            feature_grid.addWidget(model_info, 0, 4, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        self.feature_cluster_spin = QSpinBox()
        self.feature_cluster_spin.setRange(2, 64)
        self.feature_cluster_spin.setValue(5)
        
        self.feature_cluster_count_spin = QSpinBox()
        self.feature_cluster_count_spin.setRange(0, 512)
        self.feature_cluster_count_spin.setSpecialValueText(tr("Auto"))
        self.feature_cluster_count_spin.setValue(0)
        
        feature_grid.addWidget(
            self._make_label(tr("Max K")),
            1,
            0,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        feature_grid.addWidget(
            self.feature_cluster_spin,
            1,
            1,
        )
        feature_grid.addWidget(QLabel(tr("Clusters")), 1, 2)
        feature_grid.addWidget(self.feature_cluster_count_spin, 1, 3)
        k_info = self._make_info("controls.som.features.k")
        if k_info is not None:
            feature_grid.addWidget(k_info, 1, 4, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        self.feature_cluster_scorer_combo = QComboBox()
        self.feature_cluster_scorer_combo.addItem(tr("Silhouette"), "silhouette")
        self.feature_cluster_scorer_combo.addItem(tr("Calinski-Harabasz"), "calinski_harabasz")
        self.feature_cluster_scorer_combo.addItem(tr("Davies-Bouldin"), "davies_bouldin")
        feature_grid.addWidget(
            self._make_label(tr("Score")),
            2,
            0,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        feature_grid.addWidget(
            self.feature_cluster_scorer_combo,
            2,
            1,
            1,
            3,
        )
        score_info = self._make_info("controls.som.features.score")
        if score_info is not None:
            feature_grid.addWidget(score_info, 2, 4, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        feature_body.addLayout(feature_grid)
        buttons_row = QHBoxLayout()
        self.feature_cluster_button = QPushButton(tr("Cluster features"))
        buttons_row.addWidget(self.feature_cluster_button)
        feature_button_info = self._make_info("controls.som.features.cluster_button")
        if feature_button_info is not None:
            buttons_row.addWidget(feature_button_info)
        buttons_row.addStretch(1)
        feature_body.addLayout(buttons_row)
        layout.addWidget(self.feature_cluster_section)

        # ---- Neuron clustering
        self.neuron_cluster_section = CollapsibleSection(
            tr("Neuron clustering"), collapsed=True, parent=self
        )
        neuron_body = self.neuron_cluster_section.bodyLayout()
        auto_timeline_row = QHBoxLayout()
        self.auto_cluster_timeline_checkbox = QCheckBox(tr("Auto cluster timeline"))
        self.auto_cluster_timeline_checkbox.setChecked(True)
        auto_timeline_row.addWidget(self.auto_cluster_timeline_checkbox)
        auto_timeline_info = self._make_info("controls.som.neurons.auto_cluster_timeline")
        if auto_timeline_info is not None:
            auto_timeline_row.addWidget(auto_timeline_info)
        auto_timeline_row.addStretch(1)
        neuron_body.addLayout(auto_timeline_row)

        neuron_grid = QGridLayout()
        neuron_grid.setHorizontalSpacing(8)
        neuron_grid.setVerticalSpacing(6)
        neuron_grid.setColumnStretch(1, 1)
        neuron_grid.setColumnStretch(3, 1)
        
        self.neuron_cluster_model_combo = QComboBox()
        self.neuron_cluster_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        neuron_grid.addWidget(
            self._make_label(tr("Model")),
            0,
            0,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        neuron_grid.addWidget(
            self.neuron_cluster_model_combo,
            0,
            1,
            1,
            3,
        )
        model_info = self._make_info("controls.som.neurons.model")
        if model_info is not None:
            neuron_grid.addWidget(model_info, 0, 4, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        self.neuron_cluster_spin = QSpinBox()
        self.neuron_cluster_spin.setRange(2, 64)
        self.neuron_cluster_spin.setValue(5)
        
        self.neuron_cluster_count_spin = QSpinBox()
        self.neuron_cluster_count_spin.setRange(0, 1024)
        self.neuron_cluster_count_spin.setSpecialValueText(tr("Auto"))
        self.neuron_cluster_count_spin.setValue(0)
        
        neuron_grid.addWidget(
            self._make_label(tr("Max K")),
            1,
            0,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        neuron_grid.addWidget(
            self.neuron_cluster_spin,
            1,
            1,
        )
        neuron_grid.addWidget(QLabel(tr("Clusters")), 1, 2)
        neuron_grid.addWidget(self.neuron_cluster_count_spin, 1, 3)
        k_info = self._make_info("controls.som.neurons.k")
        if k_info is not None:
            neuron_grid.addWidget(k_info, 1, 4, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        self.neuron_cluster_scorer_combo = QComboBox()
        self.neuron_cluster_scorer_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.neuron_cluster_scorer_combo.addItem(tr("Silhouette"), "silhouette")
        self.neuron_cluster_scorer_combo.addItem(tr("Calinski-Harabasz"), "calinski_harabasz")
        self.neuron_cluster_scorer_combo.addItem(tr("Davies-Bouldin"), "davies_bouldin")
        neuron_grid.addWidget(
            self._make_label(tr("Score")),
            2,
            0,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        neuron_grid.addWidget(
            self.neuron_cluster_scorer_combo,
            2,
            1,
            1,
            3,
        )
        score_info = self._make_info("controls.som.neurons.score")
        if score_info is not None:
            neuron_grid.addWidget(score_info, 2, 4, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        neuron_body.addLayout(neuron_grid)
        neuron_buttons = QHBoxLayout()
        self.neuron_cluster_button = QPushButton(tr("Cluster neurons"))
        neuron_buttons.addWidget(self.neuron_cluster_button)
        neuron_button_info = self._make_info("controls.som.neurons.cluster_button")
        if neuron_button_info is not None:
            neuron_buttons.addWidget(neuron_button_info)
        neuron_buttons.addStretch(1)
        neuron_body.addLayout(neuron_buttons)
        layout.addWidget(self.neuron_cluster_section)

    def _wire_signals(self) -> None:
        """Wire all UI signals to their handlers."""
        self.train_button.clicked.connect(self._on_train_clicked)
        self.feature_cluster_button.clicked.connect(self._on_cluster_features_clicked)
        self.neuron_cluster_button.clicked.connect(self._on_cluster_neurons_clicked)
        self.save_timeline_clusters_button.clicked.connect(self.save_timeline_clusters_requested.emit)
        self.features_widget.selection_changed.connect(self._on_features_selection_changed)

        # React to view model signals
        if self._view_model:
            self._view_model.training_started.connect(self._on_training_started)
            self._view_model.training_finished.connect(self._on_training_finished)
            self._view_model.set_selected_feature_payloads(self.data_selector.features_widget.selected_payloads())

    def _parse_dimension(self, text: str) -> Optional[int]:
        value = (text or "").strip().lower()
        if not value or value == "auto" or value == tr("auto").lower():
            return None
        try:
            dim = int(value)
        except ValueError:
            return None
        return max(2, dim)

    def _feature_display_name(self, payload: dict) -> str:
        parts: list[str] = []
        for key in ("base_name", "source", "unit", "type"):
            value = payload.get(key)
            if value and value not in parts:
                parts.append(str(value))
        if parts:
            return " / ".join(parts)
        fid = payload.get("feature_id")
        if fid is not None:
            return tr("Feature {id}").format(id=fid)
        return ""

    # ---- Button click handlers
    def _on_train_clicked(self) -> None:
        """Handle Train SOM button click."""
        if not self._view_model:
            return

        selected_payloads = self.data_selector.features_widget.selected_payloads()

        if not selected_payloads:
            self._log(tr("Select one or more features to train the SOM before training."), level=logging.INFO)
            return

        # Keep timeline overlay feature payloads in sync with the features used for training.
        try:
            self._view_model.set_selected_feature_payloads(selected_payloads)
        except Exception:
            self._logger.warning("Exception in _on_train_clicked", exc_info=True)

        # Parse UI parameters
        map_width = self._parse_dimension(self.map_width_edit.text())
        map_height = self._parse_dimension(self.map_height_edit.text())

        epochs = self.epochs_spin.value() or None
        filters = self.data_selector.build_data_filters()
        if filters is None:
            self._log(tr("Unable to build filters for selected features."), level=logging.INFO)
            return
        preprocessing = self.data_selector.preprocessing_widget.parameters()
        data_frame = self.data_selector.fetch_base_dataframe_for_features(
            selected_payloads,
        )
        if data_frame is None or data_frame.empty:
            self._log(tr("No measurements available for the selected features."), level=logging.INFO)
            return

        try:
            self._view_model.train(
                [name for payload in selected_payloads if (name := self._feature_display_name(payload))],
                feature_payloads=selected_payloads,
                start=filters.start,
                end=filters.end,
                systems=filters.systems,
                datasets=filters.datasets,
                group_ids=filters.group_ids,
                months=filters.months,
                preprocessing=preprocessing,
                map_shape=(map_width, map_height),
                sigma=self.sigma_spin.value(),
                learning_rate=self.lr_spin.value(),
                epochs=epochs,
                normalisation=self.norm_combo.currentData(),
                training_mode=self.training_mode_combo.currentData(),
                data_frame=data_frame,
            )
        except Exception as e:
            self._log(tr("Failed to start training: {error}").format(error=e), level=logging.ERROR)

    def _on_cluster_features_clicked(self) -> None:
        """Handle Cluster features button click."""
        self.start_feature_clustering()

    def _on_cluster_neurons_clicked(self) -> None:
        """Handle Cluster neurons button click."""
        self.start_neuron_clustering()

    def _on_features_selection_changed(self, payloads: list[dict]) -> None:
        if not self._view_model:
            return
        try:
            self._view_model.set_selected_feature_payloads(payloads)
        except Exception:
            self._logger.warning("Exception in _on_features_selection_changed", exc_info=True)

    def auto_cluster_features_enabled(self) -> bool:
        checkbox = getattr(self, "auto_cluster_features_checkbox", None)
        return bool(checkbox is not None and checkbox.isChecked())

    def auto_cluster_timeline_enabled(self) -> bool:
        checkbox = getattr(self, "auto_cluster_timeline_checkbox", None)
        return bool(checkbox is not None and checkbox.isChecked())

    def start_feature_clustering(self, *, show_started_toast: bool = True) -> None:
        """Start feature clustering using current sidebar settings."""
        if not self._view_model or not self._clustering_view_model:
            return

        if self._view_model.result() is None:
            self._log(tr("Train a SOM model before clustering features."), level=logging.INFO)
            return

        max_k = self.feature_cluster_spin.value()
        n_clusters = self.feature_cluster_count_spin.value() or None
        scoring = self.feature_cluster_scorer_combo.currentData()
        method_key = self.feature_cluster_model_combo.currentData() or "kmeans"

        self.feature_cluster_button.setEnabled(False)

        def on_finished(result):
            self.feature_cluster_button.setEnabled(True)

        def on_error(message: str):
            self.feature_cluster_button.setEnabled(True)
            if message:
                self._log(tr("Failed to cluster features: {message}").format(message=message), level=logging.ERROR)

        try:
            request = ClusteringRequest(
                method=method_key,
                max_k=max_k,
                n_clusters=n_clusters,
                scoring=scoring,
            )
            self._clustering_view_model.cluster_features(
                request=request,
                on_finished=on_finished,
                on_error=on_error,
            )
            if show_started_toast:
                self._log(tr("Clustering features..."), level=logging.INFO)
        except Exception as e:
            self.feature_cluster_button.setEnabled(True)
            self._log(tr("Failed to start feature clustering: {error}").format(error=e), level=logging.ERROR)

    def start_neuron_clustering(self, *, show_started_toast: bool = True) -> None:
        """Start neuron clustering using current sidebar settings."""
        if not self._view_model or not self._clustering_view_model:
            return

        if self._view_model.result() is None:
            self._log(tr("Train a SOM model before clustering neurons."), level=logging.INFO)
            return

        max_k = self.neuron_cluster_spin.value()
        n_clusters = self.neuron_cluster_count_spin.value() or None
        scoring = self.neuron_cluster_scorer_combo.currentData()
        method_key = self.neuron_cluster_model_combo.currentData() or "kmeans"

        self.neuron_cluster_button.setEnabled(False)

        def on_finished(result):
            self.neuron_cluster_button.setEnabled(True)
            # Note: neuron_clusters_updated signal is already emitted by ClusteringViewModel

        def on_error(message: str):
            self.neuron_cluster_button.setEnabled(True)
            if message:
                self._log(tr("Failed to cluster neurons: {message}").format(message=message), level=logging.ERROR)

        try:
            request = ClusteringRequest(
                method=method_key,
                max_k=max_k,
                n_clusters=n_clusters,
                scoring=scoring,
            )
            self._clustering_view_model.cluster_neurons(
                request=request,
                on_finished=on_finished,
                on_error=on_error,
            )
            if show_started_toast:
                self._log(tr("Clustering neurons..."), level=logging.INFO)
        except Exception as e:
            self.neuron_cluster_button.setEnabled(True)
            self._log(tr("Failed to start neuron clustering: {error}").format(error=e), level=logging.ERROR)

    # ---- View model signal handlers
    def _on_training_started(self) -> None:
        """React to training start."""
        self.train_button.setEnabled(False)
        self.save_button.setEnabled(False)

    def _on_training_finished(self, result) -> None:
        """React to training completion."""
        self.train_button.setEnabled(True)
        self.save_button.setEnabled(result is not None)

    def set_timeline_cluster_save_enabled(self, enabled: bool) -> None:
        self.save_timeline_clusters_button.setEnabled(bool(enabled))
