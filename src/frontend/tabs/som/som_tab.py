
from __future__ import annotations
from typing import Optional, Any, List, Tuple, Iterable, Callable

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QItemSelectionModel
from PySide6.QtWidgets import (


    QWidget,
    QLabel,
    QDialog,
    QTabWidget,
    QInputDialog,
    QSizePolicy,
)

import logging
logger = logging.getLogger(__name__)
from ...localization import tr

from ...widgets.panel import Panel
from ...models.hybrid_pandas_model import FeatureSelection
from ...models.hybrid_pandas_model import HybridPandasModel
from ...models.log_model import LogModel, get_log_model
from backend.services import ClusteringMethodSpec
from .viewmodel import SomViewModel
from .clustering_viewmodel import ClusteringViewModel
from .map_view import SomMapView
from .som_details_dialog import SomDetailsDialog
from .som_saved_maps_dialog import SomSavedMapsDialog
from .timeline_cluster_groups_dialog import TimelineClusterGroupsDialog
from ...utils.som_details import build_som_map_prompt, build_som_map_summary_text
from ...utils.exporting import export_dataframes
from ...widgets.dataframe_table_model import DataFrameTableModel
from ...widgets.export_dialog import ExportOption, ExportSelectionDialog
from ...utils import toast_error, toast_info, toast_success, toast_warn
from ...style.cluster_colors import cluster_color_for_label

# Tabs
from .component_planes_tab import ComponentPlanesTab
from .feature_map_tab import FeatureMapTabWidget
from .sidebar import SomSidebar
from .timeline_tab import TimelineTabWidget
from ..tab_widget import TabWidget



class SomTab(TabWidget):
    """Interactive tab that exposes SOM training/visualisation controls."""
    _TIMELINE_OVERLAY_MAX_FEATURES = 5

    def __init__(
        self,
        database_model: HybridPandasModel,
        parent: Optional[QWidget] = None,
        *,
        log_model: Optional[LogModel] = None,
    ):
        self._log_model = log_model or get_log_model()
        self._view_model = SomViewModel(database_model)
        self._clustering_view_model = ClusteringViewModel(self._view_model)
        self.view_model = self._view_model
        self._clustering_method_specs: dict[str, ClusteringMethodSpec] = {}
        self._result = None
        self._neuron_clusters = None
        self._feature_clusters = None
        self._som_details_dialog: Optional[SomDetailsDialog] = None
        self._timeline_display_df = self._empty_timeline_table_dataframe()

        # Attributes populated by tab builders (_create_content_widget).
        # Declaring them here keeps static analysis aligned with runtime wiring.
        self.sidebar: Optional[SomSidebar] = None
        self.metrics_label: Any = None
        self.tabs: Any = None
        self.component_planes: Any = None
        self.feature_table: Any = None
        self.feature_group_chart: Any = None
        self.timeline_chart: Any = None
        self.timeline_table: Any = None
        self.timeline_cluster_map: Any = None
        self.timeline_display_combo: Any = None
        self.timeline_rows_splitter: Any = None
        self.timeline_cluster_panel_stack: Any = None
        self.timeline_cluster_placeholder: Any = None
        self.timeline_cluster_content: Any = None
        self._feature_table_model: Optional[DataFrameTableModel] = None
        self._timeline_table_model: Optional[DataFrameTableModel] = None

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
        try:
            self._clustering_view_model.setParent(self)
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)
        self._initialise_clustering_controls()
        self._wire_signals()

    # ------------------------------------------------------------------
    def _create_sidebar(self) -> QWidget:
        self.sidebar = SomSidebar(
            view_model=self._view_model,
            clustering_view_model=self._clustering_view_model,
            log_model=self._log_model,
            data_model=self.view_model._data_model,
            parent=self,
        )
        return self.sidebar

    def _create_content_widget(self) -> QWidget:
        results_panel = Panel("")
        results_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        results_layout = results_panel.content_layout()

        self.metrics_label = QLabel(tr("No model trained yet."))
        self.metrics_label.setWordWrap(True)
        results_layout.addWidget(self.metrics_label)

        self.tabs = QTabWidget()
        results_layout.addWidget(self.tabs, 1)

        self.component_planes = ComponentPlanesTab(self._view_model)
        self.tabs.addTab(self.component_planes, tr("Maps"))

        feature_widget = FeatureMapTabWidget(self._view_model, parent=self.tabs)
        self.feature_table = feature_widget.feature_table
        self.feature_group_chart = feature_widget.feature_group_chart
        self.tabs.addTab(feature_widget, tr("Features"))

        timeline_widget = TimelineTabWidget(self._view_model, parent=self.tabs)
        self.timeline_chart = timeline_widget.timeline_chart
        self.timeline_table = timeline_widget.timeline_table
        self.timeline_cluster_map = timeline_widget.timeline_cluster_map
        self.timeline_display_combo = timeline_widget.timeline_display_combo
        self.timeline_rows_splitter = timeline_widget.timeline_rows_splitter
        self.timeline_cluster_panel_stack = timeline_widget.timeline_cluster_panel_stack
        self.timeline_cluster_placeholder = timeline_widget.timeline_cluster_placeholder
        self.timeline_cluster_content = timeline_widget.timeline_cluster_content
        self.tabs.addTab(timeline_widget, tr("Timeline"))
        self._init_table_models()
        self._update_timeline_cluster_panel_state()

        self._training_widgets: List[Tuple[Any, Optional[str]]] = []
        # All clustering controls live on the sidebar now
        if self.sidebar is not None:
            self._register_training_widget(getattr(self.sidebar, "feature_cluster_model_combo", None))
            self._register_training_widget(getattr(self.sidebar, "feature_cluster_spin", None))
            self._register_training_widget(getattr(self.sidebar, "feature_cluster_count_spin", None))
            self._register_training_widget(getattr(self.sidebar, "feature_cluster_scorer_combo", None))
            self._register_training_widget(
                getattr(self.sidebar, "feature_cluster_button", None),
                tr("Train the SOM to cluster features."),
            )
            self._register_training_widget(getattr(self.sidebar, "neuron_cluster_model_combo", None))
            self._register_training_widget(getattr(self.sidebar, "neuron_cluster_spin", None))
            self._register_training_widget(getattr(self.sidebar, "neuron_cluster_count_spin", None))
            self._register_training_widget(getattr(self.sidebar, "neuron_cluster_scorer_combo", None))
            self._register_training_widget(
                getattr(self.sidebar, "neuron_cluster_button", None),
                tr("Train the SOM to cluster neurons."),
            )
        self._register_training_widget(getattr(self, "timeline_display_combo", None))
        self._set_training_actions_enabled(False)

        return results_panel

    def _init_table_models(self) -> None:
        feature_table = getattr(self, "feature_table", None)
        if feature_table is not None and feature_table.model() is None:
            self._feature_table_model = DataFrameTableModel(pd.DataFrame(), include_index=False)
            feature_table.setModel(self._feature_table_model)

        timeline_table = getattr(self, "timeline_table", None)
        if timeline_table is not None and timeline_table.model() is None:
            self._timeline_table_model = DataFrameTableModel(
                self._empty_timeline_table_dataframe(),
                include_index=False,
            )
            timeline_table.setModel(self._timeline_table_model)

    # ------------------------------------------------------------------
    def _register_training_widget(self, widget: Any, tooltip: Optional[str] = None) -> None:
        if widget is None:
            return
        if tooltip:
            try:
                widget.setAttribute(Qt.WA_AlwaysShowToolTips, True)
            except Exception:
                logger.warning("Exception in _register_training_widget", exc_info=True)
        self._training_widgets.append((widget, tooltip))

    def _current_method_key(self, target: str) -> Optional[str]:
        combo = None
        if self.sidebar is not None:
            if target == "feature":
                combo = getattr(self.sidebar, "feature_cluster_model_combo", None)
            elif target == "neuron":
                combo = getattr(self.sidebar, "neuron_cluster_model_combo", None)
        if combo is None:
            return None
        value = combo.currentData()
        return None if value is None else str(value)

    def _current_method_spec(self, target: str) -> Optional[ClusteringMethodSpec]:
        key = self._current_method_key(target)
        if key is None:
            return None
        return self._clustering_method_specs.get(key)

    def _update_clustering_method_state(self, target: str) -> None:
        spec = self._current_method_spec(target)
        supports_auto = True if spec is None else bool(spec.supports_auto_k)
        requires_clusters = True if spec is None else bool(spec.requires_n_clusters)
        uses_clusters = supports_auto or requires_clusters

        if self.sidebar is not None:
            if target == "feature":
                max_spin = getattr(self.sidebar, "feature_cluster_spin", None)
                count_spin = getattr(self.sidebar, "feature_cluster_count_spin", None)
                scorer_combo = getattr(self.sidebar, "feature_cluster_scorer_combo", None)
            else:
                max_spin = getattr(self.sidebar, "neuron_cluster_spin", None)
                count_spin = getattr(self.sidebar, "neuron_cluster_count_spin", None)
                scorer_combo = getattr(self.sidebar, "neuron_cluster_scorer_combo", None)
        else:
            max_spin = count_spin = scorer_combo = None

        for widget, enabled in ((max_spin, supports_auto), (scorer_combo, supports_auto)):
            if widget is not None:
                widget.setEnabled(enabled)

        if count_spin is not None:
            count_spin.setEnabled(uses_clusters)
            block = count_spin.blockSignals(True)
            try:
                if not uses_clusters:
                    try:
                        count_spin.setValue(0)
                    except Exception:
                        logger.warning("Exception in _update_clustering_method_state", exc_info=True)
                elif uses_clusters:
                    try:
                        current = int(count_spin.value())
                    except Exception:
                        current = 0
                    if not supports_auto:
                        if current < 2:
                            try:
                                count_spin.setValue(2)
                            except Exception:
                                logger.warning("Exception in _update_clustering_method_state", exc_info=True)
                    else:
                        if current < 0:
                            try:
                                count_spin.setValue(0)
                            except Exception:
                                logger.warning("Exception in _update_clustering_method_state", exc_info=True)
            finally:
                count_spin.blockSignals(block)

        self._sync_cluster_controls()

    def _on_cluster_method_changed(self, target: str) -> None:
        self._update_clustering_method_state(target)

    def _initialise_clustering_controls(self) -> None:
        try:
            methods = self._clustering_view_model.available_methods()
        except Exception:
            methods = []
        self._clustering_method_specs = {spec.key: spec for spec in methods}

        if self.sidebar is not None:
            for combo in (
                getattr(self.sidebar, "feature_cluster_model_combo", None),
                getattr(self.sidebar, "neuron_cluster_model_combo", None),
            ):
                if combo is None:
                    continue
                block = combo.blockSignals(True)
                try:
                    combo.clear()
                    for spec in methods:
                        combo.addItem(spec.label, spec.key)
                    if combo.count() > 0:
                        combo.setCurrentIndex(0)
                finally:
                    combo.blockSignals(block)

        self._update_clustering_method_state("feature")
        self._update_clustering_method_state("neuron")

    # ------------------------------------------------------------------
    def _wire_signals(self) -> None:
        # Model signals
        self._view_model.training_started.connect(self._on_training_started)
        self._view_model.training_finished.connect(self._on_training_finished)
        self._view_model.error_occurred.connect(self._on_training_error)
        self._view_model.cluster_names_changed.connect(self._on_cluster_names_changed)
        self._view_model.database_changed.connect(self._on_database_changed)
        self._view_model.selected_feature_payloads_changed.connect(
            self._on_selected_features_from_viewmodel
        )
        self._view_model.timeline_display_options_changed.connect(
            self._on_timeline_display_options_changed
        )
        
        # Clustering signals come from clustering view model
        self._clustering_view_model.feature_clusters_updated.connect(self._on_feature_clusters_updated)
        self._clustering_view_model.neuron_clusters_updated.connect(self._on_neuron_clusters_updated)
        self._clustering_view_model.clustering_error.connect(self._on_clustering_error)

        if self.timeline_display_combo is not None:
            self.timeline_display_combo.selection_changed.connect(self._on_timeline_display_ui_changed)

        # Connect cluster rename signal from timeline cluster map
        timeline_cluster_map = getattr(self, "timeline_cluster_map", None)
        if timeline_cluster_map is not None:
            timeline_cluster_map.cluster_rename_requested.connect(self._on_cluster_rename_requested)
            timeline_cluster_map.details_requested.connect(self._on_timeline_details_requested)
            timeline_cluster_map.cell_clicked.connect(self._on_timeline_cluster_cell_clicked)

        # Cluster model combo signals are wired inside SomSidebar.

        if self.sidebar is not None:
            try:
                self.sidebar.save_button.clicked.connect(self._on_save_map_clicked)
                self.sidebar.load_button.clicked.connect(self._on_load_map_clicked)
                self.sidebar.export_button.clicked.connect(self._on_export_requested)
                self.sidebar.save_timeline_clusters_requested.connect(
                    self._on_save_timeline_clusters_requested
                )
            except Exception:
                logger.warning("Exception in _wire_signals", exc_info=True)

        if self.timeline_table is not None:
            self.timeline_table.selectionChangedInstant.connect(
                lambda *_args: self._on_timeline_rows_selected()
            )
        if self.feature_table is not None:
            self.feature_table.selectionChangedInstant.connect(
                lambda *_args: self._on_feature_table_selection_changed()
            )

    def close_database(self) -> None:
        """Release resources held by the SOM view model."""
        self._view_model.close_database()

    def _clear_filters_and_features(self) -> None:
        if self.sidebar is None:
            return
        try:
            self.sidebar.clear_filter_controls()
        except Exception:
            logger.warning("Exception in _clear_filters_and_features", exc_info=True)
        try:
            self.sidebar.features_widget.set_filters(
                systems=None,
                datasets=None,
                tags=None,
                reload=True,
            )
        except Exception:
            logger.warning("Exception in _clear_filters_and_features", exc_info=True)

    def _on_database_changed(self, _db) -> None:
        self._result = None
        self._neuron_clusters = None
        self._feature_clusters = None
        self._timeline_display_df = self._empty_timeline_table_dataframe()
        self.metrics_label.setText(tr("No model trained yet."))
        self._clear_filters_and_features()
        self._set_training_actions_enabled(False)
        if self.sidebar is not None:
            try:
                self.sidebar.export_button.setEnabled(False)
            except Exception:
                logger.warning("Exception in _on_database_changed", exc_info=True)

        if self.timeline_display_combo is not None:
            block = self.timeline_display_combo.blockSignals(True)
            try:
                self.timeline_display_combo.set_selected_values(["bmu"])
            finally:
                self.timeline_display_combo.blockSignals(block)
        self._view_model.set_timeline_display_options(selected=["bmu"])

        try:
            self.timeline_chart.clear()
        except Exception:
            logger.warning("Exception in _on_database_changed", exc_info=True)
        self._fill_table(self.feature_table, pd.DataFrame())
        self._view_model.set_timeline_table_dataframe(self._empty_timeline_table_dataframe())
        try:
            self.feature_group_chart.clear()
        except Exception:
            logger.warning("Exception in _on_database_changed", exc_info=True)
        self._render_result()
        self._update_timeline_cluster_map()
        self._update_timeline_cluster_save_state()

    def _on_save_map_clicked(self) -> None:
        if self._view_model.result() is None:
            toast_info(tr("Train a SOM model before saving."), title=tr("SOM"), tab_key="som")
            return
        name, ok = QInputDialog.getText(self, tr("Save SOM"), tr("Map name:"))
        if not ok:
            return
        try:
            model_id = self._view_model.save_map(name)
        except Exception as exc:
            toast_warn(
                tr("Failed to save SOM.\n\n{error}").format(error=exc),
                title=tr("Save failed"),
                tab_key="som",
            )
            return
        if model_id:
            toast_success(tr("SOM saved successfully."), title=tr("SOM saved"), tab_key="som")

    def _on_load_map_clicked(self) -> None:
        maps = self._view_model.list_saved_maps()
        if not maps:
            toast_info(tr("No saved SOM were found."), title=tr("No saved maps"), tab_key="som")
            return

        dialog = SomSavedMapsDialog(maps, parent=self)

        def _refresh() -> None:
            dialog.set_maps(self._view_model.list_saved_maps())

        def _handle_load(model_id: int) -> None:
            try:
                result = self._view_model.load_saved_map(model_id)
            except Exception as exc:
                toast_warn(
                    tr("Failed to load SOM.\n\n{error}").format(error=exc),
                    title=tr("Load failed"),
                    tab_key="som",
                )
                return
            self._apply_loaded_result(result)
            dialog.accept()

        def _handle_delete(model_id: int) -> None:
            try:
                self._view_model.delete_saved_map(model_id)
            except Exception as exc:
                toast_warn(
                    tr("Failed to delete SOM.\n\n{error}").format(error=exc),
                    title=tr("Delete failed"),
                    tab_key="som",
                )
                return
            _refresh()

        dialog.load_requested.connect(_handle_load)
        dialog.delete_requested.connect(_handle_delete)
        dialog.exec()

    def _apply_loaded_result(self, result) -> None:
        if result is None:
            return
        self._result = result
        self._neuron_clusters = self._view_model.last_neuron_clusters()
        self._apply_feature_clusters_state(
            self._view_model.last_feature_clusters(),
            announce=False,
        )
        self._set_training_actions_enabled(True)
        self._sync_cluster_controls()
        if self.sidebar is not None:
            try:
                self.sidebar.save_button.setEnabled(True)
                self.sidebar.export_button.setEnabled(True)
            except Exception:
                logger.warning("Exception in _apply_loaded_result", exc_info=True)
        if isinstance(self.component_planes, ComponentPlanesTab):
            try:
                self.component_planes.set_neuron_clusters(self._neuron_clusters)
            except Exception:
                logger.warning("Exception in _apply_loaded_result", exc_info=True)
        if self._neuron_clusters is not None:
            selected = self._view_model.timeline_display_selected()
            if "cluster" not in selected:
                selected = [key for key in selected if key != "bmu"]
                selected.insert(0, "cluster")
                self._view_model.set_timeline_display_options(selected=selected)
        self._render_result()
        self._update_timeline_cluster_panel_state()
        self._update_timeline_cluster_save_state()

    def _selected_feature_labels(self) -> list[str]:
        payloads = self._selected_feature_payloads()
        labels: list[str] = []
        for payload in payloads:
            parts: list[str] = []
            for key in ("base_name", "source", "unit", "type"):
                value = payload.get(key)
                if value and value not in parts:
                    parts.append(str(value))
            if parts:
                labels.append(" / ".join(parts))
                continue
            fid = payload.get("feature_id")
            if fid is not None:
                labels.append(tr("Feature {id}").format(id=fid))
        return labels

    def _selected_feature_payloads(self) -> list[dict]:
        return self._view_model.selected_feature_payloads()

    # ------------------------------------------------------------------
    def _sync_cluster_controls(self) -> None:
        if not self._result or self.sidebar is None:
            return

        max_k = self._spin_value(getattr(self.sidebar, "feature_cluster_spin", None))
        n_clusters = self._cluster_count_value(getattr(self.sidebar, "feature_cluster_count_spin", None))
        self._feature_cluster_limits(max_k, n_clusters)

        max_k = self._spin_value(getattr(self.sidebar, "neuron_cluster_spin", None))
        n_clusters = self._cluster_count_value(getattr(self.sidebar, "neuron_cluster_count_spin", None))
        self._neuron_cluster_limits(max_k, n_clusters)

    # ------------------------------------------------------------------
    def _spin_value(self, spin_box) -> int:
        if spin_box is None:
            return 2
        try:
            value = int(spin_box.value())
        except Exception:
            value = 2
        value = max(2, value)
        if value != spin_box.value():
            try:
                spin_box.setValue(value)
            except Exception:
                logger.warning("Exception in _spin_value", exc_info=True)
        return value

    def _cluster_count_value(self, spin_box) -> Optional[int]:
        if spin_box is None:
            return None
        try:
            value = int(spin_box.value())
        except Exception:
            return None
        if value <= 0:
            return None
        if value < 2:
            try:
                spin_box.setValue(2)
            except Exception:
                logger.warning("Exception in _cluster_count_value", exc_info=True)
            return 2
        return value

    def _current_scorer(self, combo_box) -> str:
        if combo_box is None:
            return "silhouette"
        return combo_box.currentData() or "silhouette"

    def _feature_cluster_limits(self, max_k: int, n_clusters: Optional[int]) -> tuple[int, Optional[int]]:
        limit = None
        if self._result is not None:
            positions = getattr(self._result, "feature_positions", None)
            if isinstance(positions, pd.DataFrame) and not positions.empty and "feature" in positions.columns:
                limit = max(2, int(positions["feature"].nunique()))
            if limit is None:
                norm_df = getattr(self._result, "normalized_dataframe", None)
                if isinstance(norm_df, pd.DataFrame) and not norm_df.empty:
                    limit = max(2, int(len(norm_df.columns)))
        if limit is None:
            labels = self._selected_feature_labels()
            if labels:
                limit = max(2, len(labels))
        if limit is None:
            limit = max(2, max_k)

        safe_max_k = min(max_k, limit)
        if safe_max_k != max_k and self.sidebar is not None:
            try:
                self.sidebar.feature_cluster_spin.setValue(safe_max_k)
            except Exception:
                logger.warning("Exception in _feature_cluster_limits", exc_info=True)

        safe_n_clusters = n_clusters
        if n_clusters is not None:
            safe_n_clusters = min(max(2, n_clusters), limit)
            if self.sidebar is not None:
                try:
                    self.sidebar.feature_cluster_count_spin.setValue(safe_n_clusters)
                except Exception:
                    logger.warning("Exception in _feature_cluster_limits", exc_info=True)

        return safe_max_k, safe_n_clusters

    def _neuron_cluster_limits(self, max_k: int, n_clusters: Optional[int]) -> tuple[int, Optional[int]]:
        limit = None
        if self._result is not None:
            map_shape = getattr(self._result, "map_shape", None)
            if isinstance(map_shape, (tuple, list)) and len(map_shape) >= 2:
                try:
                    map_units = int(map_shape[0]) * int(map_shape[1])
                except Exception:
                    map_units = None
                else:
                    if map_units and map_units > 0:
                        limit = max(2, map_units)
            rows_df = getattr(self._result, "row_bmus", None)
            row_count = None
            if isinstance(rows_df, pd.DataFrame):
                row_count = len(rows_df)
            if row_count and row_count > 0:
                limit = min(limit, row_count) if limit is not None else max(2, row_count)
        if limit is None:
            limit = max(2, max_k)

        safe_max_k = min(max_k, limit)
        if safe_max_k != max_k and self.sidebar is not None:
            try:
                self.sidebar.neuron_cluster_spin.setValue(safe_max_k)
            except Exception:
                logger.warning("Exception in _neuron_cluster_limits", exc_info=True)

        safe_n_clusters = n_clusters
        if n_clusters is not None:
            safe_n_clusters = min(max(2, n_clusters), limit)
            if self.sidebar is not None:
                try:
                    self.sidebar.neuron_cluster_count_spin.setValue(safe_n_clusters)
                except Exception:
                    logger.warning("Exception in _neuron_cluster_limits", exc_info=True)

        return safe_max_k, safe_n_clusters

    # ------------------------------------------------------------------
    def _parse_map_dimension(self, text: str) -> Optional[int]:
        value = (text or "").strip().lower()
        if not value or value == "auto" or value == tr("auto").lower():
            return None
        try:
            dim = int(value)
        except ValueError:
            return None
        return max(2, dim)

    # ------------------------------------------------------------------
    def _on_training_started(self) -> None:
        show_overlay = self._view_model.timeline_overlay_enabled()
        if self.sidebar is not None:
            self.sidebar.train_button.setEnabled(False)
        try:
            toast_info(tr("Training SOM…"), title=tr("SOM"), tab_key="som")
        except Exception:
            logger.warning("Exception in _on_training_started", exc_info=True)
        self._result = None
        self._neuron_clusters = None
        self._feature_clusters = None
        self._timeline_display_df = self._empty_timeline_table_dataframe()
        self._view_model.set_timeline_table_dataframe(self._timeline_display_df)
        self._on_neuron_clusters_updated(None)
        self._set_training_actions_enabled(False)
        selected = ["bmu"]
        if show_overlay:
            selected.append("selected_features")
        if self.timeline_display_combo is not None:
            block = self.timeline_display_combo.blockSignals(True)
            try:
                self.timeline_display_combo.set_selected_values(selected)
            finally:
                self.timeline_display_combo.blockSignals(block)
        self._view_model.set_timeline_display_options(selected=selected)
        self.timeline_chart.clear()
        self._update_timeline_cluster_save_state()

    def _on_training_finished(self, result) -> None:
        if self.sidebar is not None:
            self.sidebar.train_button.setEnabled(True)
        self._result = result
        self._set_training_actions_enabled(True)
        if self.sidebar is not None:
            try:
                self.sidebar.export_button.setEnabled(True)
            except Exception:
                logger.warning("Exception in _on_training_finished", exc_info=True)
        self._sync_cluster_controls()
        self._render_result()
        self._update_timeline_cluster_save_state()
        self._run_auto_clustering_after_training()
        try:
            toast_success(tr("SOM training finished."), title=tr("SOM"), tab_key="som")
        except Exception:
            logger.warning("Exception in _on_training_finished", exc_info=True)

    def _on_training_error(self, message: str) -> None:
        if self.sidebar is not None:
            self.sidebar.train_button.setEnabled(True)
        if message:
            try:
                toast_error(message, title=tr("SOM training failed"), tab_key="som")
            except Exception:
                logger.warning("Exception in _on_training_error", exc_info=True)
        self._set_training_actions_enabled(self._result is not None)

    def _on_clustering_error(self, message: str) -> None:
        """Handle clustering errors from the clustering view model."""
        if message:
            try:
                toast_error(message, title=tr("Clustering failed"), tab_key="som")
            except Exception:
                logger.warning("Exception in _on_clustering_error", exc_info=True)

    def _run_auto_clustering_after_training(self) -> None:
        sidebar = self.sidebar
        if sidebar is None or self._result is None:
            return
        try:
            if sidebar.auto_cluster_features_enabled():
                sidebar.start_feature_clustering(show_started_toast=False)
            if sidebar.auto_cluster_timeline_enabled():
                sidebar.start_neuron_clustering(show_started_toast=False)
        except Exception:
            logger.warning("Exception in _run_auto_clustering_after_training", exc_info=True)

    # ------------------------------------------------------------------
    def _set_training_actions_enabled(self, enabled: bool) -> None:
        for widget, tooltip in getattr(self, "_training_widgets", []):
            if widget is not None:
                widget.setEnabled(enabled)
                if tooltip:
                    try:
                        widget.setToolTip("" if enabled else tooltip)
                    except Exception:
                        logger.warning("Exception in _set_training_actions_enabled", exc_info=True)
        # Auto-clustering toggles should always remain user-editable.
        if self.sidebar is not None:
            for checkbox_name in ("auto_cluster_features_checkbox", "auto_cluster_timeline_checkbox"):
                checkbox = getattr(self.sidebar, checkbox_name, None)
                if checkbox is not None:
                    checkbox.setEnabled(True)
            # Clustering model choices should remain editable even before SOM training.
            for combo_name in ("feature_cluster_model_combo", "neuron_cluster_model_combo"):
                combo = getattr(self.sidebar, combo_name, None)
                if combo is not None:
                    combo.setEnabled(True)
                    try:
                        combo.setToolTip("")
                    except Exception:
                        logger.warning("Exception in _set_training_actions_enabled", exc_info=True)
        if enabled:
            self._update_clustering_method_state("feature")
            self._update_clustering_method_state("neuron")

    # ------------------------------------------------------------------
    def _render_result(self) -> None:
        if not self._result:
            # ensure component-planes UI clears too
            if isinstance(self.component_planes, ComponentPlanesTab):
                self.component_planes.clear()
            return

        res = self._result
        qe = getattr(res, "quantization_error", float("nan"))
        te = getattr(res, "topographic_error", float("nan"))
        self.metrics_label.setText(
            tr(
                "Grid {rows}×{cols} · Quantisation error {qe:.4f} · Topographic error {te:.4f}"
            ).format(rows=res.map_shape[0], cols=res.map_shape[1], qe=qe, te=te)
        )

        # Delegate planes + aux map rendering
        if isinstance(self.component_planes, ComponentPlanesTab):
            self.component_planes.set_result(res)

        # --- The rest of the tabs rely on tables/charts owned by SomTab ----
        self._update_feature_table()
        self._update_timeline_views()
        self._sync_cluster_controls()
        self._update_timeline_cluster_map()

    def _on_export_requested(self) -> None:
        if not self._result:
            toast_info(tr("Train a SOM model before exporting."), title=tr("SOM"), tab_key="som")
            return

        datasets: dict[str, pd.DataFrame] = {}
        feature_df = self._feature_positions_dataframe()
        if not feature_df.empty:
            datasets[tr("Feature clusters")] = feature_df

        timeline_df = self._timeline_dataframe()
        if not timeline_df.empty:
            datasets[tr("Timeline")] = timeline_df
        elif self._timeline_display_df is not None and not self._timeline_display_df.empty:
            datasets[tr("Timeline")] = self._timeline_display_df.copy()

        correlations = getattr(self._result, "correlations", pd.DataFrame())
        if isinstance(correlations, pd.DataFrame) and not correlations.empty:
            corr_export = correlations.copy().reset_index().rename(columns={"index": "feature"})
            datasets[tr("Feature correlations")] = corr_export

        distance_map = getattr(self._result, "distance_map", None)
        if isinstance(distance_map, pd.DataFrame) and not distance_map.empty:
            datasets[tr("Distance map")] = distance_map.copy().reset_index(drop=True)

        activation = getattr(self._result, "activation_response", None)
        if isinstance(activation, pd.DataFrame) and not activation.empty:
            datasets[tr("Activation response")] = activation.copy().reset_index(drop=True)

        quant = getattr(self._result, "quantization_map", None)
        if isinstance(quant, pd.DataFrame) and not quant.empty:
            datasets[tr("Quantization map")] = quant.copy().reset_index(drop=True)

        metrics_df = pd.DataFrame(
            [
                {"metric": "quantization_error", "value": getattr(self._result, "quantization_error", None)},
                {"metric": "topographic_error", "value": getattr(self._result, "topographic_error", None)},
            ]
        )
        datasets[tr("SOM metrics")] = metrics_df

        if not datasets:
            toast_info(tr("No SOM outputs available to export."), title=tr("SOM"), tab_key="som")
            return

        options = [
            ExportOption(key=name, label=name, description=tr("SOM output")) for name in datasets.keys()
        ]
        dialog = ExportSelectionDialog(
            title=tr("Export SOM results"),
            heading=tr("Choose which SOM outputs to export."),
            options=options,
            parent=self,
        )
        if dialog.exec() != ExportSelectionDialog.DialogCode.Accepted:
            return

        selected = dialog.selected_keys()
        if not selected:
            toast_info(tr("Select at least one export item."), title=tr("Export"), tab_key="som")
            return

        chosen = {name: datasets[name] for name in selected if name in datasets}
        ok, message = export_dataframes(
            parent=self,
            title=tr("Export SOM"),
            selected_format=dialog.selected_format(),
            datasets=chosen,
        )
        if not message:
            return
        if ok:
            toast_success(message, title=tr("Export complete"), tab_key="som")
        else:
            toast_warn(message, title=tr("Export"), tab_key="som")

    # -------------------------- generic tables/heatmaps (unchanged) -------
    def _on_neuron_clusters_updated(self, clusters) -> None:
        self._neuron_clusters = clusters
        # Update the view model's internal state so get_unique_cluster_ids() works
        self._view_model._last_neuron_clusters = clusters
        # Clear cluster names when new clustering is performed
        self._view_model._cluster_names = {}
        if isinstance(self.component_planes, ComponentPlanesTab):
            try:
                self.component_planes.set_neuron_clusters(clusters)
            except Exception:
                logger.warning("Exception in _on_neuron_clusters_updated", exc_info=True)
        # Auto-switch to cluster mode when clusters are available
        if clusters is not None:
            selected = self._view_model.timeline_display_selected()
            if "cluster" not in selected:
                selected = [key for key in selected if key != "bmu"]
                selected.insert(0, "cluster")
                self._view_model.set_timeline_display_options(selected=selected)
        self._update_timeline_views()
        self._sync_cluster_controls()
        self._on_timeline_rows_selected()
        self._update_timeline_cluster_panel_state()
        self._update_timeline_cluster_save_state()
        if clusters is not None:
            try:
                toast_success(
                    tr("Neuron clustering finished."),
                    title=tr("SOM"),
                    tab_key="som",
                    on_click=lambda: self._activate_subtab("timeline"),
                )
            except Exception:
                logger.warning("Exception in _on_neuron_clusters_updated", exc_info=True)

    def _on_cluster_names_changed(self) -> None:
        """Called when cluster names are updated in the view model."""
        preserved_range: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None
        try:
            if self._view_model.timeline_show_clusters():
                preserved_range = self._capture_timeline_x_range()
        except Exception:
            logger.warning("Exception in _on_cluster_names_changed", exc_info=True)
        self._update_timeline_cluster_map()
        self._update_timeline_views()
        if preserved_range is not None:
            self._restore_timeline_x_range(preserved_range)

    def _on_cluster_rename_requested(self, cluster_id: int, new_name: str) -> None:
        """Handle cluster rename request from cluster map context menu."""
        self._view_model.set_cluster_name(cluster_id, new_name)
        # Update will happen via cluster_names_changed signal

    def _on_feature_clusters_updated(self, clusters) -> None:
        """Called when feature clustering completes."""
        self._apply_feature_clusters_state(clusters, announce=True)

    def _feature_positions_dataframe(self) -> pd.DataFrame:
        if not self._result:
            return pd.DataFrame()
        positions = self._result.feature_positions.copy()
        if positions.empty:
            return positions
        if self._feature_clusters is not None:
            cluster_df = pd.DataFrame(
                {
                    "feature": list(self._feature_clusters.index),
                    "cluster": list(self._feature_clusters.labels),
                }
            )
            positions = positions.merge(cluster_df, on="feature", how="left")
        if "cluster" not in positions.columns:
            positions = positions.copy()
            positions["cluster"] = pd.NA
        else:
            positions = positions.copy()
        positions["cluster"] = positions["cluster"].apply(self._format_cluster_label)
        positions = positions.sort_values(by=["x", "y", "feature"])
        return positions

    def _apply_feature_clusters_state(self, clusters, *, announce: bool) -> None:
        self._feature_clusters = clusters
        # Keep the SOM view model in sync so saved maps persist feature clusters.
        self._view_model._last_feature_clusters = clusters
        self._update_feature_table()
        self._sync_cluster_controls()
        if announce and clusters is not None:
            try:
                toast_success(
                    tr("Feature clustering finished."),
                    title=tr("SOM"),
                    tab_key="som",
                    on_click=lambda: self._activate_subtab("features"),
                )
            except Exception:
                logger.warning("Exception in _apply_feature_clusters_state", exc_info=True)

    def _activate_subtab(self, key: str) -> None:
        tabs = getattr(self, "tabs", None)
        if tabs is None:
            return
        target = str(key or "").strip().lower()
        if not target:
            return
        for index in range(tabs.count()):
            text = str(tabs.tabText(index) or "").strip().lower()
            if text == target:
                tabs.setCurrentIndex(index)
                return

    def _update_feature_table(self) -> None:
        df = self._feature_positions_dataframe()
        drop_columns = [c for c in ("label", "max_value", "mean_value", "min_value") if c in df.columns]
        if drop_columns:
            df = df.drop(columns=drop_columns)
        if not df.empty and "feature" in df.columns:
            id_map = self._feature_id_by_name()
            if id_map:
                df = df.copy()
                df["feature_id"] = df["feature"].map(id_map)
        if not df.empty and "feature" in df.columns:
            stats = self._feature_stats_by_name()
            if stats:
                df = df.copy()
                df["min"] = df["feature"].map(stats.get("min", {}))
                df["mean"] = df["feature"].map(stats.get("mean", {}))
                df["max"] = df["feature"].map(stats.get("max", {}))
        if "feature" in df.columns:
            try:
                df = df.copy()
                df["feature"] = df["feature"].apply(self._view_model.feature_display_name)
            except Exception:
                logger.warning("Exception in _update_feature_table", exc_info=True)
        if not df.empty:
            preferred_order = ["feature_id", "feature", "x", "y", "min", "mean", "max", "cluster"]
            ordered = [col for col in preferred_order if col in df.columns]
            ordered.extend([col for col in df.columns if col not in ordered])
            df = df.loc[:, ordered]
        self._fill_table(self.feature_table, df)
        self._update_feature_group_chart()

    def _feature_id_by_name(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        payloads = self._selected_feature_payloads()
        for payload in payloads:
            try:
                selection = FeatureSelection.from_payload(payload)
            except TypeError:
                selection = FeatureSelection(
                    feature_id=payload.get("feature_id"),
                    label=payload.get("label"),
                    base_name=payload.get("base_name"),
                    source=payload.get("source"),
                    unit=payload.get("unit"),
                    type=payload.get("type"),
                    lag_seconds=payload.get("lag_seconds"),
                )
            feature_id = self._view_model._safe_feature_id(payload)
            if feature_id is None:
                continue
            feature_name = str(selection.display_name() or "").strip()
            if feature_name:
                mapping[feature_name] = int(feature_id)
        return mapping

    def _feature_stats_by_name(self) -> dict[str, dict[str, float]]:
        if not self._result:
            return {"min": {}, "mean": {}, "max": {}}
        value_df = self._view_model.last_dataframe()
        if value_df is None or value_df.empty:
            value_df = getattr(self._result, "normalized_dataframe", pd.DataFrame())
        if value_df is None or value_df.empty:
            return {"min": {}, "mean": {}, "max": {}}
        min_out: dict[str, float] = {}
        mean_out: dict[str, float] = {}
        max_out: dict[str, float] = {}
        for column in value_df.columns:
            numeric = pd.to_numeric(value_df[column], errors="coerce")
            finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
            if finite.empty:
                continue
            name = str(column)
            min_out[name] = float(finite.min())
            mean_out[name] = float(finite.mean())
            max_out[name] = float(finite.max())
        return {"min": min_out, "mean": mean_out, "max": max_out}

    def _feature_table_selected_display_names(self) -> list[str]:
        table = getattr(self, "feature_table", None)
        if table is None:
            return []
        model = table.model()
        if model is None:
            return []
        selection = table.selectionModel()
        if selection is None:
            return []

        feature_col = None
        for col in range(model.columnCount()):
            header = model.headerData(col, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
            if str(header).strip().lower() == "feature":
                feature_col = col
                break
        if feature_col is None:
            return []

        names: list[str] = []
        for row_idx in sorted({index.row() for index in selection.selectedRows()}):
            idx = model.index(row_idx, feature_col)
            if not idx.isValid():
                continue
            name = str(idx.data(Qt.ItemDataRole.DisplayRole) or "").strip()
            if name and name not in names:
                names.append(name)
        return names

    def _feature_values_dataframe(self) -> pd.DataFrame:
        if not self._result:
            return pd.DataFrame()
        value_df = self._view_model.last_dataframe()
        if value_df is None or value_df.empty:
            value_df = getattr(self._result, "normalized_dataframe", pd.DataFrame())
        if value_df is None or value_df.empty:
            return pd.DataFrame()
        return value_df.copy()

    def _feature_group_chart_dataframe(
        self,
        selected_display_names: Optional[list[str]] = None,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        if not self._result:
            return pd.DataFrame(), {}

        value_df = self._feature_values_dataframe()
        if value_df.empty:
            return pd.DataFrame(), {}

        numeric_df = value_df.apply(pd.to_numeric, errors="coerce")
        if numeric_df.empty:
            return pd.DataFrame(), {}

        available_actual = [str(col) for col in numeric_df.columns]
        display_to_actual: dict[str, str] = {}
        for actual_name in available_actual:
            display = self._view_model.feature_display_name(actual_name)
            display_to_actual[str(display)] = actual_name

        selected = [name for name in (selected_display_names or []) if name in display_to_actual]
        if not selected:
            return pd.DataFrame(), {}

        selected_actual = [display_to_actual[name] for name in selected if name in display_to_actual]
        if not selected_actual:
            return pd.DataFrame(), {}

        chart_df = pd.DataFrame({"feature": selected})
        cluster_columns: dict[str, int] = {}

        labels = getattr(self._neuron_clusters, "bmu_cluster_labels", None) if self._neuron_clusters is not None else None
        if labels is None or len(labels) <= 0:
            base_means = numeric_df[selected_actual].mean(skipna=True)
            chart_df[tr("Mean")] = [float(base_means.get(name, np.nan)) for name in selected_actual]
            return chart_df, cluster_columns

        row_count = min(len(numeric_df), len(labels))
        if row_count <= 0:
            base_means = numeric_df[selected_actual].mean(skipna=True)
            chart_df[tr("Mean")] = [float(base_means.get(name, np.nan)) for name in selected_actual]
            return chart_df, cluster_columns
        selected_values = numeric_df.iloc[:row_count][selected_actual].copy()
        cluster_series = pd.to_numeric(pd.Series(labels[:row_count]), errors="coerce")
        if cluster_series.isna().all():
            base_means = numeric_df[selected_actual].mean(skipna=True)
            chart_df[tr("Mean")] = [float(base_means.get(name, np.nan)) for name in selected_actual]
            return chart_df, cluster_columns

        unique_clusters = sorted(cluster_series.dropna().astype(int).unique().tolist())
        cluster_array = cluster_series.to_numpy(dtype=float, copy=False)
        label_to_mask: dict[str, np.ndarray] = {}
        for cluster_id in unique_clusters:
            mask = cluster_array == float(int(cluster_id))
            if not bool(mask.any()):
                continue
            cluster_name = self._view_model.get_cluster_name(int(cluster_id)).strip()
            label = cluster_name or f"{tr('Cluster')} {int(cluster_id)}"
            if label in label_to_mask:
                label_to_mask[label] = np.logical_or(label_to_mask[label], mask)
            else:
                label_to_mask[label] = mask

        for idx, (column_name, mask) in enumerate(label_to_mask.items(), start=1):
            cluster_means = selected_values.iloc[mask].mean(skipna=True)
            chart_df[column_name] = [float(cluster_means.get(name, np.nan)) for name in selected_actual]
            cluster_columns[column_name] = int(idx)

        return chart_df, cluster_columns

    def _update_feature_group_chart(self) -> None:
        chart = getattr(self, "feature_group_chart", None)
        if chart is None:
            return
        selected = self._feature_table_selected_display_names()
        chart_df, cluster_columns = self._feature_group_chart_dataframe(selected)
        if chart_df.empty or "feature" not in chart_df.columns:
            chart.clear()
            return
        value_cols = [col for col in chart_df.columns if col != "feature"]
        if not value_cols:
            chart.clear()
            return
        series_colors = {
            str(column): cluster_color_for_label(cluster_id, border=False)
            for column, cluster_id in cluster_columns.items()
        }
        chart.set_title(tr("Feature means by neuron cluster"))
        chart.set_y_label(tr("Mean value"))
        chart.set_multi_series(
            chart_df,
            category_col="feature",
            value_cols=value_cols,
            series_colors=series_colors or None,
        )

    def _on_feature_table_selection_changed(self) -> None:
        self._update_feature_group_chart()

    def _format_cluster_label(self, value: object) -> object:
        if value is None or pd.isna(value):
            return pd.NA
        try:
            cluster_id = int(value)
        except Exception:
            return value
        name = self._view_model.get_cluster_name(cluster_id)
        if name:
            return name
        return cluster_id

    def _timeline_dataframe(self) -> pd.DataFrame:
        if not self._result:
            return pd.DataFrame()
        row_df = self._result.row_bmus.copy()
        if row_df.empty:
            return row_df
        width = 0
        if isinstance(getattr(self._result, "map_shape", None), tuple) and len(self._result.map_shape) >= 2:
            try:
                width = int(self._result.map_shape[1])
            except Exception:
                width = 0
        if "bmu" not in row_df.columns:
            row_df = row_df.copy()
            bmu_x = pd.to_numeric(row_df.get("bmu_x"), errors="coerce")
            bmu_y = pd.to_numeric(row_df.get("bmu_y"), errors="coerce")
            row_df["bmu"] = bmu_x * max(1, width) + bmu_y
        else:
            bmu_numeric = pd.to_numeric(row_df.get("bmu"), errors="coerce")
            missing_mask = bmu_numeric.isna()
            if bool(missing_mask.any()):
                row_df = row_df.copy()
                bmu_x = pd.to_numeric(row_df.get("bmu_x"), errors="coerce")
                bmu_y = pd.to_numeric(row_df.get("bmu_y"), errors="coerce")
                computed = bmu_x * max(1, width) + bmu_y
                row_df.loc[missing_mask, "bmu"] = computed.loc[missing_mask]
        if self._neuron_clusters is not None:
            labels = getattr(self._neuron_clusters, "bmu_cluster_labels", None)
            if labels is not None and len(labels) == len(row_df):
                row_df = row_df.copy()
                row_df["cluster"] = pd.Series(labels, index=row_df.index)
        if "cluster" not in row_df.columns:
            row_df = row_df.copy()
            row_df["cluster"] = pd.NA
        row_df = row_df.copy()
        row_df["cluster"] = row_df["cluster"].apply(self._format_cluster_label)
        return row_df

    def _timeline_overlay_dataframe(self) -> tuple[pd.DataFrame, list[str]]:
        if self.sidebar is None:
            return pd.DataFrame(), []

        payloads = self._selected_feature_payloads()
        if not payloads:
            return pd.DataFrame(), []

        selector_filters = self.sidebar.data_selector.build_data_filters()
        try:
            df = self.sidebar.data_selector.fetch_base_dataframe_for_features(
                payloads,
            )
        except Exception:
            return pd.DataFrame(), []

        if df is None or df.empty:
            return pd.DataFrame(), []

        working = df.copy()
        working["t"] = pd.to_datetime(working.get("t"), errors="coerce")
        working = working.dropna(subset=["t"]).sort_values("t")
        if working.empty:
            return working, []

        months = set((selector_filters.months or []) if selector_filters is not None else [])
        if months:
            working = working[working["t"].dt.month.isin(months)]
        if working.empty:
            return working, []
        working = working.sort_values("t")

        preferred_columns: list[str] = []
        for payload in payloads:
            try:
                selection = FeatureSelection.from_payload(payload)
            except TypeError:
                selection = FeatureSelection(
                    feature_id=payload.get("feature_id"),
                    label=payload.get("label"),
                    base_name=payload.get("base_name"),
                    source=payload.get("source"),
                    unit=payload.get("unit"),
                    type=payload.get("type"),
                )
            preferred_columns.append(selection.display_name())
        # Keep overlay deterministic and user-driven: first N selected features.
        selected_columns = [col for col in preferred_columns if col in working.columns]
        if not selected_columns:
            selected_columns = [col for col in working.columns if col != "t"]
        selected_columns = selected_columns[: self._TIMELINE_OVERLAY_MAX_FEATURES]
        if not selected_columns:
            return pd.DataFrame(), []
        return working[["t", *selected_columns]].copy(), selected_columns

    @staticmethod
    def _coerce_utc_naive_timestamps(series: pd.Series) -> pd.Series:
        ts = pd.to_datetime(series, errors="coerce", utc=True)
        try:
            return ts.dt.tz_localize(None)
        except Exception:
            return pd.to_datetime(series, errors="coerce")

    def _merge_timeline_overlay(
        self,
        base_df: pd.DataFrame,
        overlay_df: pd.DataFrame,
        overlay_cols: list[str],
    ) -> pd.DataFrame:
        merged = base_df.copy()
        if overlay_df is None or overlay_df.empty or not overlay_cols:
            return merged

        left = merged.copy()
        left["t"] = self._coerce_utc_naive_timestamps(left["t"])
        left = left.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
        if left.empty:
            return merged

        right = overlay_df.loc[:, [c for c in ["t", *overlay_cols] if c in overlay_df.columns]].copy()
        if "t" not in right.columns or len(right.columns) <= 1:
            return left
        right["t"] = self._coerce_utc_naive_timestamps(right["t"])
        right = right.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
        if right.empty:
            return left

        exact = left.merge(right, on="t", how="left")
        if any(exact[col].notna().any() for col in overlay_cols if col in exact.columns):
            return exact

        numeric_right = right.copy()
        for col in overlay_cols:
            if col in numeric_right.columns:
                numeric_right[col] = pd.to_numeric(numeric_right[col], errors="coerce")
        grouped_right = (
            numeric_right.groupby("t", as_index=False)[overlay_cols]
            .mean(numeric_only=True)
            .sort_values("t")
            .reset_index(drop=True)
        )
        if grouped_right.empty:
            return exact

        try:
            return pd.merge_asof(
                left.sort_values("t"),
                grouped_right.sort_values("t"),
                on="t",
                direction="nearest",
            )
        except Exception:
            return exact

    def _timeline_cluster_map_dataframe(self) -> Optional[pd.DataFrame]:
        if self._neuron_clusters is None:
            return None
        grid = getattr(self._neuron_clusters, "labels_grid", None)
        if isinstance(grid, pd.DataFrame) and not grid.empty:
            return grid
        return None

    def _cluster_map_tooltip(self, clusters: pd.DataFrame) -> Callable[[int, int], str]:
        hits_df = getattr(self._result, "activation_response", None) if self._result else None
        distance_df = getattr(self._result, "distance_map", None) if self._result else None
        qe_df = getattr(self._result, "quantization_map", None) if self._result else None
        view_model = self._view_model

        def formatter(row: int, col: int) -> str:
            if clusters is None or row >= clusters.shape[0] or col >= clusters.shape[1]:
                return ""
            parts = [tr("Neuron ({col}, {row})").format(col=col, row=row)]
            if distance_df is not None and not distance_df.empty and row < distance_df.shape[0] and col < distance_df.shape[1]:
                try:
                    parts.append(
                        tr("Distance: {value:.4f}").format(
                            value=float(distance_df.iat[row, col])
                        )
                    )
                except Exception:
                    logger.warning("Exception in formatter", exc_info=True)
            if hits_df is not None and not hits_df.empty and row < hits_df.shape[0] and col < hits_df.shape[1]:
                parts.append(tr("Hits: {value}").format(value=int(hits_df.iat[row, col])))
            if qe_df is not None and not qe_df.empty and row < qe_df.shape[0] and col < qe_df.shape[1]:
                try:
                    parts.append(
                        tr("QE: {value:.4f}").format(
                            value=float(qe_df.iat[row, col])
                        )
                    )
                except Exception:
                    logger.warning("Exception in formatter", exc_info=True)
            label = clusters.iat[row, col]
            if pd.notna(label):
                try:
                    label_int = int(label)
                    # Get custom name if available
                    custom_name = view_model.get_cluster_name(label_int)
                    if custom_name:
                        parts.append(
                            tr("Cluster: {label}").format(label=custom_name)
                        )
                    else:
                        parts.append(tr("Cluster: {label}").format(label=label_int))
                except (ValueError, TypeError):
                    parts.append(tr("Cluster: {label}").format(label=label))
            return "\n".join(parts)

        return formatter

    def _update_timeline_views(self) -> None:
        row_df = self._timeline_dataframe()
        splitter = getattr(self, "timeline_rows_splitter", None)
        splitter_sizes = splitter.sizes() if splitter is not None else []
        if row_df.empty:
            self._timeline_display_df = self._empty_timeline_table_dataframe()
            self._view_model.set_timeline_table_dataframe(self._timeline_display_df)
            try:
                self.timeline_chart.clear()
            except Exception:
                logger.warning("Exception in _update_timeline_views", exc_info=True)
            self._update_timeline_cluster_map()
            self._update_timeline_row_highlight([])
            if splitter is not None and splitter_sizes:
                try:
                    splitter.setSizes(splitter_sizes)
                except Exception:
                    logger.warning("Exception in _update_timeline_views", exc_info=True)
            return

        preview_limit = 10_000
        display_df = row_df.head(preview_limit).copy()
        if "index" in display_df.columns:
            display_df = display_df.assign(index=display_df["index"].apply(lambda v: str(v)))
        self._timeline_display_df = display_df
        self._view_model.set_timeline_table_dataframe(self._timeline_display_df)

        self._update_timeline_chart(row_df)
        self._update_timeline_cluster_map()
        self._update_timeline_row_highlight([])
        if splitter is not None and splitter_sizes:
            try:
                splitter.setSizes(splitter_sizes)
            except Exception:
                logger.warning("Exception in _update_timeline_views", exc_info=True)

    def _update_timeline_chart(self, timeline_df: Optional[pd.DataFrame] = None) -> None:
        if timeline_df is None:
            timeline_df = self._timeline_dataframe()
        if timeline_df is None or timeline_df.empty:
            self.timeline_chart.clear()
            return

        if "index" in timeline_df.columns:
            times = pd.to_datetime(timeline_df["index"], errors="coerce")
        else:
            times = pd.to_datetime(timeline_df.index, errors="coerce")

        selected_options = set(self._view_model.timeline_display_selected())
        show_cluster = "cluster" in selected_options
        show_bmu = "bmu" in selected_options
        if not selected_options:
            self.timeline_chart.clear()
            return

        mode = "cluster" if show_cluster else "bmu"
        if show_cluster:
            valid = (
                self._neuron_clusters is not None
                and getattr(self._neuron_clusters, "bmu_cluster_labels", None) is not None
                and len(self._neuron_clusters.bmu_cluster_labels) == len(timeline_df)
            )
            if not valid:
                selected = [key for key in self._view_model.timeline_display_selected() if key != "cluster"]
                if not selected:
                    selected = ["bmu"]
                self._view_model.set_timeline_display_options(selected=selected)
                mode = "bmu"
                show_cluster = False

        width = 0
        if self._result and isinstance(self._result.map_shape, tuple) and len(self._result.map_shape) >= 2:
            width = int(self._result.map_shape[1])
        bmu_x = pd.to_numeric(
            timeline_df.get("bmu_x", pd.Series(index=timeline_df.index, dtype=float)),
            errors="coerce",
        )
        bmu_y = pd.to_numeric(
            timeline_df.get("bmu_y", pd.Series(index=timeline_df.index, dtype=float)),
            errors="coerce",
        )
        bmu_numeric_all = pd.to_numeric(bmu_x * max(1, width) + bmu_y, errors="coerce")

        if show_cluster:
            base_col_name = tr("Cluster")
            base_series_raw = pd.Series(
                self._neuron_clusters.bmu_cluster_labels,
                index=timeline_df.index,
                dtype=float,
            )
            base_series = pd.to_numeric(base_series_raw, errors="coerce")
            cluster_group_names: dict[int, str] = {}
            try:
                unique_cluster_ids = sorted(
                    {
                        int(value)
                        for value in base_series.dropna().tolist()
                    }
                )
            except Exception:
                unique_cluster_ids = []
            for cluster_id in unique_cluster_ids:
                custom = self._view_model.get_cluster_name(cluster_id).strip()
                cluster_group_names[int(cluster_id)] = custom or f"{tr('Cluster')} {cluster_id}"
        else:
            base_col_name = tr("BMU")
            base_series = bmu_numeric_all
            cluster_group_names = {}

        base_numeric = pd.to_numeric(base_series, errors="coerce")
        mask = times.notna() & base_numeric.notna()
        if not mask.any():
            self.timeline_chart.clear()
            return

        chart_df = pd.DataFrame(
            {
                "t": pd.to_datetime(times[mask], errors="coerce"),
                base_col_name: base_numeric[mask].to_numpy(dtype=float),
            }
        )
        hover_group_df: Optional[pd.DataFrame] = None
        cluster_display = timeline_df.get("cluster")
        if cluster_display is not None:
            try:
                hover_group_df = pd.DataFrame(
                    {
                        "t": pd.to_datetime(times[mask], errors="coerce"),
                        "group": cluster_display[mask].to_numpy(),
                    }
                )
            except Exception:
                logger.warning("Exception in _update_timeline_chart", exc_info=True)
                hover_group_df = None

        overlay_df = pd.DataFrame()
        overlay_columns: list[str] = []
        extra_overlay_cols: list[str] = []
        if show_cluster and show_bmu:
            bmu_overlay_col = tr("BMU")
            chart_df[bmu_overlay_col] = bmu_numeric_all[mask].to_numpy(dtype=float)
            extra_overlay_cols.append(bmu_overlay_col)
        if self._view_model.timeline_overlay_enabled():
            overlay_df, overlay_columns = self._timeline_overlay_dataframe()

        if (overlay_df is not None and not overlay_df.empty) or extra_overlay_cols:
            available_cols = [col for col in overlay_columns if col in overlay_df.columns]
            if not available_cols:
                available_cols = (
                    [col for col in overlay_df.columns if col != "t"][: self._TIMELINE_OVERLAY_MAX_FEATURES]
                    if overlay_df is not None and not overlay_df.empty
                    else []
                )
            else:
                available_cols = available_cols[: self._TIMELINE_OVERLAY_MAX_FEATURES]

            scale_input_cols = list(dict.fromkeys([*available_cols, *extra_overlay_cols]))

            if scale_input_cols:
                merged = chart_df.copy()
                if overlay_df is not None and not overlay_df.empty and available_cols:
                    merged = self._merge_timeline_overlay(merged, overlay_df, available_cols)

                base_vals = merged[base_col_name].to_numpy(dtype=float)
                if base_vals.size:
                    base_min = float(np.nanmin(base_vals))
                    base_max = float(np.nanmax(base_vals))
                else:
                    base_min, base_max = 0.0, 1.0
                if not np.isfinite(base_min):
                    base_min = 0.0
                if not np.isfinite(base_max):
                    base_max = base_min + 1.0
                base_range = base_max - base_min
                if not np.isfinite(base_range) or base_range == 0.0:
                    base_range = 1.0

                for col in scale_input_cols:
                    numeric = pd.to_numeric(merged[col], errors="coerce")
                    finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
                    if finite.empty:
                        scaled = pd.Series(np.nan, index=merged.index)
                    else:
                        vmin = float(finite.min())
                        vmax = float(finite.max())
                        if not np.isfinite(vmin):
                            vmin = 0.0
                        if not np.isfinite(vmax):
                            vmax = vmin + 1.0
                        denom = vmax - vmin
                        if denom == 0.0:
                            denom = 1.0
                        scaled = (numeric - vmin) / denom * base_range + base_min
                    merged[f"{col} (scaled)"] = scaled

                merged = merged.drop(columns=scale_input_cols)
                chart_df = merged

        chart_df = chart_df.sort_values("t")
        chart_df = chart_df.dropna(subset=["t"]).reset_index(drop=True)
        try:
            visible_series = [col for col in chart_df.columns if col != "t"]
            self.timeline_chart.set_max_series(max(1, len(visible_series)))
        except Exception:
            logger.warning("Exception in _update_timeline_chart", exc_info=True)

        if mode == "cluster":
            overlay_scaled_cols = [
                col for col in chart_df.columns if col not in {"t", base_col_name} and col.endswith("(scaled)")
            ]
            self.timeline_chart.set_group_timeline(
                chart_df,
                time_col="t",
                group_col=base_col_name,
                group_names=cluster_group_names,
                overlay_cols=overlay_scaled_cols,
            )
            self.timeline_chart.set_hover_group_series(None)
            return

        self.timeline_chart.set_dataframe(chart_df)
        self.timeline_chart.set_hover_group_series(hover_group_df, time_col="t", group_col="group")

    def _capture_timeline_x_range(self) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
        chart = getattr(self, "timeline_chart", None)
        axis = getattr(chart, "axis_x", None) if chart is not None else None
        if axis is None:
            return None
        try:
            qmin = axis.min()
            qmax = axis.max()
            start = pd.to_datetime(int(qmin.toMSecsSinceEpoch()), unit="ms")
            end = pd.to_datetime(int(qmax.toMSecsSinceEpoch()), unit="ms")
        except Exception:
            return None
        if pd.isna(start) or pd.isna(end) or end <= start:
            return None
        return start, end

    def _restore_timeline_x_range(self, value: Optional[tuple[pd.Timestamp, pd.Timestamp]]) -> None:
        if value is None:
            return
        chart = getattr(self, "timeline_chart", None)
        if chart is None:
            return
        start, end = value
        try:
            chart.set_x_range(start, end)
        except Exception:
            logger.warning("Exception in _restore_timeline_x_range", exc_info=True)

    def _on_timeline_display_ui_changed(self, *_args: Any) -> None:
        selected = (
            self.timeline_display_combo.selected_values()
            if self.timeline_display_combo is not None
            else ["bmu"]
        )
        self._view_model.set_timeline_display_options(selected=selected)

    def _on_timeline_display_options_changed(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        selected = payload.get("selected") if isinstance(payload.get("selected"), list) else ["bmu"]
        if self.timeline_display_combo is not None:
            block = self.timeline_display_combo.blockSignals(True)
            try:
                self.timeline_display_combo.set_selected_values(selected)
            finally:
                self.timeline_display_combo.blockSignals(block)
        self._update_timeline_chart()

    def _timeline_clusters_available(self) -> bool:
        if self._result is None or self._neuron_clusters is None:
            return False
        row_bmus = getattr(self._result, "row_bmus", None)
        labels = getattr(self._neuron_clusters, "bmu_cluster_labels", None)
        if not isinstance(row_bmus, pd.DataFrame) or row_bmus.empty:
            return False
        if labels is None or len(labels) != len(row_bmus):
            return False
        timestamps = pd.to_datetime(row_bmus.get("index"), errors="coerce")
        return bool(timestamps.notna().any())

    def _update_timeline_cluster_save_state(self) -> None:
        if self.sidebar is None:
            return
        try:
            self.sidebar.set_timeline_cluster_save_enabled(self._timeline_clusters_available())
        except Exception:
            logger.warning("Exception in _update_timeline_cluster_save_state", exc_info=True)

    def _on_save_timeline_clusters_requested(self) -> None:
        if not self._timeline_clusters_available():
            toast_info(
                tr("Cluster neurons to create timeline clusters before saving."),
                title=tr("SOM"),
                tab_key="som",
            )
            return

        cluster_ids = self._view_model.get_unique_cluster_ids()
        if not cluster_ids:
            toast_info(
                tr("No neuron clusters were found to save."),
                title=tr("SOM"),
                tab_key="som",
            )
            return

        defaults: dict[int, str] = {}
        for cluster_id in cluster_ids:
            custom_name = self._view_model.get_cluster_name(cluster_id).strip()
            defaults[int(cluster_id)] = custom_name or f"SOM_Cluster {cluster_id}"

        dialog = TimelineClusterGroupsDialog(defaults, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        group_label = dialog.group_label()
        if not group_label:
            toast_warn(
                tr("Group label cannot be empty."),
                title=tr("Save timeline clusters"),
                tab_key="som",
            )
            return

        names = dialog.group_names()
        if any(not value.strip() for value in names.values()):
            toast_warn(
                tr("Each cluster needs a non-empty group name."),
                title=tr("Save timeline clusters"),
                tab_key="som",
            )
            return
        normalized_names = {int(cluster_id): str(value).strip() for cluster_id, value in names.items()}
        save_as_timeframes = dialog.save_as_timeframes()

        try:
            self.sidebar.set_timeline_cluster_save_enabled(False)
        except Exception:
            logger.warning("Exception in _on_save_timeline_clusters_requested", exc_info=True)

        toast_info(
            tr("Saving timeline clusters to database..."),
            title=tr("SOM"),
            tab_key="som",
        )

        def _on_finished(summary: dict[str, int]) -> None:
            self._update_timeline_cluster_save_state()
            toast_success(
                tr("Saved {groups} cluster groups with {points} timeline ranges.").format(
                    groups=int(summary.get("group_labels", 0)),
                    points=int(summary.get("group_points", 0)),
                ),
                title=tr("SOM"),
                tab_key="som",
            )

        def _on_error(message: str) -> None:
            self._update_timeline_cluster_save_state()
            toast_error(
                tr("Failed to save timeline clusters.\n\n{error}").format(error=message or tr("Unknown error")),
                title=tr("Save failed"),
                tab_key="som",
            )

        self._view_model.save_timeline_clusters_as_groups(
            normalized_names,
            kind=group_label,
            save_as_timeframes=save_as_timeframes,
            on_finished=_on_finished,
            on_error=_on_error,
        )

    def _on_selected_features_from_viewmodel(self, *_args) -> None:
        if self._view_model.timeline_overlay_enabled():
            self._update_timeline_chart()

    def _update_timeline_cluster_map(self) -> None:
        view = getattr(self, "timeline_cluster_map", None)
        if not isinstance(view, SomMapView):
            return
        cluster_df = self._timeline_cluster_map_dataframe()
        if cluster_df is None or cluster_df.empty:
            view.clear_map()
            self._update_timeline_cluster_panel_state()
            return
        tooltip = self._cluster_map_tooltip(cluster_df)
        view.set_map_data(
            cluster_df,
            tooltip,
            annotations=cluster_df,
            value_formatter=lambda v: "" if pd.isna(v) else str(self._format_cluster_label(v)),
        )
        view.set_cluster_overlay(cluster_df, fill_only=True)
        # Enable centered labels with custom names instead of per-cell values
        view.set_cluster_names(self._view_model.get_all_cluster_names())
        view.set_show_cluster_centered_labels(True)
        view.set_highlight_cells(self._timeline_selected_cells())
        self._update_timeline_cluster_panel_state()

    def _update_timeline_cluster_panel_state(self) -> None:
        stack = getattr(self, "timeline_cluster_panel_stack", None)
        placeholder = getattr(self, "timeline_cluster_placeholder", None)
        content = getattr(self, "timeline_cluster_content", None)
        if stack is None or placeholder is None or content is None:
            return
        splitter = getattr(self, "timeline_rows_splitter", None)
        splitter_sizes = splitter.sizes() if splitter is not None else []
        show_cluster = self._neuron_clusters is not None
        stack.setCurrentWidget(content if show_cluster else placeholder)
        if splitter is not None and splitter_sizes:
            try:
                splitter.setSizes(splitter_sizes)
            except Exception:
                logger.warning("Exception in _update_timeline_cluster_panel_state", exc_info=True)

    def _timeline_selected_cells(self) -> list[tuple[int, int]]:
        df = getattr(self, "_timeline_display_df", pd.DataFrame())
        if df is None or df.empty:
            return []
        table = getattr(self, "timeline_table", None)
        if table is None:
            return []
        selection = table.selectionModel()
        if selection is None:
            return []
        rows = {index.row() for index in selection.selectedRows()}
        coords: list[tuple[int, int]] = []
        for row in sorted(rows):
            if row < 0 or row >= len(df):
                continue
            record = df.iloc[row]
            bmu_x = pd.to_numeric(pd.Series([record.get("bmu_x")]), errors="coerce").iat[0]
            bmu_y = pd.to_numeric(pd.Series([record.get("bmu_y")]), errors="coerce").iat[0]
            if pd.isna(bmu_x) or pd.isna(bmu_y):
                continue
            coords.append((int(bmu_x), int(bmu_y)))
        return coords

    def _update_timeline_row_highlight(self, cells: Optional[Iterable[tuple[int, int]]]) -> None:
        view = getattr(self, "timeline_cluster_map", None)
        if isinstance(view, SomMapView):
            view.set_highlight_cells(cells or [])

    def _on_timeline_rows_selected(self) -> None:
        self._update_timeline_row_highlight(self._timeline_selected_cells())

    def _on_timeline_cluster_cell_clicked(self, row: int, col: int) -> None:
        df = getattr(self, "_timeline_display_df", pd.DataFrame())
        if df is None or df.empty:
            return
        table = getattr(self, "timeline_table", None)
        if table is None:
            return
        selection = table.selectionModel()
        if selection is None:
            return
        model = table.model()
        if model is None:
            return

        bmu_x = pd.to_numeric(df.get("bmu_x"), errors="coerce")
        bmu_y = pd.to_numeric(df.get("bmu_y"), errors="coerce")
        mask = (bmu_x == row) & (bmu_y == col)
        if not mask.any():
            selection.clearSelection()
            return

        selection.clearSelection()
        rows = mask.to_numpy()
        first = True
        for row_idx, matched in enumerate(rows):
            if not matched:
                continue
            idx = model.index(row_idx, 0)
            if not idx.isValid():
                continue
            if first:
                selection.select(
                    idx,
                    QItemSelectionModel.SelectionFlag.ClearAndSelect
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
                table.setCurrentIndex(idx)
                first = False
            else:
                selection.select(
                    idx,
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )

    def _on_timeline_details_requested(self, row: int, col: int) -> None:
        if self._result is None:
            return
        cluster_df = self._timeline_cluster_map_dataframe()
        summary_text = build_som_map_summary_text(
            selection_key="clusters",
            selection_label=tr("Neuron clusters"),
            result=self._result,
            row=row,
            col=col,
            cluster_map=cluster_df,
            cluster_names=self._view_model.get_all_cluster_names(),
        )
        self._show_som_details_dialog(summary_text)

    def _show_som_details_dialog(self, summary_text: str) -> None:
        if self._som_details_dialog is None:
            self._som_details_dialog = SomDetailsDialog(
                summary_text=summary_text,
                on_ask_ai=self._ask_som_details_from_ai,
                parent=self,
            )
            try:
                self._som_details_dialog.finished.connect(
                    lambda _res: setattr(self, "_som_details_dialog", None)
                )
            except Exception:
                logger.warning("Exception in _show_som_details_dialog", exc_info=True)
        else:
            try:
                self._som_details_dialog.set_summary_text(summary_text)
            except Exception:
                logger.warning("Exception in _show_som_details_dialog", exc_info=True)
        try:
            self._som_details_dialog.show()
            self._som_details_dialog.raise_()
            self._som_details_dialog.activateWindow()
        except Exception:
            logger.warning("Exception in _show_som_details_dialog", exc_info=True)

    def _ask_som_details_from_ai(self, summary_text: str) -> None:
        prompt = build_som_map_prompt(summary_text.strip())
        if not prompt:
            return
        log_view_model = self._resolve_log_view_model()
        if log_view_model is None:
            return
        self._show_log_window()
        try:
            log_view_model.ask_llm(prompt)
        except Exception:
            logger.warning("Exception in _ask_som_details_from_ai", exc_info=True)

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

    def _fill_table(self, table, df: pd.DataFrame) -> None:
        model = table.model() if table is not None else None
        if not isinstance(model, DataFrameTableModel):
            return
        model.set_dataframe(df)

    @staticmethod
    def _empty_timeline_table_dataframe() -> pd.DataFrame:
        return pd.DataFrame(columns=["index", "bmu_x", "bmu_y", "bmu", "cluster"])
