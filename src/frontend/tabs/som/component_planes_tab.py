
from __future__ import annotations
from typing import Callable, Optional, Union, Literal, TYPE_CHECKING

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (


    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

import logging
logger = logging.getLogger(__name__)
from ...localization import tr

from ...widgets.panel import Panel
from ...utils.som_details import build_som_map_summary_text, build_som_map_prompt
from .map_view import SomMapView
from .som_details_dialog import SomDetailsDialog

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from .viewmodel import SomViewModel



SpecialMap = Literal["umatrix", "hits", "quantization", "clusters"]


class ComponentPlanesTab(QWidget):
    """
    Encapsulates all UI and logic for component planes and special maps (U-Matrix / Hits / QE).
    - Shows a 3x3 grid (nine panels).
    - Each panel has a combo where the user can pick either a feature or one of the special maps.
    - Bottom row defaults to: U-Matrix, Hit map, Quantisation error (left -> right).
    """

    SPECIAL_LABELS: dict[SpecialMap, str] = {
        "umatrix": "Distance (U-Matrix)",
        "hits": "Hit map",
        "quantization": "Quantisation error",
        "clusters": "Neuron clusters",
    }

    def __init__(
        self,
        view_model: Optional["SomViewModel"] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._view_model = view_model  # Store for cluster name lookups
        self._result = None  # set via set_result(...)
        self._plane_combos: list[QComboBox] = []
        self._plane_views: list[SomMapView] = []
        self._details_dialog: Optional[SomDetailsDialog] = None

        # Tracks what is selected in each of the 9 panels.
        # Each entry is either a column name (feature) or a special key ("umatrix", "hits", "quantization").
        self._plane_selected: list[Optional[Union[str, SpecialMap]]] = [None] * 9
        self._cluster_map: Optional[pd.DataFrame] = None

        # ---- Layout root ---------------------------------------------------
        root = QVBoxLayout(self)

        overlay_row = QHBoxLayout()
        overlay_row.addStretch(1)
        # Single checkbox for cluster boundaries - always shows as borders (no color overlay)
        self.cluster_overlay_checkbox = QCheckBox(tr("Show clusters"))
        self.cluster_overlay_checkbox.setEnabled(False)
        self.cluster_overlay_checkbox.setToolTip(
            tr(
                "When enabled, cluster boundaries are shown as colored edges\n"
                "on all maps, making clusters clearly visible"
            )
        )
        self.cluster_overlay_checkbox.toggled.connect(
            lambda _checked: self._update_cluster_overlays()
        )
        overlay_row.addWidget(self.cluster_overlay_checkbox)
        root.addLayout(overlay_row)

        # Splitter with 9 panels arranged as 3 horizontal splitters inside one vertical splitter.
        self.component_splitter = QSplitter(Qt.Orientation.Vertical, self)

        row_splitters: list[QSplitter] = []
        for _ in range(3):
            row_split = QSplitter(Qt.Orientation.Horizontal, self.component_splitter)
            row_splitters.append(row_split)

        # Create 9 panels
        for idx in range(9):
            panel, combo, view = self._create_component_plane(idx)
            target_split = row_splitters[idx // 3]
            target_split.addWidget(panel)
            self._plane_combos.append(combo)
            self._plane_views.append(view)

        # Add the three rows to the main vertical splitter
        for rs in row_splitters:
            self.component_splitter.addWidget(rs)

        root.addWidget(self.component_splitter, 1)

        if view_model is not None:
            view_model.training_started.connect(self.clear)
            view_model.training_finished.connect(self.set_result)
            view_model.cluster_names_changed.connect(self._on_cluster_names_changed)

        # Connect cluster rename signals from all plane views
        for idx, view in enumerate(self._plane_views):
            view.cluster_rename_requested.connect(self._on_cluster_rename_requested)
            view.details_requested.connect(
                lambda row, col, index=idx: self._on_plane_details_requested(index, row, col)
            )

    def _on_cluster_names_changed(self) -> None:
        """Called when cluster names are updated in the view model."""
        # Update cluster names in all plane views and refresh cluster maps
        for idx, sel in enumerate(self._plane_selected):
            if sel == "clusters":
                self._update_single_plane(idx, sel)
            else:
                # Update cluster names for border mode views
                self._plane_views[idx].set_cluster_names(self._get_all_cluster_names())

    def _on_cluster_rename_requested(self, cluster_id: int, new_name: str) -> None:
        """Handle cluster rename request from a map view."""
        if self._view_model is None:
            return
        try:
            self._view_model.set_cluster_name(cluster_id, new_name)
        except (AttributeError, KeyError, TypeError):
            logger.warning("Exception in _on_cluster_rename_requested", exc_info=True)

    def _on_plane_details_requested(self, index: int, row: int, col: int) -> None:
        if self._result is None:
            return
        selection = self._plane_selected[index]
        if selection is None:
            return
        if selection in self.SPECIAL_LABELS:
            selection_label = tr(self.SPECIAL_LABELS.get(selection, str(selection)))
        else:
            selection_label = str(selection)
        summary_text = build_som_map_summary_text(
            selection_key=str(selection),
            selection_label=selection_label,
            result=self._result,
            row=row,
            col=col,
            cluster_map=self._cluster_map,
            cluster_names=self._get_all_cluster_names(),
        )
        self._show_details_dialog(summary_text)

    def _show_details_dialog(self, summary_text: str) -> None:
        if self._details_dialog is None:
            self._details_dialog = SomDetailsDialog(
                summary_text=summary_text,
                on_ask_ai=self._ask_map_from_ai,
                parent=self,
            )
            try:
                self._details_dialog.finished.connect(
                    lambda _res: setattr(self, "_details_dialog", None)
                )
            except Exception:
                logger.warning("Exception in _show_details_dialog", exc_info=True)
        else:
            try:
                self._details_dialog.set_summary_text(summary_text)
            except Exception:
                logger.warning("Exception in _show_details_dialog", exc_info=True)
        try:
            self._details_dialog.show()
            self._details_dialog.raise_()
            self._details_dialog.activateWindow()
        except Exception:
            logger.warning("Exception in _show_details_dialog", exc_info=True)

    def _ask_map_from_ai(self, summary_text: str) -> None:
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
            logger.warning("Exception in _ask_map_from_ai", exc_info=True)

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

    # ------------------------------------------------------------------ API
    def set_result(self, result) -> None:
        """
        Provide a new SOM result object. This updates combos and all nine maps.
        The 'result' object is expected to have:
          - normalized_dataframe (pd.DataFrame)
          - component_planes: dict[str, pd.DataFrame]
          - activation_response: pd.DataFrame
          - quantization_map: Optional[pd.DataFrame]
          - distance_map: pd.DataFrame
          - map_shape: tuple[int, int]
        """
        self._result = result
        self._populate_feature_combos()
        self._apply_default_selections_if_needed()
        self._update_all_planes()

    def clear(self) -> None:
        """Clear all views."""
        self._result = None
        for combo in self._plane_combos:
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)
        for view in self._plane_views:
            view.clear_map()
        self._plane_selected = [None] * 9
        self.set_neuron_clusters(None)

    def set_neuron_clusters(self, clusters) -> None:
        cluster_df = None
        if clusters is not None:
            candidate = getattr(clusters, "labels_grid", None)
            if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                cluster_df = candidate.copy()
        self._cluster_map = cluster_df
        
        # Enable/disable cluster overlay checkbox and default to checked when clusters available
        checkbox = getattr(self, "cluster_overlay_checkbox", None)
        if checkbox is not None:
            checkbox.setEnabled(cluster_df is not None)
            block = checkbox.blockSignals(True)
            if cluster_df is None:
                checkbox.setChecked(False)
            else:
                # Default to checked when clusters become available
                checkbox.setChecked(True)
            checkbox.blockSignals(block)
        
        self._update_cluster_overlays()
        for idx, sel in enumerate(self._plane_selected):
            if sel == "clusters":
                self._update_single_plane(idx, sel)

    # ------------------------------------------------------------ UI factory
    def _create_component_plane(self, index: int):
        panel = Panel(tr("Plane {index}").format(index=index + 1))
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = panel.content_layout()

        # Top row: label + combo
        top_line = QHBoxLayout()
        top_line.addWidget(QLabel(tr("Map")))
        combo = QComboBox()
        combo.setPlaceholderText(tr("Select feature or map"))
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        combo.setMinimumContentsLength(12)
        top_line.addWidget(combo, 1)
        layout.addLayout(top_line)

        # Map view
        view = SomMapView()
        view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(view, 1)

        combo.currentIndexChanged.connect(
            lambda _i, idx=index: self._on_plane_combo_changed(idx)
        )
        return panel, combo, view

    # -------------------------------------------------------------- Updates
    def _populate_feature_combos(self) -> None:
        """
        Fill each combo with: [special maps] + [all features]
        Preserve current selections where possible.
        """
        feature_names: list[str] = []
        if self._result is not None:
            feature_names = list(self._result.normalized_dataframe.columns)

        # We rebuild items for each combo
        for i, combo in enumerate(self._plane_combos):
            # Remember current selection (by data) if any
            current_data = combo.currentData() if combo.count() > 0 else None

            combo.blockSignals(True)
            combo.clear()

            # Special maps section
            for key, label in self.SPECIAL_LABELS.items():
                combo.addItem(tr(label), key)

            # Separator-like visual hint (not a true separator in QComboBox)
            # We just add a disabled "— Features —" label
            if feature_names:
                combo.insertSeparator(combo.count())
                combo.addItem(tr("— Features —"))
                # Disable this pseudo-header
                header_index = combo.count() - 1
                combo.model().item(header_index).setEnabled(False)

            # Features
            for name in feature_names:
                label = name
                if self._view_model is not None:
                    try:
                        label = self._view_model.feature_display_name(name)
                    except Exception:
                        label = name
                combo.addItem(label, name)

            # Try to re-select previous
            if current_data is not None:
                index_to_select = self._find_data_index(combo, current_data)
                if index_to_select >= 0:
                    combo.setCurrentIndex(index_to_select)

            combo.blockSignals(False)

        # If no features exist, reset internal selections
        if not feature_names and self._result is None:
            self._plane_selected = [None] * 9

    def _apply_default_selections_if_needed(self) -> None:
        """
        If a plane has nothing selected, assign a default:
         - Top & middle rows: unique feature names cycling through available features.
         - Bottom row (indices 6,7,8): U-Matrix, Hits, Quantisation error (left -> right).
        """
        # First gather features
        features: list[str] = []
        if self._result is not None:
            features = list(self._result.normalized_dataframe.columns)

        # Bottom row defaults
        bottom_defaults: list[SpecialMap] = ["umatrix", "hits", "quantization"]

        used: set[str] = set()
        for idx in range(9):
            if self._plane_selected[idx] is not None:
                # If the selection is no longer available, we clear it so we reassign.
                if isinstance(self._plane_selected[idx], str):
                    # Could be feature or special key. Special keys are in SPECIAL_LABELS.
                    if self._plane_selected[idx] in self.SPECIAL_LABELS:
                        continue
                    # Feature: ensure still exists
                    if self._plane_selected[idx] not in features:
                        self._plane_selected[idx] = None
                else:
                    continue

            if idx >= 6:
                # Bottom row hard defaults
                self._plane_selected[idx] = bottom_defaults[idx - 6]
                self._select_combo_data(idx, bottom_defaults[idx - 6])
            else:
                # Top two rows: distribute features (fall back to specials if no features)
                if features:
                    # Avoid duplicates when possible
                    candidate = features[idx % len(features)]
                    if candidate in used and len(used) < len(features):
                        # find next unused
                        for f in features:
                            if f not in used:
                                candidate = f
                                break
                    used.add(candidate)
                    self._plane_selected[idx] = candidate
                    self._select_combo_data(idx, candidate)
                else:
                    # If no features, just default to U-Matrix on top-left, etc.
                    fallback = "umatrix" if idx == 0 else "hits" if idx == 1 else "quantization"
                    self._plane_selected[idx] = fallback  # type: ignore[assignment]
                    self._select_combo_data(idx, fallback)

    def _on_plane_combo_changed(self, index: int) -> None:
        combo = self._plane_combos[index]
        sel = combo.currentData()
        # sel can be a feature name (str) or a special map key ("umatrix"/"hits"/"quantization")
        if sel is None:
            self._plane_selected[index] = None
            self._plane_views[index].clear_map()
            return
        self._plane_selected[index] = sel
        self._update_single_plane(index, sel)
        self._update_cluster_overlays()

    def _update_all_planes(self) -> None:
        for idx, sel in enumerate(self._plane_selected):
            if sel:
                self._update_single_plane(idx, sel)
            else:
                self._plane_views[idx].clear_map()
        self._update_cluster_overlays()

    def _update_single_plane(self, index: int, selection: Union[str, SpecialMap]) -> None:
        """
        Update one cell according to its selection.
        - If selection is a feature name: show its component plane.
        - If selection is a special key: show that special map.
        """
        if not self._result:
            self._plane_views[index].clear_map()
            return

        view = self._plane_views[index]

        if selection == "clusters":
            cluster_df = self._cluster_map
            if cluster_df is None or cluster_df.empty:
                view.show_placeholder(
                    tr("No neuron clusters yet.\nUse 'Cluster neurons' to create a cluster map.")
                )
                return
            tooltip = self._make_standard_tooltip()
            view.set_map_data(
                cluster_df,
                tooltip,
                annotations=cluster_df,
                value_formatter=lambda v: str(int(v)) if pd.notna(v) else "",
            )
            view.set_cluster_overlay(cluster_df, fill_only=True)
            # Enable centered labels with custom names (like timeline tab)
            view.set_cluster_names(self._get_all_cluster_names())
            view.set_show_cluster_centered_labels(True)
            return

        if isinstance(selection, str) and selection not in self.SPECIAL_LABELS:
            # Feature plane
            plane: Optional[pd.DataFrame] = self._result.component_planes.get(selection)
            if plane is None or plane.empty:
                view.clear_map()
                return
            view.set_map_data(
                plane,
                self._make_standard_tooltip(primary_df=plane, primary_label="Value"),
                symmetric=True,  # component planes typically look better symmetric
            )
            return

        # Special maps
        hits_df = self._result.activation_response
        qe_df = getattr(self._result, "quantization_map", None)

        if selection == "hits":
            matrix = hits_df
            if matrix is not None and not matrix.empty:
                tooltip = self._make_standard_tooltip()
                view.set_map_data(matrix, tooltip)
                return

        if selection == "quantization":
            matrix = qe_df
            if matrix is not None and not matrix.empty:
                tooltip = self._make_standard_tooltip()
                view.set_map_data(matrix, tooltip)
                return

        # Default / "umatrix"
        matrix = self._result.distance_map
        if matrix is not None and not matrix.empty:
            view.set_map_data(matrix, self._make_standard_tooltip())
            return

        view.clear_map()

    def _cluster_overlay_enabled(self) -> bool:
        checkbox = getattr(self, "cluster_overlay_checkbox", None)
        if checkbox is None:
            return False
        try:
            return bool(checkbox.isChecked()) and self._cluster_map is not None
        except Exception:
            return False

    def _update_cluster_overlays(self) -> None:
        """Update cluster overlay data and rendering mode for all plane views.
        
        When clusters are shown, we always use border mode (no color overlay)
        to preserve the underlying SOM colors.
        """
        show_clusters = self._cluster_overlay_enabled()
        overlay = self._cluster_map if show_clusters else None
        
        for idx, view in enumerate(self._plane_views):
            if self._plane_selected[idx] == "clusters":
                continue
            view.set_cluster_overlay(overlay, fill_only=False)
            # Always use border mode for cluster visualization (no color overlay)
            view.set_cluster_border_mode(show_clusters)
            view.set_cluster_names(self._get_all_cluster_names() if show_clusters else {})
            view.set_show_cluster_centered_labels(show_clusters)

    def _get_cluster_name(self, cluster_id: int) -> str:
        """Get custom name for a cluster from the view model."""
        if self._view_model is None:
            return ""
        try:
            return self._view_model.get_cluster_name(cluster_id)
        except (AttributeError, KeyError, TypeError):
            return ""

    def _get_all_cluster_names(self) -> dict[int, str]:
        """Get all custom cluster names from the view model."""
        if self._view_model is None:
            return {}
        try:
            return self._view_model.get_all_cluster_names()
        except (AttributeError, KeyError, TypeError):
            return {}

    def _format_cluster_label(self, label) -> str:
        """Format cluster label with custom name if available."""
        try:
            label_int = int(label)
            custom_name = self._get_cluster_name(label_int)
            if custom_name:
                return f"{label_int} ({custom_name})"
            return str(label_int)
        except (ValueError, TypeError):
            return str(label)

    # ---------------------------------------------------------- Tooltips ---
    def _make_standard_tooltip(
        self,
        *,
        primary_df: Optional[pd.DataFrame] = None,
        primary_label: str = "Value",
    ) -> Callable[[int, int], str]:
        distance_df = self._result.distance_map if self._result else None
        hits_df = self._result.activation_response if self._result else None
        qe_df = getattr(self._result, "quantization_map", None) if self._result else None

        def _value(df: Optional[pd.DataFrame], row: int, col: int) -> Optional[float]:
            if df is None or df.empty or row >= df.shape[0] or col >= df.shape[1]:
                return None
            value = df.iat[row, col]
            if pd.isna(value):
                return None
            try:
                return float(value)
            except Exception:
                return None

        def formatter(row: int, col: int) -> str:
            parts: list[str] = []
            if primary_df is not None:
                primary = _value(primary_df, row, col)
                if primary is None:
                    parts.append(f"{primary_label}: unavailable")
                else:
                    parts.append(f"{primary_label}: {primary:.4f}")
            parts.append(f"Neuron ({col}, {row})")
            distance = _value(distance_df, row, col)
            if distance is not None:
                parts.append(f"Distance: {distance:.4f}")
            hits = _value(hits_df, row, col)
            if hits is not None:
                parts.append(f"Hits: {int(hits)}")
            qe = _value(qe_df, row, col)
            if qe is not None:
                parts.append(f"QE: {qe:.4f}")
            cluster_df = self._cluster_map
            if cluster_df is not None and not cluster_df.empty and row < cluster_df.shape[0] and col < cluster_df.shape[1]:
                label = cluster_df.iat[row, col]
                if pd.notna(label):
                    parts.append(f"Cluster: {self._format_cluster_label(label)}")
            return "\n".join(parts)

        return formatter

    # ---------------------------------------------------------- Utilities ---
    @staticmethod
    def _find_data_index(combo: QComboBox, data: object) -> int:
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                return i
        return -1

    def _select_combo_data(self, combo_index: int, data: object) -> None:
        combo = self._plane_combos[combo_index]
        idx = self._find_data_index(combo, data)
        if idx >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

