
from __future__ import annotations
import math
import threading
from functools import partial
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal

from backend.services import (

    SOMService,
    SomResult,
    NeuronClusteringResult,
    FeatureClusteringResult,
    ClusteringInputs,
    ClusteringMethodSpec,
)
from backend.services.modeling_shared import display_name
from core.training_settings import (
    TRAINING_SPARSE_FEATURE_NAN_RATIO_THRESHOLD,
    TRAINING_STATIC_FEATURE_MAX_UNIQUE_NON_NULL,
)

import logging
logger = logging.getLogger(__name__)
from ...models.hybrid_pandas_model import HybridPandasModel, FeatureSelection

from ...threading.runner import run_in_thread
from ...threading.utils import run_in_main_thread
from ...utils import set_status_text, set_progress, clear_progress, clear_status_text
from ...utils.model_persistence import frame_to_records, normalize_for_json, records_to_frame


class SomViewModel(QObject):
    """Qt-friendly wrapper around the SOM backend service."""

    training_started = Signal()
    training_finished = Signal(object)
    error_occurred = Signal(str)
    status_changed = Signal(str)
    database_changed = Signal(object)
    
    # Clustering-related signals
    neuron_clusters_updated = Signal(object)  # emitted with NeuronClusteringResult or None
    feature_clusters_updated = Signal(object)  # emitted with FeatureClusteringResult or None
    clustering_error = Signal(str)  # emitted on clustering error
    cluster_names_changed = Signal()  # emitted when cluster names are updated
    selected_feature_payloads_changed = Signal(object)
    timeline_display_options_changed = Signal(object)
    timeline_table_dataframe_changed = Signal(object)
    sparse_features_dropped = Signal(object)
    static_features_dropped = Signal(object)
    _TIMELINE_DISPLAY_ALLOWED = {"bmu", "cluster", "selected_features"}
    _TIMELINE_DISPLAY_DEFAULT = ("bmu",)

    def __init__(self, model: HybridPandasModel, parent=None):
        """
        SomViewModel works purely against a HybridPandasModel instance.
        The model is responsible for managing DB connections via SettingsModel.
        """
        super().__init__(parent)
        self._data_model: Optional[HybridPandasModel] = model
        self._service = SOMService()
        self._result: Optional[SomResult] = None
        self._last_dataframe: Optional[pd.DataFrame] = None
        self._last_neuron_clusters: Optional[NeuronClusteringResult] = None
        self._last_feature_clusters: Optional[FeatureClusteringResult] = None
        self._current_status_text: str = ""
        self._feature_display_map: dict[str, str] = {}
        self._cluster_names: dict[int, str] = {}  # Maps cluster ID to custom name
        self._last_training_context: dict[str, object] = {}
        self._selected_feature_payloads: list[dict] = []
        self._timeline_display_selected: list[str] = list(self._TIMELINE_DISPLAY_DEFAULT)
        self._timeline_table_df: pd.DataFrame = pd.DataFrame(
            columns=["index", "cluster", "bmu_x", "bmu_y", "bmu", "distance"]
        )

        self._training_running = False
        self._neuron_cluster_running = False
        self._feature_cluster_running = False

        # React to DB path changes via the hybrid model, and forward a notification
        try:
            self._data_model.database_changed.connect(self._on_database_changed)
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)

        self.status_changed.connect(self._apply_status_text)

    def _close_database(self) -> None:
        """Clear model-related state; does NOT touch global SettingsModel."""
        self._result = None
        self._last_dataframe = None
        self._last_neuron_clusters = None
        self._last_feature_clusters = None
        self._cluster_names = {}
        self._feature_display_map = {}
        self._last_training_context = {}
        self._selected_feature_payloads = []
        self._timeline_display_selected = list(self._TIMELINE_DISPLAY_DEFAULT)
        self._timeline_table_df = pd.DataFrame(
            columns=["index", "cluster", "bmu_x", "bmu_y", "bmu", "distance"]
        )

    def close_database(self) -> None:
        """Release the data model reference and related state."""
        self._close_database()
        self.database_changed.emit(None)

    def _on_database_changed(self, _path) -> None:
        """
        Called when the underlying HybridPandasModel reports a DB path change.
        We clear any cached SOM results and forward a database_changed signal.
        """
        self._result = None
        self._last_dataframe = None
        self._last_neuron_clusters = None
        self._last_feature_clusters = None
        self._cluster_names = {}
        self._feature_display_map = {}
        self._last_training_context = {}
        self._selected_feature_payloads = []
        self._timeline_display_selected = list(self._TIMELINE_DISPLAY_DEFAULT)
        self._timeline_table_df = pd.DataFrame(
            columns=["index", "cluster", "bmu_x", "bmu_y", "bmu", "distance"]
        )
        self.selected_feature_payloads_changed.emit([])
        self.timeline_display_options_changed.emit(self._timeline_display_payload())
        self.timeline_table_dataframe_changed.emit(self._timeline_table_df.copy())

    # ------------------------------------------------------------------
    def _ensure_data_model(self) -> HybridPandasModel:
        if self._data_model is None:
            raise RuntimeError("Data model is not initialised")
        return self._data_model

    def set_selected_feature_payloads(self, payloads: Optional[Sequence[dict]]) -> None:
        normalized: list[dict] = []
        for payload in payloads or []:
            if isinstance(payload, dict):
                normalized.append(dict(payload))
        if normalized == self._selected_feature_payloads:
            return
        self._selected_feature_payloads = normalized
        self.selected_feature_payloads_changed.emit(list(self._selected_feature_payloads))

    def selected_feature_payloads(self) -> list[dict]:
        return [dict(payload) for payload in self._selected_feature_payloads]

    def set_timeline_display_options(
        self,
        *,
        selected: Optional[Sequence[str]] = None,
        mode: Optional[str] = None,
        overlay_enabled: Optional[bool] = None,
    ) -> None:
        changed = False
        if selected is not None:
            normalized_selected = self._normalize_timeline_display_selected(selected)
            if normalized_selected != self._timeline_display_selected:
                self._timeline_display_selected = normalized_selected
                changed = True
        else:
            # Backward-compatible path for old callers.
            normalized_selected = list(self._timeline_display_selected)
            if mode is not None:
                normalized_mode = str(mode or "bmu").strip().lower()
                if normalized_mode not in {"bmu", "cluster"}:
                    normalized_mode = "bmu"
                normalized_selected = [
                    key for key in normalized_selected if key not in {"bmu", "cluster"}
                ]
                normalized_selected.insert(0, normalized_mode)
            if overlay_enabled is not None:
                if bool(overlay_enabled):
                    if "selected_features" not in normalized_selected:
                        normalized_selected.append("selected_features")
                else:
                    normalized_selected = [
                        key for key in normalized_selected if key != "selected_features"
                    ]
            normalized_selected = self._normalize_timeline_display_selected(normalized_selected)
            if normalized_selected != self._timeline_display_selected:
                self._timeline_display_selected = normalized_selected
                changed = True
        if changed:
            self.timeline_display_options_changed.emit(self._timeline_display_payload())

    def timeline_mode(self) -> str:
        if "cluster" in self._timeline_display_selected:
            return "cluster"
        if "bmu" in self._timeline_display_selected:
            return "bmu"
        return "none"

    def timeline_overlay_enabled(self) -> bool:
        return "selected_features" in self._timeline_display_selected

    def timeline_display_selected(self) -> list[str]:
        return list(self._timeline_display_selected)

    def timeline_show_clusters(self) -> bool:
        return "cluster" in self._timeline_display_selected

    def timeline_show_bmu(self) -> bool:
        return "bmu" in self._timeline_display_selected

    def _normalize_timeline_display_selected(self, values: Sequence[str]) -> list[str]:
        out: list[str] = []
        for value in values or []:
            key = str(value or "").strip().lower()
            if key in self._TIMELINE_DISPLAY_ALLOWED and key not in out:
                out.append(key)
        return out

    def _timeline_display_payload(self) -> dict[str, object]:
        return {
            "selected": list(self._timeline_display_selected),
            "mode": self.timeline_mode(),
            "overlay_enabled": self.timeline_overlay_enabled(),
            "show_bmu": self.timeline_show_bmu(),
            "show_cluster": self.timeline_show_clusters(),
        }

    def set_timeline_table_dataframe(self, df: Optional[pd.DataFrame]) -> None:
        base_columns = ["index", "cluster", "bmu_x", "bmu_y", "bmu", "distance"]
        if df is None or df.empty:
            normalized = pd.DataFrame(columns=base_columns)
        else:
            normalized = df.copy()
            for column in base_columns:
                if column not in normalized.columns:
                    normalized[column] = pd.NA
            ordered = list(base_columns) + [c for c in normalized.columns if c not in set(base_columns)]
            normalized = normalized.loc[:, ordered]
        self._timeline_table_df = normalized
        self.timeline_table_dataframe_changed.emit(self._timeline_table_df.copy())

    def timeline_table_dataframe(self) -> pd.DataFrame:
        return self._timeline_table_df.copy()

    # ------------------------------------------------------------------
    def result(self) -> Optional[SomResult]:
        return self._result

    def last_dataframe(self) -> Optional[pd.DataFrame]:
        return None if self._last_dataframe is None else self._last_dataframe.copy()

    def clustering_inputs(self) -> ClusteringInputs:
        if self._result is None:
            raise RuntimeError("Train a SOM model before clustering")
        return self._service.clustering_inputs()

    def available_clustering_methods(self) -> list[ClusteringMethodSpec]:
        return list(self._service.available_clustering_methods())

    # ------------------------------------------------------------------
    def save_map(self, name: str) -> Optional[int]:
        if self._result is None:
            raise RuntimeError("Train a SOM model before saving")
        model = self._ensure_data_model()
        db = model.db

        display_name = (name or "").strip()
        if not display_name:
            raise ValueError("Map name cannot be empty")

        training_context = dict(self._last_training_context)
        feature_payloads = training_context.get("feature_payloads") or []
        features: list[tuple[int, str]] = []
        for payload in feature_payloads:
            fid = self._safe_feature_id(payload)
            if fid is not None:
                features.append((fid, "input"))

        weights = None
        if self._service.som is not None:
            try:
                weights = np.asarray(self._service.som.get_weights(), dtype=float)
            except Exception:
                weights = None
        if weights is None and self._result.som_object is not None:
            try:
                weights = np.asarray(self._result.som_object.get_weights(), dtype=float)
            except Exception:
                weights = None
        if weights is None:
            raise RuntimeError("Unable to resolve SOM weights for saving")

        component_planes = {
            name: frame_to_records(frame)
            for name, frame in (self._result.component_planes or {}).items()
        }
        neuron_clusters_payload = self._serialize_neuron_clusters(self._last_neuron_clusters)
        feature_clusters_payload = self._serialize_feature_clusters(self._last_feature_clusters)

        artifacts = {
            "weights": normalize_for_json(weights.tolist()),
            "component_planes": component_planes,
            "feature_positions": frame_to_records(self._result.feature_positions),
            "row_bmus": frame_to_records(self._result.row_bmus),
            "bmu_counts": frame_to_records(self._result.bmu_counts),
            "distance_map": frame_to_records(self._result.distance_map),
            "activation_response": frame_to_records(self._result.activation_response),
            "quantization_map": frame_to_records(self._result.quantization_map),
            "correlations": frame_to_records(self._result.correlations),
            "normalized_dataframe": frame_to_records(self._result.normalized_dataframe),
            "scaler": normalize_for_json(self._result.scaler),
            "quantization_error": normalize_for_json(self._result.quantization_error),
            "topographic_error": normalize_for_json(self._result.topographic_error),
            "normalized_columns": list(self._result.normalized_dataframe.columns),
            "neuron_clusters": neuron_clusters_payload,
            "feature_clusters": feature_clusters_payload,
        }

        parameters = {
            "training": normalize_for_json(training_context.get("training") or {}),
            "filters": normalize_for_json(training_context.get("filters") or {}),
            "preprocessing": normalize_for_json(training_context.get("preprocessing") or {}),
            "feature_payloads": normalize_for_json(feature_payloads),
            "feature_labels": normalize_for_json(training_context.get("feature_labels") or []),
            "feature_display_map": normalize_for_json(self._feature_display_map),
            "cluster_names": normalize_for_json(self._cluster_names),
        }

        model_id = db.save_model_run(
            name=display_name,
            model_type="som",
            algorithm_key="som",
            selector_key=None,
            preprocessing=normalize_for_json(training_context.get("preprocessing") or {}),
            filters=normalize_for_json(training_context.get("filters") or {}),
            hyperparameters=normalize_for_json(training_context.get("training") or {}),
            parameters=parameters,
            artifacts=artifacts,
            features=features,
            results=None,
        )
        return model_id

    def list_saved_maps(self) -> list[dict]:
        try:
            db = self._ensure_data_model().db
            models = db.list_models(model_type="som")
        except Exception:
            return []
        if models is None or models.empty:
            return []
        maps: list[dict] = []
        for _, row in models.iterrows():
            model_id = int(row.get("model_id"))
            entry = {
                "model_id": model_id,
                "name": row.get("name"),
                "created_at": row.get("created_at"),
            }
            try:
                details = db.fetch_model(model_id)
            except Exception:
                details = None
            if details:
                try:
                    entry.update(self._summarize_saved_map(details))
                except Exception:
                    logger.warning("Exception in list_saved_maps", exc_info=True)
            maps.append(entry)
        return maps

    def delete_saved_map(self, model_id: int) -> None:
        if model_id is None:
            return
        db = self._ensure_data_model().db
        db.delete_model(int(model_id))

    def load_saved_map(self, model_id: int) -> SomResult:
        db = self._ensure_data_model().db
        details = db.fetch_model(int(model_id))
        if not details:
            raise RuntimeError("Saved SOM not found")

        parameters = details.get("parameters") or {}
        artifacts = details.get("artifacts") or {}
        training = parameters.get("training") or {}

        weights = np.asarray(artifacts.get("weights") or [], dtype=float)
        if weights.ndim != 3:
            raise RuntimeError("Saved SOM weights are missing or invalid")

        map_shape = (
            int(weights.shape[1]),
            int(weights.shape[0]),
        )

        normalized_records = artifacts.get("normalized_dataframe") or []
        normalized_df = records_to_frame(normalized_records)
        normalized_columns = artifacts.get("normalized_columns") or []
        if normalized_columns and not normalized_df.empty:
            normalized_df = normalized_df.reindex(columns=list(normalized_columns))

        som_obj = self._service.load_from_state(
            weights=weights,
            sigma=float(training.get("sigma", 1.0)),
            learning_rate=float(training.get("learning_rate", 0.5)),
            neighborhood_function=str(training.get("neighborhood_function", "gaussian")),
            normalized_dataframe=normalized_df if not normalized_df.empty else None,
        )

        component_planes_payload = artifacts.get("component_planes") or {}
        component_planes = {
            name: self._map_frame_from_records(records, map_shape)
            for name, records in component_planes_payload.items()
        }

        result = SomResult(
            map_shape=map_shape,
            component_planes=component_planes,
            feature_positions=records_to_frame(artifacts.get("feature_positions")),
            row_bmus=records_to_frame(artifacts.get("row_bmus")),
            bmu_counts=records_to_frame(artifacts.get("bmu_counts")),
            distance_map=self._map_frame_from_records(artifacts.get("distance_map"), map_shape),
            activation_response=self._map_frame_from_records(artifacts.get("activation_response"), map_shape),
            quantization_map=self._map_frame_from_records(artifacts.get("quantization_map"), map_shape),
        correlations=self._restore_square_dataframe(
            records_to_frame(artifacts.get("correlations"))
        ),
            quantization_error=float(artifacts.get("quantization_error", float("nan"))),
            topographic_error=float(artifacts.get("topographic_error", float("nan"))),
            normalized_dataframe=normalized_df,
            scaler=artifacts.get("scaler") or {},
            som_object=som_obj,
        )

        self._result = result
        self._last_dataframe = None
        self._last_neuron_clusters = self._deserialize_neuron_clusters(
            artifacts.get("neuron_clusters") or {},
            map_shape,
        )
        self._last_feature_clusters = self._deserialize_feature_clusters(
            artifacts.get("feature_clusters") or {},
        )
        self._cluster_names = self._normalize_cluster_name_map(parameters.get("cluster_names") or {})
        self._feature_display_map = parameters.get("feature_display_map") or {}
        self._last_training_context = {
            "training": training,
            "filters": parameters.get("filters") or {},
            "preprocessing": parameters.get("preprocessing") or {},
            "feature_payloads": parameters.get("feature_payloads") or [],
            "feature_labels": parameters.get("feature_labels") or [],
        }
        return result

    @staticmethod
    def _normalize_cluster_name_map(payload: object) -> dict[int, str]:
        if not isinstance(payload, dict):
            return {}
        names: dict[int, str] = {}
        for key, value in payload.items():
            try:
                cluster_id = int(key)
            except (TypeError, ValueError):
                continue
            if value is None:
                continue
            name = str(value).strip()
            if name:
                names[cluster_id] = name
        return names

    @staticmethod
    def _serialize_neuron_clusters(
        clusters: Optional[NeuronClusteringResult],
    ) -> Optional[dict]:
        if clusters is None:
            return None
        labels_grid = frame_to_records(getattr(clusters, "labels_grid", None))
        return {
            "k": int(getattr(clusters, "k", 0) or 0),
            "labels_1d": normalize_for_json(np.asarray(clusters.labels_1d).tolist()),
            "centers": normalize_for_json(np.asarray(clusters.centers).tolist()),
            "counts": normalize_for_json(np.asarray(clusters.counts).tolist()),
            "labels_grid": labels_grid,
            "bmu_cluster_labels": normalize_for_json(np.asarray(clusters.bmu_cluster_labels).tolist()),
        }

    @staticmethod
    def _serialize_feature_clusters(
        clusters: Optional[FeatureClusteringResult],
    ) -> Optional[dict]:
        if clusters is None:
            return None
        return {
            "k": int(getattr(clusters, "k", 0) or 0),
            "labels": normalize_for_json(np.asarray(clusters.labels).tolist()),
            "centers": normalize_for_json(np.asarray(clusters.centers).tolist()),
            "counts": normalize_for_json(np.asarray(clusters.counts).tolist()),
            "index": normalize_for_json(list(getattr(clusters, "index", []) or [])),
        }

    def _deserialize_neuron_clusters(
        self,
        payload: object,
        map_shape: tuple[int, int],
    ) -> Optional[NeuronClusteringResult]:
        if not isinstance(payload, dict) or not payload:
            return None
        labels_grid = self._map_frame_from_records(payload.get("labels_grid"), map_shape)
        labels_1d = np.asarray(payload.get("labels_1d") or [], dtype=int)
        centers = np.asarray(payload.get("centers") or [], dtype=float)
        counts = np.asarray(payload.get("counts") or [], dtype=int)
        bmu_cluster_labels = np.asarray(payload.get("bmu_cluster_labels") or [], dtype=int)
        k = payload.get("k")
        try:
            k_val = int(k)
        except (TypeError, ValueError):
            k_val = int(counts.size) if counts.size else 0
        return NeuronClusteringResult(
            k=k_val,
            labels_1d=labels_1d,
            centers=centers,
            counts=counts,
            labels_grid=labels_grid,
            bmu_cluster_labels=bmu_cluster_labels,
        )

    @staticmethod
    def _deserialize_feature_clusters(payload: object) -> Optional[FeatureClusteringResult]:
        if not isinstance(payload, dict) or not payload:
            return None
        labels = np.asarray(payload.get("labels") or [], dtype=int)
        centers = np.asarray(payload.get("centers") or [], dtype=float)
        counts = np.asarray(payload.get("counts") or [], dtype=int)
        index = list(payload.get("index") or [])
        k = payload.get("k")
        try:
            k_val = int(k)
        except (TypeError, ValueError):
            k_val = int(counts.size) if counts.size else 0
        return FeatureClusteringResult(
            k=k_val,
            labels=labels,
            centers=centers,
            counts=counts,
            index=index,
        )

    @staticmethod
    def _infer_map_shape(weights_payload: object) -> Optional[tuple[int, int]]:
        if not isinstance(weights_payload, list) or not weights_payload:
            return None
        try:
            width = len(weights_payload)
            height = len(weights_payload[0]) if width else 0
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None
        return (height, width)

    def _summarize_saved_map(self, details: dict) -> dict:
        parameters = details.get("parameters") or {}
        artifacts = details.get("artifacts") or {}

        map_shape = self._infer_map_shape(artifacts.get("weights"))
        map_shape_display = ""
        if map_shape is None:
            training = parameters.get("training") or {}
            raw_shape = training.get("map_shape") or []
            if isinstance(raw_shape, (list, tuple)) and len(raw_shape) == 2:
                try:
                    width = int(raw_shape[0])
                    height = int(raw_shape[1])
                    if width > 0 and height > 0:
                        map_shape_display = f"{height}x{width}"
                except Exception:
                    map_shape_display = ""
        else:
            map_shape_display = f"{map_shape[0]}x{map_shape[1]}"

        feature_payloads = parameters.get("feature_payloads") or []
        feature_labels = parameters.get("feature_labels") or []
        normalized_columns = artifacts.get("normalized_columns") or []
        features_count = 0
        for candidate in (feature_payloads, feature_labels, normalized_columns):
            if isinstance(candidate, (list, tuple)) and candidate:
                features_count = len(candidate)
                break
        if not features_count:
            features_count = len(details.get("features") or [])

        neuron_clusters = artifacts.get("neuron_clusters") or {}
        feature_clusters = artifacts.get("feature_clusters") or {}
        return {
            "map_shape": map_shape_display,
            "features": int(features_count) if features_count else "",
            "neuron_clusters": neuron_clusters.get("k", ""),
            "feature_clusters": feature_clusters.get("k", ""),
            "quantization_error": artifacts.get("quantization_error", ""),
            "topographic_error": artifacts.get("topographic_error", ""),
        }

    # ------------------------------------------------------------------
    # Cluster naming methods
    # ------------------------------------------------------------------
    def get_cluster_name(self, cluster_id: int) -> str:
        """Get the custom name for a cluster, or empty string if not set."""
        return self._cluster_names.get(cluster_id, "")

    def set_cluster_name(self, cluster_id: int, name: str) -> None:
        """Set a custom name for a cluster."""
        if name.strip():
            self._cluster_names[cluster_id] = name.strip()
        elif cluster_id in self._cluster_names:
            del self._cluster_names[cluster_id]
        self.cluster_names_changed.emit()

    def get_all_cluster_names(self) -> dict[int, str]:
        """Get a copy of all cluster names."""
        return dict(self._cluster_names)

    def clear_cluster_names(self) -> None:
        """Clear all cluster names."""
        self._cluster_names = {}
        self.cluster_names_changed.emit()

    def get_unique_cluster_ids(self) -> list[int]:
        """Get unique cluster IDs from the last neuron clustering result."""
        if self._last_neuron_clusters is None:
            return []
        labels_grid = getattr(self._last_neuron_clusters, "labels_grid", None)
        if labels_grid is None or labels_grid.empty:
            return []
        try:
            unique_labels = labels_grid.to_numpy().ravel()
            unique_ids = sorted(set(int(x) for x in unique_labels if pd.notna(x)))
            return unique_ids
        except (ValueError, TypeError, AttributeError):
            return []

    def _build_timeline_cluster_group_frames(
        self,
        cluster_labels: dict[int, str],
        *,
        save_as_timeframes: bool,
    ) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame]:
        if self._result is None or self._last_neuron_clusters is None:
            raise RuntimeError("Cluster neurons before saving timeline clusters")

        row_bmus = getattr(self._result, "row_bmus", None)
        bmu_cluster_labels = getattr(self._last_neuron_clusters, "bmu_cluster_labels", None)
        if not isinstance(row_bmus, pd.DataFrame) or row_bmus.empty:
            raise ValueError("Timeline data is empty")
        if bmu_cluster_labels is None or len(bmu_cluster_labels) != len(row_bmus):
            raise ValueError("Timeline cluster labels are not aligned with timeline rows")

        ts = pd.to_datetime(row_bmus.get("index"), errors="coerce")
        timeline = pd.DataFrame(
            {
                "ts": ts,
                "cluster_id": pd.to_numeric(
                    pd.Series(bmu_cluster_labels, index=row_bmus.index),
                    errors="coerce",
                ),
            }
        ).dropna(subset=["ts", "cluster_id"])
        if timeline.empty:
            raise ValueError("No valid timeline timestamps were found for clusters")
        timeline["cluster_id"] = timeline["cluster_id"].astype(int)
        timeline["ts"] = timeline["ts"].dt.tz_localize(None)

        normalized_names: dict[int, str] = {}
        for cluster_id, label in (cluster_labels or {}).items():
            try:
                cid = int(cluster_id)
            except (TypeError, ValueError):
                continue
            text = str(label or "").strip()
            if text:
                normalized_names[cid] = text
        if not normalized_names:
            raise ValueError("At least one cluster group name is required")

        timeline["label"] = timeline["cluster_id"].map(normalized_names)
        timeline = timeline.dropna(subset=["label"])
        if timeline.empty:
            raise ValueError("No timeline rows matched the selected cluster names")

        labels_df = pd.DataFrame(
            {
                "label": sorted(set(timeline["label"].astype(str))),
            }
        )
        timeline = timeline.sort_values("ts").reset_index(drop=True)
        timeline["end_ts"] = self._timeline_point_end_times(timeline["ts"])
        timeline = timeline.dropna(subset=["end_ts"])
        if timeline.empty:
            raise ValueError("No valid timeline ranges were found for clusters")

        if save_as_timeframes:
            run_ids = timeline["label"].ne(timeline["label"].shift(1)).cumsum()
            ranges_df = (
                timeline.assign(_run_id=run_ids)
                .groupby("_run_id", as_index=False, sort=True)
                .agg(
                    label=("label", "first"),
                    start_ts=("ts", "min"),
                    end_ts=("end_ts", "max"),
                )
            )
            points_df = ranges_df[["start_ts", "end_ts", "label"]].drop_duplicates().reset_index(drop=True)
        else:
            points_df = timeline.rename(columns={"ts": "start_ts"}).copy()
            points_df = points_df[["start_ts", "end_ts", "label"]].drop_duplicates().reset_index(drop=True)

        scope_rows = self._timeline_group_scope_rows()
        if scope_rows is not None and not scope_rows.empty:
            scoped = scope_rows.copy()
            scoped["system_id"] = pd.to_numeric(scoped.get("system_id"), errors="coerce")
            scoped["dataset_id"] = pd.to_numeric(scoped.get("dataset_id"), errors="coerce")
            scoped["import_id"] = pd.to_numeric(scoped.get("import_id"), errors="coerce")
            scoped = scoped.dropna(subset=["system_id", "dataset_id"])
            scoped = scoped[scoped["dataset_id"] > 0]
            scoped = scoped[["system_id", "dataset_id", "import_id"]].drop_duplicates().reset_index(drop=True)
            if not scoped.empty:
                points_df = (
                    points_df.assign(_scope_join_key=1)
                    .merge(scoped.assign(_scope_join_key=1), on="_scope_join_key", how="inner")
                    .drop(columns=["_scope_join_key"])
                    .reset_index(drop=True)
                )

        return {
            "clusters_saved": int(len(normalized_names)),
            "group_labels": int(labels_df["label"].nunique()),
            "group_points": int(len(points_df)),
        }, labels_df, points_df

    def _timeline_group_scope_rows(self) -> pd.DataFrame:
        columns = ["system_id", "dataset_id", "import_id"]
        filters = self._last_training_context.get("filters") if isinstance(self._last_training_context, dict) else {}
        if not isinstance(filters, dict):
            filters = {}

        systems = [str(item).strip() for item in (filters.get("systems") or []) if str(item).strip()]
        datasets = [str(item).strip() for item in (filters.get("datasets") or []) if str(item).strip()]
        import_ids: list[int] = []
        for item in (filters.get("import_ids") or []):
            if item is None:
                continue
            try:
                import_ids.append(int(item))
            except Exception:
                continue
        feature_payloads = self._last_training_context.get("feature_payloads") if isinstance(self._last_training_context, dict) else []
        feature_ids: list[int] = []
        for payload in feature_payloads or []:
            fid = self._safe_feature_id(payload)
            if fid is not None:
                feature_ids.append(int(fid))

        try:
            db = self._ensure_data_model().db
            with db.connection() as con:
                if import_ids:
                    ph = ",".join(["?"] * len(import_ids))
                    frame = con.execute(
                        f"""
                        SELECT DISTINCT
                            ds.system_id AS system_id,
                            i.dataset_id AS dataset_id,
                            i.id AS import_id
                        FROM imports i
                        JOIN datasets ds ON ds.id = i.dataset_id
                        WHERE i.id IN ({ph})
                        ORDER BY ds.system_id, i.dataset_id, i.id
                        """,
                        import_ids,
                    ).df()
                elif systems or datasets:
                    where: list[str] = []
                    params: list[object] = []
                    if systems:
                        ph = ",".join(["?"] * len(systems))
                        where.append(f"sy.name IN ({ph})")
                        params.extend(systems)
                    if datasets:
                        ph = ",".join(["?"] * len(datasets))
                        where.append(f"ds.name IN ({ph})")
                        params.extend(datasets)
                    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
                    frame = con.execute(
                        f"""
                        SELECT DISTINCT
                            ds.system_id AS system_id,
                            ds.id AS dataset_id,
                            CAST(-1 AS INTEGER) AS import_id
                        FROM datasets ds
                        JOIN systems sy ON sy.id = ds.system_id
                        {where_sql}
                        ORDER BY ds.system_id, ds.id
                        """,
                        params,
                    ).df()
                elif feature_ids:
                    ph = ",".join(["?"] * len(feature_ids))
                    frame = con.execute(
                        f"""
                        SELECT DISTINCT
                            fdm.system_id AS system_id,
                            fdm.dataset_id AS dataset_id,
                            CAST(-1 AS INTEGER) AS import_id
                        FROM feature_dataset_map fdm
                        WHERE fdm.feature_id IN ({ph})
                        ORDER BY fdm.system_id, fdm.dataset_id
                        """,
                        feature_ids,
                    ).df()
                else:
                    frame = pd.DataFrame(columns=columns)
        except Exception:
            frame = pd.DataFrame(columns=columns)

        if frame is None or frame.empty:
            return pd.DataFrame(columns=columns)
        normalized = frame.copy()
        for column in columns:
            normalized[column] = pd.to_numeric(normalized.get(column), errors="coerce")
        normalized = (
            normalized.dropna(subset=["system_id", "dataset_id"])
            .astype({"system_id": int, "dataset_id": int, "import_id": int})
            .drop_duplicates()
            .reset_index(drop=True)
        )
        return normalized.loc[:, columns]

    @staticmethod
    def _timeline_point_end_times(ts: pd.Series) -> pd.Series:
        """Return point end boundaries using next distinct timestamp, with inferred tail step."""
        times = pd.to_datetime(ts, errors="coerce")
        out = pd.Series([pd.NaT] * len(times), index=times.index, dtype="datetime64[ns]")
        if times.empty:
            return out

        unique_times = pd.Series(pd.unique(times.dropna())).sort_values(kind="stable").reset_index(drop=True)
        if unique_times.empty:
            return out

        diffs = unique_times.diff().dropna()
        positive_diffs = diffs[diffs > pd.Timedelta(0)]
        if positive_diffs.empty:
            inferred_step = pd.Timedelta(seconds=1)
        else:
            inferred_step = pd.Timedelta(positive_diffs.min())

        next_times = unique_times.shift(-1)
        next_times.iloc[-1] = unique_times.iloc[-1] + inferred_step
        end_map = {pd.Timestamp(start): pd.Timestamp(end) for start, end in zip(unique_times, next_times)}

        mapped = times.map(end_map)
        out.loc[mapped.index] = mapped
        return out

    def save_timeline_clusters_as_groups(
        self,
        cluster_labels: dict[int, str],
        *,
        kind: str = "som_cluster",
        save_as_timeframes: bool = True,
        on_finished: Optional[Callable[[dict[str, int]], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        def _run():
            base_summary, labels_df, points_df = self._build_timeline_cluster_group_frames(
                cluster_labels,
                save_as_timeframes=save_as_timeframes,
            )
            model = self._ensure_data_model()
            model_summary = model.insert_group_labels_and_points(
                kind=kind,
                labels_df=labels_df,
                points_df=points_df,
            )
            return {
                "clusters_saved": int(base_summary.get("clusters_saved", 0)),
                "group_labels": int(model_summary.get("group_labels", 0)),
                "group_points": int(model_summary.get("group_points", 0)),
            }

        def _on_finished(summary: dict[str, int]):
            if callable(on_finished):
                on_finished(summary)

        def _on_error(message: str):
            if callable(on_error):
                on_error(message)

        run_in_thread(
            _run,
            on_result=_on_finished,
            on_error=_on_error,
            owner=self,
            key="som_save_timeline_clusters",
            cancel_previous=True,
        )

    # ------------------------------------------------------------------
    def _payloads_for_labels(self, labels: Sequence[str]) -> list[dict]:
        try:
            model = self._ensure_data_model()
            df = model.features_for_systems_datasets()
        except Exception:
            df = pd.DataFrame(
                columns=["feature_id", "label", "base_name", "source", "unit", "type"]
            )
        if df is None or df.empty:
            return []
        mapping = {
            str(row["label"]): row
            for _, row in df.iterrows()
            if pd.notna(row.get("label"))
        }
        payloads: list[dict] = []
        for label in labels:
            row = mapping.get(str(label))
            if row is None:
                continue
            payloads.append(
                dict(
                    label=str(label),
                    feature_id=row.get("feature_id"),
                    base_name=row.get("base_name"),
                    source=row.get("source"),
                    unit=row.get("unit"),
                    type=row.get("type"),
                )
            )
        return payloads

    @staticmethod
    def _label_free_name(selection: FeatureSelection, index: int) -> str:
        parts: list[str] = []
        for piece in (selection.base_name, selection.source, selection.unit, selection.type):
            if piece and piece not in parts:
                parts.append(str(piece))
        if parts:
            return " / ".join(parts)
        if selection.feature_id is not None:
            return f"Feature {selection.feature_id}"
        return f"Feature {index + 1}"

    @staticmethod
    def _unique_name(name: str, used: set[str]) -> str:
        base_name = name or "Feature"
        candidate = base_name
        suffix = 2
        while candidate in used:
            candidate = f"{base_name} ({suffix})"
            suffix += 1
        used.add(candidate)
        return candidate

    def _build_feature_display_map(self, payloads: Sequence[dict]) -> dict[str, str]:
        selections: list[FeatureSelection] = []
        for payload in payloads:
            try:
                selections.append(FeatureSelection.from_payload(payload))
            except TypeError:
                selections.append(
                    FeatureSelection(
                        feature_id=payload.get("feature_id"),
                        label=payload.get("label"),
                        base_name=payload.get("base_name"),
                        source=payload.get("source"),
                        unit=payload.get("unit"),
                        type=payload.get("type"),
                        lag_seconds=payload.get("lag_seconds"),
                    )
                )

        used_actual: set[str] = set()
        used_display: set[str] = set()
        mapping: dict[str, str] = {}
        for idx, sel in enumerate(selections):
            actual = sel.display_name() or f"Feature {idx + 1}"
            actual = self._unique_name(actual, used_actual)
            display = self._label_free_name(sel, idx)
            display = self._unique_name(display, used_display)
            mapping[actual] = display
        return mapping

    def feature_display_name(self, name: str) -> str:
        return self._feature_display_map.get(name, name)

    @staticmethod
    def _safe_feature_id(payload: object) -> Optional[int]:
        if not isinstance(payload, dict):
            return None
        fid = payload.get("feature_id")
        if fid is None:
            return None
        try:
            return int(fid)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _map_frame_from_records(
        records: Optional[Sequence[dict]],
        map_shape: tuple[int, int],
    ) -> pd.DataFrame:
        frame = records_to_frame(records)
        if frame.empty:
            return pd.DataFrame()
        height, width = map_shape
        trimmed = frame.iloc[:height, :width].copy()
        trimmed.index = pd.RangeIndex(start=0, stop=height, step=1, name="x")
        trimmed.columns = pd.RangeIndex(start=0, stop=width, step=1, name="y")
        return trimmed

    @staticmethod
    def _restore_square_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if df.shape[0] == len(df.columns):
            df = df.copy()
            df.index = pd.Index(df.columns)
        return df

    def preprocessed_features_dataframe(
        self,
        feature_payloads: Sequence[dict],
        *,
        data_frame: pd.DataFrame,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
        group_ids: Optional[Sequence[int]] = None,
        preprocessing: Optional[dict] = None,
    ) -> pd.DataFrame:
        if not feature_payloads:
            return pd.DataFrame()

        if isinstance(data_frame, pd.DataFrame):
            return data_frame.copy()
        raise RuntimeError("SomViewModel requires a pre-fetched DataFrame from DataSelectorWidget.")

    # ------------------------------------------------------------------
    # @ai(gpt-5, codex, refactor, 2026-03-02)
    def train_from_labels(
        self,
        feature_labels: Sequence[str],
        *,
        feature_payloads: Optional[Sequence[dict]] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
        group_ids: Optional[Sequence[int]] = None,
        months: Optional[Sequence[int]] = None,
        preprocessing: Optional[dict] = None,
        map_shape: tuple[Optional[int], Optional[int]] = (None, None),
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        epochs: Optional[int] = None,
        normalisation: str = "zscore",
        training_mode: str = "batch",
        stop_event: Optional[threading.Event] = None,
        progress_callback: Optional[callable] = None,
        status_callback: Optional[callable] = None,
        data_frame: Optional[pd.DataFrame] = None,
    ) -> SomResult:
        if not feature_labels:
            raise ValueError("Select at least one feature")
        if len(feature_labels) < 2:
            raise ValueError("Select at least two features to train SOM.")

        self._ensure_data_model()

        if feature_payloads is None:
            feature_payloads = self._payloads_for_labels(feature_labels)
        if not feature_payloads:
            raise ValueError("Unable to resolve feature metadata for the selected features")

        self._last_neuron_clusters = None
        self._last_feature_clusters = None
        full_feature_display_map = self._build_feature_display_map(feature_payloads)

        params = dict(preprocessing or {})
        params.pop("target_points", None)

        months_set = {int(m) for m in (months or []) if m is not None}
        base_df = self.preprocessed_features_dataframe(
            [dict(payload) for payload in feature_payloads],
            data_frame=data_frame,
            start=start,
            end=end,
            systems=systems,
            datasets=datasets,
            import_ids=import_ids,
            group_ids=group_ids,
            preprocessing=params,
        )
        if base_df is None or base_df.empty:
            raise ValueError("No measurements available for the selected features")

        if months_set:
            tcol = pd.to_datetime(base_df["t"], errors="coerce")
            base_df = base_df[tcol.dt.month.isin(months_set)]

        base_df["t"] = pd.to_datetime(base_df["t"], errors="coerce")
        base_df = base_df.dropna(subset=["t"])
        feature_cols = [c for c in base_df.columns if c != "t"]
        if not feature_cols:
            raise ValueError("No measurements available for the selected features")

        numeric_df = base_df.set_index("t")[feature_cols].apply(pd.to_numeric, errors="coerce")
        nan_ratio = numeric_df.isna().mean(axis=0)
        dropped_sparse_cols = [
            name
            for name in feature_cols
            if float(nan_ratio.get(name, 0.0)) > TRAINING_SPARSE_FEATURE_NAN_RATIO_THRESHOLD
        ]
        if dropped_sparse_cols:
            dropped_display = [full_feature_display_map.get(name, name) for name in dropped_sparse_cols]

            def _emit_sparse_warning(names: list[str]) -> None:
                self.sparse_features_dropped.emit(list(names))

            run_in_main_thread(_emit_sparse_warning, dropped_display)
            feature_cols = [name for name in feature_cols if name not in set(dropped_sparse_cols)]

        dropped_static_cols: list[str] = []
        if feature_cols:
            dropped_static_cols = [
                name
                for name in feature_cols
                if int(pd.Series(numeric_df[name]).dropna().nunique()) <= TRAINING_STATIC_FEATURE_MAX_UNIQUE_NON_NULL
            ]
            if dropped_static_cols:
                dropped_display = [full_feature_display_map.get(name, name) for name in dropped_static_cols]

                def _emit_static_warning(names: list[str]) -> None:
                    self.static_features_dropped.emit(list(names))

                run_in_main_thread(_emit_static_warning, dropped_display)
                feature_cols = [name for name in feature_cols if name not in set(dropped_static_cols)]

        if not feature_cols:
            raise ValueError("All selected features were dropped by training rules (>95% missing or static values).")
        if len(feature_cols) < 2:
            raise ValueError("SOM needs at least two non-static features with enough data.")

        feature_cols_set = set(feature_cols)
        filtered_feature_payloads = [
            dict(payload)
            for payload in feature_payloads
            if display_name(dict(payload or {})) in feature_cols_set
        ]
        if not filtered_feature_payloads:
            raise ValueError("Unable to resolve feature metadata for the filtered SOM feature set")

        self._feature_display_map = self._build_feature_display_map(filtered_feature_payloads)
        filtered_feature_labels = [self._feature_display_map.get(name, name) for name in feature_cols]

        combined = numeric_df.loc[:, feature_cols].dropna(how="any")

        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        if combined.empty:
            raise ValueError("No rows remain after aligning the selected features")

        auto_dim = max(2, math.ceil(math.sqrt(5 * math.sqrt(max(1, len(combined))))))
        width_hint, height_hint = map_shape
        width = int(width_hint) if width_hint else auto_dim
        height = int(height_hint) if height_hint else auto_dim

        result = self._service.train(
            combined,
            map_shape=(width, height),
            sigma=sigma,
            learning_rate=learning_rate,
            num_epochs=epochs,
            normalisation=normalisation,
            training_mode=training_mode,
            stop_event=stop_event,
            progress_callback=progress_callback,
            status_callback=status_callback,
        )
        self._result = result
        self._last_dataframe = combined
        self._last_training_context = {
            "feature_labels": list(filtered_feature_labels),
            "feature_payloads": filtered_feature_payloads,
            "sparse_features_dropped": [full_feature_display_map.get(name, name) for name in dropped_sparse_cols],
            "static_features_dropped": [full_feature_display_map.get(name, name) for name in dropped_static_cols],
            "filters": normalize_for_json(
                {
                    "features": filtered_feature_payloads,
                    "start": start,
                    "end": end,
                    "group_ids": list(group_ids) if group_ids else None,
                    "systems": list(systems) if systems else None,
                    "datasets": list(datasets) if datasets else None,
                    "import_ids": list(import_ids) if import_ids else None,
                }
            ),
            "preprocessing": normalize_for_json(params),
            "training": normalize_for_json(
                {
                    "map_shape": [int(width), int(height)],
                    "sigma": float(sigma),
                    "learning_rate": float(learning_rate),
                    "epochs": int(epochs) if epochs is not None else None,
                    "normalisation": str(normalisation),
                    "training_mode": str(training_mode),
                    "neighborhood_function": "gaussian",
                }
            ),
        }
        return result

    # ------------------------------------------------------------------
    def _reset_status(self, *, preserve_text: bool = False) -> None:
        if not preserve_text:
            self.status_changed.emit("SOM ready")
        clear_progress()

    def _apply_status_text(self, message: Optional[str]) -> None:
        text = "" if message is None else str(message)
        self._current_status_text = text
        if text:
            try:
                set_status_text(text)
            except Exception:
                logger.warning("Exception in _apply_status_text", exc_info=True)
        else:
            try:
                clear_status_text()
            except Exception:
                logger.warning("Exception in _apply_status_text", exc_info=True)

    def _handle_status_update(self, message: Optional[str]) -> None:
        text = "" if message is None else str(message)
        self.status_changed.emit(text)

    def _handle_service_status_update(self, message: Optional[str]) -> None:
        text = str(message or "").strip()
        if self._training_running and text:
            if self._current_status_text != "Training SOM...":
                self.status_changed.emit("Training SOM...")

    def _cluster_set_running_status(self, target: str) -> None:
        def _update() -> None:
            set_progress(0)
            target_label = "neurons" if target == "neuron" else "features"
            self.status_changed.emit(f"Clustering SOM {target_label}...")

        run_in_main_thread(_update)

    def _cluster_set_finished_status(self, target: str) -> None:
        def _update() -> None:
            clear_progress()
            self.status_changed.emit(f"SOM {target} clustering finished.")

        run_in_main_thread(_update)

    def _cluster_set_failed_status(self, target: str, reason: Optional[str] = None) -> None:
        def _update() -> None:
            clear_progress()
            text = str(reason).strip() if reason else "unknown error"
            text = f"SOM {target} clustering failed: {text}"
            self.status_changed.emit(text)

        run_in_main_thread(_update)

    def _cluster_update_progress(self, target: str, percent: int) -> None:
        def _update() -> None:
            try:
                pct = int(percent)
            except Exception:
                pct = 0
            set_progress(max(0, min(100, pct)))

        run_in_main_thread(_update)

    # ------------------------------------------------------------------
    # @ai(gpt-5, codex, refactor, 2026-02-28)
    def train_som(
        self,
        feature_labels: Sequence[str],
        *,
        feature_payloads: Optional[Sequence[dict]] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
        group_ids: Optional[Sequence[int]] = None,
        months: Optional[Sequence[int]] = None,
        preprocessing: Optional[dict] = None,
        map_shape: tuple[Optional[int], Optional[int]] = (None, None),
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        epochs: Optional[int] = None,
        normalisation: str = "zscore",
        training_mode: str = "batch",
        on_finished=None,
        on_error=None,
        data_frame: Optional[pd.DataFrame] = None,
    ) -> None:
        if feature_payloads is not None:
            # Synchronize overlay selections with the features passed to training.
            self.set_selected_feature_payloads(feature_payloads)
        set_progress(0)
        self.status_changed.emit("Training SOM...")
        self._training_running = True

        def _run(stop_event=None, progress_callback=None, status_callback=None):
            self.training_started.emit()
            try:
                try:
                    result = self.train_from_labels(
                        feature_labels,
                        feature_payloads=feature_payloads,
                        start=start,
                        end=end,
                        systems=systems,
                        datasets=datasets,
                        import_ids=import_ids,
                        group_ids=group_ids,
                        months=months,
                        preprocessing=preprocessing,
                        map_shape=map_shape,
                        sigma=sigma,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        normalisation=normalisation,
                        training_mode=training_mode,
                        stop_event=stop_event,
                        progress_callback=progress_callback,
                        status_callback=status_callback,
                        data_frame=data_frame,
                    )
                    return result
                except ValueError as exc:
                    return {"validation_error": str(exc)}
            finally:
                run_in_main_thread(partial(self._reset_status, preserve_text=True))

        def _on_finished(result: SomResult):
            if isinstance(result, dict) and result.get("validation_error"):
                _on_error(str(result.get("validation_error")))
                return
            self._training_running = False
            self.status_changed.emit("SOM training finished.")
            self.training_finished.emit(result)
            if callable(on_finished):
                on_finished(result)

        def _on_error(message: str):
            self._training_running = False
            text = str(message).strip() if message else "Unknown error"
            self.status_changed.emit(f"SOM training failed: {text}")
            clear_progress()
            self.error_occurred.emit(message)
            if callable(on_error):
                on_error(message)

        # Start the worker and keep a reference so we can stop it on close
        # Map backend callbacks to GUI helpers via run_in_main_thread
        run_in_thread(
            _run,
            on_result=_on_finished,
            on_progress=lambda p: run_in_main_thread(
                lambda: set_progress(max(0, min(100, int(p))))
            ),
            on_error=_on_error,
            status_callback=self._handle_service_status_update,
            owner=self,
            key="som_train",
            cancel_previous=True,
        )

    # ------------------------------------------------------------------
    # @ai(gpt-5, codex, refactor, 2026-02-28)
    def cluster_neurons(
        self,
        *,
        max_k: int,
        n_clusters: Optional[int],
        scoring: str,
        on_finished: Optional[Callable[[NeuronClusteringResult], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        if self._result is None:
            raise RuntimeError("Train a SOM model before clustering neurons")

        self._neuron_cluster_running = True
        self._cluster_set_running_status("neuron")

        def _run(*, progress_callback=None):
            return self._service.cluster_neurons(
                n_clusters=n_clusters,
                k_list_max=int(max_k),
                scoring=scoring,
                progress_callback=progress_callback,
            )

        def _on_finished(result: NeuronClusteringResult):
            self._neuron_cluster_running = False
            self._cluster_set_finished_status("neuron")
            self._last_neuron_clusters = result
            # Clear cluster names when new clustering is performed
            self._cluster_names = {}
            self.neuron_clusters_updated.emit(result)
            if callable(on_finished):
                on_finished(result)

        def _on_error(message: str):
            self._neuron_cluster_running = False
            self._cluster_set_failed_status("neuron", message)
            self.clustering_error.emit(message)
            if callable(on_error):
                on_error(message)

        run_in_thread(
            _run,
            on_result=_on_finished,
            on_error=_on_error,
            on_progress=lambda p: self._cluster_update_progress("neurons", p),
            owner=self,
            key="som_cluster_neurons",
            cancel_previous=True,
        )

    # @ai(gpt-5, codex, refactor, 2026-02-28)
    def cluster_features(
        self,
        *,
        max_k: int,
        n_clusters: Optional[int],
        scoring: str,
        on_finished: Optional[Callable[[FeatureClusteringResult], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        if self._result is None:
            raise RuntimeError("Train a SOM model before clustering features")

        self._feature_cluster_running = True
        self._cluster_set_running_status("feature")

        def _run(*, progress_callback=None):
            return self._service.cluster_features(
                n_clusters=n_clusters,
                k_list_max=int(max_k) if max_k is not None else None,
                scoring=scoring,
                progress_callback=progress_callback,
            )

        def _on_finished(result: FeatureClusteringResult):
            self._feature_cluster_running = False
            self._cluster_set_finished_status("feature")
            self._last_feature_clusters = result
            self.feature_clusters_updated.emit(result)
            if callable(on_finished):
                on_finished(result)

        def _on_error(message: str):
            self._feature_cluster_running = False
            self._cluster_set_failed_status("feature", message)
            self.clustering_error.emit(message)
            if callable(on_error):
                on_error(message)

        run_in_thread(
            _run,
            on_result=_on_finished,
            on_error=_on_error,
            on_progress=lambda p: self._cluster_update_progress("features", p),
            owner=self,
            key="som_cluster_features",
            cancel_previous=True,
        )

    # ------------------------------------------------------------------
    def last_neuron_clusters(self) -> Optional[NeuronClusteringResult]:
        return self._last_neuron_clusters

    def last_feature_clusters(self) -> Optional[FeatureClusteringResult]:
        return self._last_feature_clusters

    def close(self) -> None:
        self._close_database()
