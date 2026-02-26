
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Mapping, TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from backend.services import (

    ClusteringMethodSpec,
    ClusteringService,
    FeatureClusteringResult,
    NeuronClusteringResult,
    ScoreType,
)

import logging
logger = logging.getLogger(__name__)

from ...threading.runner import run_in_thread
from ...threading.utils import run_in_main_thread
from ...utils import set_progress, clear_progress, set_status_text

if TYPE_CHECKING:
    from .viewmodel import SomViewModel


@dataclass(frozen=True)
class ClusteringRequest:
    method: str
    max_k: Optional[int]
    n_clusters: Optional[int]
    scoring: ScoreType
    method_params: Optional[Mapping[str, object]] = None


class ClusteringViewModel(QObject):
    """Qt-friendly wrapper responsible for SOM clustering tasks."""
    
    # Signals
    neuron_clusters_updated = Signal(object)  # NeuronClusteringResult
    feature_clusters_updated = Signal(object)  # FeatureClusteringResult
    clustering_error = Signal(str)

    def __init__(self, som_viewmodel: "SomViewModel", parent: Optional[QObject] = None):
        super().__init__(parent)
        self._som_viewmodel = som_viewmodel
        self._service = ClusteringService()
        self._neuron_running = False
        self._feature_running = False

    # ------------------------------------------------------------------
    def available_methods(self) -> list[ClusteringMethodSpec]:
        return list(self._service.available_methods())

    def method_spec(self, method: str) -> ClusteringMethodSpec:
        return self._service.method_spec(method)

    # ------------------------------------------------------------------
    def _cluster_status_message(self, target: str, percent: Optional[int]) -> str:
        base = f"Clustering SOM {target}…"
        if percent is None:
            return base
        try:
            pct = int(percent)
        except Exception:
            pct = 0
        pct = max(0, min(100, pct))
        return f"{base} ({pct}%)"

    def _cluster_update_progress(self, target: str, percent: int) -> None:
        if target == "neurons" and not self._neuron_running:
            return
        if target == "features" and not self._feature_running:
            return
        message = self._cluster_status_message(target, percent)

        def _update() -> None:
            try:
                pct = int(percent)
            except Exception:
                pct = 0
            set_progress(max(0, min(100, pct)))
            try:
                self._som_viewmodel.status_changed.emit(message)
            except Exception:
                logger.warning("Exception in _update", exc_info=True)

        run_in_main_thread(_update)

    def _cluster_reset_status(self) -> None:
        def _reset() -> None:
            clear_progress()
            try:
                set_status_text("SOM ready")
            except Exception:
                logger.warning("Exception in _reset", exc_info=True)
            try:
                self._som_viewmodel.status_changed.emit("SOM ready")
            except Exception:
                logger.warning("Exception in _reset", exc_info=True)

        run_in_main_thread(_reset)

    # ------------------------------------------------------------------
    def _normalise_max_k(self, value: Optional[int], default: int) -> int:
        if value is None:
            return default
        try:
            k = int(value)
        except Exception:
            k = default
        return max(2, k)

    # ------------------------------------------------------------------
    def cluster_neurons(
        self,
        *,
        request: ClusteringRequest,
        on_finished: Optional[Callable[[NeuronClusteringResult], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        if self._som_viewmodel.result() is None:
            raise RuntimeError("Train a SOM model before clustering neurons")

        spec = self.method_spec(request.method)
        if request.n_clusters is None and not spec.supports_auto_k:
            raise ValueError("Selected method requires an explicit number of clusters")

        inputs = self._som_viewmodel.clustering_inputs()
        k_list_max = self._normalise_max_k(request.max_k, default=16)

        self._neuron_running = True
        self._cluster_update_progress("neurons", 0)

        def _run(*, progress_callback=None):
            try:
                return self._service.cluster_neurons(
                    codebook=inputs.codebook,
                    map_shape=inputs.map_shape,
                    n_clusters=request.n_clusters,
                    k_list_max=k_list_max,
                    random_state=42,
                    scoring=request.scoring,
                    bmu_indices=inputs.bmu_indices,
                    progress_callback=progress_callback,
                    method=request.method,
                    method_params=request.method_params,
                )
            finally:
                self._cluster_reset_status()

        def _on_finished(result: NeuronClusteringResult):
            self._neuron_running = False
            self.neuron_clusters_updated.emit(result)
            if callable(on_finished):
                on_finished(result)

        def _on_error(message: str):
            self._neuron_running = False
            self._cluster_reset_status()
            self.clustering_error.emit(message)
            if callable(on_error):
                on_error(message)

        run_in_thread(
            _run,
            on_result=_on_finished,
            on_error=_on_error,
            on_progress=lambda p: self._cluster_update_progress("neurons", p),
            owner=self,
            key="cluster_neurons",
            cancel_previous=True,
        )

    # ------------------------------------------------------------------
    def cluster_features(
        self,
        *,
        request: ClusteringRequest,
        on_finished: Optional[Callable[[FeatureClusteringResult], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        if self._som_viewmodel.result() is None:
            raise RuntimeError("Train a SOM model before clustering features")

        spec = self.method_spec(request.method)
        if request.n_clusters is None and not spec.supports_auto_k:
            raise ValueError("Selected method requires an explicit number of clusters")

        inputs = self._som_viewmodel.clustering_inputs()
        k_list_max = self._normalise_max_k(request.max_k, default=20)

        self._feature_running = True
        self._cluster_update_progress("features", 0)

        def _run(*, progress_callback=None):
            try:
                return self._service.cluster_features(
                    codebook=inputs.codebook,
                    feature_names=inputs.feature_names,
                    n_clusters=request.n_clusters,
                    k_list_max=k_list_max,
                    random_state=0,
                    scoring=request.scoring,
                    progress_callback=progress_callback,
                    method=request.method,
                    method_params=request.method_params,
                )
            finally:
                self._cluster_reset_status()

        def _on_finished(result: FeatureClusteringResult):
            self._feature_running = False
            self.feature_clusters_updated.emit(result)
            if callable(on_finished):
                on_finished(result)

        def _on_error(message: str):
            self._feature_running = False
            self._cluster_reset_status()
            self.clustering_error.emit(message)
            if callable(on_error):
                on_error(message)

        run_in_thread(
            _run,
            on_result=_on_finished,
            on_error=_on_error,
            on_progress=lambda p: self._cluster_update_progress("features", p),
            owner=self,
            key="cluster_features",
            cancel_previous=True,
        )
