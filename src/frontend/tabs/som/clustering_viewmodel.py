
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Mapping, TYPE_CHECKING
import time

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
from ...utils import set_progress, clear_progress
from backend.services.logging.perf_debug import perf_debug_log

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
    def _cluster_set_running_status(self, target: str) -> None:
        def _update() -> None:
            set_progress(0)
            target_label = "neurons" if target == "neuron" else "features"
            status_text = f"Clustering SOM {target_label}..."
            try:
                self._som_viewmodel.status_changed.emit(status_text)
            except Exception:
                logger.warning("Exception in _cluster_set_running_status", exc_info=True)

        run_in_main_thread(_update)

    def _cluster_set_preparing_status(self, target: str) -> None:
        def _update() -> None:
            set_progress(0)
            target_label = "neurons" if target == "neuron" else "features"
            status_text = f"Preparing SOM {target_label} clustering..."
            try:
                self._som_viewmodel.status_changed.emit(status_text)
            except Exception:
                logger.warning("Exception in _cluster_set_preparing_status", exc_info=True)

        run_in_main_thread(_update)

    def _cluster_update_progress(self, target: str, percent: int) -> None:
        if target == "neurons" and not self._neuron_running:
            return
        if target == "features" and not self._feature_running:
            return

        def _update() -> None:
            try:
                pct = int(percent)
            except Exception:
                pct = 0
            set_progress(max(0, min(100, pct)))

        run_in_main_thread(_update)

    def _cluster_set_finished_status(self, target: str) -> None:
        def _reset() -> None:
            clear_progress()
            status_text = f"SOM {target} clustering finished."
            try:
                self._som_viewmodel.status_changed.emit(status_text)
            except Exception:
                logger.warning("Exception in _cluster_set_finished_status", exc_info=True)

        run_in_main_thread(_reset)

    def _cluster_set_failed_status(self, target: str, reason: Optional[str] = None) -> None:
        def _reset() -> None:
            clear_progress()
            reason_text = str(reason).strip() if reason else "unknown error"
            text = f"SOM {target} clustering failed: {reason_text}"
            try:
                self._som_viewmodel.status_changed.emit(text)
            except Exception:
                logger.warning("Exception in _cluster_set_failed_status", exc_info=True)

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
    # @ai(gpt-5, codex, refactor, 2026-02-28)
    def cluster_neurons(
        self,
        *,
        request: ClusteringRequest,
        on_finished: Optional[Callable[[NeuronClusteringResult], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        # @ai(gpt-5, codex, observability, 2026-03-23)
        if self._som_viewmodel.result() is None:
            raise RuntimeError("Train a SOM model before clustering neurons")

        spec = self.method_spec(request.method)
        if request.n_clusters is None and not spec.supports_auto_k:
            raise ValueError("Selected method requires an explicit number of clusters")

        k_list_max = self._normalise_max_k(request.max_k, default=16)

        self._neuron_running = True
        self._cluster_set_preparing_status("neuron")
        started_at = time.perf_counter()
        perf_debug_log(
            "som.clustering.neurons",
            "cluster_started",
            method=request.method,
            n_clusters=request.n_clusters,
            max_k=k_list_max,
            scoring=request.scoring,
        )

        def _run(*, progress_callback=None):
            prep_started_at = time.perf_counter()
            perf_debug_log("som.clustering.neurons", "prepare_inputs_started")
            inputs = self._som_viewmodel.clustering_inputs(include_bmu_indices=True)
            perf_debug_log(
                "som.clustering.neurons",
                "prepare_inputs_finished",
                elapsed_ms=(time.perf_counter() - prep_started_at) * 1000.0,
                samples=(
                    len(inputs.bmu_indices)
                    if getattr(inputs, "bmu_indices", None) is not None
                    else 0
                ),
                units=(
                    int(inputs.codebook.shape[0])
                    if getattr(inputs, "codebook", None) is not None
                    else 0
                ),
                features=len(getattr(inputs, "feature_names", []) or []),
            )
            self._cluster_set_running_status("neuron")
            result = self._service.cluster_neurons(
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
            perf_debug_log(
                "som.clustering.neurons",
                "cluster_backend_finished",
                elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                k=getattr(result, "k", None),
            )
            return result

        def _on_finished(result: NeuronClusteringResult):
            self._neuron_running = False
            self._cluster_set_finished_status("neuron")
            self.neuron_clusters_updated.emit(result)
            perf_debug_log(
                "som.clustering.neurons",
                "cluster_ui_finished",
                elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                k=getattr(result, "k", None),
            )
            if callable(on_finished):
                on_finished(result)

        def _on_error(message: str):
            self._neuron_running = False
            self._cluster_set_failed_status("neuron", message)
            perf_debug_log(
                "som.clustering.neurons",
                "cluster_failed",
                elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                error=message,
            )
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
    # @ai(gpt-5, codex, refactor, 2026-02-28)
    def cluster_features(
        self,
        *,
        request: ClusteringRequest,
        on_finished: Optional[Callable[[FeatureClusteringResult], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        # @ai(gpt-5, codex, observability, 2026-03-23)
        if self._som_viewmodel.result() is None:
            raise RuntimeError("Train a SOM model before clustering features")

        spec = self.method_spec(request.method)
        if request.n_clusters is None and not spec.supports_auto_k:
            raise ValueError("Selected method requires an explicit number of clusters")

        k_list_max = self._normalise_max_k(request.max_k, default=20)

        self._feature_running = True
        self._cluster_set_preparing_status("feature")
        started_at = time.perf_counter()
        perf_debug_log(
            "som.clustering.features",
            "cluster_started",
            method=request.method,
            n_clusters=request.n_clusters,
            max_k=k_list_max,
            scoring=request.scoring,
        )

        def _run(*, progress_callback=None):
            prep_started_at = time.perf_counter()
            perf_debug_log("som.clustering.features", "prepare_inputs_started")
            inputs = self._som_viewmodel.clustering_inputs(include_bmu_indices=False)
            perf_debug_log(
                "som.clustering.features",
                "prepare_inputs_finished",
                elapsed_ms=(time.perf_counter() - prep_started_at) * 1000.0,
                units=(
                    int(inputs.codebook.shape[0])
                    if getattr(inputs, "codebook", None) is not None
                    else 0
                ),
                features=len(getattr(inputs, "feature_names", []) or []),
            )
            self._cluster_set_running_status("feature")
            result = self._service.cluster_features(
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
            perf_debug_log(
                "som.clustering.features",
                "cluster_backend_finished",
                elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                k=getattr(result, "k", None),
            )
            return result

        def _on_finished(result: FeatureClusteringResult):
            self._feature_running = False
            self._cluster_set_finished_status("feature")
            self.feature_clusters_updated.emit(result)
            perf_debug_log(
                "som.clustering.features",
                "cluster_ui_finished",
                elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                k=getattr(result, "k", None),
            )
            if callable(on_finished):
                on_finished(result)

        def _on_error(message: str):
            self._feature_running = False
            self._cluster_set_failed_status("feature", message)
            perf_debug_log(
                "som.clustering.features",
                "cluster_failed",
                elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                error=message,
            )
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
