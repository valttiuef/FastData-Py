from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
"""Service for unsupervised clustering utilities."""

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from .scoring import ScoreType, score_dispatch

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from sklearn.base import ClusterMixin  # type: ignore[import-not-found]

try:  # pragma: no cover - optional dependency
    from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - fallback when sklearn is absent
    ConvergenceWarning = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ClusteringMethodSpec:
    """Metadata describing an available clustering algorithm."""

    key: str
    label: str
    supports_auto_k: bool = True
    requires_n_clusters: bool = True
    accepts_random_state: bool = True
    default_params: Mapping[str, object] = field(default_factory=dict)


def _method_registry() -> Dict[str, ClusteringMethodSpec]:
    """Return the clustering method registry."""

    return {
        "kmeans": ClusteringMethodSpec(
            key="kmeans",
            label="K-Means",
            supports_auto_k=True,
            requires_n_clusters=True,
            accepts_random_state=True,
            default_params={"n_init": 10},
        ),
        "minibatch_kmeans": ClusteringMethodSpec(
            key="minibatch_kmeans",
            label="Mini-Batch K-Means",
            supports_auto_k=True,
            requires_n_clusters=True,
            accepts_random_state=True,
            default_params={"n_init": 10},
        ),
        "agglomerative": ClusteringMethodSpec(
            key="agglomerative",
            label="Agglomerative",
            supports_auto_k=True,
            requires_n_clusters=True,
            accepts_random_state=False,
            default_params={"linkage": "ward"},
        ),
        "spectral": ClusteringMethodSpec(
            key="spectral",
            label="Spectral",
            supports_auto_k=True,
            requires_n_clusters=True,
            accepts_random_state=True,
            default_params={"assign_labels": "kmeans"},
        ),
    }


_METHOD_REGISTRY = _method_registry()


def _candidate_k_values(n_samples: int, k_min: int, k_max: int) -> List[int]:
    """Return the candidate K values within dataset limits."""
    k_min_adj = max(2, min(k_min, n_samples - 1))
    k_max_adj = max(k_min_adj, min(k_max, n_samples - 1))
    return list(range(k_min_adj, k_max_adj + 1))


def _merge_params(
    spec: ClusteringMethodSpec,
    overrides: Optional[Mapping[str, object]],
) -> Dict[str, object]:
    params: Dict[str, object] = dict(spec.default_params)
    if overrides:
        for key, value in overrides.items():
            params[key] = value
    return params


def _instantiate_estimator(
    method: str,
    *,
    spec: ClusteringMethodSpec,
    n_clusters: Optional[int],
    random_state: int,
    params: Mapping[str, object],
) -> "ClusterMixin":
    if spec.requires_n_clusters and n_clusters is None:
        raise ValueError(f"Method '{method}' requires 'n_clusters'.")

    kwargs = dict(params)
    if spec.requires_n_clusters and n_clusters is not None:
        kwargs["n_clusters"] = int(n_clusters)
    if spec.accepts_random_state:
        kwargs.setdefault("random_state", int(random_state))

    try:
        if method == "kmeans":
            from sklearn.cluster import KMeans  # type: ignore[import-not-found]

            return KMeans(**kwargs)
        if method == "minibatch_kmeans":
            from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-not-found]

            return MiniBatchKMeans(**kwargs)
        if method == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-not-found]

            return AgglomerativeClustering(**kwargs)
        if method == "spectral":
            from sklearn.cluster import SpectralClustering  # type: ignore[import-not-found]

            return SpectralClustering(**kwargs)
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "scikit-learn is required for clustering operations."
        ) from exc

    raise ValueError(f"Unknown clustering method '{method}'.")


def _fit_predict(estimator, X):
    with warnings.catch_warnings():
        if ConvergenceWarning is not None:
            warnings.simplefilter("ignore", ConvergenceWarning)

        fit_predict = getattr(estimator, "fit_predict", None)
        if callable(fit_predict):
            labels = fit_predict(X)
        else:
            fit = getattr(estimator, "fit", None)
            if not callable(fit):
                raise TypeError("Estimator must implement fit_predict(X) or fit(X).")

            fit(X)
            labels = getattr(estimator, "labels_", None)
            if labels is None:
                raise AttributeError("Estimator fit(), but does not provide labels_.")
    return np.asarray(labels)


def _normalise_labels(labels: np.ndarray) -> Tuple[np.ndarray, int]:
    unique, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(int), int(len(unique))


def _compute_centers(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    centers = np.zeros((k, X.shape[1]), dtype=float)
    for idx in range(k):
        members = X[labels == idx]
        if members.size:
            centers[idx] = np.nanmean(members, axis=0)
    return centers


def _auto_choose_k(
    X: np.ndarray,
    *,
    method: str,
    spec: ClusteringMethodSpec,
    params: Mapping[str, object],
    k_min: int,
    k_max: int,
    random_state: int,
    scoring: ScoreType,
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Try K=k_min..k_max and return the K that maximizes the score."""

    n_samples = X.shape[0]
    candidate_ks = _candidate_k_values(n_samples, k_min, k_max)

    best_k, best_score = None, -np.inf
    for idx, k in enumerate(candidate_ks, start=1):
        try:
            estimator = _instantiate_estimator(
                method,
                spec=spec,
                n_clusters=k,
                random_state=random_state,
                params=params,
            )
            labels = _fit_predict(estimator, X)
            score = score_dispatch(X, labels, scoring)
        except Exception:
            score = -np.inf

        if score > best_score:
            best_k, best_score = k, score

        if progress is not None:
            try:
                progress(idx, len(candidate_ks))
            except Exception:
                logger.warning("Exception in _auto_choose_k", exc_info=True)

    if best_k is not None:
        return best_k
    if candidate_ks:
        return candidate_ks[-1]
    return max(2, k_min)


@dataclass
class NeuronClusteringResult:
    k: int
    labels_1d: np.ndarray          # length = n_units
    centers: np.ndarray            # (k, n_features)
    counts: np.ndarray             # (k,)
    labels_grid: pd.DataFrame      # (height x width) cluster id per unit
    bmu_cluster_labels: np.ndarray # per-row cluster label for data (like dataClusterLabelsOriginal)


@dataclass
class FeatureClusteringResult:
    k: int
    labels: np.ndarray             # (n_features,)
    centers: np.ndarray            # (k, n_units)
    counts: np.ndarray             # (k,)
    index: List[str]


class ClusteringService:
    """Utility collection for re-usable clustering helpers."""

    def available_methods(self) -> List[ClusteringMethodSpec]:
        """Return the supported clustering algorithms."""

        return sorted(_METHOD_REGISTRY.values(), key=lambda spec: spec.label.lower())

    def method_spec(self, method: str) -> ClusteringMethodSpec:
        """Return the specification for the given clustering method."""

        try:
            return _METHOD_REGISTRY[method]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown clustering method '{method}'.") from exc

    @staticmethod
    def _zscore_safe(v: np.ndarray) -> np.ndarray:
        m = np.nanmean(v)
        s = np.nanstd(v)
        if not np.isfinite(s) or s == 0:
            s = 1.0
        r = (v - m) / s
        r[~np.isfinite(r)] = 0.0
        return r

    @staticmethod
    def _progress_emitter(
        callback: Optional[Callable[[int], None]],
        total_steps: int,
    ) -> Callable[[int], None]:
        if callback is None or total_steps <= 0:
            return lambda _completed: None

        def _emit(completed: int) -> None:
            try:
                fraction = float(completed) / float(total_steps)
            except Exception:
                fraction = 0.0
            try:
                callback(int(max(0, min(100, round(fraction * 100)))))
            except Exception:
                logger.warning("Exception in _emit", exc_info=True)

        return _emit

    def cluster_neurons(
        self,
        *,
        codebook: np.ndarray,
        map_shape: Tuple[int, int],
        n_clusters: Optional[int] = None,
        k_list_max: int = 16,
        random_state: int = 42,
        scoring: ScoreType = "silhouette",
        bmu_indices: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        method: str = "kmeans",
        method_params: Optional[Mapping[str, object]] = None,
    ) -> NeuronClusteringResult:
        """Cluster SOM neurons using one of the supported algorithms."""
        if codebook.ndim != 2:
            raise ValueError("codebook must be a 2D array")

        n_units = codebook.shape[0]

        spec = self.method_spec(method)
        params = _merge_params(spec, method_params)

        candidate_ks: List[int] = []
        if n_clusters is None:
            if not spec.supports_auto_k:
                raise ValueError(
                    f"Method '{method}' does not support automatic K selection."
                )
            candidate_ks = _candidate_k_values(
                n_samples=n_units,
                k_min=2,
                k_max=min(k_list_max, n_units - 1),
            )

        total_steps = max(1, len(candidate_ks) + 1)
        emit_progress = self._progress_emitter(progress_callback, total_steps)

        if n_clusters is None:

            def _auto_progress(completed: int, _total: int) -> None:
                emit_progress(completed)

            n_clusters = _auto_choose_k(
                codebook,
                method=method,
                spec=spec,
                params=params,
                k_min=2,
                k_max=min(k_list_max, n_units - 1),
                random_state=random_state,
                scoring=scoring,
                progress=_auto_progress,
            )

        estimator = _instantiate_estimator(
            method,
            spec=spec,
            n_clusters=n_clusters,
            random_state=random_state,
            params=params,
        )
        labels_raw = _fit_predict(estimator, codebook)
        labels_1d, k_final = _normalise_labels(labels_raw)
        counts = np.bincount(labels_1d, minlength=k_final)

        centers_attr = getattr(estimator, "cluster_centers_", None)
        if isinstance(centers_attr, np.ndarray) and centers_attr.shape[0] == k_final:
            centers = np.asarray(centers_attr, dtype=float)
        else:
            centers = _compute_centers(codebook, labels_1d, k_final)

        emit_progress(total_steps)

        width, height = map_shape
        if width * height != n_units:
            raise ValueError("map_shape does not match codebook dimensions")

        labels_grid = pd.DataFrame(
            labels_1d.reshape(width, height).T,
            index=pd.RangeIndex(start=0, stop=height, name="x"),
            columns=pd.RangeIndex(start=0, stop=width, name="y"),
        )

        if bmu_indices is not None and len(bmu_indices) > 0:
            bmu_cluster_labels = labels_1d[bmu_indices]
        else:
            bmu_cluster_labels = np.array([], dtype=int)

        return NeuronClusteringResult(
            k=k_final,
            labels_1d=labels_1d,
            centers=centers,
            counts=counts,
            labels_grid=labels_grid,
            bmu_cluster_labels=bmu_cluster_labels,
        )

    def cluster_features(
        self,
        *,
        codebook: np.ndarray,
        feature_names: Sequence[str],
        n_clusters: Optional[int] = None,
        k_list_max: Optional[int] = None,
        random_state: int = 0,
        scoring: ScoreType = "silhouette",
        progress_callback: Optional[Callable[[int], None]] = None,
        method: str = "kmeans",
        method_params: Optional[Mapping[str, object]] = None,
    ) -> FeatureClusteringResult:
        """Cluster SOM features by similarity of their component planes."""
        if codebook.ndim != 2:
            raise ValueError("codebook must be a 2D array")

        n_units, n_features = codebook.shape
        if len(feature_names) != n_features:
            raise ValueError("feature_names length must match number of features")

        spec = self.method_spec(method)
        params = _merge_params(spec, method_params)

        feature_planes = np.zeros((n_features, n_units), dtype=float)
        for j in range(n_features):
            v = codebook[:, j]
            feature_planes[j, :] = self._zscore_safe(v)

        if k_list_max is None:
            k_list_max = min(20, max(2, n_features - 1))

        candidate_ks: List[int] = []
        if n_clusters is None:
            if not spec.supports_auto_k:
                raise ValueError(
                    f"Method '{method}' does not support automatic K selection."
                )
            candidate_ks = _candidate_k_values(
                n_samples=n_features,
                k_min=2,
                k_max=min(k_list_max, n_features - 1),
            )

        total_steps = max(1, len(candidate_ks) + 1)
        emit_progress = self._progress_emitter(progress_callback, total_steps)

        if n_clusters is None:

            def _auto_progress(completed: int, _total: int) -> None:
                emit_progress(completed)

            n_clusters = _auto_choose_k(
                feature_planes,
                method=method,
                spec=spec,
                params=params,
                k_min=2,
                k_max=min(k_list_max, n_features - 1),
                random_state=random_state,
                scoring=scoring,
                progress=_auto_progress,
            )

        estimator = _instantiate_estimator(
            method,
            spec=spec,
            n_clusters=n_clusters,
            random_state=random_state,
            params=params,
        )
        labels_raw = _fit_predict(estimator, feature_planes)
        labels, k_final = _normalise_labels(labels_raw)
        counts = np.bincount(labels, minlength=k_final)

        centers_attr = getattr(estimator, "cluster_centers_", None)
        if isinstance(centers_attr, np.ndarray) and centers_attr.shape[0] == k_final:
            centers = np.asarray(centers_attr, dtype=float)
        else:
            centers = _compute_centers(feature_planes, labels, k_final)

        emit_progress(total_steps)

        return FeatureClusteringResult(
            k=k_final,
            labels=labels,
            centers=centers,
            counts=counts,
            index=list(feature_names),
        )
