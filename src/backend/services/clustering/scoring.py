from __future__ import annotations
"""Scoring utilities for clustering algorithms."""

from typing import Callable, TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from sklearn.metrics import (  # type: ignore[import-not-found]
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

ScoreType = Union[str, Callable[[np.ndarray, np.ndarray], float]]


def score_dispatch(X: np.ndarray, labels: np.ndarray, scoring: ScoreType) -> float:
    """Return a scalar 'higher is better' score for given labels on X."""
    if callable(scoring):
        return float(scoring(X, labels))

    s = (scoring or "silhouette").lower()
    try:
        from sklearn.metrics import (  # type: ignore[import-not-found]
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "scikit-learn is required for clustering score computations."
        ) from exc

    if s in ("silhouette", "sil"):
        return float(silhouette_score(X, labels, metric="euclidean"))
    if s in ("calinski_harabasz", "ch", "calinski", "harabasz"):
        return float(calinski_harabasz_score(X, labels))
    if s in ("davies_bouldin", "db", "davies"):
        # DB: lower is better -> negate so "higher is better"
        return -float(davies_bouldin_score(X, labels))
    raise ValueError(f"Unknown scoring='{scoring}'")
