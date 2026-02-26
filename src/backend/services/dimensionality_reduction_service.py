"""Dimensionality reduction utilities (e.g. PCA, PLS, ICA)."""
from __future__ import annotations
from typing import Callable, Mapping, Optional

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import FactorAnalysis, FastICA, PCA, TruncatedSVD


class _PLSFeatureProjector(BaseEstimator, TransformerMixin):
    """Pipeline-safe wrapper for PLSRegression that emits only X scores."""

    def __init__(
        self,
        n_components: int = 2,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-06,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self._pls: Optional[PLSRegression] = None

    def fit(self, X, y=None):
        self._pls = PLSRegression(
            n_components=self.n_components,
            scale=self.scale,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self._pls.fit(X, y)
        return self

    def transform(self, X):
        if self._pls is None:
            raise RuntimeError("PLSFeatureProjector is not fitted")
        transformed = self._pls.transform(X)
        if isinstance(transformed, tuple):
            transformed = transformed[0]
        return np.asarray(transformed)


class DimensionalityReductionService:
    """Build dimensionality reduction transformers for regression pipelines."""

    def __init__(self) -> None:
        self._reducer_factories: dict[
            str, tuple[str, Callable[[dict[str, object]], Optional[object]], dict[str, object]]
        ] = {
            "none": (
                "Disabled",
                lambda _params: None,
                {},
            ),
            "pca": (
                "PCA",
                lambda params: PCA(**params),
                {"n_components": None, "svd_solver": "auto", "whiten": False, "random_state": 0},
            ),
            "pls_regression": (
                "PLSRegression",
                lambda params: _PLSFeatureProjector(**params),
                {"n_components": 2, "scale": True, "max_iter": 500, "tol": 0.000001},
            ),
            "fast_ica": (
                "FastICA",
                lambda params: FastICA(**params),
                {
                    "n_components": None,
                    "algorithm": "parallel",
                    "whiten": "unit-variance",
                    "fun": "logcosh",
                    "max_iter": 1000,
                    "tol": 0.001,
                    "random_state": 0,
                },
            ),
            "factor_analysis": (
                "FactorAnalysis",
                lambda params: FactorAnalysis(**params),
                {
                    "n_components": None,
                    "svd_method": "randomized",
                    "iterated_power": 3,
                    "rotation": None,
                    "tol": 0.01,
                    "random_state": 0,
                },
            ),
            "truncated_svd": (
                "TruncatedSVD",
                lambda params: TruncatedSVD(**params),
                {
                    "n_components": 2,
                    "algorithm": "randomized",
                    "n_iter": 5,
                    "tol": 0.0,
                    "random_state": 0,
                },
            ),
        }

    def available_dimensionality_reducers(self) -> list[tuple[str, str, dict[str, object]]]:
        return [
            (key, label, defaults.copy())
            for key, (label, _factory, defaults) in self._reducer_factories.items()
        ]

    def has_reducer(self, reducer_key: str) -> bool:
        return reducer_key in self._reducer_factories

    def definition_for(
        self, reducer_key: str
    ) -> tuple[str, Callable[[dict[str, object]], Optional[object]], dict[str, object]]:
        if reducer_key not in self._reducer_factories:
            raise KeyError(reducer_key)
        return self._reducer_factories[reducer_key]

    def build_reducer(
        self,
        reducer_key: str,
        params: Optional[Mapping[str, object]],
        *,
        n_features: int,
        n_samples: int,
    ) -> tuple[str, Optional[object], dict[str, object]]:
        label, factory, defaults = self.definition_for(reducer_key)
        if reducer_key == "truncated_svd" and n_features < 2:
            return label, None, {}
        reducer_cfg = defaults.copy()
        if isinstance(params, Mapping):
            reducer_cfg.update(params)
        reducer_cfg = self._clean_reducer_params(reducer_cfg)
        reducer_cfg = self._normalize_reducer_config(
            reducer_key,
            reducer_cfg,
            n_features=n_features,
            n_samples=n_samples,
        )
        reducer_obj = factory(reducer_cfg)
        return label, reducer_obj, reducer_cfg

    def _clean_reducer_params(
        self,
        params: Optional[Mapping[str, object]],
    ) -> dict[str, object]:
        if not params:
            return {}
        cleaned: dict[str, object] = {}
        for key, value in dict(params).items():
            if value is None or value == "":
                continue
            cleaned[str(key)] = value
        return cleaned

    def _normalize_reducer_config(
        self,
        reducer_key: str,
        config: Mapping[str, object],
        *,
        n_features: int,
        n_samples: int,
    ) -> dict[str, object]:
        normalized = dict(config or {})
        if reducer_key == "none":
            return {}

        n_components = normalized.get("n_components")
        if n_components is None:
            return normalized
        try:
            n_comp = int(n_components)
        except Exception:
            return normalized
        if n_comp <= 0:
            normalized.pop("n_components", None)
            return normalized

        if reducer_key == "truncated_svd":
            upper = max(1, n_features - 1)
        else:
            upper = max(1, min(n_features, n_samples))

        normalized["n_components"] = max(1, min(n_comp, upper))
        return normalized
