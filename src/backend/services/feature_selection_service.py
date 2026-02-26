"""Reusable feature selection utilities."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_regression,
    mutual_info_regression,
)
from sklearn.linear_model import Ridge


def _mutual_info_score_func(X, y, random_state: int = 0):
    """Score function wrapper for mutual information regression."""
    return mutual_info_regression(X, y, random_state=random_state)


@dataclass(frozen=True)
class FeatureSelectorDefinition:
    """Metadata required to build a selector instance."""

    key: str
    label: str
    defaults: dict[str, object]


class FeatureSelectionService:
    """Factory for feature selection methods used across the application."""

    def __init__(self) -> None:
        self._selector_factories: dict[
            str, tuple[str, Callable[[dict[str, object]], Optional[object]], dict[str, object]]
        ] = {
            "none": (
                "No Feature Selection",
                lambda params: None,
                {},
            ),
            "variance_threshold": (
                "Low Variance Filter",
                lambda params: VarianceThreshold(**params),
                {"threshold": 0.0},
            ),
            "select_k_best": (
                "K Best (F-Score)",
                lambda params: SelectKBest(score_func=f_regression, **params),
                {"k": "all"},
            ),
            "mutual_info": (
                "K Best (Mutual Information)",
                lambda params: SelectKBest(
                    score_func=partial(_mutual_info_score_func, random_state=params.get("random_state", 0)),
                    k=params.get("k", "all"),
                ),
                {"k": "all", "random_state": 0},
            ),
            "random_forest_importance": (
                "Random Forest Importance",
                lambda params: SelectFromModel(
                    RandomForestRegressor(
                        random_state=params.get("random_state", 0),
                        n_estimators=params.get("n_estimators", 100),
                        n_jobs=-1,
                    ),
                    **{k: v for k, v in params.items() if k not in {"random_state", "n_estimators"}},
                ),
                {"threshold": "median", "random_state": 0, "n_estimators": 100},
            ),
            "extra_trees_importance": (
                "Extra Trees Importance",
                lambda params: SelectFromModel(
                    ExtraTreesRegressor(
                        random_state=params.get("random_state", 0),
                        n_estimators=params.get("n_estimators", 200),
                        n_jobs=-1,
                    ),
                    **{k: v for k, v in params.items() if k not in {"random_state", "n_estimators"}},
                ),
                {"threshold": "median", "random_state": 0, "n_estimators": 200},
            ),
            "gradient_boosting_importance": (
                "Gradient Boosting Importance",
                lambda params: SelectFromModel(
                    GradientBoostingRegressor(
                        random_state=params.get("random_state", 0),
                        n_estimators=params.get("n_estimators", 100),
                    ),
                    **{k: v for k, v in params.items() if k not in {"random_state", "n_estimators"}},
                ),
                {"threshold": "median", "random_state": 0, "n_estimators": 100},
            ),
            "rfe_ridge": (
                "Recursive Feature Elimination (Ridge)",
                lambda params: RFE(
                    estimator=Ridge(random_state=params.get("random_state", 0)),
                    n_features_to_select=params.get("n_features_to_select"),
                    step=params.get("step", 1),
                ),
                {"n_features_to_select": None, "step": 1, "random_state": 0},
            ),
        }

    # ------------------------------------------------------------------
    def available_feature_selectors(self) -> list[tuple[str, str, dict[str, object]]]:
        """Return available selectors and their default parameters."""

        return [
            (key, label, defaults.copy())
            for key, (label, _factory, defaults) in self._selector_factories.items()
        ]

    def definition_for(self, key: str) -> FeatureSelectorDefinition:
        """Return selector metadata for UI representation."""

        if key not in self._selector_factories:
            raise KeyError(f"Unknown feature selector '{key}'")
        label, _factory, defaults = self._selector_factories[key]
        return FeatureSelectorDefinition(key=key, label=label, defaults=defaults.copy())

    def build(self, key: str, overrides: Optional[dict[str, object]] = None) -> tuple[str, Optional[object]]:
        """Instantiate the selector for the provided key."""

        if key not in self._selector_factories:
            raise KeyError(f"Unknown feature selector '{key}'")
        label, factory, defaults = self._selector_factories[key]
        params = defaults.copy()
        if overrides:
            params.update(overrides)
        selector = factory(params)
        return label, selector

