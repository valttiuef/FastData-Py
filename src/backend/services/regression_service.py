from __future__ import annotations
"""Utilities for building and training regression pipelines."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional, Sequence
import warnings

import threading

import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
    cross_val_predict,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .dimensionality_reduction_service import DimensionalityReductionService
from .feature_selection_service import FeatureSelectionService
from .modeling_shared import (
    display_name,
    normalize_preprocessed_frame,
    parse_hidden_layer_sizes,
    prepare_wide_frame,
)


# Pipeline step name constants for polynomial regression
_POLY_FEATURES_STEP = "polynomial_features"
_LINEAR_REGRESSOR_STEP = "linear_regressor"


@dataclass(frozen=True)
class RegressionRunResult:
    """Container for a single regression pipeline execution."""

    key: str
    model_key: str
    model_label: str
    selector_key: str
    selector_label: str
    reducer_key: str
    reducer_label: str
    metrics: dict[str, float]
    cv_scores: dict[str, list[float]]
    progress_frame: pd.DataFrame
    timeline_frame: pd.DataFrame
    scatter_frame: pd.DataFrame
    trained_at: str = ""
    model_id: Optional[int] = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RegressionSummary:
    """Aggregate of regression runs for UI consumption."""

    runs: list[RegressionRunResult] = field(default_factory=list)


def _build_polynomial_regressor(params: dict[str, object]) -> Pipeline:
    """Build a polynomial regression pipeline with polynomial features and linear regression."""
    return Pipeline([
        (_POLY_FEATURES_STEP, PolynomialFeatures(degree=params.get("degree", 2), include_bias=False)),
        (_LINEAR_REGRESSOR_STEP, LinearRegression(fit_intercept=params.get("fit_intercept", True))),
    ])


class RegressionService:
    """Build and execute regression pipelines over measurement data."""

    def __init__(
        self,
        database,
        feature_selection_service: Optional[FeatureSelectionService] = None,
        dimensionality_reduction_service: Optional[DimensionalityReductionService] = None,
    ):
        self._db = database
        self._feature_selection_service = feature_selection_service or FeatureSelectionService()
        self._dimensionality_reduction_service = (
            dimensionality_reduction_service or DimensionalityReductionService()
        )

        self._model_factories: dict[str, tuple[str, Callable[[dict[str, object]], RegressorMixin], dict[str, object]]] = {
            "linear_regression": (
                "Linear Regression",
                lambda params: LinearRegression(**params),
                {"fit_intercept": True, "positive": False},
            ),
            "ridge": (
                "Ridge Regression",
                lambda params: Ridge(**params),
                {"alpha": 1.0, "solver": "auto", "random_state": 0},
            ),
            "lasso": (
                "Lasso Regression",
                lambda params: Lasso(**params),
                {"alpha": 1.0, "max_iter": 1000, "random_state": 0},
            ),
            "elastic_net": (
                "Elastic Net",
                lambda params: ElasticNet(**params),
                {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 1000, "random_state": 0},
            ),
            "polynomial_regression": (
                "Polynomial Regression",
                _build_polynomial_regressor,
                {"degree": 2, "fit_intercept": True},
            ),
            "random_forest": (
                "Random Forest",
                lambda params: RandomForestRegressor(**params),
                {
                    "n_estimators": 200,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 0,
                    "n_jobs": -1,
                },
            ),
            "extra_trees": (
                "Extra Trees",
                lambda params: ExtraTreesRegressor(**params),
                {
                    "n_estimators": 300,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 0,
                    "n_jobs": -1,
                },
            ),
            "gradient_boosting": (
                "Gradient Boosting",
                lambda params: GradientBoostingRegressor(**params),
                {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 0,
                },
            ),
            "adaboost": (
                "AdaBoost",
                lambda params: AdaBoostRegressor(**params),
                {
                    "n_estimators": 50,
                    "learning_rate": 1.0,
                    "random_state": 0,
                },
            ),
            "svr": (
                "Support Vector Regression (SVR)",
                lambda params: SVR(**params),
                {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
            ),
            "knn": (
                "K-Nearest Neighbors",
                lambda params: KNeighborsRegressor(**params),
                {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto"},
            ),
            "decision_tree": (
                "Decision Tree",
                lambda params: DecisionTreeRegressor(**params),
                {
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 0,
                },
            ),
            "mlp": (
                "MLP Regressor",
                lambda params: MLPRegressor(
                    hidden_layer_sizes=parse_hidden_layer_sizes(
                        params.get("hidden_layer_sizes", (100,))
                    )
                    or (100,),
                    activation=params.get("activation", "relu"),
                    solver=params.get("solver", "adam"),
                    alpha=params.get("alpha", 0.0001),
                    learning_rate=params.get("learning_rate", "constant"),
                    max_iter=params.get("max_iter", 200),
                    random_state=params.get("random_state", 0),
                ),
                {
                    "hidden_layer_sizes": (100,),
                    "activation": "relu",
                    "solver": "adam",
                    "alpha": 0.0001,
                    "learning_rate": "constant",
                    "max_iter": 200,
                    "random_state": 0,
                },
            ),
        }
        self._cv_labels: dict[str, str] = {
            "none": "No cross-validation",
            "kfold": "K-Fold",
            "stratified_kfold": "Stratified K-Fold",
            "time_series": "Time series split",
            "group_kfold": "Group K-Fold",
        }

        self._test_split_labels: dict[str, str] = {
            "random": "Random",
            "time": "Time ordered",
            "stratified": "Stratified",
        }

    # ------------------------------------------------------------------
    def available_models(self) -> list[tuple[str, str, dict[str, object]]]:
        return [
            (key, label, defaults.copy())
            for key, (label, _factory, defaults) in self._model_factories.items()
        ]

    def available_feature_selectors(self) -> list[tuple[str, str, dict[str, object]]]:
        return self._feature_selection_service.available_feature_selectors()

    def available_dimensionality_reducers(self) -> list[tuple[str, str, dict[str, object]]]:
        return self._dimensionality_reduction_service.available_dimensionality_reducers()

    def available_cv_strategies(self) -> list[tuple[str, str]]:
        return list(self._cv_labels.items())

    def available_test_strategies(self) -> list[tuple[str, str]]:
        return list(self._test_split_labels.items())

    # ------------------------------------------------------------------
    def run_regressions(
        self,
        *,
        input_features: Sequence[Mapping[str, object]],
        target_feature: Mapping[str, object],
        selectors: Sequence[str],
        models: Sequence[str],
        systems: Optional[Sequence[str]] = None,
        Datasets: Optional[Sequence[str]] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        selector_params: Optional[Mapping[str, Mapping[str, object]]] = None,
        model_params: Optional[Mapping[str, Mapping[str, object]]] = None,
        reducers: Optional[Sequence[str]] = None,
        reducer_params: Optional[Mapping[str, Mapping[str, object]]] = None,
        cv_strategy: str = "none",
        cv_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 0,
        test_size: Optional[float] = 0.2,
        test_strategy: str = "random",
        stratify_bins: int = 5,
        time_series_gap: int = 0,
        cv_group_kind: Optional[str] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        result_callback: Optional[Callable[[RegressionRunResult], None]] = None,
        data_frame: Optional[pd.DataFrame] = None,
        stratify_feature: Optional[Mapping[str, object]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> RegressionSummary:
        if not input_features:
            raise ValueError("Select at least one input feature")
        if not target_feature:
            raise ValueError("Select a target feature")
        if not selectors:
            selectors = ["none"]
        if not models:
            raise ValueError("Select at least one regression model")

        selector_params = selector_params or {}
        model_params = model_params or {}
        reducer_params = reducer_params or {}

        selector_keys: list[str] = []
        for key in selectors:
            try:
                self._feature_selection_service.definition_for(key)
            except KeyError:
                continue
            selector_keys.append(key)
        if not selector_keys:
            raise ValueError("No valid feature selection methods selected")
        model_keys = [m for m in models if m in self._model_factories]
        if not model_keys:
            raise ValueError("No valid regression models selected")
        reducer_keys: list[str] = []
        requested_reducers = list(reducers or ["none"])
        if not requested_reducers:
            requested_reducers = ["none"]
        for key in requested_reducers:
            if self._dimensionality_reduction_service.has_reducer(key):
                reducer_keys.append(key)
        if not reducer_keys:
            reducer_keys = ["none"]

        def _emit_progress(pct: int) -> None:
            if progress_callback is None:
                return
            progress_callback(int(max(0, min(100, pct))))

        def _emit_status(msg: str) -> None:
            if status_callback is None:
                return
            status_callback(str(msg))

        def _check_cancel() -> None:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Regression run cancelled")

        def _emit_result(run: RegressionRunResult) -> None:
            if result_callback is None:
                return
            result_callback(run)

        _emit_progress(0)
        _emit_status("Preparing regression dataset…")
        _check_cancel()

        input_payloads = [dict(p) for p in input_features]
        target_payload = dict(target_feature)
        stratify_payload = dict(stratify_feature) if stratify_feature else None

        expected_payloads: list[Mapping[str, object]] = [target_payload] + input_payloads
        if stratify_payload:
            expected_payloads.append(stratify_payload)

        combined_payloads: list[Mapping[str, object]] = []
        feature_ids: list[int] = []
        seen_ids: set[int] = set()
        for payload in expected_payloads:
            fid = payload.get("feature_id")
            if fid is None:
                continue
            try:
                fid_int = int(fid)
            except Exception as exc:
                raise ValueError("Invalid feature identifier in payload") from exc
            if fid_int in seen_ids:
                continue
            seen_ids.add(fid_int)
            feature_ids.append(fid_int)
            combined_payloads.append(payload)

        if data_frame is not None:
            _check_cancel()
            data = normalize_preprocessed_frame(data_frame, expected_payloads)
        else:
            if not feature_ids:
                raise ValueError("Missing feature identifiers for database query")
            df = self._db.query_raw(
                systems=systems,
                datasets=Datasets,
                feature_ids=feature_ids,
                start=start,
                end=end,
            )
            _check_cancel()
            if df is None or df.empty:
                _emit_status("No data available for regression.")
                return RegressionSummary()
            data = prepare_wide_frame(df, combined_payloads)

        _check_cancel()
        if data is None or data.empty:
            _emit_status("No data available for regression.")
            return RegressionSummary()

        _emit_status("Configuring regression splits…")

        target_name = display_name(target_payload)
        input_names = [display_name(p) for p in input_payloads]

        required_cols = ["t", target_name] + input_names
        stratify_label = display_name(stratify_payload) if stratify_payload else None
        if stratify_label and stratify_label not in required_cols:
            required_cols.append(stratify_label)

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing expected columns in dataset: {missing}")

        data = data.dropna(subset=required_cols)
        if data.empty:
            raise ValueError("No rows remain after dropping missing values")

        data = data.sort_values("t").reset_index(drop=True)

        X = data[input_names]
        y = data[target_name]

        stratify_series = self._prepare_stratify_series(data, stratify_payload, target_name, stratify_bins)
        group_series = None
        if (cv_strategy or "").lower() == "group_kfold":
            group_series = self._prepare_group_kfold_series(
                data,
                group_kind=cv_group_kind,
            )

        _check_cancel()

        X_train, X_test, y_train, y_test, strat_train, _ = self._train_test_split(
            X,
            y,
            test_size=test_size,
            strategy=test_strategy,
            stratify_series=stratify_series,
            stratify_bins=stratify_bins,
            shuffle=shuffle,
            random_state=random_state,
        )

        cv = self._build_cv(
            X_train,
            y_train,
            strategy=cv_strategy,
            folds=cv_folds,
            shuffle=shuffle,
            random_state=random_state,
            stratify_series=strat_train,
            stratify_bins=stratify_bins,
            gap=time_series_gap,
            group_series=group_series.reindex(X_train.index) if group_series is not None else None,
            status_callback=_emit_status,
        )

        _emit_status("Training regression models…")

        progress_rows: list[dict[str, object]] = []
        runs: list[RegressionRunResult] = []
        failures: list[str] = []

        total_runs = len(selector_keys) * len(reducer_keys) * len(model_keys)
        done = 0

        for selector_key in selector_keys:
            _check_cancel()
            overrides = selector_params.get(selector_key, {})
            selector_label, selector = self._feature_selection_service.build(
                selector_key, overrides if isinstance(overrides, Mapping) else None
            )

            for reducer_key in reducer_keys:
                _check_cancel()
                reducer_label, reducer_obj, reducer_cfg = self._dimensionality_reduction_service.build_reducer(
                    reducer_key,
                    reducer_params.get(reducer_key, {}),
                    n_features=max(1, int(X_train.shape[1])),
                    n_samples=max(1, int(len(X_train))),
                )
                effective_reducer_label = reducer_label if reducer_obj is not None else ""

                for model_key in model_keys:
                    _check_cancel()
                    model_label, model_factory, model_defaults = self._model_factories[model_key]
                    label_parts = [model_label]
                    if selector_label:
                        label_parts.append(selector_label)
                    if reducer_key != "none" and effective_reducer_label:
                        label_parts.append(effective_reducer_label)
                    current_label = " + ".join(label_parts)
                    _emit_status(f"Training {current_label}.")
                    try:
                        model_cfg = model_defaults.copy()
                        model_cfg.update(model_params.get(model_key, {}))

                        regressor = model_factory(model_cfg)

                        steps = [("scaler", StandardScaler())]
                        if selector is not None:
                            steps.append(("selector", selector))
                        if reducer_obj is not None:
                            steps.append(("reducer", reducer_obj))
                        steps.append(("regressor", regressor))

                        base_pipeline = Pipeline(steps)

                        cv_scores = self._cross_validate(clone(base_pipeline), X_train, y_train, cv)
                        _check_cancel()
                        cv_predictions = self._cross_val_predictions(clone(base_pipeline), X_train, y_train, cv)
                        cv_split_labels = self._cross_val_split_labels(X_train, y_train, cv)
                        _check_cancel()

                        pipeline = clone(base_pipeline)
                        pipeline.fit(X_train, y_train)
                        _check_cancel()

                        metrics = self._compute_metrics(pipeline, X_train, y_train, X_test, y_test)
                        inputs_selected = self._selected_input_count(pipeline, X_train)
                        selected_names = self._selected_input_names(pipeline, input_names)

                        if cv_predictions is not None:
                            train_pred_series = pd.Series(np.asarray(cv_predictions, dtype=float), index=X_train.index)
                        else:
                            train_pred_series = pd.Series(np.asarray(pipeline.predict(X_train), dtype=float), index=X_train.index)

                        test_pred_series: Optional[pd.Series] = None
                        if not X_test.empty:
                            test_pred_series = pd.Series(np.asarray(pipeline.predict(X_test), dtype=float), index=X_test.index)

                        overall_progress = {
                            "step": len(progress_rows) + 1,
                            "label": current_label,
                            "r2": metrics.get("r2_test", metrics.get("r2_train", np.nan)),
                            "rmse": metrics.get("rmse_test", metrics.get("rmse_train", np.nan)),
                        }
                        progress_rows.append(overall_progress)

                        timeline = self._build_timeline_frame(
                            data,
                            target_name,
                            train_pred_series,
                            test_pred_series,
                            train_split_labels=cv_split_labels,
                        )
                        scatter = self._build_scatter_frame(y_train, train_pred_series, y_test, test_pred_series)

                        trained_at = datetime.now(timezone.utc).isoformat()
                        run_key = (
                            f"{model_key}:{selector_key}:{reducer_key}:"
                            f"{int(pd.Timestamp.now('UTC').timestamp() * 1000)}:{done + 1}"
                        )
                        run = RegressionRunResult(
                            key=run_key,
                            model_key=model_key,
                            model_label=model_label,
                            selector_key=selector_key,
                            selector_label=selector_label,
                            reducer_key=reducer_key,
                            reducer_label="" if reducer_key == "none" else effective_reducer_label,
                            metrics=metrics,
                            cv_scores=cv_scores,
                            progress_frame=pd.DataFrame(progress_rows),
                            timeline_frame=timeline,
                            scatter_frame=scatter,
                            trained_at=trained_at,
                            metadata={
                                "rows_total": int(len(X_train) + len(X_test)),
                                "rows_train": int(len(X_train)),
                                "rows_test": int(len(X_test)),
                                "inputs_selected": int(inputs_selected),
                                "inputs_total": int(len(input_names)),
                                "inputs_selected_names": list(selected_names),
                            },
                        )
                        runs.append(run)
                        _emit_result(run)
                    except Exception as exc:
                        failures.append(f"{current_label}: {exc}")
                        _emit_status(f"Failed {current_label}: {exc}")
                    finally:
                        done += 1
                        if total_runs:
                            pct = int(round((done / total_runs) * 100))
                            _emit_progress(pct)

                        if total_runs:
                            status_msg = f"Completed {current_label} ({done}/{total_runs})"
                        else:
                            status_msg = f"Completed {current_label}"
                        _emit_status(status_msg)

        if done >= total_runs:
            _emit_progress(100)
        if total_runs > 0 and not runs and failures:
            preview = "; ".join(failures[:3])
            raise RuntimeError(f"All regression pipelines failed. {preview}")
        _emit_status("Regression experiments completed.")
        return RegressionSummary(runs=runs)

    def _train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        test_size: Optional[float | int],
        strategy: str,
        stratify_series: Optional[pd.Series],
        stratify_bins: int,
        shuffle: bool,
        random_state: int,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        Optional[pd.Series],
        Optional[pd.Series],
    ]:
        if not test_size or test_size <= 0:
            strat_train = stratify_series.copy() if stratify_series is not None else None
            empty_df = pd.DataFrame(columns=X.columns)
            empty_series = pd.Series(dtype=y.dtype)
            return X, empty_df, y, empty_series, strat_train, None

        strategy = (strategy or "random").lower()
        indices = np.arange(len(X))

        if strategy == "time":
            if isinstance(test_size, (int, np.integer)) and test_size >= 1:
                test_count = min(int(test_size), max(1, len(X) - 1))
                split_index = max(1, len(X) - test_count)
            else:
                split_index = int(round(len(X) * (1 - float(test_size))))
                split_index = max(1, min(len(X) - 1, split_index))
            train_idx = np.arange(split_index)
            test_idx = np.arange(split_index, len(X))
        else:
            stratify_values = None
            if strategy == "stratified" and stratify_series is not None:
                stratify_values = stratify_series.to_numpy()
            if isinstance(test_size, (int, np.integer)) and test_size >= 1:
                test_size = min(int(test_size), max(1, len(indices) - 1))
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                shuffle=shuffle,
                random_state=random_state,
                stratify=stratify_values,
            )
            train_idx = np.sort(train_idx)
            test_idx = np.sort(test_idx)

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy() if len(test_idx) else pd.DataFrame(columns=X.columns)
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy() if len(test_idx) else pd.Series(dtype=y.dtype)

        strat_train: Optional[pd.Series] = None
        strat_test: Optional[pd.Series] = None
        if stratify_series is not None:
            strat_train = stratify_series.iloc[train_idx].copy()
            strat_test = stratify_series.iloc[test_idx].copy() if len(test_idx) else None

        return X_train, X_test, y_train, y_test, strat_train, strat_test

    def _bin_target(self, y: pd.Series, bins: int) -> Optional[np.ndarray]:
        if bins <= 1:
            return None
        try:
            categories = pd.qcut(y, q=bins, duplicates="drop")
        except Exception:
            return None
        return categories.to_numpy()

    def _prepare_stratify_series(
        self,
        data: pd.DataFrame,
        stratify_payload: Optional[Mapping[str, object]],
        target_label: str,
        bins: int,
    ) -> Optional[pd.Series]:
        if stratify_payload is not None:
            if isinstance(stratify_payload, Mapping) and stratify_payload.get("group_kind"):
                return self._prepare_group_kfold_series(
                    data,
                    group_kind=str(stratify_payload.get("group_kind")),
                )
            column = display_name(stratify_payload)
            if column in data.columns:
                series = data[column]
            else:
                series = data[target_label]
        else:
            series = data[target_label]

        if pd.api.types.is_numeric_dtype(series):
            binned = self._bin_target(pd.Series(series), bins)
            if binned is None:
                return None
            return pd.Series(binned, index=series.index)

        return pd.Series(series, index=series.index).astype(str)

    def _build_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        strategy: str,
        folds: int,
        shuffle: bool,
        random_state: int,
        stratify_series: Optional[pd.Series],
        stratify_bins: int,
        gap: int,
        group_series: Optional[pd.Series] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        strategy = (strategy or "none").lower()
        if strategy == "none" or folds <= 1 or len(y) < folds:
            return None
        if strategy == "kfold":
            return KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
        if strategy == "group_kfold":
            if group_series is None:
                if status_callback is not None:
                    status_callback("Group K-Fold requires a valid group; falling back to K-Fold.")
                return KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
            groups = pd.Series(group_series, index=X.index).astype(str)
            if groups is None or groups.dropna().empty:
                if status_callback is not None:
                    status_callback("Group K-Fold has no group assignments; falling back to K-Fold.")
                return KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
            unique_groups = pd.Series(groups).dropna().unique()
            if len(unique_groups) < folds:
                if status_callback is not None:
                    status_callback("Not enough groups for Group K-Fold; falling back to K-Fold.")
                return KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
            splitter = GroupKFold(n_splits=folds)
            return list(splitter.split(X, y, groups=groups))
        if strategy == "stratified_kfold":
            labels = self._prepare_stratify_labels(stratify_series, y, stratify_bins)
            if labels is None:
                return KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
            splitter = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
            return list(splitter.split(np.zeros(len(y)), labels))
        if strategy == "time_series":
            splitter = TimeSeriesSplit(n_splits=folds, gap=max(0, int(gap)))
            return splitter
        return None

    def _prepare_group_kfold_series(
        self,
        data: pd.DataFrame,
        *,
        group_kind: Optional[str],
    ) -> Optional[pd.Series]:
        if data is None or data.empty or "t" not in data.columns:
            return None
        if not group_kind:
            return None
        group_kind = str(group_kind).strip()
        if not group_kind:
            return None
        labels_df = self._db.list_group_labels(kind=group_kind)
        if labels_df is None or labels_df.empty:
            return None
        group_ids = labels_df["group_id"].tolist()
        if not group_ids:
            return None
        start_ts = data["t"].min()
        end_ts = data["t"].max()
        group_points = self._db.group_points(group_ids, start=start_ts, end=end_ts)
        if group_points is None or group_points.empty:
            return None

        label_map = dict(zip(labels_df["group_id"], labels_df["label"]))
        group_points = group_points.copy()
        group_points["group_label"] = group_points["group_id"].map(label_map)
        group_points["start_ts"] = pd.to_datetime(group_points["start_ts"], errors="coerce")
        group_points["end_ts"] = pd.to_datetime(group_points["end_ts"], errors="coerce")
        group_points = group_points.dropna(subset=["start_ts", "end_ts", "group_label"])
        group_points = group_points[group_points["end_ts"] >= group_points["start_ts"]]
        if group_points.empty:
            return None

        df = data[["t"]].copy()
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df = df.dropna(subset=["t"])
        if df.empty:
            return None

        df = df.sort_values("t")
        gp = group_points.sort_values(["start_ts", "end_ts"]).reset_index(drop=True)
        starts = gp["start_ts"].to_numpy(dtype="datetime64[ns]")
        ends = gp["end_ts"].to_numpy(dtype="datetime64[ns]")
        labels = gp["group_label"].astype(str).to_numpy(dtype=object)
        values = df["t"].to_numpy(dtype="datetime64[ns]")
        idx = np.searchsorted(starts, values, side="right") - 1
        assigned: list[object] = [np.nan] * len(df)
        for i, ridx in enumerate(idx):
            if ridx < 0:
                continue
            value_ts = values[i]
            if np.isnat(value_ts) or np.isnat(ends[ridx]) or value_ts > ends[ridx]:
                continue
            assigned[i] = labels[ridx]
        series = pd.Series(assigned, index=df.index)
        return series.reindex(data.index)

    def _selected_input_count(self, pipeline: Pipeline, X_train: pd.DataFrame) -> int:
        selector = pipeline.named_steps.get("selector")
        if selector is None:
            return int(X_train.shape[1])
        if hasattr(selector, "get_support"):
            support = selector.get_support()
            if support is not None:
                return int(np.sum(support))
        return int(getattr(selector, "n_features_in_", X_train.shape[1]))

    def _prepare_stratify_labels(
        self,
        stratify_series: Optional[pd.Series],
        y: pd.Series,
        bins: int,
    ) -> Optional[np.ndarray]:
        if stratify_series is not None:
            series = stratify_series
        else:
            series = y
        if series is None or len(series) == 0:
            return None
        if pd.api.types.is_numeric_dtype(series):
            return self._bin_target(pd.Series(series), bins)
        return pd.Series(series).astype(str).to_numpy()

    def _cross_validate(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv) -> dict[str, list[float]]:
        if cv is None:
            return {}
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*sklearn\.utils\.parallel\.delayed.*",
                category=UserWarning,
                module=r"sklearn\.utils\.parallel",
            )
            result = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
                error_score="raise",
            )
        scores = {
            "r2": list(result.get("test_r2", [])),
            "rmse": [abs(v) for v in result.get("test_rmse", [])],
        }
        return scores

    def _cross_val_predictions(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv):
        if cv is None:
            return None
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*sklearn\.utils\.parallel\.delayed.*",
                category=UserWarning,
                module=r"sklearn\.utils\.parallel",
            )
            return cross_val_predict(pipeline, X, y, cv=cv)

    def _selected_input_names(self, pipeline: Pipeline, input_names: list[str]) -> list[str]:
        selector = pipeline.named_steps.get("selector")
        if selector is None:
            return list(input_names)
        if hasattr(selector, "get_support"):
            support = selector.get_support()
            if support is not None and len(support) == len(input_names):
                return [name for name, keep in zip(input_names, support) if bool(keep)]
        return list(input_names)

    def _cross_val_split_labels(self, X: pd.DataFrame, y: pd.Series, cv) -> Optional[pd.Series]:
        if cv is None:
            return None
        labels = pd.Series(index=X.index, dtype=object)
        if hasattr(cv, "split"):
            split_iter = cv.split(X, y)
        else:
            split_iter = iter(cv)
        for fold_idx, (_train_idx, test_idx) in enumerate(split_iter, start=1):
            labels.iloc[np.asarray(test_idx, dtype=int)] = f"fold {fold_idx}"
        if labels.notna().sum() == 0:
            return None
        return labels

    def _compute_metrics(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        train_pred = pipeline.predict(X_train)
        metrics["r2_train"] = float(r2_score(y_train, train_pred))
        metrics["rmse_train"] = float(np.sqrt(mean_squared_error(y_train, train_pred)))

        if not X_test.empty and not y_test.empty:
            test_pred = pipeline.predict(X_test)
            metrics["r2_test"] = float(r2_score(y_test, test_pred))
            metrics["rmse_test"] = float(np.sqrt(mean_squared_error(y_test, test_pred)))
        return metrics

    def _build_timeline_frame(
        self,
        data: pd.DataFrame,
        target_label: str,
        train_predictions: pd.Series,
        test_predictions: Optional[pd.Series],
        train_split_labels: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        timeline = data[["t", target_label]].copy()
        actual_label = "Actual"
        timeline = timeline.rename(columns={target_label: actual_label})

        cv_label = "Prediction (train)"
        split_label = "Split (train)"
        test_label = "Prediction (test)"

        timeline[cv_label] = np.nan
        if train_predictions is not None:
            timeline.loc[train_predictions.index, cv_label] = train_predictions.values
            split_series = pd.Series([None] * len(timeline), index=timeline.index, dtype=object)
            if train_split_labels is not None:
                aligned = train_split_labels.reindex(train_predictions.index)
                split_series.loc[train_predictions.index] = aligned.values
            else:
                split_series.loc[train_predictions.index] = "train"
            timeline[split_label] = split_series

            split_values = (
                pd.Series(split_series.loc[train_predictions.index], dtype=object)
                .dropna()
                .astype(str)
                .str.strip()
            )
            split_values = split_values[split_values != ""]
            split_names = list(dict.fromkeys(split_values.tolist()))
            split_names = sorted(
                split_names,
                key=lambda name: (
                    0,
                    int(name.split(" ", 1)[1]),
                )
                if str(name).lower().startswith("fold ")
                and str(name).split(" ", 1)[1].isdigit()
                else (1, str(name)),
            )

            # When CV is used, expose split-specific train series for charting.
            # Without CV splits, keep only "Prediction (train)".
            if len(split_names) > 1:
                for split_name in split_names:
                    col = f"Prediction (train split {split_name})"
                    timeline[col] = np.nan
                    member_index = split_series[
                        split_series.astype(str).str.strip() == split_name
                    ].index
                    if len(member_index):
                        timeline.loc[member_index, col] = train_predictions.reindex(member_index).values
                timeline = timeline.drop(columns=[cv_label], errors="ignore")

        if test_predictions is not None:
            timeline[test_label] = np.nan
            timeline.loc[test_predictions.index, test_label] = test_predictions.values
        elif test_label in timeline.columns:
            timeline = timeline.drop(columns=[test_label])

        if cv_label in timeline.columns and timeline[cv_label].notna().sum() == 0:
            timeline = timeline.drop(columns=[cv_label])
        if split_label in timeline.columns and timeline[split_label].notna().sum() == 0:
            timeline = timeline.drop(columns=[split_label])
        if test_label in timeline.columns and timeline[test_label].notna().sum() == 0:
            timeline = timeline.drop(columns=[test_label])

        return timeline

    def _build_scatter_frame(
        self,
        y_train: pd.Series,
        train_predictions: Optional[pd.Series],
        y_test: pd.Series,
        test_predictions: Optional[pd.Series],
    ) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        if train_predictions is not None:
            train_df = pd.concat([y_train, train_predictions], axis=1, join="inner")
            train_df.columns = ["actual", "predicted"]
            train_df["dataset"] = "Train"
            parts.append(train_df)
        if test_predictions is not None and not test_predictions.empty:
            test_df = pd.concat([y_test, test_predictions], axis=1, join="inner")
            test_df.columns = ["actual", "predicted"]
            test_df["dataset"] = "Test"
            parts.append(test_df)
        if not parts:
            return pd.DataFrame(columns=["actual", "predicted", "dataset"])
        scatter = pd.concat(parts, axis=0, ignore_index=True)
        scatter = scatter.dropna()
        return scatter

