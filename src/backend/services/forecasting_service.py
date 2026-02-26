"""Forecasting utilities built on scikit-learn regressors."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Callable, Mapping, Optional, Sequence

import threading

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_selection_service import FeatureSelectionService
from .modeling_shared import display_name, normalize_preprocessed_frame, prepare_wide_frame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastRunResult:
    key: str
    model_key: str
    model_label: str
    feature_key: str
    feature_label: str
    metrics: dict[str, float]
    progress_frame: pd.DataFrame
    forecast_frame: pd.DataFrame
    model_id: Optional[int] = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ForecastSummary:
    runs: list[ForecastRunResult] = field(default_factory=list)


class ForecastingService:
    """Build and execute forecasting pipelines from lagged-regression design matrices."""

    def __init__(self, database, feature_selection_service: Optional[FeatureSelectionService] = None):
        self._db = database
        self._feature_selection_service = feature_selection_service or FeatureSelectionService()
        self._default_selector = "rfe_ridge"
        self._selector_fallback = "none"
        self._model_factories: dict[str, tuple[str, Callable[[dict[str, object]], RegressorMixin], dict[str, object]]] = {
            "linear_regression": (
                "Linear Regression",
                lambda params: LinearRegression(fit_intercept=params.get("fit_intercept", True)),
                {"fit_intercept": True},
            ),
            "ridge": (
                "Ridge Regression",
                lambda params: Ridge(
                    alpha=params.get("alpha", 1.0),
                    solver=params.get("solver", "auto"),
                    random_state=params.get("random_state", 0),
                ),
                {"alpha": 1.0, "solver": "auto", "random_state": 0},
            ),
            "lasso": (
                "Lasso Regression",
                lambda params: Lasso(
                    alpha=params.get("alpha", 0.1),
                    max_iter=params.get("max_iter", 1000),
                    random_state=params.get("random_state", 0),
                ),
                {"alpha": 0.1, "max_iter": 1000, "random_state": 0},
            ),
            "random_forest": (
                "Random Forest",
                lambda params: RandomForestRegressor(
                    n_estimators=params.get("n_estimators", 200),
                    max_depth=params.get("max_depth"),
                    min_samples_split=params.get("min_samples_split", 2),
                    min_samples_leaf=params.get("min_samples_leaf", 1),
                    random_state=params.get("random_state", 0),
                    n_jobs=-1,
                ),
                {
                    "n_estimators": 200,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 0,
                },
            ),
            "gradient_boosting": (
                "Gradient Boosting",
                lambda params: GradientBoostingRegressor(
                    n_estimators=params.get("n_estimators", 100),
                    learning_rate=params.get("learning_rate", 0.1),
                    max_depth=params.get("max_depth", 3),
                    random_state=params.get("random_state", 0),
                ),
                {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 0},
            ),
        }

    def available_models(self) -> list[tuple[str, str, dict[str, object]]]:
        return [(key, label, defaults.copy()) for key, (label, _factory, defaults) in self._model_factories.items()]

    def run_forecasts(
        self,
        *,
        features: Sequence[Mapping[str, object]],
        models: Sequence[str],
        systems: Optional[Sequence[str]] = None,
        Datasets: Optional[Sequence[str]] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        model_params: Optional[Mapping[str, Mapping[str, object]]] = None,
        forecast_horizon: int = 12,
        test_size: Optional[float] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        result_callback: Optional[Callable[[ForecastRunResult], None]] = None,
        window_callback: Optional[Callable[[dict], None]] = None,
        data_frame: Optional[pd.DataFrame] = None,
        stop_event: Optional[threading.Event] = None,
        window_strategy: str = "single",
        initial_window: Optional[int] = None,
        target_feature: Optional[Mapping[str, object]] = None,
    ) -> ForecastSummary:
        del test_size, window_callback, initial_window
        if not features:
            raise ValueError("Select at least one feature to forecast")
        model_keys = [m for m in models if m in self._model_factories]
        if not model_keys:
            raise ValueError("No valid forecasting models selected")
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive")

        def _check_cancel() -> None:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Forecast run cancelled")

        def _emit_progress(pct: int) -> None:
            if progress_callback:
                progress_callback(int(max(0, min(100, pct))))

        def _emit_status(msg: str) -> None:
            if status_callback:
                status_callback(str(msg))

        model_params = model_params or {}
        _emit_progress(0)

        payloads = [dict(p) for p in features]
        combined_payloads = list(payloads)
        target_label: Optional[str] = None
        if target_feature:
            target_payload = dict(target_feature)
            target_label = display_name(target_payload)
            if target_payload not in combined_payloads:
                combined_payloads.append(target_payload)

        feature_ids = [int(p["feature_id"]) for p in combined_payloads if p.get("feature_id") is not None]
        if data_frame is not None:
            data = normalize_preprocessed_frame(data_frame, combined_payloads)
        else:
            df = self._db.query_raw(systems=systems, datasets=Datasets, feature_ids=feature_ids, start=start, end=end)
            if df is None or df.empty:
                _emit_status("No data available for forecasting.")
                return ForecastSummary()
            data = prepare_wide_frame(df, combined_payloads)

        if data is None or data.empty:
            _emit_status("No data available for forecasting.")
            return ForecastSummary()

        total_runs = len(payloads) * len(model_keys)
        done = 0
        progress_rows: list[dict[str, object]] = []
        runs: list[ForecastRunResult] = []

        for payload in payloads:
            _check_cancel()
            feature_label = display_name(payload)
            if feature_label not in data.columns:
                continue
            required_cols = [feature_label]
            if target_label and target_label in data.columns and target_label != feature_label:
                required_cols.append(target_label)

            ds = data[["t", *required_cols]].dropna().sort_values("t").reset_index(drop=True)
            if len(ds) <= forecast_horizon + 5:
                continue

            for model_key in model_keys:
                _check_cancel()
                model_label, model_factory, defaults = self._model_factories[model_key]
                cfg = defaults.copy()
                cfg.update(model_params.get(model_key, {}))

                selector_key = str(cfg.pop("selector_key", self._default_selector))
                if selector_key not in {k for k, _, _ in self._feature_selection_service.available_feature_selectors()}:
                    selector_key = self._selector_fallback

                if window_strategy == "single":
                    metrics, frame = self._fit_single_split(ds, feature_label, target_label, model_factory, cfg, selector_key, forecast_horizon)
                else:
                    metrics, frame = self._fit_window_backtest(ds, feature_label, target_label, model_factory, cfg, selector_key, forecast_horizon, window_strategy)

                done += 1
                progress_rows.append({
                    "t": pd.Timestamp.utcnow(),
                    "model": model_label,
                    "feature": feature_label,
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                    "r2": -abs(metrics.get("rmse") or 0.0),
                    "step": len(progress_rows) + 1,
                })
                run = ForecastRunResult(
                    key=f"{model_key}:{payload.get('feature_id', feature_label)}:{int(pd.Timestamp.utcnow().timestamp()*1000)}:{done}",
                    model_key=model_key,
                    model_label=model_label,
                    feature_key=str(payload.get("feature_id", feature_label)),
                    feature_label=feature_label,
                    metrics=metrics,
                    progress_frame=pd.DataFrame(progress_rows),
                    forecast_frame=frame,
                    metadata={"selector_key": selector_key},
                )
                runs.append(run)
                if result_callback:
                    result_callback(run)
                _emit_progress(int(round((done / max(1, total_runs)) * 100)))

        _emit_status("Forecast experiments completed.")
        _emit_progress(100)
        return ForecastSummary(runs=runs)

    def _build_lagged_dataset(
        self,
        frame: pd.DataFrame,
        value_col: str,
        target_col: Optional[str],
        window_length: int,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        work = frame.copy().sort_values("t").reset_index(drop=True)
        for i in range(1, window_length + 1):
            work[f"lag_{i}"] = work[value_col].shift(i)
        if target_col and target_col in work.columns and target_col != value_col:
            work["exog"] = work[target_col]
        work = work.dropna().reset_index(drop=True)
        feature_cols = [c for c in work.columns if c.startswith("lag_") or c == "exog"]
        return work[feature_cols], work[value_col], work["t"]

    def _build_pipeline(self, model_factory, model_cfg: Mapping[str, object], selector_key: str) -> Pipeline:
        model = model_factory(dict(model_cfg))
        selector_label, selector = self._feature_selection_service.build(selector_key, None)
        logger.debug("Forecast selector in use: %s", selector_label)
        steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
        if selector is not None:
            steps.append(("selector", selector))
        steps.append(("regressor", model))
        return Pipeline(steps)

    def _fit_single_split(self, ds, feature_label, target_label, model_factory, cfg, selector_key, forecast_horizon):
        window_length = int(cfg.pop("window_length", 12))
        X, y, t = self._build_lagged_dataset(ds, feature_label, target_label, window_length)
        if len(X) <= forecast_horizon:
            raise ValueError("Not enough samples for selected horizon")
        train_end = len(X) - forecast_horizon
        X_train, y_train, t_train = X.iloc[:train_end], y.iloc[:train_end], t.iloc[:train_end]
        X_test, y_test, t_test = X.iloc[train_end:], y.iloc[train_end:], t.iloc[train_end:]
        pipe = self._build_pipeline(model_factory, cfg, selector_key)
        pipe.fit(X_train, y_train)
        y_pred_test = pd.Series(np.asarray(pipe.predict(X_test), dtype=float), index=t_test)
        metrics = self._compute_metrics(y_test.set_axis(t_test), y_pred_test)

        pipe_all = self._build_pipeline(model_factory, cfg, selector_key)
        pipe_all.fit(X, y)
        future_X, future_t = self._recursive_future_features(ds, feature_label, target_label, window_length, forecast_horizon)
        y_future = pd.Series(np.asarray(pipe_all.predict(future_X), dtype=float), index=future_t)
        return metrics, self._build_forecast_frame(t_train, y_train, t_test, y_test, y_pred_test, y_future)

    def _fit_window_backtest(self, ds, feature_label, target_label, model_factory, cfg, selector_key, forecast_horizon, strategy):
        window_length = int(cfg.pop("window_length", 12))
        X, y, t = self._build_lagged_dataset(ds, feature_label, target_label, window_length)
        min_train = max(window_length * 2, forecast_horizon + 5)
        preds: list[pd.Series] = []
        trues: list[pd.Series] = []
        step = max(1, forecast_horizon)
        for train_end in range(min_train, len(X) - forecast_horizon + 1, step):
            train_start = 0 if strategy == "expanding" else max(0, train_end - min_train)
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[train_end:train_end + forecast_horizon]
            y_test = y.iloc[train_end:train_end + forecast_horizon]
            t_test = t.iloc[train_end:train_end + forecast_horizon]
            if X_test.empty:
                continue
            pipe = self._build_pipeline(model_factory, cfg, selector_key)
            pipe.fit(X_train, y_train)
            preds.append(pd.Series(np.asarray(pipe.predict(X_test), dtype=float), index=t_test))
            trues.append(pd.Series(np.asarray(y_test, dtype=float), index=t_test))

        if not preds:
            return self._fit_single_split(ds, feature_label, target_label, model_factory, cfg, selector_key, forecast_horizon)

        y_pred_cv = pd.concat(preds).groupby(level=0).mean().sort_index()
        y_true_cv = pd.concat(trues).groupby(level=0).mean().sort_index()
        metrics = self._compute_metrics(y_true_cv, y_pred_cv)

        pipe_all = self._build_pipeline(model_factory, cfg, selector_key)
        pipe_all.fit(X, y)
        future_X, future_t = self._recursive_future_features(ds, feature_label, target_label, window_length, forecast_horizon)
        y_future = pd.Series(np.asarray(pipe_all.predict(future_X), dtype=float), index=future_t)

        train_mask = t < y_true_cv.index.min()
        return metrics, self._build_forecast_frame(
            t[train_mask], y[train_mask], y_true_cv.index, y_true_cv.values, y_pred_cv, y_future
        )

    def _recursive_future_features(self, ds, value_col, target_col, window_length, horizon):
        values = ds[value_col].astype(float).to_list()
        times = pd.to_datetime(ds["t"]).tolist()
        freq = pd.infer_freq(pd.DatetimeIndex(times)) or "h"
        exog_value = float(ds[target_col].dropna().iloc[-1]) if target_col and target_col in ds.columns else None

        rows = []
        future_times = []
        for i in range(horizon):
            lag_values = [values[-j] for j in range(1, window_length + 1)]
            row = {f"lag_{j}": lag_values[j - 1] for j in range(1, window_length + 1)}
            if exog_value is not None:
                row["exog"] = exog_value
            rows.append(row)
            next_t = pd.date_range(start=times[-1], periods=2, freq=freq)[1]
            future_times.append(next_t)
            times.append(next_t)
            values.append(values[-1])
        return pd.DataFrame(rows), pd.DatetimeIndex(future_times)

    def _compute_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        aligned_true = pd.Series(y_true).astype(float)
        aligned_pred = pd.Series(y_pred).astype(float).reindex(aligned_true.index)
        mask = ~(aligned_true.isna() | aligned_pred.isna())
        yt = aligned_true[mask]
        yp = aligned_pred[mask]
        if yt.empty:
            return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}
        mse = float(mean_squared_error(yt, yp))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(yt, yp))
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = float(np.nanmean(np.abs((yt - yp) / yt.replace(0, np.nan))) * 100)
        return {"rmse": rmse, "mae": mae, "mape": mape}

    def _build_forecast_frame(self, t_train, y_train, t_test, y_test, y_pred_test, y_pred_future):
        train_frame = pd.DataFrame({"t": pd.to_datetime(t_train), "Actual": np.asarray(y_train, dtype=float), "Period": "Training"})
        test_frame = pd.DataFrame({
            "t": pd.to_datetime(t_test),
            "Actual": np.asarray(y_test, dtype=float),
            "Forecast": np.asarray(pd.Series(y_pred_test), dtype=float),
            "Period": "Testing",
        })
        future_frame = pd.DataFrame({"t": pd.to_datetime(y_pred_future.index), "Forecast": np.asarray(y_pred_future, dtype=float), "Period": "Forecast"})
        return pd.concat([train_frame, test_frame, future_frame], ignore_index=True).dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

