from __future__ import annotations
"""Legacy forecasting utilities built on sktime forecasters (reference only)."""

from dataclasses import dataclass, field
import importlib.util
import logging
import re
from typing import Callable, Mapping, Optional, Sequence

import threading

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

logger = logging.getLogger(__name__)

try:
    from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.theta import ThetaForecaster
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    from sktime.split import SlidingWindowSplitter, ExpandingWindowSplitter

    SKTIME_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    BaseForecaster = object  # type: ignore[misc,assignment]
    ForecastingHorizon = None  # type: ignore[assignment]
    NaiveForecaster = ThetaForecaster = ExponentialSmoothing = None  # type: ignore[assignment]
    SlidingWindowSplitter = ExpandingWindowSplitter = None  # type: ignore[assignment]
    SKTIME_AVAILABLE = False

# Optional advanced forecasters
try:
    from sktime.forecasting.arima import AutoARIMA
    AUTOARIMA_AVAILABLE = True
except ImportError:
    AutoARIMA = None  # type: ignore[assignment]
    AUTOARIMA_AVAILABLE = False

try:
    from sktime.forecasting.compose import TransformedTargetForecaster
    from sktime.transformations.series.boxcox import BoxCoxTransformer
    COMPOSED_AVAILABLE = True
except ImportError:
    TransformedTargetForecaster = BoxCoxTransformer = None  # type: ignore[assignment]
    COMPOSED_AVAILABLE = False

try:
    from sktime.forecasting.trend import PolynomialTrendForecaster
    TREND_AVAILABLE = True
except ImportError:
    PolynomialTrendForecaster = None  # type: ignore[assignment]
    TREND_AVAILABLE = False

try:
    from sktime.forecasting.compose import EnsembleForecaster
    ENSEMBLE_AVAILABLE = True
except ImportError:
    EnsembleForecaster = None  # type: ignore[assignment]
    ENSEMBLE_AVAILABLE = False

# BATS model
try:
    from sktime.forecasting.bats import BATS
    BATS_AVAILABLE = True
except ImportError:
    BATS = None  # type: ignore[assignment]
    BATS_AVAILABLE = False

# TBATS model
try:
    from sktime.forecasting.tbats import TBATS
    TBATS_AVAILABLE = True
except ImportError:
    TBATS = None  # type: ignore[assignment]
    TBATS_AVAILABLE = False

# Croston method for intermittent demand forecasting
try:
    from sktime.forecasting.croston import Croston
    CROSTON_AVAILABLE = True
except ImportError:
    Croston = None  # type: ignore[assignment]
    CROSTON_AVAILABLE = False

# Time series regression using sklearn regressors (make_reduction)
try:
    from sktime.forecasting.compose import make_reduction
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    REDUCTION_AVAILABLE = True
except ImportError:
    make_reduction = None  # type: ignore[assignment]
    Ridge = Lasso = RandomForestRegressor = GradientBoostingRegressor = None  # type: ignore[assignment]
    MLPRegressor = None  # type: ignore[assignment]
    REDUCTION_AVAILABLE = False

from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class ForecastRunResult:
    """Container for a single forecasting pipeline execution."""

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
    """Aggregate of forecasting runs for UI consumption."""

    runs: list[ForecastRunResult] = field(default_factory=list)


def _display_name(payload: Mapping[str, object]) -> str:
    parts: list[str] = []
    for key in ("label", "base_name", "source", "unit", "type"):
        val = payload.get(key)
        if val and str(val) not in parts:
            parts.append(str(val))
    if not parts:
        fid = payload.get("feature_id")
        return f"Feature {fid}" if fid is not None else "Feature"
    return " · ".join(parts)


def _parse_hidden_layer_sizes(value: object) -> Optional[tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, str):
        parts = [p for p in re.split(r"[,\\s]+", value.strip()) if p]
        if not parts:
            return None
        numbers: list[int] = []
        for part in parts:
            try:
                numbers.append(int(part))
            except ValueError:
                return None
        return tuple(numbers)
    return None


class ForecastingService:
    """Build and execute time-series forecasting pipelines."""

    def __init__(self, database):
        if not SKTIME_AVAILABLE:
            raise ImportError("sktime is required for forecasting features but is not installed")
        self._db = database

        # Build model factories dynamically, only including available forecasters
        statsmodels_available = importlib.util.find_spec("statsmodels") is not None

        def _fallback_forecaster(strategy: str = "last") -> BaseForecaster:
            class _LocalFallback(BaseForecaster):
                def __init__(self):
                    super().__init__()
                    self._strategy = strategy
                    self._last_value = 0.0
                    self._trend = 0.0
                    self._freq = None
                    self._last_index = None
                    self._tags.update({"requires-fh-in-fit": False})

                def _fit(self, y, X=None, fh=None):  # type: ignore[override]
                    self._last_value = float(y.iloc[-1]) if len(y) else 0.0
                    if len(y) > 1 and self._strategy == "drift":
                        try:
                            self._trend = float(y.iloc[-1] - y.iloc[0]) / max(1, len(y) - 1)
                        except Exception:
                            self._trend = 0.0
                    self._freq = y.index.freqstr or y.attrs.get("_inferred_freq")
                    self._last_index = y.index[-1] if len(y) else None
                    return self

                def _predict(self, fh=None, X=None):  # type: ignore[override]
                    horizon = len(fh) if fh is not None else 0
                    steps = np.arange(1, horizon + 1)
                    if self._strategy == "drift":
                        values = self._last_value + self._trend * steps
                    else:
                        values = np.full(horizon, self._last_value)
                    try:
                        freq = self._freq
                        if freq and self._last_index is not None:
                            index = pd.date_range(start=self._last_index, periods=horizon + 1, freq=freq)[1:]
                        else:
                            index = pd.RangeIndex(start=0, stop=horizon)
                    except Exception:
                        index = pd.RangeIndex(start=0, stop=horizon)
                    return pd.Series(values, index=index)

            return _LocalFallback()

        self._model_factories: dict[
            str, tuple[str, Callable[[dict[str, object]], BaseForecaster], dict[str, object]]
        ] = {
            "naive": (
                "Naive forecaster",
                lambda params: NaiveForecaster(**params),
                {"strategy": "drift"},
            ),
            "theta": (
                "Theta forecaster",
                lambda params: ThetaForecaster(**params)
                if statsmodels_available
                else _fallback_forecaster("drift"),
                {"sp": 12, "deseasonalize": False},
            ),
            "exp_smoothing": (
                "Exponential smoothing",
                lambda params: ExponentialSmoothing(**params)
                if statsmodels_available
                else _fallback_forecaster("last"),
                {"trend": "add", "sp": 1},
            ),
        }

        # Add AutoARIMA if available (only if dependencies are compatible)
        # AutoARIMA requires numpy<2, so it may not be available in all environments
        # Skip it for now due to compatibility issues
        if False and AUTOARIMA_AVAILABLE:  # Disabled for now
            self._model_factories["auto_arima"] = (
                "Auto ARIMA",
                lambda params: AutoARIMA(**params),
                {"suppress_warnings": True, "seasonal": False, "max_p": 5, "max_q": 5},
            )

        # Add Polynomial Trend if available
        if TREND_AVAILABLE:
            self._model_factories["poly_trend"] = (
                "Polynomial Trend",
                lambda params: PolynomialTrendForecaster(**params),
                {"degree": 1},
            )

        # Add BATS if available
        if BATS_AVAILABLE:
            self._model_factories["bats"] = (
                "BATS",
                lambda params: BATS(**params),
                {"use_box_cox": False, "use_trend": True, "use_damped_trend": False},
            )

        # Add TBATS if available
        if TBATS_AVAILABLE:
            self._model_factories["tbats"] = (
                "TBATS",
                lambda params: TBATS(**params),
                {"use_box_cox": False, "use_trend": True, "use_damped_trend": False},
            )

        # Add Croston if available (for intermittent demand forecasting)
        if CROSTON_AVAILABLE:
            self._model_factories["croston"] = (
                "Croston",
                lambda params: Croston(**params),
                {"smoothing": 0.1},
            )

        # Time Series Regression models (using make_reduction with sklearn regressors)
        # These models support exogenous variables (X) for forecasting
        if REDUCTION_AVAILABLE:
            self._model_factories["ts_ridge"] = (
                "Time Series Ridge Regression",
                lambda params: make_reduction(
                    Ridge(alpha=params.get("alpha", 1.0)),
                    window_length=params.get("window_length", 10),
                    strategy=params.get("strategy", "recursive"),
                ),
                {"alpha": 1.0, "window_length": 10, "strategy": "recursive"},
            )
            self._model_factories["ts_lasso"] = (
                "Time Series Lasso Regression",
                lambda params: make_reduction(
                    Lasso(alpha=params.get("alpha", 0.1)),
                    window_length=params.get("window_length", 10),
                    strategy=params.get("strategy", "recursive"),
                ),
                {"alpha": 0.1, "window_length": 10, "strategy": "recursive"},
            )
            self._model_factories["ts_random_forest"] = (
                "Time Series Random Forest",
                lambda params: make_reduction(
                    RandomForestRegressor(
                        n_estimators=params.get("n_estimators", 100),
                        max_depth=params.get("max_depth"),
                        random_state=params.get("random_state", 0),
                        n_jobs=-1,
                    ),
                    window_length=params.get("window_length", 10),
                    strategy=params.get("strategy", "recursive"),
                ),
                {"n_estimators": 100, "max_depth": None, "window_length": 10, "strategy": "recursive", "random_state": 0},
            )
            self._model_factories["ts_gradient_boosting"] = (
                "Time Series Gradient Boosting",
                lambda params: make_reduction(
                    GradientBoostingRegressor(
                        n_estimators=params.get("n_estimators", 100),
                        max_depth=params.get("max_depth", 3),
                        learning_rate=params.get("learning_rate", 0.1),
                        random_state=params.get("random_state", 0),
                    ),
                    window_length=params.get("window_length", 10),
                    strategy=params.get("strategy", "recursive"),
                ),
                {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "window_length": 10, "strategy": "recursive", "random_state": 0},
            )
            if MLPRegressor is not None:
                self._model_factories["ts_mlp"] = (
                    "Time Series MLP",
                    lambda params: make_reduction(
                        MLPRegressor(
                            hidden_layer_sizes=_parse_hidden_layer_sizes(
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
                        window_length=params.get("window_length", 10),
                        strategy=params.get("strategy", "recursive"),
                    ),
                    {
                        "hidden_layer_sizes": (100,),
                        "activation": "relu",
                        "solver": "adam",
                        "alpha": 0.0001,
                        "learning_rate": "constant",
                        "max_iter": 200,
                        "window_length": 10,
                        "strategy": "recursive",
                        "random_state": 0,
                    },
                )

    def available_models(self) -> list[tuple[str, str, dict[str, object]]]:
        return [
            (key, label, defaults.copy())
            for key, (label, _factory, defaults) in self._model_factories.items()
        ]

    # ------------------------------------------------------------------
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
        logger.info(f"Starting forecasting with {len(features)} features and {len(models)} models")
        logger.debug(f"Features: {[p.get('label', p.get('feature_id')) for p in features]}")
        logger.debug(f"Models: {models}")
        logger.debug(f"Parameters: {model_params}")
        if target_feature:
            logger.debug(f"Target feature: {target_feature.get('label', target_feature.get('feature_id'))}")
        
        if not features:
            raise ValueError("Select at least one feature to forecast")
        model_keys = [m for m in models if m in self._model_factories]
        if not model_keys:
            raise ValueError("No valid forecasting models selected")
        if forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive")

        model_params = model_params or {}

        def _emit_progress(pct: int) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(int(max(0, min(100, pct))))
            except Exception:
                logger.warning("Exception in _emit_progress", exc_info=True)

        def _emit_status(msg: str) -> None:
            if status_callback is None:
                return
            try:
                status_callback(str(msg))
            except Exception:
                logger.warning("Exception in _emit_status", exc_info=True)

        def _check_cancel() -> None:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Forecast run cancelled")

        def _emit_result(run: ForecastRunResult) -> None:
            if result_callback is None:
                return
            try:
                result_callback(run)
            except Exception:
                logger.warning("Exception in _emit_result", exc_info=True)

        _emit_progress(0)
        _emit_status("Preparing forecasting dataset…")
        _check_cancel()

        feature_payloads = [dict(p) for p in features]
        combined_payloads: list[Mapping[str, object]] = []
        feature_ids: list[int] = []
        seen_ids: set[int] = set()
        for payload in feature_payloads:
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

        # Add target feature to combined payloads if provided and not already included
        target_payload = dict(target_feature) if target_feature else None
        target_label: Optional[str] = None
        if target_payload:
            target_fid = target_payload.get("feature_id")
            if target_fid is not None:
                try:
                    target_fid_int = int(target_fid)
                    if target_fid_int not in seen_ids:
                        seen_ids.add(target_fid_int)
                        feature_ids.append(target_fid_int)
                        combined_payloads.append(target_payload)
                    target_label = _display_name(target_payload)
                    logger.debug(f"Target feature for exogenous variables: {target_label}")
                except Exception:
                    target_payload = None
                    target_label = None

        if data_frame is not None:
            _check_cancel()
            data = self._normalize_preprocessed_frame(data_frame, combined_payloads)
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
                _emit_status("No data available for forecasting.")
                return ForecastSummary()
            data = self._prepare_wide_frame(df, combined_payloads)

        _check_cancel()
        if data is None or data.empty:
            _emit_status("No data available for forecasting.")
            return ForecastSummary()

        data = data.dropna(subset=["t"])
        if data.empty:
            _emit_status("No data available for forecasting.")
            return ForecastSummary()
        data = data.sort_values("t").reset_index(drop=True)

        runs: list[ForecastRunResult] = []
        progress_rows: list[dict[str, object]] = []
        total_runs = len(feature_payloads) * len(model_keys)
        done = 0

        for payload in feature_payloads:
            _check_cancel()
            feature_label = _display_name(payload)
            col = feature_label
            if col not in data.columns:
                continue
            series = self._prepare_series(data[["t", col]], value_column=col)
            if series.empty:
                continue

            # Prepare exogenous variables (X) if target feature is specified
            X_series: Optional[pd.DataFrame] = None
            if target_label and target_label in data.columns and target_label != col:
                try:
                    X_data = data[["t", target_label]].copy()
                    X_data["t"] = pd.to_datetime(X_data["t"], errors="coerce")
                    X_data = X_data.dropna()
                    X_data = X_data.sort_values("t")
                    X_series = pd.DataFrame(
                        {target_label: X_data[target_label].values},
                        index=pd.DatetimeIndex(X_data["t"])
                    )
                    # Align X with the series index
                    X_series = X_series.reindex(series.index, method="nearest")
                    X_series = X_series.dropna()
                    logger.debug(f"Prepared exogenous data X with {len(X_series)} rows for {feature_label}")
                except Exception as e:
                    logger.warning(f"Failed to prepare exogenous data: {e}")
                    X_series = None

            for model_key in model_keys:
                _check_cancel()
                model_label, model_factory, model_defaults = self._model_factories[model_key]
                cfg = model_defaults.copy()
                cfg.update(model_params.get(model_key, {}))
                # Filter out None values to avoid passing unsupported parameters to model constructors
                cfg_before_filter = cfg.copy()
                cfg = {k: v for k, v in cfg.items() if v is not None}
                
                if cfg_before_filter != cfg:
                    logger.debug(f"Filtered None values for {model_key}: {set(cfg_before_filter.keys()) - set(cfg.keys())}")
                logger.debug(f"Creating {model_label} forecaster with config: {cfg}")

                try:
                    forecaster = model_factory(cfg)
                except Exception as e:
                    logger.error(
                        f"Failed to create forecaster for {model_label}: {e}", exc_info=True
                    )
                    continue

                # Handle different window strategies
                # Check if this is a time series regression model that supports exogenous variables
                is_ts_regression = model_key.startswith("ts_")
                use_exog = is_ts_regression and X_series is not None and not X_series.empty
                
                if window_strategy in ("sliding", "expanding"):
                    # Use window-based cross-validation (sliding or expanding)
                    _emit_status(f"Setting up splitter for {feature_label}…")
                    if window_strategy == "expanding":
                        splitter = self._create_expanding_window_splitter(
                            len(series),
                            forecast_horizon=forecast_horizon,
                            initial_window=initial_window,
                        )
                    else:
                        splitter = self._create_sliding_window_splitter(
                            len(series),
                            forecast_horizon=forecast_horizon,
                            initial_window=initial_window
                        )
                    logger.debug(f"Splitter configured for {feature_label}")

                    # Run cross-validation with splitter
                    logger.debug(f"Running cross-validation for {model_label} on {feature_label}")
                    try:
                        y_pred_cv, y_true_cv, splits_info = self._run_backtest(
                            forecaster, series, splitter, forecast_horizon,
                            window_callback=window_callback,
                            model_label=model_label,
                            feature_label=feature_label,
                            X=X_series if use_exog else None,
                        )
                        logger.debug(f"Cross-validation complete: {len(splits_info)} splits")
                    except Exception as e:
                        logger.error(f"Failed to run backtest for {model_label}: {e}", exc_info=True)
                        raise

                    # Train final model on all data and forecast future
                    logger.debug(f"Training final {model_label} on complete dataset")
                    try:
                        if use_exog:
                            forecaster.fit(series, X=X_series.loc[series.index])
                        else:
                            forecaster.fit(series)
                        fh_final = ForecastingHorizon(np.arange(1, forecast_horizon + 1), is_relative=True)
                        # For future predictions with exogenous variables, we need future X values
                        # Since we don't have them, we'll use the last known values
                        if use_exog:
                            future_X = self._prepare_future_exogenous(X_series, series, forecast_horizon)
                            y_pred_future = forecaster.predict(fh_final, X=future_X)
                        else:
                            y_pred_future = forecaster.predict(fh_final)
                        logger.debug(f"Final forecast complete: {len(y_pred_future)} future predictions")
                    except Exception as e:
                        logger.error(f"Failed final fit/predict for {model_label}: {e}", exc_info=True)
                        raise

                    metrics = self._compute_metrics(y_true_cv, y_pred_cv)
                    logger.debug(f"Metrics for {model_label}: RMSE={metrics.get('rmse'):.4f}, MAE={metrics.get('mae'):.4f}")
                    forecast_frame = self._build_forecast_frame_with_cv(series, y_true_cv, y_pred_cv, y_pred_future, splits_info)
                else:
                    # Use simple train/test split (single fold) based on forecast_horizon
                    if forecast_horizon is None or forecast_horizon <= 0:
                        raise ValueError(
                            "forecast_horizon must be a positive integer for simple train/test split."
                        )

                    if len(series) <= forecast_horizon:
                        raise ValueError(
                            f"Series length ({len(series)}) must be greater than "
                            f"forecast_horizon ({forecast_horizon}) for simple split."
                        )

                    logger.debug(
                        f"Using simple train/test split (last {forecast_horizon} rows as test) "
                        f"for {model_label} on {feature_label}"
                    )

                    # Train = all but last `forecast_horizon` rows
                    # Test  = last `forecast_horizon` rows
                    y_train = series.iloc[:-forecast_horizon]
                    y_test = series.iloc[-forecast_horizon:]
                    
                    # Prepare exogenous data splits if using time series regression
                    X_train: Optional[pd.DataFrame] = None
                    X_test: Optional[pd.DataFrame] = None
                    if use_exog and X_series is not None:
                        X_train = X_series.loc[y_train.index].copy() if y_train.index.isin(X_series.index).any() else None
                        X_test = X_series.loc[y_test.index].copy() if y_test.index.isin(X_series.index).any() else None

                    logger.debug(
                        f"Training {model_label} on {len(y_train)} samples, "
                        f"testing on {len(y_test)} samples"
                    )

                    try:
                        if use_exog and X_train is not None:
                            forecaster.fit(y_train, X=X_train)
                        else:
                            forecaster.fit(y_train)
                        logger.debug("Model training complete")
                    except Exception as e:
                        logger.error(f"Failed to fit {model_label}: {e}", exc_info=True)
                        raise

                    try:
                        # Predict for test period:
                        # we ask the model to forecast `len(y_test)` steps ahead
                        fh_test = ForecastingHorizon(
                            np.arange(1, len(y_test) + 1), is_relative=True
                        )
                        if use_exog and X_test is not None:
                            y_pred_test_raw = forecaster.predict(fh_test, X=X_test)
                        else:
                            y_pred_test_raw = forecaster.predict(fh_test)

                        # Align predictions to the actual test index (last `forecast_horizon` rows)
                        y_pred_test = pd.Series(
                            y_pred_test_raw.values[:len(y_test)],
                            index=y_test.index[:len(y_pred_test_raw)],
                        )
                        logger.debug(f"Test predictions complete: {len(y_pred_test)}")

                        # Predict future forecast from the end of the full series.
                        # Re-use the same forecast_horizon to keep behavior consistent
                        fh_future = ForecastingHorizon(
                            np.arange(1, forecast_horizon + 1), is_relative=True
                        )
                        # For future predictions with exogenous variables, use last known values
                        if use_exog and X_series is not None:
                            future_X = self._prepare_future_exogenous(X_series, series, forecast_horizon)
                            y_pred_future = forecaster.predict(fh_future, X=future_X)
                        else:
                            y_pred_future = forecaster.predict(fh_future)
                        logger.debug(f"Future forecast complete: {len(y_pred_future)}")

                    except Exception as e:
                        logger.error(f"Failed to predict with {model_label}: {e}", exc_info=True)
                        raise

                    metrics = self._compute_metrics(y_test, y_pred_test)
                    logger.debug(
                        f"Metrics for {model_label}: "
                        f"RMSE={metrics.get('rmse'):.4f}, MAE={metrics.get('mae'):.4f}"
                    )
                    forecast_frame = self._build_forecast_frame(
                        y_train, y_test, y_pred_test, y_pred_future
                    )

                key = f"{model_key}:{payload.get('feature_id', feature_label)}:{int(pd.Timestamp.utcnow().timestamp() * 1000)}:{done + 1}"
                progress_rows.append(
                    {
                        "t": pd.Timestamp.utcnow(),
                        "model": model_label,
                        "feature": feature_label,
                        "rmse": metrics.get("rmse"),
                        "mae": metrics.get("mae"),
                        "r2": -abs(metrics.get("rmse") or 0.0),
                        "step": len(progress_rows) + 1,
                    }
                )

                run = ForecastRunResult(
                    key=key,
                    model_key=model_key,
                    model_label=model_label,
                    feature_key=str(payload.get("feature_id", feature_label)),
                    feature_label=feature_label,
                    metrics=metrics,
                    progress_frame=pd.DataFrame(progress_rows),
                    forecast_frame=forecast_frame,
                )

                runs.append(run)
                done += 1
                logger.info(f"Completed {model_label} for {feature_label}")
                if total_runs:
                    pct = int(round((done / total_runs) * 100))
                    _emit_progress(pct)
                    _emit_status(f"Completed {model_label} for {feature_label} ({done}/{total_runs})")
                _emit_result(run)

        if done >= total_runs:
            _emit_progress(100)
        _emit_status("Forecasting experiments completed.")
        return ForecastSummary(runs=runs)

    # ------------------------------------------------------------------
    def _prepare_wide_frame(
        self,
        df: pd.DataFrame,
        payloads: Sequence[Mapping[str, object]],
    ) -> pd.DataFrame:
        df = df.copy()
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df = df.dropna(subset=["t"])
        if df.empty:
            return pd.DataFrame(columns=["t"])

        df["feature_id"] = pd.to_numeric(df["feature_id"], errors="coerce")
        df = df.dropna(subset=["feature_id"])
        if df.empty:
            return pd.DataFrame(columns=["t"])
        df["feature_id"] = df["feature_id"].astype(int)

        pivot = (
            df.pivot_table(index="t", columns="feature_id", values="v", aggfunc="mean", dropna=False)
            .sort_index()
        )
        pivot = pivot.reset_index()

        name_map = {int(p["feature_id"]): _display_name(p) for p in payloads}
        rename = {}
        for fid in pivot.columns:
            if isinstance(fid, (int, np.integer)) and fid in name_map:
                rename[fid] = name_map[fid]
        wide = pivot.rename(columns=rename)
        return wide

    def _normalize_preprocessed_frame(
        self,
        frame: pd.DataFrame,
        payloads: Sequence[Mapping[str, object]],
    ) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["t"])
        data = frame.copy()
        if "t" not in data.columns:
            raise ValueError("Preprocessed data is missing the 't' column")
        data["t"] = pd.to_datetime(data["t"], errors="coerce")
        data = data.dropna(subset=["t"])
        if data.empty:
            return pd.DataFrame(columns=["t"])

        ordered: list[str] = []
        for payload in payloads:
            name = _display_name(payload)
            if name not in ordered:
                ordered.append(name)

        missing = [name for name in ordered if name not in data.columns]
        if missing:
            raise ValueError(f"Missing expected columns in preprocessed dataset: {missing}")

        columns = ["t"] + ordered
        return data.loc[:, columns]

    def _prepare_series(self, df: pd.DataFrame, *, value_column: str) -> pd.Series:
        df = df.copy()
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df = df.dropna(subset=["t", value_column])
        if df.empty:
            return pd.Series(dtype=float)

        # Sort by time and use the timestamps exactly as given
        df = df.sort_values("t")
        series = pd.Series(df[value_column].to_numpy(), index=pd.DatetimeIndex(df["t"]))

        freq = series.index.freqstr or pd.infer_freq(series.index)
        if freq is None and len(series.index) > 1:
            try:
                deltas = pd.Series(series.index[1:] - series.index[:-1]).dropna()
                median_delta = deltas.median()
                if pd.notna(median_delta) and median_delta > pd.Timedelta(0):
                    freq = to_offset(median_delta).freqstr
            except Exception:
                freq = None

        if freq:
            try:
                series = series.asfreq(freq, method="pad")
                series.attrs["_inferred_freq"] = freq
            except Exception:
                logger.warning("Exception in _prepare_series", exc_info=True)

        return series

    def _compute_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        if y_true is None or y_true.empty:
            return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}
        try:
            aligned_true = y_true.iloc[: len(y_pred)]
        except Exception:
            aligned_true = y_true
        if aligned_true.empty:
            return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}
        
        # Drop NaN values that may result from resampling irregular data
        mask = ~(aligned_true.isna() | y_pred[: len(aligned_true)].isna())
        aligned_true_clean = aligned_true[mask]
        y_pred_clean = y_pred[: len(aligned_true)][mask]
        
        if aligned_true_clean.empty or y_pred_clean.empty:
            return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}
        
        # Compute MSE and then take sqrt to get RMSE (avoiding deprecated squared parameter)
        mse = float(mean_squared_error(aligned_true_clean, y_pred_clean))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(aligned_true_clean, y_pred_clean))
        with np.errstate(divide="ignore", invalid="ignore"):
            mape_vals = np.abs((aligned_true_clean - y_pred_clean) / aligned_true_clean.replace(0, np.nan))
            mape = float(np.nanmean(mape_vals) * 100)
        return {"rmse": rmse, "mae": mae, "mape": mape}

    def _build_forecast_frame(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        y_pred_test: pd.Series,
        y_pred_future: pd.Series,
    ) -> pd.DataFrame:
        """Build a combined frame with training data, test data with predictions, and future forecast."""
        logger.debug(f"Building forecast frame: train={len(y_train)}, test={len(y_test)}, pred_test={len(y_pred_test)}, pred_future={len(y_pred_future)}")
        
        # Build training frame with actual values only (no predictions)
        train_frame = pd.DataFrame({
            "t": y_train.index,
            "Actual": y_train.to_numpy(),
            "Period": "Training"
        })
        
        # Build test frame with both actual values and predictions
        test_frame = pd.DataFrame({
            "t": y_test.index,
            "Actual": y_test.to_numpy(),
            "Forecast": y_pred_test.to_numpy() if len(y_pred_test) == len(y_test) else np.full(len(y_test), np.nan),
            "Period": "Testing"
        })
        
        # Build forecast frame for the future predictions beyond test set
        last_time = y_test.index[-1] if len(y_test) > 0 else y_train.index[-1]
        
        # Try to get frequency from series metadata or infer from data
        freq = y_train.attrs.get('_inferred_freq') or y_test.attrs.get('_inferred_freq')
        
        if freq is None:
            # Try to infer frequency from the combined train + test series
            try:
                combined_series = pd.concat([y_train, y_test])
                freq = pd.infer_freq(combined_series)
            except Exception:
                logger.warning("Exception in _build_forecast_frame", exc_info=True)
        
        logger.debug(f"Detected frequency for forecast: {freq}")
        
        forecast_frame = pd.DataFrame()
        if len(y_pred_future) > 0:
            # Generate forecast index
            if freq is not None:
                try:
                    # Generate future dates with the detected frequency
                    forecast_index = pd.date_range(start=last_time, periods=len(y_pred_future) + 1, freq=freq)[1:]
                    logger.debug(f"Generated {len(forecast_index)} forecast dates with frequency {freq}")
                except Exception as e:
                    logger.warning(f"Failed to generate forecast dates with freq {freq}: {e}")
                    # Fallback to hourly
                    forecast_index = pd.date_range(start=last_time, periods=len(y_pred_future), freq='h')
            else:
                logger.warning("Could not determine frequency, creating hourly index for forecast")
                # No frequency detected - create hourly increments (common for resampled data)
                forecast_index = pd.date_range(start=last_time, periods=len(y_pred_future), freq='h')
            
            # Create forecast dataframe for future predictions
            forecast_frame = pd.DataFrame({
                "t": forecast_index,
                "Forecast": y_pred_future.to_numpy(),
                "Period": "Forecast"
            })
        
        logger.debug(f"Frame sizes: train={len(train_frame)}, test={len(test_frame)}, forecast={len(forecast_frame)}")
        
        # Combine all frames
        combined = pd.concat([train_frame, test_frame, forecast_frame], ignore_index=True)
        combined = combined.dropna(subset=["t"]).sort_values("t")
        
        logger.debug(f"Combined frame shape: {combined.shape}, date range: {combined['t'].min()} to {combined['t'].max()}")
        
        return combined.reset_index(drop=True)
    
    def _auto_initial_window(
        self,
        series_length: int,
        forecast_horizon: int,
        initial_window: Optional[int] = None,
    ) -> int:
        """Compute a reasonable initial window size.

        - At least forecast_horizon + 1
        - At least 20 samples (to avoid tiny, unstable fits)
        - At least 20% of the series length, if possible
        - Not larger than series_length - forecast_horizon
        """
        if initial_window is not None and initial_window >= 0:
            return initial_window

        min_window = forecast_horizon + 1
        min_samples = 20
        dynamic_window = int(series_length * 0.2)

        # Base candidate
        window = max(min_window, min_samples, dynamic_window)

        # Ensure we leave enough room for at least one horizon
        max_window = max(forecast_horizon + 1, series_length - forecast_horizon)
        window = min(window, max_window)

        return window

    def _prepare_future_exogenous(
        self,
        X_series: pd.DataFrame,
        series: pd.Series,
        forecast_horizon: int,
    ) -> pd.DataFrame:
        """Prepare exogenous variables for future predictions.
        
        Since we don't have actual future X values, we use the last known values
        repeated for the forecast horizon.
        
        Args:
            X_series: DataFrame of exogenous variables with datetime index
            series: The target time series (used for frequency inference)
            forecast_horizon: Number of steps to forecast
        
        Returns:
            DataFrame with future X values indexed by future dates
        """
        last_X = X_series.iloc[[-1] * forecast_horizon].copy()
        freq = series.index.freq or pd.infer_freq(series.index) or "h"
        last_X.index = pd.date_range(
            start=series.index[-1],
            periods=forecast_horizon + 1,
            freq=freq
        )[1:]
        return last_X

    def _create_sliding_window_splitter(
        self,
        series_length: int,
        forecast_horizon: int,
        initial_window: Optional[int] = None,
    ):
        """Create a sliding window splitter for time series cross-validation.
        
        Args:
            series_length: Total length of the time series
            forecast_horizon: Number of periods to forecast ahead (multi-step)
            initial_window: Size of initial training window. If None or < 0, auto-calculated.
        
        Returns:
            SlidingWindowSplitter configured for cross-validation
        """
        # Auto-determine initial window if needed
        initial_window = self._auto_initial_window(
            series_length=series_length,
            forecast_horizon=forecast_horizon,
            initial_window=initial_window,
        )

        # Build multi-step forecast horizon: [1, 2, ..., H]
        fh_array = np.arange(1, forecast_horizon + 1)

        logger.debug(
            f"SlidingWindowSplitter: length={series_length}, "
            f"initial_window={initial_window}, fh={fh_array}, "
            f"step_length={forecast_horizon}"
        )

        return SlidingWindowSplitter(
            window_length=initial_window,
            fh=fh_array,
            step_length=forecast_horizon,
        )

    def _create_expanding_window_splitter(
        self,
        series_length: int,
        forecast_horizon: int,
        initial_window: Optional[int] = None,
    ):
        """Create an expanding window splitter for time series cross-validation.
        
        Args:
            series_length: Total length of the time series
            forecast_horizon: Number of periods to forecast ahead (multi-step)
            initial_window: Size of initial training window. If None or < 0, auto-calculated.
        
        Returns:
            ExpandingWindowSplitter configured for cross-validation
        """
        # Auto-determine initial window if needed
        initial_window = self._auto_initial_window(
            series_length=series_length,
            forecast_horizon=forecast_horizon,
            initial_window=initial_window,
        )

        # Build multi-step forecast horizon: [1, 2, ..., H]
        fh_array = np.arange(1, forecast_horizon + 1)

        logger.debug(
            f"ExpandingWindowSplitter: length={series_length}, "
            f"initial_window={initial_window}, fh={fh_array}, "
            f"step_length={forecast_horizon}"
        )

        return ExpandingWindowSplitter(
            initial_window=initial_window,
            fh=fh_array,
            step_length=forecast_horizon,
        )

    def _run_backtest(
        self,
        forecaster: BaseForecaster,
        y: pd.Series,
        splitter,
        forecast_horizon: int,
        window_callback: Optional[Callable[[dict], None]] = None,
        model_label: str = "",
        feature_label: str = "",
        X: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.Series, pd.Series, list[dict]]:
        """Run backtest using time series splitter.
        
        Emits window-level callbacks for each split step so you can see detailed progress.
        
        Args:
            forecaster: sktime forecaster instance
            y: Time series to backtest on
            splitter: Splitter instance (SlidingWindowSplitter or ExpandingWindowSplitter)
            forecast_horizon: Number of periods to forecast (used for logging / sanity checks)
            window_callback: Optional callback(window_info_dict) emitted for each window step
            model_label: Model name for window callback context
            feature_label: Feature name for window callback context
            X: Optional exogenous variables DataFrame with same index as y
        
        Returns:
            Tuple of (predictions, actuals, split_info)
        """
        # Normalize target series to a flat, univariate time index expected by sktime
        if not isinstance(y, pd.Series):
            try:
                y = pd.Series(y.squeeze())
            except Exception:
                y = pd.Series(pd.array(y))

        if isinstance(y.index, pd.MultiIndex):
            y = y.copy()
            y.index = y.index.get_level_values(-1)

        try:
            y.index = pd.DatetimeIndex(y.index)
        except Exception:
            y = y.reset_index(drop=True)
            y.index = pd.RangeIndex(len(y))

        y = y.sort_index()
        y = pd.Series(pd.to_numeric(y, errors="coerce"), index=y.index)
        y = y.dropna()
        if y.empty:
            raise ValueError("No valid target values for backtesting")

        if X is not None:
            if isinstance(X, pd.Series):
                X = X.to_frame()
            try:
                X.index = pd.DatetimeIndex(X.index)
            except Exception:
                X = X.reset_index(drop=True)
                X.index = pd.RangeIndex(len(X))
            X = X.reindex(y.index).dropna(how="all")
            if X.empty:
                X = None

        y_preds: list[pd.Series] = []
        y_trues: list[pd.Series] = []
        splits_info: list[dict] = []
        total_splits = None
        
        for i, (train_idx, test_idx) in enumerate(splitter.split(y)):
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Sanity: test length should generally match forecast_horizon
            if len(y_test) != forecast_horizon:
                logger.debug(
                    f"Split {i}: test_size={len(y_test)} differs from "
                    f"forecast_horizon={forecast_horizon}"
                )

            # Prepare exogenous data for this split if available
            X_train: Optional[pd.DataFrame] = None
            X_test: Optional[pd.DataFrame] = None
            if X is not None:
                try:
                    X_train = X.loc[y_train.index].copy() if y_train.index.isin(X.index).any() else None
                    X_test = X.loc[y_test.index].copy() if y_test.index.isin(X.index).any() else None
                except Exception as e:
                    logger.warning(f"Failed to align exogenous data for split {i}: {e}")
                    X_train = None
                    X_test = None

            # Fit forecaster on training data
            if X_train is not None and not X_train.empty:
                forecaster.fit(y_train, X=X_train)
            else:
                forecaster.fit(y_train)

            # Predict for the test period using relative steps 1..len(y_test)
            fh_test = ForecastingHorizon(
                np.arange(1, len(y_test) + 1),
                is_relative=True,
            )
            if X_test is not None and not X_test.empty:
                y_pred_raw = forecaster.predict(fh_test, X=X_test)
            else:
                y_pred_raw = forecaster.predict(fh_test)

            # Align predictions with the test indices
            y_pred = pd.Series(
                y_pred_raw.values[: len(y_test)],
                index=y_test.index[: len(y_pred_raw)],
            )

            y_preds.append(y_pred)
            y_trues.append(y_test)

            window_info = {
                "split": i,
                "train_size": len(y_train),
                "test_size": len(y_test),
                "train_start": y_train.index[0],
                "train_end": y_train.index[-1],
                "test_start": y_test.index[0],
                "test_end": y_test.index[-1],
            }
            splits_info.append(window_info)

            logger.debug(
                f"Split {i}: train={len(y_train)}, "
                f"test={len(y_test)}, pred={len(y_pred)}"
            )
            
            # Emit window callback with detailed information about this step
            if window_callback is not None:
                try:
                    window_callback({
                        "window_index": i,
                        "model": model_label,
                        "feature": feature_label,
                        "train_size": len(y_train),
                        "test_size": len(y_test),
                        "status": f"Completed window {i + 1}: {len(y_train)} train rows, {len(y_test)} test rows"
                    })
                except Exception as exc:
                    logger.warning(f"Window callback error: {exc}")

        # Combine all predictions and actuals
        y_pred_combined = pd.concat(y_preds, ignore_index=False)
        y_true_combined = pd.concat(y_trues, ignore_index=False)

        logger.debug(
            f"Backtest complete: {len(splits_info)} splits, "
            f"total predictions={len(y_pred_combined)}"
        )

        return y_pred_combined, y_true_combined, splits_info

    def _build_forecast_frame_with_cv(
        self,
        y: pd.Series,
        y_true_cv: pd.Series,
        y_pred_cv: pd.Series,
        y_pred_future: pd.Series,
        splits_info: list[dict],
    ) -> pd.DataFrame:
        """Build forecast frame combining cross-validation results and future forecast.
        
        Args:
            y: Full time series
            y_true_cv: Actual values from cross-validation
            y_pred_cv: Predictions from cross-validation
            y_pred_future: Future forecast predictions
            splits_info: Information about cross-validation splits
        
        Returns:
            DataFrame with combined results
        """
        logger.debug(
            f"Building CV forecast frame: series_len={len(y)}, cv_actuals={len(y_true_cv)}, "
            f"cv_preds={len(y_pred_cv)}, future_preds={len(y_pred_future)}"
        )
        
        # Create frame for cross-validation period (where we have both actual and predicted)
        cv_frame = pd.DataFrame({
            "t": y_true_cv.index,
            "Actual": y_true_cv.values,
            "Forecast": y_pred_cv.values,
            "Period": "Validation"
        })
        
        # Create frame for training period (data before any cross-validation test points)
        if len(splits_info) > 0:
            first_test_start = splits_info[0]["test_start"]
            train_mask = y.index < first_test_start
            if train_mask.any():
                train_frame = pd.DataFrame({
                    "t": y.index[train_mask],
                    "Actual": y.values[train_mask],
                    "Period": "Training"
                })
            else:
                train_frame = pd.DataFrame(columns=["t", "Actual", "Period"])
        else:
            train_frame = pd.DataFrame(columns=["t", "Actual", "Period"])
        
        # Create frame for future forecast
        last_time = y.index[-1]
        freq = y.attrs.get('_inferred_freq')
        
        if len(y_pred_future) > 0:
            if freq is not None:
                try:
                    forecast_index = pd.date_range(start=last_time, periods=len(y_pred_future) + 1, freq=freq)[1:]
                except Exception as e:
                    logger.warning(f"Failed to generate future dates with freq {freq}: {e}")
                    forecast_index = pd.date_range(start=last_time, periods=len(y_pred_future), freq='h')
            else:
                forecast_index = pd.date_range(start=last_time, periods=len(y_pred_future), freq='h')
            
            forecast_frame = pd.DataFrame({
                "t": forecast_index,
                "Forecast": y_pred_future.values,
                "Period": "Forecast"
            })
        else:
            forecast_frame = pd.DataFrame(columns=["t", "Forecast", "Period"])
        
        # Combine all frames
        combined = pd.concat([train_frame, cv_frame, forecast_frame], ignore_index=True)
        combined = combined.dropna(subset=["t"]).sort_values("t")
        
        logger.debug(f"Combined CV frame shape: {combined.shape}, date range: {combined['t'].min()} to {combined['t'].max()}")
        
        return combined.reset_index(drop=True)


__all__ = ["ForecastingService", "ForecastSummary", "ForecastRunResult"]

