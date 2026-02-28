
from __future__ import annotations
import logging
from typing import Mapping, Optional, Sequence

import pandas as pd
from PySide6.QtCore import QObject, Signal

from backend.services.forecasting_service import ForecastRunResult, ForecastingService, ForecastSummary

from ...models.hybrid_pandas_model import DataFilters, HybridPandasModel
from ...utils.model_persistence import frame_to_records, normalize_for_json, records_to_frame
from ...threading.runner import run_in_thread
from ...threading.utils import run_in_main_thread
from ...viewmodels.log_view_model import get_log_view_model
from ...utils import (
    clear_progress,
    set_progress,
    set_status_text,
    toast_error,
    toast_info,
    toast_success,
    toast_warn,
)


class ForecastingViewModel(QObject):
    """Qt-friendly interface for forecasting experiments used by the tab UI."""

    database_changed = Signal(object)
    run_requested = Signal()
    features_changed = Signal()
    run_started = Signal(object)
    run_progress = Signal(int)
    run_window_progress = Signal(dict)  # Emits window-level progress info
    run_partial = Signal(object)
    run_finished = Signal(object)
    run_failed = Signal(str)
    status_changed = Signal(str)

    def __init__(
        self,
        data_model: HybridPandasModel,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._data_model: HybridPandasModel = data_model
        self._service: Optional[ForecastingService] = None
        self._running = False
        self._last_status_message: Optional[str] = None

        self._data_model.database_changed.connect(self._on_database_changed)
        self._on_database_changed(self._data_model.path)

    # ------------------------------------------------------------------
    @property
    def data_model(self) -> HybridPandasModel:
        return self._data_model

    # ------------------------------------------------------------------
    def request_run(self) -> None:
        self.run_requested.emit()

    def notify_features_changed(self) -> None:
        self.features_changed.emit()

    # ------------------------------------------------------------------
    def _close_database(self) -> None:
        self._running = False
        self._service = None
        self._last_status_message = None

    def close_database(self) -> None:
        self._close_database()
        self.database_changed.emit(None)

    def _on_database_changed(self, _path) -> None:
        self._close_database()
        try:
            db = self._data_model.db
            self._service = ForecastingService(db)
        except Exception as exc:
            self._service = None
            self._log_error(str(exc))
            self.database_changed.emit(None)
            return
        self.database_changed.emit(db)

    # ------------------------------------------------------------------
    def available_models(self) -> list[tuple[str, str, dict[str, object]]]:
        if not self._service:
            return []
        return self._service.available_models()

    def save_run(
        self,
        run: ForecastRunResult,
        context: Mapping[str, object],
    ) -> Optional[int]:
        if not self._service:
            raise RuntimeError("Database is not initialised")
        if run is None:
            return None
        db = self._data_model.db

        feature = context.get("feature") or {}
        target = context.get("target") or {}

        features: list[tuple[int, str]] = []
        feature_id = _safe_feature_id(feature)
        if feature_id is not None:
            features.append((feature_id, "feature"))
        target_id = _safe_feature_id(target)
        if target_id is not None and target_id not in {fid for fid, _ in features}:
            features.append((target_id, "target"))

        model_params = context.get("model_params") or {}
        model_label = context.get("model_label") or run.model_label

        artifacts = {
            "progress_frame": frame_to_records(run.progress_frame),
            "forecast_frame": frame_to_records(run.forecast_frame),
            "metrics": normalize_for_json(run.metrics),
        }

        parameters = {
            "run_key": run.key,
            "display": {
                "model_label": model_label,
                "feature_label": context.get("feature_label") or run.feature_label,
            },
            "feature": normalize_for_json(feature),
            "target": normalize_for_json(target),
            "forecast": normalize_for_json(context.get("forecast") or {}),
        }

        results = _build_forecast_results(run)

        model_id = db.save_model_run(
            name=f"{model_label} · {context.get('feature_label') or run.feature_label}",
            model_type="forecasting",
            algorithm_key=run.model_key,
            selector_key=None,
            preprocessing=normalize_for_json(context.get("preprocessing") or {}),
            filters=normalize_for_json(context.get("filters") or {}),
            hyperparameters={"model_params": normalize_for_json(model_params)},
            parameters=parameters,
            artifacts=normalize_for_json(artifacts),
            features=features,
            results=results,
        )
        return model_id

    def delete_saved_model(self, model_id: int) -> None:
        if not self._service:
            raise RuntimeError("Database is not initialised")
        if model_id is None:
            return
        self._data_model.db.delete_model(int(model_id))

    def load_saved_runs(self) -> list[ForecastRunResult]:
        if not self._service:
            return []
        db = self._data_model.db
        saved = []
        try:
            models = db.list_models(model_type="forecasting")
        except Exception:
            return []
        for _, row in models.iterrows():
            model_id = int(row.get("model_id"))
            details = db.fetch_model(model_id)
            if not details:
                continue
            run = _forecast_run_from_details(details)
            if run is not None:
                saved.append(run)
        return saved

    # ------------------------------------------------------------------
    def run_forecasts(
        self,
        *,
        data_frame: pd.DataFrame,
        features: Sequence[Mapping[str, object]],
        models: Sequence[str],
        data_filters: Optional[DataFilters] = None,
        systems: Optional[Sequence[str]] = None,
        Datasets: Optional[Sequence[str]] = None,
        start=None,
        end=None,
        model_params: Optional[Mapping[str, Mapping[str, object]]] = None,
        forecast_horizon: int = 12,
        preprocessing: Optional[Mapping[str, object]] = None,
        window_strategy: str = "sliding",
        initial_window: Optional[int] = None,
        target_feature: Optional[Mapping[str, object]] = None,
        progress_callback=None,
        status_callback=None,
        result_callback=None,
        window_callback=None,
        stop_event=None,
    ) -> ForecastSummary:
        if not self._service:
            raise RuntimeError("Database is not initialised")

        logger = logging.getLogger(__name__)
        logger.info(f"Preparing forecasts with {len(features)} features and {len(models)} models")

        if data_filters is not None:
            systems = data_filters.systems or systems
            Datasets = data_filters.Datasets or Datasets
            start = data_filters.start or start
            end = data_filters.end or end
        
        payloads = [dict(p) for p in features]
        logger.debug(f"Feature payloads: {[p.get('label', p.get('feature_id')) for p in payloads]}")
        
        target_payload = dict(target_feature) if target_feature else None
        
        if not isinstance(data_frame, pd.DataFrame):
            raise RuntimeError(
                "ForecastingViewModel requires a pre-fetched DataFrame from DataSelectorWidget."
            )
        data_frame = data_frame.copy()
        logger.info(f"Preprocessed data: {data_frame.shape[0]} rows, {data_frame.shape[1]} columns")

        return self._service.run_forecasts(
            features=payloads,
            models=models,
            systems=systems,
            Datasets=Datasets,
            start=start,
            end=end,
            model_params=model_params,
            forecast_horizon=forecast_horizon,
            progress_callback=progress_callback,
            status_callback=status_callback,
            result_callback=result_callback,
            window_callback=window_callback,
            data_frame=data_frame,
            stop_event=stop_event,
            window_strategy=window_strategy,
            initial_window=initial_window,
            target_feature=target_payload,
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._close_database()

    def is_running(self) -> bool:
        return self._running

    # @ai(gpt-5, codex, refactor, 2026-02-28)
    def start_forecasts(
        self,
        *,
        data_frame: pd.DataFrame,
        features: Sequence[Mapping[str, object]],
        models: Sequence[str],
        data_filters: Optional[DataFilters] = None,
        systems: Optional[Sequence[str]] = None,
        Datasets: Optional[Sequence[str]] = None,
        start=None,
        end=None,
        model_params: Optional[Mapping[str, Mapping[str, object]]] = None,
        forecast_horizon: int = 12,
        preprocessing: Optional[Mapping[str, object]] = None,
        window_strategy: str = "sliding",
        initial_window: Optional[int] = None,
        target_feature: Optional[Mapping[str, object]] = None,
    ) -> None:
        if not self._service:
            error_msg = "Database is not initialised or ForecastingService failed to load"
            self._log_error(error_msg)
            raise RuntimeError(error_msg)
        if self.is_running():
            raise RuntimeError("Forecast run already in progress")

        self._last_status_message = None
        expected_runs = max(0, len(features) * len(models))
        self._log_info("Starting forecasting experiments…")
        try:
            toast_info("Starting forecasting experiments…", title="Forecasting", tab_key="forecasting")
        except Exception:
            logger.warning("Exception in start_forecasts", exc_info=True)
        self.run_started.emit(expected_runs)
        self._handle_status_update("Generating forecasts...")
        set_progress(0)
        self._running = True

        def _run(*, progress_callback=None, stop_event=None, result_callback=None):
            return self.run_forecasts(
                data_frame=data_frame,
                features=features,
                models=models,
                data_filters=data_filters,
                systems=systems,
                Datasets=Datasets,
                start=start,
                end=end,
                model_params=model_params,
                forecast_horizon=forecast_horizon,
                preprocessing=preprocessing,
                window_strategy=window_strategy,
                initial_window=initial_window,
                target_feature=target_feature,
                progress_callback=progress_callback,
                status_callback=lambda message: run_in_main_thread(
                    self._handle_service_status_update, message
                ),
                result_callback=result_callback,
                window_callback=lambda window_info: run_in_main_thread(self._handle_window_progress, window_info),
                stop_event=stop_event,
            )

        run_in_thread(
            _run,
            on_result=lambda summary: run_in_main_thread(self._handle_run_success, summary),
            on_error=lambda message: run_in_main_thread(self._handle_run_error, message),
            on_progress=lambda value: run_in_main_thread(self._handle_run_progress, value),
            on_intermediate_result=lambda run: run_in_main_thread(self._handle_partial_run, run),
            owner=self,
            key="forecast_run",
        )

    # @ai(gpt-5, codex, refactor, 2026-02-28)
    def _handle_run_success(self, summary: ForecastSummary) -> None:
        self._finalise_thread()
        clear_progress()
        try:
            count = len(getattr(summary, "runs", []) or [])
        except Exception:
            count = 0
        if count > 0:
            message = f"Forecasting finished ({count} runs)."
            status_text = "Forecasting finished."
            toast_fn = toast_success
        else:
            message = "No forecasting runs were produced."
            status_text = "Forecasting produced no results."
            toast_fn = toast_warn
        try:
            toast_fn(message, title="Forecasting", tab_key="forecasting")
        except Exception:
            logger.warning("Exception in _handle_run_success", exc_info=True)
        self._handle_status_update(status_text)
        self.run_finished.emit(summary)

    def _handle_partial_run(self, run) -> None:
        if run is None:
            return
        self.run_partial.emit(run)

    # @ai(gpt-5, codex, refactor, 2026-02-28)
    def _handle_run_error(self, message: str) -> None:
        self._finalise_thread()
        text = str(message).strip() if message else "Unknown error"
        is_cancelled = text.lower() == "forecast run cancelled"
        if is_cancelled:
            status_text = "Forecasting cancelled."
            self._log_info(status_text)
            try:
                toast_info(status_text, title="Forecasting", tab_key="forecasting")
            except Exception:
                logger.warning("Exception in _handle_run_error", exc_info=True)
        else:
            status_text = f"Forecasting failed: {text}"
            self._log_error(status_text)
            try:
                toast_error(text, title="Forecasting failed", tab_key="forecasting")
            except Exception:
                logger.warning("Exception in _handle_run_error", exc_info=True)
        self._handle_status_update(status_text)
        clear_progress()
        self.run_failed.emit(status_text)

    def _handle_run_progress(self, value) -> None:
        try:
            percent = int(value)
        except Exception:
            percent = 0
        percent = max(0, min(100, percent))
        self.run_progress.emit(percent)
        run_in_main_thread(set_progress, percent)

    def _handle_status_update(self, message: Optional[str]) -> None:
        text = str(message).strip() if message else ""
        if text == self._last_status_message:
            return
        self._last_status_message = text
        if text:
            self._log_info(text)
        try:
            set_status_text(text)
        except Exception:
            logger.warning("Exception in _handle_status_update", exc_info=True)
        self.status_changed.emit(text)

    def _handle_service_status_update(self, message: Optional[str]) -> None:
        text = str(message or "").strip()
        if text:
            self._log_info(text)
        if self._running:
            self._handle_status_update("Generating forecasts...")

    def _handle_window_progress(self, window_info: dict) -> None:
        """Handle window-level progress updates during forecasting.
        
        Args:
            window_info: Dictionary containing window progress information:
                - window_index: Current window number
                - model: Model label
                - feature: Feature label
                - train_size: Number of training samples
                - test_size: Number of test samples
                - status: Human-readable status message
        """
        try:
            self.run_window_progress.emit(window_info)
        except Exception as exc:
            logger = logging.getLogger(__name__)
            logger.warning(f"Error handling window progress: {exc}")

    def _finalise_thread(self) -> None:
        self._running = False
        self._last_status_message = None

    def _log_info(self, message: str) -> None:
        self._log(message)

    def _log_error(self, message: str) -> None:
        self._log(message, level="error")

    def _log(self, message: str, *, level: str = "info") -> None:
        levels = {
            "info": logging.INFO,
            "error": logging.ERROR,
        }
        lvl = levels.get(level, logging.INFO)
        try:
            get_log_view_model().log_message(message, level=lvl, origin="forecasting")
        except Exception:
            logger.warning("Exception in _log", exc_info=True)


def _safe_feature_id(payload: object) -> Optional[int]:
    if not isinstance(payload, Mapping):
        return None
    fid = payload.get("feature_id")
    try:
        return int(fid) if fid is not None else None
    except Exception:
        return None


def _build_forecast_results(run: ForecastRunResult) -> list[Mapping[str, object]]:
    results: list[Mapping[str, object]] = []
    metrics = run.metrics or {}
    for metric_name in ("rmse", "mae", "mape"):
        if metric_name in metrics:
            results.append(
                {
                    "stage": "test",
                    "metric_name": metric_name,
                    "metric_value": metrics.get(metric_name),
                }
            )
    return results


def _forecast_run_from_details(details: Mapping[str, object]) -> Optional[ForecastRunResult]:
    try:
        artifacts = details.get("artifacts") or {}
        parameters = details.get("parameters") or {}
        display = parameters.get("display") or {}
        model_label = display.get("model_label") or details.get("name") or details.get("algorithm_key")
        feature_label = display.get("feature_label") or ""
        if not feature_label:
            feature = parameters.get("feature") or {}
            feature_label = feature.get("label") or feature.get("base_name") or ""
        key = parameters.get("run_key") or f"db:{details.get('model_id')}"

        metrics = _forecast_metrics_from_results(details.get("results") or [])

        metadata = {
            "feature": parameters.get("feature") or {},
            "target": parameters.get("target") or {},
            "filters": details.get("filters") or {},
            "preprocessing": details.get("preprocessing") or {},
            "hyperparameters": details.get("hyperparameters") or {},
            "forecast": parameters.get("forecast") or {},
            "metrics": metrics,
            "progress_frame": artifacts.get("progress_frame") or [],
            "forecast_frame": artifacts.get("forecast_frame") or [],
            "model_label": model_label,
            "feature_label": feature_label,
        }

        return ForecastRunResult(
            key=str(key),
            model_key=str(details.get("algorithm_key") or ""),
            model_label=str(model_label),
            feature_key=str((parameters.get("feature") or {}).get("feature_id") or ""),
            feature_label=str(feature_label),
            metrics=metrics,
            progress_frame=records_to_frame(artifacts.get("progress_frame") or []),
            forecast_frame=records_to_frame(artifacts.get("forecast_frame") or []),
            model_id=int(details.get("model_id")),
            metadata=metadata,
        )
    except Exception:
        return None


def _forecast_metrics_from_results(results: Sequence[Mapping[str, object]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in results:
        if row.get("stage") != "test":
            continue
        metric = row.get("metric_name")
        value = row.get("metric_value")
        if metric and value is not None:
            metrics[str(metric)] = float(value)
    return metrics


__all__ = ["ForecastingViewModel"]
