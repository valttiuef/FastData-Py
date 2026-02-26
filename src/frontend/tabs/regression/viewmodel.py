
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence

import pandas as pd
from PySide6.QtCore import QObject, Signal

from backend.services.feature_selection_service import FeatureSelectionService
from backend.services.modeling_shared import display_name
from backend.services.regression_service import RegressionRunResult, RegressionService, RegressionSummary

from ...models.hybrid_pandas_model import HybridPandasModel
from ...models.log_model import LogModel, get_log_model
from ...utils.model_persistence import frame_to_records, normalize_for_json, records_to_frame

from ...threading.runner import run_in_thread
from ...threading.utils import run_in_main_thread
from ...utils import clear_progress, set_progress, toast_error, toast_info, toast_success

logger = logging.getLogger(__name__)


class RegressionViewModel(QObject):
    """Qt-friendly interface for regression experiments used by the tab UI."""

    database_changed = Signal(object)
    run_requested = Signal()
    features_changed = Signal()
    run_started = Signal()
    run_progress = Signal(int)
    run_partial = Signal(object)
    run_finished = Signal(object)
    run_failed = Signal(str)
    status_changed = Signal(str)
    run_context_changed = Signal(object)
    save_completed = Signal(object)

    def __init__(
        self,
        data_model: HybridPandasModel,
        parent: Optional[QObject] = None,
        *,
        log_model: Optional[LogModel] = None,
    ):
        """
        Parameters
        ----------
        data_model:
            Shared HybridPandasModel instance providing both DB access helpers
            (list_systems, features_for_systems_datasets, etc.) and
            preprocessed time-series data for regression.
        """
        super().__init__(parent)
        self._data_model: HybridPandasModel = data_model
        self._service: Optional[RegressionService] = None
        self._feature_service: Optional[FeatureSelectionService] = None
        self._running = False
        self._save_running = False
        self._save_predictions_running = False
        self._pending_auto_saves: dict[str, tuple[RegressionRunResult, dict[str, object]]] = {}
        self._last_status_message: Optional[str] = None
        self._log_model: LogModel = log_model or get_log_model()

        # React to DB path changes inside the HybridPandasModel
        self._data_model.database_changed.connect(self._on_database_changed)
        self._on_database_changed(self._data_model.path)

    # ------------------------------------------------------------------
    @property
    def data_model(self) -> HybridPandasModel:
        return self._data_model

    # ------------------------------------------------------------------
    def request_run(self) -> None:
        """Notify listeners that a regression run has been requested."""
        self.run_requested.emit()

    def notify_features_changed(self) -> None:
        """Notify listeners that feature selections have changed."""
        self.features_changed.emit()

    # ------------------------------------------------------------------
    def _close_database(self) -> None:
        """Internal reset when DB changes or is closed."""
        self._running = False
        self._service = None
        self._feature_service = None
        self._last_status_message = None
        # NOTE: we do NOT clear self._data_model – it is owned by the caller.

    def close_database(self) -> None:
        """Release held regression-related resources and notify listeners."""
        self._close_database()
        self.database_changed.emit(None)

    def _on_database_changed(self, _path) -> None:
        """
        Called when the underlying DB path in HybridPandasModel changes.
        Rebuilds the RegressionService while continuing to use the shared
        HybridPandasModel for data access.
        """
        self._close_database()
        try:
            db = self._data_model.db
            self._feature_service = FeatureSelectionService()
            self._service = RegressionService(db, feature_selection_service=self._feature_service)
        except Exception:
            self._service = None
            self.database_changed.emit(None)
            return
        self.database_changed.emit(db)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def available_models(self) -> list[tuple[str, str, dict[str, object]]]:
        if not self._service:
            return []
        return self._service.available_models()

    def available_feature_selectors(self) -> list[tuple[str, str, dict[str, object]]]:
        if not self._service:
            return []
        return self._service.available_feature_selectors()

    def available_dimensionality_reducers(self) -> list[tuple[str, str, dict[str, object]]]:
        if not self._service:
            return []
        return self._service.available_dimensionality_reducers()

    def available_cv_strategies(self) -> list[tuple[str, str]]:
        if not self._service:
            return []
        return self._service.available_cv_strategies()

    def available_test_strategies(self) -> list[tuple[str, str]]:
        if not self._service:
            return []
        return self._service.available_test_strategies()

    def available_group_kinds(self) -> list[tuple[str, str]]:
        try:
            kinds = self._data_model.db.list_group_kinds()
        except Exception:
            return []
        if not kinds:
            return []
        return [(str(k), str(k)) for k in kinds if str(k)]

    def save_run(
        self,
        run: RegressionRunResult,
        context: Mapping[str, object],
    ) -> Optional[int]:
        if not self._service:
            raise RuntimeError("Database is not initialised")
        if run is None:
            return None
        db = self._data_model.db

        inputs = context.get("inputs") or []
        target = context.get("target") or {}
        stratify = context.get("stratify") or {}

        features: list[tuple[int, str]] = []
        for payload in inputs:
            fid = _safe_feature_id(payload)
            if fid is not None:
                features.append((fid, "input"))
        target_id = _safe_feature_id(target)
        if target_id is not None:
            features.append((target_id, "target"))
        stratify_id = _safe_feature_id(stratify)
        if stratify_id is not None and stratify_id not in {fid for fid, _ in features}:
            features.append((stratify_id, "stratify"))

        model_params = context.get("model_params") or {}
        selector_params = context.get("selector_params") or {}
        reducer_config = (
            context.get("dimensionality_reduction")
            or context.get("pca")
            or {}
        )

        model_name = context.get("model_label") or run.model_label
        selector_label = context.get("selector_label") or run.selector_label
        reducer_label = context.get("reducer_label") or run.reducer_label
        display_parts = [str(model_name)]
        if selector_label:
            display_parts.append(str(selector_label))
        if reducer_label:
            display_parts.append(str(reducer_label))
        display_name = " + ".join(display_parts)

        results = _build_regression_results(run)

        metadata = getattr(run, "metadata", None)
        row_counts: dict[str, object] = {}
        inputs_selected: Optional[int] = None
        inputs_selected_names: Optional[list[str]] = None
        inputs_total: Optional[int] = None
        if isinstance(metadata, dict):
            if "rows_total" in metadata:
                row_counts["total"] = metadata.get("rows_total")
            if "rows_train" in metadata:
                row_counts["train"] = metadata.get("rows_train")
            if "rows_test" in metadata:
                row_counts["test"] = metadata.get("rows_test")
            if "inputs_selected" in metadata:
                try:
                    inputs_selected = int(metadata.get("inputs_selected"))  # type: ignore[arg-type]
                except Exception:
                    inputs_selected = None
            if "inputs_selected_names" in metadata and isinstance(metadata.get("inputs_selected_names"), list):
                inputs_selected_names = [str(v) for v in metadata.get("inputs_selected_names") if str(v)]
            if "inputs_total" in metadata:
                try:
                    inputs_total = int(metadata.get("inputs_total"))  # type: ignore[arg-type]
                except Exception:
                    inputs_total = None

        artifacts = {
            "progress_frame": frame_to_records(run.progress_frame),
            "timeline_frame": frame_to_records(run.timeline_frame),
            "scatter_frame": frame_to_records(run.scatter_frame),
            "metrics": normalize_for_json(run.metrics),
            "cv_scores": normalize_for_json(run.cv_scores),
        }

        parameters = {
            "run_key": run.key,
            "trained_at": str(getattr(run, "trained_at", "") or ""),
            "display": {
                "model_label": model_name,
                "selector_label": selector_label,
                "reducer_label": reducer_label,
            },
            "row_counts": normalize_for_json(row_counts),
            "inputs_selected": inputs_selected,
            "inputs_selected_names": normalize_for_json(inputs_selected_names),
            "inputs_total": inputs_total,
            "inputs": normalize_for_json(inputs),
            "target": normalize_for_json(target),
            "stratify": normalize_for_json(stratify),
            "split": normalize_for_json(context.get("split") or {}),
            "dimensionality_reduction": normalize_for_json(reducer_config),
            "pca": normalize_for_json(reducer_config),
        }

        model_id = db.save_model_run(
            name=display_name,
            model_type="regression",
            algorithm_key=run.model_key,
            selector_key=run.selector_key,
            preprocessing=normalize_for_json(context.get("preprocessing") or {}),
            filters=normalize_for_json(context.get("filters") or {}),
            hyperparameters={
                "model_params": normalize_for_json(model_params),
                "selector_params": normalize_for_json(selector_params),
                "reducer_params": normalize_for_json(context.get("reducer_params") or {}),
                "dimensionality_reduction": normalize_for_json(reducer_config),
                "pca": normalize_for_json(reducer_config),
            },
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

    def start_save_predictions(
        self,
        *,
        feature: Mapping[str, object],
        measurements: pd.DataFrame,
        source_feature_id: Optional[int] = None,
        on_finished=None,
        on_error=None,
    ) -> None:
        if self._save_predictions_running:
            try:
                toast_info("Prediction save already in progress.", title="Regression", tab_key="regression")
            except Exception:
                logger.warning("Exception in start_save_predictions", exc_info=True)
            return
        self._save_predictions_running = True
        try:
            toast_info("Saving predictions to database...", title="Regression", tab_key="regression")
        except Exception:
            logger.warning("Exception in start_save_predictions", exc_info=True)

        payload = dict(feature or {})
        frame = measurements.copy() if isinstance(measurements, pd.DataFrame) else pd.DataFrame()

        def _save() -> dict[str, object]:
            return self._data_model.save_feature_with_measurements(
                feature=payload,
                measurements=frame,
                source_feature_id=source_feature_id,
            )

        def _handle_success(result: dict[str, object]) -> None:
            self._save_predictions_running = False
            feature_payload = result.get("feature")
            if isinstance(feature_payload, dict):
                try:
                    self._data_model.notify_features_changed(new_features=[feature_payload])
                except Exception:
                    logger.warning("Exception in _handle_success", exc_info=True)
            feature_name = str(payload.get("base_name") or "")
            if isinstance(feature_payload, dict):
                feature_name = str(feature_payload.get("base_name") or feature_name)
            measurement_count = int(result.get("measurement_count") or 0)
            try:
                toast_success(
                    f"Saved {measurement_count} prediction rows as '{feature_name}'.",
                    title="Regression saved",
                    tab_key="regression",
                )
            except Exception:
                logger.warning("Exception in _handle_success", exc_info=True)
            if callable(on_finished):
                on_finished(result)

        def _handle_error(message: str) -> None:
            self._save_predictions_running = False
            text = str(message).strip() if message else "Unknown error"
            try:
                toast_error(text, title="Regression save failed", tab_key="regression")
            except Exception:
                logger.warning("Exception in _handle_error", exc_info=True)
            if callable(on_error):
                on_error(text)

        run_in_thread(
            _save,
            on_result=lambda result: run_in_main_thread(_handle_success, result),
            on_error=lambda message: run_in_main_thread(_handle_error, message),
            owner=self,
            key="regression_save_predictions",
            cancel_previous=True,
        )

    def load_saved_runs(self) -> list[RegressionRunResult]:
        if not self._service:
            return []
        db = self._data_model.db
        saved = []
        try:
            models = db.list_models(model_type="regression")
        except Exception:
            return []
        for _, row in models.iterrows():
            model_id = int(row.get("model_id"))
            details = db.fetch_model(model_id)
            if not details:
                continue
            run = _regression_run_from_details(details)
            if run is not None:
                saved.append(run)
        return saved

    # ------------------------------------------------------------------
    def start_save_runs(
        self,
        runs: Sequence[RegressionRunResult],
        contexts: Mapping[str, Mapping[str, object]],
        *,
        auto: bool = False,
    ) -> None:
        if not self._service:
            raise RuntimeError("Database is not initialised")
        pending: list[tuple[RegressionRunResult, dict[str, object]]] = []
        for run in runs:
            if run is None or getattr(run, "model_id", None):
                continue
            context = contexts.get(run.key) if isinstance(contexts, Mapping) else None
            if not isinstance(context, Mapping):
                continue
            pending.append((run, self._ensure_context_labels(run, dict(context))))

        if not pending:
            if not auto:
                try:
                    toast_info("Selected runs are already saved.", title="Regression", tab_key="regression")
                except Exception:
                    logger.warning("Exception in start_save_runs", exc_info=True)
            return

        if self._save_running:
            if auto:
                for run, context in pending:
                    self._pending_auto_saves[str(run.key)] = (run, context)
                return
            try:
                toast_info("Regression save already in progress.", title="Regression", tab_key="regression")
            except Exception:
                logger.warning("Exception in start_save_runs", exc_info=True)
            return

        self._start_save_thread(pending, auto=auto)

    def queue_auto_save(self, run: RegressionRunResult, context: Mapping[str, object]) -> None:
        if run is None or getattr(run, "model_id", None):
            return
        if not isinstance(context, Mapping):
            return
        payload = self._ensure_context_labels(run, dict(context))
        self._pending_auto_saves[str(run.key)] = (run, payload)
        if self._save_running:
            return
        batch = list(self._pending_auto_saves.values())
        self._pending_auto_saves.clear()
        if batch:
            self._start_save_thread(batch, auto=True)

    def _start_save_thread(
        self,
        batch: Sequence[tuple[RegressionRunResult, dict[str, object]]],
        *,
        auto: bool,
    ) -> None:
        self._save_running = True
        count = len(batch)
        if auto:
            label = "model" if count == 1 else "models"
            try:
                toast_info(f"Auto-saving regression {label}...", title="Regression", tab_key="regression")
            except Exception:
                logger.warning("Exception in _start_save_thread", exc_info=True)
        else:
            try:
                toast_info("Saving regression models...", title="Regression", tab_key="regression")
            except Exception:
                logger.warning("Exception in _start_save_thread", exc_info=True)

        def _save_batch() -> dict[str, object]:
            saved: list[dict[str, object]] = []
            failures: list[str] = []
            for run, context in batch:
                try:
                    model_id = self.save_run(run, context)
                except Exception:
                    failures.append(str(run.key))
                    continue
                if model_id is not None:
                    saved.append({"key": run.key, "model_id": int(model_id), "context": context})
                else:
                    failures.append(str(run.key))
            return {"saved": saved, "failures": failures, "total": len(batch), "auto": auto}

        run_in_thread(
            _save_batch,
            on_result=lambda payload: run_in_main_thread(self._handle_save_success, payload),
            on_error=lambda message: run_in_main_thread(self._handle_save_error, message, auto),
            owner=self,
            key="regression_save_models",
            cancel_previous=not auto,
        )

    def _handle_save_success(self, payload: Mapping[str, object]) -> None:
        self._save_running = False
        saved = payload.get("saved") or []
        failures = payload.get("failures") or []
        total = int(payload.get("total") or 0)
        auto = bool(payload.get("auto"))
        saved_count = len(saved)
        label = "model" if total == 1 else "models"
        if auto:
            if saved_count:
                try:
                    toast_success("Regression model auto-saved.", title="Regression saved", tab_key="regression")
                except Exception:
                    logger.warning("Exception in _handle_save_success", exc_info=True)
            else:
                try:
                    toast_error("Auto-save failed.", title="Regression save failed", tab_key="regression")
                except Exception:
                    logger.warning("Exception in _handle_save_success", exc_info=True)
        else:
            if failures:
                try:
                    toast_error(
                        f"Saved {saved_count} of {total} regression {label}.",
                        title="Regression save failed",
                        tab_key="regression",
                    )
                except Exception:
                    logger.warning("Exception in _handle_save_success", exc_info=True)
            else:
                try:
                    toast_success(
                        f"Saved {saved_count} regression {label}.",
                        title="Regression saved",
                        tab_key="regression",
                    )
                except Exception:
                    logger.warning("Exception in _handle_save_success", exc_info=True)
        self.save_completed.emit(dict(payload))

        if self._pending_auto_saves:
            batch = list(self._pending_auto_saves.values())
            self._pending_auto_saves.clear()
            if batch:
                self._start_save_thread(batch, auto=True)

    def _handle_save_error(self, message: str, auto: bool) -> None:
        self._save_running = False
        text = str(message).strip() if message else "Unknown error"
        title = "Regression save failed"
        if auto:
            try:
                toast_error("Auto-save failed.", title=title, tab_key="regression")
            except Exception:
                logger.warning("Exception in _handle_save_error", exc_info=True)
        else:
            try:
                toast_error(text, title=title, tab_key="regression")
            except Exception:
                logger.warning("Exception in _handle_save_error", exc_info=True)
        if self._pending_auto_saves:
            batch = list(self._pending_auto_saves.values())
            self._pending_auto_saves.clear()
            if batch:
                self._start_save_thread(batch, auto=True)

    def run_regressions(
        self,
        *,
        data_frame: pd.DataFrame,
        input_features: Sequence[Mapping[str, object]],
        target_features: Sequence[Mapping[str, object]],
        selectors: Sequence[str],
        models: Sequence[str],
        systems: Optional[Sequence[str]] = None,
        Datasets: Optional[Sequence[str]] = None,
        groups: Optional[Sequence[int]] = None,
        start=None,
        end=None,
        selector_params: Optional[Mapping[str, Mapping[str, object]]] = None,
        model_params: Optional[Mapping[str, Mapping[str, object]]] = None,
        reducers: Optional[Sequence[str]] = None,
        reducer_params: Optional[Mapping[str, Mapping[str, object]]] = None,
        cv_strategy: str = "none",
        cv_folds: int = 5,
        cv_group_kind: Optional[str] = None,
        shuffle: bool = True,
        random_state: int = 0,
        test_size: Optional[float] = 0.2,
        test_strategy: str = "random",
        stratify_bins: int = 5,
        time_series_gap: int = 0,
        preprocessing: Optional[Mapping[str, object]] = None,
        stratify_feature: Optional[Mapping[str, object]] = None,
        target_contexts: Optional[Sequence[Mapping[str, object]]] = None,
        context_callback=None,
        progress_callback=None,
        status_callback=None,
        result_callback=None,
        stop_event=None,
    ) -> RegressionSummary:
        if not self._service:
            raise RuntimeError("Database is not initialised")
        if not isinstance(data_frame, pd.DataFrame):
            raise RuntimeError(
                "RegressionViewModel requires a pre-fetched DataFrame from DataSelectorWidget."
            )

        input_payloads = [dict(p) for p in input_features]
        stratify_payload = dict(stratify_feature) if stratify_feature else None

        targets = [dict(t) for t in target_features if t]
        if not targets:
            raise ValueError("Select at least one target feature")

        total_targets = len(targets)
        reducer_list = list(reducers or ["none"])
        if not reducer_list:
            reducer_list = ["none"]
        total_runs_per_target = len(selectors) * len(reducer_list) * len(models)
        total_runs = total_runs_per_target * total_targets if total_runs_per_target else total_targets

        all_runs: list[RegressionRunResult] = []
        target_failures: list[str] = []

        for idx, target_payload in enumerate(targets):
            if target_contexts and idx < len(target_contexts) and context_callback is not None:
                try:
                    run_in_main_thread(context_callback, target_contexts[idx])
                except Exception:
                    logger.warning("Exception in run_regressions", exc_info=True)

            target_data_frame = data_frame.copy()

            def _progress_adapter(pct: int, *, _idx=idx) -> None:
                if progress_callback is None:
                    return
                try:
                    per_target = max(1, total_runs_per_target)
                    completed = _idx * per_target
                    overall = int(round(((completed + (pct / 100) * per_target) / max(1, total_runs)) * 100))
                    progress_callback(overall)
                except Exception:
                    logger.warning("Exception in _progress_adapter", exc_info=True)

            target_label = str(display_name(target_payload)).strip() or f"target-{idx + 1}"
            try:
                summary = self._service.run_regressions(
                    input_features=input_payloads,
                    target_feature=target_payload,
                    selectors=selectors,
                    models=models,
                    systems=systems,
                    Datasets=Datasets,
                    start=start,
                    end=end,
                    selector_params=selector_params,
                    model_params=model_params,
                    reducers=reducers,
                    reducer_params=reducer_params,
                    cv_strategy=cv_strategy,
                    cv_folds=cv_folds,
                    cv_group_kind=cv_group_kind,
                    shuffle=shuffle,
                    random_state=random_state,
                    test_size=test_size,
                    test_strategy=test_strategy,
                    stratify_bins=stratify_bins,
                    time_series_gap=time_series_gap,
                    progress_callback=_progress_adapter,
                    data_frame=target_data_frame,
                    stratify_feature=stratify_payload,
                    status_callback=status_callback,
                    result_callback=result_callback,
                    stop_event=stop_event,
                )
            except Exception as exc:
                text = str(exc).strip() if exc else "Unknown regression error"
                if text.lower() == "regression run cancelled":
                    raise
                target_failures.append(f"{target_label}: {text}")
                if status_callback is not None:
                    try:
                        status_callback(f"Skipped target {target_label}: {text}")
                    except Exception:
                        logger.warning("Exception in _progress_adapter", exc_info=True)
                continue

            target_runs = getattr(summary, "runs", []) or []
            if target_runs:
                all_runs.extend(target_runs)
                continue
            target_failures.append(f"{target_label}: no runs produced")

        if not all_runs:
            detail = "; ".join(target_failures[:3]) if target_failures else "No runs were produced."
            raise RuntimeError(f"Regression run produced no successful models. {detail}")
        return RegressionSummary(runs=all_runs)

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._close_database()

    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        return self._running

    def start_regressions(
        self,
        *,
        data_frame: pd.DataFrame,
        input_features: Sequence[Mapping[str, object]],
        target_features: Sequence[Mapping[str, object]],
        target_contexts: Optional[Sequence[Mapping[str, object]]] = None,
        selectors: Sequence[str],
        models: Sequence[str],
        systems: Optional[Sequence[str]] = None,
        Datasets: Optional[Sequence[str]] = None,
        groups: Optional[Sequence[int]] = None,
        start=None,
        end=None,
        selector_params: Optional[Mapping[str, Mapping[str, object]]] = None,
        model_params: Optional[Mapping[str, Mapping[str, object]]] = None,
        reducers: Optional[Sequence[str]] = None,
        reducer_params: Optional[Mapping[str, Mapping[str, object]]] = None,
        cv_strategy: str = "none",
        cv_folds: int = 5,
        cv_group_kind: Optional[str] = None,
        shuffle: bool = True,
        random_state: int = 0,
        test_size: Optional[float] = 0.2,
        test_strategy: str = "random",
        stratify_bins: int = 5,
        time_series_gap: int = 0,
        preprocessing: Optional[Mapping[str, object]] = None,
        stratify_feature: Optional[Mapping[str, object]] = None,
    ) -> None:
        if not self._service:
            raise RuntimeError("Database is not initialised")
        if self.is_running():
            raise RuntimeError("Regression run already in progress")

        self._last_status_message = None
        try:
            toast_info("Starting regression experiments…", title="Regression", tab_key="regression")
        except Exception:
            logger.warning("Exception in start_regressions", exc_info=True)
        self.run_started.emit()
        self._handle_status_update("Running regression experiments…")
        set_progress(0)
        self._running = True

        def _run(*, progress_callback=None, stop_event=None, result_callback=None):
            return self.run_regressions(
                data_frame=data_frame,
                input_features=input_features,
                target_features=target_features,
                selectors=selectors,
                models=models,
                systems=systems,
                Datasets=Datasets,
                groups=groups,
                start=start,
                end=end,
                selector_params=selector_params,
                model_params=model_params,
                reducers=reducers,
                reducer_params=reducer_params,
                cv_strategy=cv_strategy,
                cv_folds=cv_folds,
                cv_group_kind=cv_group_kind,
                shuffle=shuffle,
                random_state=random_state,
                test_size=test_size,
                test_strategy=test_strategy,
                stratify_bins=stratify_bins,
                time_series_gap=time_series_gap,
                preprocessing=preprocessing,
                stratify_feature=stratify_feature,
                target_contexts=target_contexts,
                context_callback=lambda context: run_in_main_thread(
                    self._handle_context_update, context
                ),
                progress_callback=progress_callback,
                status_callback=lambda message: run_in_main_thread(
                    self._handle_status_update, message
                ),
                result_callback=result_callback,
                stop_event=stop_event,
            )

        run_in_thread(
            _run,
            on_result=lambda summary: run_in_main_thread(self._handle_run_success, summary),
            on_error=lambda message: run_in_main_thread(self._handle_run_error, message),
            on_progress=lambda value: run_in_main_thread(self._handle_run_progress, value),
            on_intermediate_result=lambda run: run_in_main_thread(self._handle_partial_run, run),
            owner=self,
            key="regression_run",
        )

    def _handle_run_success(self, summary: RegressionSummary) -> None:
        self._finalise_thread()
        clear_progress()
        try:
            count = len(getattr(summary, "runs", []) or [])
            toast_success(f"Regression finished ({count} runs).", title="Regression", tab_key="regression")
        except Exception:
            try:
                toast_success("Regression finished.", title="Regression", tab_key="regression")
            except Exception:
                logger.warning("Exception in _handle_run_success", exc_info=True)
        self.run_finished.emit(summary)

    def _handle_partial_run(self, run) -> None:
        if run is None:
            return
        self.run_partial.emit(run)

    def _handle_run_error(self, message: str) -> None:
        self._finalise_thread()
        text = str(message).strip() if message else "Unknown error"
        is_cancelled = text.lower() == "regression run cancelled"
        if is_cancelled:
            status_text = "Regression run cancelled."
            try:
                toast_info(status_text, title="Regression", tab_key="regression")
            except Exception:
                logger.warning("Exception in _handle_run_error", exc_info=True)
        else:
            status_text = f"Regression failed: {text}"
            try:
                toast_error(text, title="Regression failed", tab_key="regression")
            except Exception:
                logger.warning("Exception in _handle_run_error", exc_info=True)
        self._last_status_message = status_text
        self.status_changed.emit(status_text)
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
        self.status_changed.emit(text)

    def _handle_context_update(self, context: Optional[Mapping[str, object]]) -> None:
        self.run_context_changed.emit(dict(context) if context else None)

    def _finalise_thread(self) -> None:
        self._running = False
        self._last_status_message = None

    # ------------------------------------------------------------------
    def _log_info(self, message: str) -> None:
        self._log(message)

    def _log(self, message: str, *, level: str = "info") -> None:
        levels = {
            "info": logging.INFO,
            "error": logging.ERROR,
        }
        lvl = levels.get(level, logging.INFO)
        self._log_model.log_text(message, level=lvl, origin="regression")

    def _ensure_context_labels(self, run: RegressionRunResult, context: dict[str, object]) -> dict[str, object]:
        context = dict(context)
        context.setdefault("model_label", run.model_label)
        context.setdefault("selector_label", run.selector_label)
        context.setdefault("reducer_label", getattr(run, "reducer_label", ""))
        return context


def _safe_feature_id(payload: object) -> Optional[int]:
    if not isinstance(payload, Mapping):
        return None
    fid = payload.get("feature_id")
    try:
        return int(fid) if fid is not None else None
    except Exception:
        return None


def _build_regression_results(run: RegressionRunResult) -> list[Mapping[str, object]]:
    results: list[Mapping[str, object]] = []
    metrics = run.metrics or {}
    if "r2_train" in metrics:
        results.append({"stage": "train", "metric_name": "r2", "metric_value": metrics.get("r2_train")})
    if "rmse_train" in metrics:
        results.append({"stage": "train", "metric_name": "rmse", "metric_value": metrics.get("rmse_train")})
    if "r2_test" in metrics:
        results.append({"stage": "test", "metric_name": "r2", "metric_value": metrics.get("r2_test")})
    if "rmse_test" in metrics:
        results.append({"stage": "test", "metric_name": "rmse", "metric_value": metrics.get("rmse_test")})
    cv_scores = run.cv_scores or {}
    for metric_name, values in cv_scores.items():
        for idx, value in enumerate(values):
            results.append(
                {
                    "stage": "cv",
                    "metric_name": metric_name,
                    "metric_value": value,
                    "fold": idx,
                }
            )
    return results


def _regression_run_from_details(details: Mapping[str, object]) -> Optional[RegressionRunResult]:
    try:
        artifacts = details.get("artifacts") or {}
        parameters = details.get("parameters") or {}
        display = parameters.get("display") or {}
        model_label = display.get("model_label") or details.get("name") or details.get("algorithm_key")
        selector_label = display.get("selector_label") or details.get("selector_key") or ""
        display_reducer_label = display.get("reducer_label") or ""
        key = parameters.get("run_key") or f"db:{details.get('model_id')}"

        metrics = _metrics_from_results(details.get("results") or [])
        cv_scores = _cv_scores_from_results(details.get("results") or [])

        hyperparameters = details.get("hyperparameters") or {}
        dim_reduction = (
            parameters.get("dimensionality_reduction")
            or hyperparameters.get("dimensionality_reduction")
            or parameters.get("pca")
            or hyperparameters.get("pca")
            or {}
        )
        reducer_key = str(details.get("reducer_key") or "")
        reducer_label = str(details.get("reducer_label") or display_reducer_label or "")
        if isinstance(dim_reduction, dict):
            method_key = dim_reduction.get("method")
            method_label = dim_reduction.get("label")
            if (not reducer_key) and method_key:
                reducer_key = str(method_key)
            if (not reducer_label) and method_label:
                reducer_label = str(method_label)

        metadata = {
            "trained_at": _to_iso_datetime(parameters.get("trained_at") or details.get("created_at")),
            "row_counts": parameters.get("row_counts") or {},
            "inputs_selected": parameters.get("inputs_selected"),
            "inputs_selected_names": parameters.get("inputs_selected_names") or [],
            "inputs_total": parameters.get("inputs_total"),
            "inputs": parameters.get("inputs") or [],
            "target": parameters.get("target") or {},
            "stratify": parameters.get("stratify") or {},
            "filters": details.get("filters") or {},
            "preprocessing": details.get("preprocessing") or {},
            "hyperparameters": hyperparameters,
            "split": parameters.get("split") or {},
            "dimensionality_reduction": dim_reduction,
            "pca": dim_reduction,
            "metrics": metrics,
            "cv_scores": cv_scores,
            "progress_frame": artifacts.get("progress_frame") or [],
            "timeline_frame": artifacts.get("timeline_frame") or [],
            "scatter_frame": artifacts.get("scatter_frame") or [],
            "model_label": model_label,
            "selector_label": selector_label,
            "reducer_label": reducer_label,
        }
        trained_at = _to_iso_datetime(parameters.get("trained_at") or details.get("created_at"))

        return RegressionRunResult(
            key=str(key),
            model_key=str(details.get("algorithm_key") or ""),
            model_label=str(model_label),
            selector_key=str(details.get("selector_key") or "none"),
            selector_label=str(selector_label),
            reducer_key=reducer_key or "none",
            reducer_label=reducer_label,
            metrics=metrics,
            cv_scores=cv_scores,
            progress_frame=records_to_frame(artifacts.get("progress_frame") or []),
            timeline_frame=records_to_frame(artifacts.get("timeline_frame") or []),
            scatter_frame=records_to_frame(artifacts.get("scatter_frame") or []),
            trained_at=trained_at,
            model_id=int(details.get("model_id")),
            metadata=metadata,
        )
    except Exception:
        return None


def _metrics_from_results(results: Sequence[Mapping[str, object]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in results:
        stage = row.get("stage")
        metric = row.get("metric_name")
        value = row.get("metric_value")
        if value is None:
            continue
        if stage == "train" and metric in {"r2", "rmse"}:
            metrics[f"{metric}_train"] = float(value)
        if stage == "test" and metric in {"r2", "rmse"}:
            metrics[f"{metric}_test"] = float(value)
    return metrics


def _cv_scores_from_results(results: Sequence[Mapping[str, object]]) -> dict[str, list[float]]:
    cv_scores: dict[str, list[float]] = {}
    for row in results:
        if row.get("stage") != "cv":
            continue
        metric = row.get("metric_name")
        if not metric:
            continue
        value = row.get("metric_value")
        if value is None:
            continue
        cv_scores.setdefault(str(metric), []).append(float(value))
    return cv_scores


def _to_iso_datetime(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    text = str(value).strip()
    if not text:
        return ""
    try:
        parsed = pd.to_datetime(text, errors="coerce", utc=True)
        if pd.isna(parsed):
            return text
        return parsed.isoformat()
    except Exception:
        return text
