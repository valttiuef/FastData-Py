
from __future__ import annotations
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QGroupBox, QVBoxLayout
from ..localization import tr

from ..models.hybrid_pandas_model import DataFilters, FeatureSelection, HybridPandasModel
from ..tabs.data.features_info_dialog import FeaturesInfoDialog
from ..utils.feature_details import build_features_prompt
from ..threading.runner import run_in_thread
from ..threading.utils import run_in_main_thread
from .features_list_widget import FeaturesListWidget
from .filters_widget import FiltersWidget
from .preprocessing_widget import PreprocessingWidget
from ..viewmodels.help_viewmodel import HelpViewModel
import logging

logger = logging.getLogger(__name__)


def _format_date(ts) -> str:
    if ts is None:
        return "?"
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return "?"


def _append_correlation_table(lines: list[str], df: pd.DataFrame, columns: Sequence[str]) -> None:
    if df.empty:
        return
    valid_columns = [col for col in columns if col in df.columns]
    if len(valid_columns) < 2:
        return
    try:
        numeric_df = df[valid_columns].apply(pd.to_numeric, errors="coerce")
        corr_df = numeric_df.corr()
    except Exception:
        return
    if corr_df.empty:
        return
    lines.append("\nCorrelation table (Pearson, rounded to 3 decimals):")
    lines.append(corr_df.round(3).to_string())


def _build_feature_summary_text(
    model: HybridPandasModel,
    filters: DataFilters,
    start_ts,
    end_ts,
    *,
    preprocessing: Mapping[str, object] | None = None,
) -> str:
    try:
        params = dict(preprocessing or {})
    except Exception:
        params = {}

    try:
        model.load_base(filters, **params)
    except Exception:
        logger.warning("Exception in _build_feature_summary_text", exc_info=True)

    df = model.base_dataframe()
    columns = [c for c in df.columns if c != "t"]

    try:
        col_map = model._feature_column_names(filters.features, df)  # type: ignore[attr-defined]
    except Exception:
        col_map = {}
    reverse_map = {name: key for key, name in col_map.items()}

    s = _format_date(start_ts)
    e = _format_date(end_ts)
    lines: list[str] = [f"Date range: {s} -> {e}"]

    if not columns:
        lines.append("No feature values are available for the current selection.")
        return "\n".join(lines)

    for col in columns:
        sel = None
        key = reverse_map.get(col)
        if key is not None:
            for item in filters.features:
                if item.identity_key() == key:
                    sel = item
                    break

        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            stats_text = "no numeric values in the visible range"
        else:
            stats_text = (
                f"count={len(valid)}, mean={valid.mean():.3g}, min={valid.min():.3g}, "
                f"max={valid.max():.3g}, std={valid.std():.3g}"
            )

        meta_parts: list[str] = []
        if sel is not None:
            for attr, label in (
                ("base_name", "base"),
                ("source", "source"),
                ("unit", "unit"),
                ("type", "type"),
            ):
                val = getattr(sel, attr, None)
                if val:
                    meta_parts.append(f"{label}={val}")
            if getattr(sel, "feature_id", None) is not None:
                meta_parts.append(f"id={sel.feature_id}")

        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
        lines.append(f"- {col}{meta}: {stats_text}")

    _append_correlation_table(lines, df, columns)
    lines.append("\nAdd any extra context or questions here before sending to the AI.")
    return "\n".join(lines)



class DataSelectorViewModel(QObject):
    """View-model that keeps :class:`DataSelectorWidget` in sync with a model."""

    filters_changed = Signal(dict)
    preprocessing_changed = Signal(dict)
    features_selection_changed = Signal(list)
    data_requirements_changed = Signal(dict)

    def __init__(
        self,
        *,
        widget: "DataSelectorWidget",
        data_model: Optional[HybridPandasModel] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._widget = widget
        self._data_model: Optional[HybridPandasModel] = None
        self._selection_refresh_pending = False
        self._selection_refresh_waiting_features_reload = False
        self._widget.filters_changed.connect(self.filters_changed)
        self._widget.preprocessing_changed.connect(self.preprocessing_changed)
        self._widget.features_selection_changed.connect(self.features_selection_changed)
        self._widget.data_requirements_changed.connect(self.data_requirements_changed)
        if self._widget.filters_widget is not None:
            self._widget.filters_widget.filters_refreshed.connect(self._apply_selection_state_after_filters_refresh)
        if self._widget.features_widget is not None:
            self._widget.features_widget.features_reloaded.connect(self._on_features_reloaded)
        self._data_model = data_model
        if self._data_model is not None:
            self._data_model.database_changed.connect(self._on_database_changed)
            self._data_model.selection_state_changed.connect(self._on_selection_state_changed)
        self.refresh_from_model()

    def refresh_from_model(self) -> None:
        """Refresh filter choices and feature list from the current model."""
        if self._widget.filters_widget is not None:
            try:
                self._widget.filters_widget.refresh_filters()
            except Exception:
                logger.warning("Exception in refresh_from_model", exc_info=True)
        try:
            self._apply_selection_state_to_widget()
        except Exception:
            logger.warning("Exception in refresh_from_model", exc_info=True)
        if self._widget.features_widget is not None:
            try:
                self._widget.features_widget.reload_features()
            except Exception:
                logger.warning("Exception in refresh_from_model", exc_info=True)

    def _on_database_changed(self, *_args) -> None:
        if self._selection_refresh_pending:
            self._finish_selection_refresh()
        if self._widget.features_widget is not None:
            try:
                self._widget.features_widget.clear_selection_and_suppress_autoselect()
            except Exception:
                logger.warning("Exception in _on_database_changed", exc_info=True)

    def _on_selection_state_changed(self) -> None:
        """Apply saved selection filters/preprocessing and refresh affected widgets."""
        already_pending = self._selection_refresh_pending
        self._selection_refresh_pending = True
        self._selection_refresh_waiting_features_reload = False
        if not already_pending:
            self._widget.begin_feature_reload_batch()
            self._widget.begin_data_requirements_batch()
        if self._widget.filters_widget is not None:
            try:
                self._widget.filters_widget.refresh_filters()
            except Exception:
                logger.warning("Exception in _on_selection_state_changed", exc_info=True)
                self._apply_selection_state_after_filters_refresh()
        else:
            self._apply_selection_state_after_filters_refresh()

    def _apply_selection_state_after_filters_refresh(self) -> None:
        if not self._selection_refresh_pending:
            return
        self._apply_selection_state_to_widget()
        if self._widget.features_widget is None:
            self._finish_selection_refresh()
            return
        if self._widget.features_widget.use_selection_filter:
            # Selection-filtered feature lists are refreshed through selected_features_changed.
            self._finish_selection_refresh()
            return
        try:
            self._selection_refresh_waiting_features_reload = True
            self._widget.features_widget.reload_features()
        except Exception:
            logger.warning("Exception in _apply_selection_state_after_filters_refresh", exc_info=True)
            self._finish_selection_refresh()

    def _on_features_reloaded(self, _df: object) -> None:
        if not self._selection_refresh_pending or not self._selection_refresh_waiting_features_reload:
            return
        self._finish_selection_refresh()

    def _finish_selection_refresh(self) -> None:
        if not self._selection_refresh_pending:
            return
        self._selection_refresh_pending = False
        self._selection_refresh_waiting_features_reload = False
        self._widget.end_feature_reload_batch(trigger_reload=False)
        self._widget.end_data_requirements_batch()

    def _apply_selection_state_to_widget(self) -> None:
        model = self._data_model
        if model is None:
            return
        try:
            filters_state = dict(getattr(model, "selection_filters", {}) or {})
        except Exception:
            filters_state = {}
        try:
            preprocessing_state = dict(getattr(model, "selection_preprocessing", {}) or {})
        except Exception:
            preprocessing_state = {}

        try:
            self._widget.apply_settings(
                {
                    "filters": filters_state,
                    "preprocessing": preprocessing_state,
                },
                reload_features=False,
            )
        except Exception:
            logger.warning("Exception in _apply_selection_state_to_widget", exc_info=True)

    # ------------------------------------------------------------------
    def build_data_filters(self) -> Optional[DataFilters]:
        return self._widget.build_data_filters()

    def fetch_base_dataframe(
        self, *, preprocessing_override: Optional[Mapping[str, object]] = None
    ) -> Optional[pd.DataFrame]:
        return self._widget.fetch_base_dataframe(
            preprocessing_override=preprocessing_override
        )

    def fetch_base_dataframe_for_features(
        self,
        feature_payloads: Sequence[Mapping[str, object]],
        *,
        preprocessing_override: Optional[Mapping[str, object]] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        group_ids: Optional[Sequence[int]] = None,
    ) -> Optional[pd.DataFrame]:
        return self._widget.fetch_base_dataframe_for_features(
            feature_payloads,
            preprocessing_override=preprocessing_override,
            start=start,
            end=end,
            systems=systems,
            datasets=datasets,
            group_ids=group_ids,
        )


class DataSelectorWidget(QGroupBox):
    """Composite widget bundling preprocessing, filters and features list.

    The widget keeps the features list in sync with the selected filters and
    exposes convenience helpers for building :class:`DataFilters` instances and
    fetching preprocessed data frames from :class:`HybridPandasModel`.
    """

    filters_changed = Signal(dict)
    preprocessing_changed = Signal(dict)
    features_selection_changed = Signal(list)
    data_requirements_changed = Signal(dict)

    def __init__(
        self,
        *,
        title: str = "Data selection",
        parent=None,
        data_model: Optional[HybridPandasModel] = None,
        help_viewmodel: Optional[HelpViewModel] = None,
        show_preprocessing: bool = True,
        show_filters: bool = True,
        show_features_list: bool = True,
    ) -> None:
        super().__init__(tr(title), parent)
        self._data_model = data_model
        self._help_viewmodel = help_viewmodel
        if self._data_model is None:
            logger.warning("DataSelectorWidget initialised without data_model.")
        if self._help_viewmodel is None:
            logger.warning("DataSelectorWidget initialised without help_viewmodel.")
        self._feature_dialog: FeaturesInfoDialog | None = None
        self._requirements_batch_depth = 0
        self._requirements_emit_pending = False
        self._feature_reload_batch_depth = 0
        self._feature_reload_pending = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.preprocessing_widget: PreprocessingWidget | None = None
        self.filters_widget: FiltersWidget | None = None
        self.features_widget: FeaturesListWidget | None = None

        if show_preprocessing:
            self.preprocessing_widget = PreprocessingWidget(
                collapsed=True,
                parent=self,
                help_viewmodel=self._help_viewmodel,
            )
            layout.addWidget(self.preprocessing_widget)

        if show_filters:
            self.filters_widget = FiltersWidget(
                parent=self,
                model=data_model,
                help_viewmodel=self._help_viewmodel,
            )
            layout.addWidget(self.filters_widget)

        if show_features_list:
            self.features_widget = FeaturesListWidget(
                parent=self,
                data_model=data_model,
            )
            self.features_widget.setVisible(True)
            layout.addWidget(self.features_widget, 1)
            self.features_widget.details_requested.connect(self._show_features_info_dialog)

        # Wire inner widget signals to the composite signals
        if self.preprocessing_widget is not None:
            self.preprocessing_widget.parameters_changed.connect(self._on_preprocessing_changed)
        if self.filters_widget is not None:
            # Connect to specific filter signals that affect features (systems, datasets, tags)
            self.filters_widget.systems_changed.connect(self._on_feature_affecting_filters_changed)
            self.filters_widget.datasets_changed.connect(self._on_feature_affecting_filters_changed)
            self.filters_widget.imports_changed.connect(self._on_feature_affecting_filters_changed)
            self.filters_widget.tags_changed.connect(self._on_feature_affecting_filters_changed)
            # Connect to all filter changes for general notification (without reloading features)
            self.filters_widget.filters_changed.connect(self._on_filters_changed)
        if self.features_widget is not None:
            self.features_widget.selection_changed.connect(self._on_features_selection_changed)

        self.view_model = DataSelectorViewModel(widget=self, data_model=data_model, parent=self)

        # Initial propagation
        self._on_feature_affecting_filters_changed()

    # ------------------------------------------------------------------
    def _on_preprocessing_changed(self, params: Mapping[str, Any]) -> None:
        params = dict(params or {})
        self.preprocessing_changed.emit(params)
        self._emit_data_requirements()

    def _on_feature_affecting_filters_changed(self, *_args) -> None:
        """Handle changes to filters that affect the features list (systems, datasets, tags)."""
        if self.features_widget is None or self.filters_widget is None:
            return
        if self._feature_reload_batch_depth > 0:
            self._feature_reload_pending = True
            return
        self.features_widget.set_filters(
            systems=self.filters_widget.selected_systems(),
            datasets=self.filters_widget.selected_datasets(),
            import_ids=self.filters_widget.selected_import_ids(),
            tags=self.filters_widget.selected_tags(),
            reload=True,
        )

    def _on_filters_changed(self, *_args) -> None:
        """Handle all filter changes - emit signals and update data requirements."""
        if self.filters_widget is None:
            self.filters_changed.emit({})
            self._emit_data_requirements()
            return
        self.filters_changed.emit(self.filters_widget.filter_state())
        self._emit_data_requirements()

    def _on_features_selection_changed(self, payloads: list[dict]) -> None:
        self.features_selection_changed.emit(payloads)
        self._emit_data_requirements()

    def _emit_data_requirements(self) -> None:
        if self._requirements_batch_depth > 0:
            self._requirements_emit_pending = True
            return
        self._emit_data_requirements_now()

    def _emit_data_requirements_now(self) -> None:
        filters = self.filters_widget.filter_state() if self.filters_widget is not None else {}
        preprocessing = self.preprocessing_widget.parameters() if self.preprocessing_widget is not None else {}
        features = self.features_widget.selected_payloads() if self.features_widget is not None else []
        self.data_requirements_changed.emit(
            {
                "filters": filters,
                "preprocessing": preprocessing,
                "features": features,
            }
        )

    def begin_data_requirements_batch(self) -> None:
        self._requirements_batch_depth += 1

    def end_data_requirements_batch(self) -> None:
        if self._requirements_batch_depth <= 0:
            self._requirements_batch_depth = 0
            return
        self._requirements_batch_depth -= 1
        if self._requirements_batch_depth == 0 and self._requirements_emit_pending:
            self._requirements_emit_pending = False
            self._emit_data_requirements_now()

    def begin_feature_reload_batch(self) -> None:
        self._feature_reload_batch_depth += 1

    def end_feature_reload_batch(self, *, trigger_reload: bool = True) -> None:
        if self._feature_reload_batch_depth <= 0:
            self._feature_reload_batch_depth = 0
            self._feature_reload_pending = False
            return
        self._feature_reload_batch_depth -= 1
        if self._feature_reload_batch_depth == 0:
            pending = self._feature_reload_pending
            self._feature_reload_pending = False
            if pending and trigger_reload:
                if self.features_widget is None or self.filters_widget is None:
                    return
                self.features_widget.set_filters(
                    systems=self.filters_widget.selected_systems(),
                    datasets=self.filters_widget.selected_datasets(),
                    import_ids=self.filters_widget.selected_import_ids(),
                    tags=self.filters_widget.selected_tags(),
                    reload=True,
                )

    def schedule_data_requirements_emit(self) -> None:
        self._emit_data_requirements()

    def get_settings(self) -> dict[str, dict]:
        return {
            "preprocessing": (
                self.preprocessing_widget.get_settings()
                if self.preprocessing_widget is not None
                else {}
            ),
            "filters": (
                self.filters_widget.get_settings()
                if self.filters_widget is not None
                else {}
            ),
        }

    def apply_settings(
        self,
        settings: Mapping[str, object] | None,
        *,
        reload_features: bool = True,
    ) -> None:
        payload = dict(settings or {})
        preprocessing = payload.get("preprocessing")
        filters = payload.get("filters")
        self.begin_feature_reload_batch()
        self.begin_data_requirements_batch()
        try:
            if self.preprocessing_widget is not None:
                self.preprocessing_widget.set_settings(
                    preprocessing if isinstance(preprocessing, dict) else {}
                )
            if self.filters_widget is not None:
                self.filters_widget.set_settings(
                    filters if isinstance(filters, dict) else {}
                )
            if reload_features and self.features_widget is not None and self.filters_widget is not None:
                self.features_widget.set_filters(
                    systems=self.filters_widget.selected_systems(),
                    datasets=self.filters_widget.selected_datasets(),
                    import_ids=self.filters_widget.selected_import_ids(),
                    tags=self.filters_widget.selected_tags(),
                    reload=True,
                )
            self.schedule_data_requirements_emit()
        finally:
            self.end_feature_reload_batch(trigger_reload=False)
            self.end_data_requirements_batch()

    def clear_filters_and_features(self, *, refresh_filter_options: bool = True) -> None:
        self.apply_settings({"filters": {}}, reload_features=True)
        if refresh_filter_options and self.filters_widget is not None:
            self.filters_widget.refresh_filters()

    # ------------------------------------------------------------------
    def build_data_filters(self) -> Optional[DataFilters]:
        if self.features_widget is None or self.filters_widget is None:
            return None
        payloads = self.features_widget.selected_payloads()
        if not payloads:
            return None
        start = self.filters_widget.start_timestamp()
        end = self.filters_widget.end_timestamp()
        return DataFilters(
            features=[FeatureSelection.from_payload(p) for p in payloads],
            start=start,
            end=end,
            group_ids=self.filters_widget.selected_group_ids(),
            months=self.filters_widget.selected_months(),
            systems=self.filters_widget.selected_systems(),
            datasets=self.filters_widget.selected_datasets(),
            import_ids=self.filters_widget.selected_import_ids(),
        )

    # ------------------------------------------------------------------
    def fetch_base_dataframe(self, *, preprocessing_override: Optional[Mapping[str, object]] = None) -> Optional[pd.DataFrame]:
        model = self._data_model
        filters = self.build_data_filters()
        if model is None or filters is None:
            return None

        params = dict(self.preprocessing_widget.parameters()) if self.preprocessing_widget is not None else {}
        if preprocessing_override:
            params.update(dict(preprocessing_override))

        try:
            model.load_base(filters, **params)
            return model.base_dataframe().copy()
        except Exception:
            return None

    def fetch_base_dataframe_for_features(
        self,
        feature_payloads: Sequence[Mapping[str, object]],
        *,
        preprocessing_override: Optional[Mapping[str, object]] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        group_ids: Optional[Sequence[int]] = None,
    ) -> Optional[pd.DataFrame]:
        model = self._data_model
        payloads = [dict(p) for p in (feature_payloads or []) if isinstance(p, Mapping)]
        if not payloads:
            return None
        if self.filters_widget is None:
            return None
        filters = DataFilters(
            features=[FeatureSelection.from_payload(p) for p in payloads],
            start=self.filters_widget.start_timestamp() if start is None else start,
            end=self.filters_widget.end_timestamp() if end is None else end,
            group_ids=list(group_ids) if group_ids is not None else self.filters_widget.selected_group_ids(),
            months=self.filters_widget.selected_months(),
            systems=list(systems) if systems is not None else self.filters_widget.selected_systems(),
            datasets=list(datasets) if datasets is not None else self.filters_widget.selected_datasets(),
            import_ids=self.filters_widget.selected_import_ids(),
        )
        if model is None or filters is None:
            return None

        params = dict(self.preprocessing_widget.parameters()) if self.preprocessing_widget is not None else {}
        if preprocessing_override:
            params.update(dict(preprocessing_override))

        try:
            model.load_base(filters, **params)
            return model.base_dataframe().copy()
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _resolve_log_view_model(self):
        try:
            win = self.window() or self.parent()
        except Exception:
            win = None
        if win is not None:
            return getattr(win, "log_view_model", None)
        return None

    def _show_log_window(self) -> None:
        try:
            win = self.window() or self.parent()
            if win is None:
                return
            show_chat = getattr(win, "show_chat_window", None)
            if callable(show_chat):
                show_chat()
                return
            set_log_visible = getattr(win, "set_log_visible", None)
            if callable(set_log_visible):
                set_log_visible(True)
        except Exception:
            logger.warning("Exception in _show_log_window", exc_info=True)

    def _ask_features_from_ai(self, summary_text: str) -> None:
        prompt = build_features_prompt(summary_text)
        if not prompt:
            return

        log_view_model = self._resolve_log_view_model()
        if log_view_model is None:
            return

        self._show_log_window()
        try:
            log_view_model.ask_llm(prompt)
        except Exception:
            logger.warning("Exception in _ask_features_from_ai", exc_info=True)

    def _show_features_info_dialog(self, payloads: list[dict]) -> None:
        """Show feature details dialog for the selected features.
        
        Args:
            payloads: List of feature payload dicts containing feature metadata
        """
        if not payloads:
            return

        self._ensure_feature_dialog(tr("Loading feature details…"))

        model = self._data_model
        filters = None
        params = dict(self.preprocessing_widget.parameters()) if self.preprocessing_widget is not None else {}
        if model is not None and self.filters_widget is not None:
            try:
                filters = DataFilters(
                    features=[FeatureSelection.from_payload(p) for p in payloads],
                    start=self.filters_widget.start_timestamp(),
                    end=self.filters_widget.end_timestamp(),
                    group_ids=self.filters_widget.selected_group_ids(),
                    months=self.filters_widget.selected_months(),
                    systems=self.filters_widget.selected_systems(),
                    datasets=self.filters_widget.selected_datasets(),
                    import_ids=self.filters_widget.selected_import_ids(),
                )
            except Exception:
                filters = None

        self._load_feature_summary(payloads=payloads, model=model, filters=filters, params=params)

    def show_feature_details(self, payloads: Optional[Sequence[Mapping[str, object]]] = None) -> None:
        resolved_payloads: list[dict] = []
        if payloads is None:
            if self.features_widget is None:
                resolved_payloads = []
            else:
                resolved_payloads = [dict(p) for p in self.features_widget.selected_payloads()]
        else:
            resolved_payloads = [dict(p) for p in payloads if isinstance(p, Mapping)]
        if not resolved_payloads:
            return
        self._show_features_info_dialog(resolved_payloads)

    def _ensure_feature_dialog(self, summary_text: str) -> None:
        if self._feature_dialog is None:
            self._feature_dialog = FeaturesInfoDialog(
                summary_text=summary_text,
                on_ask_ai=self._ask_features_from_ai,
                parent=self,
            )
            try:
                self._feature_dialog.finished.connect(
                    lambda _res: setattr(self, "_feature_dialog", None)
                )
            except Exception:
                logger.warning("Exception in _ensure_feature_dialog", exc_info=True)
        else:
            try:
                self._feature_dialog.set_summary_text(summary_text)
            except Exception:
                logger.warning("Exception in _ensure_feature_dialog", exc_info=True)

        self._show_feature_dialog()

    def _show_feature_dialog(self) -> None:
        try:
            if self._feature_dialog is None:
                return
            self._feature_dialog.show()
            self._feature_dialog.raise_()
            self._feature_dialog.activateWindow()
        except Exception:
            logger.warning("Exception in _show_feature_dialog", exc_info=True)

    def _load_feature_summary(
        self,
        *,
        payloads: list[dict],
        model: Optional[HybridPandasModel],
        filters: Optional[DataFilters],
        params: dict[str, Any],
    ) -> None:
        def _build_summary() -> str:
            display_start = None
            display_end = None
            if model is not None and filters is not None:
                try:
                    # Show the true data bounds for the selected features, regardless of UI time filters.
                    bounds_filters = filters.clone_with_range(None, None)
                    display_start, display_end = model.time_bounds(bounds_filters)
                except Exception:
                    display_start, display_end = None, None
                if display_start is None and display_end is None:
                    display_start = filters.start
                    display_end = filters.end
                try:
                    return _build_feature_summary_text(
                        model,
                        filters,
                        display_start,
                        display_end,
                        preprocessing=params,
                    )
                except Exception:
                    logger.warning("Exception in _build_summary", exc_info=True)
            from ..utils.feature_details import build_feature_summary_from_payloads
            return build_feature_summary_from_payloads(payloads)

        def _apply_summary(summary_text: str) -> None:
            if self._feature_dialog is None:
                return
            try:
                self._feature_dialog.set_summary_text(summary_text)
            except Exception:
                logger.warning("Exception in _apply_summary", exc_info=True)

        run_in_thread(
            _build_summary,
            on_result=lambda summary: run_in_main_thread(_apply_summary, summary),
            owner=self,
            key="feature_summary",
            cancel_previous=True,
        )
        self._show_feature_dialog()


__all__ = ["DataSelectorWidget", "DataSelectorViewModel"]
