
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional

import pandas as pd

from PySide6.QtWidgets import QWidget, QLabel, QSplitter, QPushButton, QHBoxLayout, QSizePolicy
from PySide6.QtCore import Qt, QTimer, QDateTime
from ...localization import tr

logger = logging.getLogger(__name__)

# Use shared chart caps to keep visualization work bounded for large selections.
from frontend.charts import (
    MAX_FEATURES_SHOWN,
    MAX_FEATURES_SHOWN_LEGEND,
    MonthlyBarChart,
    TimeSeriesChart,
)
from frontend.utils.feature_details import (
    format_feature_label,
)
from ...widgets.panel import Panel
from ...models.hybrid_pandas_model import DataFilters, HybridPandasModel
from ...viewmodels.help_viewmodel import get_help_viewmodel
from ...utils.file_dialog_history import get_dialog_directory, remember_dialog_path
from .sidebar import Sidebar
from .viewmodel import DataViewModel

from backend.models import ImportOptions
from ..tab_widget import TabWidget


# @ai(gpt-5, codex, fix, 2026-03-10)
def _build_import_filter_state(
    *,
    current_state: dict | None,
    import_options: ImportOptions,
    import_ids: list[int] | None,
    imports_frame: pd.DataFrame | None,
) -> dict:
    state = dict(current_state or {})
    previous_systems = [
        str(value).strip()
        for value in (state.get("systems") or [])
        if str(value).strip()
    ]
    previous_datasets = [
        str(value).strip()
        for value in (state.get("datasets") or state.get("Datasets") or [])
        if str(value).strip()
    ]
    previous_import_ids: list[int] = []
    for value in state.get("import_ids") or []:
        try:
            previous_import_ids.append(int(value))
        except Exception:
            continue
    resolved_import_ids = [int(v) for v in (import_ids or [])]
    resolved_systems: list[str] = []
    resolved_datasets: list[str] = []

    if imports_frame is not None and not imports_frame.empty and resolved_import_ids:
        scoped = imports_frame.copy()
        if "import_id" in scoped.columns:
            scoped["import_id"] = pd.to_numeric(scoped["import_id"], errors="coerce")
            scoped = scoped[scoped["import_id"].isin(resolved_import_ids)]
        else:
            scoped = scoped.iloc[0:0]
        if not scoped.empty:
            if "system" in scoped.columns:
                resolved_systems = [
                    str(value).strip()
                    for value in scoped["system"].tolist()
                    if str(value).strip()
                ]
            if "dataset" in scoped.columns:
                resolved_datasets = [
                    str(value).strip()
                    for value in scoped["dataset"].tolist()
                    if str(value).strip()
                ]

    if not resolved_systems:
        system_name = str(import_options.system_name or "").strip()
        if system_name:
            resolved_systems = [system_name]

    if not resolved_datasets:
        dataset_name = str(import_options.dataset_name or "").strip()
        if dataset_name and dataset_name != "__sheet__":
            resolved_datasets = [dataset_name]

    state["systems"] = list(dict.fromkeys(previous_systems + resolved_systems))
    state["datasets"] = list(dict.fromkeys(previous_datasets + resolved_datasets))
    state["Datasets"] = list(state["datasets"])
    state["import_ids"] = list(dict.fromkeys(previous_import_ids + resolved_import_ids))
    return state


class DataTab(TabWidget):
    """
    Controller: keeps charts, sidebar and HybridPandasModel in sync.

    Uses a shared HybridPandasModel instance that is passed in from the
    main window; no new models are created here.
    """
    MAX_FEATURES_DETAILS_SUMMARY = 40

    def __init__(
        self,
        database_model: HybridPandasModel,
        parent=None,
    ):
        # Shared model used throughout the app
        self._database_model = database_model
        # Wrap the shared HybridPandasModel in a view-model for signal & UI helpers
        self._view_model = DataViewModel(self._database_model)
        self._import_dialog_active = False

        super().__init__(parent)

        try:
            self._view_model.setParent(self)
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._reload_now)

        self._reload_epoch = 0
        self._last_requirements_key = None
        self._last_reload_epoch = -1
        self._owned_base_cache_key = None
        self._selector_requirements_key: tuple | None = None
        self._selection_sync_pending = False
        self._reload_request_id = 0
        self._active_reload_request_id = 0
        self._inflight_requirements_key: tuple | None = None

        self._selection_sync_fallback = QTimer(self)
        self._selection_sync_fallback.setSingleShot(True)
        self._selection_sync_fallback.setInterval(450)
        self._selection_sync_fallback.timeout.connect(self._on_selection_sync_timeout)

        self._last_import_progress_message: Optional[str] = None
        self._last_progress_phase: Optional[str] = None
        self._last_import_warning_message: Optional[str] = None
        self._show_auto_timestep_status: bool = False
        self._initial_filter_timeframe_applied: bool = False
        self._pending_post_import_filter_state: dict | None = None

        self._wire_signals()

        # React to DB changes from the shared HybridPandasModel
        self._database_model.database_changed.connect(self.reload_from_db)
        self._database_model.selection_state_changed.connect(self._on_selection_state_changed)

    def reload_from_db(self, db_path: Path | str | None = None) -> None:
        """Reload UI state when the underlying database path/connection changes.

        The shared HybridPandasModel instance is responsible for reconnecting
        to the new DB. Here we just repopulate the UI from the updated model.
        """
        self._reload_epoch += 1
        self._selection_sync_pending = False
        self._selection_sync_fallback.stop()
        self._initial_filter_timeframe_applied = False
        self._clear_charts(tr("No data loaded"))

    def _on_selection_state_changed(self) -> None:
        self._reload_epoch += 1
        # Selection application updates filters/features asynchronously in the
        # DataSelector. Wait for the selector's consolidated requirements signal
        # before reloading charts so we do not render stale feature selections.
        self._selection_sync_pending = True
        self._debounce.stop()
        self._selection_sync_fallback.start()

    def close_database(self) -> None:
        # For shared model, this just notifies listeners; it does not destroy the model
        try:
            self._view_model.close_database()
        except Exception:
            logger.warning("Exception in close_database", exc_info=True)

    # --- UI
    def _create_sidebar(self) -> QWidget:
        self.sidebar = Sidebar(self._view_model, parent=self)
        return self.sidebar

    def _create_content_widget(self) -> QWidget:
        right = Panel(title="", parent=self)
        right_layout = right.content_layout()

        self.data_info = QLabel(tr("No data loaded"))
        self.data_info.setObjectName("DataInfo")
        self.data_info.setWordWrap(True)
        self.data_info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.data_info.setMinimumWidth(0)

        info_row = QHBoxLayout()
        info_row.addWidget(self.data_info, 1)

        self.features_info_button = QPushButton("ℹ", right)
        self.features_info_button.setObjectName("infoButton")
        self.features_info_button.setToolTip(tr("Show selected feature details"))
        self.features_info_button.setFixedSize(16, 16)
        self.features_info_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.features_info_button.setFlat(True)
        self.features_info_button.clicked.connect(self._show_features_info_dialog)
        self.features_info_button.setEnabled(False)
        info_row.addWidget(self.features_info_button)

        right_layout.addLayout(info_row)

        self.charts_splitter = QSplitter(Qt.Orientation.Vertical, right)
        self.charts_splitter.setChildrenCollapsible(False)
        self.charts_splitter.setHandleWidth(6)
        right_layout.addWidget(self.charts_splitter, 1)

        self.monthly_chart = MonthlyBarChart(title="")
        self.charts_splitter.addWidget(self.monthly_chart)
        try:
            self.monthly_chart.set_append_timeframe_to_title(False)
            self.monthly_chart.bucket_selected.connect(self._on_monthly_bucket_selected)
            self.monthly_chart.reset_requested.connect(self._on_monthly_reset_requested)
            self.monthly_chart.chart.legend().setVisible(True)
        except Exception:
            logger.warning("Exception in _create_content_widget", exc_info=True)

        self.timeseries_chart = TimeSeriesChart(title="")
        self.timeseries_chart.set_delegate_x_reset_to_controller(True)
        # Keep live pan updates enabled so viewport changes can fetch/window data continuously.
        self.timeseries_chart.set_live_pan_emit_enabled(True)
        self.timeseries_chart.range_changed.connect(self._on_chart_range_changed)
        self.timeseries_chart.reset_requested.connect(self._on_chart_reset_requested)
        
        try:
            self.timeseries_chart.chart.legend().setVisible(True)
        except Exception:
            logger.warning("Exception in _create_content_widget", exc_info=True)
        self.charts_splitter.addWidget(self.timeseries_chart)
        for index in range(self.charts_splitter.count()):
            self.charts_splitter.setStretchFactor(index, 1)

        return right

    # --- signals
    def _wire_signals(self):
        self._view_model.import_requested.connect(self._import_dialog)
        self._view_model.new_requested.connect(self._on_sidebar_new_database)
        self._view_model.load_requested.connect(self._on_sidebar_open_database)
        self._view_model.save_requested.connect(self._on_sidebar_save_database)
        self._view_model.progress.connect(self._on_progress)

        selector_vm = self.sidebar.data_selector.view_model
        selector_vm.data_requirements_changed.connect(self._on_selector_data_requirements_changed)
        self.sidebar.data_selector.filters_widget.filters_refreshed.connect(self._apply_pending_post_import_filters)

    def _on_selector_data_requirements_changed(self, _requirements: dict) -> None:
        requirements = _requirements if isinstance(_requirements, dict) else {}
        self._selector_requirements_key = self._freeze_value(
            {
                "filters": requirements.get("data_filters", requirements.get("filters", {})),
                "preprocessing": requirements.get("preprocessing", {}),
            }
        )
        if self._selection_sync_pending:
            self._selection_sync_pending = False
            self._selection_sync_fallback.stop()
        self._maybe_reload()

    def _on_selection_sync_timeout(self) -> None:
        if not self._selection_sync_pending:
            return
        self._selection_sync_pending = False
        self._maybe_reload()

    def _apply_pending_post_import_filters(self) -> None:
        pending = self._pending_post_import_filter_state
        if pending is None:
            return
        self._pending_post_import_filter_state = None
        try:
            preprocessing = self.sidebar.data_selector.preprocessing_widget.get_settings()
        except Exception:
            preprocessing = {}
        self.sidebar.data_selector.apply_settings(
            {
                "filters": pending,
                "preprocessing": preprocessing,
            },
            reload_features=True,
        )

    def _queue_post_import_scope_selection(self, *, import_options: ImportOptions, import_ids: list[int]) -> None:
        try:
            current_state = self.sidebar.data_selector.filters_widget.get_settings()
        except Exception:
            current_state = {}
        try:
            imports_frame = self._view_model.db.list_imports()
        except Exception:
            imports_frame = pd.DataFrame()
        self._pending_post_import_filter_state = _build_import_filter_state(
            current_state=current_state,
            import_options=import_options,
            import_ids=import_ids,
            imports_frame=imports_frame,
        )
        try:
            self.sidebar.data_selector.filters_widget.refresh_filters()
        except Exception:
            logger.warning("Failed to refresh filters after import; applying imported scope immediately.", exc_info=True)
            self._apply_pending_post_import_filters()

    def _on_progress(self, phase: str, cur: int, tot: int, msg: str) -> None:
        """Handle progress updates from the view model (e.g., during data loading/import)."""
        from frontend import utils as futils
        if str(phase) == "preprocess_auto_timestep" and not self._show_auto_timestep_status:
            return
        phase_text = str(phase or "").strip()
        message = str(msg or "").strip()
        if str(phase) == "file" and message.startswith("Importing "):
            if message != self._last_import_progress_message:
                self._last_import_progress_message = message
                file_name = Path(message.replace("Importing ", "", 1)).name
                if tot and int(tot) > 0:
                    index = min(int(cur) + 1, int(tot))
                    toast_text = tr("Loading file {index}/{total}: {name}").format(
                        index=index,
                        total=int(tot),
                        name=file_name,
                    )
                else:
                    toast_text = tr("Loading file: {name}").format(name=file_name)
                futils.toast_info(toast_text, title=tr("Import"), tab_key="data")
        if str(phase) == "import_warning" and message:
            if message != self._last_import_warning_message:
                self._last_import_warning_message = message
                futils.toast_warn(message, title=tr("Import warning"), tab_key="data")
        if phase_text and phase_text != self._last_progress_phase:
            self._last_progress_phase = phase_text
            phase_map = {
                "file": tr("Importing files..."),
                "preprocess": tr("Preprocessing data..."),
                "preprocess_auto_timestep": tr("Resolving timestep..."),
                "import_warning": tr("Import finished with warnings."),
            }
            futils.set_status_text(phase_map.get(phase_text, tr("Updating data...")))

    def _invoke_main_window_action(self, method_name: str) -> None:
        """Call the given method on the top-level window if available."""
        win = self.window() or self.parent()
        try:
            handler = getattr(win, method_name, None) if win is not None else None
            if callable(handler):
                handler()
        except Exception:
            logger.warning("Exception in _invoke_main_window_action", exc_info=True)

    def _on_sidebar_new_database(self):
        self._invoke_main_window_action("new_database")

    def _on_sidebar_open_database(self):
        self._invoke_main_window_action("open_database")

    def _on_sidebar_save_database(self):
        self._invoke_main_window_action("save_database_as")

    # --- features
    def _format_feature_label(self, feature) -> str:
        if isinstance(feature, dict):
            payload = dict(feature)
        else:
            payload = {
                "feature_id": getattr(feature, "feature_id", None),
                "name": getattr(feature, "name", getattr(feature, "base_name", None)),
                "base_name": getattr(feature, "base_name", getattr(feature, "name", None)),
                "source": getattr(feature, "source", None),
                "unit": getattr(feature, "unit", None),
                "type": getattr(feature, "type", None),
            }
        payload.pop("notes", None)
        payload.pop("label", None)
        return format_feature_label(payload)

    def _format_date(self, ts) -> str:
        """Safely format a timestamp as YYYY-MM-DD or return "?"."""
        if ts is None:
            return "?"
        try:
            t = pd.Timestamp(ts)
            return t.strftime("%Y-%m-%d")
        except Exception:
            return "?"

    def _build_feature_label_list(self, features: list | None, max_count: int | None = None) -> tuple[str, int]:
        """Return (display_names, display_count)."""
        if not features:
            return "—", 0
        limit = MAX_FEATURES_SHOWN if max_count is None else int(max_count)
        requested = len(features)
        display_count = min(requested, limit)
        names = ", ".join(
            self._format_feature_label(feature)
            for feature in (features or [])[:display_count]
        )
        if requested > display_count:
            names = f"{names} (+{requested - display_count} more)"
        return names, display_count

    def _build_data_info_text(self, flt: DataFilters, start_ts, end_ts, rows: int | None = None) -> str:
        """
        Build the info line text displaying features, date range, row count, and timestep.
        
        Args:
            flt: DataFilters containing feature and time information
            start_ts: Start timestamp
            end_ts: End timestamp
            rows: Number of rows
        
        Returns:
            Formatted info text string
        """
        s = self._format_date(start_ts)
        e = self._format_date(end_ts)
        names, _ = self._build_feature_label_list(flt.features)
        
        # Get the resolved timestep from the model (accounts for auto-resolution)
        try:
            timestep = self._view_model.resolved_timestep_display()
            timestep_str = tr(" — Timestep: {timestep}").format(timestep=timestep) if timestep else ""
        except Exception:
            timestep_str = ""

        return tr("Features: {names} — Range: {start} → {end} — Showing {rows} rows{timestep}").format(
            names=names, start=s, end=e, rows=rows, timestep=timestep_str
        )

    def _update_features_info_button(self, flt: DataFilters | None) -> None:
        try:
            self.features_info_button.setEnabled(bool(flt and getattr(flt, "features", None)))
        except Exception:
            logger.warning("Exception in _update_features_info_button", exc_info=True)

    # --- monthly interactions
    def _on_monthly_bucket_selected(self, start_ts, end_ts, level_code: str):
        if start_ts is None or end_ts is None:
            return
        self._set_dt_controls_from_range(start_ts, end_ts)
        level = str(level_code or "")
        if level == "h":
            # Hour clicks should only update the time-series viewport; keep hourly bars unchanged.
            self._view_model.set_view_window(start_ts, end_ts)
            series_df = self._view_model.series_for_chart()
            self.timeseries_chart.set_dataframe(series_df)
            try:
                display_end = self._chart_display_end(pd.Timestamp(start_ts), pd.Timestamp(end_ts))
                self.timeseries_chart.set_x_range(pd.Timestamp(start_ts), display_end)
            except Exception:
                logger.warning("Failed to apply exclusive-end viewport for hourly bucket selection.", exc_info=True)
            flt = self._build_filters()
            if flt:
                self._sync_monthly_chart_title(flt.clone_with_range(start_ts, end_ts))
                self.data_info.setText(
                    self._build_data_info_text(flt, start_ts, end_ts, rows=len(series_df))
                )
            self._update_features_info_button(flt)
            return
        self._debounce.start()

    def _on_monthly_reset_requested(self):
        flt = self._build_filters()
        if not flt:
            return
        # Match time-series reset behavior: reset to true feature bounds,
        # not the current narrowed sidebar range.
        flt_for_bounds = flt.clone_with_range(None, None)
        b0, b1 = self._view_model.time_bounds(flt_for_bounds)
        if b0 is None or b1 is None:
            return
        self._set_dt_controls_from_range(b0, b1)
        self._reload_now(force=True)

    def _set_monthly_chart_title(self, flt: DataFilters | None) -> None:
        if flt is None:
            return
        level = str(getattr(self.monthly_chart, "_current_level", "") or "")
        start = flt.start
        end = flt.end
        try:
            if level == "D" and start is not None:
                title = pd.Timestamp(start).strftime("%B %Y")
            elif level == "h" and start is not None:
                title = pd.Timestamp(start).strftime("%Y-%m-%d")
            else:
                if start is None or end is None:
                    title = tr("Date range")
                else:
                    s_ts = pd.Timestamp(start)
                    e_ts = pd.Timestamp(end)
                    # End is exclusive in filters; use inclusive end for display logic.
                    e_inclusive = e_ts - pd.Timedelta(microseconds=1)
                    if s_ts.date() == e_inclusive.date():
                        title = s_ts.strftime("%Y-%m-%d")
                    elif s_ts.year == e_inclusive.year and s_ts.month == e_inclusive.month:
                        title = s_ts.strftime("%B %Y")
                    else:
                        s = s_ts.strftime("%Y-%m-%d")
                        e = e_inclusive.strftime("%Y-%m-%d")
                        title = f"{s} - {e}"
            self.monthly_chart.set_title(title)
        except Exception:
            logger.warning("Exception in _set_monthly_chart_title", exc_info=True)

    # @ai(gpt-5, codex-cli, fix, 2026-03-13)
    def _sync_monthly_chart_title(self, flt: DataFilters | None) -> None:
        try:
            has_data = bool(self.monthly_chart.series.barSets())
        except Exception:
            has_data = False
        if has_data:
            self._set_monthly_chart_title(flt)
            return
        try:
            self.monthly_chart.set_title("")
        except Exception:
            logger.warning("Exception in _sync_monthly_chart_title", exc_info=True)

    # --- chart interactions
    def _on_chart_range_changed(self, start_ts, end_ts):
        """
        User finished a zoom or ended a pan/zoom scroll.
        Only update the model's view window; it will decide if it needs to fetch/window.
        """
        self._ensure_data_tab_cache_active()
        y_range = None
        try:
            y_min = float(self.timeseries_chart.axis_y.min())
            y_max = float(self.timeseries_chart.axis_y.max())
            if y_max > y_min:
                y_range = (y_min, y_max)
        except Exception:
            logger.warning("Exception in _on_chart_range_changed", exc_info=True)

        self._view_model.set_view_window(start_ts, end_ts)
        series_df = self._view_model.series_for_chart()
        self.timeseries_chart.update_window_dataframe(series_df)
        try:
            # Keep the dragged viewport stable. set_dataframe() derives X-range
            # from data bounds, which can slightly snap on first drag event.
            self.timeseries_chart.set_x_range(pd.Timestamp(start_ts), pd.Timestamp(end_ts))
        except Exception:
            logger.warning("Failed to restore dragged X-range after time-series refresh.", exc_info=True)
        if y_range is not None:
            try:
                self.timeseries_chart.axis_y.setRange(*y_range)
            except Exception:
                logger.warning("Exception in _on_chart_range_changed", exc_info=True)
        # mirror to dt edits but do NOT trigger expensive reloads
        self._set_dt_controls_from_range(start_ts, end_ts)
        # info line
        flt = self._build_filters()
        if flt:
            self._sync_monthly_chart_title(flt.clone_with_range(start_ts, end_ts))
            self.data_info.setText(
                self._build_data_info_text(flt, start_ts, end_ts, rows=len(series_df))
            )
        self._update_features_info_button(flt)

    def _on_chart_reset_requested(self):
        """
        Reset to true DB min/max for the currently selected features.
        """
        flt = self._build_filters()
        if not flt:
            return
        # ignore current start/end in flt for bounds; DB gives absolute per feature
        flt_for_bounds = flt.clone_with_range(None, None)
        b0, b1 = self._view_model.time_bounds(flt_for_bounds)
        if b0 is None or b1 is None:
            return
        already_at_bounds = (
            self._timestamps_match(flt.start, b0)
            and self._timestamps_match(flt.end, b1)
        )
        self._set_dt_controls_from_range(b0, b1)
        if already_at_bounds:
            return
        # Full reload with these new dt edits
        self._reload_now(force=True)

    # --- helpers
    def _set_dt_controls_from_range(self, start_ts, end_ts):
        if start_ts is None or end_ts is None:
            return
        try:
            s = pd.Timestamp(start_ts)
            e = pd.Timestamp(end_ts)
        except Exception:
            return

        def _to_qdt(ts: pd.Timestamp) -> QDateTime:
            ts = pd.Timestamp(ts)  # ensure Timestamp
            return QDateTime(
                ts.year, ts.month, ts.day,
                ts.hour, ts.minute, ts.second, int(ts.microsecond / 1000),
            )

        s_qdt = _to_qdt(s)
        e_qdt = _to_qdt(e)
        self.sidebar.set_date_range_controls(s_qdt, e_qdt)

    def _chart_display_end(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.Timestamp:
        """Return a chart-only right edge that respects exclusive filter end bounds."""
        try:
            s = pd.Timestamp(start_ts)
            e = pd.Timestamp(end_ts)
        except Exception:
            return pd.Timestamp(end_ts)
        if e <= s:
            return e
        is_midnight = (
            e.hour == 0
            and e.minute == 0
            and e.second == 0
            and e.microsecond == 0
        )
        if is_midnight and (e - s) >= pd.Timedelta(days=1):
            # Keep data/query end exclusive (< end) and avoid showing next-day
            # midnight as the last X-axis label for full-day/month ranges.
            return e - pd.Timedelta(milliseconds=1)
        return e

    # --- reload plumbing
    def _maybe_reload(self, *args):
        if not self._should_reload():
            return
        self._debounce.start()

    def _build_filters(self) -> DataFilters | None:
        return self.sidebar.data_selector.build_data_filters()

    def _build_chart_filters(self, flt: DataFilters | None = None) -> DataFilters | None:
        """Build filters used for chart/cache work, capped to visible feature count."""
        if flt is None:
            flt = self._build_filters()
        if flt is None:
            return None
        if len(flt.features) <= MAX_FEATURES_SHOWN_LEGEND:
            return flt
        return DataFilters(
            features=list(flt.features[:MAX_FEATURES_SHOWN_LEGEND]),
            start=flt.start,
            end=flt.end,
            group_ids=list(flt.group_ids) if flt.group_ids is not None else None,
            months=list(flt.months) if flt.months is not None else None,
            systems=list(flt.systems) if flt.systems is not None else None,
            datasets=list(flt.datasets) if flt.datasets is not None else None,
            import_ids=list(flt.import_ids) if flt.import_ids is not None else None,
        )

    def _next_reload_request_id(self) -> int:
        self._reload_request_id += 1
        return self._reload_request_id

    def _freeze_value(self, value: object) -> tuple:
        if isinstance(value, dict):
            return tuple(sorted((str(k), self._freeze_value(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple, set)):
            return tuple(self._freeze_value(v) for v in value)
        if isinstance(value, pd.Timestamp):
            return (str(pd.Timestamp(value)),)
        try:
            hash(value)
        except Exception:
            return (repr(value),)
        return (value,)

    def _requirements_key(self) -> tuple | None:
        flt = self._build_chart_filters()
        if not flt:
            return None

        def _ts_key(value) -> str | None:
            if value is None:
                return None
            try:
                return str(pd.Timestamp(value))
            except Exception:
                return str(value)

        def _sort_key(items: tuple) -> tuple:
            return tuple("" if v is None else str(v) for v in items)

        feature_items = [
            (
                int(sel.feature_id) if sel.feature_id is not None else None,
                sel.base_name or None,
                sel.source or None,
                sel.unit or None,
                sel.type or None,
                int(sel.lag_seconds) if sel.lag_seconds is not None else 0,
            )
            for sel in (flt.features or [])
        ]
        features_key = tuple(sorted(feature_items, key=_sort_key))
        systems_key = tuple(sorted(str(s) for s in (flt.systems or [])))
        datasets_key = tuple(sorted(str(l) for l in (flt.Datasets or [])))
        imports_key = tuple(sorted(int(i) for i in (flt.import_ids or [])))
        groups_key = tuple(sorted(int(g) for g in (flt.group_ids or [])))
        months_key = tuple(sorted(int(m) for m in self.sidebar.selected_months()))
        return (
            features_key,
            _ts_key(flt.start),
            _ts_key(flt.end),
            systems_key,
            datasets_key,
            imports_key,
            groups_key,
            months_key,
            self._selector_requirements_key,
        )

    def _should_reload(self) -> bool:
        key = self._requirements_key()
        if key is None:
            return self._last_requirements_key is not None or self._last_reload_epoch != self._reload_epoch
        if self._last_reload_epoch != self._reload_epoch:
            return True
        return key != self._last_requirements_key

    def _capture_owned_base_cache_key(self) -> None:
        try:
            self._owned_base_cache_key = self._view_model.current_base_cache_key()
        except Exception:
            self._owned_base_cache_key = None

    def _timestamps_match(self, left, right) -> bool:
        try:
            if left is None and right is None:
                return True
            if left is None or right is None:
                return False
            return pd.Timestamp(left) == pd.Timestamp(right)
        except Exception:
            return False

    def _apply_initial_timeframe_before_first_fetch(self, flt: DataFilters) -> DataFilters:
        """Resolve initial filter range from feature bounds before first base-data fetch."""
        if self._initial_filter_timeframe_applied:
            return flt
        flt_for_bounds = flt.clone_with_range(None, None)
        b0, b1 = self._view_model.time_bounds(flt_for_bounds)
        if b0 is None or b1 is None:
            return flt
        self._initial_filter_timeframe_applied = True
        if self._timestamps_match(flt.start, b0) and self._timestamps_match(flt.end, b1):
            return flt
        self._set_dt_controls_from_range(b0, b1)
        refreshed = self._build_chart_filters(self._build_filters())
        return refreshed if refreshed is not None else flt.clone_with_range(b0, b1)

    @staticmethod
    def _feature_payloads_from_filters(flt: DataFilters) -> list[dict]:
        payloads: list[dict] = []
        for feature in list(flt.features or []):
            payloads.append(
                {
                    "feature_id": feature.feature_id,
                    "notes": feature.label,
                    "name": feature.base_name,
                    "source": feature.source,
                    "unit": feature.unit,
                    "type": feature.type,
                    "lag_seconds": feature.lag_seconds,
                }
            )
        return payloads

    def _ensure_data_tab_cache_active(self) -> None:
        flt = self._build_chart_filters()
        if not flt:
            self._owned_base_cache_key = None
            return
        try:
            current_key = self._view_model.current_base_cache_key()
        except Exception:
            current_key = None
        if self._owned_base_cache_key is not None and current_key == self._owned_base_cache_key:
            return
        # Trigger an async reload so the cache is built without blocking the UI.
        self._reload_now(force=True)

    def _reload_now(self, *, force: bool = False):
        self._debounce.stop()
        requirements_key = self._requirements_key()
        if not force:
            if (
                requirements_key is None
                and self._last_requirements_key is None
                and self._last_reload_epoch == self._reload_epoch
            ):
                return
            if (
                requirements_key is not None
                and requirements_key == self._last_requirements_key
                and self._last_reload_epoch == self._reload_epoch
            ):
                return
            if (
                requirements_key is not None
                and requirements_key == self._inflight_requirements_key
                and self._last_reload_epoch == self._reload_epoch
            ):
                return
        flt = self._build_filters()
        if not flt:
            self._clear_charts(tr("Select at least one feature to load."))
            self._update_features_info_button(None)
            self._last_requirements_key = None
            self._last_reload_epoch = self._reload_epoch
            self._inflight_requirements_key = None
            self._owned_base_cache_key = None
            return
        chart_filters = self._build_chart_filters(flt)
        if not chart_filters:
            self._clear_charts(tr("Select at least one feature to load."))
            self._update_features_info_button(None)
            self._last_requirements_key = None
            self._last_reload_epoch = self._reload_epoch
            self._inflight_requirements_key = None
            self._owned_base_cache_key = None
            return

        chart_filters = self._apply_initial_timeframe_before_first_fetch(chart_filters)
        requirements_key = self._requirements_key()

        request_id = self._next_reload_request_id()
        self._active_reload_request_id = request_id
        self._inflight_requirements_key = requirements_key

        def _on_fetch_result(frame_token: str) -> None:
            if request_id != self._active_reload_request_id:
                return
            self._inflight_requirements_key = None
            base_df = self.sidebar.data_selector.resolve_dataframe_token(frame_token, consume=True)
            self._apply_reload_result(
                flt=chart_filters,
                requirements_key=requirements_key,
                base_df=base_df,
            )

        def _on_fetch_error(_message: str) -> None:
            if request_id != self._active_reload_request_id:
                return
            self._inflight_requirements_key = None
            self._clear_charts(tr("Failed to load selected data."))
            self._update_features_info_button(None)
            self._last_requirements_key = requirements_key
            self._last_reload_epoch = self._reload_epoch

        started = self.sidebar.data_selector.fetch_base_dataframe_for_features_token_async(
            self._feature_payloads_from_filters(chart_filters),
            start=chart_filters.start,
            end=chart_filters.end,
            systems=chart_filters.systems,
            datasets=chart_filters.datasets,
            group_ids=chart_filters.group_ids,
            on_result=_on_fetch_result,
            on_error=_on_fetch_error,
            owner=self,
            key="data_tab_reload_fetch",
            cancel_previous=True,
        )
        if started:
            return
        self._inflight_requirements_key = None
        self._clear_charts(tr("Failed to start data reload."))
        self._update_features_info_button(None)
        self._last_requirements_key = requirements_key
        self._last_reload_epoch = self._reload_epoch

    def _apply_reload_result(
        self,
        *,
        flt: DataFilters,
        requirements_key: tuple | None,
        base_df: pd.DataFrame | None,
    ) -> None:
        self._capture_owned_base_cache_key()

        flt_for_bounds = flt.clone_with_range(None, None)
        b0, b1 = self._view_model.time_bounds(flt_for_bounds)
        effective_start = flt.start
        effective_end = flt.end
        if effective_start is None or effective_end is None:
            effective_start = b0 if effective_start is None else effective_start
            effective_end = b1 if effective_end is None else effective_end

        if effective_start is not None and effective_end is not None:
            self._view_model.set_view_window(effective_start, effective_end)

        try:
            working_df = (
                pd.DataFrame(columns=["t"])
                if base_df is None
                else base_df.copy()
            )
            feature_cols = [c for c in working_df.columns if c != "t"]
            display_cols = feature_cols[:MAX_FEATURES_SHOWN_LEGEND]
            if display_cols:
                if len(display_cols) == 1:
                    col = display_cols[0]
                    values = pd.to_numeric(working_df[col], errors="coerce")
                    self.monthly_chart.set_data(
                        working_df["t"],
                        values,
                        series_name=col,
                        show_legend=True,
                    )
                else:
                    self.monthly_chart.set_frame(working_df[["t"] + display_cols])
            else:
                self.monthly_chart.clear()
        except Exception:
            logger.warning("Exception in _apply_reload_result", exc_info=True)

        series_df = self._view_model.series_for_chart()
        self.timeseries_chart.set_dataframe(series_df)
        if effective_start is not None and effective_end is not None:
            try:
                display_end = self._chart_display_end(pd.Timestamp(effective_start), pd.Timestamp(effective_end))
                self.timeseries_chart.set_x_range(pd.Timestamp(effective_start), display_end)
            except Exception:
                logger.warning("Failed to restore requested X-range after reload.", exc_info=True)
        title_filters = flt.clone_with_range(effective_start, effective_end)
        self._sync_monthly_chart_title(title_filters)

        self.data_info.setText(
            self._build_data_info_text(flt, b0, b1, rows=len(series_df))
        )
        self._update_features_info_button(flt)
        self._last_requirements_key = requirements_key
        self._last_reload_epoch = self._reload_epoch

    def _clear_charts(self, message: str) -> None:
        self.data_info.setText(message)
        self._owned_base_cache_key = None
        try:
            self.monthly_chart.clear()
            self.monthly_chart.set_title("")
        except Exception:
            logger.warning("Exception in _clear_charts", exc_info=True)
        try:
            self.timeseries_chart.clear()
        except Exception:
            logger.warning("Exception in _clear_charts", exc_info=True)

    def _show_features_info_dialog(self) -> None:
        flt = self._build_filters()
        if not flt or not flt.features:
            self.data_info.setText(tr("Select at least one feature to view details."))
            self._update_features_info_button(None)
            return

        # Convert FeatureSelection objects to payloads for the dialog
        payloads = []
        for feat_sel in flt.features[: self.MAX_FEATURES_DETAILS_SUMMARY]:
            feature_name = feat_sel.base_name
            feature_type = feat_sel.type
            payloads.append({
                "feature_id": feat_sel.feature_id,
                "notes": feat_sel.label,
                "name": feature_name,
                "source": feat_sel.source,
                "unit": feat_sel.unit,
                "type": feature_type,
            })

        try:
            self.sidebar.data_selector.show_feature_details(payloads)
        except Exception:
            logger.warning("Exception in _show_features_info_dialog", exc_info=True)

    # --- import
    def _import_dialog(self):
        if self._import_dialog_active:
            return

        from PySide6.QtWidgets import QFileDialog, QDialog
        from frontend.windows.import_options_dialog import ImportOptionsDialog
        from frontend.threading import run_in_main_thread, run_in_thread
        from frontend import utils as futils

        self._import_dialog_active = True
        try:
            parent_win = self.window() or self
            start_dir = str(get_dialog_directory(parent_win, "import", Path.cwd()))
            files, _ = QFileDialog.getOpenFileNames(
                parent_win,
                tr("Import data (CSV/XLSX)"),
                start_dir,
                tr("Data files (*.csv *.xlsx *.xls);;All files (*.*)"),
            )
            if not files:
                return
            remember_dialog_path(parent_win, "import", files[0])

            base_opts = ImportOptions(system_name="DefaultSystem", dataset_name="DefaultDataset")
            try:
                help_viewmodel = get_help_viewmodel()
            except Exception:
                help_viewmodel = None
            dlg = ImportOptionsDialog(
                files,
                base_opts,
                database_model=self._database_model,
                data_view_model=self._view_model,
                help_viewmodel=help_viewmodel,
                parent=parent_win,
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return

            opts = dlg.build_options()
        finally:
            self._import_dialog_active = False

        self._last_import_progress_message = None
        self._last_progress_phase = None
        self._last_import_warning_message = None

        # Define worker function to run in background thread
        def _worker(files_list, options, progress_callback=None, stop_event=None):
            # progress_callback expects int percent (0-100)
            try:
                # Update status in main thread
                run_in_main_thread(futils.set_progress, 0)
                run_in_main_thread(futils.set_status_text, tr("Importing data..."))
                run_in_main_thread(
                    futils.toast_info,
                    tr("Importing data…"),
                    title=tr("Import"),
                    tab_key="data",
                )
                # Delegate to model import; model will call the provided progress callback
                import_ids = self._view_model.import_files(
                    files_list,
                    options=options,
                    progress_callback=progress_callback,
                )
                run_in_main_thread(futils.set_status_text, tr("Import finished."))
                run_in_main_thread(
                    futils.toast_success,
                    tr("Import complete."),
                    title=tr("Import"),
                    tab_key="data",
                )
                run_in_main_thread(futils.set_progress, None)
                return {"ok": True, "import_ids": list(import_ids or [])}
            except Exception as e:
                run_in_main_thread(
                    futils.set_status_text,
                    tr("Import failed: {error}").format(error=e),
                )
                run_in_main_thread(
                    futils.toast_error,
                    f"{e}",
                    title=tr("Import failed"),
                    tab_key="data",
                )
                run_in_main_thread(futils.set_progress, None)
                return {"ok": False, "import_ids": []}

        # GUI progress handler (called from thread via signals)
        def _on_progress(percent: int):
            # Ensure GUI updates happen on main thread
            run_in_main_thread(futils.set_progress, percent)

        def _on_result(result):
            payload = result if isinstance(result, dict) else {}
            if not bool(payload.get("ok")):
                return
            import_ids = [int(v) for v in (payload.get("import_ids") or [])]
            run_in_main_thread(
                self._queue_post_import_scope_selection,
                import_options=opts,
                import_ids=import_ids,
            )
            run_in_main_thread(self._reload_now, force=True)

        def _on_error(msg: str):
            try:
                futils.set_status_text(tr("Import failed: {error}").format(error=msg))
                futils.toast_error(
                    tr("Failed to import files: {message}").format(message=msg),
                    title=tr("Import failed"),
                    tab_key="data",
                )
            except Exception:
                logger.warning("Exception in _on_error", exc_info=True)
            run_in_main_thread(futils.set_progress, None)

        # Start background thread; runner tracks lifetime and cleanup
        run_in_thread(
            _worker,
            on_result=_on_result,
            on_progress=_on_progress,
            on_error=_on_error,
            files_list=files,
            options=opts,
            owner=self,
            key="import_files",
            cancel_previous=True,
        )

    def closeEvent(self, ev):
        super().closeEvent(ev)
