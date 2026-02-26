
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional

import pandas as pd

from PySide6.QtWidgets import QWidget, QLabel, QSplitter, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt, QTimer, QDateTime
from ...localization import tr

logger = logging.getLogger(__name__)

# Use the global cap from the shared charts package so it's consistent across all charts
from frontend.charts import MAX_FEATURES_SHOWN_LEGEND, MonthlyBarChart, TimeSeriesChart
from frontend.utils.feature_details import (
    build_feature_label_list,
    format_feature_label,
)
from ...widgets.panel import Panel
from ...models.hybrid_pandas_model import DataFilters, HybridPandasModel
from ...models.log_model import LogModel, get_log_model
from .sidebar import Sidebar
from .viewmodel import DataViewModel

from backend.models import ImportOptions
from ..tab_widget import TabWidget



def _parse_qdatetime(qdt: QDateTime) -> pd.Timestamp | None:
    if not qdt or not qdt.isValid():
        return None
    # IMPORTANT: build a naive (timezone-free) wall-clock timestamp directly
    d = qdt.date()
    t = qdt.time()
    try:
        return pd.Timestamp(
            year=d.year(),
            month=d.month(),
            day=d.day(),
            hour=t.hour(),
            minute=t.minute(),
            second=t.second(),
            microsecond=t.msec() * 1000,
        )
    except Exception:
        return None


class DataTab(TabWidget):
    """
    Controller: keeps charts, sidebar and HybridPandasModel in sync.

    Uses a shared HybridPandasModel instance that is passed in from the
    main window; no new models are created here.
    """

    def __init__(
        self,
        database_model: HybridPandasModel,
        parent=None,
        *,
        log_model: Optional[LogModel] = None,
    ):
        # Shared model used throughout the app
        self._database_model = database_model
        self._log_model = log_model or get_log_model()
        # Wrap the shared HybridPandasModel in a view-model for signal & UI helpers
        self._view_model = DataViewModel(self._database_model)
        self._import_dialog_active = False

        super().__init__(parent)

        if log_model is None and self._log_model.parent() is None:
            try:
                self._log_model.setParent(self)
            except Exception:
                logger.warning("Exception in __init__", exc_info=True)
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
        self._selection_sync_fallback = QTimer(self)
        self._selection_sync_fallback.setSingleShot(True)
        self._selection_sync_fallback.setInterval(450)
        self._selection_sync_fallback.timeout.connect(self._on_selection_sync_timeout)
        self._last_import_progress_message: Optional[str] = None
        self._show_auto_timestep_status: bool = False

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

        font = self.data_info.font()
        font.setPointSize(12)        # increase size (points)
        font.setBold(True)           # make bold
        self.data_info.setFont(font)

        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.setSpacing(8)
        info_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        info_row.addWidget(self.data_info)

        self.features_info_button = QPushButton("ℹ", right)
        self.features_info_button.setToolTip(tr("Show selected feature details"))
        self.features_info_button.setFixedSize(22, 22)
        self.features_info_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.features_info_button.clicked.connect(self._show_features_info_dialog)
        self.features_info_button.setEnabled(False)
        info_row.addWidget(self.features_info_button)
        info_row.addStretch(1)

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

    def _on_selector_data_requirements_changed(self, _requirements: dict) -> None:
        requirements = _requirements if isinstance(_requirements, dict) else {}
        self._selector_requirements_key = self._freeze_value(
            {
                "filters": requirements.get("filters", {}),
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

    def _on_progress(self, phase: str, cur: int, tot: int, msg: str) -> None:
        """Handle progress updates from the view model (e.g., during data loading/import)."""
        from frontend import utils as futils
        if str(phase) == "preprocess_auto_timestep" and not self._show_auto_timestep_status:
            return
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
        futils.set_status_text(f"[{phase}] {cur}/{tot} {msg}")

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
        return format_feature_label(feature)

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
        return build_feature_label_list(features or [], max_count=max_count)

    def _build_data_info_text(self, flt: DataFilters, start_ts, end_ts, rows: int | None = None) -> str:
        """
        Build the info line text displaying features, date range, row count, and timestep.
        
        Args:
            flt: DataFilters containing feature and time information
            start_ts: Start timestamp
            end_ts: End timestamp
            rows: Number of rows (if None, fetches from current_dataframe)
        
        Returns:
            Formatted info text string
        """
        if rows is None:
            rows = len(self._view_model.current_dataframe())
        
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
            flt = self._build_filters()
            if flt:
                self._set_monthly_chart_title(flt)
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
        b0, b1 = self._view_model.time_bounds(flt)
        if b0 is None or b1 is None:
            return
        self._set_dt_controls_from_range(b0, b1)
        self._debounce.start()

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
        self.timeseries_chart.set_dataframe(series_df)
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
        self._set_dt_controls_from_range(b0, b1)
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

    # --- reload plumbing
    def _maybe_reload(self, *args):
        if not self._should_reload():
            return
        self._debounce.start()

    def _build_filters(self) -> DataFilters | None:
        return self.sidebar.data_selector.build_data_filters()

    def _build_chart_filters(self) -> DataFilters | None:
        """Build filters used for chart/cache work, capped to visible feature count."""
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

    @staticmethod
    def _feature_payload_from_selection(selection) -> dict[str, object]:
        return {
            "feature_id": selection.feature_id,
            "notes": selection.label,
            "name": selection.base_name,
            "source": selection.source,
            "unit": selection.unit,
            "type": selection.type,
            "lag_seconds": selection.lag_seconds,
        }

    def _fetch_chart_base_dataframe(
        self,
        chart_flt: DataFilters,
    ) -> pd.DataFrame | None:
        payloads = [
            self._feature_payload_from_selection(selection)
            for selection in (chart_flt.features or [])
        ]
        if not payloads:
            return None
        return self.sidebar.data_selector.fetch_base_dataframe_for_features(
            payloads,
            start=chart_flt.start,
            end=chart_flt.end,
            systems=chart_flt.systems,
            datasets=chart_flt.datasets,
            group_ids=chart_flt.group_ids,
        )

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
        self._fetch_chart_base_dataframe(flt)
        self._capture_owned_base_cache_key()

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
        flt = self._build_filters()
        if not flt:
            self._clear_charts(tr("Select at least one feature to load."))
            self._update_features_info_button(None)
            self._last_requirements_key = None
            self._last_reload_epoch = self._reload_epoch
            self._owned_base_cache_key = None
            return
        chart_flt = self._build_chart_filters()
        if not chart_flt:
            self._clear_charts(tr("Select at least one feature to load."))
            self._update_features_info_button(None)
            self._last_requirements_key = None
            self._last_reload_epoch = self._reload_epoch
            self._owned_base_cache_key = None
            return

        # Build/refresh base cache according to preprocessing params (postprocessed!)
        base_df = self._fetch_chart_base_dataframe(chart_flt)
        self._capture_owned_base_cache_key()

        # Set initial view window to sidebar dt edits (model may refine/fetch as needed)
        if flt.start is not None and flt.end is not None:
            self._view_model.set_view_window(flt.start, flt.end)

        # ---- Monthly/Daily/Hourly bars: derive from POSTPROCESSED BASE DATA ----
        try:
            if base_df is None:
                base_df = pd.DataFrame(columns=["t"])
            df = base_df.copy()

            feature_cols = [c for c in df.columns if c != "t"]
            requested_features = len(feature_cols)
            display_cols = feature_cols[:MAX_FEATURES_SHOWN_LEGEND]
            if display_cols:
                if len(display_cols) == 1:
                    col = display_cols[0]
                    values = pd.to_numeric(df[col], errors="coerce")
                    self.monthly_chart.set_data(df["t"], values, series_name=col)
                else:
                    self.monthly_chart.set_frame(df[["t"] + display_cols])
                self.monthly_chart.chart.legend().setVisible(True)
        except Exception:
            logger.warning("Exception in _reload_now", exc_info=True)

        # ---- line chart (raw chart) from model view slice as before ----
        series_df = self._view_model.series_for_chart()
        self.timeseries_chart.set_dataframe(series_df)
        self._set_monthly_chart_title(flt)

        # Info line: show true data bounds for selected features (ignore UI time filters)
        flt_for_bounds = flt.clone_with_range(None, None)
        b0, b1 = self._view_model.time_bounds(flt_for_bounds)
        start_ts = b0
        end_ts = b1
        self.data_info.setText(
            self._build_data_info_text(flt, start_ts, end_ts, rows=len(series_df))
        )
        self._update_features_info_button(flt)
        self._last_requirements_key = requirements_key
        self._last_reload_epoch = self._reload_epoch

    def _clear_charts(self, message: str) -> None:
        self.data_info.setText(message)
        self._owned_base_cache_key = None
        try:
            self.monthly_chart.clear()
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
        for feat_sel in flt.features:
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
            files, _ = QFileDialog.getOpenFileNames(
                parent_win,
                tr("Import data (CSV/XLSX)"),
                str(Path.cwd()),
                tr("Data files (*.csv *.xlsx *.xls);;All files (*.*)"),
            )
            if not files:
                return

            base_opts = ImportOptions(system_name="DefaultSystem", dataset_name="DefaultDataset")
            dlg = ImportOptionsDialog(
                files,
                base_opts,
                database_model=self._database_model,
                data_view_model=self._view_model,
                help_viewmodel=getattr(parent_win, "help_viewmodel", None),
                parent=parent_win,
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return

            opts = dlg.build_options()
        finally:
            self._import_dialog_active = False

        self._last_import_progress_message = None

        # Define worker function to run in background thread
        def _worker(files_list, options, progress_callback=None, stop_event=None):
            # progress_callback expects int percent (0-100)
            try:
                # Update status in main thread
                run_in_main_thread(futils.set_status_text, tr("Importing data..."))
                run_in_main_thread(
                    futils.toast_info,
                    tr("Importing data…"),
                    title=tr("Import"),
                    tab_key="data",
                )
                # Delegate to model import; model will call the provided progress callback
                self._view_model.import_files(files_list, options=options, progress_callback=progress_callback)
                run_in_main_thread(futils.set_status_text, tr("Import complete."))
                run_in_main_thread(
                    futils.toast_success,
                    tr("Import complete."),
                    title=tr("Import"),
                    tab_key="data",
                )
                run_in_main_thread(futils.set_progress, None)
                return True
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
                return False

        # GUI progress handler (called from thread via signals)
        def _on_progress(percent: int):
            # Ensure GUI updates happen on main thread
            run_in_main_thread(futils.set_progress, percent)

        def _on_result(result):
            # result is True/False
            if result:
                run_in_main_thread(self._reload_now)
            else:
                # already set by worker
                pass

        def _on_error(msg: str):
            try:
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
