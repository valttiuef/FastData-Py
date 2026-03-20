
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import time

import pandas as pd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from ...localization import tr

from . import MAX_FEATURES_SHOWN_LEGEND
from ...models.hybrid_pandas_model import FeatureSelection, HybridPandasModel
from .viewmodel import ChartsViewModel
from ...threading import run_in_main_thread, run_in_thread

from .chart_card import ChartCard
from .charts_sidebar import ChartsSidebar
from ...utils.exporting import ExportPlan, execute_export_plan, prepare_charts_excel_export_plan, prepare_dataframes_export_plan
from ...utils import set_status_text, toast_error, toast_file_saved, toast_info, toast_success, toast_warn
from ...widgets.export_dialog import ExportOption, ExportSelectionDialog
from ...widgets.panel import Panel
from ..tab_widget import TabWidget

class ChartsTab(TabWidget):
    """Interactive charts workspace with configurable chart cards."""
    MAX_CHART_CARDS = 9
    MIN_CHART_CARD_HEIGHT = 220

    def __init__(self, database_model: HybridPandasModel, parent=None):
        self._database_model = database_model
        self._view_model = ChartsViewModel(database_model)
        self._chart_cards: List[ChartCard] = []
        self._selected_card: ChartCard | None = None
        self._refresh_epoch = 0
        self._card_render_state: dict[ChartCard, tuple[int, tuple | None]] = {}
        self._suppress_card_updates = False
        self._correlation_bar_payload: dict[str, object] | None = None
        self._last_progress_toast: Optional[str] = None
        self._last_progress_toast_at: float = 0.0
        self._show_auto_timestep_toast: bool = False
        self._allow_auto_timestep_toast: bool = False
        self._last_preprocessing_key: tuple | None = None
        self._selector_requirements_key: tuple | None = None
        self._selection_sync_pending = False
        self._card_fetch_nonce: int = 0
        self._card_fetch_tokens: dict[ChartCard, int] = {}
        self._shared_frame_cache: dict[tuple, pd.DataFrame] = {}
        self._inflight_frame_fetch_waiters: dict[tuple, list[tuple[object, object]]] = {}

        super().__init__(parent)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._refresh_all_charts)
        self._selection_sync_fallback = QTimer(self)
        self._selection_sync_fallback.setSingleShot(True)
        self._selection_sync_fallback.setInterval(450)
        self._selection_sync_fallback.timeout.connect(self._on_selection_sync_timeout)

        self._connect_signals()
        self._refresh_feature_items()
        self._ensure_chart_count(2)
        self._configure_default_cards()
        self._refresh_all_charts()
        self._allow_auto_timestep_toast = False

        self._database_model.database_changed.connect(self.reload_from_db)
        self._database_model.selection_state_changed.connect(self._on_selection_state_changed)
        self._database_model.progress.connect(self._on_model_progress)

    # ------------------------------------------------------------------
    def _create_sidebar(self) -> QWidget:
        self.sidebar = ChartsSidebar(view_model=self._view_model, parent=self)
        return self.sidebar

    # @ai(gpt-5, codex-cli, refactor, 2026-03-12)
    def _create_content_widget(self) -> QWidget:
        right_panel = Panel("", parent=self)
        right_layout = right_panel.content_layout()

        header = QLabel(tr("Select features and find correlations or plot charts."))
        header.setObjectName("panelStatusLabel")
        header.setWordWrap(True)
        right_layout.addWidget(header)

        self._charts_scroll = QScrollArea(right_panel)
        self._charts_scroll.setObjectName("chartsScrollArea")
        self._charts_scroll.setWidgetResizable(True)
        self._charts_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._charts_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._charts_container = QWidget(self._charts_scroll)
        self._charts_container.setObjectName("chartsScrollContainer")
        self._charts_layout = QVBoxLayout(self._charts_container)
        self._charts_layout.setContentsMargins(0, 0, 0, 0)
        self._charts_layout.setSpacing(8)

        self._charts_scroll.setWidget(self._charts_container)
        right_layout.addWidget(self._charts_scroll, 1)

        self._row_splitters: list[QSplitter] = []

        return right_panel

    def _connect_signals(self) -> None:
        self._view_model.add_chart_requested.connect(self._on_add_chart_requested)
        self._view_model.remove_chart_requested.connect(self._on_remove_chart_requested)
        self._view_model.chart_configuration_changed.connect(self._on_card_configuration_changed)
        self._view_model.correlation_search_finished.connect(self._on_correlation_search_finished)
        self._view_model.correlation_search_failed.connect(self._on_correlation_search_failed)
        self.sidebar.btn_export.clicked.connect(self._export_results)
        self.sidebar.features_widget.features_reloaded.connect(lambda _df: self._refresh_feature_items())
        self.sidebar.features_widget.selection_changed.connect(
            lambda _payloads: self._on_sidebar_feature_selection_changed()
        )
        self.sidebar.data_selector.view_model.data_requirements_changed.connect(
            self._on_selector_data_requirements_changed
        )

    # ------------------------------------------------------------------
    def reload_from_db(self, db_path: Path | str | None = None) -> None:
        self._refresh_epoch += 1
        self._selection_sync_pending = False
        self._selection_sync_fallback.stop()
        self._card_render_state.clear()
        self._invalidate_shared_frame_cache()
        for card in self._chart_cards:
            card.clear_chart(tr("Select features for the new database"))
        self._suppress_card_updates = True
        try:
            for card in self._chart_cards:
                card.set_available_features([])
        finally:
            self._suppress_card_updates = False
        self._debounce.start()

    def _on_selection_state_changed(self) -> None:
        self._refresh_epoch += 1
        self._selection_sync_pending = True
        self._invalidate_shared_frame_cache()
        self._debounce.stop()
        self._selection_sync_fallback.start()

    def _on_selector_data_requirements_changed(self, _requirements: dict) -> None:
        requirements = _requirements if isinstance(_requirements, dict) else {}
        previous_selector_key = self._selector_requirements_key
        preprocessing_key = self._freeze_value(requirements.get("preprocessing", {}))
        self._selector_requirements_key = self._freeze_value(
            {
                "filters": requirements.get("data_filters", requirements.get("filters", {})),
                "preprocessing": requirements.get("preprocessing", {}),
            }
        )
        if self._selector_requirements_key != previous_selector_key:
            self._invalidate_shared_frame_cache()
        if preprocessing_key != self._last_preprocessing_key:
            self._allow_auto_timestep_toast = self._show_auto_timestep_toast
            self._last_preprocessing_key = preprocessing_key
        if self._selection_sync_pending:
            self._selection_sync_pending = False
            self._selection_sync_fallback.stop()
            self._refresh_feature_items()
        self._debounce.start()

    def _on_selection_sync_timeout(self) -> None:
        if not self._selection_sync_pending:
            return
        self._selection_sync_pending = False
        self._refresh_feature_items()
        self._debounce.start()

    def _on_model_progress(self, phase: str, _cur: int, _tot: int, msg: str) -> None:
        if str(phase) != "preprocess_auto_timestep":
            return
        if not self._show_auto_timestep_toast:
            self._allow_auto_timestep_toast = False
            return
        message = str(msg or "").strip()
        if not self._allow_auto_timestep_toast or not message:
            self._allow_auto_timestep_toast = False
            return
        now = time.monotonic()
        should_toast = message != self._last_progress_toast or (now - self._last_progress_toast_at) > 5.0
        if not should_toast:
            self._allow_auto_timestep_toast = False
            return
        self._last_progress_toast = message
        self._last_progress_toast_at = now
        toast_info(message, title=tr("Auto timestep"), tab_key="charts")
        self._allow_auto_timestep_toast = False

    def _on_sidebar_feature_selection_changed(self) -> None:
        self._debounce.start()

    def close_database(self) -> None:
        self._view_model.close()

    # ------------------------------------------------------------------
    def _refresh_feature_items(self) -> None:
        items = self.sidebar.available_feature_items()
        self._suppress_card_updates = True
        try:
            for card in self._chart_cards:
                card.set_available_features(items)
        finally:
            self._suppress_card_updates = False

    # ------------------------------------------------------------------
    def _ensure_chart_count(self, count: int) -> None:
        target = max(0, min(int(count), self.MAX_CHART_CARDS))
        while len(self._chart_cards) < target:
            self._add_chart_card()
        self._update_add_chart_button_state()
        self._layout_chart_cards()

    def _set_chart_count_exact(self, count: int) -> None:
        count = max(1, min(int(count), self.MAX_CHART_CARDS))
        while len(self._chart_cards) < count:
            self._add_chart_card()
        while len(self._chart_cards) > count:
            card = self._chart_cards.pop()
            self._card_render_state.pop(card, None)
            self._card_fetch_tokens.pop(card, None)
            if self._selected_card is card:
                self._selected_card = None
            card.set_view_model(None)
            card.setParent(None)
            card.deleteLater()
        if self._selected_card not in self._chart_cards:
            fallback = self._chart_cards[0] if self._chart_cards else None
            self._set_selected_card(fallback)
        self._update_add_chart_button_state()
        self._layout_chart_cards()

    # @ai(gpt-5, codex-cli, refactor, 2026-03-12)
    def _on_add_chart_requested(self) -> None:
        if not self._add_chart_card():
            set_status_text(tr("Chart panel limit reached."))
            toast_warn(
                tr("Maximum of {count} chart panels reached.").format(count=self.MAX_CHART_CARDS),
                title=tr("Charts"),
                tab_key="charts",
            )
            self._update_add_chart_button_state()
            return
        self._layout_chart_cards()
        self._refresh_all_charts()

    def _add_chart_card(self) -> bool:
        if len(self._chart_cards) >= self.MAX_CHART_CARDS:
            self._update_add_chart_button_state()
            return False
        card = ChartCard(parent=self, view_model=self._view_model)
        self._apply_chart_card_height(card)
        self._chart_cards.append(card)
        items = self.sidebar.available_feature_items()
        card.set_available_features(items)
        card.correlation_feature_clicked.connect(self._on_correlation_feature_clicked)
        card.selection_requested.connect(self._on_card_selection_requested)
        if self._selected_card is None:
            self._set_selected_card(card)
        self._update_add_chart_button_state()
        return True

    def _apply_chart_card_height(self, card: ChartCard) -> None:
        card.setMinimumHeight(self.MIN_CHART_CARD_HEIGHT)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _on_remove_chart_requested(self) -> None:
        if not self._chart_cards:
            return
        card = self._chart_cards.pop()
        self._card_render_state.pop(card, None)
        self._card_fetch_tokens.pop(card, None)
        if self._selected_card is card:
            self._selected_card = None
        card.set_view_model(None)
        card.setParent(None)
        card.deleteLater()
        if self._selected_card not in self._chart_cards:
            fallback = self._chart_cards[0] if self._chart_cards else None
            self._set_selected_card(fallback)
        self._update_add_chart_button_state()
        self._layout_chart_cards()
        self._refresh_all_charts()

    def _update_add_chart_button_state(self) -> None:
        sidebar = getattr(self, "sidebar", None)
        button = getattr(sidebar, "btn_add_chart", None) if sidebar is not None else None
        if button is None:
            return
        button.setEnabled(len(self._chart_cards) < self.MAX_CHART_CARDS)

    def _configure_default_cards(self) -> None:
        self._set_chart_count_exact(2)
        self._suppress_card_updates = True
        try:
            self._chart_cards[0].set_chart_type("correlation_bar", emit_changed=False)
            self._chart_cards[0].set_selected_features([], emit_changed=False)
            self._chart_cards[1].set_chart_type("scatter", emit_changed=False)
            self._chart_cards[1].set_selected_features([], emit_changed=False)
        finally:
            self._suppress_card_updates = False
        self._set_selected_card(self._chart_cards[1] if len(self._chart_cards) > 1 else self._chart_cards[0])

    def _layout_chart_cards(self) -> None:
        # remove existing row splitters from the rows layout
        for splitter in self._row_splitters:
            splitter.setParent(None)
            splitter.deleteLater()
        self._row_splitters = []

        while self._charts_layout.count():
            item = self._charts_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        if not self._chart_cards:
            self._charts_container.setMinimumHeight(0)
            return

        if len(self._chart_cards) == 1:
            self._add_row(self._chart_cards[:1], full_width=True)
        elif len(self._chart_cards) == 2:
            self._add_row(self._chart_cards[0:1], full_width=True)
            self._add_row(self._chart_cards[1:2], full_width=True)
        else:
            self._add_row(self._chart_cards[0:1], full_width=True)
            for start in range(1, len(self._chart_cards), 2):
                self._add_row(self._chart_cards[start : start + 2], full_width=False)
        for idx, _row in enumerate(self._row_splitters):
            self._charts_layout.setStretch(idx, 1)

        row_count = max(0, len(self._row_splitters))
        min_height = 0
        if row_count > 0:
            min_height = row_count * self.MIN_CHART_CARD_HEIGHT + max(0, row_count - 1) * 8
        self._charts_container.setMinimumHeight(min_height)

    def _add_row(self, row_cards: list[ChartCard], *, full_width: bool) -> None:
        row_splitter = QSplitter(Qt.Orientation.Horizontal, self._charts_container)
        row_splitter.setChildrenCollapsible(False)
        row_splitter.setMinimumHeight(self.MIN_CHART_CARD_HEIGHT)
        row_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        for idx, card in enumerate(row_cards):
            self._apply_chart_card_height(card)
            row_splitter.addWidget(card)
            row_splitter.setStretchFactor(idx, 1)
        if not full_width and len(row_cards) > 1:
            row_splitter.setSizes([1, 1])
        self._charts_layout.addWidget(row_splitter)
        self._row_splitters.append(row_splitter)

    # ------------------------------------------------------------------
    def _on_card_configuration_changed(self, card: ChartCard) -> None:
        if self._selected_card is card and not self._is_card_selectable_for_correlation_target(card):
            self._set_selected_card(None)
        if self._suppress_card_updates:
            return
        self._update_chart(card, force=True)

    def _on_card_selection_requested(self, card: ChartCard) -> None:
        if card not in self._chart_cards:
            return
        if not self._is_card_selectable_for_correlation_target(card):
            return
        self._set_selected_card(card)

    def _set_selected_card(self, card: ChartCard | None) -> None:
        if card in self._chart_cards and self._is_card_selectable_for_correlation_target(card):
            self._selected_card = card
        else:
            self._selected_card = self._first_selectable_target_card()
        for item in self._chart_cards:
            item.set_selected_for_correlation_target(item is self._selected_card)

    @staticmethod
    def _is_card_selectable_for_correlation_target(card: ChartCard) -> bool:
        return card.chart_type() != "correlation_bar"

    def _first_selectable_target_card(self) -> ChartCard | None:
        for card in self._chart_cards:
            if self._is_card_selectable_for_correlation_target(card):
                return card
        return None

    def _refresh_all_charts(self) -> None:
        for card in self._chart_cards:
            self._update_chart(card)

    # ------------------------------------------------------------------
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

    def _card_requirements_key(
        self,
        *,
        chart_type: str,
        features: list[FeatureSelection],
        selector_requirements_key: tuple | None,
    ) -> tuple:
        features_key = tuple(
            (
                int(sel.feature_id) if sel.feature_id is not None else None,
                sel.label or None,
                sel.base_name or None,
                sel.source or None,
                sel.unit or None,
                sel.type or None,
                int(sel.lag_seconds) if sel.lag_seconds is not None else 0,
            )
            for sel in features
        )
        return (
            chart_type,
            features_key,
            selector_requirements_key,
        )

    def _should_update_card(self, card: ChartCard, key: tuple | None, *, force: bool) -> bool:
        if force:
            return True
        state = self._card_render_state.get(card)
        if state is None:
            return True
        last_epoch, last_key = state
        return not (last_epoch == self._refresh_epoch and last_key == key)

    def _record_card_state(self, card: ChartCard, key: tuple | None) -> None:
        self._card_render_state[card] = (self._refresh_epoch, key)

    def _next_card_fetch_token(self, card: ChartCard) -> int:
        self._card_fetch_nonce += 1
        token = self._card_fetch_nonce
        self._card_fetch_tokens[card] = token
        return token

    # @ai(gpt-5, codex, refactor, 2026-03-20)
    def _invalidate_shared_frame_cache(self) -> None:
        self._shared_frame_cache.clear()
        self._inflight_frame_fetch_waiters.clear()

    def _frame_fetch_key(self, features: list[FeatureSelection]) -> tuple:
        feature_key = tuple(
            (
                int(sel.feature_id) if sel.feature_id is not None else None,
                sel.base_name or None,
                sel.source or None,
                sel.unit or None,
                sel.type or None,
                int(sel.lag_seconds) if sel.lag_seconds is not None else 0,
            )
            for sel in features
        )
        return (feature_key, self._selector_requirements_key)

    def _request_shared_frame_for_features(
        self,
        *,
        feature_payloads: list[dict],
        frame_key: tuple,
        on_result,
        on_error,
    ) -> bool:
        cached = self._shared_frame_cache.get(frame_key)
        if isinstance(cached, pd.DataFrame):
            if callable(on_result):
                on_result(cached.copy())
            return True

        existing_waiters = self._inflight_frame_fetch_waiters.get(frame_key)
        if existing_waiters is not None:
            existing_waiters.append((on_result, on_error))
            return True

        self._inflight_frame_fetch_waiters[frame_key] = [(on_result, on_error)]

        def _drain_waiters_with_result(frame: pd.DataFrame) -> None:
            waiters = self._inflight_frame_fetch_waiters.pop(frame_key, [])
            for result_cb, _error_cb in waiters:
                if callable(result_cb):
                    result_cb(frame.copy())

        def _drain_waiters_with_error(message: str) -> None:
            waiters = self._inflight_frame_fetch_waiters.pop(frame_key, [])
            for _result_cb, error_cb in waiters:
                if callable(error_cb):
                    error_cb(message)

        def _on_fetch_result(frame_token: str) -> None:
            try:
                frame = self.sidebar.data_selector.resolve_dataframe_token(frame_token, consume=True)
                if not isinstance(frame, pd.DataFrame):
                    frame = pd.DataFrame(columns=["t"])
            except Exception as exc:
                _drain_waiters_with_error(str(exc))
                return
            self._shared_frame_cache[frame_key] = frame
            while len(self._shared_frame_cache) > 12:
                self._shared_frame_cache.pop(next(iter(self._shared_frame_cache)))
            _drain_waiters_with_result(frame)

        def _on_fetch_error(message: str) -> None:
            _drain_waiters_with_error(str(message))

        started = self.sidebar.data_selector.fetch_base_dataframe_for_features_token_async(
            feature_payloads,
            on_result=_on_fetch_result,
            on_error=_on_fetch_error,
            owner=self,
            key=("charts_shared_frame_fetch", frame_key),
            cancel_previous=True,
        )
        if started:
            return True

        self._inflight_frame_fetch_waiters.pop(frame_key, None)
        return False

    def _update_chart(self, card: ChartCard, *, force: bool = False) -> None:
        chart_type = card.chart_type()
        if chart_type == "correlation_bar":
            target = self.sidebar.selected_correlation_feature()
            selected = [f for f in card.selected_features() if isinstance(f, FeatureSelection)]

            if isinstance(target, FeatureSelection):
                self._update_correlation_chart_from_selection(
                    card,
                    target=target,
                    candidates=selected,
                    force=force,
                )
                return

            payload = self._correlation_bar_payload
            key = self._correlation_payload_key()
            if not self._should_update_card(card, key, force=force):
                return
            if not payload:
                card.show_message(tr("Select a target feature and at least one comparison feature"))
                self._record_card_state(card, key)
                return
            self._render_correlation_payload(card, payload)
            self._record_card_state(card, key)
            return

        features = card.selected_features()
        if not features:
            card.show_message(tr("Select at least one feature"))
            self._record_card_state(card, None)
            return

        selected_features = [f for f in features if isinstance(f, FeatureSelection)]
        feature_payloads = [self._feature_payload(sel) for sel in selected_features]
        key = self._card_requirements_key(
            chart_type=chart_type,
            features=selected_features,
            selector_requirements_key=self._selector_requirements_key,
        )
        if not self._should_update_card(card, key, force=force):
            return
        feature_names = [f.display_name() for f in selected_features][:MAX_FEATURES_SHOWN_LEGEND]
        title = self._build_chart_title(feature_names)
        token = self._next_card_fetch_token(card)
        frame_key = self._frame_fetch_key(selected_features)

        def _on_result(frame: pd.DataFrame) -> None:
            if self._card_fetch_tokens.get(card) != token:
                return
            try:
                if chart_type == "monthly":
                    card.set_monthly_frame(frame, title)
                elif chart_type == "time_series":
                    card.set_time_series_frame(frame, title)
                elif chart_type == "scatter":
                    self._render_scatter_chart(card, len(selected_features), frame, feature_names)
                else:
                    card.show_scatter_placeholder()
                self._record_card_state(card, key)
            except Exception as exc:
                card.show_message(tr("Failed to load chart data: {error}").format(error=exc))

        def _on_error(message: str) -> None:
            if self._card_fetch_tokens.get(card) != token:
                return
            card.show_message(tr("Failed to load chart data: {error}").format(error=str(message)))

        started = self._request_shared_frame_for_features(
            feature_payloads=feature_payloads,
            frame_key=frame_key,
            on_result=_on_result,
            on_error=_on_error,
        )
        if not started:
            card.show_message(tr("Failed to load chart data."))

    def _correlation_payload_key(self) -> tuple | None:
        payload = self._correlation_bar_payload
        if not payload:
            return None
        labels = tuple(str(item) for item in (payload.get("labels") or []))
        values = tuple(round(float(item), 8) for item in (payload.get("values") or []))
        target_name = str(payload.get("target_name") or "")
        return ("correlation_bar", target_name, labels, values)

    def _render_correlation_payload(self, card: ChartCard, payload: dict[str, object]) -> None:
        labels = payload.get("labels") or []
        values = payload.get("values") or []
        target_name = str(payload.get("target_name") or tr("Target feature"))
        card.set_group_bar_data(
            [str(item) for item in labels],
            [float(item) for item in values],
            title=tr("Feature correlations vs {feature}").format(feature=target_name),
        )

    def _update_correlation_chart_from_selection(
        self,
        card: ChartCard,
        *,
        target: FeatureSelection,
        candidates: list[FeatureSelection],
        force: bool = False,
    ) -> None:
        unique: list[FeatureSelection] = [target]
        seen: set[tuple] = {target.identity_key()}
        for feature in candidates:
            key = feature.identity_key()
            if key in seen:
                continue
            seen.add(key)
            unique.append(feature)

        if len(unique) < 2:
            card.show_message(tr("Select at least two features"))
            self._record_card_state(card, None)
            return

        key = self._card_requirements_key(
            chart_type="correlation_bar",
            features=unique,
            selector_requirements_key=self._selector_requirements_key,
        )
        if not self._should_update_card(card, key, force=force):
            return
        token = self._next_card_fetch_token(card)
        feature_payloads = [self._feature_payload(feature) for feature in unique]
        frame_key = self._frame_fetch_key(unique)

        def _on_result(frame: pd.DataFrame) -> None:
            if self._card_fetch_tokens.get(card) != token:
                return
            payload = self._build_correlation_payload_from_frame(
                target=target,
                candidates=unique,
                frame=frame,
            )
            if not payload:
                card.show_message(tr("Unable to calculate correlations for this selection"))
                self._record_card_state(card, None)
                return
            self._correlation_bar_payload = payload
            self._render_correlation_payload(card, payload)
            self._record_card_state(card, key)

        def _on_error(_message: str) -> None:
            if self._card_fetch_tokens.get(card) != token:
                return
            card.show_message(tr("Unable to calculate correlations for this selection"))
            self._record_card_state(card, None)

        started = self._request_shared_frame_for_features(
            feature_payloads=feature_payloads,
            frame_key=frame_key,
            on_result=_on_result,
            on_error=_on_error,
        )
        if not started:
            card.show_message(tr("Unable to calculate correlations for this selection"))
            self._record_card_state(card, None)

    def _build_correlation_payload_from_frame(
        self,
        *,
        target: FeatureSelection,
        candidates: list[FeatureSelection],
        frame: pd.DataFrame | None,
        limit: int | None = None,
    ) -> dict[str, object] | None:
        if frame is None or frame.empty:
            return None

        columns = [col for col in frame.columns if col != "t"]
        if len(columns) < 2:
            return None

        target_index = None
        target_key = target.identity_key()
        for idx, feature in enumerate(candidates[: len(columns)]):
            if feature.identity_key() == target_key:
                target_index = idx
                break
        if target_index is None or target_index >= len(columns):
            return None

        target_col = columns[target_index]
        mapping: dict[str, FeatureSelection] = {}
        for idx, col in enumerate(columns):
            if idx < len(candidates):
                mapping[str(col)] = candidates[idx]

        numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        target_series = numeric[target_col]
        valid_target = target_series.notna()
        if int(valid_target.sum()) < 3:
            return None

        target_aligned = target_series.loc[valid_target]
        matrix = numeric.loc[valid_target]
        if target_aligned.nunique(dropna=True) < 2:
            return None

        pair_counts = matrix.notna().sum(axis=0)
        variable_mask = matrix.nunique(dropna=True) >= 2
        matrix = matrix.loc[:, (pair_counts >= 3) & variable_mask]
        if matrix.empty:
            return None

        corr_series = matrix.corrwith(target_aligned, axis=0, drop=True)
        pair_counts = matrix.notna().sum(axis=0)
        corr_series = corr_series.drop(labels=[target_col], errors="ignore")
        corr_series = corr_series[pair_counts.reindex(corr_series.index).fillna(0) >= 3]
        corr_series = corr_series[pd.notna(corr_series)]
        corr_series = corr_series[~corr_series.isin([float("inf"), float("-inf")])]
        if corr_series.empty:
            return None

        entries: list[tuple[FeatureSelection, float]] = []
        for col, corr in corr_series.items():
            feature = mapping.get(str(col))
            if feature is None:
                continue
            entries.append((feature, float(corr)))
        if not entries:
            return None

        entries.sort(key=lambda item: abs(item[1]), reverse=True)
        shown_entries = entries if limit is None else entries[: max(0, int(limit))]
        return {
            "target_feature": target,
            "target_name": target.display_name(),
            "labels": [feature.display_name() for feature, _corr in shown_entries],
            "values": [corr for _feature, corr in shown_entries],
            "entries": [{"feature": feature, "correlation": corr} for feature, corr in shown_entries],
        }

    def _on_correlation_search_finished(self, result) -> None:
        top10 = list(getattr(result, "top10", []) or [])
        if not top10:
            set_status_text(tr("Correlation analysis produced no results."))
            toast_warn(tr("No correlations were found."), title=tr("Charts"), tab_key="charts")
            return

        target = getattr(result, "target_feature", None)
        if not isinstance(target, FeatureSelection):
            set_status_text(tr("Correlation analysis produced no results."))
            toast_warn(tr("Target feature is unavailable."), title=tr("Charts"), tab_key="charts")
            return

        self._correlation_bar_payload = {
            "target_feature": target,
            "target_name": target.display_name(),
            "labels": [entry.feature.display_name() for entry in top10],
            "values": [float(entry.correlation) for entry in top10],
            "entries": [{"feature": entry.feature, "correlation": float(entry.correlation)} for entry in top10],
        }

        if not self._chart_cards:
            self._add_chart_card()
            self._layout_chart_cards()

        self._suppress_card_updates = True
        selected_target: ChartCard | None = None
        try:
            self._chart_cards[0].set_chart_type("correlation_bar", emit_changed=False)
            selected_for_ranking = [target] + [entry.feature for entry in top10]
            self._chart_cards[0].set_selected_features(selected_for_ranking, emit_changed=False)

            correlation_features = [entry.feature for entry in top10]
            for idx, card in enumerate(self._chart_cards[1:], start=0):
                if idx >= len(correlation_features):
                    break
                card.set_chart_type("scatter", emit_changed=False)
                card.set_selected_features([target, correlation_features[idx]], emit_changed=False)
                if selected_target is None:
                    selected_target = card
        finally:
            self._suppress_card_updates = False
        self._set_selected_card(selected_target)

        self._card_render_state.clear()
        for card in self._chart_cards:
            self._update_chart(card, force=True)

        updated_scatter_count = min(max(len(self._chart_cards) - 1, 0), len(top10))
        updated_panel_count = 1 + updated_scatter_count
        toast_success(
            tr("Correlation analysis finished. Updated {count} chart panels.").format(
                count=updated_panel_count
            ),
            title=tr("Charts"),
            tab_key="charts",
        )
        set_status_text(tr("Correlation analysis finished."))

    def _on_correlation_search_failed(self, message: str) -> None:
        text = str(message).strip() if message else tr("Unknown error")
        set_status_text(tr("Correlation analysis failed: {error}").format(error=text))
        toast_error(text, title=tr("Correlation failed"), tab_key="charts")

    def _on_correlation_feature_clicked(self, category: str) -> None:
        payload = self._correlation_bar_payload or {}
        entries = payload.get("entries") or []
        target = payload.get("target_feature")
        if not isinstance(target, FeatureSelection):
            return

        selected: FeatureSelection | None = None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            feature = entry.get("feature")
            if not isinstance(feature, FeatureSelection):
                continue
            if feature.display_name() == category:
                selected = feature
                break
        if selected is None:
            return

        target_card = self._selected_card
        if target_card is None or target_card not in self._chart_cards:
            return
        target_card.set_selected_features([target, selected], emit_changed=False)
        self._update_chart(target_card, force=True)

    def _render_scatter_chart(
        self,
        card: ChartCard,
        feature_count: int,
        frame: pd.DataFrame,
        feature_names: list[str],
    ) -> None:
        if feature_count not in (2, 3):
            card.show_message(tr("Scatter charts require exactly 2 or 3 features"))
            return

        if frame is None or frame.empty:
            card.show_message(tr("No data for the selected filters"))
            return

        columns = self._feature_columns(frame, feature_count)
        if len(columns) != feature_count:
            card.show_message(tr("Scatter data unavailable for the selected features"))
            return

        card.set_scatter_frame(
            frame,
            columns=columns,
            labels=feature_names[:feature_count],
        )

    # ------------------------------------------------------------------
    def _build_chart_title(self, feature_names: list[str], *, max_total: int = 72, max_each: int = 28) -> str:
        if not feature_names:
            return tr("Chart")

        parts: list[str] = []
        hidden = 0
        current_len = 0

        for name in feature_names:
            label = str(name).strip()
            if not label:
                continue
            if len(label) > max_each:
                label = label[: max_each - 1] + "…"

            addition = (", " if parts else "") + label
            if parts and current_len + len(addition) > max_total:
                hidden += 1
                continue
            parts.append(label)
            current_len += len(addition)

        if hidden:
            parts.append(tr("+{count} more").format(count=hidden))

        return ", ".join(parts) if parts else tr("Chart")

    # ------------------------------------------------------------------
    @staticmethod
    def _feature_columns(frame: pd.DataFrame | None, count: int) -> list[str]:
        if frame is None or frame.empty:
            return []
        columns = [col for col in frame.columns if col != "t"]
        return columns[:count]

    # ------------------------------------------------------------------
    @staticmethod
    def _feature_payload(selection: FeatureSelection) -> dict:
        return {
            "feature_id": selection.feature_id,
            "notes": selection.label,
            "name": selection.base_name,
            "source": selection.source,
            "unit": selection.unit,
            "type": selection.type,
            "lag_seconds": selection.lag_seconds,
        }

    def _aggregate_monthly_export_frame(self, frame: pd.DataFrame | None) -> tuple[pd.DataFrame, str]:
        if frame is None or frame.empty or "t" not in frame.columns:
            return pd.DataFrame(), "M"

        df = frame.copy()
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df = df.dropna(subset=["t"]).sort_values("t")
        if df.empty:
            return pd.DataFrame(), "M"

        feature_cols = [c for c in df.columns if c != "t"][:MAX_FEATURES_SHOWN_LEGEND]
        if not feature_cols:
            return pd.DataFrame(), "M"

        t_valid = df["t"][df["t"].notna()]
        level = "M"
        if not t_valid.empty:
            nunique_months = t_valid.dt.to_period("M").nunique()
            if nunique_months <= 1:
                nunique_days = t_valid.dt.to_period("D").nunique()
                if nunique_days <= 1:
                    if t_valid.dt.to_period("h").nunique() >= 1:
                        level = "h"
                    else:
                        level = "D"
                else:
                    level = "D"

        freq_map = {"A": "A", "M": "M", "W": "W", "D": "D", "h": "h"}
        freq = freq_map.get(level, "M")

        long_df = df[["t"] + feature_cols].melt(id_vars="t", var_name="feature", value_name="value")
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
        long_df = long_df.dropna(subset=["value"])
        if long_df.empty:
            return pd.DataFrame(), level

        long_df["period"] = long_df["t"].dt.to_period(freq)
        grouped = long_df.groupby(["period", "feature"], as_index=False)["value"].mean()
        grouped = grouped.sort_values(["period", "feature"])

        start_per = grouped["period"].min()
        end_per = grouped["period"].max()
        full = pd.period_range(start=start_per, end=end_per, freq=freq)
        features = sorted(grouped["feature"].unique().tolist())

        pivot = (
            grouped.pivot(index="period", columns="feature", values="value")
            .reindex(full)
            .reindex(columns=features)
        )

        labels = [self._format_period_label(level, per) for per in pivot.index]
        export_df = pivot.reset_index(drop=True)
        export_df.insert(0, tr("Period"), labels)
        return export_df, level

    @staticmethod
    def _format_period_label(level_code: str, period: pd.Period) -> str:
        try:
            start = period.start_time
        except Exception:
            start = pd.Timestamp(period.to_timestamp())
        if level_code == "A":
            return start.strftime("%Y")
        if level_code == "M":
            return start.strftime("%Y-%m")
        if level_code == "W":
            iso = start.isocalendar()
            return f"{iso.year}-W{int(iso.week):02d}"
        if level_code == "D":
            return start.strftime("%Y-%m-%d")
        return start.strftime("%Y-%m-%d %H:00")

    @staticmethod
    def _correlation_export_frame(payload: dict[str, object] | None) -> pd.DataFrame:
        if not isinstance(payload, dict):
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        entries = payload.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                feature = entry.get("feature")
                try:
                    label = feature.display_name() if feature is not None else ""
                except Exception:
                    label = ""
                label = str(label or "").strip()
                if not label:
                    continue
                corr_value = pd.to_numeric(entry.get("correlation"), errors="coerce")
                if pd.isna(corr_value):
                    continue
                rows.append({"Feature": label, "Correlation": float(corr_value)})

        if not rows:
            labels = payload.get("labels")
            values = payload.get("values")
            if isinstance(labels, list) and isinstance(values, list):
                for label, value in zip(labels, values):
                    text = str(label or "").strip()
                    corr_value = pd.to_numeric(value, errors="coerce")
                    if not text or pd.isna(corr_value):
                        continue
                    rows.append({"Feature": text, "Correlation": float(corr_value)})

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=["Feature", "Correlation"])

    @staticmethod
    def _prepare_time_series_export_frame(frame: pd.DataFrame | None, *, feature_count: int) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        feature_cols = ChartsTab._feature_columns(frame, feature_count)
        if not feature_cols:
            feature_cols = [col for col in frame.columns if col != "t"]
        export_columns = []
        if "t" in frame.columns:
            export_columns.append("t")
        export_columns.extend(feature_cols)
        export_columns = [col for col in export_columns if col in frame.columns]
        if not export_columns:
            return pd.DataFrame()
        out = frame.loc[:, export_columns].copy()
        if "t" in out.columns:
            out["t"] = pd.to_datetime(out["t"], errors="coerce")
            out = out.dropna(subset=["t"]).sort_values("t", kind="stable")
            out = out.rename(columns={"t": tr("Date")})
        return out.reset_index(drop=True)

    @staticmethod
    def _prepare_scatter_export_frame(
        frame: pd.DataFrame | None,
        *,
        feature_count: int,
    ) -> tuple[pd.DataFrame, list[str]]:
        if frame is None or frame.empty:
            return pd.DataFrame(), []
        scatter_cols = ChartsTab._feature_columns(frame, feature_count)
        if len(scatter_cols) not in (2, 3):
            return pd.DataFrame(), []
        out = frame.loc[:, scatter_cols].copy()
        for col in scatter_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=scatter_cols).reset_index(drop=True)
        if out.empty:
            return pd.DataFrame(), []
        return out, list(scatter_cols)

    # @ai(gpt-5, codex, refactor, 2026-03-12)
    def _export_results(self) -> None:
        export_items: dict[str, dict[str, object]] = {}
        for idx, card in enumerate(self._chart_cards, start=1):
            chart_type = card.chart_type()
            if chart_type == "correlation_bar":
                corr_df = self._correlation_export_frame(self._correlation_bar_payload)
                if corr_df.empty:
                    continue
                name = tr("Chart {index} (correlation ranking)").format(index=idx)
                export_items[name] = {
                    "kind": "correlation",
                    "frame": corr_df,
                }
                continue
            if chart_type not in {"monthly", "time_series", "scatter"}:
                continue
            selected_features = [f for f in card.selected_features() if isinstance(f, FeatureSelection)]
            if not selected_features:
                continue
            name = tr("Chart {index} ({kind})").format(index=idx, kind=chart_type)
            export_items[name] = {
                "kind": "card_fetch",
                "index": idx,
                "chart_type": chart_type,
                "selected_features": selected_features,
                "feature_payloads": [self._feature_payload(sel) for sel in selected_features],
            }

        self._run_export_dialog(export_items)

    # @ai(gpt-5, codex, refactor, 2026-03-12)
    def _run_export_dialog(
        self,
        export_items: dict[str, dict[str, object]],
    ) -> None:
        if not export_items:
            toast_info(tr("No chart data available to export."), title=tr("Charts"), tab_key="charts")
            return

        dialog = ExportSelectionDialog(
            title=tr("Export chart results"),
            heading=tr("Choose which chart datasets to export."),
            options=[ExportOption(key=k, label=k) for k in export_items],
            show_chart_data_options=True,
            chart_label="Charts",
            data_label="Data",
            parent=self,
        )
        if dialog.exec() != ExportSelectionDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_keys()
        if not selected:
            toast_info(tr("Select at least one dataset to export."), title=tr("Export"), tab_key="charts")
            return
        include_charts = dialog.include_charts()
        include_data = dialog.include_data()
        if not include_charts and not include_data:
            toast_info(tr("Select charts and/or data to export."), title=tr("Export"), tab_key="charts")
            return

        chosen_items = {name: export_items[name] for name in selected if name in export_items}
        if not chosen_items:
            toast_info(tr("No chart data available to export."), title=tr("Charts"), tab_key="charts")
            return

        selected_format = dialog.selected_format()
        destination_plan = self._prepare_export_destination_plan(
            selected_format=selected_format,
            include_charts=include_charts,
            include_data=include_data,
        )
        if destination_plan is None:
            return
        self._collect_and_export_selected_items(
            chosen_items=chosen_items,
            selected_format=selected_format,
            include_charts=include_charts,
            include_data=include_data,
            destination_plan=destination_plan,
        )

    def _prepare_export_destination_plan(
        self,
        *,
        selected_format: str,
        include_charts: bool,
        include_data: bool,
    ) -> ExportPlan | None:
        # File selection should happen before any data fetch work starts.
        placeholder = {tr("Selection"): pd.DataFrame({"Value": [0]})}
        if selected_format == "excel":
            return prepare_charts_excel_export_plan(
                parent=self,
                title=tr("Export charts"),
                datasets=placeholder,
                chart_specs={},
                include_charts=include_charts,
                include_data=include_data,
                chart_first=True,
            )
        return prepare_dataframes_export_plan(
            parent=self,
            title=tr("Export charts"),
            selected_format=selected_format,
            datasets=placeholder,
        )

    def _collect_and_export_selected_items(
        self,
        *,
        chosen_items: dict[str, dict[str, object]],
        selected_format: str,
        include_charts: bool,
        include_data: bool,
        destination_plan: ExportPlan,
    ) -> None:
        export_button = getattr(self.sidebar, "btn_export", None)
        if export_button is not None:
            export_button.setEnabled(False)
        set_status_text(tr("Fetching chart data..."))
        toast_info(tr("Fetching chart data..."), title=tr("Charts"), tab_key="charts")

        datasets: dict[str, pd.DataFrame] = {}
        chart_specs: dict[str, dict[str, object]] = {}
        fetch_requests: list[tuple[str, int, str, list[FeatureSelection], list[dict]]] = []

        for name, item in chosen_items.items():
            kind = str(item.get("kind") or "")
            if kind == "correlation":
                frame = item.get("frame")
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    datasets[name] = frame.copy()
                    chart_specs[name] = {
                        "type": "monthly",
                        "x_column": "Feature",
                        "y_columns": ["Correlation"],
                        "title": name,
                    }
                continue
            if kind != "card_fetch":
                continue
            idx = int(item.get("index") or 0)
            chart_type = str(item.get("chart_type") or "")
            selected_features = [
                f for f in (item.get("selected_features") or []) if isinstance(f, FeatureSelection)
            ]
            feature_payloads = [dict(p) for p in (item.get("feature_payloads") or []) if isinstance(p, dict)]
            if idx <= 0 or chart_type not in {"monthly", "time_series", "scatter"}:
                continue
            if not selected_features or not feature_payloads:
                continue
            fetch_requests.append((name, idx, chart_type, selected_features, feature_payloads))

        request_index = 0

        def _finish_collection() -> None:
            if not datasets:
                if export_button is not None:
                    export_button.setEnabled(True)
                set_status_text(tr("Chart export warning."))
                toast_warn(tr("No chart data available to export."), title=tr("Export"), tab_key="charts")
                return
            set_status_text(tr("Chart data fetched. Exporting..."))
            self._execute_chart_export_plan(
                datasets=datasets,
                chart_specs=chart_specs,
                selected_format=selected_format,
                include_charts=include_charts,
                include_data=include_data,
                destination_plan=destination_plan,
                export_button=export_button,
            )

        def _process_frame(
            *,
            name: str,
            chart_type: str,
            selected_features: list[FeatureSelection],
            frame: pd.DataFrame | None,
        ) -> None:
            if frame is None or frame.empty:
                return
            try:
                scatter_cols: list[str] = []
                if chart_type == "monthly":
                    frame, _chart_level = self._aggregate_monthly_export_frame(frame)
                elif chart_type == "time_series":
                    frame = self._prepare_time_series_export_frame(frame, feature_count=len(selected_features))
                elif chart_type == "scatter":
                    frame, scatter_cols = self._prepare_scatter_export_frame(
                        frame,
                        feature_count=len(selected_features),
                    )
                else:
                    return
            except Exception:
                return
            if frame is None or frame.empty:
                return
            datasets[name] = frame.copy()
            columns = [str(c) for c in frame.columns]
            if chart_type == "scatter":
                if len(scatter_cols) == 2:
                    chart_specs[name] = {
                        "type": "scatter",
                        "x_column": scatter_cols[0],
                        "y_columns": [scatter_cols[1]],
                        "title": name,
                    }
                return
            x_col = tr("Date") if tr("Date") in columns else ("t" if "t" in columns else (columns[0] if columns else ""))
            y_cols = [c for c in columns if c and c != x_col]
            chart_specs[name] = {
                "type": chart_type,
                "x_column": x_col,
                "y_columns": y_cols,
                "title": name,
            }

        def _fetch_next() -> None:
            nonlocal request_index
            if request_index >= len(fetch_requests):
                _finish_collection()
                return
            name, idx, chart_type, selected_features, feature_payloads = fetch_requests[request_index]
            request_index += 1

            def _on_result(frame_token: str) -> None:
                frame = self.sidebar.data_selector.resolve_dataframe_token(frame_token, consume=True)
                _process_frame(
                    name=name,
                    chart_type=chart_type,
                    selected_features=selected_features,
                    frame=frame,
                )
                _fetch_next()

            def _on_error(_message: str) -> None:
                _fetch_next()

            started = self.sidebar.data_selector.fetch_base_dataframe_for_features_token_async(
                feature_payloads,
                on_result=_on_result,
                on_error=_on_error,
                owner=self,
                key=("charts_export_fetch", idx),
                cancel_previous=True,
            )
            if not started:
                _fetch_next()

        _fetch_next()

    def _execute_chart_export_plan(
        self,
        *,
        datasets: dict[str, pd.DataFrame],
        chart_specs: dict[str, dict[str, object]],
        selected_format: str,
        include_charts: bool,
        include_data: bool,
        destination_plan: ExportPlan,
        export_button,
    ) -> None:
        if selected_format == "excel":
            chosen_specs = {name: chart_specs.get(name, {}) for name in datasets}
            plan = ExportPlan(
                kind="charts_excel",
                selected_format="excel",
                destination=destination_plan.destination,
                datasets=datasets,
                chart_specs=chosen_specs,
                include_charts=include_charts,
                include_data=include_data,
                chart_first=True,
            )
        else:
            plan = ExportPlan(
                kind="dataframes",
                selected_format=selected_format,
                destination=destination_plan.destination,
                datasets=datasets,
            )

        set_status_text(tr("Exporting charts..."))
        toast_info(tr("Exporting charts..."), title=tr("Charts"), tab_key="charts")

        def _on_export_finished(outcome) -> None:
            if export_button is not None:
                export_button.setEnabled(True)
            if not getattr(outcome, "message", ""):
                return
            if bool(getattr(outcome, "ok", False)):
                set_status_text(tr("Chart export finished."))
                toast_success(tr("Chart export finished."), title=tr("Charts"), tab_key="charts")
                open_path = getattr(outcome, "open_path", None)
                if open_path:
                    toast_file_saved(open_path, title=tr("Export saved"), tab_key="charts")
                return
            set_status_text(tr("Chart export warning."))
            toast_warn(str(outcome.message), title=tr("Export"), tab_key="charts")

        def _on_export_error(message: str) -> None:
            if export_button is not None:
                export_button.setEnabled(True)
            set_status_text(tr("Chart export failed."))
            text = tr("Chart export failed: {error}").format(error=str(message or tr("Unknown error")))
            toast_error(text, title=tr("Export failed"), tab_key="charts")

        run_in_thread(
            execute_export_plan,
            on_result=lambda outcome: run_in_main_thread(_on_export_finished, outcome),
            on_error=lambda message: run_in_main_thread(_on_export_error, message),
            plan=plan,
            owner=self,
            key="charts_export",
            cancel_previous=True,
        )
