
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import time

import pandas as pd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QLabel,
    QSplitter,
    QWidget,
)
from ...localization import tr

from . import MAX_FEATURES_SHOWN_LEGEND
from ...models.hybrid_pandas_model import FeatureSelection, HybridPandasModel
from .viewmodel import ChartsViewModel

from .chart_card import ChartCard
from .charts_sidebar import ChartsSidebar
from ...utils.exporting import export_dataframes, export_charts_excel
from ...utils import set_status_text, toast_error, toast_info, toast_success, toast_warn
from ...widgets.export_dialog import ExportOption, ExportSelectionDialog
from ...widgets.panel import Panel
from ..tab_widget import TabWidget

class ChartsTab(TabWidget):
    """Interactive charts workspace with configurable chart cards."""

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

    def _create_content_widget(self) -> QWidget:
        right_panel = Panel("", parent=self)
        right_layout = right_panel.content_layout()

        header = QLabel(tr("Configure filters on the left and choose chart types/features for each panel."))
        header.setWordWrap(True)
        right_layout.addWidget(header)

        self._chart_splitter = QSplitter(Qt.Orientation.Vertical, right_panel)
        self._chart_splitter.setChildrenCollapsible(False)
        right_layout.addWidget(self._chart_splitter, 1)

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
        self._debounce.stop()
        self._selection_sync_fallback.start()

    def _on_selector_data_requirements_changed(self, _requirements: dict) -> None:
        requirements = _requirements if isinstance(_requirements, dict) else {}
        preprocessing_key = self._freeze_value(requirements.get("preprocessing", {}))
        self._selector_requirements_key = self._freeze_value(
            {
                "filters": requirements.get("filters", {}),
                "preprocessing": requirements.get("preprocessing", {}),
            }
        )
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
        while len(self._chart_cards) < count:
            self._add_chart_card()
        self._layout_chart_cards()

    def _set_chart_count_exact(self, count: int) -> None:
        count = max(1, int(count))
        while len(self._chart_cards) < count:
            self._add_chart_card()
        while len(self._chart_cards) > count:
            card = self._chart_cards.pop()
            self._card_render_state.pop(card, None)
            if self._selected_card is card:
                self._selected_card = None
            card.set_view_model(None)
            card.setParent(None)
            card.deleteLater()
        if self._selected_card not in self._chart_cards:
            fallback = self._chart_cards[0] if self._chart_cards else None
            self._set_selected_card(fallback)
        self._layout_chart_cards()

    def _on_add_chart_requested(self) -> None:
        self._add_chart_card()
        self._layout_chart_cards()
        self._refresh_all_charts()

    def _add_chart_card(self) -> None:
        card = ChartCard(parent=self, view_model=self._view_model)
        self._chart_cards.append(card)
        items = self.sidebar.available_feature_items()
        card.set_available_features(items)
        card.correlation_feature_clicked.connect(self._on_correlation_feature_clicked)
        card.selection_requested.connect(self._on_card_selection_requested)
        if self._selected_card is None:
            self._set_selected_card(card)

    def _on_remove_chart_requested(self) -> None:
        if not self._chart_cards:
            return
        card = self._chart_cards.pop()
        self._card_render_state.pop(card, None)
        if self._selected_card is card:
            self._selected_card = None
        card.set_view_model(None)
        card.setParent(None)
        card.deleteLater()
        if self._selected_card not in self._chart_cards:
            fallback = self._chart_cards[0] if self._chart_cards else None
            self._set_selected_card(fallback)
        self._layout_chart_cards()
        self._refresh_all_charts()

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
        # remove existing row splitters from the vertical splitter
        for splitter in self._row_splitters:
            splitter.setParent(None)
            splitter.deleteLater()
        self._row_splitters = []

        while self._chart_splitter.count():
            widget = self._chart_splitter.widget(0)
            if widget is not None:
                widget.setParent(None)

        if not self._chart_cards:
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

        for index in range(self._chart_splitter.count()):
            self._chart_splitter.setStretchFactor(index, 1)
        self._chart_splitter.setSizes([1] * self._chart_splitter.count())

    def _add_row(self, row_cards: list[ChartCard], *, full_width: bool) -> None:
        row_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        row_splitter.setChildrenCollapsible(False)
        for idx, card in enumerate(row_cards):
            row_splitter.addWidget(card)
            row_splitter.setStretchFactor(idx, 1)
        if not full_width and len(row_cards) > 1:
            row_splitter.setSizes([1, 1])
        self._chart_splitter.addWidget(row_splitter)
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

        try:
            frame = self.sidebar.data_selector.fetch_base_dataframe_for_features(feature_payloads)
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

        payload = self._build_correlation_payload(target=target, candidates=unique)
        if not payload:
            card.show_message(tr("Unable to calculate correlations for this selection"))
            self._record_card_state(card, None)
            return

        self._correlation_bar_payload = payload
        self._render_correlation_payload(card, payload)
        self._record_card_state(card, key)

    def _build_correlation_payload(
        self,
        *,
        target: FeatureSelection,
        candidates: list[FeatureSelection],
        limit: int | None = None,
    ) -> dict[str, object] | None:
        feature_payloads = [self._feature_payload(feature) for feature in candidates]
        frame = self.sidebar.data_selector.fetch_base_dataframe_for_features(feature_payloads)
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
            .fillna(0.0)
        )

        labels = [self._format_period_label(level, per) for per in pivot.index]
        export_df = pivot.reset_index(drop=True)
        export_df.insert(0, "label", labels)
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

    def _export_results(self) -> None:
        datasets: dict[str, pd.DataFrame] = {}
        chart_specs: dict[str, dict[str, object]] = {}
        for idx, card in enumerate(self._chart_cards, start=1):
            features = card.selected_features()
            if not features:
                continue
            selected_features = [f for f in features if isinstance(f, FeatureSelection)]
            if not selected_features:
                continue
            feature_payloads = [self._feature_payload(sel) for sel in selected_features]
            df = self.sidebar.data_selector.fetch_base_dataframe_for_features(feature_payloads)
            if df is None or df.empty:
                continue
            chart_type = card.chart_type()
            try:
                if chart_type == "monthly":
                    df, chart_level = self._aggregate_monthly_export_frame(df)
                elif chart_type == "scatter":
                    pass
                else:
                    continue
            except Exception:
                continue
            if df is None or df.empty:
                continue
            name = tr("Chart {index} ({kind})").format(index=idx, kind=chart_type)
            datasets[name] = df.copy()
            columns = [str(c) for c in df.columns]
            if chart_type == "scatter":
                feature_count = len(selected_features)
                scatter_cols = self._feature_columns(df, feature_count)
                if len(scatter_cols) == 2:
                    chart_specs[name] = {
                        "type": "scatter",
                        "x_column": scatter_cols[0],
                        "y_columns": [scatter_cols[1]],
                        "title": name,
                    }
            else:
                x_col = "t" if "t" in columns else (columns[0] if columns else "")
                y_cols = [c for c in columns if c and c != x_col]
                chart_specs[name] = {
                    "type": chart_type,
                    "x_column": x_col,
                    "y_columns": y_cols,
                    "title": name,
                }

        if not datasets:
            toast_info(tr("No chart data available to export."), title=tr("Charts"), tab_key="charts")
            return

        dialog = ExportSelectionDialog(
            title=tr("Export chart results"),
            heading=tr("Choose which chart datasets to export."),
            options=[ExportOption(key=k, label=k) for k in datasets],
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
        chosen = {name: datasets[name] for name in selected if name in datasets}
        selected_format = dialog.selected_format()
        if selected_format == "excel":
            chosen_specs = {name: chart_specs.get(name, {}) for name in chosen}
            ok, message = export_charts_excel(
                parent=self,
                title=tr("Export charts"),
                datasets=chosen,
                chart_specs=chosen_specs,
                include_charts=include_charts,
                include_data=include_data,
                chart_first=True,
            )
        else:
            ok, message = export_dataframes(
                parent=self,
                title=tr("Export charts"),
                selected_format=selected_format,
                datasets=chosen,
            )
        if not message:
            return
        if ok:
            toast_success(message, title=tr("Export complete"), tab_key="charts")
        else:
            toast_warn(message, title=tr("Export"), tab_key="charts")
