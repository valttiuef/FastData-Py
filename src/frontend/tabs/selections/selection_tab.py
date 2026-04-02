
from __future__ import annotations
# --- @ai START ---
# model: gpt-5
# tool: codex-cli
# role: behavior-update
# reviewed: yes
# date: 2026-03-02
# --- @ai END ---
from typing import List, Optional

import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QAbstractItemView, QComboBox, QInputDialog, QLabel, QLineEdit, QMessageBox, QStyledItemDelegate, QWidget
from ...localization import tr

from ...models.database_model import DatabaseModel
from ...utils import toast_error, toast_info, toast_success, toast_warn
from ...models.selection_settings import (
    FILTER_SCOPE_CHOICES,
    SELECTION_MODE_EXCLUDE,
    SelectionSettingsPayload,
    normalize_filter_scope,
)
from ...widgets.fast_table import FastPandasProxyModel, FastTable
from ...widgets.panel import Panel
from ..tab_widget import TabWidget
from .models import FeatureSelectionTableModel
from .feature_group_conversion_dialog import FeatureGroupConversionDialog
from .sidebar import Sidebar
from .viewmodel import SelectionsViewModel
import logging

logger = logging.getLogger(__name__)


def _default_selection_payload() -> SelectionSettingsPayload:
    # @ai(gpt-5, codex-cli, fix, 2026-03-10)
    """Default payload for the built-in startup selection state."""
    return SelectionSettingsPayload(
        include_selections=True,
        include_filters=False,
        include_preprocessing=False,
    )


class FilterScopeDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        for scope in FILTER_SCOPE_CHOICES:
            editor.addItem(scope.title(), scope)
        return editor

    def setEditorData(self, editor, index):
        current = normalize_filter_scope(index.model().data(index, Qt.ItemDataRole.EditRole))
        current_index = editor.findData(current)
        editor.setCurrentIndex(max(0, current_index))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentData(), Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

class SelectionsTab(TabWidget):
    def __init__(self, database_model: DatabaseModel, parent: Optional[QWidget] = None) -> None:
        self._database_model = database_model
        self._table_model = FeatureSelectionTableModel()
        self._current_payload = _default_selection_payload()
        self._settings: List[dict] = []
        self._settings_with_default: List[dict] = []
        self._select_all_default = True
        self._pending_toast_action: Optional[str] = None
        self._group_filters_refresh_pending = False
        self._features_refresh_pending = False
        self._full_refresh_pending = False
        super().__init__(parent)

        self._table_model.setParent(self)
        self._view_model = SelectionsViewModel(self._database_model, parent=self)

        self._connect_signals()

        # Important:
        # Do NOT manually refresh sidebar.data_selector here.
        # Its own DataSelectorViewModel already runs refresh_from_model() during
        # construction. A second startup refresh here is selections-tab-specific
        # and can overwrite the final scoped datasets/imports with all items.
        self._view_model.refresh()

        self._database_model.database_changed.connect(self._on_database_changed)
        self._database_model.selection_database_changed.connect(self._on_selection_database_changed)
        self._database_model.features_list_changed.connect(self._on_features_list_changed)

    # ------------------------------------------------------------------
    def _create_sidebar(self) -> QWidget:
        self.sidebar = Sidebar(self, model=self._database_model)
        return self.sidebar

    def _create_content_widget(self) -> QWidget:
        panel = Panel(title=tr("Features"))
        layout = panel.content_layout()
        self.title = QLabel(tr("Select features and manage selection settings."), panel)
        self.title.setObjectName("DataInfo")
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        self._search_edit = QLineEdit(panel)
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.setPlaceholderText(tr("Search features…"))
        layout.addWidget(self._search_edit)
        self._proxy_model = FastPandasProxyModel(self)
        self._proxy_model.setSourceModel(self._table_model)
        self.features_table = FastTable(
            select="rows",
            single_selection=False,
            tint_current_selection=True,
            editable=True,
            initial_uniform_column_widths=True,
            min_column_width=120,
            context_menu_builder=self._build_table_context_menu,
            parent=panel,
        )
        # Selections editing should not start from AnyKeyPressed; that path can
        # trigger Qt editor ownership warnings on Enter commit in this table.
        self.features_table.setEditTriggers(
            QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.DoubleClicked
        )
        self.features_table.setModel(self._proxy_model)
        filter_scope_column = next(
            (idx for idx, (key, _label) in enumerate(self._table_model.COLUMN_DEFINITIONS) if key == "filter_scope"),
            -1,
        )
        if filter_scope_column >= 0:
            self.features_table.setItemDelegateForColumn(filter_scope_column, FilterScopeDelegate(self.features_table))
        self.features_table.sortByColumn(1, Qt.SortOrder.AscendingOrder)  # Id
        layout.addWidget(self.features_table, 1)
        return panel

    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self._view_model.features_changed.connect(self._on_features_changed)
        self._view_model.settings_changed.connect(self._on_settings_loaded)
        self._view_model.active_setting_changed.connect(self._on_active_setting_changed)
        self._view_model.feature_group_preview_ready.connect(self._on_feature_group_preview_ready)
        self._view_model.feature_group_conversion_done.connect(self._on_feature_group_conversion_done)
        self._view_model.features_save_completed.connect(self._on_features_save_completed)
        self._view_model.error_occurred.connect(self._log_error)

        self.sidebar.remove_button.clicked.connect(self._on_remove_feature)
        self.sidebar.reload_button.clicked.connect(self._on_reload_features)
        self.sidebar.save_button.clicked.connect(self._on_save_features)
        self.sidebar.load_setting_button.clicked.connect(self._on_load_setting)
        self.sidebar.save_setting_button.clicked.connect(self._on_save_setting)
        self.sidebar.delete_setting_button.clicked.connect(self._on_delete_setting)
        self.sidebar.settings_tree.currentItemChanged.connect(self._on_setting_list_changed)
        self.sidebar.load_db_button.clicked.connect(self._on_load_settings_database)
        self.sidebar.save_db_button.clicked.connect(self._on_save_settings_database)
        self.sidebar.reset_db_button.clicked.connect(self._on_reset_settings_database)
        if self.sidebar.filters_widget is not None:
            self.sidebar.filters_widget.filters_changed.connect(self._on_scope_filters_changed)
        self._search_edit.textChanged.connect(self._on_search_text_changed)
        self._table_model.dataChanged.connect(self._on_table_data_changed)

    def _on_search_text_changed(self, text: str) -> None:
        self._proxy_model.set_filter(text, columns=None, case_sensitive=False, debounce_ms=0)

    def _invoke_main_window_action(self, method_name: str) -> None:
        win = self.window() or self.parent()
        try:
            handler = getattr(win, method_name, None) if win is not None else None
            if callable(handler):
                handler()
        except Exception:
            logger.warning("Exception in _invoke_main_window_action", exc_info=True)

    # ------------------------------------------------------------------
    def _on_database_changed(self, *_args) -> None:
        # DataSelectorWidget already listens to DatabaseModel signals directly.
        # Coalesce bursty DB change notifications to one Selections refresh pass.
        if self._full_refresh_pending:
            return
        self._full_refresh_pending = True
        QTimer.singleShot(0, self._flush_full_refresh)

    def _flush_full_refresh(self) -> None:
        self._full_refresh_pending = False
        self._view_model.refresh()

    def _on_selection_database_changed(self, *_args) -> None:
        self._view_model._refresh_settings()

    def _on_features_list_changed(self) -> None:
        # DataSelectorWidget and filter widgets already subscribe to
        # features_list_changed. Keep Selections tab refresh coalesced to avoid
        # stacked duplicate refresh work during rapid feature-update bursts.
        if self._features_refresh_pending:
            return
        self._features_refresh_pending = True
        QTimer.singleShot(0, self._flush_features_refresh)

    def _flush_features_refresh(self) -> None:
        self._features_refresh_pending = False
        self._view_model.refresh_features()

    def _on_scope_filters_changed(self, *_args) -> None:
        fw = self.sidebar.filters_widget
        if fw is None:
            return
        self._view_model.set_feature_scope_filters(
            systems=fw.selected_systems(),
            datasets=fw.selected_datasets(),
            import_ids=fw.selected_import_ids(),
            tags=fw.selected_tags(),
        )

    def _on_features_changed(self, df) -> None:
        self._table_model.set_features(df)
        self._table_model.apply_selection(self._current_payload, select_all_by_default=self._select_all_default)
        self._reapply_table_sort()
        self._queue_group_filter_refresh()
        if self._pending_toast_action == "reload_features":
            self._pending_toast_action = None
            try:
                toast_success(tr("Features reloaded from database."), title=tr("Selections"), tab_key="selections")
            except Exception:
                logger.warning("Exception in _on_features_changed", exc_info=True)

    def _reapply_table_sort(self) -> None:
        """Re-run current sort after model updates when dynamic sort is disabled."""
        header = self.features_table.horizontalHeader()
        column = header.sortIndicatorSection() if header is not None else -1
        if column is None or int(column) < 0:
            column = 1  # Id column
        order = header.sortIndicatorOrder() if header is not None else Qt.SortOrder.AscendingOrder
        QTimer.singleShot(0, lambda: self._proxy_model.sort(int(column), order))

    def _default_record(self) -> dict:
        return dict(
            id=None,
            name=tr("Default"),
            notes="",
            created_at="",
            auto_load=False,
            is_active=False,
            payload=_default_selection_payload(),
        )

    def _on_settings_loaded(self, records: List[dict]) -> None:
        self._settings = list(records)
        combined = [self._default_record()] + self._settings
        active_index = next((idx for idx, rec in enumerate(combined) if rec.get("is_active")), -1)
        if active_index < 0:
            combined[0]["is_active"] = True
            active_index = 0
        self._settings_with_default = combined
        self.sidebar.set_settings(combined, active_index=active_index)
        self._on_setting_list_changed()

        if self._pending_toast_action in {"save_setting", "delete_setting"}:
            action = self._pending_toast_action
            self._pending_toast_action = None
            try:
                if action == "delete_setting":
                    toast_success(tr("Selection setting deleted."), title=tr("Selections"), tab_key="selections")
                else:
                    toast_success(tr("Selection setting saved."), title=tr("Selections"), tab_key="selections")
            except Exception:
                logger.warning("Exception in _on_settings_loaded", exc_info=True)

    def _on_active_setting_changed(self, record: Optional[dict]) -> None:
        payload = record.get("payload") if record else _default_selection_payload()
        if payload is None:
            payload = _default_selection_payload()
        if not isinstance(payload, SelectionSettingsPayload):
            payload = SelectionSettingsPayload.from_dict(payload if isinstance(payload, dict) else {})
            if not record:
                payload = _default_selection_payload()

        self._current_payload = payload
        self._select_all_default = not bool(record and record.get("id")) or not payload.selections_enabled()

        self._view_model.set_apply_scope_filters(payload.filters_enabled())
        self._table_model.apply_selection(payload, select_all_by_default=self._select_all_default)

        name = record.get("name") if record else ""
        notes = record.get("notes") if record else ""
        self.sidebar.set_setting_name(name or "")
        self.sidebar.set_setting_notes(notes or "")
        self.sidebar.set_payload_options(
            include_selections=payload.selections_enabled(),
            include_filters=payload.filters_enabled(),
            include_preprocessing=payload.preprocessing_enabled(),
        )

        target_id = record.get("id") if record else None
        self._select_setting_in_list(target_id)

        # Important:
        # Let DatabaseModel selection state be the single source of truth.
        # Do not apply sidebar selector settings directly here, and do not manually
        # trigger scope-filter handling from the current widget state. At startup
        # that current widget state can still be the unscoped/all-data one.
        payload_to_apply = payload if (record and record.get("id")) else None
        self._apply_active_selection_payload(payload_to_apply)

        self._queue_group_filter_refresh()

        if self._pending_toast_action == "activate_setting":
            self._pending_toast_action = None
            try:
                toast_success(tr("Selection setting activated."), title=tr("Selections"))
            except Exception:
                logger.warning("Exception in _on_active_setting_changed", exc_info=True)

    def _queue_group_filter_refresh(self) -> None:
        if self._group_filters_refresh_pending:
            return
        self._group_filters_refresh_pending = True
        QTimer.singleShot(0, self._flush_group_filter_refresh)

    def _flush_group_filter_refresh(self) -> None:
        self._group_filters_refresh_pending = False
        self._refresh_group_filters_for_enabled_group_features()

    def _apply_active_selection_payload(self, payload: Optional[SelectionSettingsPayload]) -> None:
        try:
            if payload is None:
                self._database_model.apply_selection_payload(None)
                return
            self._database_model.apply_selection_payload(SelectionSettingsPayload.from_dict(payload.to_dict()))
        except Exception:
            logger.warning("Exception in _apply_active_selection_payload", exc_info=True)

    def _select_setting_in_list(self, setting_id: Optional[int]) -> None:
        count = self.sidebar.settings_tree.topLevelItemCount()
        for idx in range(count):
            item = self.sidebar.settings_tree.topLevelItem(idx)
            if not item:
                continue
            data = item.data(0, Qt.ItemDataRole.UserRole)
            rec_id = data.get("id") if isinstance(data, dict) else None
            if rec_id == setting_id:
                current_item = self.sidebar.settings_tree.currentItem()
                if current_item is not item:
                    was = self.sidebar.settings_tree.blockSignals(True)
                    self.sidebar.settings_tree.setCurrentItem(item)
                    self.sidebar.settings_tree.blockSignals(was)
                return
        if setting_id is None and count:
            was = self.sidebar.settings_tree.blockSignals(True)
            self.sidebar.settings_tree.setCurrentItem(self.sidebar.settings_tree.topLevelItem(0))
            self.sidebar.settings_tree.blockSignals(was)

    def _on_setting_list_changed(self, *_args) -> None:
        record = self.sidebar.current_setting()
        if not record:
            self.sidebar.set_setting_name("")
            self.sidebar.set_setting_notes("")
            self.sidebar.delete_setting_button.setEnabled(False)
            self.sidebar.set_payload_options(
                include_selections=True,
                include_filters=False,
                include_preprocessing=False,
            )
            return
        can_delete = record.get("id") is not None
        self.sidebar.delete_setting_button.setEnabled(bool(can_delete))
        self.sidebar.set_setting_name(record.get("name", "") if record.get("id") is not None else "")
        self.sidebar.set_setting_notes(record.get("notes", "") if record.get("id") is not None else "")
        payload = record.get("payload")
        if not isinstance(payload, SelectionSettingsPayload):
            payload = SelectionSettingsPayload.from_dict(payload if isinstance(payload, dict) else {})
        self.sidebar.set_payload_options(
            include_selections=payload.selections_enabled(),
            include_filters=payload.filters_enabled(),
            include_preprocessing=payload.preprocessing_enabled(),
        )

    def _selected_rows(self) -> List[int]:
        selection = self.features_table.selectionModel()
        if selection is None:
            return []
        rows: set[int] = set()
        for index in selection.selectedIndexes():
            if not index.isValid():
                continue
            source_index = self._proxy_model.mapToSource(index)
            if source_index.isValid():
                rows.add(int(source_index.row()))
        return sorted(set(rows))

    def _single_selected_row(self) -> Optional[int]:
        rows = self._selected_rows()
        if len(rows) != 1:
            return None
        return int(rows[0])

    def _selected_source_indexes(self) -> List[tuple[int, int]]:
        selection = self.features_table.selectionModel()
        if selection is None:
            return []
        pairs: list[tuple[int, int]] = []
        for view_index in selection.selectedIndexes():
            if not view_index.isValid():
                continue
            source_index = self._proxy_model.mapToSource(view_index)
            if source_index.isValid():
                pairs.append((int(source_index.row()), int(source_index.column())))
        return sorted(set(pairs))

    def _coerce_bulk_value(self, column_key: str, text: str):
        raw = (text or "").strip()
        if column_key == "selected":
            normalized = raw.lower()
            if normalized in {"1", "true", "yes", "on", "enabled"}:
                return True
            if normalized in {"0", "false", "no", "off", "disabled"}:
                return False
            raise ValueError(tr("Enter a boolean value (true/false, yes/no, 1/0)."))
        if column_key == "filter_scope":
            normalized = normalize_filter_scope(raw, default="")
            if normalized in FILTER_SCOPE_CHOICES:
                return normalized
            allowed = ", ".join(scope.title() for scope in FILTER_SCOPE_CHOICES)
            raise ValueError(tr("Enter one of: {allowed}.").format(allowed=allowed))
        if column_key in {"filter_min", "filter_max"}:
            if raw == "":
                return ""
            try:
                return float(raw)
            except Exception as exc:
                raise ValueError(tr("Enter a numeric value for filters.")) from exc
        if column_key == "lag_seconds":
            if raw == "":
                return "0"
            try:
                int(raw)
                return raw
            except Exception as exc:
                raise ValueError(tr("Enter an integer value for lag seconds.")) from exc
        return raw

    def _set_value_for_selected_cells(self) -> None:
        cells = self._selected_source_indexes()
        if not cells:
            return
        value_text, ok = QInputDialog.getText(
            self,
            tr("Set value"),
            tr("Value for selected cells:"),
        )
        if not ok:
            return
        changed = 0
        for row, col in cells:
            if col < 0 or col >= self._table_model.columnCount():
                continue
            try:
                column_key = self._table_model.COLUMN_DEFINITIONS[col][0]
            except Exception:
                continue
            try:
                value = self._coerce_bulk_value(column_key, value_text)
            except ValueError as exc:
                self._log_error(str(exc))
                return
            idx = self._table_model.index(row, col)
            if idx.isValid() and self._table_model.setData(idx, value, Qt.ItemDataRole.EditRole):
                changed += 1
        if changed:
            try:
                toast_success(
                    tr("Updated {count} cell(s).").format(count=changed),
                    title=tr("Selections"),
                    tab_key="selections",
                )
            except Exception:
                logger.warning("Exception in _set_value_for_selected_cells", exc_info=True)

    def _on_remove_feature(self) -> None:
        rows = self._selected_rows()
        if not rows:
            try:
                toast_info(
                    tr("No features selected. Select one or more rows, then remove."),
                    title=tr("Selections"),
                    tab_key="selections",
                )
            except Exception:
                logger.warning("Exception in _on_remove_feature", exc_info=True)
            return
        removed = []
        if rows:
            removed = self._table_model.remove_rows(rows)
        if removed:
            delete_ids = [int(row["feature_id"]) for row in removed if row.get("feature_id")]
            if delete_ids:
                self._view_model.queue_feature_deletions(delete_ids)

    def _on_save_features(self) -> None:
        new_features, updated_features = self._table_model.feature_changes()
        deleted_ids = self._view_model.pending_deletes()
        if not new_features and not updated_features and not deleted_ids:
            try:
                toast_info(
                    tr("No changes to save. Edit feature values, add/remove features, then save."),
                    title=tr("Selections"),
                    tab_key="selections",
                )
            except Exception:
                logger.warning("Exception in _on_save_features", exc_info=True)
            return
        summary: List[str] = []
        if updated_features:
            summary.append(tr("Update {count} feature(s)").format(count=len(updated_features)))
        if new_features:
            summary.append(tr("Add {count} new feature(s)").format(count=len(new_features)))
        if deleted_ids:
            summary.append(tr("Delete {count} feature(s)").format(count=len(deleted_ids)))
        detail = "\n".join(summary) or tr("Apply feature changes?")
        result = QMessageBox.question(
            self,
            tr("Apply feature changes"),
            tr("{detail}\n\nDo you want to continue?").format(detail=detail),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            return
        self._pending_toast_action = "save_features"
        try:
            toast_info(tr("Saving feature changes…"), title=tr("Selections"), tab_key="selections")
        except Exception:
            logger.warning("Exception in _on_save_features", exc_info=True)
        self._view_model.save_features(new_features, updated_features, deleted_feature_ids=deleted_ids)

    def _on_features_save_completed(self, summary: Optional[dict]) -> None:
        if self._pending_toast_action != "save_features":
            return
        self._pending_toast_action = None
        inserted = int((summary or {}).get("inserted_count") or 0)
        updated = int((summary or {}).get("updated_count") or 0)
        deleted = int((summary or {}).get("deleted_count") or 0)
        message = tr("Feature changes saved.")
        if inserted or updated or deleted:
            message = tr("Feature changes saved ({inserted} added, {updated} updated, {deleted} deleted).").format(
                inserted=inserted,
                updated=updated,
                deleted=deleted,
            )
        try:
            toast_success(message, title=tr("Selections"), tab_key="selections")
        except Exception:
            logger.warning("Exception in _on_features_save_completed", exc_info=True)

    def _on_reload_features(self) -> None:
        self._pending_toast_action = "reload_features"
        try:
            toast_info(tr("Reloading features from database…"), title=tr("Selections"), tab_key="selections")
        except Exception:
            logger.warning("Exception in _on_reload_features", exc_info=True)
        self._view_model.refresh_features()

    def _gather_payload(self) -> SelectionSettingsPayload:
        _, filters = self._table_model.selection_state()
        _, label_filters = self._table_model.selection_fallback_state()
        excluded_features, excluded_labels = self._table_model.selection_exclusion_state()
        selector_settings = self.sidebar.get_selection_settings()
        payload_options = self.sidebar.payload_options()
        include_selections = bool(payload_options.get("include_selections"))
        include_filters = bool(payload_options.get("include_filters"))
        include_preprocessing = bool(payload_options.get("include_preprocessing"))
        filter_state = dict(selector_settings.get("filters") or {})
        preprocessing = dict(selector_settings.get("preprocessing") or {})
        return SelectionSettingsPayload(
            feature_ids=excluded_features if include_selections else [],
            feature_labels=excluded_labels if include_selections else [],
            filters=filter_state if include_filters else {},
            preprocessing=preprocessing if include_preprocessing else {},
            feature_filters=filters if include_selections else [],
            feature_filter_labels=label_filters if include_selections else [],
            include_selections=include_selections,
            include_filters=include_filters,
            include_preprocessing=include_preprocessing,
            selection_mode=SELECTION_MODE_EXCLUDE if include_selections else None,
        )

    def _on_save_setting(self) -> None:
        record = self.sidebar.current_setting() or {}
        payload = self._gather_payload()
        current_setting_id = record.get("id") if record else None
        name = self.sidebar.setting_display_name()
        if not name:
            try:
                toast_warn(
                    tr("Enter a name before saving selection settings."),
                    title=tr("Selections"),
                    tab_key="selections",
                )
            except Exception:
                logger.warning("Exception in _on_save_setting", exc_info=True)
            return
        notes = self.sidebar.setting_notes_text()
        activate = bool(record.get("is_active")) if record else False

        existing = self._find_setting_by_name(name)
        setting_id = None
        if existing and existing.get("id") != current_setting_id:
            result = QMessageBox.question(
                self,
                tr("Replace selection setting"),
                tr("A selection setting named '{name}' already exists.\n\nReplace it?").format(name=name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if result != QMessageBox.StandardButton.Yes:
                return
            setting_id = existing.get("id")
            activate = bool(existing.get("is_active"))
        elif existing and existing.get("id") == current_setting_id:
            # Saving with the same loaded name updates that setting.
            setting_id = current_setting_id

        self._pending_toast_action = "save_setting"
        try:
            toast_info(tr("Saving selection setting…"), title=tr("Selections"), tab_key="selections")
        except Exception:
            logger.warning("Exception in _on_save_setting", exc_info=True)
        self._view_model.save_selection_setting(
            name=name,
            notes=notes,
            payload=payload,
            auto_load=False,
            setting_id=setting_id,
            activate=activate,
        )

    def _find_setting_by_name(self, name: str) -> Optional[dict]:
        target = str(name or "").strip().casefold()
        if not target:
            return None
        for rec in self._settings:
            rec_name = str(rec.get("name") or "").strip().casefold()
            if rec_name == target:
                return rec
        return None

    def _on_table_data_changed(self, top_left, bottom_right, roles=None) -> None:
        if top_left is None or bottom_right is None:
            return
        start = top_left.column()
        end = bottom_right.column()
        keys = set()
        for col in range(start, end + 1):
            try:
                keys.add(self._table_model.COLUMN_DEFINITIONS[col][0])
            except Exception:
                continue
        if keys.intersection({"selected", "type"}):
            self._queue_group_filter_refresh()
        if not keys.intersection({"filter_min", "filter_max", "filter_scope"}):
            return
        # Do not apply value filters immediately; wait for selection settings save/activate.
        return

    def _is_group_feature_row(self, row: int) -> bool:
        payload = self._table_model.row_payload(row)
        if not payload:
            return False
        feature_type = str(payload.get("type") or "").strip().casefold()
        if feature_type == "group":
            return True
        feature_id = payload.get("feature_id")
        if feature_id in (None, ""):
            return False
        try:
            fid = int(feature_id)
        except Exception:
            return False
        try:
            db = self._database_model.database()
            kinds = db.group_kinds_for_feature_ids([fid])
            return bool(kinds)
        except Exception:
            return False

    def _refresh_group_filters_for_enabled_group_features(self) -> None:
        fw = self.sidebar.filters_widget
        if fw is None:
            return
        try:
            selected_ids, _filters = self._table_model.selection_state()
            db = self._database_model.database()
            all_groups = self._database_model.groups_df(respect_selection=False)
            if not selected_ids:
                fw.set_groups(pd.DataFrame(columns=["group_id", "kind", "label"]))
                return
            enabled_kinds = set(db.group_kinds_for_feature_ids(selected_ids))
            if not enabled_kinds:
                fw.set_groups(pd.DataFrame(columns=["group_id", "kind", "label"]))
                return
            kinds = all_groups.get("kind", pd.Series(dtype=object)).astype(str).str.strip()
            fw.set_groups(all_groups.loc[kinds.isin(enabled_kinds)].reset_index(drop=True))
        except Exception as exc:
            if self._database_model._is_database_in_use_error(exc):
                logger.info("Skipped group filter refresh because database file is in use.")
                return
            logger.warning("Exception in _refresh_group_filters_for_enabled_group_features", exc_info=True)

    def _on_load_setting(self) -> None:
        record = self.sidebar.current_setting() or {}
        setting_id = record.get("id")
        self._pending_toast_action = "activate_setting"
        try:
            toast_info(tr("Activating selection…"), title=tr("Selections"))
        except Exception:
            logger.warning("Exception in _on_load_setting", exc_info=True)
        self._view_model.activate_selection(setting_id)

    def _on_delete_setting(self) -> None:
        record = self.sidebar.current_setting() or {}
        setting_id = record.get("id")
        if setting_id is None:
            return
        name = str(record.get("name") or tr("Selection")).strip() or tr("Selection")
        result = QMessageBox.question(
            self,
            tr("Delete selection setting"),
            tr("Delete selection setting '{name}'?\n\nThis action cannot be undone.").format(name=name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            return
        self._pending_toast_action = "delete_setting"
        try:
            toast_info(tr("Deleting selection setting…"), title=tr("Selections"), tab_key="selections")
        except Exception:
            logger.warning("Exception in _on_delete_setting", exc_info=True)
        self._view_model.delete_selection_setting(setting_id)

    def _on_load_settings_database(self) -> None:
        self._invoke_main_window_action("open_selection_settings_database")

    def _on_save_settings_database(self) -> None:
        self._invoke_main_window_action("save_selection_settings_database_as")

    def _on_reset_settings_database(self) -> None:
        self._invoke_main_window_action("reset_selection_settings_database")

    def _log_error(self, message: str) -> None:
        if self._pending_toast_action == "save_features":
            self._pending_toast_action = None
        try:
            toast_error(message, title=tr("Selections error"), tab_key="selections")
        except Exception:
            logger.warning("Exception in _log_error", exc_info=True)

    def _on_convert_selected_feature_to_group(self, row: int) -> None:
        if self._is_group_feature_row(int(row)):
            try:
                toast_info(
                    tr("Selected feature is already a group feature."),
                    title=tr("Selections"),
                    tab_key="selections",
                )
            except Exception:
                logger.warning("Exception in _on_convert_selected_feature_to_group", exc_info=True)
            return
        payload = self._table_model.row_payload(row)
        if not payload:
            return
        feature_id = payload.get("feature_id")
        if feature_id in (None, ""):
            try:
                toast_info(
                    tr("Only saved features with measurements can be converted to groups."),
                    title=tr("Selections"),
                    tab_key="selections",
                )
            except Exception:
                logger.warning("Exception in _on_convert_selected_feature_to_group", exc_info=True)
            return
        try:
            fid = int(feature_id)
        except Exception:
            return
        feature_name = str(payload.get("name") or payload.get("notes") or "").strip() or f"feature_{fid}"
        try:
            toast_info(
                tr("Loading unique values for selected feature…"),
                title=tr("Selections"),
                tab_key="selections",
            )
        except Exception:
            logger.warning("Exception in _on_convert_selected_feature_to_group", exc_info=True)
        self._view_model.request_feature_group_preview(feature_id=fid, feature_name=feature_name)

    def _on_feature_group_preview_ready(self, payload: Optional[dict]) -> None:
        if not payload:
            return
        dialog = FeatureGroupConversionDialog(
            feature_name=str(payload.get("feature_name") or ""),
            unique_values=list(payload.get("unique_values") or []),
            unique_count=int(payload.get("unique_count") or 0),
            is_csv_import_feature=bool(payload.get("is_csv_import_feature")),
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        values = dialog.values()
        group_name = str(values.get("group_name") or "").strip()
        if not group_name:
            try:
                toast_error(
                    tr("Group feature name cannot be empty."),
                    title=tr("Selections error"),
                    tab_key="selections",
                )
            except Exception:
                logger.warning("Exception in _on_feature_group_preview_ready", exc_info=True)
            return
        try:
            toast_info(
                tr("Converting feature values to groups…"),
                title=tr("Selections"),
                tab_key="selections",
            )
        except Exception:
            logger.warning("Exception in _on_feature_group_preview_ready", exc_info=True)
        self._view_model.convert_feature_values_to_group(
            feature_id=int(payload.get("feature_id") or 0),
            group_kind=group_name,
            save_as_timeframes=bool(values.get("save_as_timeframes", True)),
            link_only_as_group_label=bool(values.get("link_only_as_group_label", False)),
            value_name_map=dict(values.get("value_name_map") or {}),
        )

    def _on_feature_group_conversion_done(self, payload: Optional[dict]) -> None:
        if not payload:
            return
        try:
            if bool(payload.get("link_only_as_group_label")):
                message = tr("Saved {groups} group label(s) and linked {links} CSV column mapping(s).").format(
                    groups=int(payload.get("group_labels") or 0),
                    links=int(payload.get("csv_group_links") or 0),
                )
            else:
                message = tr("Created {groups} groups with {points} entries.").format(
                    groups=int(payload.get("group_labels") or 0),
                    points=int(payload.get("group_points") or 0),
                )
            toast_success(message, title=tr("Selections"), tab_key="selections")
        except Exception:
            logger.warning("Exception in _on_feature_group_conversion_done", exc_info=True)

    def _build_table_context_menu(self, menu, _pos, _table) -> None:
        if self._table_model.rowCount() <= 0:
            return
        selected_cells = self._selected_source_indexes()
        if selected_cells:
            set_selected_cells = menu.addAction(tr("Set value for selected cells…"))
            set_selected_cells.triggered.connect(lambda _: self._set_value_for_selected_cells())
            menu.addSeparator()
        rows = self._selected_rows()
        group_rows = [row for row in rows if self._is_group_feature_row(row)]
        non_group_rows = [row for row in rows if row not in group_rows]
        if rows:
            if group_rows and not non_group_rows:
                enable_selected = menu.addAction(tr("Enable selected group features"))
                disable_selected = menu.addAction(tr("Disable selected group features"))
                remove_selected = menu.addAction(tr("Remove selected group features"))
            else:
                enable_selected = menu.addAction(tr("Enable selected features"))
                disable_selected = menu.addAction(tr("Disable selected features"))
                remove_selected = menu.addAction(tr("Remove selected features"))
            enable_selected.triggered.connect(
                lambda _, r=list(rows): self._table_model.set_rows_selected(r, True)
            )
            disable_selected.triggered.connect(
                lambda _, r=list(rows): self._table_model.set_rows_selected(r, False)
            )
            remove_selected.triggered.connect(lambda _: self._on_remove_feature())
        single_row = self._single_selected_row()
        if single_row is not None:
            feature_id = self._table_model.feature_id_for_row(single_row)
            if feature_id is not None:
                if not self._is_group_feature_row(int(single_row)):
                    menu.addSeparator()
                    convert_action = menu.addAction(tr("Convert feature values to groups…"))
                    convert_action.triggered.connect(
                        lambda _, row=int(single_row): self._on_convert_selected_feature_to_group(row)
                    )
