
from __future__ import annotations
from typing import List, Optional

from PySide6.QtCore import QSortFilterProxyModel, Qt, QTimer
from PySide6.QtWidgets import QAbstractItemView, QInputDialog, QLineEdit, QMessageBox, QWidget
from ...localization import tr

from ...models.database_model import DatabaseModel
from ...models.log_model import LogModel, get_log_model
from ...utils import toast_error, toast_info, toast_success
from ...models.selection_settings import SelectionSettingsPayload
from ...widgets.fast_table import FastTable
from ...widgets.panel import Panel
from ..tab_widget import TabWidget
from .models import FeatureSelectionTableModel
from .feature_group_conversion_dialog import FeatureGroupConversionDialog
from .sidebar import Sidebar
from .viewmodel import SelectionsViewModel
import logging

logger = logging.getLogger(__name__)



class _SelectionSortProxy(QSortFilterProxyModel):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._search_text = ""
        self._search_terms: tuple[str, ...] = ()

    def set_search_text(self, text: str) -> None:
        raw = (text or "").strip().lower()
        terms = tuple(part.strip() for part in raw.split(",") if part.strip())
        if raw == self._search_text and terms == self._search_terms:
            return
        self._search_text = raw
        self._search_terms = terms
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):  # noqa: N802
        if not self._search_terms:
            return True
        source = self.sourceModel()
        if source is None:
            return False
        col_count = source.columnCount(source_parent)
        for col in range(col_count):
            idx = source.index(source_row, col, source_parent)
            value = source.data(idx, Qt.ItemDataRole.DisplayRole)
            text = str(value or "").lower()
            if any(term in text for term in self._search_terms):
                return True
        return False

    def lessThan(self, left, right):  # noqa: N802
        left_data = self.sourceModel().data(left, Qt.ItemDataRole.EditRole)
        right_data = self.sourceModel().data(right, Qt.ItemDataRole.EditRole)
        try:
            if left_data in (None, "") and right_data not in (None, ""):
                return True
            if right_data in (None, "") and left_data not in (None, ""):
                return False
            left_num = float(left_data)
            right_num = float(right_data)
            return left_num < right_num
        except Exception:
            return str(left_data) < str(right_data)


class SelectionsTab(TabWidget):
    def __init__(self, database_model: DatabaseModel, parent: Optional[QWidget] = None, *, log_model: Optional[LogModel] = None) -> None:
        self._database_model = database_model
        self._log_model = log_model or get_log_model()
        self._table_model = FeatureSelectionTableModel()
        self._current_payload = SelectionSettingsPayload()
        self._settings: List[dict] = []
        self._settings_with_default: List[dict] = []
        self._select_all_default = True
        self._pending_toast_action: Optional[str] = None
        super().__init__(parent)

        # QObject-derived helpers must be created after the QWidget base class
        # has been initialised to avoid "base class not called" errors when we
        # pass ``self`` as their parent.
        self._table_model.setParent(self)
        self._view_model = SelectionsViewModel(self._database_model, log_model=self._log_model, parent=self)

        self._connect_signals()
        self.sidebar.filters_widget.refresh_filters()
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
        self._search_edit = QLineEdit(panel)
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.setPlaceholderText(tr("Search features…"))
        layout.addWidget(self._search_edit)
        self.features_table = FastTable(
            select="items",
            single_selection=False,
            tint_current_selection=False,
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
        self.features_table.set_stretch_column(-1)
        self._proxy_model = _SelectionSortProxy(self)
        self._proxy_model.setSourceModel(self._table_model)
        self._proxy_model.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        # Keep sorting stable while cell editors are committing values.
        self._proxy_model.setDynamicSortFilter(False)
        self.features_table.setModel(self._proxy_model)
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
        self.sidebar.settings_list.currentRowChanged.connect(self._on_setting_list_changed)
        self.sidebar.load_db_button.clicked.connect(self._on_load_settings_database)
        self.sidebar.save_db_button.clicked.connect(self._on_save_settings_database)
        self.sidebar.reset_db_button.clicked.connect(self._on_reset_settings_database)
        self._search_edit.textChanged.connect(self._proxy_model.set_search_text)
        self._table_model.dataChanged.connect(self._on_table_data_changed)

    def _invoke_main_window_action(self, method_name: str) -> None:
        win = self.window() or self.parent()
        try:
            handler = getattr(win, method_name, None) if win is not None else None
            if callable(handler):
                handler()
        except Exception:
            logger.warning("Exception in _invoke_main_window_action", exc_info=True)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _on_database_changed(self, *_args) -> None:
        self.sidebar.filters_widget.refresh_filters()
        self._view_model.refresh()

    def _on_selection_database_changed(self, *_args) -> None:
        self._view_model._refresh_settings()

    def _on_features_list_changed(self) -> None:
        self.sidebar.filters_widget.refresh_filters()
        self._view_model.refresh_features()

    def _on_features_changed(self, df) -> None:
        self._table_model.set_features(df)
        self._table_model.apply_selection(self._current_payload, select_all_by_default=self._select_all_default)
        self._reapply_table_sort()
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
        return dict(id=None, name=tr("Default"), auto_load=False, is_active=False, payload=SelectionSettingsPayload())

    def _on_settings_loaded(self, records: List[dict]) -> None:
        self._settings = list(records)
        combined = [self._default_record()] + self._settings
        active_index = next((idx for idx, rec in enumerate(combined) if rec.get("is_active")), -1)
        if active_index < 0:
            combined[0]["is_active"] = True
            active_index = 0
        self._settings_with_default = combined
        self.sidebar.set_settings(combined, active_index=active_index)
        self._on_setting_list_changed(self.sidebar.settings_list.currentRow())

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
        payload = record.get("payload") if record else SelectionSettingsPayload()
        if payload is None:
            payload = SelectionSettingsPayload()
        self._current_payload = payload
        self._select_all_default = not bool(record and record.get("id"))
        self._table_model.apply_selection(payload, select_all_by_default=self._select_all_default)
        self.sidebar.set_preprocessing_parameters(payload.preprocessing)
        self.sidebar.apply_filter_state(payload.filters)
        name = record.get("name") if record else ""
        self.sidebar.set_setting_name(name or "")
        target_id = record.get("id") if record else None
        self._select_setting_in_list(target_id)
        self._apply_active_selection_payload(payload)
        if self._pending_toast_action == "activate_setting":
            self._pending_toast_action = None
            try:
                toast_success(tr("Selection setting activated."), title=tr("Selections"), tab_key="selections")
            except Exception:
                logger.warning("Exception in _on_active_setting_changed", exc_info=True)

    def _apply_active_selection_payload(self, payload: SelectionSettingsPayload) -> None:
        try:
            self._database_model.apply_selection_payload(SelectionSettingsPayload.from_dict(payload.to_dict()))
        except Exception:
            logger.warning("Exception in _apply_active_selection_payload", exc_info=True)

    def _select_setting_in_list(self, setting_id: Optional[int]) -> None:
        count = self.sidebar.settings_list.count()
        for idx in range(count):
            item = self.sidebar.settings_list.item(idx)
            if not item:
                continue
            data = item.data(Qt.ItemDataRole.UserRole)
            rec_id = data.get("id") if isinstance(data, dict) else None
            if rec_id == setting_id:
                if idx != self.sidebar.settings_list.currentRow():
                    was = self.sidebar.settings_list.blockSignals(True)
                    self.sidebar.settings_list.setCurrentRow(idx)
                    self.sidebar.settings_list.blockSignals(was)
                return
        if setting_id is None and count:
            was = self.sidebar.settings_list.blockSignals(True)
            self.sidebar.settings_list.setCurrentRow(0)
            self.sidebar.settings_list.blockSignals(was)

    def _on_setting_list_changed(self, index: int) -> None:
        record = self.sidebar.current_setting()
        if not record:
            self.sidebar.set_setting_name("")
            return
        self.sidebar.set_setting_name(record.get("name", "") if record.get("id") is not None else "")

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
        if column_key in {"selected", "filter_global"}:
            normalized = raw.lower()
            if normalized in {"1", "true", "yes", "on", "enabled"}:
                return True
            if normalized in {"0", "false", "no", "off", "disabled"}:
                return False
            raise ValueError(tr("Enter a boolean value (true/false, yes/no, 1/0)."))
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
        selected_features, filters = self._table_model.selection_state()
        selected_labels, label_filters = self._table_model.selection_state_labels()
        filter_state = self.sidebar.filter_state()
        preprocessing = self.sidebar.preprocessing_parameters()
        return SelectionSettingsPayload(
            feature_ids=selected_features,
            feature_labels=selected_labels,
            filters=filter_state,
            preprocessing=preprocessing,
            feature_filters=filters,
            feature_filter_labels=label_filters,
        )

    def _on_save_setting(self) -> None:
        record = self.sidebar.current_setting() or {}
        payload = self._gather_payload()
        setting_id = record.get("id") if record else None
        name = self.sidebar.setting_display_name() or tr("Custom")
        activate = bool(record.get("is_active")) if record else False
        self._pending_toast_action = "save_setting"
        try:
            toast_info(tr("Saving selection setting…"), title=tr("Selections"), tab_key="selections")
        except Exception:
            logger.warning("Exception in _on_save_setting", exc_info=True)
        self._view_model.save_selection_setting(
            name=name,
            payload=payload,
            auto_load=False,
            setting_id=setting_id,
            activate=activate,
        )

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
        if not keys.intersection({"filter_min", "filter_max", "filter_global"}):
            return
        # Do not apply value filters immediately; wait for selection settings save/activate.
        return

    def _on_load_setting(self) -> None:
        record = self.sidebar.current_setting() or {}
        setting_id = record.get("id")
        self._pending_toast_action = "activate_setting"
        try:
            toast_info(tr("Activating selection…"), title=tr("Selections"), tab_key="selections")
        except Exception:
            logger.warning("Exception in _on_load_setting", exc_info=True)
        self._view_model.activate_selection(setting_id)

    def _on_delete_setting(self) -> None:
        record = self.sidebar.current_setting() or {}
        setting_id = record.get("id")
        if setting_id is None:
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
            keep_original_feature=not bool(values.get("remove_original_feature", True)),
            save_as_timeframes=bool(values.get("save_as_timeframes", True)),
            value_name_map=dict(values.get("value_name_map") or {}),
        )

    def _on_feature_group_conversion_done(self, payload: Optional[dict]) -> None:
        if not payload:
            return
        try:
            message = tr("Created {groups} groups with {points} entries.").format(
                groups=int(payload.get("group_labels") or 0),
                points=int(payload.get("group_points") or 0),
            )
            if bool(payload.get("feature_removed")):
                message = tr("{message} Original feature was removed.").format(message=message)
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
        if rows:
            enable_selected = menu.addAction(tr("Enable selected features"))
            enable_selected.triggered.connect(
                lambda _, r=list(rows): self._table_model.set_rows_selected(r, True)
            )
            disable_selected = menu.addAction(tr("Disable selected features"))
            disable_selected.triggered.connect(
                lambda _, r=list(rows): self._table_model.set_rows_selected(r, False)
            )
            remove_selected = menu.addAction(tr("Remove selected features"))
            remove_selected.triggered.connect(lambda _: self._on_remove_feature())
        single_row = self._single_selected_row()
        if single_row is not None:
            feature_id = self._table_model.feature_id_for_row(single_row)
            if feature_id is not None:
                menu.addSeparator()
                convert_action = menu.addAction(tr("Convert feature values to groups…"))
                convert_action.triggered.connect(
                    lambda _, row=int(single_row): self._on_convert_selected_feature_to_group(row)
                )
