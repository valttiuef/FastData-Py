
from __future__ import annotations
# --- @ai START ---
# model: gpt-5
# tool: codex-cli
# role: ui-refactor
# reviewed: yes
# date: 2026-03-02
# --- @ai END ---
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from ...localization import tr

from ...widgets.data_selector_widget import DataSelectorWidget
from ...widgets.sidebar_widget import SidebarWidget
from ...models.selection_settings import SelectionSettingsPayload



class Sidebar(SidebarWidget):
    def __init__(self, parent=None, *, model=None):
        super().__init__(title=tr("Selections"), parent=parent)
        layout = self.content_layout()

        # Feature actions
        actions = QGroupBox(tr("Feature actions"))
        actions_layout = QVBoxLayout(actions)
        self.remove_button = QPushButton(tr("Remove selected features"))
        self.reload_button = QPushButton(tr("Reload features"))
        self.save_button = QPushButton(tr("Save feature changes"))
        actions_layout.addWidget(self.remove_button)
        actions_layout.addWidget(self.reload_button)
        actions_layout.addWidget(self.save_button)
        self.set_sticky_actions(actions)

        self.data_selector = DataSelectorWidget(
            parent=self,
            data_model=model,
            show_preprocessing=True,
            show_filters=True,
            show_features_list=False,
        )
        layout.addWidget(self.data_selector)
        self.preprocessing_widget = self.data_selector.preprocessing_widget
        self.filters_widget = self.data_selector.filters_widget
        if self.preprocessing_widget is not None:
            self.preprocessing_widget.set_collapsed(False)
        if self.filters_widget is not None:
            self.filters_widget.set_collapsed(False)

        # Selection settings controls
        settings = QGroupBox(tr("Selection settings"))
        settings_layout = QVBoxLayout(settings)

        self.settings_tree = QTreeWidget()
        self.settings_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.settings_tree.setUniformRowHeights(True)
        self.settings_tree.setRootIsDecorated(False)
        self.settings_tree.setAlternatingRowColors(True)
        self.settings_tree.setHeaderLabels(
            [
                tr("Name"),
                tr("Created"),
                tr("Notes"),
                tr("Includes"),
            ]
        )
        self.settings_tree.header().setStretchLastSection(True)
        settings_layout.addWidget(self.settings_tree, 1)

        form = QFormLayout()
        self.setting_name = QLineEdit()
        self.setting_notes = QLineEdit()
        form.addRow(tr("Name"), self.setting_name)
        form.addRow(tr("Notes"), self.setting_notes)
        self.include_selections_checkbox = QCheckBox(tr("Selections"))
        self.include_filters_checkbox = QCheckBox(tr("Filtering"))
        self.include_preprocessing_checkbox = QCheckBox(tr("Preprocessing"))
        self.include_selections_checkbox.setChecked(True)
        self.include_filters_checkbox.setChecked(False)
        self.include_preprocessing_checkbox.setChecked(False)
        includes_row = QHBoxLayout()
        includes_row.addWidget(self.include_selections_checkbox)
        includes_row.addWidget(self.include_filters_checkbox)
        includes_row.addWidget(self.include_preprocessing_checkbox)
        form.addRow(tr("Save"), includes_row)
        settings_layout.addLayout(form)

        buttons_row = QHBoxLayout()
        self.load_setting_button = QPushButton(tr("Load"))
        self.save_setting_button = QPushButton(tr("Save"))
        self.delete_setting_button = QPushButton(tr("Delete"))
        buttons_row.addWidget(self.load_setting_button)
        buttons_row.addWidget(self.save_setting_button)
        buttons_row.addWidget(self.delete_setting_button)
        settings_layout.addLayout(buttons_row)

        layout.addWidget(settings, 1)

        db_box = QGroupBox(tr("Settings database"))
        db_layout = QVBoxLayout(db_box)
        buttons_row = QHBoxLayout()
        self.load_db_button = QPushButton(tr("Load…"))
        self.save_db_button = QPushButton(tr("Save as…"))
        self.reset_db_button = QPushButton(tr("Reset"))
        buttons_row.addWidget(self.load_db_button)
        buttons_row.addWidget(self.save_db_button)
        buttons_row.addWidget(self.reset_db_button)
        db_layout.addLayout(buttons_row)
        layout.addWidget(db_box)

    # --- Settings helpers ------------------------------------------------
    def set_settings(self, records: list[dict], *, active_index: int = 0) -> None:
        was = self.settings_tree.blockSignals(True)
        self.settings_tree.clear()
        for record in records:
            name = record.get("name") or tr("Selection")
            setting_id = record.get("id")
            created = str(record.get("created_at") or "")
            notes = str(record.get("notes") or "")
            includes = self._includes_text(record)
            item = QTreeWidgetItem([str(name), created, notes, includes])
            item.setData(0, Qt.ItemDataRole.UserRole, record)
            if record.get("is_active"):
                font = item.font(0)
                font.setBold(True)
                for col in range(item.columnCount()):
                    item.setFont(col, font)
            tooltip = str(name or tr("Selection"))
            for col in range(item.columnCount()):
                item.setToolTip(col, tooltip)
            self.settings_tree.addTopLevelItem(item)
        index = active_index if 0 <= active_index < self.settings_tree.topLevelItemCount() else 0
        if self.settings_tree.topLevelItemCount():
            self.settings_tree.setCurrentItem(self.settings_tree.topLevelItem(index))
        self.settings_tree.blockSignals(was)

    def current_setting(self) -> dict | None:
        item = self.settings_tree.currentItem()
        if not item:
            return None
        data = item.data(0, Qt.ItemDataRole.UserRole)
        return data if isinstance(data, dict) else None

    def selected_setting_id(self) -> int | None:
        record = self.current_setting()
        if not record:
            return None
        value = record.get("id")
        try:
            return int(value) if value is not None else None
        except Exception:
            return None

    def set_setting_name(self, name: str) -> None:
        was = self.setting_name.blockSignals(True)
        self.setting_name.setText(name or "")
        self.setting_name.blockSignals(was)

    def set_setting_notes(self, notes: str) -> None:
        was = self.setting_notes.blockSignals(True)
        self.setting_notes.setText(notes or "")
        self.setting_notes.blockSignals(was)

    def setting_display_name(self) -> str:
        return (self.setting_name.text() or "").strip()

    def setting_notes_text(self) -> str:
        return (self.setting_notes.text() or "").strip()

    def set_payload_options(
        self,
        *,
        include_selections: bool,
        include_filters: bool,
        include_preprocessing: bool,
    ) -> None:
        was_sel = self.include_selections_checkbox.blockSignals(True)
        was_flt = self.include_filters_checkbox.blockSignals(True)
        was_pre = self.include_preprocessing_checkbox.blockSignals(True)
        self.include_selections_checkbox.setChecked(bool(include_selections))
        self.include_filters_checkbox.setChecked(bool(include_filters))
        self.include_preprocessing_checkbox.setChecked(bool(include_preprocessing))
        self.include_selections_checkbox.blockSignals(was_sel)
        self.include_filters_checkbox.blockSignals(was_flt)
        self.include_preprocessing_checkbox.blockSignals(was_pre)

    def payload_options(self) -> dict:
        return {
            "include_selections": bool(self.include_selections_checkbox.isChecked()),
            "include_filters": bool(self.include_filters_checkbox.isChecked()),
            "include_preprocessing": bool(self.include_preprocessing_checkbox.isChecked()),
        }

    def _includes_text(self, record: dict) -> str:
        payload = record.get("payload")
        if not isinstance(payload, SelectionSettingsPayload):
            payload = SelectionSettingsPayload.from_dict(payload if isinstance(payload, dict) else {})
        labels: list[str] = []
        if payload.selections_enabled():
            labels.append(tr("Selections"))
        if payload.filters_enabled():
            labels.append(tr("Filtering"))
        if payload.preprocessing_enabled():
            labels.append(tr("Preprocessing"))
        return ", ".join(labels) if labels else tr("None")

    # --- Filters/preprocessing -------------------------------------------
    def preprocessing_parameters(self) -> dict:
        return self.data_selector.get_settings().get("preprocessing", {})

    def set_preprocessing_parameters(self, params: dict | None) -> None:
        self.data_selector.apply_settings({"preprocessing": params or {}, "filters": self.filter_state()})

    def filter_state(self) -> dict:
        return self.data_selector.get_settings().get("filters", {})

    def apply_filter_state(self, state: dict | None) -> None:
        self.data_selector.apply_settings({"preprocessing": self.preprocessing_parameters(), "filters": state or {}})

    def get_selection_settings(self) -> dict:
        return self.data_selector.get_settings()

    def apply_selection_settings(self, settings: dict | None) -> None:
        self.data_selector.apply_settings(settings or {})

    def set_systems(self, items: list[tuple[str, object]], *, check_all: bool = True) -> None:
        if self.filters_widget is not None:
            self.filters_widget.set_systems(items, check_all=check_all)

    def set_datasets(self, items: list[tuple[str, object]], *, check_all: bool = True) -> None:
        if self.filters_widget is not None:
            self.filters_widget.set_datasets(items, check_all=check_all)

    def set_groups(self, df) -> None:
        if self.filters_widget is not None:
            self.filters_widget.set_groups(df)

    def set_tags(self, items: list[tuple[str, object]], *, check_all: bool = True) -> None:
        if self.filters_widget is not None:
            self.filters_widget.set_tags(items, check_all=check_all)
