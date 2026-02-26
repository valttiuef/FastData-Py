
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (

    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from ...localization import tr

from ...widgets.preprocessing_widget import PreprocessingWidget
from ...widgets.filters_widget import FiltersWidget
from ...widgets.sidebar_widget import SidebarWidget



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

        self.preprocessing_widget = PreprocessingWidget(collapsed=True, parent=self)
        layout.addWidget(self.preprocessing_widget)

        self.filters_widget = FiltersWidget(collapsed=True, parent=self, model=model)
        layout.addWidget(self.filters_widget)

        # Selection settings controls
        settings = QGroupBox(tr("Selection settings"))
        settings_layout = QVBoxLayout(settings)

        self.settings_list = QListWidget()
        self.settings_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.settings_list.setUniformItemSizes(True)
        settings_layout.addWidget(self.settings_list, 1)

        form = QFormLayout()
        self.setting_name = QLineEdit()
        form.addRow(tr("Name"), self.setting_name)
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
        was = self.settings_list.blockSignals(True)
        self.settings_list.clear()
        for record in records:
            name = record.get("name") or tr("Selection")
            item = QListWidgetItem(str(name))
            item.setData(Qt.ItemDataRole.UserRole, record)
            if record.get("is_active"):
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            item.setToolTip(str(name or tr("Selection")))
            self.settings_list.addItem(item)
        index = active_index if 0 <= active_index < self.settings_list.count() else 0
        if self.settings_list.count():
            self.settings_list.setCurrentRow(index)
        self.settings_list.blockSignals(was)

    def current_setting(self) -> dict | None:
        item = self.settings_list.currentItem()
        if not item:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
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

    def setting_display_name(self) -> str:
        return (self.setting_name.text() or "").strip()

    # --- Filters/preprocessing -------------------------------------------
    def preprocessing_parameters(self) -> dict:
        return self.preprocessing_widget.parameters()

    def set_preprocessing_parameters(self, params: dict | None) -> None:
        self.preprocessing_widget.set_parameters(params or {})

    def filter_state(self) -> dict:
        return self.filters_widget.filter_state()

    def apply_filter_state(self, state: dict | None) -> None:
        self.filters_widget.apply_filter_state(state or {})

    def set_systems(self, items: list[tuple[str, object]], *, check_all: bool = True) -> None:
        self.filters_widget.set_systems(items, check_all=check_all)

    def set_datasets(self, items: list[tuple[str, object]], *, check_all: bool = True) -> None:
        self.filters_widget.set_datasets(items, check_all=check_all)

    def set_groups(self, df) -> None:
        self.filters_widget.set_groups(df)

    def set_tags(self, items: list[tuple[str, object]], *, check_all: bool = True) -> None:
        self.filters_widget.set_tags(items, check_all=check_all)
