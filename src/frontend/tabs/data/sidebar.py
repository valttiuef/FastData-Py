
from __future__ import annotations
from PySide6.QtWidgets import (

    QVBoxLayout, QLabel, QCheckBox, QPushButton,
    QGroupBox
)
from ...localization import tr

from ...widgets.data_selector_widget import DataSelectorWidget
from ...widgets.sidebar_widget import SidebarWidget

from .viewmodel import DataViewModel

import pandas as pd
from PySide6.QtCore import QDateTime



class Sidebar(SidebarWidget):
    def __init__(self, view_model: DataViewModel, parent=None):
        self._view_model: DataViewModel = view_model
        super().__init__(title=tr("Data"), parent=parent)
        layout = self.content_layout()

        # --- Actions
        actions = QGroupBox(tr("Actions"))
        a_lay = QVBoxLayout(actions)
        self.btn_import = QPushButton(tr("Import Files…"))
        self.btn_import.clicked.connect(self._view_model.request_import)
        self.btn_new = QPushButton(tr("New Database…"))
        self.btn_new.clicked.connect(self._view_model.request_new_database)
        self.btn_load = QPushButton(tr("Open Database…"))
        self.btn_load.clicked.connect(self._view_model.request_load_database)
        self.btn_save = QPushButton(tr("Save Database…"))
        self.btn_save.clicked.connect(self._view_model.request_save_database)
        a_lay.addWidget(self.btn_import)
        a_lay.addWidget(self.btn_new)
        a_lay.addWidget(self.btn_load)
        a_lay.addWidget(self.btn_save)
        self.set_sticky_actions(actions)

        self.data_selector = DataSelectorWidget(parent=self, data_model=view_model.model)
        layout.addWidget(self.data_selector, 1)

    def selected_months(self) -> list[int]:
        # returns month numbers (ints)
        return self.data_selector.filters_widget.selected_months()

    # --- Systems/Datasets helpers ---
    def set_systems(self, items: list[tuple[str, object]], check_all: bool = True):
        """Populate the systems multi-select. Items are (label, value) pairs."""
        self.data_selector.filters_widget.set_systems(items, check_all=check_all)

    def set_datasets(self, items: list[tuple[str, object]], check_all: bool = True):
        """Populate the datasets multi-select. Items are (label, value) pairs."""
        self.data_selector.filters_widget.set_datasets(items, check_all=check_all)

    def set_tags(self, items: list[tuple[str, object]], check_all: bool = True):
        self.data_selector.filters_widget.set_tags(items, check_all=check_all)

    def selected_systems(self) -> list[str]:
        return self.data_selector.filters_widget.selected_systems()

    def selected_datasets(self) -> list[str]:
        return self.data_selector.filters_widget.selected_datasets()

    def selected_tags(self) -> list[str]:
        return self.data_selector.filters_widget.selected_tags()

    def preprocessing_params(self) -> dict:
        return self.data_selector.preprocessing_widget.parameters()
    
    def set_groups(self, df: pd.DataFrame):
        """
        df columns: group_id, kind, label
        """
        self.data_selector.filters_widget.set_groups(df)

    def selected_group_ids(self) -> list[int]:
        return self.data_selector.filters_widget.selected_group_ids()

    def set_date_range_controls(self, start: QDateTime, end: QDateTime) -> None:
        for widget, value in (
            (self.data_selector.filters_widget.dt_from, start),
            (self.data_selector.filters_widget.dt_to, end),
        ):
            was_blocked = widget.blockSignals(True)
            widget.setDateTime(value)
            widget.blockSignals(was_blocked)
