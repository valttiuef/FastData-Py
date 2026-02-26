
from __future__ import annotations
# @ai(gpt-5, codex, refactor, 2026-02-26)
from typing import Iterable, Mapping, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (

    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QHeaderView,
)
from ...localization import tr

import pandas as pd

from ...widgets.fast_table import FastTable



class SomSavedMapsDialog(QDialog):
    load_requested = Signal(int)
    delete_requested = Signal(int)

    def __init__(
        self,
        maps: Iterable[Mapping[str, object]] = (),
        parent: Optional[QDialog] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Saved SOM"))
        self.setModal(True)

        self._columns = [
            tr("Name"),
            tr("Created"),
            tr("Map"),
            tr("Features"),
            tr("Neuron clusters"),
            tr("Feature clusters"),
            tr("QE"),
            tr("TE"),
            tr("Model ID"),
        ]

        self._table = FastTable(
            self,
            select="rows",
            single_selection=True,
            editable=False,
            initial_uniform_column_widths=True,
            sorting_enabled=False,
        )
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._table.set_stretch_column(-1)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.set_dataframe(pd.DataFrame(columns=self._columns), include_index=False)
        self._table.hideColumn(8)
        self._table.selectionChangedInstant.connect(self._update_action_buttons)

        info = QLabel(tr("Select a saved SOM to load or delete."), self)
        info.setWordWrap(True)

        self._load_button = QPushButton(tr("Load"), self)
        self._load_button.clicked.connect(self._emit_load_selected)
        self._delete_button = QPushButton(tr("Delete"), self)
        self._delete_button.clicked.connect(self._confirm_delete_selected)

        close_button = QPushButton(tr("Close"), self)
        close_button.clicked.connect(self.accept)

        actions = QHBoxLayout()
        actions.addWidget(self._load_button)
        actions.addWidget(self._delete_button)
        actions.addStretch(1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addWidget(self._table, 1)
        layout.addLayout(actions)
        layout.addLayout(footer)

        self.set_maps(maps)
        self._update_action_buttons()
        self.setMinimumSize(960, 460)
        self.resize(960, 460)

    def set_maps(self, maps: Iterable[Mapping[str, object]]) -> None:
        rows = list(maps)
        data = []
        for item in rows:
            data.append(
                {
                    self._columns[0]: str(item.get("name") or tr("Untitled")),
                    self._columns[1]: str(item.get("created_at") or ""),
                    self._columns[2]: str(item.get("map_shape") or ""),
                    self._columns[3]: item.get("features") or "",
                    self._columns[4]: item.get("neuron_clusters") or "",
                    self._columns[5]: item.get("feature_clusters") or "",
                    self._columns[6]: item.get("quantization_error") or "",
                    self._columns[7]: item.get("topographic_error") or "",
                    self._columns[8]: int(item.get("model_id")),
                }
            )
        self._table.set_dataframe(pd.DataFrame(data, columns=self._columns), include_index=False)
        self._table.hideColumn(8)
        self._update_action_buttons()

    def _selected_model_id(self) -> Optional[int]:
        index = self._table.currentIndex()
        if not index.isValid():
            return None
        row = index.row()
        model_index = self._table.model().index(row, 8)
        model_id = model_index.data(Qt.ItemDataRole.UserRole)
        if model_id is None:
            return None
        return int(model_id)

    def _update_action_buttons(self) -> None:
        has_selection = self._selected_model_id() is not None
        self._load_button.setEnabled(has_selection)
        self._delete_button.setEnabled(has_selection)

    def _emit_load_selected(self) -> None:
        model_id = self._selected_model_id()
        if model_id is not None:
            self.load_requested.emit(model_id)

    def _confirm_delete_selected(self) -> None:
        model_id = self._selected_model_id()
        if model_id is not None:
            self._confirm_delete(model_id)

    def _confirm_delete(self, model_id: int) -> None:
        confirm = QMessageBox.question(
            self,
            tr("Delete SOM"),
            tr("Delete this saved SOM?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.delete_requested.emit(model_id)

