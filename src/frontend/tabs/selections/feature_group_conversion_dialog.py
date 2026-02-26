from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...localization import tr


class FeatureGroupConversionDialog(QDialog):
    def __init__(
        self,
        *,
        feature_name: str,
        unique_values: List[str],
        unique_count: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._unique_values = list(unique_values or [])
        self._unique_count = int(unique_count)
        self.setModal(True)
        self.setWindowTitle(tr("Convert feature values to groups"))
        self.setMinimumWidth(560)
        self._build_ui(feature_name=str(feature_name or "").strip())
        self.adjustSize()

    def _build_ui(self, *, feature_name: str) -> None:
        root = QVBoxLayout(self)
        info_form = QFormLayout()

        selected_feature = QLabel(feature_name or tr("(Unnamed feature)"), self)
        selected_feature.setWordWrap(True)
        info_form.addRow(tr("Selected feature:"), selected_feature)

        unique_count_label = QLabel(str(self._unique_count), self)
        info_form.addRow(tr("Unique values:"), unique_count_label)

        self.group_name_edit = QLineEdit(feature_name, self)
        self.group_name_edit.setPlaceholderText(tr("Group feature name"))
        info_form.addRow(tr("Group feature name:"), self.group_name_edit)
        root.addLayout(info_form)

        self.remove_original_checkbox = QCheckBox(tr("Remove original measurement feature"), self)
        self.remove_original_checkbox.setChecked(True)
        root.addWidget(self.remove_original_checkbox)

        self.save_as_timeframes_checkbox = QCheckBox(tr("Save as timeframes"), self)
        self.save_as_timeframes_checkbox.setChecked(True)
        root.addWidget(self.save_as_timeframes_checkbox)

        if self._unique_count < 20:
            rename_label = QLabel(
                tr("Optionally rename unique values before saving as database groups."),
                self,
            )
            rename_label.setWordWrap(True)
            root.addWidget(rename_label)

            self.rename_table = QTableWidget(len(self._unique_values), 2, self)
            self.rename_table.setHorizontalHeaderLabels([tr("Original value"), tr("Group label")])
            self.rename_table.verticalHeader().setVisible(False)
            self.rename_table.setAlternatingRowColors(True)
            self.rename_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.rename_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            self.rename_table.setEditTriggers(
                QTableWidget.EditTrigger.DoubleClicked
                | QTableWidget.EditTrigger.EditKeyPressed
                | QTableWidget.EditTrigger.AnyKeyPressed
            )
            self.rename_table.horizontalHeader().setStretchLastSection(True)
            for idx, value in enumerate(self._unique_values):
                original_item = QTableWidgetItem(value)
                original_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                renamed_item = QTableWidgetItem(value)
                self.rename_table.setItem(idx, 0, original_item)
                self.rename_table.setItem(idx, 1, renamed_item)
            row_count = max(1, int(len(self._unique_values)))
            row_height = int(self.rename_table.verticalHeader().defaultSectionSize() or 30)
            header_height = int(self.rename_table.horizontalHeader().height() or 30)
            frame_height = int(self.rename_table.frameWidth() * 2)
            desired_height = min(300, header_height + frame_height + (row_count * row_height) + 4)
            self.rename_table.setMinimumHeight(desired_height)
            self.rename_table.setMaximumHeight(desired_height)
            root.addWidget(self.rename_table)
        else:
            self.rename_table = None
            too_many_label = QLabel(
                tr("Value renaming is available when unique values are fewer than 20."),
                self,
            )
            too_many_label.setWordWrap(True)
            root.addWidget(too_many_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(buttons)
        root.addLayout(button_row)

    def values(self) -> dict:
        name_map: Dict[str, str] = {}
        if self.rename_table is not None:
            for row in range(self.rename_table.rowCount()):
                original_item = self.rename_table.item(row, 0)
                renamed_item = self.rename_table.item(row, 1)
                original = str(original_item.text() if original_item else "").strip()
                renamed = str(renamed_item.text() if renamed_item else "").strip()
                if original and renamed:
                    name_map[original] = renamed
        return {
            "group_name": str(self.group_name_edit.text() or "").strip(),
            "remove_original_feature": bool(self.remove_original_checkbox.isChecked()),
            "save_as_timeframes": bool(self.save_as_timeframes_checkbox.isChecked()),
            "value_name_map": name_map,
        }
