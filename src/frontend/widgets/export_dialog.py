
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..localization import tr


@dataclass(frozen=True)
class ExportOption:
    key: str
    label: str
    description: str = ""


class ExportSelectionDialog(QDialog):
    """Shared dialog that lets the user choose datasets and output format."""

    def __init__(
        self,
        *,
        title: str,
        heading: str,
        options: Iterable[ExportOption],
        default_selected_keys: Optional[Iterable[str]] = None,
        show_format: bool = True,
        show_chart_data_options: bool = False,
        chart_label: str = "Charts",
        data_label: str = "Data",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(520, 460)

        layout = QVBoxLayout(self)

        self._heading_label = QLabel(heading, self)
        self._heading_label.setWordWrap(True)
        layout.addWidget(self._heading_label)

        self.format_combo = QComboBox(self)
        self.format_combo.addItem(tr("CSV"), "csv")
        self.format_combo.addItem(tr("Excel (.xlsx)"), "excel")
        self.format_combo.setCurrentIndex(1)
        self.format_combo.setVisible(bool(show_format))
        if show_format:
            format_row = QHBoxLayout()
            format_row.addWidget(QLabel(tr("Format:"), self))
            format_row.addWidget(self.format_combo, 1)
            layout.addLayout(format_row)

        self._chart_checkbox: QCheckBox | None = None
        self._data_checkbox: QCheckBox | None = None
        if show_chart_data_options:
            include_row = QHBoxLayout()
            include_row.addWidget(QLabel(tr("Include:"), self))
            self._chart_checkbox = QCheckBox(tr(chart_label), self)
            self._data_checkbox = QCheckBox(tr(data_label), self)
            self._chart_checkbox.setChecked(True)
            self._data_checkbox.setChecked(True)
            include_row.addWidget(self._chart_checkbox)
            include_row.addWidget(self._data_checkbox)
            include_row.addStretch(1)
            layout.addLayout(include_row)

        self._options_group = QGroupBox(tr("Data to export"), self)
        options_layout = QVBoxLayout(self._options_group)

        self.select_all_checkbox = QCheckBox(tr("Select all"), self._options_group)
        self.select_all_checkbox.setChecked(True)
        options_layout.addWidget(self.select_all_checkbox)

        scroll = QScrollArea(self._options_group)
        scroll.setWidgetResizable(True)
        options_container = QWidget(scroll)
        self._option_layout = QVBoxLayout(options_container)
        self._option_layout.setContentsMargins(0, 0, 0, 0)
        self._option_layout.setSpacing(6)
        scroll.setWidget(options_container)
        options_layout.addWidget(scroll, 1)
        layout.addWidget(self._options_group, 1)

        selected_defaults = set(default_selected_keys or [])
        use_default_subset = bool(selected_defaults)
        self._option_checkboxes: dict[str, QCheckBox] = {}
        for option in options:
            cb_text = option.label
            if option.description:
                cb_text = f"{option.label} — {option.description}"
            checkbox = QCheckBox(cb_text, options_container)
            checkbox.setChecked(option.key in selected_defaults if use_default_subset else True)
            checkbox.setToolTip(option.description or option.label)
            checkbox.stateChanged.connect(self._sync_select_all_checkbox)
            self._option_layout.addWidget(checkbox)
            self._option_checkboxes[option.key] = checkbox

        self._option_layout.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.select_all_checkbox.toggled.connect(self._set_all_options_checked)
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        self._sync_select_all_checkbox()
        self._on_format_changed()

    def selected_keys(self) -> list[str]:
        return [key for key, cb in self._option_checkboxes.items() if cb.isChecked()]

    def selected_format(self) -> str:
        return str(self.format_combo.currentData() or "csv")

    def include_charts(self) -> bool:
        return True if self._chart_checkbox is None else bool(self._chart_checkbox.isChecked())

    def include_data(self) -> bool:
        return True if self._data_checkbox is None else bool(self._data_checkbox.isChecked())

    def _set_all_options_checked(self, checked: bool) -> None:
        for checkbox in self._option_checkboxes.values():
            checkbox.setChecked(bool(checked))

    def _sync_select_all_checkbox(self) -> None:
        if not self._option_checkboxes:
            self.select_all_checkbox.setChecked(False)
            self.select_all_checkbox.setEnabled(False)
            return
        states = [cb.isChecked() for cb in self._option_checkboxes.values()]
        all_checked = all(states)
        any_checked = any(states)
        block = self.select_all_checkbox.blockSignals(True)
        try:
            self.select_all_checkbox.setChecked(all_checked)
            self.select_all_checkbox.setTristate(True)
            if any_checked and not all_checked:
                self.select_all_checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
            elif all_checked:
                self.select_all_checkbox.setCheckState(Qt.CheckState.Checked)
            else:
                self.select_all_checkbox.setCheckState(Qt.CheckState.Unchecked)
        finally:
            self.select_all_checkbox.blockSignals(block)

    def _on_format_changed(self) -> None:
        """Enable/disable charts checkbox based on selected format."""
        if self._chart_checkbox is not None:
            is_excel = self.selected_format() == "excel"
            self._chart_checkbox.setEnabled(is_excel)


__all__ = ["ExportOption", "ExportSelectionDialog"]
