
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette, QTextCharFormat, QTextCursor, QTextDocument
from PySide6.QtWidgets import (


    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)
from ..localization import tr

from ..models.log_model import LogEvent
from ..models.settings_model import SettingsModel
from ..viewmodels import LogViewModel
from ..widgets.collapsible_section import CollapsibleSection
from ..widgets.multi_check_combo import MultiCheckCombo



class LogWindow(QWidget):
    """Widget showing application logs with filtering and search."""

    def __init__(
        self,
        view_model: LogViewModel,
        settings_model: SettingsModel,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._view_model = view_model
        self._model = view_model.log_model
        self._settings_model = settings_model
        self._preferred_sources = set(self._settings_model.log_sources)
        self._sources_restored = False
        self._last_search_text = ""
        self._search_dirty = True

        self.setObjectName("logWindow")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        settings_section = CollapsibleSection(tr("Settings"), collapsed=True, parent=self)
        settings_layout = settings_section.body_layout()
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(6)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        settings_layout.addLayout(grid)

        self._filter_bar = QWidget(self)
        filter_layout = QHBoxLayout(self._filter_bar)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(8)
        self._logger_selector = MultiCheckCombo(self._filter_bar, placeholder=tr("All sources"))
        self._logger_selector.set_summary_max(6)
        self._logger_selector.selection_changed.connect(self._on_logger_selection_changed)
        filter_layout.addWidget(self._logger_selector, stretch=1)
        self._filter_bar.setVisible(False)
        sources_label = QLabel(tr("Sources:"), self)
        grid.addWidget(sources_label, 0, 0)
        grid.addWidget(self._filter_bar, 0, 1)

        labels = [sources_label]

        max_width = max(label.sizeHint().width() for label in labels)
        grid.setColumnMinimumWidth(0, max_width)

        for label in labels:
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(6)

        self._load_log_button = QPushButton(tr("Load log…"), self)
        self._load_log_button.clicked.connect(self._load_log_database)
        button_row.addWidget(self._load_log_button, 1)

        self._save_log_button = QPushButton(tr("Save log as…"), self)
        self._save_log_button.clicked.connect(self._save_log_database_as)
        button_row.addWidget(self._save_log_button, 1)

        self._reset_log_button = QPushButton(tr("Reset log"), self)
        self._reset_log_button.clicked.connect(self._confirm_reset_log)
        button_row.addWidget(self._reset_log_button, 1)

        settings_layout.addLayout(button_row)

        layout.addWidget(settings_section)

        self._view = QTextEdit(self)
        self._view.setReadOnly(True)
        self._view.setObjectName("logWindowView")
        layout.addWidget(self._view, stretch=1)

        search_row = QHBoxLayout()
        search_row.setContentsMargins(0, 4, 0, 0)
        search_row.setSpacing(8)
        search_row.addWidget(QLabel(tr("Search:"), self))
        self._search_input = QLineEdit(self)
        self._search_input.setPlaceholderText(tr("Search logs…"))
        self._search_input.returnPressed.connect(self._search_next)
        self._search_input.textChanged.connect(self._mark_search_dirty)
        search_row.addWidget(self._search_input, stretch=1)
        self._search_button = QPushButton(tr("Search"), self)
        self._search_button.clicked.connect(self._search_next)
        search_row.addWidget(self._search_button)
        layout.addLayout(search_row)

        self._model.entry_added.connect(self._append_entry)
        self._model.cleared.connect(self._clear_view)
        self._model.loggers_changed.connect(self._on_loggers_changed)
        self._model.filter_changed.connect(self._refresh_view)

        self._updating_logger_selector = False

        self._on_loggers_changed(self._model.available_loggers())
        for entry in self._model.filtered_entries():
            self._append_entry(entry)

        try:
            self._settings_model.set_log_database_path(self._view_model.current_log_database_path())
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)

        self._mark_search_dirty()

    # ------------------------------------------------------------------
    def _append_entry(self, event: LogEvent) -> None:
        cursor = self._view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        default_color = self._view.palette().color(QPalette.Text)
        fmt.setForeground(default_color)
        color = self._color_for_level(event.level)
        if color is not None:
            fmt.setForeground(QColor(color))

        cursor.insertText(f"{event.formatted}\n", fmt)
        self._view.setTextCursor(cursor)
        self._view.ensureCursorVisible()
        self._mark_search_dirty()

    def _clear_view(self) -> None:
        self._view.clear()
        self._mark_search_dirty()

    def _refresh_view(self) -> None:
        entries = self._model.filtered_entries()
        text = "\n".join(entry.formatted for entry in entries)
        self._view.setUpdatesEnabled(False)
        try:
            self._clear_view()
            if text:
                self._view.setPlainText(text)
            self._update_logger_selector()
            cursor = self._view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._view.setTextCursor(cursor)
            self._mark_search_dirty()
        finally:
            self._view.setUpdatesEnabled(True)

    def _on_loggers_changed(self, loggers: list[str]) -> None:
        self._filter_bar.setVisible(bool(loggers))
        items = [(name, name) for name in sorted(loggers)]
        self._updating_logger_selector = True
        try:
            if items:
                self._logger_selector.set_items(items, check_all=True)
            else:
                self._logger_selector.clear_items()
        finally:
            self._updating_logger_selector = False
        self._apply_saved_sources(loggers)
        self._update_logger_selector()

    def _on_logger_selection_changed(self) -> None:
        if self._updating_logger_selector:
            return
        enabled = self._logger_selector.selected_values()
        self._model.set_enabled_loggers([str(name) for name in enabled])
        self._preferred_sources = set(self._model.enabled_logger_names())
        self._save_sources()

    def _update_logger_selector(self) -> None:
        self._updating_logger_selector = True
        try:
            enabled = self._model.enabled_logger_names()
            self._logger_selector.set_selected_values(enabled)
        finally:
            self._updating_logger_selector = False

    def _apply_saved_sources(self, loggers: list[str]) -> None:
        if self._updating_logger_selector:
            return

        available = [name for name in loggers if name]
        if not available:
            return

        if not self._sources_restored:
            preferred = [name for name in self._preferred_sources if name in available]
            if preferred:
                self._model.set_enabled_loggers(preferred)
                self._logger_selector.set_selected_values(preferred)
            else:
                self._logger_selector.set_selected_values(self._model.enabled_logger_names())
            self._sources_restored = True

        self._save_sources()

    def _save_sources(self) -> None:
        enabled = self._model.enabled_logger_names()
        self._settings_model.set_log_sources(enabled)

    # ------------------------------------------------------------------
    def _prepare_for_storage_change(self) -> None:
        saved_sources = self._settings_model.log_sources
        self._preferred_sources = set(saved_sources or self._model.enabled_logger_names())
        self._sources_restored = False

    def _handle_storage_success(self, path: Path, message: str) -> None:
        self._settings_model.set_log_database_path(path)
        self._model.log_text(message, origin="ui.log")
        self._save_sources()

    def _show_storage_error(self, title: str, error: Exception) -> None:
        self._model.log_text(f"{title}: {error}", level=logging.ERROR, origin="ui.log")

    def _confirm_reset_log(self) -> None:
        reply = QMessageBox.question(
            self,
            tr("Reset log"),
            tr("This will clear the log and reset it to the default Dataset. Continue?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._reset_log_database()

    def _load_log_database(self) -> None:
        start_dir = str(self._settings_model.log_database_path)
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            tr("Open log database"),
            start_dir,
            tr("Log database files (*.db);;All Files (*)"),
        )
        if not path_str:
            return

        try:
            self._prepare_for_storage_change()
            target = Path(path_str)
            self._view_model.set_log_database(target)
        except Exception as exc:  # pragma: no cover - UI guard
            self._show_storage_error(tr("Failed to load log database"), exc)
            return

        self._handle_storage_success(
            target,
            tr("Loaded log database from {name}.").format(name=Path(path_str).name),
        )

    def _save_log_database_as(self) -> None:
        start_dir = str(self._settings_model.log_database_path)
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            tr("Save log database as"),
            start_dir,
            tr("Log database files (*.db);;All Files (*)"),
        )
        if not path_str:
            return

        try:
            self._prepare_for_storage_change()
            target = Path(path_str)
            new_path = self._view_model.save_log_database_as(target)
        except Exception as exc:  # pragma: no cover - UI guard
            self._show_storage_error(tr("Failed to save log database"), exc)
            return

        self._handle_storage_success(
            new_path,
            tr("Saved log database to {name}.").format(name=Path(path_str).name),
        )

    def _reset_log_database(self) -> None:
        try:
            self._prepare_for_storage_change()
            path = self._view_model.reset_log_database()
        except Exception as exc:  # pragma: no cover - UI guard
            self._show_storage_error(tr("Failed to reset log database"), exc)
            return

        self._handle_storage_success(path, tr("Cleared log and reset database to default Dataset."))

    def _mark_search_dirty(self) -> None:
        self._search_dirty = True

    def _search_next(self) -> None:
        query = self._search_input.text().strip()
        if not query:
            return

        if self._search_dirty or query != self._last_search_text:
            cursor = self._view.textCursor()
            cursor.clearSelection()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._view.setTextCursor(cursor)
            self._last_search_text = query
            self._search_dirty = False
        else:
            cursor = self._view.textCursor()
            if cursor.hasSelection():
                cursor.setPosition(cursor.selectionStart())
                self._view.setTextCursor(cursor)

        found = self._view.find(query, QTextDocument.FindFlag.FindBackward)
        if not found:
            self._search_dirty = True

    @staticmethod
    def _color_for_level(level: int) -> Optional[str]:
        if level >= logging.ERROR:
            return "#f06292"
        if level >= logging.WARNING:
            return "#ffb74d"
        return None

