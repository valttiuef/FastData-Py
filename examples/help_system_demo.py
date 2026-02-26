#!/usr/bin/env python
"""
Widget System Demo for quick UI testing.

This example demonstrates:
1. Help system widgets (InfoButton / attach_help)
2. Toast notifications (ToastManager) with spam controls
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from backend.help_manager import get_help_manager
from frontend.style.styles import apply_theme, detect_system_theme
from frontend.viewmodels.help_viewmodel import get_help_viewmodel
from frontend.widgets.help_widgets import InfoButton, attach_help
from frontend.widgets.toast import ToastManager


class WidgetSystemDemo(QWidget):
    """Demo window for quickly testing widgets in isolation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Widget System Demo")
        self.resize(780, 560)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )

        # Initialize help system
        help_folder = Path(__file__).parent.parent / "resources" / "help"
        help_manager = get_help_manager(help_folder)
        self.help_viewmodel = get_help_viewmodel(help_manager, parent=self)

        # Toasts
        self.toast_manager = ToastManager(anchor=self)

        self._create_ui()

    def _create_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("<h1>Widget System Demo</h1>", self)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        instructions = QLabel(
            "<p>Use this window to quickly test widgets without launching the full app.</p>", self
        )
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Theme:", self))
        self.theme_combo = QComboBox(self)
        self.theme_combo.addItems(["dark", "light"])
        theme_row.addWidget(self.theme_combo)
        theme_row.addStretch(1)
        layout.addLayout(theme_row)

        tabs = QTabWidget(self)
        tabs.addTab(self._build_toast_tab(), "Toast")
        tabs.addTab(self._build_help_tab(), "Help System")
        layout.addWidget(tabs, 1)

        stats = QLabel(
            f"<i>Help system loaded with version {self.help_viewmodel.version}, "
            f"app: {self.help_viewmodel.metadata.get('app', 'Unknown')}</i>",
            self,
        )
        stats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(stats)

        close_btn = QPushButton("Close", self)
        close_btn.setMaximumWidth(120)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def _build_toast_tab(self) -> QWidget:
        page = QWidget(self)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(12)

        group = QGroupBox("Toast Tester", page)
        form = QFormLayout(group)

        self.toast_kind = QComboBox(group)
        self.toast_kind.addItems(["info", "success", "warn", "error"])

        self.toast_title = QLineEdit(group)
        self.toast_title.setText("Info")

        self.toast_msec = QSpinBox(group)
        self.toast_msec.setRange(250, 30000)
        self.toast_msec.setSingleStep(250)
        self.toast_msec.setValue(2500)

        self.toast_message = QPlainTextEdit(group)
        self.toast_message.setPlainText("This is a toast message.\nSecond line wraps nicely.")
        self.toast_message.setMinimumHeight(90)

        self.toast_count = QSpinBox(group)
        self.toast_count.setRange(1, 200)
        self.toast_count.setValue(5)

        self.toast_interval = QSpinBox(group)
        self.toast_interval.setRange(0, 5000)
        self.toast_interval.setSingleStep(25)
        self.toast_interval.setValue(125)

        form.addRow("Kind:", self.toast_kind)
        form.addRow("Title:", self.toast_title)
        form.addRow("Duration (ms):", self.toast_msec)
        form.addRow("Message:", self.toast_message)
        form.addRow("Spam count:", self.toast_count)
        form.addRow("Spam interval (ms):", self.toast_interval)

        btn_row = QHBoxLayout()
        show_btn = QPushButton("Show toast", group)
        spam_btn = QPushButton("Spam", group)
        show_btn.clicked.connect(self._show_one_toast)
        spam_btn.clicked.connect(self._spam_toasts)
        btn_row.addWidget(show_btn)
        btn_row.addWidget(spam_btn)
        btn_row.addStretch(1)
        form.addRow("", btn_row)

        page_layout.addWidget(group)
        page_layout.addStretch(1)
        return page

    def _build_help_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        header = QLabel(
            "<p>Hover over the <b>?</b> icons to see tooltips.<br>"
            "Click the icons to open detailed help popups.</p>",
            page,
        )
        layout.addWidget(header)

        layout.addWidget(QLabel("<h3>Dataset</h3>", page))
        row = QHBoxLayout()
        row.addWidget(QLabel("Current Dataset:", page))
        row.addWidget(InfoButton("app.dataset", self.help_viewmodel))
        row.addStretch(1)
        layout.addLayout(row)

        layout.addWidget(QLabel("<h3>Regression</h3>", page))
        row = QHBoxLayout()
        row.addWidget(QLabel("Regression Model:", page))
        row.addWidget(InfoButton("app.regression", self.help_viewmodel))
        row.addStretch(1)
        layout.addLayout(row)

        layout.addWidget(QLabel("<h3>Dynamic Feature (Wildcard)</h3>", page))
        row = QHBoxLayout()
        row.addWidget(QLabel("Feature: AE_RMS", page))
        row.addWidget(InfoButton("feature:AE_RMS", self.help_viewmodel))
        row.addStretch(1)
        layout.addLayout(row)

        layout.addWidget(QLabel("<h3>Property-Based Help</h3>", page))
        help_label = QLabel("(Help key attached via widget property)", page)
        attach_help(help_label, "app.preprocessing")
        help_key = help_label.property("help_key")
        row = QHBoxLayout()
        row.addWidget(QLabel(f"Attached help key: '{help_key}'", page))
        row.addWidget(InfoButton("app.preprocessing", self.help_viewmodel))
        row.addStretch(1)
        layout.addLayout(row)

        layout.addWidget(QLabel("<h3>Self-Organizing Maps</h3>", page))
        row = QHBoxLayout()
        row.addWidget(QLabel("SOM Clustering:", page))
        row.addWidget(InfoButton("app.som", self.help_viewmodel))
        row.addStretch(1)
        layout.addLayout(row)

        layout.addStretch(1)
        return page

    def _show_one_toast(self) -> None:
        kind = self.toast_kind.currentText().strip().lower()
        title = self.toast_title.text().strip()
        message = self.toast_message.toPlainText()
        msec = int(self.toast_msec.value())

        if kind == "success":
            self.toast_manager.success(message, title=title or "Done", msec=msec)
        elif kind == "warn":
            self.toast_manager.warn(message, title=title or "Warning", msec=msec)
        elif kind == "error":
            self.toast_manager.error(message, title=title or "Error", msec=msec)
        else:
            self.toast_manager.info(message, title=title or "Info", msec=msec)

    def _spam_toasts(self) -> None:
        count = int(self.toast_count.value())
        interval = int(self.toast_interval.value())
        for i in range(count):
            QTimer.singleShot(i * interval, self._show_one_toast)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    help_folder = Path(__file__).parent.parent / "resources" / "help"
    if not help_folder.exists():
        print(f"Error: Help files not found at {help_folder}")
        print("Please run this script from the repo root.")
        sys.exit(1)

    demo = WidgetSystemDemo()

    initial_theme = detect_system_theme()
    try:
        apply_theme(app, initial_theme)
    except Exception:
        pass
    demo.theme_combo.setCurrentText(initial_theme if initial_theme in ("dark", "light") else "dark")

    def _apply_selected_theme() -> None:
        theme = demo.theme_combo.currentText().strip().lower()
        if theme not in ("dark", "light"):
            return
        try:
            apply_theme(app, theme)
        except Exception:
            pass

    demo.theme_combo.currentTextChanged.connect(lambda *_: _apply_selected_theme())
    demo.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
