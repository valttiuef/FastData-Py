
from __future__ import annotations
from typing import Callable, Optional

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (


    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import logging
logger = logging.getLogger(__name__)
from ..localization import tr



class AskAiDialog(QDialog):
    """Reusable dialog with an editor and an Ask from AI action."""

    def __init__(
        self,
        *,
        title: str,
        header_text: Optional[str],
        summary_text: str,
        on_ask_ai: Optional[Callable[[str], None]],
        minimum_size: Optional[tuple[int, int]] = None,
        body_widget: Optional[QWidget] = None,
        editor_widget: Optional[QTextEdit] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self._on_ask_ai = on_ask_ai
        self._adjusting_position = False
        self._editor: Optional[QTextEdit] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        if header_text:
            header = QLabel(header_text)
            header.setWordWrap(True)
            layout.addWidget(header)

        if editor_widget is not None:
            self._editor = editor_widget
            if minimum_size:
                self._editor.setMinimumSize(*minimum_size)
        else:
            self._editor = QTextEdit(self)
            self._editor.setPlainText(summary_text)
            if minimum_size:
                self._editor.setMinimumSize(*minimum_size)

        if body_widget is not None:
            layout.addWidget(body_widget, 1)
        else:
            layout.addWidget(self._editor, 1)

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch(1)

        self._ask_button = QPushButton(tr("Ask from AI"), self)
        self._ask_button.clicked.connect(self._handle_ask_ai)
        buttons_layout.addWidget(self._ask_button)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.reject)
        buttons_layout.addWidget(buttons)

        layout.addLayout(buttons_layout)

    def set_summary_text(self, text: str) -> None:
        """Replace the contents of the editor with new summary text."""
        if self._editor is None:
            return
        self._editor.setPlainText(text)

    def get_summary_text(self) -> str:
        """Return the current contents of the editor."""
        if self._editor is None:
            return ""
        return self._editor.toPlainText()

    def _handle_ask_ai(self) -> None:
        if callable(self._on_ask_ai):
            try:
                self._on_ask_ai(self.get_summary_text())
            except Exception:
                logger.warning("Exception in _handle_ask_ai", exc_info=True)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._ensure_on_screen()

    def moveEvent(self, event) -> None:  # type: ignore[override]
        super().moveEvent(event)
        self._ensure_on_screen()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._ensure_on_screen()

    def _ensure_on_screen(self) -> None:
        if self._adjusting_position:
            return
        self._adjusting_position = True
        try:
            frame = self.frameGeometry()
            screen = QGuiApplication.screenAt(frame.center()) or QGuiApplication.primaryScreen()
            if screen is None:
                return
            available = screen.availableGeometry()
            max_x = available.x() + max(0, available.width() - frame.width())
            max_y = available.y() + max(0, available.height() - frame.height())
            x = min(max(frame.x(), available.x()), max_x)
            y = min(max(frame.y(), available.y()), max_y)
            if x != frame.x() or y != frame.y():
                self.move(x, y)
        finally:
            self._adjusting_position = False


__all__ = ["AskAiDialog"]
