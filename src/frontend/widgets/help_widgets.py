from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
"""
UI widgets for context-sensitive help system.

Provides InfoButton and HelpPopup for displaying help content in the GUI.
"""


from typing import Callable, Optional

from PySide6.QtCore import Qt, QPoint, QSize, Signal
from PySide6.QtGui import QFont, QCursor
from PySide6.QtWidgets import (

    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from ..localization import tr

from ..viewmodels.help_viewmodel import HelpViewModel
from .ask_ai_dialog import AskAiDialog



class InfoButton(QPushButton):
    """
    Small info icon button that shows help on hover and displays a popup on click.
    
    Usage:
        info_btn = InfoButton("app.dataset", help_viewmodel)
        layout.addWidget(info_btn)
    """

    def __init__(
        self,
        help_key: str,
        help_viewmodel: HelpViewModel,
        parent: Optional[QWidget] = None,
    ):
        """
        Initialize the info button.
        
        Args:
            help_key: The help key to display
            help_viewmodel: The help view model to use
            parent: Optional parent widget
        """
        super().__init__("ℹ", parent)
        self._help_key = help_key
        self._help_viewmodel = help_viewmodel
        self._popup: Optional[HelpPopup] = None

        self.setStyleSheet("""
        QPushButton {
            padding: 0px;
            margin: 0px;
            border: none;
            background: transparent;
        }
        """)

        self.setFixedSize(14, 14)
        
        self.setFlat(True)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        
        # Set tooltip from help system
        tooltip = self._help_viewmodel.get_tooltip(help_key)
        if tooltip:
            self.setToolTip(tooltip)
        
        # Connect click to show popup
        self.clicked.connect(self._show_popup)

    def _show_popup(self) -> None:
        """Show the help popup near this button."""
        if self._popup is None:
            parent_window = self.window()
            show_log_callback = self._resolve_log_visibility_callback(parent_window)
            log_view_model = getattr(parent_window, "log_view_model", None)
            self._popup = HelpPopup(
                self._help_key,
                self._help_viewmodel,
                parent=parent_window,
                log_view_model=log_view_model,
                show_log_callback=show_log_callback,
            )
        
        # Position popup near the button
        global_pos = self.mapToGlobal(QPoint(0, self.height()))
        self._popup.move(global_pos)
        self._popup.show()
        self._popup.raise_()
        self._popup.activateWindow()

    def set_help_key(self, help_key: str) -> None:
        """
        Change the help key for this button.
        
        Args:
            help_key: The new help key
        """
        self._help_key = help_key
        
        # Update tooltip
        tooltip = self._help_viewmodel.get_tooltip(help_key)
        if tooltip:
            self.setToolTip(tooltip)
        
        # Update popup if it exists
        if self._popup is not None:
            self._popup.set_help_key(help_key)

    def _resolve_log_visibility_callback(self, parent_window: Optional[QWidget]) -> Optional[Callable[[], None]]:
        if parent_window is None:
            return None

        show_chat = getattr(parent_window, "show_chat_window", None)
        if callable(show_chat):
            return lambda: show_chat()

        set_log_visible = getattr(parent_window, "set_log_visible", None)
        if callable(set_log_visible):
            return lambda: set_log_visible(True)

        log_window = getattr(parent_window, "log_window", None)
        if log_window is not None:
            return lambda: log_window.setVisible(True)

        return None


class HelpPopup(AskAiDialog):
    """
    Popup window that displays help title and body content.
    
    Shown as a Qt.Popup window with title and HTML body.
    """

    closed = Signal()  # Emitted when the popup is closed

    def __init__(
        self,
        help_key: str,
        help_viewmodel: HelpViewModel,
        parent: Optional[QWidget] = None,
        *,
        log_view_model=None,
        show_log_callback: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize the help popup.
        
        Args:
            help_key: The help key to display
            help_viewmodel: The help view model to use
            parent: Optional parent widget
        """
        self._help_key = help_key
        self._help_viewmodel = help_viewmodel
        self._log_view_model = log_view_model or getattr(parent, "log_view_model", None)
        self._show_log_callback = show_log_callback
        self._initial_context_text = ""

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        # Title label
        self._title_label = QLabel()
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        self._title_label.setWordWrap(True)
        content_layout.addWidget(self._title_label)

        # Body text editor (supports HTML and is editable)
        self._body_editor = QTextEdit()
        self._body_editor.setFrameStyle(QFrame.Shape.NoFrame)
        content_layout.addWidget(self._body_editor, stretch=1)

        super().__init__(
            title=tr("Help"),
            header_text=None,
            summary_text="",
            on_ask_ai=None,
            minimum_size=None,
            body_widget=content,
            editor_widget=self._body_editor,
            parent=parent,
        )
        self.setMinimumSize(350, 200)
        self.setMaximumSize(600, 800)
        
        # Load content
        self._load_content()

    def _load_content(self) -> None:
        """Load help content from the view model."""
        # Get title
        title = self._help_viewmodel.get_title(self._help_key)
        if title:
            self._title_label.setText(title)
        else:
            self._title_label.setText(tr("Help"))
        
        # Get body
        body = self._help_viewmodel.get_body(self._help_key)
        if body:
            self._body_editor.setHtml(body)
        else:
            self._body_editor.setPlainText(
                tr("No help available for '{help_key}'").format(help_key=self._help_key)
            )
        self._initial_context_text = self.get_summary_text().strip()

    def set_help_key(self, help_key: str) -> None:
        """
        Change the help key and reload content.
        
        Args:
            help_key: The new help key
        """
        self._help_key = help_key
        self._load_content()

    def _ask_from_ai(self) -> None:
        """Send a contextual prompt about this help topic to the LLM log."""
        prompt, _iteration = self._help_viewmodel.build_ai_prompt(self._help_key)

        if not prompt:
            return

        context_text = self.get_summary_text().strip()
        if context_text and context_text != self._initial_context_text:
            user_context = context_text
            if self._initial_context_text and context_text.startswith(self._initial_context_text):
                user_context = context_text[len(self._initial_context_text):].strip()
            if user_context:
                prompt = f"{prompt}\n\nUser context:\n{user_context}"

        log_view_model = self._resolve_log_view_model()
        if log_view_model is None:
            return

        self._show_log_window()
        log_view_model.ask_llm(prompt)

    def _handle_ask_ai(self) -> None:
        self._ask_from_ai()

    def _resolve_log_view_model(self):
        if self._log_view_model is not None:
            return self._log_view_model

        parent_window = self.parent()
        if parent_window is not None:
            self._log_view_model = getattr(parent_window, "log_view_model", None)

        return self._log_view_model

    def _show_log_window(self) -> None:
        if self._show_log_callback is not None:
            try:
                self._show_log_callback()
                return
            except Exception:
                logger.warning("Exception in _show_log_window", exc_info=True)

        parent_window = self.parent()
        if parent_window is not None:
            show_chat = getattr(parent_window, "show_chat_window", None)
            if callable(show_chat):
                try:
                    show_chat()
                    return
                except Exception:
                    logger.warning("Exception in _show_log_window", exc_info=True)
            set_log_visible = getattr(parent_window, "set_log_visible", None)
            if callable(set_log_visible):
                try:
                    set_log_visible(True)
                except Exception:
                    logger.warning("Exception in _show_log_window", exc_info=True)

    def closeEvent(self, event: "QCloseEvent") -> None:  # type: ignore
        """Override close event to emit signal."""
        self.closed.emit()
        super().closeEvent(event)

    def sizeHint(self) -> QSize:
        """Provide a reasonable default size."""
        return QSize(450, 400)


def attach_help(widget: QWidget, help_key: str) -> None:
    """
    Attach a help key to a widget via Qt property.
    
    This allows the widget to be associated with help content without
    modifying its implementation.
    
    Args:
        widget: The widget to attach help to
        help_key: The help key to associate
        
    Example:
        attach_help(my_label, "app.dataset")
    """
    widget.setProperty("help_key", help_key)


def get_help_key(widget: QWidget) -> Optional[str]:
    """
    Retrieve the help key attached to a widget.
    
    Args:
        widget: The widget to check
        
    Returns:
        The help key string, or None if not set
    """
    value = widget.property("help_key")
    return value if isinstance(value, str) else None
