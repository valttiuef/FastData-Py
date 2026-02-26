
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .panel import Panel


SIDEBAR_WIDGET_MARGINS = (4, 0, 4, 0)
SIDEBAR_CONTENT_MARGINS = (4, 8, 4, 8)
SIDEBAR_CONTENT_SPACING = 4
SIDEBAR_ACTIONS_MARGINS = (4, 8, 4, 4)
SIDEBAR_ACTIONS_SPACING = 4

SIDEBAR_MINIMUM_WIDTH = 250

class SidebarWidget(QWidget):
    """Common sidebar container with scrollable panel content."""

    def __init__(
        self,
        title: str = "",
        parent: QWidget | None = None,
        *,
        minimum_width: int = SIDEBAR_MINIMUM_WIDTH,
    ) -> None:
        super().__init__(parent)

        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # type: ignore
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # type: ignore

        self._actions_container = QWidget(self)
        self._actions_layout = QVBoxLayout(self._actions_container)
        self._actions_layout.setContentsMargins(*SIDEBAR_ACTIONS_MARGINS)
        self._actions_layout.setSpacing(SIDEBAR_ACTIONS_SPACING)
        self._actions_container.setVisible(False)

        # no title
        self._panel = Panel(title="", parent=self)
        self._panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        self._scroll_area.setWidget(self._panel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*SIDEBAR_WIDGET_MARGINS)
        layout.setSpacing(0)
        layout.addWidget(self._actions_container)
        layout.addWidget(self._scroll_area)

        content_layout = self._panel.content_layout()
        content_layout.setContentsMargins(*SIDEBAR_CONTENT_MARGINS)
        content_layout.setSpacing(SIDEBAR_CONTENT_SPACING)

        # Preferred so splitter honors our hint but still allows expansion
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        if minimum_width > 0:
            self.setMinimumWidth(minimum_width)

    def content_layout(self) -> QVBoxLayout:
        """Expose the underlying panel layout for subclasses."""

        return self._panel.content_layout()

    def panel(self) -> Panel:
        """Return the panel used as scrollable content."""

        return self._panel

    def scroll_area(self) -> QScrollArea:
        """Return the internal scroll area widget."""

        return self._scroll_area

    def set_sticky_actions(self, widget: QWidget) -> None:
        """Place *widget* in the non-scrolling actions area."""

        if widget is None:
            return
        self._actions_layout.addWidget(widget)
        self._actions_container.setVisible(True)


__all__ = ["SidebarWidget"]
