
from __future__ import annotations
from PySide6.QtWidgets import QVBoxLayout, QWidget

# Margins INSIDE the panel content area (padding between frame and inner widgets)
PANEL_CONTENT_MARGINS = (4, 4, 4, 4)

# Margins OUTSIDE the panel itself (spacing around the panel in its parent layout)
PANEL_OUTER_MARGINS = (4, 0, 4, 0)

PANEL_CONTENT_SPACING = 4


class Panel(QWidget):
    """
    Lightweight titled panel with adjustable inner and outer margins.

    Usage:
        pnl = Panel("My Panel")
        pnl.content_layout().addWidget(someWidget)
    """
    def __init__(self, title: str = "", parent: QWidget | None = None):
        super().__init__(parent)

        # --- Outer wrapper layout to give the panel space in parent layout
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(*PANEL_OUTER_MARGINS)
        outer_layout.setSpacing(0)  # spacing between outer frame and content

        # --- Inner layout for content inside the panel frame
        self._content_layout = QVBoxLayout()
        self._content_layout.setContentsMargins(*PANEL_CONTENT_MARGINS)
        self._content_layout.setSpacing(PANEL_CONTENT_SPACING)

        # --- Apply layout hierarchy
        outer_layout.addLayout(self._content_layout)

    # Accessors ------------------------------------------------------------
    def content_layout(self) -> QVBoxLayout:
        """Return the inner content layout for adding widgets."""
        return self._content_layout

