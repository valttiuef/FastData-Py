
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QSplitter,
    QWidget,
)

from ..widgets.sidebar_widget import SIDEBAR_MINIMUM_WIDTH

SIDEBAR_MAXIMUM_WIDTH = SIDEBAR_MINIMUM_WIDTH + 550
SIDEBAR_INITIAL_WIDTH = SIDEBAR_MINIMUM_WIDTH + 250


class TabWidget(QWidget):
    """Base widget that provides a sidebar/content splitter layout."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        sidebar_minimum: int | None = SIDEBAR_MINIMUM_WIDTH,
        sidebar_maximum: int | None = SIDEBAR_MAXIMUM_WIDTH,
        sidebar_initial: int | None = SIDEBAR_INITIAL_WIDTH,
    ) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setHandleWidth(6)
        layout.addWidget(self.splitter, 1)


        # Sidebar
        self.sidebar = self._create_sidebar()
        if self.sidebar is not None:
            if sidebar_minimum is not None:
                self.sidebar.setMinimumWidth(sidebar_minimum)
            if sidebar_maximum is not None:
                self.sidebar.setMaximumWidth(sidebar_maximum)
            self.sidebar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            self.splitter.addWidget(self.sidebar)
            self.splitter.setStretchFactor(self.splitter.indexOf(self.sidebar), 0)

        # Content
        self.content_widget = self._create_content_widget()
        if self.content_widget is not None:
            self.content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.splitter.addWidget(self.content_widget)
            self.splitter.setStretchFactor(self.splitter.indexOf(self.content_widget), 1)

        # --- Set initial width AFTER widgets are added ---
        if sidebar_initial is None:
            sidebar_initial = sidebar_minimum or 320  # sensible fallback
        # first value = sidebar width, second = content width
        self.splitter.setSizes([sidebar_initial, 10000])

    def _create_sidebar(self) -> QWidget:
        """Return the sidebar widget used by the tab."""
        raise NotImplementedError

    def _create_content_widget(self) -> QWidget:
        """Return the main content widget for the tab."""
        raise NotImplementedError


__all__ = ["TabWidget", "SIDEBAR_MAXIMUM_WIDTH"]
