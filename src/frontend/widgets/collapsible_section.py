
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QToolButton, QFrame, QVBoxLayout, QGroupBox
from ..localization import tr


class CollapsibleSection(QWidget):
    """
    Simple collapsible container: a header toolbutton with an arrow and a body frame.
    Use `body_layout()` to add your content.
    """
    def __init__(self, title: str, collapsed: bool = True, parent=None):
        super().__init__(parent)

        # Outer frame container
        self._group = QGroupBox()       # everything goes INSIDE this groupbox
        self._group.setTitle("")        # no native title; we’ll use the button text
        self._group.setFlat(False)
        self._group.setMinimumWidth(240) #match sidebar minimum width
        self._group.setStyleSheet("QGroupBox { margin-top: 4px; }")

        # Header button (inside the groupbox)
        self._btn = QToolButton(parent=self)
        self._btn.setText(title)
        self._btn.setCheckable(True)
        self._btn.setChecked(not collapsed)
        self._btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)  # type: ignore
        self._btn.setMinimumHeight(24)
        self._btn.setMinimumWidth(160)
        self._btn.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips, True)  # type: ignore
        self._btn.setArrowType(Qt.ArrowType.DownArrow if not collapsed else Qt.ArrowType.RightArrow)  # type: ignore
        self._btn.toggled.connect(self._on_toggled)
        self._btn.setAccessibleName(tr("Toggle {title}").format(title=title))

        # Collapsible body (inside the groupbox)
        self._body = QFrame()
        self._body.setFrameShape(QFrame.Shape.NoFrame)  # type: ignore
        self._body.setVisible(not collapsed)

        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)

        # Layout INSIDE the groupbox
        group_layout = QVBoxLayout()
        group_layout.setContentsMargins(4, 0, 4, 0)  # typical groupbox padding / 2
        group_layout.setSpacing(4)
        group_layout.addWidget(self._btn)
        group_layout.addWidget(self._body)
        self._group.setLayout(group_layout)

        # Layout of this widget: just hold the groupbox
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._group)

    def setToolTip(self, text: str) -> None:  # type: ignore[override]
        super().setToolTip(text)
        self._btn.setToolTip(text)

    def _on_toggled(self, on: bool):
        self._btn.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)  # type: ignore
        self._body.setVisible(on)

    def body_layout(self) -> QVBoxLayout:
        """Return the layout inside the body to add widgets into."""
        return self._body_layout

    # Convenience aliases (if you prefer camelCase like in the original snippet)
    def bodyLayout(self) -> QVBoxLayout:  # noqa: N802
        return self.body_layout()

    def set_title(self, title: str):
        self._btn.setText(title)

    def is_collapsed(self) -> bool:
        return not self._body.isVisible()

    def set_collapsed(self, collapsed: bool):
        if self.is_collapsed() != collapsed:
            self._btn.toggle()  # toggling updates arrow + body visibility
