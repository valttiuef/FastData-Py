from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QDate, QDateTime, QEvent, QPoint, QTime, Qt, Signal, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QCalendarWidget,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QLineEdit,
    QToolButton,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

# --- @ai START ---
# model: gpt-5-codex
# tool: codex-cli
# role: widget-refactor
# reviewed: no
# date: 2026-03-24
# --- @ai END ---


class _ClickableLineEdit(QLineEdit):
    pass


class _DateTimePopup(QFrame):
    value_changed = Signal(QDateTime)
    enabled_changed = Signal(bool)

    def __init__(
        self,
        *,
        enabled_label: str,
        initial_enabled: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("dateTimeFilterPopup")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.enabled_checkbox = QCheckBox(enabled_label, self)
        self.enabled_checkbox.setChecked(bool(initial_enabled))
        layout.addWidget(self.enabled_checkbox)

        self.calendar = QCalendarWidget(self)
        self.calendar.setGridVisible(False)
        layout.addWidget(self.calendar)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(QLabel("Time", self))

        self.time_edit = QTimeEdit(self)
        self.time_edit.setDisplayFormat("HH:mm:ss")
        self.time_edit.setKeyboardTracking(False)
        self.time_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.addWidget(self.time_edit, 1)

        self.now_button = QPushButton("Now", self)
        row.addWidget(self.now_button)
        layout.addLayout(row)

        self.enabled_checkbox.toggled.connect(self._on_enabled_toggled)
        self.calendar.selectionChanged.connect(self._emit_value)
        self.time_edit.timeChanged.connect(lambda _time: self._emit_value())
        self.now_button.clicked.connect(self._set_now)
        self._sync_editors_enabled()

    def _sync_editors_enabled(self) -> None:
        enabled = bool(self.enabled_checkbox.isChecked())
        self.calendar.setEnabled(enabled)
        self.time_edit.setEnabled(enabled)
        self.now_button.setEnabled(enabled)

    def _on_enabled_toggled(self, checked: bool) -> None:
        self._sync_editors_enabled()
        self.enabled_changed.emit(bool(checked))

    def _emit_value(self) -> None:
        date = self.calendar.selectedDate()
        time = self.time_edit.time()
        self.value_changed.emit(QDateTime(date, time))

    def _set_now(self) -> None:
        now = QDateTime.currentDateTime()
        self.set_date_time(now, emit_signal=True)

    def set_date_time(self, value: QDateTime, *, emit_signal: bool) -> None:
        qdt = value if value.isValid() else QDateTime.currentDateTime()
        old_cal = self.calendar.blockSignals(True)
        old_time = self.time_edit.blockSignals(True)
        try:
            self.calendar.setSelectedDate(qdt.date())
            self.time_edit.setTime(qdt.time())
        finally:
            self.calendar.blockSignals(old_cal)
            self.time_edit.blockSignals(old_time)
        if emit_signal:
            self._emit_value()

    def date_time(self) -> QDateTime:
        return QDateTime(self.calendar.selectedDate(), self.time_edit.time())

    def set_enabled_flag(self, enabled: bool, *, emit_signal: bool) -> None:
        flag = bool(enabled)
        if self.enabled_checkbox.isChecked() == flag:
            self._sync_editors_enabled()
            return
        if emit_signal:
            self.enabled_checkbox.setChecked(flag)
            return
        old = self.enabled_checkbox.blockSignals(True)
        try:
            self.enabled_checkbox.setChecked(flag)
        finally:
            self.enabled_checkbox.blockSignals(old)
        self._sync_editors_enabled()

    def enabled_flag(self) -> bool:
        return bool(self.enabled_checkbox.isChecked())


class DateTimeFilterControl(QFrame):
    """Single compact control with popup calendar+time editor and in-popup enable toggle."""

    enabled_changed = Signal(bool)
    date_time_changed = Signal(QDateTime)

    def __init__(
        self,
        *,
        enabled_label: str,
        watermark_text: str,
        initial_datetime: QDateTime,
        initial_enabled: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("dateTimeFilterControl")
        self._watermark_text = str(watermark_text)
        self._popup_open = False
        self._control_press_active = False
        self._press_intent_toggle = False
        self._suppress_next_open = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._display = _ClickableLineEdit(self)
        self._display.setObjectName("dateTimeFilterDisplay")
        self._display.setReadOnly(True)
        self._display.setPlaceholderText(self._watermark_text)
        self._display.setTextMargins(6, 0, 6, 0)
        layout.addWidget(self._display, 1)

        self._button = QToolButton(self)
        self._button.setObjectName("dateTimeFilterButton")
        self._button.setArrowType(Qt.ArrowType.DownArrow)
        self._button.setAutoRaise(True)
        self._button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._button.setFixedWidth(20)
        layout.addWidget(self._button)

        self._popup = _DateTimePopup(
            enabled_label=enabled_label,
            initial_enabled=bool(initial_enabled),
            parent=self,
        )
        self._popup.value_changed.connect(self._on_popup_value_changed)
        self._popup.enabled_changed.connect(self._on_popup_enabled_changed)

        initial = initial_datetime if initial_datetime.isValid() else QDateTime(QDate.currentDate(), QTime(0, 0))
        self._popup.set_date_time(initial, emit_signal=False)
        self._refresh_display()

        self.installEventFilter(self)
        self._display.installEventFilter(self)
        self._button.installEventFilter(self)
        self._popup.installEventFilter(self)

    def _on_popup_value_changed(self, value: QDateTime) -> None:
        self._refresh_display()
        self.date_time_changed.emit(value)

    def _on_popup_enabled_changed(self, enabled: bool) -> None:
        self._refresh_display()
        self.enabled_changed.emit(bool(enabled))

    def _refresh_display(self) -> None:
        enabled = self.is_enabled()
        self._display.setProperty("dateControlDisabled", not enabled)
        if enabled:
            self._display.setText(self._format_display_datetime(self.date_time()))
        else:
            self._display.setText(self._watermark_text)
        self._display.style().unpolish(self._display)
        self._display.style().polish(self._display)
        self._display.update()

    def _format_display_datetime(self, value: QDateTime) -> str:
        if not value or not value.isValid():
            return ""
        t = value.time()
        if t.hour() == 0 and t.minute() == 0 and t.second() == 0 and t.msec() == 0:
            return value.toString("dd/MM/yyyy")
        if t.second() == 0 and t.msec() == 0:
            return value.toString("dd/MM/yyyy HH:mm")
        return value.toString("dd/MM/yyyy HH:mm:ss")

    def show_popup(self) -> None:
        bottom_left = self.mapToGlobal(self.rect().bottomLeft())
        self._popup.adjustSize()
        self._popup.move(QPoint(bottom_left.x(), bottom_left.y() + 2))
        self._popup.show()
        self._popup.raise_()
        self._popup.activateWindow()
        self._popup_open = True

    def hide_popup(self) -> None:
        self._popup.hide()
        self._popup_open = False
        try:
            over_control = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        except Exception:
            over_control = False
        if over_control:
            self._suppress_next_open = True
            QTimer.singleShot(250, self._clear_suppress_next_open)

    def _clear_suppress_next_open(self) -> None:
        self._suppress_next_open = False

    def eventFilter(self, obj, ev):
        try:
            et = ev.type()
        except Exception:
            return super().eventFilter(obj, ev)

        if obj is self._popup and et == QEvent.Type.Hide:  # type: ignore
            self._popup_open = False
            try:
                over_control = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
            except Exception:
                over_control = False
            if over_control:
                self._suppress_next_open = True
                QTimer.singleShot(250, self._clear_suppress_next_open)
            return False

        if obj in (self, self._display, self._button):
            if et == QEvent.Type.MouseButtonPress:  # type: ignore
                if getattr(ev, "button", lambda: None)() == Qt.MouseButton.LeftButton:
                    self._control_press_active = True
                    if self._suppress_next_open:
                        self._press_intent_toggle = False
                        return True
                    self._press_intent_toggle = True
                    return True
                return True

            if et == QEvent.Type.MouseButtonRelease:  # type: ignore
                if getattr(ev, "button", lambda: None)() == Qt.MouseButton.LeftButton:
                    was_press = self._control_press_active
                    self._control_press_active = False
                    if self._suppress_next_open:
                        self._suppress_next_open = False
                        return True
                    if was_press and self._press_intent_toggle:
                        if self._popup_open:
                            self.hide_popup()
                        else:
                            QTimer.singleShot(0, self.show_popup)
                    return True
                self._control_press_active = False
                return True

            if et == QEvent.Type.MouseButtonDblClick:  # type: ignore
                return True

        return super().eventFilter(obj, ev)

    def date_time(self) -> QDateTime:
        return self._popup.date_time()

    def set_date_time(self, value: QDateTime, *, emit_signal: bool = False) -> None:
        self._popup.set_date_time(value, emit_signal=emit_signal)
        self._refresh_display()
        if emit_signal:
            self.date_time_changed.emit(self.date_time())

    def is_enabled(self) -> bool:
        return self._popup.enabled_flag()

    def set_enabled(self, enabled: bool, *, emit_signal: bool = True) -> None:
        prev = self.is_enabled()
        self._popup.set_enabled_flag(enabled, emit_signal=False)
        self._refresh_display()
        cur = self.is_enabled()
        if emit_signal and prev != cur:
            self.enabled_changed.emit(cur)
