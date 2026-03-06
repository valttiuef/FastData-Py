from __future__ import annotations
from typing import Callable, Iterable, List, Tuple, Any, Optional

from PySide6.QtCore import Signal, Qt, QEvent, QTimer
from PySide6.QtGui import QStandardItemModel, QStandardItem, QCursor
from PySide6.QtWidgets import QComboBox, QListView, QAbstractItemView, QMenu

from ..localization import tr
import logging

logger = logging.getLogger(__name__)


class MultiCheckCombo(QComboBox):
    """
    Reusable multi-check combobox.

    API:
      - set_items([(label, value), ...], check_all=True)
      - clear_items()
      - selected_values() -> list
      - set_selected_values(values: list)
      - set_placeholder("All ...")
      - set_summary_max(n: int)
      - set_summary_formatter(callable(labels:list[str], checked:int, total:int) -> str)

    Behavior:
      - Clicking anywhere on a row toggles the checkbox (we handle it on viewport events).
      - Clicking anywhere on the widget (line edit, not just the arrow) opens the popup.
      - Emits selection_changed on every toggle/change.
    """
    selection_changed = Signal()
    context_action_triggered = Signal(str, object)

    _ACTION_ROLE = Qt.ItemDataRole.UserRole + 1
    _ACTION_TOGGLE_ALL = "__action_toggle_all__"

    def __init__(self, parent=None, placeholder: str = "All", summary_max: int = 4):
        super().__init__(parent)

        self._placeholder = tr(placeholder) if placeholder else tr("All")
        self._summary_max = max(1, int(summary_max))
        self._internal_change = False
        self._summary_formatter: Optional[Callable[[List[str], int, int], str]] = None
        self._max_checked: Optional[int] = None
        self._context_actions: list[tuple[str, str]] = []
        self._empty_selection_means_all = True
        self._preserve_missing_selected_values = False
        self._remembered_selected_values: Optional[list[Any]] = None

        # --- popup toggle robustness ---
        self._popup_open = False                 # our own truth
        self._combo_press_active = False
        self._press_intent_toggle = False        # whether we want to toggle on release
        self._suppress_next_open = False         # set when popup closed due to clicking the combo

        # Model
        model = QStandardItemModel(self)
        self.setModel(model)
        model.itemChanged.connect(self._on_item_changed)

        # View
        view = QListView(self)
        view.setSelectionMode(QListView.SelectionMode.NoSelection)  # type: ignore
        view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)  # type: ignore
        view.setUniformItemSizes(True)
        self.setView(view)

        # Consume clicks on viewport to toggle check state ourselves
        self.view().viewport().installEventFilter(self)

        # Read-only line edit for display
        self.setEditable(True)
        le = self.lineEdit()
        if le is not None:
            le.setReadOnly(True)
            le.setPlaceholderText(self._placeholder)

        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)  # type: ignore
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # type: ignore
        self.setModelColumn(0)

        # Open/close popup when clicking anywhere on combo or line edit
        self.installEventFilter(self)
        if le is not None:
            le.installEventFilter(self)

        self._refresh_summary()

    # ---------- public API ----------
    def set_placeholder(self, text: str):
        self._placeholder = tr(text) if text else tr("All")
        self._refresh_summary()

    def set_summary_max(self, n: int):
        self._summary_max = max(1, int(n))
        self._refresh_summary()

    def set_summary_formatter(self, formatter: Callable[[List[str], int, int], str]):
        self._summary_formatter = formatter
        self._refresh_summary()

    def set_max_checked(self, maximum: Optional[int]):
        self._max_checked = None if maximum is None else max(1, int(maximum))
        self._enforce_max_checked()

    def max_checked(self) -> Optional[int]:
        return self._max_checked

    def set_empty_selection_means_all(self, enabled: bool) -> None:
        self._empty_selection_means_all = bool(enabled)

    def empty_selection_means_all(self) -> bool:
        return bool(self._empty_selection_means_all)

    def set_preserve_missing_selected_values(self, enabled: bool) -> None:
        self._preserve_missing_selected_values = bool(enabled)
        if not self._preserve_missing_selected_values:
            self._remembered_selected_values = list(self.selected_values())

    def preserve_missing_selected_values(self) -> bool:
        return bool(self._preserve_missing_selected_values)

    def remembered_selected_values(self) -> list[Any]:
        if not self._preserve_missing_selected_values:
            return self.selected_values()
        return list(self._remembered_selected_values or [])

    def set_remembered_selected_values(self, values: Iterable[Any] | None) -> None:
        self._remembered_selected_values = list(values or [])

    def clear_items(self):
        m: QStandardItemModel = self.model()  # type: ignore[assignment]
        self._internal_change = True
        try:
            m.blockSignals(True)
            m.clear()
        finally:
            m.blockSignals(False)
            self._internal_change = False
        if not self._preserve_missing_selected_values:
            self._remembered_selected_values = []
        self._refresh_summary()
        if not self.signalsBlocked():
            self.selection_changed.emit()

    def set_items(self, items: Iterable[Tuple[str, Any]], check_all: bool = False):
        m: QStandardItemModel = self.model()  # type: ignore[assignment]
        self._internal_change = True
        remembered = (
            None if self._remembered_selected_values is None else list(self._remembered_selected_values)
            if self._preserve_missing_selected_values
            else list(self.selected_values())
        )

        if self._popup_open:
            self.hidePopup()

        try:
            m.blockSignals(True)
            m.clear()
            self._append_action_items(m)
            for label, value in items:
                it = QStandardItem(str(label))
                it.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )  # type: ignore
                it.setCheckable(True)
                it.setCheckState(Qt.CheckState.Checked if check_all else Qt.CheckState.Unchecked)  # type: ignore
                it.setData(value, Qt.ItemDataRole.UserRole)
                m.appendRow(it)
        finally:
            m.blockSignals(False)
            self._internal_change = False

        visible_values = self._all_item_values()
        if self._preserve_missing_selected_values:
            if remembered is None:
                remembered = list(visible_values) if check_all else []
            remembered_full = list(remembered or [])
            self._remembered_selected_values = list(remembered_full)
            visible_selected = [value for value in remembered_full if value in set(visible_values)]
            self.set_selected_values(visible_selected)
            self._remembered_selected_values = list(remembered_full)
        else:
            self._remembered_selected_values = list(self.selected_values())
        self._enforce_max_checked()
        self._refresh_summary()
        self._clear_current_index(self.lineEdit().text() if self.lineEdit() is not None else None)

        try:
            self.view().reset()
        except Exception:
            logger.warning("Exception in set_items", exc_info=True)

        if not self.signalsBlocked():
            self.selection_changed.emit()

    def selected_values(self) -> list:
        out: List[Any] = []
        m: QStandardItemModel = self.model()  # type: ignore[assignment]
        for row in range(m.rowCount()):
            it = m.item(row)
            if it and not self._is_action_item(it) and it.checkState() == Qt.CheckState.Checked:  # type: ignore
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def set_selected_values(self, values: Iterable[Any]):
        vs = set(values or [])
        m: QStandardItemModel = self.model()  # type: ignore[assignment]
        self._internal_change = True
        try:
            m.blockSignals(True)
            for row in range(m.rowCount()):
                it = m.item(row)
                if not it or self._is_action_item(it):
                    continue
                val = it.data(Qt.ItemDataRole.UserRole)
                it.setCheckState(Qt.CheckState.Checked if val in vs else Qt.CheckState.Unchecked)  # type: ignore
        finally:
            m.blockSignals(False)
            self._internal_change = False

        if self._preserve_missing_selected_values:
            ordered = [value for value in self._remembered_selected_values if value in vs]
            extras = [value for value in values or [] if value in vs and value not in ordered]
            self._remembered_selected_values = ordered + extras
        else:
            self._remembered_selected_values = list(values or [])
        self._enforce_max_checked()
        self._refresh_summary()
        self._clear_current_index(self.lineEdit().text() if self.lineEdit() is not None else None)

        if not self.signalsBlocked():
            self.selection_changed.emit()

    def clear_selection(self) -> None:
        self.set_selected_values([])

    def set_context_actions(self, actions: Iterable[Tuple[str, str]]) -> None:
        out: list[tuple[str, str]] = []
        for key, label in actions or []:
            action_key = str(key or "").strip()
            action_label = str(label or "").strip()
            if action_key and action_label:
                out.append((action_key, action_label))
        self._context_actions = out

    # ---------- events ----------
    def eventFilter(self, obj, ev):
        try:
            et = ev.type()
        except Exception:
            return super().eventFilter(obj, ev)

        # 1) popup viewport: toggle check state ourselves
        if obj is self.view().viewport():
            if et in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease, QEvent.Type.MouseButtonDblClick):  # type: ignore
                if et == QEvent.Type.MouseButtonPress:  # type: ignore
                    idx = self.view().indexAt(ev.pos())
                    if idx.isValid():
                        m: QStandardItemModel = self.model()  # type: ignore[assignment]
                        it = m.itemFromIndex(idx)
                        if it is not None:
                            if (
                                getattr(ev, "button", lambda: None)() == Qt.MouseButton.RightButton
                                and self._context_actions
                                and not self._is_action_item(it)
                            ):
                                self._show_context_menu_for_item(ev.globalPosition().toPoint(), it)
                                return True
                            if self._handle_action_item(it):
                                return True
                            it.setCheckState(
                                Qt.CheckState.Unchecked
                                if it.checkState() == Qt.CheckState.Checked
                                else Qt.CheckState.Checked
                            )  # type: ignore
                return True
            return False

        # 2) combo / lineEdit: deterministic toggle
        if obj is self or obj is self.lineEdit():
            if et == QEvent.Type.MouseButtonPress:  # type: ignore
                if getattr(ev, "button", lambda: None)() == Qt.MouseButton.LeftButton:
                    self._combo_press_active = True

                    # If the popup just closed because user clicked the combo, do NOT reopen.
                    if self._suppress_next_open:
                        self._press_intent_toggle = False
                        return True

                    # Decide intent based on our flag (NOT isVisible()).
                    self._press_intent_toggle = True
                    return True
                return True

            if et == QEvent.Type.MouseButtonRelease:  # type: ignore
                if getattr(ev, "button", lambda: None)() == Qt.MouseButton.LeftButton:
                    was_press = self._combo_press_active
                    self._combo_press_active = False

                    # Clear suppression after the click cycle finishes.
                    if self._suppress_next_open:
                        self._suppress_next_open = False
                        return True

                    if was_press and self._press_intent_toggle:
                        if self._popup_open:
                            self.hidePopup()
                        else:
                            QTimer.singleShot(0, self.showPopup)
                    return True

                self._combo_press_active = False
                return True

            if et == QEvent.Type.MouseButtonDblClick:  # type: ignore
                return True

        return super().eventFilter(obj, ev)

    def showPopup(self):
        self.view().setMinimumWidth(max(self.view().sizeHintForColumn(0) + 24, self.width()))
        super().showPopup()
        self._popup_open = True

    def hidePopup(self):
        # If the cursor is over the combo at the moment the popup closes,
        # it was very likely closed by clicking the combo again. Suppress reopening.
        try:
            over_combo = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        except Exception:
            over_combo = False

        super().hidePopup()
        self._popup_open = False

        if over_combo:
            self._suppress_next_open = True
            # also auto-clear in case some weird path never gets the release
            QTimer.singleShot(250, self._clear_suppress_next_open)

    def _clear_suppress_next_open(self):
        self._suppress_next_open = False

    # ---------- context menu ----------
    def _show_context_menu_for_item(self, global_pos, item: QStandardItem) -> None:
        if item is None or self._is_action_item(item) or not self._context_actions:
            return
        menu = QMenu(self)
        for action_key, action_label in self._context_actions:
            action = menu.addAction(action_label)
            action.setData(action_key)
        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        action_key = str(chosen.data() or "").strip()
        if not action_key:
            return
        payload = {"value": item.data(Qt.ItemDataRole.UserRole), "label": item.text()}
        self.context_action_triggered.emit(action_key, payload)

    # ---------- model changes ----------
    def _on_item_changed(self, it: QStandardItem):
        if self._internal_change or it is None:
            return
        self._enforce_max_checked(prefer=it)
        current = self.selected_values()
        if self._preserve_missing_selected_values:
            self._remembered_selected_values = list(current)
        else:
            self._remembered_selected_values = list(current)
        self._refresh_summary()
        if not self.signalsBlocked():
            self.selection_changed.emit()

    def _enforce_max_checked(self, prefer: Optional[QStandardItem] = None) -> None:
        if self._max_checked is None:
            return
        maximum = max(1, int(self._max_checked))
        m: QStandardItemModel = self.model()  # type: ignore[assignment]

        checked_items: list[QStandardItem] = []
        for row in range(m.rowCount()):
            item = m.item(row)
            if item is None or self._is_action_item(item):
                continue
            if item.checkState() == Qt.CheckState.Checked:  # type: ignore
                checked_items.append(item)

        if len(checked_items) <= maximum:
            return

        keep: list[QStandardItem] = []
        if prefer is not None and prefer in checked_items:
            keep.append(prefer)
        for item in checked_items:
            if item in keep:
                continue
            keep.append(item)
            if len(keep) >= maximum:
                break

        self._internal_change = True
        try:
            m.blockSignals(True)
            for item in checked_items:
                if item not in keep:
                    item.setCheckState(Qt.CheckState.Unchecked)  # type: ignore
        finally:
            m.blockSignals(False)
            self._internal_change = False

    # ---------- summary ----------
    def _refresh_summary(self):
        m: QStandardItemModel = self.model()  # type: ignore[assignment]
        total = 0
        for row in range(m.rowCount()):
            it = m.item(row)
            if it is not None and not self._is_action_item(it):
                total += 1

        if total == 0:
            le = self.lineEdit()
            if le is not None:
                le.setText(self._placeholder)
            self.setToolTip(self._placeholder)
            self._clear_current_index()
            self._update_toggle_action_label()
            return

        checked = 0
        labels: List[str] = []
        for row in range(m.rowCount()):
            it = m.item(row)
            if it is None or self._is_action_item(it):
                continue
            if it.checkState() == Qt.CheckState.Checked:  # type: ignore
                checked += 1
                if len(labels) < self._summary_max:
                    labels.append(it.text())

        if self._summary_formatter:
            text = self._summary_formatter(labels, checked, total)
        else:
            if checked == 0:
                text = (
                    self._placeholder
                    if self._empty_selection_means_all
                    else tr("None selected")
                )
            elif checked == total:
                text = tr("All selected ({count})").format(count=total)
            else:
                text = ", ".join(labels) + ("…" if checked > self._summary_max else "")

        text = text or self._placeholder
        le = self.lineEdit()
        if le is not None:
            le.setText(text)

        self._clear_current_index(text)

        all_checked = []
        for i in range(m.rowCount()):
            item = m.item(i)
            if item is None or self._is_action_item(item):
                continue
            if item.checkState() == Qt.CheckState.Checked:  # type: ignore
                all_checked.append(item.text())
        self.setToolTip(", ".join(all_checked) if all_checked else self._placeholder)
        self._update_toggle_action_label()

    # ---------- action row helpers ----------
    def _append_action_items(self, model: QStandardItemModel) -> None:
        item = QStandardItem(tr("Check all"))
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)  # type: ignore
        item.setCheckable(False)
        item.setData(self._ACTION_TOGGLE_ALL, self._ACTION_ROLE)
        model.appendRow(item)

    def _is_action_item(self, item: Optional[QStandardItem]) -> bool:
        if item is None:
            return False
        return item.data(self._ACTION_ROLE) == self._ACTION_TOGGLE_ALL

    def _handle_action_item(self, item: QStandardItem) -> bool:
        action = item.data(self._ACTION_ROLE)
        if action == self._ACTION_TOGGLE_ALL:
            self._set_all_checked(not self._all_items_checked())
            return True
        return False

    def _all_items_checked(self) -> bool:
        m: QStandardItemModel = self.model()  # type: ignore[assignment]
        total = 0
        checked = 0
        for row in range(m.rowCount()):
            item = m.item(row)
            if item is None or self._is_action_item(item):
                continue
            total += 1
            if item.checkState() == Qt.CheckState.Checked:  # type: ignore
                checked += 1
        return total > 0 and checked == total

    def _update_toggle_action_label(self) -> None:
        m: QStandardItemModel = self.model()  # type: ignore[assignment]
        for row in range(m.rowCount()):
            item = m.item(row)
            if item is None or not self._is_action_item(item):
                continue
            item.setText(tr("Uncheck all") if self._all_items_checked() else tr("Check all"))
            break

    def _set_all_checked(self, checked: bool) -> None:
        m: QStandardItemModel = self.model()  # type: ignore[assignment]
        self._internal_change = True
        try:
            for row in range(m.rowCount()):
                item = m.item(row)
                if item is None or self._is_action_item(item):
                    continue
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)  # type: ignore
        finally:
            self._internal_change = False

        current = self.selected_values()
        self._remembered_selected_values = list(current)
        self._enforce_max_checked()
        self._refresh_summary()
        self._clear_current_index(self.lineEdit().text() if self.lineEdit() is not None else None)

        try:
            self.view().viewport().update()
        except Exception:
            logger.warning("Exception in _set_all_checked", exc_info=True)

        if not self.signalsBlocked():
            self.selection_changed.emit()

    def _clear_current_index(self, text: Optional[str] = None) -> None:
        was_blocked = self.blockSignals(True)
        try:
            self.setCurrentIndex(-1)
        finally:
            self.blockSignals(was_blocked)
        if text is not None:
            le = self.lineEdit()
            if le is not None:
                le.setText(text)

    def _all_item_values(self) -> list[Any]:
        values: list[Any] = []
        m: QStandardItemModel = self.model()  # type: ignore[assignment]
        for row in range(m.rowCount()):
            item = m.item(row)
            if item is None or self._is_action_item(item):
                continue
            values.append(item.data(Qt.ItemDataRole.UserRole))
        return values
